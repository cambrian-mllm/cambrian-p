#!/bin/bash

set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi

benchmark=vsibench
timestamp=$(TZ="America/New_York" date "+%Y%m%d")
num_processes=8
num_frames=32
launcher=accelerate
use_camera_tokens="false"
camera_tokens_mode="camera_tokens"
camera_tokens_place="prepend_to_frame"
query_mode="query_after_question"
conv_template="qwen_1_5"
output_path=""

available_models="cambrian_p_0p5b,cambrian_p_7b,llava_one_vision_qwen2_0p5b_ov_32f,llava_one_vision_qwen2_7b_ov_32f,llava_next_video_7b_qwen2_32f,llama3_vila1p5_8b_32f,llama3_longvila_8b_128frames_32f,longva_7b_32f,internvl2_2b_8f,internvl2_8b_8f"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --num_processes)
        num_processes="$2"
        shift 2
        ;;
    --model)
        IFS=',' read -r -a models <<<"$2"
        shift 2
        ;;
    --finetuned_model_path)
        finetuned_model_path="$2"
        shift 2
        ;;
    --limit)
        limit="$2"
        shift 2
        ;;
    --num_frames)
        num_frames="$2"
        shift 2
        ;;
    --use_camera_tokens)
        use_camera_tokens="$2"
        shift 2
        ;;
    --camera_tokens_mode)
        camera_tokens_mode="$2"
        shift 2
        ;;
    --camera_tokens_place)
        camera_tokens_place="$2"
        shift 2
        ;;
    --query_mode)
        query_mode="$2"
        shift 2
        ;;
    --conv_template)
        conv_template="$2"
        shift 2
        ;;
    --output_path)
        output_path="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

if [ "$models" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

for model in "${models[@]}"; do
    echo "Start evaluating $model..."
    log_suffix="$timestamp"
    output_model_name="$([ -n "$finetuned_model_path" ] && basename "$finetuned_model_path" || echo "$model")"

    case "$model" in
    "gemini_1p5_flash")
        model_family="gemini_api"
        model_args="model_version=gemini-1.5-flash,modality=video"
        ;;
    "gemini_1p5_pro_002")
        model_family="gemini_api"
        model_args="model_version=gemini-1.5-pro,modality=video"
        ;;
    "gemini_2p0_flash_exp")
        model_family="gemini_api"
        model_args="model_version=gemini-2.0-flash-exp,modality=video"
        ;;
    "gpt_4o_2024_08_06_f16")
        model_family="gpt4v"
        model_args="model_version=gpt-4o-2024-08-06,modality=video,max_frames_num=16"
        ;;
    "llava_one_vision_qwen2_0p5b_ov_32f")
        model_family="cambrian_p"
        model="llava_one_vision_qwen2_0p5b_ov_${num_frames}f"
        model_args="pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames"
        ;;
    "llava_one_vision_qwen2_7b_ov_32f")
        model_family="cambrian_p"
        model="llava_one_vision_qwen2_7b_ov_${num_frames}f"
        model_args="pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames"
        ;;
    "llava_one_vision_qwen2_72b_ov_32f")
        model_family="cambrian_p"
        model_args="pretrained=lmms-lab/llava-onevision-qwen2-72b-ov-sft,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32,device_map=auto"
        num_processes=1
        ;;
    "llava_next_video_7b_qwen2_32f")
        model_family="llava_vid"
        model="llava_next_video_7b_qwen2_${num_frames}f"
        model_args="pretrained=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,max_frames_num=$num_frames"
        ;;
    "llava_next_video_72b_qwen2_32f")
        model_family="llava_vid"
        model_args="pretrained=lmms-lab/LLaVA-NeXT-Video-72B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,max_frames_num=32,device_map=auto"
        num_processes=1
        ;;
    "llama3_vila1p5_8b_32f")
        model_family="vila"
        model="llama3_vila1p5_8b_${num_frames}f"
        model_args="pretrained=Efficient-Large-Model/Llama-3-VILA1.5-8B,attn_implementation=flash_attention_2,video_decode_backend=decord,conv_template=llama_3,max_frames_num=$num_frames"
        ;;
    "llama3_vila1p5_40b_32f")
        model_family="vila"
        model_args="pretrained=Efficient-Large-Model/VILA1.5-40b,attn_implementation=flash_attention_2,video_decode_backend=decord,conv_template=hermes-2,max_frames_num=32,device_map=auto"
        num_processes=1
        ;;
    "llama3_longvila_8b_128frames_32f")
        model_family="vila"
        model="llama3_longvila_8b_128frames_${num_frames}f"
        model_args="pretrained=Efficient-Large-Model/Llama-3-LongVILA-8B-128Frames,attn_implementation=flash_attention_2,video_decode_backend=decord,conv_template=llama_3,device_map=cuda,max_frames_num=$num_frames"
        ;;
    "longva_7b_32f")
        model_family="longva"
        model="longva_7b_${num_frames}f"
        model_args="pretrained=lmms-lab/LongVA-7B,video_decode_backend=decord,conv_template=qwen_1_5,model_name=longva_qwen_7b,max_frames_num=$num_frames"
        ;;
    "internvl2_2b_8f")
        model_family="internvl2"
        model_args="pretrained=OpenGVLab/InternVL2-2B,modality=video,max_frames_num=8"
        ;;
    "internvl2_8b_8f")
        model_family="internvl2"
        model_args="pretrained=OpenGVLab/InternVL2-8B,modality=video,max_frames_num=8"
        ;;
    "internvl2_40b_8f")
        model_family="internvl2"
        model_args="pretrained=OpenGVLab/InternVL2-40B,modality=video,max_frames_num=8,device_map=auto"
        num_processes=1
        ;;
    "cambrian_p_0p5b")
        model_family="cambrian_p"
        model_args="pretrained=$finetuned_model_path,conv_template=$conv_template,model_name=llava_qwen,max_frames_num=$num_frames,use_camera_tokens=$use_camera_tokens,camera_tokens_mode=$camera_tokens_mode,camera_tokens_place=$camera_tokens_place,query_mode=$query_mode"
        ;;
    "cambrian_p_7b")
        model_family="cambrian_p"
        model_args="pretrained=$finetuned_model_path,conv_template=$conv_template,model_name=llava_qwen,max_frames_num=$num_frames,use_camera_tokens=$use_camera_tokens,camera_tokens_mode=$camera_tokens_mode,camera_tokens_place=$camera_tokens_place,query_mode=$query_mode"
        ;;
        *)
        echo "Unknown model: $model"
        exit -1
        ;;
    esac

    if [ "$launcher" = "python" ]; then
        export LMMS_EVAL_LAUNCHER="python"
        evaluate_script="python \
            "
    elif [ "$launcher" = "accelerate" ]; then
        export LMMS_EVAL_LAUNCHER="accelerate"
        evaluate_script="accelerate launch \
            --num_processes=$num_processes \
            "
    fi

    evaluate_script="$evaluate_script -m lmms_eval \
        --model $model_family \
        --model_args $model_args \
        --tasks $benchmark \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $log_suffix \
        --output_path ${output_path:-./logs}/$benchmark/$output_model_name \
        "

    if [ -n "$limit" ]; then
        evaluate_script="$evaluate_script \
            --limit $limit \
        "
    fi
    echo $evaluate_script
    eval $evaluate_script
done
