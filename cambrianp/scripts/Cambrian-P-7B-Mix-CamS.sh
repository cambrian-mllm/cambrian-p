#!/bin/bash

set -e  # Exit on any error

source scripts/basic/setup_env.sh

init_common_env
init_wandb_config
setup_exp_name
init_deepspeed_config
init_batch_size_config
setup_exp_dir

export DECORD_EOF_RETRY_MAX=20480

# Display GPU information
nvidia-smi

# auto calculate the gradient accumulation steps according to the number of GPUs and global batch size
num_workers=4
global_batch_size=256
per_device_train_batch_size=1
total_gpus=$((NNODES * NPROC_PER_NODE))


if [ "$total_gpus" -eq 0 ]; then
    log "Error: Total number of GPUs is zero. Exiting."
    exit 1
fi

if [ $((global_batch_size % (total_gpus * per_device_train_batch_size))) -ne 0 ]; then
    log "Warning: global_batch_size is not perfectly divisible by (total_gpus * per_device_train_batch_size)."
    log "This may result in an effective global batch size that is smaller than intended."
fi

gradient_accumulation_steps=$((global_batch_size / total_gpus / per_device_train_batch_size))
log "Detected $NNODES nodes and $NPROC_PER_NODE GPUs per node for a total of $total_gpus GPUs."
log "Calculated gradient accumulation steps: $gradient_accumulation_steps"

# export DEEPSPEED_CONFIG_FILE="scripts/zero3_stage2.json"

log "Using gradient accumulation steps: $gradient_accumulation_steps"

log "Using DeepSpeed config file: $DEEPSPEED_CONFIG_FILE"

log "Using master port: $MASTER_PORT"

torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    cambrianp/train/train_mem.py \
    --deepspeed $DEEPSPEED_CONFIG_FILE \
    --model_name_or_path nyu-visionx/Cambrian-S-7B-S3 \
    --version qwen_1_5 \
    --data_path $DATA_DIR \
    --image_folder $DATA_DIR \
    --json_file data/cambrianp_train.jsonl \
    --vipe_cambrians_path data/vipe_cambrians_with_vqa.json \
    --vipe_cambrians_data_root ${VIPE_CAMBRIANS_DATA_ROOT:-/path/to/Cambrian-S-3M} \
    --vipe_cambrians_results_root ${VIPE_CAMBRIANS_RESULTS_ROOT:-/path/to/vipe_pose_results} \
    --vipe_cambrians_source_config data/vipe_source_config_high.json \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model,mm_rec_head" \
    --mm_vision_tower_lr 2e-6 \
    --downstream_head_lr 1e-4 \
    --vision_tower "google/siglip-so400m-patch14-384" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_img_tok_num 64 \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $EXP_NAME \
    --output_dir $OUTPUT_DIR/$EXP_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers $num_workers \
    --video_load_timeout 300 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 128 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --load_rec_model True \
    --load_rec_data True \
    --enable_camera True \
    --enable_point False \
    --enable_depth False \
    --data_mode unified \
    --sample_jitter 0.005 \
    --rec_data_mode cut3r \
    --cut3r_scannet_max_interval 100 \
    --cut3r_scannet_min_interval 30 \
    --cut3r_scannetpp_max_interval 100 \
    --cut3r_scannetpp_min_interval 30 \
    --cut3r_arkitscenes_max_interval 100 \
    --cut3r_arkitscenes_min_interval 30 \
    --vqa_image_augs False \
    --vqa_rec_sup_augs False \
    --interleaved_training True \
    --interleaved_aug_rec_ratio 1.0 \
    --force_bp_rec True \
    --num_intermediate_layers 4 \
    --token_num 2 \
    --rec_model_path None \
    --rec_embed_dim 2048 \
    --rec_loss_weight 0.2 \
    --rec_camera_loss_weight 5.0 \
    --rec_camera_loss_type l1 \
    --rec_depth_loss_weight 1.0 \
    --rec_depth_gradient_loss_fn grad \
    --rec_depth_valid_range 0.98 \
    --rec_depth_head_patch_start_idx 1 \
    --query_loss_frame_idx 0 \
    --camera_tokens_mode camera_tokens \
    --query_mode query_after_image \
    --rec_projector_type mlp1x_gelu \
    --use_augs True \
    --rec_resolution 384 \
    --rescale True \
    --rescale_aug True \
    --landscape_check True \
    --rotation_aug True \
    --load_depth_data True \
    --camera_tokens_place append_to_frame \
    --vipe_cambrians_rec_data_mode temporal \
    --vipe_cambrians_temporal_min_interval 10 \
    --vipe_cambrians_temporal_max_interval 50 \
    --rec_free_tail_ratio 0.02


cat > $OUTPUT_DIR/$EXP_NAME/eval_config.json << 'EOF'
{
    "num_frames": 128,
    "use_camera_tokens": "true",
    "camera_tokens_mode": "camera_tokens",
    "camera_tokens_place": "append_to_frame"
}
EOF

# remove subfolder contains "checkpoint" in the output folder
rm -rf $OUTPUT_DIR/$EXP_NAME/checkpoint-*

# =============================================================================
# Completion
# =============================================================================

log "Training finished at $(TZ="America/New_York" date "+%Y%m%d_%H%M%S")"
