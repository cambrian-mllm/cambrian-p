# Evaluation

## Setup


```bash
conda activate cambrianp
export PYTHONPATH=$PWD/lmms-eval:$PWD:$PYTHONPATH
```

## Download the checkpoints

Each variant is its own HF repo. Pick the ones you want to reproduce:

```bash
mkdir -p ckpts && cd ckpts

huggingface-cli download nyu-visionx/Cambrian-P-7B      --local-dir Cambrian-P-7B

huggingface-cli download nyu-visionx/Cambrian-P-7B-Mix-MA   --local-dir Cambrian-P-7B-Mix-MA

cd ..
```

Point an env var at whichever variant you're evaluating:

```bash
export CAMBRIANP_CKPT=$PWD/ckpts/Cambrian-P-7B         
# or any other variant directory, e.g.:
# export CAMBRIANP_CKPT=$PWD/ckpts/Cambrian-P-7B-Mix-MA
```

## Video QA

Cambrian-P is evaluated on 10 spatial-reasoning and general video benchmarks (paper Section 4.1): VSI-Bench, VSI-TemporalI-Bench, SparBench, MMSIBench, MMSIVideo, MindCube, Tomato, MVBench, EgoSchema, and Perception Test.

```bash
NUM_FRAMES=128   # Cambrian-P-7B (default) and -Mix-CamS; 64 for -Mix-MA; 32 for -32f / -Mix-3R

bash lmms-eval/evaluate_all_in_one.sh \
    --benchmark vsibench \
    --model cambrian_p_7b \
    --finetuned_model_path "$CAMBRIANP_CKPT" \
    --num_frames "$NUM_FRAMES" \
    --num_processes 8 \
    --use_camera_tokens true \
    --camera_tokens_mode camera_tokens \
    --camera_tokens_place append_to_frame \
    --query_mode query_after_image \
    --conv_template qwen_1_5 \
    --output_path ./logs
```

## Streaming Pose Estimation

On-disk layout follows the MonST3R preprocessing convention [`data/evaluation_script.md`](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) for ScanNet, TUM-dynamic and Sintel:

Point at your local dataset roots via env vars (defaults in [`cambrianp/eval/relpose/metadata.py`](../cambrianp/eval/relpose/metadata.py)):

```bash
export CAMBRIANP_CKPT=$PWD/ckpts/Cambrian-P-7B-Mix-MA    # pose results come from this checkpoint
export CAMBRIANP_SCANNET_PATH=/path/to/scannetv2
export CAMBRIANP_TUM_PATH=/path/to/tum
export CAMBRIANP_SINTEL_PATH=/path/to/sintel/training/final
export CAMBRIANP_SINTEL_ANNO_PATH=/path/to/sintel/training/camdata_left
```

Then run all three datasets:

```bash
cd cambrianp/eval/relpose
for DS in scannet tum sintel; do
    python unified_pose_eval.py \
        --model_type llava_vggt \
        --model_path "$CAMBRIANP_CKPT" \
        --eval_dataset "$DS" \
        --sampling_strategy monst3r \
        --camera_tokens_place append_to_frame \
        --output_dir ./eval_out/${DS} \
        --no_timestamp \
        --viz
done
```


Outputs per dataset: `results.csv` (per-scene `ate, rpe_trans, rpe_rot` + AVERAGE), `config.txt`, `summary.txt`, `metrics.json`, per-scene trajectory plots, per-frame timing under `timing/`.

