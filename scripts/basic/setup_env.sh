#!/bin/bash
# Shared env helpers sourced by Cambrian-P training scripts in cambrianp/scripts/.
# Override any default below by exporting the corresponding env var before
# invoking the training script. Sensitive values (WANDB_API_KEY) MUST NOT
# be committed — set them in the user's shell.

log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

init_common_env() {
    log() {
        printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    }

    # Single-node by default. For multi-node, export NNODES, MASTER_ADDR,
    # MASTER_PORT, RDZV_ID, RANK before sourcing.
    log "Defaulting to single-node configuration."
    export NNODES=${NNODES:-1}
    export NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
    export RDZV_ID=${RDZV_ID:-0}
    export RANK=${RANK:-0}
    export ADDR=${ADDR:-localhost}
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export MASTER_PORT=${MASTER_PORT:-$(( ( RANDOM % 64512 ) + 1024 ))}
    log "Using $MASTER_ADDR:$MASTER_PORT for master_addr:master_port"

    # Performance settings
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
    export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
}

init_deepspeed_config() {
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1 | cut -d' ' -f1)
    export GPU_MEMORY
    export DEEPSPEED_CONFIG_FILE=${DEEPSPEED_CONFIG_FILE:-"scripts/zero3_stage1.json"}
    log "Using DeepSpeed config file: $DEEPSPEED_CONFIG_FILE"
}

init_batch_size_config() {
    # bs=1 below 90 GB GPU memory, bs=2 otherwise.
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1 | cut -d' ' -f1)
    export GPU_MEMORY
    if [[ $GPU_MEMORY -lt 90000 ]]; then
        export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
    else
        export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
    fi
    log "Using per_device_train_batch_size: $PER_DEVICE_TRAIN_BATCH_SIZE"
}

init_wandb_config() {
    export WANDB_DISABLE_CODE="true"
    export WANDB_IGNORE_GLOBS="*.patch"
    export WANDB_ENTITY=${WANDB_ENTITY:-"cambrian-mllm"}
    export WANDB_PROJECT=${WANDB_PROJECT:-"cambrian-p"}
    # WANDB_API_KEY: do NOT hardcode in this file. Export it in your shell:
    #   export WANDB_API_KEY=<your-key>
}

setup_exp_dir() {
    export OUTPUT_DIR=${OUTPUT_DIR:-"./ckpts"}
    export DATA_DIR=${DATA_DIR:-"./data"}
    export MAPANYTHING_JSON=${MAPANYTHING_JSON:-"data/mapanything_scenes.json"}
    export MAPANYTHING_ROOT=${MAPANYTHING_ROOT:-"./data/mapanythingdata"}
}

setup_exp_name() {
    export EXP_NAME=${EXP_NAME:-$(basename "$0" .sh)}
    export WANDB_NAME=$EXP_NAME
    echo "Experiment name: $EXP_NAME"
    echo "WandB name: $WANDB_NAME"
}
