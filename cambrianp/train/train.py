# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import os.path as osp
import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np
from scipy.spatial.transform import Rotation as R

import random
import re
import torch
import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig, set_seed
from torch.utils.data import Dataset
from cambrianp.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from cambrianp.train.llava_trainer import LLaVATrainer
from cambrianp import conversation as conversation_lib
from cambrianp.model import *
from cambrianp.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from cambrianp.utils import rank0_print
from cambrianp.datasets import load_3r_dataset

import cambrianp.datasets.rec_dataloading_utils as rec_utils
import cambrianp.datasets.vqa_dataloading_utils as vqa_utils
import cambrianp.datasets.mapanything_dataloading_utils as ma_utils
import cambrianp.datasets.vipe_dataloading_utils as vipe_utils

from vggt.data.augmentation_strategies import StrategyFactory
torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)
    mm_img_tok_num: Optional[int] = field(default=196)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)
    
    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)
    
    # add reconstruction related configs
    load_rec_model: Optional[bool] = False
    rec_projector_type: Optional[str] = field(default="linear")
    head_name: Optional[str] = None
    rec_model_path: Optional[str] = None
    
    token_num: Optional[int] = field(default=197)
    camera_tokens_mode: Optional[str] = field(default="camera_tokens")
    query_mode: Optional[str] = field(default="query_after_question")
    camera_tokens_place: Optional[str] = field(default="prepend_to_frame")  # options=['between_qa', 'prepend_to_frame', 'append_to_frame']
    
    rec_loss_weight: Optional[float] = field(default=0.1)
    rec_camera_loss_weight: Optional[float] = field(default=5.0)
    rec_camera_loss_type: Optional[str] = field(default="l1")
    rec_camera_loss_components: Optional[str] = field(default="all")  # ["all", "translation", "rotation", "focal"]
    
    rec_camera_loss_weight_trans: Optional[float] = field(default=1.0)
    rec_camera_loss_weight_rot: Optional[float] = field(default=1.0)
    rec_camera_loss_weight_focal: Optional[float] = field(default=0.5)

    use_point_masks: Optional[bool] = field(default=True)
    rec_depth_loss_weight: Optional[float] = field(default=1.0)
    rec_depth_gradient_loss_fn: Optional[str] = field(default="grad")
    rec_depth_valid_range: Optional[float] = field(default=0.98)
    query_loss_frame_idx: Optional[int] = field(default=None)
    rec_depth_head_patch_start_idx: Optional[int] = field(default=0)
    
    enable_point: Optional[bool] = field(default=False) 
    enable_depth: Optional[bool] = field(default=False)
    enable_camera: Optional[bool] = field(default=False)
    
    num_intermediate_layers: Optional[int] = field(default=4)
    rec_embed_dim: Optional[int] = field(default=2048)
    rec_camera_trunk_depth: Optional[int] = field(default=4)
    rec_camera_causal_attn: Optional[bool] = field(default=False)
    rec_dpt_head_type: Optional[str] = field(default="dpt_head")

    custom_llm_ce_loss: Optional[bool] = field(default=False)

    enable_rec_pose_viz: bool = field(default=True)
    rec_pose_viz_frequency: int = field(default=256)
    
    # Scale alignment direction for non-metric datasets
    # When aligning non-metric datasets, rescale predicted translations to match GT instead of rescaling GT to match pred.
    rescale_pred_not_gt: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)
    video_load_timeout: Optional[float] = field(default=120.0)
    
    load_rec_data: Optional[bool] = field(default=False)
    use_augs: Optional[bool] = field(default=False)
    rec_resolution: Optional[int] = field(default=384)
    json_file: Optional[str] = field(default=None)
    data_mode: Optional[str] = field(default=None)
    sample_jitter: Optional[float] = field(default=0.0)
    rec_data_mode: Optional[str] = field(default='cut3r')
    
    # ViPE Cambrian-S rec data mode, default should be temporal
    vipe_cambrians_rec_data_mode: Optional[str] = field(default='uniform')
    
    # interleaved training
    interleaved_training: Optional[bool] = field(default=False)
    interleaved_aug_rec_ratio: Optional[float] = field(default=1.0)
    # Separate augment ratio for ViPE-Cambrians (vipe cams) scenes. If None, falls back to
    # interleaved_aug_rec_ratio (forward-compatible with existing configs).
    vipe_interleaved_aug_rec_ratio: Optional[float] = field(default=None)
    interleaved_aug_mode: Optional[str] = field(default='scene_unified')  # options=['scene_unified', 'same_as_vqa', 'num_frame']
    interleaved_aug_frame_count_path: Optional[str] = field(default='data/video_frame_counts.json')
    spring_sample_weight: Optional[float] = field(default=1.0,
        metadata={"help": "Multiplier applied to spring dataset sampling weights in num_frame augmentation mode"})
    force_bp_rec: Optional[bool] = field(default=True)
    
    # data augs
    rescale: Optional[bool] = field(default=True) # For BothBaseDataset
    rescale_aug: Optional[bool] = field(default=False) # For BaseDataset
    landscape_check: Optional[bool] = field(default=False) # For BaseDataset
    rotation_aug: Optional[bool] = field(default=False) # For BaseDataset

    vqa_image_augs: Optional[bool] = field(default=False)
    vqa_rec_sup_augs: Optional[bool] = field(default=True)
    rec_image_augs: Optional[bool] = field(default=True)
    rec_rec_sup_augs: Optional[bool] = field(default=True)
    keep_only_dummy_vqa: Optional[bool] = field(default=False)
    
   # CUT3R mode parameters - control frame sampling behavior for Cut3r dataset
    cut3r_scannet_max_interval: Optional[int] = field(default=30,
        metadata={"help": "Maximum frame gap between consecutive frames in Cut3r video mode for ScanNet"})
    cut3r_scannet_min_interval: Optional[int] = field(default=1,
        metadata={"help": "Minimum frame gap between consecutive frames in Cut3r video mode for ScanNet"})
    cut3r_scannet_video_prob: Optional[float] = field(default=0.6,
        metadata={"help": "Probability of using temporal (video) mode vs collection mode in Cut3r video mode for ScanNet"})
    cut3r_scannet_fix_interval_prob: Optional[float] = field(default=0.6,
        metadata={"help": "Probability of using fixed interval vs variable interval in video mode for ScanNet"})
    cut3r_scannet_block_shuffle_size: Optional[int] = field(default=None,
        metadata={"help": "Block size for shuffling in collection mode (None to disable) for ScanNet"})

    cut3r_scannetpp_max_interval: Optional[int] = field(default=3,
        metadata={"help": "Maximum frame gap between consecutive frames in Cut3r video mode for scannetpp"})
    cut3r_scannetpp_min_interval: Optional[int] = field(default=1,
        metadata={"help": "Minimum frame gap between consecutive frames in Cut3r video mode for scannetpp"})
    cut3r_scannetpp_video_prob: Optional[float] = field(default=0.8,
        metadata={"help": "Probability of using temporal (video) mode vs collection mode in Cut3r for scannetpp"})
    cut3r_scannetpp_fix_interval_prob: Optional[float] = field(default=0.5,
        metadata={"help": "Probability of using fixed interval vs variable interval in video mode for scannetpp"})
    cut3r_scannetpp_block_shuffle_size: Optional[int] = field(default=None,
        metadata={"help": "Block size for shuffling in collection mode (None to disable) for scannetpp"})

    cut3r_arkitscenes_max_interval: Optional[int] = field(default=8,
        metadata={"help": "Maximum frame gap between consecutive frames in Cut3r video mode for ARKitScenes"})
    cut3r_arkitscenes_min_interval: Optional[int] = field(default=1,
        metadata={"help": "Minimum frame gap between consecutive frames in Cut3r video mode for ARKitScenes"})
    cut3r_arkitscenes_video_prob: Optional[float] = field(default=0.8,
        metadata={"help": "Probability of using temporal (video) mode vs collection mode in Cut3r for ARKitScenes"})
    cut3r_arkitscenes_fix_interval_prob: Optional[float] = field(default=0.5,
        metadata={"help": "Probability of using fixed interval vs variable interval in video mode for ARKitScenes"})
    cut3r_arkitscenes_block_shuffle_size: Optional[int] = field(default=None,
        metadata={"help": "Block size for shuffling in collection mode (None to disable) for ARKitScenes"})

    # Temporal mode parameters - used for pure temporal sampling and Cut3r fallback
    temporal_scannet_max_interval: Optional[int] = field(default=30,
        metadata={"help": "Maximum frame gap for temporal mode and Cut3r fallback in ScanNet"})
    temporal_scannet_min_interval: Optional[int] = field(default=1,
        metadata={"help": "Minimum frame gap for temporal mode and Cut3r fallback in ScanNet"})

    temporal_scannetpp_max_interval: Optional[int] = field(default=3,
        metadata={"help": "Maximum frame gap for temporal mode and Cut3r fallback in ScanNet++"})
    temporal_scannetpp_min_interval: Optional[int] = field(default=1,
        metadata={"help": "Minimum frame gap for temporal mode and Cut3r fallback in ScanNet++"})

    temporal_arkitscenes_max_interval: Optional[int] = field(default=8,
        metadata={"help": "Maximum frame gap for temporal mode and Cut3r fallback in ARKitScenes"})
    temporal_arkitscenes_min_interval: Optional[int] = field(default=1,
        metadata={"help": "Minimum frame gap for temporal mode and Cut3r fallback in ARKitScenes"})

    vipe_cambrians_temporal_min_interval: Optional[int] = field(
        default=1,
        metadata={"help": "Min frame interval for temporal sampling of ViPE-Cambrians data"})
    vipe_cambrians_temporal_max_interval: Optional[int] = field(
        default=100,
        metadata={"help": "Max frame interval for temporal sampling (falls back if video too short)"})

    scale_by_points: bool = field(default=True)
    load_depth_data: bool = field(default=True)
    
    custom_system_prompt: Optional[str] = field(default=None)
    
    # Set None in script if not using mapanything dataset
    mapanything_data_path: Optional[str] = field(
        default=None,
    )
    
    mapanything_data_root: Optional[str] = field(
        default=None,   
    )
    
    # Covisibility threshold for each dataset
    # Covisibility threshold for MPSD dataset
    mpsd_covis_thres: Optional[float] = field(default=0.15)
    
    # Covisibility threshold for ETH3D dataset
    eth3d_covis_thres: Optional[float] = field(default=0.3) 
    
    # Covisibility threshold for all other MapAnything datasets
    others_covis_thres: Optional[float] = field(default=0.15)
    
    # Apply Sim(3) alignment for non-metric datasets (MegaDepth, DL3DV, BlendedMVS)
    # Required when training on datasets with non-metric scale
    use_scale_alignment: bool = field(default=False)

    # Normalize translation loss by trajectory length to prevent large scenes dominating
    use_trajectory_balance: bool = field(default=False)
    
    # Only use metric-scale MapAnything datasets (skip MegaDepth, DL3DV, BlendedMVS)
    use_metric_only: bool = field(default=False)

    # Set None in script if not using vipe dataset (dpsp and wsdg)
    vipe_data_path: Optional[str] = field(default=None)
    
    vipe_data_root: Optional[str] = field(default=None)
    
    vipe_cambrians_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to ViPE Cambrian-S scenes JSON"}
    )
    vipe_cambrians_merge_mode: Optional[str] = field(
        default="attach",
        metadata={"help": "How to merge ViPE-Cambrians with jsonl VQA. "
                  "'replace': append ViPE entries and remove overlapping jsonl VQA (dedup). "
                  "'attach': add rec fields to matching VQA entries in-place (preserves VQA composition)."}
    )
    vipe_cambrians_data_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root for Cambrian-S RGB videos"}
    )
    vipe_cambrians_results_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root for ViPE pose/depth results"}
    )
    
    # ViPE Cambrian-S scene filtering parameters
    # we filter scenes outside [min, max] set for velocity and rotation rate
    # Upper bound on max_velocity and max_rotation_rate
    vipe_cambrians_max_velocity: Optional[float] = field(default=None)
    vipe_cambrians_max_rotation_rate: Optional[float] = field(default=None)
    
    # Lower bound on max_velocity and max_rotation_rate
    vipe_cambrians_max_velocity_lower_bound: Optional[float] = field(default=None)
    vipe_cambrians_max_rotation_rate_lower_bound: Optional[float] = field(default=None)
    
    # Path to JSON with per-source include/exclude. None=use all.
    vipe_cambrians_source_config: Optional[str] = field(default=None)
    # MapAnything frame ordering
    sort_mapanything_frames: Optional[bool] = field(default=False)

    # VLM-based quality filter for ViPE-Cambrians scenes
    vlm_filter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to VLM quality flags JSON: {rel_path: [discard_reasons], ...}"}
    )
    vlm_filter_categories_to_exclude: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated categories to exclude. "
                  "Options: text_overlay,synthetic,screen_recording,blurry,quality_issues,moving_vehicle. "
                  "Example: --vlm_filter_categories_to_exclude text_overlay,synthetic,screen_recording"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    vggt_cam_projector_lr: Optional[float] = None
    cam_token_projector_lr: Optional[float] = None
    cam_mix_mlp_lr: Optional[float] = None
    cam_out_projector_lr: Optional[float] = None
    downstream_head_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    
    force_no_bp_vqa: Optional[bool] = field(default=False)
    adam_beta2: Optional[float] = field(default=0.999)
    
    unpad_image: Optional[bool] = field(default=False)
    rec_free_tail_ratio: float = field(default=0.0, metadata={"help": "Fraction of each epoch's tail that should be free of rec samples (e.g., 0.1 = last 10% rec-free). Default 0.0 = disabled."})
    

# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
        # import pdb; pdb.set_trace()
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}
    for source in sources:
        # Check if the first message's "from" field is not a string
        if not isinstance(source[0].get("from"), str):
            # Log a warning and optionally fix it
            print(f"Warning: unexpected 'from' value {source[0].get('from')} in sample. Converting to 'human'.")
            source[0]["from"] = "human"

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<video>"], special_tokens=True)
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    video_token_index = tokenizer.convert_tokens_to_ids("<video>")
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == video_token_index or encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            # if encode_id == cam_token_index:
            #     input_id[idx] = CAM_TOKEN_INDEX
                # target[idx] = IGNORE_INDEX
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], 
               tokenizer: transformers.PreTrainedTokenizer,
               has_image: bool = False, 
               custom_system_prompt: Optional[str] = "You are a helpful assistant."
               ) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image, system_message=custom_system_prompt)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.list_data_dict = []
        self.data_mode = data_args.data_mode
        self.data_args = data_args
        rank0_print(f"\033[91mWARNING: 'rec_resolution' is hardcoded to magic number 384 in {self.__class__.__name__}\033[0m")
        self.data_args.rec_resolution = 384
        self.camera_tokens_place = getattr(data_args, 'camera_tokens_place', 'prepend_to_frame')

        # Load each data source
        json_files = [data_args.json_file]
        self._load_main_jsonl_sources(json_files)
        if data_args.mapanything_data_path is not None:
            self._load_mapanything()
        if data_args.vipe_cambrians_path is not None:
            self._load_vipe_cambrians()
        if data_args.vipe_data_path is not None:
            self._load_vipe()

        rank0_print(f"Now total loaded samples: {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")

        if self.data_args.interleaved_training and self.data_args.load_rec_data:
            self._augment_for_interleaved_training()
        else:
            rank0_print("Not augmenting data for interleaved training")

        if self.data_args.keep_only_dummy_vqa:
            self.list_data_dict = [sample for sample in self.list_data_dict if sample.get('dummy_vqa', False)]

 
    def _load_main_jsonl_sources(self, json_files):
        """Load primary VQA jsonl/json file(s) into self.list_data_dict."""
        rank0_print(f"Loading datasets from {json_files}")
        for json_file in json_files:
            try:
                rank0_print(f"Loading {json_file}")
                # Detect file format based on extension
                if json_file.endswith('.jsonl'):
                    # Load JSONL format (one JSON object per line)
                    cur_data_dict = []
                    with open(json_file, "r") as file:
                        for line in file:
                            line = line.strip()
                            if line: # Skip empty lines
                                cur_data_dict.append(json.loads(line))
                else:
                    # Load JSON array format
                    with open(json_file, "r") as file:
                        cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_file}")
                self.list_data_dict.extend(cur_data_dict)
            except FileNotFoundError:
                rank0_print(f"Warning: File not found: {json_file}, skipping...")
            except Exception as e:
                rank0_print(f"Error loading {json_file}: {e}, skipping...")
        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {self.data_path}")

    def _load_mapanything(self):
        """Load MapAnything scenes."""
        data_args = self.data_args
        assert data_args.mapanything_data_root is not None, \
            "mapanything_data_root is required when mapanything_data_path is provided"

        mapanything_entries, skipped_count, skipped_reasons = ma_utils.load_mapanything_scenes_for_training(
            data_args.mapanything_data_path,
            data_args.mapanything_data_root,
            data_args.use_metric_only,
            False,
        )
        self.list_data_dict.extend(mapanything_entries)

        rank0_print(f"Skipped {skipped_count} non-metric scenes (use_metric_only={data_args.use_metric_only}):")
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
            rank0_print(f"  - {reason}: {count}")
        ma_datasets = {}
        for entry in mapanything_entries:
            ds = entry.get('source_dataset', 'unknown')
            ma_datasets[ds] = ma_datasets.get(ds, 0) + 1
        rank0_print(f"MapAnything dataset breakdown: {ma_datasets}")
        rank0_print(f"Loaded {len(mapanything_entries)} MapAnything scenes from {data_args.mapanything_data_path}")

    def _load_vipe_cambrians(self):
        """Load + filter + merge ViPE-Cambrians scenes."""
        data_args = self.data_args
        assert data_args.vipe_cambrians_data_root is not None, \
            "vipe_cambrians_data_root is required when vipe_cambrians_path is provided"

        # Parse VLM filter categories
        categories_to_exclude = getattr(data_args, 'vlm_filter_categories_to_exclude', None)
        if categories_to_exclude:
            categories_to_exclude = [x.strip() for x in categories_to_exclude.split(',')]

        vipe_cambrians_entries, skipped, reasons = vipe_utils.load_vipe_cambrians_scenes_for_training(
            data_args.vipe_cambrians_path,
            thorough_validation=False,
            vipe_cambrians_data_root=data_args.vipe_cambrians_data_root,
            vipe_cambrians_results_root=data_args.vipe_cambrians_results_root,
            vlm_filter_path=getattr(data_args, 'vlm_filter_path', None),
            categories_to_exclude=categories_to_exclude,
        )

        # Filter scenes based on trajectory constraints
        vipe_cambrians_entries = self._filter_vipe_cambrians_entries(
            vipe_cambrians_entries, data_args
        )

        merge_mode = getattr(data_args, 'vipe_cambrians_merge_mode', 'attach')
        rank0_print(f"[ViPE-Cambrians] merge_mode={merge_mode}")

        if merge_mode == "attach":
            # Attach rec metadata to matching VQA entries in-place.
            # VQA composition is preserved — no entries added or removed.
            attached_count, attached_scenes, total_vipe = vipe_utils.attach_vipe_rec_to_vqa_entries(
                vipe_cambrians_entries, self.list_data_dict
            )
            unmatched = total_vipe - len(attached_scenes)
            rank0_print(f"[ViPE-Cambrians] {total_vipe} scenes after filtering, "
                        f"attached rec to {attached_count} VQA entries ({len(attached_scenes)} scenes)")
            if unmatched > 0:
                rank0_print(f"[ViPE-Cambrians] WARNING: {unmatched} ViPE scenes had no matching VQA in jsonl")
        elif merge_mode == "replace":
            # Append ViPE entries, then remove overlapping jsonl VQA entries.
            # The ViPE version is kept because it has rec supervision.
            deduped_count, vipe_video_names, self.list_data_dict = vipe_utils.replace_vipe_vqa_entries(
                vipe_cambrians_entries, self.list_data_dict
            )
            rank0_print(f"[ViPE-Cambrians] Appended {len(vipe_cambrians_entries)} ViPE entries, "
                        f"deduped {deduped_count} overlapping VQA entries from jsonl")
        else:
            raise ValueError(f"Unknown vipe_cambrians_merge_mode: {merge_mode}. Use 'attach' or 'replace'.")

        self.vipe_cambrians_data_root = data_args.vipe_cambrians_data_root
        self.vipe_cambrians_results_root = data_args.vipe_cambrians_results_root

        if skipped > 0:
            rank0_print(f"[ViPE-Cambrians] Skipped: {skipped}, reasons: {reasons}")

    def _load_vipe(self):
        """Load ViPE (WSDG/W360/DPSP) scenes."""
        data_args = self.data_args
        assert data_args.vipe_data_root is not None, \
            "vipe_data_root is required when vipe_data_path is provided"

        vipe_entries, skipped, reasons = vipe_utils.load_vipe_scenes_for_training(
            data_args.vipe_data_path,
            thorough_validation=False,
            vipe_data_root=data_args.vipe_data_root,
        )
        self.list_data_dict.extend(vipe_entries)
        self.vipe_data_root = data_args.vipe_data_root  # Used in _get_item

        rank0_print(f"[ViPE] Loaded {len(vipe_entries)} scenes")
        rank0_print(f"[ViPE] Skipped: {skipped}")

    def _augment_for_interleaved_training(self):
        """Augment data for interleaved training (3 modes: scene_unified, same_as_vqa, num_frame)."""
        self.interleaved_aug_rec_ratio = self.data_args.interleaved_aug_rec_ratio
        # If vipe_interleaved_aug_rec_ratio is unset, default to interleaved_aug_rec_ratio
        # so existing configs behave exactly as before.
        vipe_ratio = self.data_args.vipe_interleaved_aug_rec_ratio
        if vipe_ratio is None:
            vipe_ratio = self.interleaved_aug_rec_ratio
        self.vipe_interleaved_aug_rec_ratio = vipe_ratio

        rank0_print(
            f"Augmenting data for interleaved training with mode "
            f"{self.data_args.interleaved_aug_mode}, scannet/ark ratio "
            f"{self.interleaved_aug_rec_ratio}, vipe-cams ratio "
            f"{self.vipe_interleaved_aug_rec_ratio}"
        )
        rec_subset, unique_scenes = self._tag_loading_type_for_each_sample_and_get_rec_subset_and_unique_scenes()

        # Split rec_subset / unique_scenes into vipe-cambrians vs the rest (scannetpp/ARKit/etc.)
        vipe_rec_subset = [s for s in rec_subset if vipe_utils.is_vipe_cambrians_scene(s)]
        non_vipe_rec_subset = [s for s in rec_subset if not vipe_utils.is_vipe_cambrians_scene(s)]
        vipe_unique_scenes = {
            v: s for v, s in unique_scenes.items() if vipe_utils.is_vipe_cambrians_scene(s)
        }
        non_vipe_unique_scenes = {
            v: s for v, s in unique_scenes.items() if not vipe_utils.is_vipe_cambrians_scene(s)
        }
        rank0_print(
            f"[interleaved aug] rec_subset split: "
            f"vipe-cams={len(vipe_rec_subset)} (scenes={len(vipe_unique_scenes)}), "
            f"non-vipe={len(non_vipe_rec_subset)} (scenes={len(non_vipe_unique_scenes)})"
        )

        mode = self.data_args.interleaved_aug_mode

        def _run_augment(ratio, subset, scenes):
            if ratio <= 0 or len(subset) == 0:
                return
            if mode == 'scene_unified':
                self.list_data_dict = self.augment_data_scene_unified(ratio, subset, scenes)
            elif mode == 'same_as_vqa':
                self.list_data_dict = self.augment_data(ratio, subset)
            elif mode == 'num_frame':
                self.list_data_dict = self.augment_data_num_frame(ratio, subset, scenes)
            else:
                raise ValueError(f"Invalid interleaved training mode: {mode}")

        _run_augment(self.interleaved_aug_rec_ratio, non_vipe_rec_subset, non_vipe_unique_scenes)
        _run_augment(self.vipe_interleaved_aug_rec_ratio, vipe_rec_subset, vipe_unique_scenes)

    def _tag_loading_type_for_each_sample_and_get_rec_subset_and_unique_scenes(self):
        """
        unique_scenes values are dicts (not strings) so dummy samples for
        ViPE-Cambrians scenes carry the rec fields (data_source, rgb_path, …)
        that is_vipe_cambrians_scene() requires. ScanNet scenes copy nothing
        extra — the dummy loads via the standard scene directory path.
        """
        rec_subset = []
        scene_video_path_dict = {}
        for sample in self.list_data_dict:
            # we dont need to consider the image samples
            if 'video' in sample and 'loading_type' not in sample:
                loading_type = vqa_utils.check_video_loading_type(sample['video'], None)
                sample['loading_type'] = loading_type

            if 'loading_type' in sample and sample['loading_type'] == 'rec':
                rec_subset.append(sample)
                scene_video_path_dict[sample['video']] = sample

        return rec_subset, scene_video_path_dict
    
    def augment_data_scene_unified(self, aug_rec_ratio, rec_subset, unique_scenes):
        # num_aug_samples = int(len(self.list_data_dict) * aug_rec_ratio)
        num_aug_samples = int(len(rec_subset) * aug_rec_ratio)
        rank0_print(f"Augmenting data for interleaved training with ratio {aug_rec_ratio}, num_aug_samples = {num_aug_samples}")
        # count all unique scenes in the dataset and their video path
        
        num_sample_per_scene = num_aug_samples // len(unique_scenes)
        for video_path, rep_sample in unique_scenes.items():
            question_value = '<video>pad question'
            if self.camera_tokens_place == 'between_qa':
                question_value = '<video>pad question<cam>'

            # load the video
            qa_info = {
                'video': video_path,
                'source_dataset': rep_sample.get('source_dataset', 'unknown'),
                'question_type': 'pad',
                'conversations': [
                    {
                        'from': 'human',
                        'value': question_value
                    },
                    {
                        'from': 'gpt',
                        'value': 'answer'
                    }
                ],
                'data_sample_mode': self.data_args.rec_data_mode,
                'loading_type': 'rec',
                'dummy_vqa': True,
            }
  
            # For ViPE-Cambrians scenes, this copies data_source, rgb_path,
            # pose_path, etc. so _get_item can route to the ViPE loader.
            for field in vipe_utils.REC_META_FIELDS:
                if field in rep_sample:
                    qa_info[field] = rep_sample[field]
            self.list_data_dict.extend([qa_info] * num_sample_per_scene)

        return self.list_data_dict

    def augment_data(self, aug_rec_ratio, rec_subset):
        num_aug_samples = int(len(rec_subset) * aug_rec_ratio)
        rank0_print(f"Augmenting data for interleaved training with ratio {aug_rec_ratio}, num_aug_samples = {num_aug_samples}")

        sampled_data = random.sample(rec_subset, num_aug_samples)

        for sample in sampled_data:
            cur_sample = copy.deepcopy(sample)
            cur_sample['question_type'] = 'pad'
            question_value = '<video>pad question'
            if self.camera_tokens_place == 'between_qa':
                question_value = '<video>pad question<cam>'

            cur_sample['conversations'] = [
                {
                    'from': 'human',
                    'value': question_value  # add <cam> token if between_qa
                },
                {
                    'from': 'gpt',
                    'value': 'answer'
                }
            ]
            cur_sample['data_sample_mode'] = self.data_args.rec_data_mode
            cur_sample['loading_type'] = 'rec' 
            cur_sample['dummy_vqa'] = True
            self.list_data_dict.append(cur_sample)

        return self.list_data_dict

    def _load_video_frame_counts(self):
        frame_count_path = getattr(self.data_args, "interleaved_aug_frame_count_path", None)
        if frame_count_path is None:
            raise ValueError("interleaved_aug_frame_count_path is required for num_frame mode")

        with open(frame_count_path, "r") as f:
            frame_counts = json.load(f)

        if not isinstance(frame_counts, dict):
            raise ValueError(f"Expected dict in {frame_count_path}, got {type(frame_counts)}")

        mapanything_root = getattr(self.data_args, "mapanything_data_root", None)
        mapanything_alias_added = 0
        mapanything_scene_count = 0
        if mapanything_root:
            norm_root = osp.normpath(mapanything_root)
            for sample in self.list_data_dict:
                # Only MapAnything scenes need absolute<->relative key aliasing.
                if not ma_utils.is_mapanything_source_dataset(sample.get("source_dataset", None)):
                    continue
                
                video_key = str(sample.get("video", "")).strip()
                if not video_key:
                    continue

                mapanything_scene_count += 1
                norm_video = osp.normpath(video_key)

                rel_key = None
                if osp.isabs(norm_video):
                    try:
                        # Convert "/root/.../scene" -> ".../scene" if it is under mapanything_data_root.
                        if osp.commonpath([norm_video, norm_root]) == norm_root:
                            rel_key = osp.relpath(norm_video, norm_root)
                    except ValueError:
                        rel_key = None
                else:
                    # Frame-count files may already use relative scene keys.
                    rel_key = norm_video

                if rel_key is None:
                    continue

                # Add alias so later lookup by absolute sample['video'] succeeds.
                if rel_key in frame_counts and video_key not in frame_counts:
                    frame_counts[video_key] = frame_counts[rel_key]
                    mapanything_alias_added += 1

        rank0_print(
            f"Loaded {len(frame_counts)} frame-count entries from {frame_count_path} "
            f"(mapanything_scenes={mapanything_scene_count}, mapanything_alias_added={mapanything_alias_added})"
        )
        return frame_counts

    def augment_data_num_frame(self, aug_rec_ratio, rec_subset, unique_scenes):
        num_aug_samples = int(len(rec_subset) * aug_rec_ratio)
        rank0_print(
            f"Augmenting data for interleaved training with num_frame mode, "
            f"ratio {aug_rec_ratio}, num_aug_samples = {num_aug_samples}"
        )
        if num_aug_samples <= 0 or len(unique_scenes) == 0:
            return self.list_data_dict

        frame_counts = self._load_video_frame_counts()
        videos = list(unique_scenes.keys())
        weights = []
        matched = 0
        unmatched_keys = []

        spring_weight = getattr(self.data_args, "spring_sample_weight", 1.0)
        spring_count = 0

        for video_path in videos:
            key = str(video_path).strip()
            raw_count = frame_counts.get(key, None)
            if raw_count is None:
                unmatched_keys.append(key)
                continue
            else:
                matched += 1
                weight = max(1, int(raw_count))

            if spring_weight != 1.0 and "spring" in unique_scenes[video_path].get('source_dataset', '').lower():
                weight *= spring_weight
                spring_count += 1

            weights.append(float(weight))

        if unmatched_keys:
            preview = ", ".join(unmatched_keys[:10])
            raise ValueError(
                f"num_frame mode requires all videos to exist in {self.data_args.interleaved_aug_frame_count_path}. "
                f"Missing {len(unmatched_keys)} / {len(videos)} keys. Examples: {preview}"
            )

        raw_min_w = min(weights)
        raw_max_w = max(weights)
        if raw_max_w > raw_min_w:
            weights = [(w - raw_min_w) / (raw_max_w - raw_min_w) for w in weights]
            if sum(weights) <= 0:
                weights = [1.0] * len(weights)
        else:
            # All videos have same frame count; fall back to uniform sampling weights.
            weights = [1.0] * len(weights)

        rank0_print(
            f"\033[31mnum_frame matching: matched={matched}, unmatched=0, "
            f"raw_weight_range=[{raw_min_w}, {raw_max_w}], "
            f"normalized_weight_range=[{min(weights)}, {max(weights)}]"
            f"{f', spring_sample_weight={spring_weight} applied to {spring_count} videos' if spring_count > 0 else ''}\033[0m"
        )

        sampled_videos = random.choices(videos, weights=weights, k=num_aug_samples)
        for video_path in sampled_videos:
            rep_sample = unique_scenes[video_path]
            question_value = '<video>pad question'
            if self.camera_tokens_place == 'between_qa':
                question_value = '<video>pad question<cam>'

            qa_info = {
                'video': video_path,
                'source_dataset': rep_sample.get('source_dataset', 'unknown'),
                'question_type': 'pad',
                'conversations': [
                    {
                        'from': 'human',
                        'value': question_value
                    },
                    {
                        'from': 'gpt',
                        'value': 'answer'
                    }
                ],
                'data_sample_mode': self.data_args.rec_data_mode,
                'loading_type': 'rec',
                'dummy_vqa': True,
            }
            # Same as augment_data_scene_unified: copy rec fields so ViPE dummies can load via is_vipe_cambrians_scene(). No-op for ScanNet.
            for field in vipe_utils.REC_META_FIELDS:
                if field in rep_sample:
                    qa_info[field] = rep_sample[field]
            self.list_data_dict.append(qa_info)

        return self.list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {os.path.join(image_folder, image_file)}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"
    
    def _filter_vipe_cambrians_entries(self, entries, data_args):
        """
        Filter ViPE-Cambrians entries: keep only entries where
            min_vel <= max_velocity <= max_vel
            min_rot <= max_rotation_rate <= max_rot
        """
        source_config = None
        source_config_path = getattr(data_args, 'vipe_cambrians_source_config', None)
        if source_config_path and os.path.exists(source_config_path):
            with open(source_config_path) as f:
                source_config = json.load(f)

        max_vel_upper_bound = getattr(data_args, 'vipe_cambrians_max_velocity', None)
        max_vel_lower_bound = getattr(data_args, 'vipe_cambrians_max_velocity_lower_bound', None)
        max_rot_upper_bound = getattr(data_args, 'vipe_cambrians_max_rotation_rate', None)
        max_rot_lower_bound = getattr(data_args, 'vipe_cambrians_max_rotation_rate_lower_bound', None)

        filtered = []
        source_rejected = 0
        reject_reasons = {}

        for entry in entries:
            # 1. Source check
            if source_config is not None:
                source = entry.get("source", entry.get("source_dataset", "").replace("cambrians_", ""))
                if not source_config.get(source, True):
                    source_rejected += 1
                    continue

            vel = entry.get("max_velocity", None)
            rot = entry.get("max_rotation_rate", None)

            # 2. Velocity range: min_vel <= max_velocity <= max_vel
            if max_vel_lower_bound is not None and vel is not None and vel < max_vel_lower_bound:
                reject_reasons["vel_too_low"] = reject_reasons.get("vel_too_low", 0) + 1
                continue
            if max_vel_upper_bound is not None and vel is not None and vel > max_vel_upper_bound:
                reject_reasons["vel_too_high"] = reject_reasons.get("vel_too_high", 0) + 1
                continue

            # 3. Rotation range: min_rot <= max_rotation_rate <= max_rot
            if max_rot_lower_bound is not None and rot is not None and rot < max_rot_lower_bound:
                reject_reasons["rot_too_low"] = reject_reasons.get("rot_too_low", 0) + 1
                continue
            if max_rot_upper_bound is not None and rot is not None and rot > max_rot_upper_bound:
                reject_reasons["rot_too_high"] = reject_reasons.get("rot_too_high", 0) + 1
                continue

            filtered.append(entry)

        from cambrianp.utils import rank0_print
        rank0_print(f"[ViPE-Cambrians] Filtering: {len(entries)} → {len(filtered)}")
        if source_config:
            excluded = [k for k, v in source_config.items() if not v and not k.startswith("_")]
            rank0_print(f"[ViPE-Cambrians] Excluded sources ({source_rejected}): {excluded}")
        rank0_print(f"[ViPE-Cambrians] Range: vel=[{max_vel_lower_bound}, {max_vel_upper_bound}], rot=[{max_rot_lower_bound}, {max_rot_upper_bound}]")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            rank0_print(f"[ViPE-Cambrians]   Rejected by {reason}: {count}")

        return filtered
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        timeout_s = float(getattr(self.data_args, "video_load_timeout", 120.0))

        if timeout_s > 0:
            def _fetch(idx):
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(self._get_item, idx)
                try:
                    result = future.result(timeout=timeout_s)
                    executor.shutdown(wait=False)
                    return result
                except FutureTimeoutError:
                    executor.shutdown(wait=False)
                    raise TimeoutError(f"Video loading timed out after {timeout_s}s")

            def _source_path(idx):
                entry = self.list_data_dict[idx]
                return entry.get("video", entry.get("image", "text_only"))

            for attempt_idx in range(3):
                try:
                    return _fetch(i)
                except Exception as e:
                    print(f"[Skip] sample={i} file={_source_path(i)} attempt={attempt_idx} error={type(e).__name__}: {e}", flush=True)

            for attempt_idx in range(3):
                next_index = random.choice([j for j in range(len(self.list_data_dict)) if j != i])
                try:
                    return _fetch(next_index)
                except Exception as e:
                    print(f"[Skip] fallback sample={next_index} file={_source_path(next_index)} attempt={attempt_idx} error={type(e).__name__}: {e}", flush=True)

            try:
                return _fetch(i)
            except Exception as e:
                print(f"[Skip] final sample={i} file={_source_path(i)} error={type(e).__name__}: {e}", flush=True)
                raise

        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = random.choice([j for j in range(len(self.list_data_dict)) if j != i])
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
        
    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Expected single source per sample"
        
        sample_info = f"Sample #{i}"
        modality = "text"  # default
        video_file = None
        interval_stats = None
        rec_views = None
        # === IMAGE PROCESSING ===
        if "image" in sources[0]:
            modality = "image"
            image_file = self.list_data_dict[i]["image"]
            
            if isinstance(image_file, list):
                image = [self.process_image(f) for f in image_file]
                # Multi-image handling with padding
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        
        # === VIDEO PROCESSING ===
        elif "video" in sources[0]:
            modality = "video"
            video_file = self.list_data_dict[i]["video"]
            video_file = os.path.join(self.data_path, video_file)
            scene_ids_file = osp.basename(video_file)
            
            loading_type = self.list_data_dict[i].get('loading_type', None)

            # this variable is used to distinguish between real_vqa and the dummy padded vqa samples
            is_real_vqa = 'data_sample_mode' not in self.list_data_dict[i]

            # VQA samples: use self.data_mode
            # Reconstruction data: use the configured sample mode or the one from data dict
            sample_mode = self.list_data_dict[i].get('data_sample_mode', self.data_mode)

            if not os.path.exists(video_file):
                print(f"File {video_file} not exist!")
                return self._get_item(i + 1)
            
            num_frames_to_sample = self.data_args.frames_upbound if self.data_args.force_sample else 32
            video = None
            
            allow_repeat = getattr(self, 'allow_repeat', False)  # by default, no repeat

            # Detect dataset type from path
            dataset_type = rec_utils.detect_dataset_type(video_file)

            # Determine video extension type and load accordingly
            # 'video_file' - MP4, AVI, etc.
            # 'rec_dir' - Directory with frames for reconstruction data
            loading_type = vqa_utils.check_video_loading_type(video_file, loading_type)

            if loading_type == 'video_file':
                clip_info = vqa_utils.extract_clip_info(self.list_data_dict[i], video_file, self.data_args)
                video = clip_info.video
                selected_basenames = [f"frame_{i:06d}" for i in range(len(video))]

            elif loading_type == 'rec':
                # Check data source in priority order
                if vipe_utils.is_vipe_cambrians_scene(self.list_data_dict[i]):
                    # ViPE-annotated Cambrian-S
                    vipe_sample_mode = self.data_args.vipe_cambrians_rec_data_mode
                        
                    try:
                        video, selected_basenames, rec_views = vipe_utils.load_vipe_cambrians_scene(
                            scene_entry=self.list_data_dict[i],
                            num_frames=num_frames_to_sample,
                            sample_mode=vipe_sample_mode,
                            vipe_cambrians_data_root=getattr(self, 'vipe_cambrians_data_root', None),
                            vipe_cambrians_results_root=getattr(self, 'vipe_cambrians_results_root', None),
                            min_interval=self.data_args.vipe_cambrians_temporal_min_interval,
                            max_interval=self.data_args.vipe_cambrians_temporal_max_interval,
                        )
                        dataset_type = self.list_data_dict[i].get("source_dataset", "vipe_cambrians")
                    except Exception as e:
                        print(f"[ViPE-Cambrians] Failed to load {video_file}: {e}")
                        import traceback
                        traceback.print_exc()
                        next_index = random.choice([j for j in range(len(self.list_data_dict)) if j != i])
                        return self._get_item(next_index)
                        
                elif vipe_utils.is_vipe_scene(self.list_data_dict[i]):
                    # ViPE format (WSDG/W360/DPSP) - video-based with ZIP depth
                    try:
                        video, selected_basenames, rec_views = vipe_utils.load_vipe_scene(
                            scene_entry=self.list_data_dict[i],
                            num_frames=num_frames_to_sample,
                            target_size=(192, 192),
                            target_size_llava=(384, 384),
                            sample_mode=sample_mode,
                            vipe_data_root=self.vipe_data_root,
                        )
                        dataset_type = self.list_data_dict[i].get("source_dataset", "vipe")
                    except Exception as e:
                        print(f"[ViPE] Failed to load {video_file}: {e}")
                        next_index = random.choice([j for j in range(len(self.list_data_dict)) if j != i])
                        return self._get_item(next_index)
                                    
                elif ma_utils.is_mapanything_scene(video_file):
                    # MapAnything format - loads video + rec_views together
                    try:
                        # Resolve per-dataset covisibility threshold
                        ma_dataset_type = ma_utils.detect_mapanything_dataset_type(video_file)
                        # if ma_dataset_type and 'mpsd' in ma_dataset_type.lower():
                        #     covis_thres = self.data_args.mpsd_covis_thres
                        # elif ma_dataset_type and 'eth3d' in ma_dataset_type.lower():
                        #     covis_thres = self.data_args.eth3d_covis_thres
                        # else:
                        #     covis_thres = self.data_args.others_covis_thres

                        video, selected_basenames, rec_views = ma_utils.load_mapanything_scene(
                            video_file=video_file,
                            num_frames=num_frames_to_sample,
                            target_size=(192, 192),
                            target_size_llava=(384, 384),
                            covisibility_thres=None,
                            sort_mapanything_frames=self.data_args.sort_mapanything_frames,
                        )
                        dataset_type = ma_dataset_type

                    except Exception as e:
                        # Log error and retry with different sample
                        print(f"[MapAnything] Failed to load {video_file}: {e}")
                        # Pick a random different index
                        next_index = random.choice([j for j in range(len(self.list_data_dict)) if j != i])
                        return self._get_item(next_index)
                else:
                    # Standard scene-based dataset (ScanNet, ScanNet++, ARKitScenes, CO3D)
                    assert dataset_type is not None, "dataset_type must be provided for scene_dir"
                    cut3r_params = rec_utils.build_cut3r_params(dataset_type, self.data_args) if sample_mode == 'cut3r' else None
                    temporal_params = rec_utils.build_temporal_params(dataset_type, self.data_args)
                    
                    video, selected_basenames, interval_stats = rec_utils.load_scene_frames(
                        video_file=video_file,
                        dataset_type=dataset_type,
                        sample_mode=sample_mode,
                        num_frames=num_frames_to_sample,
                        cut3r_params=cut3r_params,
                        temporal_params=temporal_params,
                        data_args=self.data_args,
                        allow_repeat=allow_repeat,
                    )
                    
                    # CO3D needs 'apple/' prefix
                    scene_ids_file_for_rec = rec_utils.adjust_scene_id_for_dataset(scene_ids_file, dataset_type)

                    strategy = StrategyFactory.get_strategy(
                        training=True,
                        interleaved=self.data_args.interleaved_training,
                        has_vqa=is_real_vqa
                    )
                    
                    rec_dataset = load_3r_dataset(
                        self.data_args,
                        scene_ids_file_for_rec,
                        dataset_type,
                        sample_mode,
                        input_use_augs=strategy.get_input_augs(self.data_args),
                        rec_use_augs=strategy.get_rec_augs(self.data_args),
                    )
                    
                    rec_views = rec_dataset.get_data(
                        seq_index=0,
                        img_per_seq=num_frames_to_sample, 
                        basenames=selected_basenames
                    )
            else:
                raise ValueError(f"Cannot determine how to load: {video_file}")
            
            if video is None or len(video) == 0:
                raise ValueError(f"Failed to load any frames from {video_file}")
            
            processed_frames, original_size, image = vqa_utils.process_video_frames(
                processor=self.data_args.image_processor,
                video=video,
                rec_views=rec_views,
                is_real_vqa=is_real_vqa,
            )

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # Tokenization
        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        
        custom_system_prompt = self.data_args.custom_system_prompt
        if self.data_args.custom_system_prompt is not None:
            data_dict = preprocess(sources, self.tokenizer, has_image=has_image, custom_system_prompt=custom_system_prompt)
        else:
            data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        
        prompt = data_dict.get("prompt", None)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        
        # Add image/video data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # No image/video, but model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), 
                (crop_size["width"], crop_size["height"]), "text"),
            ]
        
        if prompt is not None:
            data_dict["prompt"] = prompt
        
        data_dict["id"] = self.list_data_dict[i].get("id", i)
        
        if self.data_args.load_rec_data and rec_views is not None:
            data_dict["rec_views"] = rec_views
        
        data_dict["bp_vqa"] = True if not self.data_args.force_no_bp_vqa else False
        # Set bp_rec = False for samples without rec_views (e.g., video files)
        data_dict["bp_rec"] = rec_views is not None
        if self.data_args.interleaved_training and rec_views is not None:
            data_dict["bp_vqa"] = is_real_vqa
            data_dict["bp_rec"] = not is_real_vqa or self.data_args.force_bp_rec
        
        if interval_stats is not None:
            data_dict["frame_interval_stats"] = interval_stats
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            batch["images"] = images
            
        # prompt exist in the data
        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        # Collect rec_views from instances that have them (video files won't have rec_views)
        # Use a boolean mask to track which samples have rec_views (works correctly with FSDP/DDP)
        has_rec_views_mask = [("rec_views" in instance) for instance in instances]
        if any(has_rec_views_mask):
            batch_rec_views = [instance["rec_views"] for instance in instances if "rec_views" in instance]
            batch["rec_views"] = self.stack_rec_views_dicts(batch_rec_views)
            batch["has_rec_views_mask"] = has_rec_views_mask

        if "bp_vqa" in instances[0]:
            batch["bp_vqa"] = [instance["bp_vqa"] for instance in instances]
        
        if "bp_rec" in instances[0]:
            batch["bp_rec"] = [instance["bp_rec"] for instance in instances]

        interval_stats_list = [
            inst.get("frame_interval_stats") 
            for inst in instances 
            if inst.get("frame_interval_stats") is not None
        ]
        if interval_stats_list:
            avg_intervals = [s["avg_interval"] for s in interval_stats_list]
            min_intervals = [s["min_interval"] for s in interval_stats_list]
            max_intervals = [s["max_interval"] for s in interval_stats_list]
            
            batch["batch_avg_frame_interval"] = sum(avg_intervals) / len(avg_intervals)
            batch["batch_min_frame_interval"] = min(min_intervals) if min_intervals else 0
            batch["batch_max_frame_interval"] = max(max_intervals) if max_intervals else 0

        return batch

    def stack_rec_views_dicts(self, batch_rec_views):
        """
        Stack rec_views dictionaries across batch dimension.
        
        Input: 
            [
                {'images': [L, C, H, W], 'depths': [L, H, W], ...},  # Sample 0
                {'images': [L, C, H, W], 'depths': [L, H, W], ...},  # Sample 1
            ]
        
        Output:
            {
                'images': tensor([B, L, C, H, W]),
                'depths': tensor([B, L, H, W]),
                ...
            }
        """
        if not batch_rec_views:
            return {}
        
        batch_size = len(batch_rec_views)
        keys = batch_rec_views[0].keys()
        
        stacked_dict = {}
        
        for key in keys:
            values = [rec_view[key] for rec_view in batch_rec_views]
            
            if key in ['images', 'depths', 'extrinsics', 'intrinsics', 'cam_points', 
                       'world_points', 'original_size']:
                # Convert to tensors and stack along batch dimension
                tensors = []
                for v in values:
                    if isinstance(v, np.ndarray):
                        tensor = torch.from_numpy(v).float()
                        tensors.append(tensor)
                    elif isinstance(v, torch.Tensor):
                        tensors.append(v.float())
                    else:
                        raise ValueError(f"Unexpected type for {key}: {type(v)}")         
                try:
                    stacked_dict[key] = torch.stack(tensors, dim=0)
                except RuntimeError as e:
                    print(f"Warning: Could not stack {key} due to shape mismatch: {e}")
                    print(f"Shapes: {[t.shape for t in tensors]}")
                    stacked_dict[key] = tensors  # Keep as list if stacking fails
                    
            elif key == 'point_masks':
                tensors = []
                for v in values:
                    if isinstance(v, np.ndarray):
                        tensor = torch.from_numpy(v).bool()  
                        tensors.append(tensor)
                    elif isinstance(v, torch.Tensor):
                        tensors.append(v.bool())
                    else:
                        raise ValueError(f"Unexpected type for {key}: {type(v)}") 
                try:
                    stacked_dict[key] = torch.stack(tensors, dim=0)
                except RuntimeError as e:
                    print(f"Warning: Could not stack {key} due to shape mismatch: {e}")
                    stacked_dict[key] = tensors
                    
            elif key in ['ids']:
                # Integer arrays
                tensors = [torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v) 
                          for v in values]
                stacked_dict[key] = torch.stack(tensors, dim=0).long()
                
            elif key == 'is_video' or key == 'scale_by_points' or key == 'is_metric_scale':
                # Boolean arrays/scalars
                tensors = []
                for v in values:
                    if isinstance(v, np.ndarray):
                        tensors.append(torch.from_numpy(v).bool())
                    elif isinstance(v, torch.Tensor):
                        tensors.append(v.bool())
                    else:
                        tensors.append(torch.tensor(v).bool())
                stacked_dict[key] = torch.stack(tensors, dim=0).bool()
                            
            elif key in ['seq_name', 'basename', 'image_path','sample_mode']:
                # String lists - keep as nested list
                stacked_dict[key] = values
                
            else:
                # Default: try to stack as tensor
                try:
                    if isinstance(values[0], (list, np.ndarray)):
                        tensors = [torch.tensor(v) for v in values]
                        stacked_dict[key] = torch.stack(tensors, dim=0)
                    else:
                        stacked_dict[key] = values
                except:
                    stacked_dict[key] = values
        
        return stacked_dict
    
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if model_args.use_scale_alignment:
        assert model_args.use_trajectory_balance, "use_scale_alignment requires use_trajectory_balance to be True"
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    
    # Always load the correct config class based on model type
    if ("qwen" in model_args.model_name_or_path.lower() 
        or "cambrian" in model_args.model_name_or_path.lower()
    ):
        from cambrianp.model.language_model.llava_qwen import LlavaQwenConfig
        cfg_pretrained = LlavaQwenConfig.from_pretrained(model_args.model_name_or_path)
        
        # if load a cambrian model, cast the config to LlavaQwenConfig
        if 'cambrian' in model_args.model_name_or_path.lower():
            # start to cast the config to LlavaQwenConfig
            from cambrianp.utils import cast_cambrian_config_to_llava_ov_style
            cfg_pretrained = cast_cambrian_config_to_llava_ov_style(cfg_pretrained, model_args.model_name_or_path)
        
    elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        from cambrianp.model.language_model.llava_mistral import LlavaMistralConfig
        cfg_pretrained = LlavaMistralConfig.from_pretrained(model_args.model_name_or_path)
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        from cambrianp.model.language_model.llava_llama import LlavaConfig
        cfg_pretrained = LlavaConfig.from_pretrained(model_args.model_name_or_path)
    else:
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
    
    overwrite_config = {}
    
    overwrite_config["custom_llm_ce_loss"] = model_args.custom_llm_ce_loss
    overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
     
    overwrite_config["token_num"] = model_args.token_num
    overwrite_config["load_rec_model"] = model_args.load_rec_model
    overwrite_config["enable_rec_pose_viz"] = model_args.enable_rec_pose_viz
    overwrite_config["rec_pose_viz_frequency"] = model_args.rec_pose_viz_frequency
    overwrite_config["rescale_pred_not_gt"] = model_args.rescale_pred_not_gt
    overwrite_config["camera_tokens_mode"] = model_args.camera_tokens_mode
    overwrite_config["query_mode"] = model_args.query_mode
    overwrite_config["camera_tokens_place"] = model_args.camera_tokens_place
    overwrite_config["head_name"] = model_args.head_name
    overwrite_config["rec_projector_type"] = model_args.rec_projector_type
    overwrite_config["query_loss_frame_idx"] = model_args.query_loss_frame_idx
    overwrite_config["rec_loss_weight"] = model_args.rec_loss_weight
    overwrite_config["rec_camera_loss_weight"] = model_args.rec_camera_loss_weight
    overwrite_config["rec_camera_loss_weight_trans"] = model_args.rec_camera_loss_weight_trans
    overwrite_config["rec_camera_loss_weight_rot"] = model_args.rec_camera_loss_weight_rot
    overwrite_config["rec_camera_loss_weight_focal"] = model_args.rec_camera_loss_weight_focal
    
    overwrite_config["rec_camera_loss_type"] = model_args.rec_camera_loss_type
    overwrite_config["rec_camera_loss_components"] = model_args.rec_camera_loss_components
    overwrite_config["use_scale_alignment"] = model_args.use_scale_alignment
    overwrite_config["use_trajectory_balance"] = model_args.use_trajectory_balance
    overwrite_config["use_point_masks"] = model_args.use_point_masks
    overwrite_config["rec_depth_loss_weight"] = model_args.rec_depth_loss_weight
    overwrite_config["rec_depth_gradient_loss_fn"] = model_args.rec_depth_gradient_loss_fn
    overwrite_config["rec_depth_valid_range"] = model_args.rec_depth_valid_range
    overwrite_config["enable_point"] = model_args.enable_point
    overwrite_config["enable_depth"] = model_args.enable_depth
    overwrite_config["enable_camera"] = model_args.enable_camera
    overwrite_config["num_intermediate_layers"] = model_args.num_intermediate_layers
    overwrite_config["rec_model_path"] = model_args.rec_model_path
    overwrite_config["rec_embed_dim"] = model_args.rec_embed_dim
    overwrite_config["rec_camera_trunk_depth"] = model_args.rec_camera_trunk_depth
    overwrite_config["rec_camera_causal_attn"] = model_args.rec_camera_causal_attn
    overwrite_config["rec_dpt_head_type"] = model_args.rec_dpt_head_type
    overwrite_config["rec_depth_head_patch_start_idx"] = model_args.rec_depth_head_patch_start_idx
    overwrite_config["mm_img_tok_num"] = model_args.mm_img_tok_num

    overwrite_config["unpad_image"] = training_args.unpad_image

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode
    
    # Apply any overwrite configs
    if overwrite_config:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

    # Always pass the config to ensure correct model initialization
    customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif ("qwen" in model_args.model_name_or_path.lower() 
            or "cambrian" in model_args.model_name_or_path.lower()
        ):
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                from cambrianp.utils import load_model_state_dict, verify_model_weights
                
                is_cambrian = 'cambrian' in model_args.model_name_or_path.lower()
                new_state_dict = load_model_state_dict(
                    model_args.model_name_or_path, 
                    remap_cambrian=is_cambrian,
                )
                
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                
                if new_state_dict is not None:
                    model.load_state_dict(new_state_dict, strict=False)
                    
                    verify_model_weights(model, new_state_dict)
                    
                    del new_state_dict
                    torch.cuda.empty_cache()

        elif "gemma" in model_args.model_name_or_path.lower():
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.force_no_bp_vqa = training_args.force_no_bp_vqa
    data_args.camera_tokens_place = model_args.camera_tokens_place
    data_args.enable_camera = model_args.enable_camera
    data_args.enable_point = model_args.enable_point
    data_args.enable_depth = model_args.enable_depth
    model_args.use_scale_alignment = data_args.use_scale_alignment
    model_args.use_trajectory_balance = data_args.use_trajectory_balance

    # assert data_args.load_depth_data
    
    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # Set seed before model initialization for reproducibility
    set_seed(training_args.seed)
    rank0_print(f"Set seed to {training_args.seed} before model initialization")
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif ("qwen" in model_args.model_name_or_path.lower() 
        or "cambrian" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.load_rec_model:
        rec_head = model.get_model().rec_head
        if hasattr(rec_head, "depth_head") and rec_head.depth_head is not None:
            rec_head.depth_head.apply(rec_head._init_weights)
            rank0_print("Re-initialized depth_head weights after from_pretrained")
        rec_head.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                # import ipdb;
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

        ### Deciding train which part of the model
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)

        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                        param.requires_grad_(True)
            if "mm_rec_head" in tunable_parts:
                for name, param in model.named_parameters():
                    if "rec_head" in name:
                        param.requires_grad_(True)
            
        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        
        # print all the trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                rank0_print(f"Trainable parameter: {name}")
        
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            model.get_model().rec_head.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        model.config.vggt_cam_projector_lr = training_args.vggt_cam_projector_lr
        model.config.cam_token_projector_lr = training_args.cam_token_projector_lr
        model.config.cam_mix_mlp_lr = training_args.cam_mix_mlp_lr
        model.config.cam_out_projector_lr = training_args.cam_out_projector_lr
        model.config.downstream_head_lr = training_args.downstream_head_lr
        
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    model.tokenizer = tokenizer
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # Clean up incomplete checkpoints before resuming (e.g. from preemption mid-save)
    checkpoint_dirs = sorted(
        pathlib.Path(training_args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if training_args.local_rank in (0, -1):
        import shutil
        for ckpt_dir in list(reversed(checkpoint_dirs)):
            if not (ckpt_dir / "trainer_state.json").exists():
                print(f"WARNING: Incomplete checkpoint detected (missing trainer_state.json), removing: {ckpt_dir}")
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                checkpoint_dirs.remove(ckpt_dir)
            else:
                break  # older checkpoints should be fine
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # Re-scan after cleanup so all ranks agree
    checkpoint_dirs = sorted(
        pathlib.Path(training_args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )

    if checkpoint_dirs:
        print(f">>>>>>>>>>>>>>>>!!!!!!! REACHED RESUMING POINT (resuming from {checkpoint_dirs[-1].name})")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # Added to fix config bugs when models are saved
    # if "llava-onevision" in model_args.model_name_or_path.lower():   
    #     model.config.tie_word_embeddings = True
    if '0.5b' in model_args.model_name_or_path.lower():
        model.config.tie_word_embeddings = True
    if "llava-video" in model_args.model_name_or_path.lower():
        model.config.vocab_size = 152064

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
