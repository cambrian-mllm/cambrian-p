#!/usr/bin/env python3
"""Streaming pose-estimation harness for Cambrian-P (LLaVA + VGGT reconstructor).

Supports configurable frame sampling strategies and datasets (scannet, tum, sintel).
"""

import os
import math
os.environ['NCCL_P2P_DISABLE'] = '1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import warnings
from copy import deepcopy
from datetime import datetime
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
from accelerate import PartialState
from safetensors.torch import load_file
from safetensors import safe_open

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_timing_dict(per_frame, num_frames, one_time_costs):
    """Build a standardized timing dict from per-frame measurements.
    
    Each per_frame entry should have: encode_ms, forward_ms, decode_ms, total_ms.
    one_time_costs: dict of costs that happen once (rec_head_ms, text_prefill_ms, etc.)
    """
    # Add backward-compat keys to per-frame entries
    for pf in per_frame:
        pf.setdefault("llm_time", pf["forward_ms"] / 1000)
        pf.setdefault("rec_time", pf["decode_ms"] / 1000)
        pf.setdefault("prep_time", pf["encode_ms"] / 1000)
        pf.setdefault("total_time", pf["total_ms"] / 1000)
        pf.setdefault("num_input_frames", pf["frame_id"] + 1)
    
    encode_times = [pf["encode_ms"] for pf in per_frame]
    forward_times = [pf["forward_ms"] for pf in per_frame]
    decode_times = [pf["decode_ms"] for pf in per_frame]
    total_times = [pf["total_ms"] for pf in per_frame]
    
    summary = {
        "mean_encode_ms": float(np.mean(encode_times)),
        "mean_forward_ms": float(np.mean(forward_times)),
        "mean_decode_ms": float(np.mean(decode_times)),
        "mean_total_ms": float(np.mean(total_times)),
        "median_total_ms": float(np.median(total_times)),
        "min_total_ms": float(np.min(total_times)),
        "max_total_ms": float(np.max(total_times)),
        "std_total_ms": float(np.std(total_times)),
        "sum_encode_ms": float(np.sum(encode_times)),
        "sum_forward_ms": float(np.sum(forward_times)),
        "sum_decode_ms": float(np.sum(decode_times)),
        "sum_total_ms": float(np.sum(total_times)),
    }
    
    return {
        "num_frames": num_frames,
        "per_frame_times": per_frame,
        "per_frame": per_frame,
        "one_time": one_time_costs,
        "summary": summary,
        # Backward compat keys (seconds)
        "total_llm_time": summary["sum_forward_ms"] / 1000,
        "total_rec_time": one_time_costs.get("rec_head_ms", 0) / 1000,
        "total_model_time": summary["sum_total_ms"] / 1000,
        "llm_time": summary["sum_forward_ms"] / 1000,
        "rec_time": one_time_costs.get("rec_head_ms", 0) / 1000,
        "prep_time": summary["sum_encode_ms"] / 1000,
        "decode_time": summary["sum_decode_ms"] / 1000,
        "total_time": summary["sum_total_ms"] / 1000,
    }

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

from metadata import dataset_metadata
from relpose_utils import load_traj, get_tum_poses, eval_metrics as _eval_metrics_orig


def eval_metrics(pred_traj, gt_traj=None, seq="", filename="", sample_stride=1, correct_scale=True):
    """Wrapper around relpose_utils.eval_metrics with configurable scale correction.
    
    When correct_scale=True (default): Sim(3) alignment (matches original behavior).
    When correct_scale=False: SE(3) alignment (rigid, no scale).
    
    If correct_scale=True, just delegates to the original function.
    If correct_scale=False, reimplements using evo with correct_scale=False.
    """
    if correct_scale:
        return _eval_metrics_orig(pred_traj, gt_traj, seq=seq, filename=filename, sample_stride=sample_stride)
    
    # SE(3) alignment path — same logic as original but correct_scale=False
    try:
        from evo.core import sync
        from evo.core.metrics import PoseRelation, Unit
        from evo.core.trajectory import PoseTrajectory3D
        import evo.main_ape as main_ape
        import evo.main_rpe as main_rpe
    except ImportError as e:
        print(f"Warning: evo import failed ({e}), falling back to Sim(3)")
        return _eval_metrics_orig(pred_traj, gt_traj, seq=seq, filename=filename, sample_stride=sample_stride)
    
    from relpose_utils import make_traj
    
    if sample_stride > 1:
        pred_traj = (pred_traj[0][::sample_stride], pred_traj[1][::sample_stride])
        if gt_traj is not None:
            gt_traj = (gt_traj[0][::sample_stride], gt_traj[1][::sample_stride])
    
    pred_traj_evo = make_traj(pred_traj)
    if gt_traj is not None:
        gt_traj_evo = make_traj(gt_traj)
        if pred_traj_evo.timestamps.shape[0] == gt_traj_evo.timestamps.shape[0]:
            pred_traj_evo.timestamps = gt_traj_evo.timestamps
        gt_traj_evo, pred_traj_evo = sync.associate_trajectories(gt_traj_evo, pred_traj_evo)
    else:
        return 0.0, 0.0, 0.0
    
    # ATE with SE(3) alignment
    ate_result = main_ape.ape(
        gt_traj_evo, pred_traj_evo,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True, correct_scale=False,
    )
    ate = ate_result.stats["rmse"]
    
    # RPE rotation
    rpe_rots_result = main_rpe.rpe(
        gt_traj_evo, pred_traj_evo,
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=True, correct_scale=False,
        delta=1, delta_unit=Unit.frames,
        rel_delta_tol=0.01, all_pairs=True,
    )
    rpe_rot = rpe_rots_result.stats["rmse"]
    
    # RPE translation
    rpe_transs_result = main_rpe.rpe(
        gt_traj_evo, pred_traj_evo,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True, correct_scale=False,
        delta=1, delta_unit=Unit.frames,
        rel_delta_tol=0.01, all_pairs=True,
    )
    rpe_trans = rpe_transs_result.stats["rmse"]
    
    return ate, rpe_trans, rpe_rot

MONST3R_OVERRIDES = {
    "scannet": {
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
    },
    "scannet-val": {
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
    },
    "tum": {
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_90.txt"
        ),
    },
    # Sintel: no override needed — uses default metadata paths.
    # Sintel sequences are ~50 frames, no stride-3 subsampling.
}


# =============================================================================
# Argument Parser
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser(description="Unified Pose Estimation Evaluation")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["llava_vggt"])
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./eval_results")

    # Dataset configuration
    parser.add_argument("--eval_dataset", type=str, default="scannet",
                        choices=list(dataset_metadata.keys()))
    parser.add_argument("--size", type=int, default=384,
                        help="Image size fed to the model")
    parser.add_argument("--pose_eval_stride", type=int, default=10)
    parser.add_argument("--full_seq", action="store_true", default=False)

    # Frame sampling
    parser.add_argument("--sampling_strategy", type=str, default="uniform",
                        choices=["uniform", "continuous", "monst3r"],)
    parser.add_argument("--num_frames_per_scene", type=int, default=32)

    # Model-specific options
    parser.add_argument("--rec_embed_dim", type=int, default=2048,
                        help="Reconstruction embedding dim (LLaVA-VGGT)")
    parser.add_argument("--camera_tokens_place", type=str, default="append_to_frame",
                        choices=["append_to_frame"])

    # LLaVA-VGGT inference mode
    parser.add_argument("--llava_mode", type=str, default="batch",
                        choices=["batch", "streaming"],
                        help="Inference mode: 'batch' (default, single forward) "
                             "or 'streaming' (times LLM + rec_head separately)")
    
    # Alignment mode for trajectory evaluation
    parser.add_argument("--alignment", type=str, default="sim3",
                        choices=["sim3", "se3"],
                        help="Trajectory alignment: 'sim3' (default, with scale correction) "
                             "or 'se3' (rigid, no scale correction)")
    
    # Evaluation options
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_timestamp", action="store_true", default=False)
    parser.add_argument("--random_init", action="store_true", default=False,
                        help="Use random initialization (baseline)")
    parser.add_argument("--reinit_camera_head", action="store_true", default=False,
                        help="Re-init camera head randomly after loading (llava_vggt only)")
    
    # Visualization
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz_every", type=int, default=1)
    
    return parser

def apply_monst3r_overrides(args, metadata):
    dataset = args.eval_dataset

    # Override args
    args.pose_eval_stride = 1
    args.num_frames_per_scene = None  # Use ALL frames

    if dataset in MONST3R_OVERRIDES:
        overrides = MONST3R_OVERRIDES[dataset]
        for key, val in overrides.items():
            metadata[key] = val
        print(f"[monst3r] Applied path overrides for '{dataset}'")

    print("[monst3r] Protocol settings: stride=1, all frames")

    return metadata


# =============================================================================
# Model Loading
# =============================================================================

def load_llava_vggt_model(model_path, device="cuda", rec_embed_dim=None, camera_tokens_place=None):
    """Load LLaVA model with VGGT reconstructor."""
    from cambrianp.model.builder import load_pretrained_model
    from cambrianp.mm_utils import get_model_name_from_path
    from cambrianp.model.language_model.llava_qwen import LlavaQwenConfig
    
    print(f"Loading LLaVA-VGGT model from: {model_path}")
    
    cfg = LlavaQwenConfig.from_pretrained(model_path)
    cfg.load_rec_model = True
    
    if rec_embed_dim is not None:
        cfg.rec_embed_dim = rec_embed_dim
    
    model_name = get_model_name_from_path(model_path)
    
    overwrite_config = {
        'use_camera_tokens': True,
        'camera_token_num': 2,
        'camera_tokens_mode': getattr(cfg, 'camera_tokens_mode', 'camera_tokens'),
        'mm_spatial_pool_stride': getattr(cfg, 'mm_spatial_pool_stride', 2),
        'mm_spatial_pool_mode': getattr(cfg, 'mm_spatial_pool_mode', 'stride'),
        'tie_word_embeddings': True,
        'load_rec_model': True,
        'camera_tokens_place': camera_tokens_place,
    }
    
    if rec_embed_dim is not None:
        overwrite_config["rec_embed_dim"] = rec_embed_dim
    
    device_map = {"": device} if torch.cuda.is_available() else "cpu"
    
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        multimodal=True,
        overwrite_config=overwrite_config
    )
    
    _fix_layerscale_params(model, model_path)
    
    model.eval()
    return tokenizer, model.to(dtype=torch.bfloat16, device=device), image_processor


def _fix_layerscale_params(model, checkpoint_path):
    """Load LayerScale parameters from checkpoint."""
    layerscale_params = {
        name: param for name, param in model.named_parameters()
        if '.ls1.gamma' in name or '.ls2.gamma' in name
    }
    
    if not layerscale_params:
        return
    
    single_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(single_path):
        ckpt = load_file(single_path)
        for name, param in layerscale_params.items():
            if name in ckpt:
                with torch.no_grad():
                    param.copy_(ckpt[name].to(param.dtype).to(param.device))
        return
    
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f).get("weight_map", {})
        
        shard_to_params = {}
        for name in layerscale_params:
            shard = weight_map.get(name)
            if shard:
                shard_to_params.setdefault(shard, []).append(name)
        
        for shard_name, param_names in shard_to_params.items():
            shard_path = os.path.join(checkpoint_path, shard_name)
            if os.path.exists(shard_path):
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for name in param_names:
                        try:
                            tensor = f.get_tensor(name)
                            param = layerscale_params[name]
                            with torch.no_grad():
                                param.copy_(tensor.to(param.dtype).to(param.device))
                        except Exception:
                            pass


# =============================================================================
# Pose Extraction - Model Specific
# =============================================================================

def extract_pose_random(num_frames, debug=False):
    """Generate fully random camera poses as a baseline.
    
    Each pose is a random SE(3) matrix:
    - Rotation: random quaternion → SO(3) via scipy
    - Translation: random from N(0, 1)
    
    Returns c2w poses (convention doesn't matter for random).
    """
    poses = []
    for i in range(num_frames):
        R = Rotation.random().as_matrix()
        t = np.random.randn(3)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        poses.append(torch.from_numpy(pose).float())
    
    cam_dict = {
        "focal": np.ones(num_frames) * 256.0,
        "pp": np.tile(np.array([256.0, 256.0]), (num_frames, 1)),
    }
    
    if debug:
        print(f"    [random] {num_frames} frames, pose[0]:\n{poses[0]}")
    
    return torch.stack(poses), cam_dict


def extract_pose_llava(model, video_frames, tokenizer, debug=False):
    """
    Extract poses from LLaVA-VGGT using single forward pass with detailed timing.
    """
    from cambrianp.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from cambrianp.conversation import conv_templates
    from cambrianp.mm_utils import tokenizer_image_token
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    import copy, time as _time
    
    device = model.device
    num_frames = video_frames.shape[0]
    
    # Build prompt
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\nDescribe this scene.")
    conv.append_message(conv.roles[1], None)
    
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    
    bp_rec = {
        'images': torch.zeros(1, num_frames, 3, 192, 192, device=device, dtype=torch.bfloat16),
    }
    
    raw_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        # Step 1: Prepare inputs (vision encode + spatial pool + token layout + camera tokens)
        torch.cuda.synchronize()
        t_prep = _time.perf_counter()
        
        prep_result = raw_model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None,
            [video_frames], ["video"],
            image_sizes=None,
        )
        inputs_embeds = prep_result[4]
        position_ids = prep_result[1]
        attention_mask = prep_result[2]
        info_dict_list = prep_result[6]
        
        torch.cuda.synchronize()
        prep_ms = (_time.perf_counter() - t_prep) * 1000
        
        if debug:
            print(f"    [llava-batch] Step 1 PREPARE: {prep_ms:.1f}ms "
                  f"(vision encode + spatial pool + token layout)")
            print(f"    [llava-batch] inputs_embeds: {inputs_embeds.shape}")
        
        # Step 2: Single LLM forward pass (all frames at once)
        torch.cuda.synchronize()
        t_llm = _time.perf_counter()
        
        outputs = raw_model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        
        torch.cuda.synchronize()
        llm_ms = (_time.perf_counter() - t_llm) * 1000
        
        if debug:
            print(f"    [llava-batch] Step 2 LLM FORWARD: {llm_ms:.1f}ms "
                  f"(single pass, {num_frames} frames, {inputs_embeds.shape[1]} tokens)")
        
        # Step 3: rec_head (projector + camera_head)
        torch.cuda.synchronize()
        t_rec = _time.perf_counter()
        
        rec_head = raw_model.model.rec_head
        predictions = rec_head(outputs, info_dict_list=info_dict_list, batch=bp_rec)
        
        torch.cuda.synchronize()
        rec_ms = (_time.perf_counter() - t_rec) * 1000
        
        if debug:
            print(f"    [llava-batch] Step 3 REC_HEAD: {rec_ms:.1f}ms "
                  f"(projector + CameraHead + pose decode)")
    
    # Step 4: Pose decode (pose_encoding_to_extri_intri)
    pose_enc = predictions['pose_enc']
    
    torch.cuda.synchronize()
    t_dec = _time.perf_counter()
    
    with torch.cuda.amp.autocast(dtype=torch.float32):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (384, 384))
        if extrinsic.dim() > 3:
            extrinsic = extrinsic[0]
        if intrinsic.dim() > 3:
            intrinsic = intrinsic[0]
    
    batch_size = extrinsic.shape[0]
    poses_c2w = []
    for i in range(batch_size):
        R = extrinsic[i, :3, :3]
        t = extrinsic[i, :3, 3]
        T_inv = torch.eye(4, device=device)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        poses_c2w.append(T_inv)
    
    torch.cuda.synchronize()
    dec_ms = (_time.perf_counter() - t_dec) * 1000
    
    cam_dict = {
        "focal": intrinsic[:, 0, 0].cpu().numpy(),
        "pp": intrinsic[:, :2, 2].cpu().numpy(),
    }
    
    total_ms = prep_ms + llm_ms + rec_ms + dec_ms
    
    # Print summary
    print(f"\n    [llava-batch] === BATCH TIMING SUMMARY ({num_frames} frames) ===")
    print(f"    1_prepare (encode+pool+tokens): {prep_ms:8.1f}ms ({prep_ms/total_ms*100:5.1f}%)")
    print(f"    2_llm_forward (single pass):    {llm_ms:8.1f}ms ({llm_ms/total_ms*100:5.1f}%)")
    print(f"    3_rec_head (proj+camhead):      {rec_ms:8.1f}ms ({rec_ms/total_ms*100:5.1f}%)")
    print(f"    4_pose_decode (enc→extri):       {dec_ms:8.1f}ms ({dec_ms/total_ms*100:5.1f}%)")
    print(f"    TOTAL:                          {total_ms:8.1f}ms")
    print(f"    Amortized per frame:            {total_ms/num_frames:8.1f}ms/frame")
    print()
    
    # Build timing_dict for compatibility with timing analysis
    per_frame = []
    amort = total_ms / num_frames
    for i in range(num_frames):
        per_frame.append({
            "frame_id": i,
            "encode_ms": prep_ms / num_frames,
            "forward_ms": llm_ms / num_frames,
            "decode_ms": (rec_ms + dec_ms) / num_frames,
            "total_ms": amort,
        })
    
    detailed_steps = [
        {
            "step": "1_prepare_inputs",
            "description": f"SigLIP vision tower (batch {num_frames} frames) + mm_projector + spatial pool + camera token injection + token layout",
            "time_ms": prep_ms,
            "per_frame": True,
            "per_frame_ms": prep_ms / num_frames,
            "num_frames": num_frames,
        },
        {
            "step": "2_llm_forward",
            "description": f"Qwen2 LLM single forward pass ({inputs_embeds.shape[1]} tokens, {num_frames} frames). No KV cache — processes all at once.",
            "time_ms": llm_ms,
            "per_frame": True,
            "per_frame_ms": llm_ms / num_frames,
            "num_frames": num_frames,
        },
        {
            "step": "3_rec_head",
            "description": f"Projector MLP + CameraHead (bidirectional, all {num_frames} frames at once) + pose_encoding_to_extri_intri",
            "time_ms": rec_ms + dec_ms,
            "per_frame": True,
            "per_frame_ms": (rec_ms + dec_ms) / num_frames,
            "num_frames": num_frames,
        },
    ]
    
    timing_dict = _build_timing_dict(per_frame, num_frames, {
        "detailed_steps": detailed_steps,
        "prep_ms": prep_ms,
        "llm_ms": llm_ms,
        "rec_ms": rec_ms,
        "dec_ms": dec_ms,
    })
    
    return torch.stack(poses_c2w), cam_dict, timing_dict


def extract_pose_llava_streaming(model, video_frames, tokenizer, debug=False):
    """
    Cambrian-P TRUE per-frame streaming: RGB in → pose out for each frame.
    
    Measures the honest wall-clock time from receiving a new RGB frame
    to obtaining the camera extrinsic + intrinsic for that frame.
    
    Pipeline per frame i:
    ==========================================================================
    
    STAGE 1 — ENCODE (per frame):
      1a. vision_tower(frame_i)           — SigLIP encodes 1 frame → patch features
      1b. mm_projector(patch_features)    — MLP projects to LLM embedding dim
      1c. camera_token injection          — append learnable camera token to frame tokens
      1d. add newline tokens              — add image_newline per row (LLaVA convention)
      Result: frame_embeds_i of shape [1, tokens_per_frame, D]
    
    STAGE 2 — LLM FORWARD (per frame, with KV cache):
      For frame 0: feed [system_prompt_tokens, frame_0_tokens]
      For frame i>0: feed [frame_i_tokens] only (KV cache has history)
      For last frame: also feed text_suffix_tokens ("Describe this scene.\nassistant\n")
      model.model(..., past_key_values=kv_cache, use_cache=True, output_hidden_states=True)
      Result: hidden_states at all layers for this chunk
    
    STAGE 3 — DECODE (per frame, uses all frames 0..i):
      3a. Extract cam+image tokens from hidden states at intermediate layers
          Accumulate into growing rec_feats [B, i+1, 1+P, D] per layer
      3b. forward_projector: per-layer MLP projection
      3c. camera_head: 4-layer bidirectional transformer (NO cache, re-runs on all 0..i)
          → pose_enc for frames 0..i
      3d. pose_encoding_to_extri_intri: decode pose_enc[i] → extrinsic + intrinsic
      Result: c2w pose matrix + focal/pp for frame i
    
    NOTE on CameraHead:
      The CameraHead is a small bidirectional transformer (trunk_depth=4 blocks).
      It has NO KV cache — it does cross-frame self-attention.
      So at frame i, it processes all frames 0..i. Cost grows O(i²) but the
      CameraHead is tiny compared to the LLM, so this is still cheap.
      This is the HONEST cost: you cannot get a pose without running CameraHead.
    
    Returns: (poses_c2w, cam_dict, timing_dict)
    """
    from cambrianp.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from cambrianp.conversation import conv_templates
    from cambrianp.mm_utils import tokenizer_image_token
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    import copy, time as _time

    device = model.device
    num_frames = video_frames.shape[0]

    # ======================================================================
    # PREPARATION (not timed — this is "setup", not streaming inference)
    # ======================================================================
    
    # Build conversation template to get text tokens
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\nDescribe this scene.")
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # We need to figure out the token layout.
    # Do a single-frame dummy run through prepare_inputs_labels_for_multimodal
    # to discover: pre_img_len, tokens_per_frame, text_suffix tokens, etc.
    # This is setup cost, not counted.
    
    raw_model = model.module if hasattr(model, 'module') else model
    rec_head = raw_model.model.rec_head if hasattr(raw_model.model, 'rec_head') else raw_model.rec_head
    vision_tower = raw_model.get_vision_tower()
    
    # Get text embed tokens (system prompt + post-image text)
    embed_tokens = raw_model.get_model().embed_tokens
    
    # Find where IMAGE_TOKEN_INDEX is in input_ids
    img_token_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    assert len(img_token_idx) == 1, f"Expected 1 image token placeholder, got {len(img_token_idx)}"
    img_pos = img_token_idx[0].item()
    
    # Text before image (system prompt)
    pre_img_ids = input_ids[:, :img_pos]           # tokens before <image>
    post_img_ids = input_ids[:, img_pos+1:]        # tokens after <image> (question + assistant prompt)
    
    pre_img_embeds = embed_tokens(pre_img_ids)     # [1, pre_len, D]
    post_img_embeds = embed_tokens(post_img_ids)   # [1, post_len, D]
    pre_img_len = pre_img_embeds.shape[1]
    post_img_len = post_img_embeds.shape[1]
    
    # Camera tokens from rec_head
    camera_tokens = rec_head.camera_tokens  # [K, D], K=2 typically
    camera_tokens_place = rec_head.camera_tokens_place
    token_num = rec_head.camera_tokens_num  # 2 or 3
    
    # image_newline token
    image_newline = raw_model.get_model().image_newline  # [D]
    
    # Spatial pooling config (same as prepare_inputs_labels_for_multimodal)
    mm_img_tok_num = getattr(raw_model.config, "mm_img_tok_num", 196)
    mm_spatial_pool_stride = getattr(raw_model.config, "mm_spatial_pool_stride", 2)
    mm_spatial_pool_mode = getattr(raw_model.config, "mm_spatial_pool_mode", "bilinear")
    
    # Figure out per-frame visual token count AFTER pooling
    # SigLIP: 384/14=27 → 27×27=729 patches, then pool to sqrt(mm_img_tok_num)×sqrt(mm_img_tok_num)
    pool_side = int(math.sqrt(mm_img_tok_num))  # typically 14 for tok_num=196
    patch_size = rec_head.patch_size  # should match pool_side
    vis_tokens_per_frame = pool_side * (pool_side + 1)  # with newlines: 14*15 = 210
    cam_tokens_per_frame = 1  # 1 camera token per frame
    tokens_per_frame = vis_tokens_per_frame + cam_tokens_per_frame
    
    if debug:
        print(f"    [cambrian-p] mm_img_tok_num={mm_img_tok_num}, pool_side={pool_side}, "
              f"patch_size={patch_size}")
        print(f"    [cambrian-p] vis_tok/frame={vis_tokens_per_frame}, "
              f"cam_tok/frame={cam_tokens_per_frame}, total_tok/frame={tokens_per_frame}")
        print(f"    [cambrian-p] pre_img_len={pre_img_len}, post_img_len={post_img_len}")
        print(f"    [cambrian-p] camera_tokens_place={camera_tokens_place}, token_num={token_num}")
        print(f"    [cambrian-p] spatial_pool: mode={mm_spatial_pool_mode}, stride={mm_spatial_pool_stride}")

    # Determine number of intermediate layers for rec_head
    num_intermediate_layers = rec_head.num_intermediate_layers
    
    # bp_rec dummy batch (only images field needed, used by depth head which we skip)
    bp_rec = {
        'images': torch.zeros(1, num_frames, 3, 192, 192, device=device, dtype=torch.bfloat16),
    }
    
    # Helper: spatial pooling (same as model.get_2dPool)
    def _spatial_pool_single_frame(feat, target_tok_num):
        """Pool [1, num_patches, D] → [1, target_tok_num, D]"""
        h = w = vision_tower.num_patches_per_side  # 27 for SigLIP@384
        D_feat = feat.shape[-1]
        feat_2d = feat.view(1, h, w, D_feat).permute(0, 3, 1, 2)  # [1, D, h, w]
        side = int(math.sqrt(target_tok_num))
        if mm_spatial_pool_mode == "bilinear":
            feat_2d = nn.functional.interpolate(feat_2d, size=[side, side], mode="bilinear")
        elif mm_spatial_pool_mode == "average":
            feat_2d = nn.functional.avg_pool2d(feat_2d, mm_spatial_pool_stride)
        elif mm_spatial_pool_mode == "max":
            feat_2d = nn.functional.max_pool2d(feat_2d, mm_spatial_pool_stride)
        feat_2d = feat_2d.permute(0, 2, 3, 1).reshape(1, -1, D_feat)  # [1, side*side, D]
        return feat_2d
    
    # Helper: add newline per row (same as model.add_token_per_frame for 1 frame)
    def _add_newlines_single_frame(feat, side):
        """[1, side*side, D] → [side*(side+1), D] with newline after each row"""
        D_feat = feat.shape[-1]
        feat = feat.view(side, side, D_feat)  # [side, side, D]
        nl = image_newline.unsqueeze(0).unsqueeze(0).expand(side, 1, D_feat)
        feat_with_nl = torch.cat([feat, nl], dim=1)  # [side, side+1, D]
        return feat_with_nl.reshape(-1, D_feat)  # [side*(side+1), D]

    # ======================================================================
    # STREAMING LOOP
    # ======================================================================
    per_frame = []
    detailed_steps = []
    past_key_values = None
    seq_offset = 0
    
    # LLM KV cache control
    # Set CAMBRIAN_P_LLM_NO_CACHE=1 to disable (re-runs LLM on all frames 0..i each step, O(i²))
    use_llm_cache = not bool(int(os.environ.get("CAMBRIAN_P_LLM_NO_CACHE", "0")))
    accumulated_embeds = []  # for no-cache mode: store all frame embeddings

    # CameraHead KV cache control
    # Set CAMBRIAN_P_CAMHEAD_NO_CACHE=1 to disable (re-runs CameraHead on all frames, O(i²))
    use_camhead_cache = not bool(int(os.environ.get("CAMBRIAN_P_CAMHEAD_NO_CACHE", "0")))
    cam_kv = None
    
    print(f"    [cambrian-p] LLM backbone: KV cache {'ON (O(1) per frame)' if use_llm_cache else 'OFF (O(i²), re-runs on 0..i frames)'}")
    print(f"    [cambrian-p] CameraHead:   KV cache {'ON (O(1) per frame)' if use_camhead_cache else 'OFF (O(i²), re-runs on 0..i frames)'}")
    
    # Accumulate hidden states per layer for rec_head
    # We store per-frame hidden state chunks so we can reconstruct full hidden states
    # For each intermediate layer, we keep the tokens corresponding to image+cam region
    accumulated_hidden_per_layer = {}  # layer_idx -> list of [1, tokens_per_frame, D] tensors
    
    # Track which LLM layers to extract
    # (we discover this after first forward, since we need to know total depth)
    intermediate_layer_indices = None
    
    all_poses_c2w = []
    all_focals = []
    all_pps = []

    with torch.no_grad():
        # === Optional validation: verify our manual embedding matches batch version ===
        if debug:
            print(f"    [cambrian-p] Validating manual embedding vs batch...")
            try:
                prep_result = raw_model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None,
                    [video_frames], ["video"],
                    image_sizes=None,
                )
                # prep_result is a tuple of varying length; inputs_embeds is at index 4, info_dict at index 7
                batch_embeds = prep_result[4]
                batch_info = prep_result[6]
                batch_tpf = batch_info[0]["image_token_length"] // num_frames
                print(f"    [cambrian-p] Batch: total_seq={batch_embeds.shape[1]}, "
                      f"img_tok_len={batch_info[0]['image_token_length']}, "
                      f"tok/frame={batch_tpf}, pre_img={batch_info[0]['pre_image_token_length']}")
                if batch_tpf != tokens_per_frame:
                    print(f"    [cambrian-p] WARNING: token count mismatch! "
                          f"manual={tokens_per_frame} vs batch={batch_tpf}")
                    print(f"    [cambrian-p] Adjusting tokens_per_frame to {batch_tpf}")
                    tokens_per_frame = batch_tpf
                    vis_tokens_per_frame = tokens_per_frame - cam_tokens_per_frame
                del batch_embeds, batch_info, prep_result  # free memory
            except Exception as e:
                print(f"    [cambrian-p] Validation failed (non-fatal): {e}")
        
        for i in range(num_frames):
            # ==============================================================
            # STAGE 1: ENCODE — vision_tower + mm_projector + pool + newlines + cam token
            # ==============================================================
            single_frame = video_frames[i:i+1]  # [1, 3, H, W]
            
            torch.cuda.synchronize()
            t_enc_start = _time.perf_counter()
            
            # 1a-b. encode_images: vision_tower + mm_projector
            enc_result = raw_model.encode_images(single_frame)
            # encode_images may return (image_features, llava_features) or just image_features
            if isinstance(enc_result, tuple):
                image_features = enc_result[0]  # [1, 729, D]
            else:
                image_features = enc_result  # [1, 729, D]
            
            # 1c. Spatial pooling: [1, 729, D] → [1, mm_img_tok_num, D]
            image_features = _spatial_pool_single_frame(image_features, mm_img_tok_num)
            
            # 1d. Add newline tokens per row: [pool_side*(pool_side+1), D]
            D = image_features.shape[-1]
            frame_feat_flat = _add_newlines_single_frame(image_features, pool_side)  # [210, D]
            
            # 1e. Camera token injection
            if token_num == 2:
                cam_tok = camera_tokens[0] if i == 0 else camera_tokens[1]
            elif token_num == 3:
                if i == 0:
                    cam_tok = camera_tokens[0]
                elif i == num_frames - 1:
                    cam_tok = camera_tokens[2]
                else:
                    cam_tok = camera_tokens[1]
            else:
                cam_tok = camera_tokens[0]
            
            cam_tok = cam_tok.unsqueeze(0)  # [1, D]

            frame_embeds = torch.cat([frame_feat_flat, cam_tok], dim=0)  # [210+1, D]
            
            frame_embeds = frame_embeds.unsqueeze(0)  # [1, 211, D]
            
            torch.cuda.synchronize()
            encode_ms = (_time.perf_counter() - t_enc_start) * 1000
            
            # ==============================================================
            # STAGE 2: LLM FORWARD with KV cache
            # ==============================================================
            # Build this frame's chunk of embeddings
            if i == 0:
                # First frame: [pre_img_embeds, frame_embeds]
                chunk_embeds = torch.cat([pre_img_embeds, frame_embeds], dim=1)
            elif i == num_frames - 1:
                # Last frame: [frame_embeds, post_img_embeds]
                chunk_embeds = torch.cat([frame_embeds, post_img_embeds], dim=1)
            else:
                # Middle frames: just frame_embeds
                chunk_embeds = frame_embeds
            
            chunk_len = chunk_embeds.shape[1]
            
            torch.cuda.synchronize()
            t_llm_start = _time.perf_counter()
            
            if use_llm_cache:
                # ---- LLM PATH A: KV cache ON — feed only current chunk, O(1) ----
                chunk_pos_ids = torch.arange(
                    seq_offset, seq_offset + chunk_len, device=device, dtype=torch.long
                ).unsqueeze(0)
                chunk_attn_mask = torch.ones(1, seq_offset + chunk_len, device=device, dtype=torch.long)
                
                outputs = raw_model.model(
                    input_ids=None,
                    attention_mask=chunk_attn_mask,
                    position_ids=chunk_pos_ids,
                    inputs_embeds=chunk_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                past_key_values = outputs.past_key_values
                seq_offset += chunk_len
            else:
                # ---- LLM PATH B: NO cache — re-run on all accumulated embeddings ----
                accumulated_embeds.append(chunk_embeds)
                all_embeds = torch.cat(accumulated_embeds, dim=1)  # [1, total_tokens_so_far, D]
                total_len = all_embeds.shape[1]
                
                all_pos_ids = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0)
                all_attn_mask = torch.ones(1, total_len, device=device, dtype=torch.long)
                
                outputs = raw_model.model(
                    input_ids=None,
                    attention_mask=all_attn_mask,
                    position_ids=all_pos_ids,
                    inputs_embeds=all_embeds,
                    past_key_values=None,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            torch.cuda.synchronize()
            llm_ms = (_time.perf_counter() - t_llm_start) * 1000
            
            # Discover intermediate layer indices on first forward
            if intermediate_layer_indices is None:
                total_layers = len(outputs.hidden_states) - 1  # exclude embedding layer
                intermediate_layer_indices = [
                    total_layers * (j + 1) // num_intermediate_layers
                    for j in range(num_intermediate_layers - 1)
                ]
                intermediate_layer_indices.append(-1)  # always include last layer
                if debug:
                    print(f"    [cambrian-p] LLM has {total_layers} layers, "
                          f"using intermediate layers: {intermediate_layer_indices}")
            
            # ==============================================================
            # Extract this frame's image+cam tokens from hidden states
            # ==============================================================
            # For cache mode: hidden_states are for the current chunk only
            # For no-cache mode: hidden_states are for ALL accumulated tokens;
            #   current frame's tokens are at the same absolute position
            
            if use_llm_cache:
                # Offsets relative to chunk
                if i == 0:
                    frame_tok_start = pre_img_len
                    frame_tok_end = pre_img_len + tokens_per_frame
                else:
                    frame_tok_start = 0
                    frame_tok_end = tokens_per_frame
            else:
                # No-cache: offsets are absolute within the full sequence
                # pre_img tokens + frame_0 tokens + frame_1 tokens + ... + frame_i tokens [+ post_img]
                frame_tok_start = pre_img_len + i * tokens_per_frame
                frame_tok_end = frame_tok_start + tokens_per_frame
            
            for li, layer_idx in enumerate(intermediate_layer_indices):
                hs = outputs.hidden_states[layer_idx]  # [1, chunk_len, D]
                frame_hidden = hs[:, frame_tok_start:frame_tok_end, :]  # [1, tokens_per_frame, D]
                
                if li not in accumulated_hidden_per_layer:
                    accumulated_hidden_per_layer[li] = []
                accumulated_hidden_per_layer[li].append(frame_hidden)
            
            # ==============================================================
            # STAGE 3: DECODE — projector + CameraHead + pose decode
            # ==============================================================
            torch.cuda.synchronize()
            t_dec_start = _time.perf_counter()
            
            n_seen = i + 1  # number of frames seen so far
            feat_D = accumulated_hidden_per_layer[0][-1].shape[-1]
            
            if use_camhead_cache:
                # ---- PATH A: KV cache ON — feed only current frame, O(1) ----
                rec_feats_list = []
                for li in range(len(intermediate_layer_indices)):
                    cur_hidden = accumulated_hidden_per_layer[li][-1]  # [1, tokens_per_frame, D]
                    cur_hidden = cur_hidden.view(1, 1, tokens_per_frame, feat_D)
                    
                    img_tokens_hs, cam_tokens_hs = torch.split(cur_hidden, [tokens_per_frame - 1, 1], dim=2)
                    
                    img_tokens_hs = img_tokens_hs.view(1, 1, patch_size, patch_size + 1, feat_D)
                    img_tokens_hs = img_tokens_hs[:, :, :, :patch_size, :]
                    img_tokens_hs = img_tokens_hs.reshape(1, 1, patch_size * patch_size, feat_D)
                    
                    combined = torch.cat([cam_tokens_hs, img_tokens_hs], dim=2)
                    projected = rec_head.projector_list[li](combined)
                    rec_feats_list.append(projected)
                
                with torch.amp.autocast('cuda', enabled=False):
                    pose_enc_list, cam_kv = rec_head.camera_head(
                        rec_feats_list,
                        past_key_values_camera=cam_kv,
                        use_cache=True,
                    )
                    pose_enc = pose_enc_list[-1]  # [1, 1, 8]
                
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (384, 384))
                    if extrinsic.dim() > 3: extrinsic = extrinsic[0]
                    if intrinsic.dim() > 3: intrinsic = intrinsic[0]
                
                R_i = extrinsic[0, :3, :3]
                t_i = extrinsic[0, :3, 3]
                
            else:
                # ---- PATH B: KV cache OFF — stack all frames 0..i, O(i²) ----
                rec_feats_list = []
                for li in range(len(intermediate_layer_indices)):
                    stacked = torch.cat(accumulated_hidden_per_layer[li], dim=1)  # [1, n_seen*tpf, D]
                    stacked = stacked.view(1, n_seen, tokens_per_frame, feat_D)
                    
                    img_tokens_hs, cam_tokens_hs = torch.split(stacked, [tokens_per_frame - 1, 1], dim=2)
                    
                    img_tokens_hs = img_tokens_hs.view(1, n_seen, patch_size, patch_size + 1, feat_D)
                    img_tokens_hs = img_tokens_hs[:, :, :, :patch_size, :]
                    img_tokens_hs = img_tokens_hs.reshape(1, n_seen, patch_size * patch_size, feat_D)
                    
                    combined = torch.cat([cam_tokens_hs, img_tokens_hs], dim=2)
                    projected = rec_head.projector_list[li](combined)
                    rec_feats_list.append(projected)
                
                with torch.amp.autocast('cuda', enabled=False):
                    pose_enc_list = rec_head.camera_head(rec_feats_list)
                    pose_enc = pose_enc_list[-1]  # [1, n_seen, 8]
                
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (384, 384))
                    if extrinsic.dim() > 3: extrinsic = extrinsic[0]
                    if intrinsic.dim() > 3: intrinsic = intrinsic[0]
                
                R_i = extrinsic[i, :3, :3]
                t_i = extrinsic[i, :3, 3]
            
            T_inv = torch.eye(4, device=device)
            T_inv[:3, :3] = R_i.T
            T_inv[:3, 3] = -R_i.T @ t_i
            
            torch.cuda.synchronize()
            decode_ms = (_time.perf_counter() - t_dec_start) * 1000
            
            all_poses_c2w.append(T_inv)
            focal_idx = 0 if use_camhead_cache else i
            all_focals.append(intrinsic[focal_idx, 0, 0].cpu().item())
            all_pps.append(intrinsic[focal_idx, :2, 2].cpu().numpy())
            
            # Record timing
            total_ms = encode_ms + llm_ms + decode_ms
            per_frame.append({
                "frame_id": i,
                "encode_ms": encode_ms,
                "forward_ms": llm_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "n_frames_in_camhead": n_seen,
            })
            
            if i < 5 or i % 20 == 0 or i == num_frames - 1:
                print(f"    [cambrian-p] frame {i}: enc={encode_ms:.1f}ms "
                      f"llm={llm_ms:.1f}ms dec={decode_ms:.1f}ms "
                      f"total={total_ms:.1f}ms (camhead on {n_seen} frames)")

    # ======================================================================
    # Build output
    # ======================================================================
    poses_c2w = torch.stack(all_poses_c2w)
    focal_np = np.array(all_focals)
    pp_np = np.stack(all_pps)
    cam_dict = {"focal": focal_np, "pp": pp_np}

    # Build info_dict for compatibility (describes the full sequence)
    info_dict = {
        "pre_image_token_length": pre_img_len,
        "image_token_length": num_frames * tokens_per_frame,
        "num_frames": num_frames,
        "camera_tokens_place": camera_tokens_place,
    }

    # Detailed step summary
    enc_times = [pf["encode_ms"] for pf in per_frame]
    llm_times = [pf["forward_ms"] for pf in per_frame]
    dec_times = [pf["decode_ms"] for pf in per_frame]
    
    detailed_steps = [
        {
            "step": "1_encode_vision",
            "description": f"SigLIP vision tower (1 frame) + mm_projector + camera token injection + newline tokens",
            "time_ms": sum(enc_times),
            "per_frame": True,
            "per_frame_ms": float(np.mean(enc_times)),
            "per_frame_median_ms": float(np.median(enc_times)),
            "per_frame_min_ms": float(np.min(enc_times)),
            "per_frame_max_ms": float(np.max(enc_times)),
            "num_frames": num_frames,
        },
        {
            "step": "2_llm_forward",
            "description": (
                f"Qwen2 LLM forward with KV cache (per-frame chunk). O(1) per frame after warmup."
                if use_llm_cache else
                f"Qwen2 LLM forward WITHOUT cache (re-runs on all 0..i frames). O(i²) cost."
            ),
            "time_ms": sum(llm_times),
            "per_frame": True,
            "per_frame_ms": float(np.mean(llm_times)),
            "per_frame_median_ms": float(np.median(llm_times)),
            "per_frame_min_ms": float(np.min(llm_times)),
            "per_frame_max_ms": float(np.max(llm_times)),
            "num_frames": num_frames,
        },
        {
            "step": "3_decode_camhead",
            "description": (
                f"Projector MLP + CameraHead ({getattr(rec_head.camera_head, 'trunk_depth', 4)}-layer "
                + (f"causal transformer with KV cache, O(1) per frame)"
                   if use_camhead_cache else
                   f"bidirectional transformer, NO cache, re-runs on 0..i frames, O(i²))")
                + " + pose_encoding_to_extri_intri."
            ),
            "time_ms": sum(dec_times),
            "per_frame": True,
            "per_frame_ms": float(np.mean(dec_times)),
            "per_frame_median_ms": float(np.median(dec_times)),
            "per_frame_min_ms": float(np.min(dec_times)),
            "per_frame_max_ms": float(np.max(dec_times)),
            "num_frames": num_frames,
        },
    ]
    
    # Print summary
    total_pipeline = sum(pf["total_ms"] for pf in per_frame)
    print(f"\n    [cambrian-p] === DETAILED TIMING SUMMARY ===")
    for step in detailed_steps:
        pct = step["time_ms"] / total_pipeline * 100 if total_pipeline > 0 else 0
        print(f"    {step['step']:30s}: {step['time_ms']:8.1f}ms ({pct:5.1f}%) "
              f"  avg={step['per_frame_ms']:.1f}ms/frame")
        print(f"      └─ {step['description']}")
    print(f"    {'TOTAL':30s}: {total_pipeline:8.1f}ms")
    print(f"    {'Per-frame avg':30s}: {total_pipeline/num_frames:8.1f}ms/frame")
    print()

    timing_dict = _build_timing_dict(per_frame, num_frames, {
        "detailed_steps": detailed_steps,
    })
    
    return poses_c2w, cam_dict, timing_dict


def prepare_input_llava(file_paths, image_processor, device, target_size=384):
    """Prepare input for LLaVA-VGGT model."""
    images = []
    
    for path in file_paths:
        if not os.path.exists(path):
            continue
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img.shape[:2]
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            
            start_h = (new_h - target_size) // 2
            start_w = (new_w - target_size) // 2
            img = img[start_h:start_h+target_size, start_w:start_w+target_size]
            
            if img.shape[:2] != (target_size, target_size):
                img = cv2.resize(img, (target_size, target_size))
            
            images.append(img)
        except Exception:
            continue
    
    if not images:
        return None
    
    images = np.stack(images)
    frames = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return frames.to(dtype=torch.bfloat16, device=device)


def sample_frames(filelist, num_frames, strategy="uniform"):
    """Sample frames using uniform or continuous strategy.
    """
    total = len(filelist)
    
    if strategy == "monst3r":
        # MonST3R protocol: use ALL frames, no subsampling
        return filelist, list(range(total))
    
    if num_frames is None or num_frames >= total:
        return filelist, list(range(total))
    
    if strategy == "uniform":
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    else:  # continuous
        indices = list(range(num_frames))
    
    return [filelist[i] for i in indices], indices


# =============================================================================
# Visualization
# =============================================================================

def umeyama_align(X, Y, with_scale=True):
    """Align X to Y using Umeyama similarity transformation."""
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    
    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - muX, Y - muY
    
    Sigma = (Yc.T @ Xc) / X.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)
    
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * X.shape[0] if with_scale else 1.0
    t = muY - s * (R @ muX)
    
    return (s * (R @ X.T)).T + t, (s, R, t)


def extract_xyz(tum_array):
    """Extract XYZ from TUM format array."""
    arr = np.asarray(tum_array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    if arr.shape[1] >= 8:
        return arr[:, 1:4]
    return arr[:, :3]


def visualize_trajectory(gt_tum, pred_tum, output_prefix, title="Trajectory",
                        ate=None, rpe_trans=None, rpe_rot=None, debug=False):
    """Create 4-panel trajectory visualization."""
    try:
        gt_xyz = extract_xyz(gt_tum)
        pred_xyz_raw = extract_xyz(pred_tum)
        
        if len(gt_xyz) < 2 or len(pred_xyz_raw) < 2:
            return
        
        pred_xyz, (scale, _, _) = umeyama_align(pred_xyz_raw, gt_xyz)
        
        if debug:
            print(f"[viz] Umeyama scale: {scale:.4f}")
        
        np.save(f"{output_prefix}_gt_xyz.npy", gt_xyz)
        np.save(f"{output_prefix}_pred_xyz_aligned.npy", pred_xyz)
        
        # Save full TUM arrays (with quaternions) for rotation metrics
        np.save(f"{output_prefix}_gt_tum.npy", np.asarray(gt_tum, dtype=np.float64))
        np.save(f"{output_prefix}_pred_tum.npy", np.asarray(pred_tum, dtype=np.float64))
        
        if ate is None:
            ate = np.linalg.norm(gt_xyz - pred_xyz, axis=1).mean()
        if rpe_trans is None and len(gt_xyz) > 1:
            rpe_trans = np.linalg.norm(np.diff(gt_xyz, axis=0) - np.diff(pred_xyz, axis=0), axis=1).mean()
        
        # Compute rotation errors
        rot_errors = []
        if gt_tum.shape[1] >= 7 and pred_tum.shape[1] >= 7:
            for i in range(min(len(gt_tum), len(pred_tum))):
                try:
                    gt_q = gt_tum[i, 3:7] if gt_tum.shape[1] == 7 else gt_tum[i, 4:8]
                    pred_q = pred_tum[i, 3:7] if pred_tum.shape[1] == 7 else pred_tum[i, 4:8]
                    
                    R_gt = Rotation.from_quat([gt_q[1], gt_q[2], gt_q[3], gt_q[0]]).as_matrix()
                    R_pred = Rotation.from_quat([pred_q[1], pred_q[2], pred_q[3], pred_q[0]]).as_matrix()
                    
                    angle = np.arccos(np.clip((np.trace(R_gt.T @ R_pred) - 1) / 2, -1, 1))
                    rot_errors.append(np.degrees(angle))
                except Exception:
                    pass
        
        mean_rot = np.mean(rot_errors) if rot_errors else (rpe_rot or float('nan'))
        max_rot = np.max(rot_errors) if rot_errors else float('nan')
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # 3D view
        ax3d = fig.add_subplot(221, projection='3d')
        ax3d.plot(*gt_xyz.T, 'g-', lw=2, label='GT', marker='o', ms=4, markevery=max(1, len(gt_xyz)//20))
        ax3d.plot(*pred_xyz.T, 'r-', lw=2, label='Pred', marker='s', ms=4, markevery=max(1, len(pred_xyz)//20))
        ax3d.set_title('3D View', fontsize=12, fontweight='bold')
        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
        ax3d.legend()
        
        all_xyz = np.vstack([gt_xyz, pred_xyz])
        max_range = np.ptp(all_xyz, axis=0).max() / 2
        mid = all_xyz.mean(axis=0)
        ax3d.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax3d.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax3d.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        # XY view
        ax_xy = fig.add_subplot(222)
        ax_xy.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'g-', lw=2, label='GT')
        ax_xy.plot(pred_xyz[:, 0], pred_xyz[:, 1], 'r-', lw=2, label='Pred')
        ax_xy.scatter(*gt_xyz[0, :2], c='green', marker='^', s=100, zorder=5, label='Start')
        ax_xy.scatter(*gt_xyz[-1, :2], c='purple', marker='v', s=100, zorder=5, label='End')
        ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')
        ax_xy.set_title('Top-Down (XY)', fontsize=12, fontweight='bold')
        ax_xy.axis('equal'); ax_xy.grid(True, alpha=0.3); ax_xy.legend()
        
        # XZ view
        ax_xz = fig.add_subplot(223)
        ax_xz.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', lw=2, label='GT')
        ax_xz.plot(pred_xyz[:, 0], pred_xyz[:, 2], 'r-', lw=2, label='Pred')
        ax_xz.scatter(gt_xyz[0, 0], gt_xyz[0, 2], c='green', marker='^', s=100, zorder=5)
        ax_xz.scatter(gt_xyz[-1, 0], gt_xyz[-1, 2], c='purple', marker='v', s=100, zorder=5)
        ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z')
        ax_xz.set_title('Side View (XZ)', fontsize=12, fontweight='bold')
        ax_xz.axis('equal'); ax_xz.grid(True, alpha=0.3); ax_xz.legend()
        
        # Metrics panel
        ax_txt = fig.add_subplot(224)
        ax_txt.axis('off')
        
        summary = (
            f"Frames: {len(gt_xyz)}\n"
            f"Umeyama scale: {scale:.4f}\n\n"
            f"ATE: {ate:.4f} m\n"
            f"RPE trans: {rpe_trans:.4f} m\n\n"
            f"Mean rot err: {mean_rot:.2f}°\n"
            f"Max rot err: {max_rot:.2f}°\n\n"
            f"GT extent:\n"
            f"  X: {np.ptp(gt_xyz[:, 0]):.3f} m\n"
            f"  Y: {np.ptp(gt_xyz[:, 1]):.3f} m\n"
            f"  Z: {np.ptp(gt_xyz[:, 2]):.3f} m\n"
            f"  Path: {np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)):.3f} m"
        )
        
        ax_txt.text(0.05, 0.95, "Metrics", fontsize=14, fontweight="bold", va="top", transform=ax_txt.transAxes)
        ax_txt.text(0.05, 0.85, summary, fontsize=11, va="top", family="monospace", transform=ax_txt.transAxes)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(f"{output_prefix}_trajectory.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[viz] Saved: {output_prefix}_trajectory.png")
        
    except Exception as e:
        print(f"[viz] Failed: {e}")


# =============================================================================
# Main Evaluation
# =============================================================================

def run_inference(args, model, file_paths, device, tokenizer=None, image_processor=None):
    """Run inference. Returns (poses, cam_dict, wall_time_sec, model_fwd_time_sec, timing_record).

    wall_time_sec measures data preprocessing + model forward + post-processing.
    model_fwd_time_sec measures only the model forward pass (excluding pre/post-processing);
    in batch mode it equals wall_time_sec.

    timing_record: per-scene detailed timing in streaming mode, else None.
    Uses torch.cuda.synchronize() for accurate GPU timing.
    """
    import time

    model_fwd_time = 0.0
    timing_record = None
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    if args.model_type == "llava_vggt":
        frames = prepare_input_llava(file_paths, image_processor, device)
        if frames is None:
            return None, None, 0.0, 0.0, None
        
        # Process in chunks if needed
        max_chunk = 128
        T = frames.shape[0]
        all_poses, all_focal, all_pp = [], [], []
        total_llm_time, total_rec_time = 0.0, 0.0
        
        for start in range(0, T, max_chunk):
            end = min(start + max_chunk, T)
            chunk = frames[start:end]
            
            if args.debug:
                print(f"  [inference] Chunk [{start}:{end}], shape: {chunk.shape}")
            
            if args.llava_mode == "streaming":
                chunk_poses, chunk_cam, chunk_timing = extract_pose_llava_streaming(
                    model, chunk, tokenizer, debug=args.debug
                )
                total_llm_time += chunk_timing["llm_time"]
                total_rec_time += chunk_timing["rec_time"]
                # Merge chunk timing into record
                if timing_record is None:
                    timing_record = dict(chunk_timing)  # copy all keys including per_frame_times
                else:
                    # Accumulate numeric fields
                    for k in ["num_frames", "prep_time", "llm_time", "rec_time",
                              "decode_time", "total_model_time", "total_time",
                              "total_llm_time", "total_rec_time"]:
                        if k in chunk_timing and k in timing_record:
                            timing_record[k] += chunk_timing[k]
                    # Extend per_frame_times list
                    if "per_frame_times" in chunk_timing and "per_frame_times" in timing_record:
                        timing_record["per_frame_times"].extend(chunk_timing["per_frame_times"])
            else:
                chunk_poses, chunk_cam, chunk_timing = extract_pose_llava(
                    model, chunk, tokenizer, debug=args.debug
                )
                # Accumulate batch timing
                if timing_record is None:
                    timing_record = dict(chunk_timing)  # first chunk: copy everything
                else:
                    if "per_frame_times" in chunk_timing and "per_frame_times" in timing_record:
                        timing_record["per_frame_times"].extend(chunk_timing["per_frame_times"])
                    if "per_frame" in chunk_timing and "per_frame" in timing_record:
                        timing_record["per_frame"].extend(chunk_timing["per_frame"])
                    for k in ["total_time", "total_model_time", "llm_time", "rec_time", "prep_time",
                              "total_llm_time", "total_rec_time"]:
                        if k in chunk_timing:
                            timing_record[k] = timing_record.get(k, 0) + chunk_timing[k]
            
            if torch.is_tensor(chunk_poses):
                chunk_poses = chunk_poses.cpu().numpy()
            
            all_poses.append(chunk_poses)
            all_focal.append(chunk_cam["focal"])
            all_pp.append(chunk_cam["pp"])
        
        poses = torch.from_numpy(np.concatenate(all_poses))
        cam_dict = {
            "focal": np.concatenate(all_focal),
            "pp": np.concatenate(all_pp),
        }
        if args.llava_mode == "streaming":
            model_fwd_time = total_llm_time + total_rec_time
        
    if torch.is_tensor(poses):
        poses = poses.cpu().numpy()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    wall_time = t_end - t_start
    
    # If model didn't set model_fwd_time, use wall_time as fallback
    if model_fwd_time == 0.0:
        model_fwd_time = wall_time
    
    return poses, cam_dict, wall_time, model_fwd_time, timing_record


def evaluate(args, model, tokenizer=None, image_processor=None, save_dir=None):
    """Main evaluation function."""
    metadata = deepcopy(dataset_metadata[args.eval_dataset])
    img_path = metadata["img_path"]
    anno_path = metadata.get("anno_path")

    if args.sampling_strategy == "monst3r":
        metadata = apply_monst3r_overrides(args, metadata)
        # Re-read in case overrides changed the funcs
        # (img_path / anno_path stay the same, only funcs change)
    
    # Get sequence list
    if args.full_seq or metadata.get("full_seq", False):
        seq_list = sorted([
            s for s in os.listdir(img_path)
            if os.path.isdir(os.path.join(img_path, s))
        ])
    else:
        seq_list = sorted(metadata.get("seq_list", []))
    
    # Random sampling
    if args.num_scenes and args.num_scenes < len(seq_list):
        import random
        random.seed(args.seed)
        seq_list = sorted(random.sample(seq_list, args.num_scenes))
    
    save_dir = save_dir or args.output_dir
    
    # Save config
    with open(os.path.join(save_dir, "config.txt"), 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Path: {args.model_path}\n")
        f.write(f"Dataset: {args.eval_dataset}\n")
        f.write(f"Scenes: {len(seq_list)}\n")
        f.write(f"Sampling: {args.sampling_strategy}\n")
        f.write(f"Stride: {args.pose_eval_stride}\n")
        f.write(f"Frames: {args.num_frames_per_scene}\n")
        f.write(f"Size: {args.size}\n")
        f.write(f"Random init: {args.random_init}\n")
        f.write(f"LLaVA mode: {args.llava_mode}\n")
    
    distributed_state = PartialState()
    device = distributed_state.device
    if model is not None:
        model.to(device)
    
    all_results = []
    
    with distributed_state.split_between_processes(seq_list) as seqs:
        local_results = []
        total_forward_time = 0.0
        total_model_fwd_time = 0.0
        total_frames_count = 0
        scene_timing_records = []  # Per-scene timing for streaming mode
        
        for idx, seq in enumerate(tqdm(seqs, desc=f"Process {distributed_state.process_index}")):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)
                filelist = sorted([
                    os.path.join(dir_path, f) for f in os.listdir(dir_path)
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ])
                
                if args.debug:
                    print(f"\n{'='*60}\n{seq}\n{'='*60}")
                    print(f"  Total frames in dir: {len(filelist)}")
                
                # Load GT
                gt_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format")
                gt_full = None
                
                if args.eval_dataset == "sintel" and gt_file is not None and os.path.exists(gt_file):
                    # Sintel special case: traj_format is None in metadata, but
                    # GT is loaded via load_sintel_traj from .cam files directory.
                    # This matches Point3R launch.py lines 168-171 exactly.
                    gt_full = load_traj(
                        gt_traj_file=gt_file,
                        traj_format="sintel",
                        stride=args.pose_eval_stride,
                    )
                elif traj_format and gt_file is not None and os.path.exists(gt_file):
                    gt_full = load_traj(gt_file, traj_format, stride=1)
                
                # frame selection
                if args.sampling_strategy == "monst3r":
                    # MonST3R protocol: apply stride (usually 1), then use ALL frames.
                    # The data is already pre-extracted (color_90), so stride=1 means all.
                    sampled = filelist[::args.pose_eval_stride]
                    final_idx = list(range(0, len(filelist), args.pose_eval_stride))
                else:
                    # Legacy: stride first, then subsample
                    strided = filelist[::args.pose_eval_stride]
                    strided_idx = list(range(0, len(filelist), args.pose_eval_stride))
                    
                    sampled, local_idx = sample_frames(strided, args.num_frames_per_scene, args.sampling_strategy)
                    final_idx = [strided_idx[i] for i in local_idx]
                
                if args.debug:
                    print(f"  After sampling ({args.sampling_strategy}): {len(sampled)} frames")
                
                # Verify files exist
                sampled = [f for f in sampled if os.path.exists(f)]
                if len(sampled) < 2:
                    continue
                
                # Run inference
                pr_poses, cam_dict, fwd_time, mdl_fwd_time, timing_record = run_inference(
                    args, model, sampled, device, tokenizer, image_processor
                )
                
                if pr_poses is None:
                    continue
                
                # Accumulate forward time
                total_forward_time += fwd_time
                total_model_fwd_time += mdl_fwd_time
                total_frames_count += pr_poses.shape[0]
                
                # Collect per-scene timing for streaming mode
                if timing_record is not None:
                    timing_record["scene"] = seq
                    scene_timing_records.append(timing_record)
                    # Save timing incrementally (CSV append + update plots every 5 scenes)
                    _save_timing_incremental(timing_record, scene_timing_records, save_dir,
                                            update_plots=(len(scene_timing_records) % 5 == 0))
                
                num_pred = pr_poses.shape[0]
                if args.debug:
                    print(f"  Predictions: {num_pred}, wall_time: {fwd_time:.3f}s, model_fwd: {mdl_fwd_time:.3f}s")
                
                # Convert to TUM
                pred_traj = get_tum_poses(pr_poses)
                
                # Save predictions
                os.makedirs(os.path.join(save_dir, seq), exist_ok=True)
                
                # Match GT
                ate, rpe_trans, rpe_rot = 0.0, 0.0, 0.0
                gt_traj = None
                
                if gt_full is not None:
                    gt_poses, gt_ts = gt_full
                    
                    if args.sampling_strategy == "monst3r":
                        if args.eval_dataset == "sintel":
                            # Sintel: load_traj already applied stride to GT.
                            # GT count should match image count directly.
                            num_use = min(num_pred, len(gt_poses))
                        else:
                            # ScanNet/TUM: GT file (pose_90.txt) is pre-extracted
                            # to match images 1:1. Apply same stride for consistency.
                            gt_strided = gt_poses[::args.pose_eval_stride]
                            gt_poses = gt_strided
                            num_use = min(num_pred, len(gt_poses))
                        
                        gt_sampled = gt_poses[:num_use]
                        pred_sampled = pred_traj[0][:num_use]
                        
                        pred_traj = (pred_sampled, np.arange(num_use, dtype=float))
                        gt_traj = (gt_sampled, np.arange(num_use, dtype=float))
                    else:
                        # Legacy: index into full GT using final_idx
                        valid_idx = [i for i in final_idx if i < len(gt_poses)]
                        num_use = min(num_pred, len(valid_idx))
                        
                        gt_sampled = gt_poses[valid_idx[:num_use]]
                        pred_sampled = pred_traj[0][:num_use]
                        
                        pred_traj = (pred_sampled, np.arange(num_use, dtype=float))
                        gt_traj = (gt_sampled, np.array(valid_idx[:num_use], dtype=float))
                    
                    if args.debug:
                        print(f"  Matched: Pred={num_use}, GT={len(gt_sampled)}")
                    
                    correct_scale = (args.alignment == "sim3")
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj, gt_traj, seq=seq,
                        filename=os.path.join(save_dir, f"{seq}_metrics.txt"),
                        correct_scale=correct_scale,
                    )
                
                if ate > 0 or rpe_trans > 0:
                    local_results.append({
                        'scene': seq, 'ate': ate,
                        'rpe_trans': rpe_trans, 'rpe_rot': rpe_rot
                    })
                
                # Visualization
                if args.viz and idx % args.viz_every == 0 and gt_traj is not None:
                    gt_arr = gt_traj[0] if isinstance(gt_traj, tuple) else gt_traj
                    pred_arr = pred_traj[0] if isinstance(pred_traj, tuple) else pred_traj
                    
                    if len(gt_arr) >= 2 and len(pred_arr) >= 2:
                        visualize_trajectory(
                            gt_arr, pred_arr,
                            os.path.join(save_dir, f"{seq}_traj"),
                            title=f"{args.eval_dataset} - {seq}",
                            ate=ate, rpe_trans=rpe_trans, rpe_rot=rpe_rot,
                            debug=args.debug
                        )
                
            except Exception as e:
                print(f"Error in {seq}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        
        # Gather results
        if distributed_state.num_processes > 1:
            gathered = distributed_state.gather(local_results)
            if distributed_state.is_main_process:
                all_results = [r for sublist in gathered for r in sublist]
        else:
            all_results = local_results
    
    distributed_state.wait_for_everyone()
    
    # Write summary
    if distributed_state.is_main_process and all_results:
        avg_ate = np.mean([r['ate'] for r in all_results])
        avg_rpe_t = np.mean([r['rpe_trans'] for r in all_results])
        avg_rpe_r = np.mean([r['rpe_rot'] for r in all_results])
        
        # Summary file
        with open(os.path.join(save_dir, "summary.txt"), 'w') as f:
            f.write(f"{'='*60}\nEVALUATION SUMMARY\n{'='*60}\n\n")
            f.write(f"Model: {args.model_type}\n")
            f.write(f"Dataset: {args.eval_dataset}\n")
            f.write(f"Sampling: {args.sampling_strategy}\n")
            f.write(f"Stride: {args.pose_eval_stride}\n")
            f.write(f"Frames per scene: {args.num_frames_per_scene or 'ALL'}\n")
            f.write(f"Size: {args.size}\n")
            f.write(f"Scenes: {len(all_results)}\n")
            f.write(f"Random init: {args.random_init}\n")
            f.write(f"LLaVA mode: {args.llava_mode}\n")
            f.write(f"\nAVERAGE METRICS:\n")
            f.write(f"  ATE: {avg_ate:.5f} m\n")
            f.write(f"  RPE Trans: {avg_rpe_t:.5f} m\n")
            f.write(f"  RPE Rot: {avg_rpe_r:.2f}°\n\n")
            f.write(f"{'Scene':<20} {'ATE':<12} {'RPE_T':<12} {'RPE_R':<12}\n")
            f.write('-'*60 + '\n')
            for r in sorted(all_results, key=lambda x: x['scene']):
                f.write(f"{r['scene']:<20} {r['ate']:<12.5f} {r['rpe_trans']:<12.5f} {r['rpe_rot']:<12.2f}\n")
        
        # CSV
        with open(os.path.join(save_dir, "results.csv"), 'w') as f:
            f.write("scene,ate,rpe_trans,rpe_rot\n")
            for r in sorted(all_results, key=lambda x: x['scene']):
                f.write(f"{r['scene']},{r['ate']:.6f},{r['rpe_trans']:.6f},{r['rpe_rot']:.6f}\n")
            f.write(f"AVERAGE,{avg_ate:.6f},{avg_rpe_t:.6f},{avg_rpe_r:.6f}\n")
        
        # JSON
        # Compute timing stats for metrics.json
        timing_stats = {
            "wall_time_s": round(total_forward_time, 3),
            "model_fwd_time_s": round(total_model_fwd_time, 3),
            "total_frames": total_frames_count,
            "streaming": bool(scene_timing_records),
        }
        if total_frames_count > 0 and total_model_fwd_time > 0:
            timing_stats["avg_per_frame_batch_ms"] = round(total_model_fwd_time / total_frames_count * 1000, 2)
        if scene_timing_records:
            all_frame_ms_for_json = []
            for rec in scene_timing_records:
                for ft in rec.get("per_frame_times", []):
                    all_frame_ms_for_json.append(ft["total_time"] * 1000)
            if all_frame_ms_for_json:
                timing_stats["avg_per_frame_streaming_ms"] = round(float(np.mean(all_frame_ms_for_json)), 2)
                timing_stats["median_per_frame_streaming_ms"] = round(float(np.median(all_frame_ms_for_json)), 2)
                timing_stats["streaming_frames"] = len(all_frame_ms_for_json)
        
        with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
            json.dump({
                "model_type": args.model_type,
                "sampling_strategy": args.sampling_strategy,
                "pose_eval_stride": args.pose_eval_stride,
                "num_frames_per_scene": args.num_frames_per_scene,
                "size": args.size,
                "random_init": args.random_init,
                "num_scenes": len(all_results),
                "llava_mode": args.llava_mode,
                "alignment": args.alignment,
                "ATE_m": round(avg_ate, 6),
                "RPE_T_m": round(avg_rpe_t, 6),
                "RPE_R_deg": round(avg_rpe_r, 6),
                "timing": timing_stats,
            }, f, indent=2)
        
        print(f"\n{'='*60}\nRESULTS\n{'='*60}")
        print(f"Model: {args.model_type}")
        print(f"Sampling: {args.sampling_strategy}")
        print(f"LLaVA mode: {args.llava_mode}")
        if args.random_init:
            print("*** RANDOM INITIALIZATION (BASELINE) ***")
        print(f"Scenes: {len(all_results)}")
        print(f"ATE: {avg_ate:.5f} m")
        print(f"RPE Trans: {avg_rpe_t:.5f} m")
        print(f"RPE Rot: {avg_rpe_r:.2f}°")
        print(f"Wall time: {total_forward_time:.2f}s ({total_forward_time/60:.1f}min)")
        print(f"Model fwd time: {total_model_fwd_time:.2f}s ({total_model_fwd_time/60:.1f}min)")
        
        # Compute per-frame stats if streaming timing available
        if scene_timing_records:
            all_frame_ms = []
            total_frames = 0
            for rec in scene_timing_records:
                for ft in rec.get("per_frame_times", []):
                    all_frame_ms.append(ft["total_time"] * 1000)
                total_frames += rec.get("num_frames", 0)
            if all_frame_ms:
                avg_ms = np.mean(all_frame_ms)
                med_ms = np.median(all_frame_ms)
                print(f"Avg per-frame time: {avg_ms:.1f}ms (median: {med_ms:.1f}ms, {len(all_frame_ms)} frames)")
        else:
            # Batch mode: compute avg per-frame from total model fwd time
            if total_frames_count > 0 and total_model_fwd_time > 0:
                avg_ms = total_model_fwd_time / total_frames_count * 1000
                print(f"Avg per-frame time (batch): {avg_ms:.1f}ms ({total_frames_count} frames)")
        
        print(f"Saved to: {save_dir}")
        
        # ===== Per-scene timing analysis (streaming mode) =====
        if scene_timing_records:
            _save_timing_analysis(scene_timing_records, save_dir)
        
        return avg_ate, avg_rpe_t, avg_rpe_r, total_forward_time, total_model_fwd_time, scene_timing_records
    
    return 0.0, 0.0, 0.0, 0.0, 0.0, []


def _get_method_labels(detailed_steps, pft):
    """Infer method-specific labels for plot legends from detailed_steps or per_frame keys."""
    labels = {
        "encode": "Encode",
        "forward": "Forward",
        "decode": "Decode",
    }
    
    if not detailed_steps:
        # Fallback: infer from per_frame keys
        if pft:
            has_camhead = any("n_frames_in_camhead" in f for f in pft)
            if has_camhead:
                labels["encode"] = "Encode (SigLIP + proj + pool)"
                labels["forward"] = "LLM Forward (Qwen2, KV cache)"
                labels["decode"] = "CameraHead (bidir, 0..i frames)"
        return labels
    
    # Collect all step names and descriptions
    all_descs = " ".join(s.get("description", "") for s in detailed_steps)
    all_steps = " ".join(s.get("step", "") for s in detailed_steps)
    
    # Detect model type
    is_cambrian_p = "SigLIP" in all_descs or "Qwen2" in all_descs or "LLM" in all_descs
    
    if is_cambrian_p:
        labels["encode"] = "Encode (SigLIP + proj + pool)"
        labels["forward"] = "LLM Forward (Qwen2, KV cache)"
        labels["decode"] = "CameraHead (bidir, 0..i frames)"
    else:
        # Generic: try to extract from step descriptions
        for step in detailed_steps:
            s = step.get("step", "")
            desc = step.get("description", "")
            if "encode" in s.lower() and ("ViT" in desc or "SigLIP" in desc):
                labels["encode"] = desc.split(".")[0][:40] if len(desc) > 40 else desc
            elif "forward" in s.lower() or "recurrent" in s.lower():
                labels["forward"] = desc.split(".")[0][:40] if len(desc) > 40 else desc
    
    return labels


def _save_timing_incremental(record, all_records, save_dir, update_plots=False):
    """Incrementally save per-frame timing after each scene.
    
    - Always appends to per_frame_timing.csv (append mode, includes gpu_to_cpu_ms)
    - Always saves per-scene detailed JSON (with sub-step descriptions)
    - Always saves per-scene plot
    - Optionally regenerates aggregate plots (every N scenes)
    """
    import csv
    
    timing_dir = os.path.join(save_dir, "timing")
    os.makedirs(timing_dir, exist_ok=True)
    per_scene_dir = os.path.join(timing_dir, "per_scene")
    os.makedirs(per_scene_dir, exist_ok=True)
    
    scene = record.get("scene", "unknown")
    pft = record.get("per_frame_times", [])
    
    # 1. Append per-frame rows to CSV (with gpu_to_cpu column)
    csv_path = os.path.join(timing_dir, "per_frame_timing.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["scene", "frame_id", "encode_ms", "forward_ms",
                         "gpu_to_cpu_ms", "decode_ms", "total_ms"])
        for ft in pft:
            w.writerow([
                scene, ft["frame_id"],
                f"{ft.get('encode_ms', 0.0):.3f}",
                f"{ft.get('forward_ms', 0.0):.3f}",
                f"{ft.get('gpu_to_cpu_ms', 0.0):.3f}",
                f"{ft.get('decode_ms', 0.0):.3f}",
                f"{ft.get('total_ms', 0.0):.3f}",
            ])
    
    # 2. Save per-scene detailed JSON (with sub-step descriptions)
    one_time = record.get("one_time", {})
    detailed_steps = one_time.get("detailed_steps", [])
    summary = record.get("summary", {})
    
    scene_json = {
        "scene": scene,
        "num_frames": record.get("num_frames", len(pft)),
        "pipeline_steps": [],
        "per_frame_timing": [],
        "summary": summary,
    }
    
    # Build pipeline_steps from detailed_steps
    for step in detailed_steps:
        entry = {
            "step": step["step"],
            "description": step["description"],
            "total_ms": round(step["time_ms"], 3),
        }
        if step.get("per_frame"):
            entry["per_frame_ms"] = round(step.get("per_frame_ms", 0), 3)
            if "per_frame_median_ms" in step:
                entry["per_frame_median_ms"] = round(step["per_frame_median_ms"], 3)
            if "per_frame_min_ms" in step:
                entry["per_frame_min_ms"] = round(step["per_frame_min_ms"], 3)
            if "per_frame_max_ms" in step:
                entry["per_frame_max_ms"] = round(step["per_frame_max_ms"], 3)
        scene_json["pipeline_steps"].append(entry)
    
    # Add per-frame timing (compact: only key fields)
    for ft in pft:
        scene_json["per_frame_timing"].append({
            "frame_id": ft["frame_id"],
            "encode_ms": round(ft.get("encode_ms", 0.0), 3),
            "forward_ms": round(ft.get("forward_ms", 0.0), 3),
            "gpu_to_cpu_ms": round(ft.get("gpu_to_cpu_ms", 0.0), 3),
            "decode_ms": round(ft.get("decode_ms", 0.0), 3),
            "total_ms": round(ft.get("total_ms", 0.0), 3),
        })
    
    json_path = os.path.join(per_scene_dir, f"{scene}_timing.json")
    with open(json_path, "w") as f:
        json.dump(scene_json, f, indent=2)
    
    # 3. Save this scene's per-frame plot (method-aware labels)
    if pft:
        frame_ids = np.array([f["frame_id"] for f in pft])
        encode_ms = np.array([f.get("encode_ms", 0.0) for f in pft])
        forward_ms = np.array([f.get("forward_ms", 0.0) for f in pft])
        gpu_to_cpu_ms = np.array([f.get("gpu_to_cpu_ms", 0.0) for f in pft])
        decode_ms = np.array([f.get("decode_ms", 0.0) for f in pft])
        total_ms = np.array([f.get("total_ms", 0.0) for f in pft])
        
        # Detect method from detailed_steps or per_frame keys
        method_labels = _get_method_labels(detailed_steps, pft)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        
        # Left: line plot (per-stage)
        ax = axes[0]
        ax.plot(frame_ids, encode_ms, "m-d", ms=2, lw=1.2, alpha=0.8, label=method_labels["encode"])
        ax.plot(frame_ids, forward_ms, "r-s", ms=2, lw=1.2, alpha=0.8, label=method_labels["forward"])
        if gpu_to_cpu_ms.sum() > 0:
            ax.plot(frame_ids, gpu_to_cpu_ms, "c-x", ms=2, lw=1, alpha=0.7, label="GPU→CPU")
        ax.plot(frame_ids, decode_ms, "g-^", ms=2, lw=1.2, alpha=0.8, label=method_labels["decode"])
        ax.plot(frame_ids, total_ms, "b-o", ms=3, lw=1.5, label="Total")
        ax.set_xlabel("Frame Index", fontsize=12)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title(f"{scene} — Per-Stage Timing", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Right: stacked area plot
        ax2 = axes[1]
        stack_data = [encode_ms, forward_ms, decode_ms]
        stack_labels = [method_labels["encode"], method_labels["forward"], method_labels["decode"]]
        stack_colors = ["#9b59b6", "#e74c3c", "#2ecc71"]
        if gpu_to_cpu_ms.sum() > 0:
            stack_data.insert(2, gpu_to_cpu_ms)
            stack_labels.insert(2, "GPU→CPU")
            stack_colors.insert(2, "#1abc9c")
        ax2.stackplot(frame_ids, *stack_data, labels=stack_labels, colors=stack_colors, alpha=0.7)
        ax2.plot(frame_ids, total_ms, "k-", lw=1.5, alpha=0.5, label="Total")
        ax2.set_xlabel("Frame Index", fontsize=12)
        ax2.set_ylabel("Cumulative Time (ms)", fontsize=12)
        ax2.set_title(f"{scene} — Stacked Area ({len(pft)} frames)", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(per_scene_dir, f"{scene}_frame_timing.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    
    # 4. Optionally update aggregate plots
    if update_plots and len(all_records) >= 2:
        try:
            _save_aggregate_plots(all_records, timing_dir)
        except Exception as e:
            print(f"[timing] Aggregate plot update failed: {e}")


def _save_aggregate_plots(records, timing_dir):
    """Generate aggregate overlay plots from all records so far."""
    # Infer labels
    agg_labels = {"encode": "Encode", "forward": "Forward", "decode": "Decode"}
    if records:
        ot = records[0].get("one_time", {})
        ds = ot.get("detailed_steps", [])
        pft0 = records[0].get("per_frame_times", [])
        agg_labels = _get_method_labels(ds, pft0)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    for ax, key, label, color in zip(
        axes,
        ["encode_ms", "forward_ms", "decode_ms", "total_ms"],
        [agg_labels["encode"], agg_labels["forward"], agg_labels["decode"], "Total"],
        ["#9b59b6", "#e74c3c", "#2ecc71", "#2980b9"],
    ):
        for rec in records:
            pft = rec.get("per_frame_times", [])
            if not pft: continue
            fids = [f["frame_id"] for f in pft]
            vals = [f.get(key, 0.0) for f in pft]
            ax.plot(fids, vals, alpha=0.35, lw=0.8, color=color)
        
        # Mean line
        all_vals_by_frame = {}
        for rec in records:
            for f in rec.get("per_frame_times", []):
                fid = f["frame_id"]
                all_vals_by_frame.setdefault(fid, []).append(f.get(key, 0.0))
        if all_vals_by_frame:
            sorted_fids = sorted(all_vals_by_frame.keys())
            mean_vals = [np.mean(all_vals_by_frame[fid]) for fid in sorted_fids]
            ax.plot(sorted_fids, mean_vals, color="black", lw=2, alpha=0.8, label="Mean")
            ax.legend(fontsize=9)
        
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Time (ms)")
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Streaming Per-Frame Timing ({len(records)} scenes)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(timing_dir, "all_scenes_frame_timing.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_timing_analysis(records, save_dir):
    """Save per-scene timing CSV and generate per-frame timing plots.
    
    Each record has 'per_frame_times': list of dicts with per-frame timing,
    and 'one_time.detailed_steps': list of sub-step descriptions with timing.
    
    Generates:
      - per_frame_timing.csv: per-frame timing for all scenes (with gpu_to_cpu)
      - per_scene_timing.csv: per-scene summary
      - per_scene/<scene>_timing.json: detailed sub-step breakdown per scene
      - per_scene/<scene>_frame_timing.png: per-scene plot
      - all_scenes_frame_timing.png: aggregate overlay
      - timing_summary.txt: comprehensive human-readable summary with pipeline description
    """
    import csv
    
    timing_dir = os.path.join(save_dir, "timing")
    os.makedirs(timing_dir, exist_ok=True)
    
    # 1. Save per-frame CSV (all scenes) — with gpu_to_cpu column
    csv_path = os.path.join(timing_dir, "per_frame_timing.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "frame_id", "encode_ms", "forward_ms",
                     "gpu_to_cpu_ms", "decode_ms", "total_ms"])
        for rec in records:
            scene = rec.get("scene", "unknown")
            for ft in rec.get("per_frame_times", []):
                w.writerow([
                    scene, ft["frame_id"],
                    f"{ft.get('encode_ms', 0.0):.3f}",
                    f"{ft.get('forward_ms', 0.0):.3f}",
                    f"{ft.get('gpu_to_cpu_ms', 0.0):.3f}",
                    f"{ft.get('decode_ms', 0.0):.3f}",
                    f"{ft.get('total_ms', 0.0):.3f}",
                ])
    print(f"[timing] Saved: {csv_path}")
    
    # 2. Save per-scene summary CSV
    summary_csv = os.path.join(timing_dir, "per_scene_timing.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "num_frames",
                     "encode_total_ms", "forward_total_ms", "gpu_to_cpu_total_ms",
                     "decode_total_ms", "total_ms"])
        for rec in records:
            pft = rec.get("per_frame_times", [])
            enc_t = sum(ft.get("encode_ms", 0) for ft in pft)
            fwd_t = sum(ft.get("forward_ms", 0) for ft in pft)
            cpu_t = sum(ft.get("gpu_to_cpu_ms", 0) for ft in pft)
            dec_t = sum(ft.get("decode_ms", 0) for ft in pft)
            tot_t = sum(ft.get("total_ms", 0) for ft in pft)
            w.writerow([
                rec.get("scene", "unknown"), rec["num_frames"],
                f"{enc_t:.3f}", f"{fwd_t:.3f}", f"{cpu_t:.3f}",
                f"{dec_t:.3f}", f"{tot_t:.3f}",
            ])
    print(f"[timing] Saved: {summary_csv}")
    
    # 3. Per-scene plots + JSON (detailed sub-steps)
    per_scene_dir = os.path.join(timing_dir, "per_scene")
    os.makedirs(per_scene_dir, exist_ok=True)
    
    for rec in records:
        scene = rec.get("scene", "unknown")
        pft = rec.get("per_frame_times", [])
        if not pft:
            continue
        
        # Per-scene JSON with detailed steps
        one_time = rec.get("one_time", {})
        detailed_steps = one_time.get("detailed_steps", [])
        scene_json = {
            "scene": scene,
            "num_frames": rec.get("num_frames", len(pft)),
            "pipeline_steps": [],
            "per_frame_timing": [],
            "summary": rec.get("summary", {}),
        }
        for step in detailed_steps:
            entry = {"step": step["step"], "description": step["description"],
                     "total_ms": round(step["time_ms"], 3)}
            if step.get("per_frame"):
                entry["per_frame_ms"] = round(step.get("per_frame_ms", 0), 3)
                for k in ["per_frame_median_ms", "per_frame_min_ms", "per_frame_max_ms"]:
                    if k in step:
                        entry[k] = round(step[k], 3)
            scene_json["pipeline_steps"].append(entry)
        for ft in pft:
            scene_json["per_frame_timing"].append({
                "frame_id": ft["frame_id"],
                "encode_ms": round(ft.get("encode_ms", 0.0), 3),
                "forward_ms": round(ft.get("forward_ms", 0.0), 3),
                "gpu_to_cpu_ms": round(ft.get("gpu_to_cpu_ms", 0.0), 3),
                "decode_ms": round(ft.get("decode_ms", 0.0), 3),
                "total_ms": round(ft.get("total_ms", 0.0), 3),
            })
        with open(os.path.join(per_scene_dir, f"{scene}_timing.json"), "w") as f:
            json.dump(scene_json, f, indent=2)
        
        # Per-scene plot (method-aware labels + stacked area)
        frame_ids = np.array([f["frame_id"] for f in pft])
        encode_ms = np.array([f.get("encode_ms", 0.0) for f in pft])
        forward_ms = np.array([f.get("forward_ms", 0.0) for f in pft])
        gpu_to_cpu_ms = np.array([f.get("gpu_to_cpu_ms", 0.0) for f in pft])
        decode_ms = np.array([f.get("decode_ms", 0.0) for f in pft])
        total_ms = np.array([f.get("total_ms", 0.0) for f in pft])
        
        method_labels = _get_method_labels(detailed_steps, pft)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        
        # Left: line plot
        ax = axes[0]
        ax.plot(frame_ids, encode_ms, "m-d", ms=2, lw=1.2, alpha=0.8, label=method_labels["encode"])
        ax.plot(frame_ids, forward_ms, "r-s", ms=2, lw=1.2, alpha=0.8, label=method_labels["forward"])
        if gpu_to_cpu_ms.sum() > 0:
            ax.plot(frame_ids, gpu_to_cpu_ms, "c-x", ms=2, lw=1, alpha=0.7, label="GPU→CPU")
        ax.plot(frame_ids, decode_ms, "g-^", ms=2, lw=1.2, alpha=0.8, label=method_labels["decode"])
        ax.plot(frame_ids, total_ms, "b-o", ms=3, lw=1.5, label="Total")
        ax.set_xlabel("Frame Index", fontsize=12)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title(f"{scene} — Per-Stage Timing", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Right: stacked area
        ax2 = axes[1]
        stack_data = [encode_ms, forward_ms, decode_ms]
        stack_labels = [method_labels["encode"], method_labels["forward"], method_labels["decode"]]
        stack_colors = ["#9b59b6", "#e74c3c", "#2ecc71"]
        if gpu_to_cpu_ms.sum() > 0:
            stack_data.insert(2, gpu_to_cpu_ms)
            stack_labels.insert(2, "GPU→CPU")
            stack_colors.insert(2, "#1abc9c")
        ax2.stackplot(frame_ids, *stack_data, labels=stack_labels, colors=stack_colors, alpha=0.7)
        ax2.plot(frame_ids, total_ms, "k-", lw=1.5, alpha=0.5, label="Total")
        ax2.set_xlabel("Frame Index", fontsize=12)
        ax2.set_ylabel("Cumulative Time (ms)", fontsize=12)
        ax2.set_title(f"{scene} — Stacked Area ({len(pft)} frames)", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(per_scene_dir, f"{scene}_frame_timing.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    
    print(f"[timing] Saved {len(records)} per-scene plots + JSONs to: {per_scene_dir}")
    
    # 4. Aggregate overlay plot (method-aware labels)
    # Infer labels from first record
    agg_method_labels = {"encode": "Encode", "forward": "Forward", "decode": "Decode"}
    if records:
        ot = records[0].get("one_time", {})
        ds = ot.get("detailed_steps", [])
        pft0 = records[0].get("per_frame_times", [])
        agg_method_labels = _get_method_labels(ds, pft0)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    for ax, key, label, color in zip(
        axes,
        ["encode_ms", "forward_ms", "decode_ms", "total_ms"],
        [agg_method_labels["encode"], agg_method_labels["forward"],
         agg_method_labels["decode"], "Total"],
        ["#9b59b6", "#e74c3c", "#2ecc71", "#2980b9"],
    ):
        for rec in records:
            pft = rec.get("per_frame_times", [])
            if not pft: continue
            fids = [f["frame_id"] for f in pft]
            vals = [f.get(key, 0.0) for f in pft]
            ax.plot(fids, vals, alpha=0.35, lw=0.8, color=color)
        
        # Add mean line across all scenes
        all_vals_by_frame = {}
        for rec in records:
            for f in rec.get("per_frame_times", []):
                fid = f["frame_id"]
                all_vals_by_frame.setdefault(fid, []).append(f.get(key, 0.0))
        if all_vals_by_frame:
            sorted_fids = sorted(all_vals_by_frame.keys())
            mean_vals = [np.mean(all_vals_by_frame[fid]) for fid in sorted_fids]
            ax.plot(sorted_fids, mean_vals, color="black", lw=2, alpha=0.8, label="Mean")
            ax.legend(fontsize=9)
        
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Time (ms)")
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Streaming Per-Frame Timing ({len(records)} scenes)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    agg_path = os.path.join(timing_dir, "all_scenes_frame_timing.png")
    fig.savefig(agg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[timing] Saved: {agg_path}")
    
    # 5. Comprehensive timing summary with pipeline description
    all_encode = []
    all_forward = []
    all_gpu_cpu = []
    all_decode = []
    all_total = []
    for rec in records:
        for ft in rec.get("per_frame_times", []):
            all_encode.append(ft.get("encode_ms", 0.0))
            all_forward.append(ft.get("forward_ms", 0.0))
            all_gpu_cpu.append(ft.get("gpu_to_cpu_ms", 0.0))
            all_decode.append(ft.get("decode_ms", 0.0))
            all_total.append(ft.get("total_ms", 0.0))
    
    all_encode = np.array(all_encode)
    all_forward = np.array(all_forward)
    all_gpu_cpu = np.array(all_gpu_cpu)
    all_decode = np.array(all_decode)
    all_total = np.array(all_total)
    
    # Collect pipeline step descriptions from first record
    first_steps = []
    if records:
        ot = records[0].get("one_time", {})
        first_steps = ot.get("detailed_steps", [])
    
    summary_path = os.path.join(timing_dir, "timing_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'='*80}\n")
        f.write(f"STREAMING TIMING SUMMARY\n")
        f.write(f"{'='*80}\n\n")
        
        total_frames = sum(r["num_frames"] for r in records)
        total_model = sum(r.get("total_model_time", r.get("total_time", 0)) for r in records)
        f.write(f"Scenes: {len(records)}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Total model time: {total_model:.2f}s ({total_model/60:.1f}min)\n")
        f.write(f"Avg model time per scene: {total_model/len(records):.3f}s\n\n")
        
        # Pipeline description
        f.write(f"{'='*80}\n")
        f.write(f"PIPELINE DESCRIPTION (per-step timing from first scene)\n")
        f.write(f"{'='*80}\n\n")
        for step in first_steps:
            pf_str = ""
            if step.get("per_frame"):
                pf_str = f"  [per-frame: {step.get('per_frame_ms', 0):.1f}ms avg]"
            f.write(f"  {step['step']:35s} {step['time_ms']:8.1f}ms{pf_str}\n")
            f.write(f"    └─ {step['description']}\n\n")
        
        # Per-stage aggregate statistics
        f.write(f"{'='*80}\n")
        f.write(f"PER-FRAME TIMING STATISTICS (across {len(all_total)} frames)\n")
        f.write(f"{'='*80}\n\n")
        
        def _stats_line(name, arr):
            if len(arr) == 0: return ""
            return (f"  {name:<20s}: mean={np.mean(arr):7.2f}ms  "
                    f"median={np.median(arr):7.2f}ms  "
                    f"min={np.min(arr):7.2f}ms  max={np.max(arr):7.2f}ms  "
                    f"std={np.std(arr):6.2f}ms  "
                    f"total={np.sum(arr):10.1f}ms\n")
        
        # Infer method-aware labels for the summary
        sum_labels = _get_method_labels(first_steps, [])
        
        f.write(_stats_line(sum_labels["encode"], all_encode))
        f.write(_stats_line(sum_labels["forward"], all_forward))
        if all_gpu_cpu.sum() > 0:
            f.write(_stats_line("GPU→CPU transfer", all_gpu_cpu))
        f.write(_stats_line(sum_labels["decode"], all_decode))
        f.write(_stats_line("TOTAL", all_total))
        
        # Percentage breakdown
        total_sum = all_total.sum()
        if total_sum > 0:
            f.write(f"\n  Time breakdown:\n")
            f.write(f"    Encode:    {all_encode.sum():10.1f}ms  ({all_encode.sum()/total_sum*100:5.1f}%)\n")
            f.write(f"    Forward:   {all_forward.sum():10.1f}ms  ({all_forward.sum()/total_sum*100:5.1f}%)\n")
            f.write(f"    GPU→CPU:   {all_gpu_cpu.sum():10.1f}ms  ({all_gpu_cpu.sum()/total_sum*100:5.1f}%)\n")
            f.write(f"    Decode:    {all_decode.sum():10.1f}ms  ({all_decode.sum()/total_sum*100:5.1f}%)\n")
            f.write(f"    ─────────────────────────\n")
            f.write(f"    Total:     {total_sum:10.1f}ms  (100.0%)\n")
        
        # Per-scene table
        f.write(f"\n{'='*80}\n")
        f.write(f"PER-SCENE BREAKDOWN\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Scene':<25} {'N':>5} {'Encode':>10} {'Forward':>10} {'GPU→CPU':>10} {'Decode':>10} {'Total':>10}  {'ms/frm':>8}\n")
        f.write("-" * 95 + "\n")
        for rec in records:
            pft = rec.get("per_frame_times", [])
            n = rec["num_frames"]
            enc_t = sum(ft.get("encode_ms", 0) for ft in pft)
            fwd_t = sum(ft.get("forward_ms", 0) for ft in pft)
            cpu_t = sum(ft.get("gpu_to_cpu_ms", 0) for ft in pft)
            dec_t = sum(ft.get("decode_ms", 0) for ft in pft)
            tot_t = sum(ft.get("total_ms", 0) for ft in pft)
            ms_per = tot_t / n if n > 0 else 0
            f.write(f"{rec.get('scene','?'):<25} {n:>5} "
                    f"{enc_t:>9.1f}  {fwd_t:>9.1f}  {cpu_t:>9.1f}  "
                    f"{dec_t:>9.1f}  {tot_t:>9.1f}  {ms_per:>7.1f}\n")
    
    print(f"[timing] Saved: {summary_path}")

def main():
    args = get_args_parser().parse_args()

    if args.sampling_strategy == "monst3r":
        if args.pose_eval_stride != 1:
            print(f"[monst3r] Overriding --pose_eval_stride {args.pose_eval_stride} -> 1")
            args.pose_eval_stride = 1
        if args.num_frames_per_scene is not None and args.num_frames_per_scene < 9999:
            print(f"[monst3r] Overriding --num_frames_per_scene {args.num_frames_per_scene} -> None (all frames)")
            args.num_frames_per_scene = None

    # Output directory
    suffix = f"_{args.model_type}_{args.sampling_strategy}"
    if args.random_init:
        suffix += "_random"
    if args.reinit_camera_head:
        suffix += "_reinit_cam"
    if args.llava_mode == "streaming":
        suffix += "_streaming"
    if args.alignment == "se3":
        suffix += "_se3"
    if not args.no_timestamp:
        suffix += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = args.output_dir + suffix
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"UNIFIED POSE ESTIMATION EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_type}")
    print(f"Weights: {args.model_path}")
    print(f"Random init: {args.random_init}")
    print(f"Sampling: {args.sampling_strategy}")
    print(f"LLaVA mode: {args.llava_mode}")
    if args.sampling_strategy == "monst3r":
        print(f"  [monst3r protocol] stride=1, all frames, using pre-extracted data")
    else:
        print(f"  stride={args.pose_eval_stride}, {args.num_frames_per_scene} frames")
    print(f"Size: {args.size}")
    print(f"Alignment: {args.alignment}")
    print(f"Output: {args.output_dir}\n")
    
    # Load model
    tokenizer, image_processor = None, None
    
    if args.model_type == "llava_vggt":
        tokenizer, model, image_processor = load_llava_vggt_model(
            args.model_path, device=args.device,
            rec_embed_dim=args.rec_embed_dim,
            camera_tokens_place=args.camera_tokens_place
        )
        
        # Optionally re-init camera head in-place (no separate checkpoint needed)
        if args.reinit_camera_head:
            print(">>> Re-initializing camera head parameters...")
            reinit_count = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "rec_head.camera_head" in name or "rec_head.camera_tokens" in name:
                        if "weight" in name and param.dim() >= 2:
                            torch.nn.init.trunc_normal_(param, std=0.02)
                        elif "bias" in name:
                            if "norm" in name.lower() or "ln" in name.lower():
                                param.zero_()
                            else:
                                param.zero_()
                        elif "norm" in name.lower() and "weight" in name:
                            param.fill_(1.0)
                        elif "camera_tokens" in name:
                            torch.nn.init.normal_(param, std=1e-6)
                        else:
                            torch.nn.init.trunc_normal_(param, std=0.02)
                        reinit_count += 1
            print(f">>> Re-initialized {reinit_count} camera head parameters")
        
    print("Model loaded.\n")
    
    # Run evaluation
    ate, rpe_t, rpe_r, fwd_time, mdl_fwd_time, scene_timings = evaluate(args, model, tokenizer, image_processor, args.output_dir)
    
    # Compute average per-frame LLM time from streaming timing records
    avg_frame_str = ""
    if scene_timings:
        all_frame_llm_ms = []
        for rec in scene_timings:
            for ft in rec.get("per_frame_times", []):
                all_frame_llm_ms.append(ft["llm_time"] * 1000)
        if all_frame_llm_ms:
            avg_ms = np.mean(all_frame_llm_ms)
            total_frames = len(all_frame_llm_ms)
            avg_frame_str = f", AVG_FRAME_LLM={avg_ms:.1f}ms ({total_frames} frames)"
    
    print(f"\nFinal: ATE={ate:.5f}m, RPE_T={rpe_t:.5f}m, RPE_R={rpe_r:.2f}°, WALL={fwd_time:.2f}s, MODEL_FWD={mdl_fwd_time:.2f}s{avg_frame_str}")


if __name__ == "__main__":
    main()
