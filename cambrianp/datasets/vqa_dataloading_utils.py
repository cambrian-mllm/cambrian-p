"""
VQA Data Loading Utilities

This module handles:
- Video file loading (Cambrian-S style with clip boundaries)
- Generic video loading (decord)
- Frame processing for VLM input
- Deprecated camera/frame_index prompt functions (kept for compatibility)

For scene-based datasets (ScanNet, etc.), see rec_dataloading_utils.py
"""

import os
import os.path as osp
import numpy as np
import copy
import re
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch

from cambrianp.utils import (
    process_video_with_decord,
    process_video_with_decord_byframe,
    process_video_with_decord_bytime,
    process_gif_with_imageio,
)

import cambrianp.datasets.mapanything_dataloading_utils as ma_utils

REC_DIR_PATTERNS = (
    "processed_scannet_f",
    "scannet",
    "arkitscenes",
    "scannetpp",
    "co3d",
)


def check_video_loading_type(video_file: str, loading_type: str = None) -> str:
    """
    Infer how `video_file` should be interpreted for loading.

    Returns:
        - "rec": directory containing frames for reconstruction datasets (ScanNet, ARKitScenes, MapAnything, etc.)
        - "video_file": regular video file (mp4/avi/...) or any non-matching directory
    """
    # if there are already given loading_type
    if loading_type is not None:
        return loading_type

    # for the rest of the cases
    loading_type = "video_file"

    # Handle reconstruction data from scene directories (including MapAnything)
    if osp.isdir(video_file):
        vf = video_file.lower()
        if any(pattern in vf for pattern in REC_DIR_PATTERNS):
            loading_type = "rec"
        # MapAnything scenes are also treated as 'rec' type
        elif ma_utils.is_mapanything_scene(video_file):
            loading_type = "rec"

    return loading_type

# =============================================================================
# Video Clip Info (Cambrian-S style)
# =============================================================================

@dataclass
class VideoClipInfo:
    """Clip boundaries for video-based datasets."""
    video: Optional[np.ndarray] = None
    video_time: Optional[float] = None
    frame_time: Optional[str] = None
    num_frames_to_sample: Optional[int] = None
    
    def __init__(self, data_item: dict, video_file: str, data_args: Any):
        """Initialize VideoClipInfo from data item and video file.
        
        modified based on https://github.com/cambrian-mllm/cambrian-s/blob/e4b88142ffa6d427e87c513f61fcffc2afc4c7e2/cambrian/train/train_fsdp.py#L1298-L1304
        
        Args:
            data_item (dict): Data item containing start_frame, end_frame, fps, current_observation_frame
            video_file (str): Path to video file or directory
            data_args (Any): Data arguments (frames_upbound, force_sample, etc.)
        """
        if os.path.isdir(video_file):
            if "shareVideoGPTV" in video_file:  # shareVideoGPTV use 2FPS
                avg_fps = 2
            elif "TVQA" in video_file:  # TVQA use 3FPS
                avg_fps = 3
            else:  # for unknown video frames, we assume it is 1FPS
                avg_fps = 1

            frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
            # Ensure the frames are sorted if they are named sequentially
            frame_files.sort()
            
            video_time = len(frame_files) / avg_fps

            if 'start_frame' in data_item:
                start_frame = data_item['start_frame']
                end_time = data_item['end_frame']
                start_time = start_frame / avg_fps
                end_frame = int(end_time * avg_fps)
                end_frame = min(len(frame_files) - 1, end_frame)
                frame_files = frame_files[start_frame:end_frame + 1]  # from start to end
                video_time = end_time - start_time
            
            frame_idx = [i for i in range(0, len(frame_files), avg_fps)]
            frame_time = [i / avg_fps for i in frame_idx]

            if data_args.frames_upbound > 0:
                if len(frame_files) > data_args.frames_upbound or data_args.force_sample:
                    frame_idx = np.linspace(0, len(frame_files) - 1, data_args.frames_upbound, dtype=int).tolist()
                    frame_time = [i/avg_fps for i in frame_idx]
        
            frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

            # Read and store the sampled frames
            num_frames_to_sample = len(frame_idx)
            video = []
            for idx in frame_idx:
                frame_path = frame_files[idx]
                try:
                    with Image.open(frame_path) as img:
                        frame = img.convert("RGB")
                        video.append(np.array(frame))
                except IOError:
                    print(f"Failed to read frame at path: {frame_path}")
            video = np.stack(video)
        elif video_file.endswith(".gif"):
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))
                raise FileNotFoundError
            assert "start" not in data_item and "end" not in data_item, "start and end should not be in gif video"
            assert "start_frame" not in data_item and "end_frame" not in data_item, "start_frame and end_frame should not be in gif video"
            video, video_time, frame_time, num_frames_to_sample = process_gif_with_imageio(video_file, data_args)
        else:
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))
                raise FileNotFoundError

            if 'start_frame' in data_item:
                start_frame = data_item['start_frame']
                end_frame = data_item['end_frame']
                current_observation_frame = data_item.get('current_observation_frame', None)

                video, video_time, frame_time, num_frames_to_sample = process_video_with_decord_byframe(video_file, data_args, start_frame, end_frame, current_observation_frame)
                if not video.size > 0:
                    raise ValueError(f"Video {video_file} is empty")
            elif 'start' in data_item:
                start_time = data_item['start']
                end_time = data_item['end']
                video, video_time, frame_time, num_frames_to_sample = process_video_with_decord_bytime(video_file, data_args, start_time, end_time)
            else:
                video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, data_args)
        
        self.video = video
        self.video_time = video_time
        self.frame_time = frame_time
        self.num_frames_to_sample = num_frames_to_sample
        

def extract_clip_info(data_item: dict, video_file: str, data_args: Any) -> Optional[VideoClipInfo]:
    """
    Extract VideoClipInfo from a data item if present.
    
    Supports two formats:
    - Frame-based: {'start_frame': 100, 'end_frame': 200, ...}
    - Time-based: {'start': '00:01:30', 'end': '00:01:45', ...}
    
    Returns None if no clip info found (scene-based datasets).
    """
    return VideoClipInfo(data_item, video_file, data_args)


# =============================================================================
# Frame Processing for VLM
# =============================================================================

# def _rec_views_to_pil(images_chw: np.ndarray) -> List[Image.Image]:
#     """Convert rec_views images from (L, C, H, W) to list of PIL Images."""
#     images_hwc = images_chw.transpose(0, 2, 3, 1).astype(np.uint8)
#     return [Image.fromarray(img) for img in images_hwc]

def _rec_views_to_pil(images_chw) -> List[Image.Image]:
    """Convert rec_views images from (L, C, H, W) to list of PIL Images."""
    if isinstance(images_chw, torch.Tensor):
        images_chw = images_chw.numpy()
    images_hwc = images_chw.transpose(0, 2, 3, 1)
    # Scale from [0, 1] to [0, 255] if in float range
    if images_hwc.max() <= 1.0:
        images_hwc = (images_hwc * 255).astype(np.uint8)
    else:
        images_hwc = images_hwc.astype(np.uint8)
    return [Image.fromarray(img) for img in images_hwc]

def get_video_frames_for_vlm(
    video: Optional[List[Image.Image]],
    rec_views: Optional[dict],
    is_real_vqa: bool,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Get frames for VLM processing, choosing appropriate source.
    
    Priority:
    1. VQA mode with video -> use video frames
    2. Rec mode with rec_views -> use rec_views['images_for_llava']
    3. Fallback to video if available
    
    Returns:
        (frames_pil, original_size)
    """
    # VQA mode: use original frames
    if video is not None and is_real_vqa:
        return video, video[0].size
    
    # Rec mode: use processed frames from rec_views
    if rec_views is not None and 'images_for_llava' in rec_views:
        frames_pil = _rec_views_to_pil(rec_views['images_for_llava'])
        return frames_pil, (384, 384)
    
    # Fallback
    if video is not None:
        return video, video[0].size
    
    raise ValueError("No video frames or rec_views available")


def process_video_frames(
    processor,
    video: Optional[List[Image.Image]],
    rec_views: Optional[dict],
    is_real_vqa: bool,
) -> Tuple[torch.Tensor, Tuple[int, int], List[Tuple]]:
    """
    Process video frames for VLM input using image processor.
    
    Returns:
        (processed_frames, original_size, image_payload)
        
    image_payload is [(processed_frames, original_size, "video")]
    """
    frames_pil, original_size = get_video_frames_for_vlm(video, rec_views, is_real_vqa)
    
    processed_frames = processor.preprocess(
        frames_pil, return_tensors="pt"
    )["pixel_values"]
    
    image = [(processed_frames, original_size, "video")]
    
    return processed_frames, original_size, image


# =============================================================================
# Debug Utilities
# =============================================================================

def print_data_stats(sample_idx: int, rec_views: Optional[dict], data_dict: dict, list_data_dict: dict):
    """Print debug stats for loaded data."""
    def _print_stats(name, arr):
        if hasattr(arr, 'shape'):
            print(f"[Sample {sample_idx}] {name}: shape={arr.shape}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")
    
    if rec_views is not None:
        for key in ['images', 'images_for_llava', 'cam2world', 'world2cam', 'intrinsics', 'depths']:
            if key in rec_views:
                _print_stats(f"rec_views['{key}']", rec_views[key])
    
    if "video" in list_data_dict and 'image' in data_dict:
        _print_stats("processed_frames", data_dict["image"][0][0])


def validate_extracted_features_config(sample_mode: str, interleaved_training: bool, 
                                       is_real_vqa: bool, feature_fusion_type: str):
    """
    [DEPRECATED] Validate config for extracted features mode.
    
    Only used when data_args.load_extracted_features=True
    """
    if sample_mode != "unified":
        raise ValueError(
            f"Feature fusion mode '{feature_fusion_type}' requires unified sampling for vqa, "
            f"but got sample_mode='{sample_mode}'"
        )
    if interleaved_training and not is_real_vqa:
        raise ValueError(
            "The vqa should always in data, and we are not using interleaved mode here"
        )


def append_cam_token_to_questions(
    sources: List[Dict],
) -> List:
    """
    [DEPRECATED] Append <cam> token to human questions for 'between_qa' camera token placement.
    
    Returns the modified sources.
    """
    for conv in sources[0]['conversations']:
        if conv['from'] == 'human' and not conv['value'].endswith('<cam>'):
            conv['value'] = conv['value'] + '<cam>'
    
    return sources
