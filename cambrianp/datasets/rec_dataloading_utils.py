"""
Reconstruction Data Loading Utilities

This module handles loading frames from scene-based datasets:
- ScanNet
- ScanNet++ 
- ARKitScenes
- CO3D

For video-based datasets (Cambrian-S), see vqa_dataloading_utils.py
"""

import os
import os.path as osp
import numpy as np
import random
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a scene-based dataset."""
    images_subdir: str
    image_extension: str
    metadata_file: str
    fallback_metadata_file: Optional[str]
    supports_cut3r_pairs: bool
    name_transform: Optional[Callable] = None
    scene_id_prefix: Optional[str] = None
    supports_extracted_features: bool = True
    supports_cut3r_params: bool = True


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "scannet": DatasetConfig(
        images_subdir="color",
        image_extension=".jpg",
        metadata_file="scene_metadata_all.npz",
        fallback_metadata_file="new_scene_metadata.npz",
        supports_cut3r_pairs=False,
    ),
    "scannetpp": DatasetConfig(
        images_subdir="images",
        image_extension=".jpg",
        metadata_file="scene_metadata_all.npz",
        fallback_metadata_file=None,
        supports_cut3r_pairs=True,
    ),
    "arkitscenes": DatasetConfig(
        images_subdir="vga_wide",
        image_extension=".jpg",
        metadata_file="scene_metadata_all.npz",
        fallback_metadata_file=None,
        supports_cut3r_pairs=True,
        name_transform=lambda x: x.replace(".png", ".jpg"),
    ),
    "co3d": DatasetConfig(
        images_subdir="images",
        image_extension=".jpg",
        metadata_file="scene_metadata.npz",
        fallback_metadata_file=None,
        supports_cut3r_pairs=False,
        scene_id_prefix="apple",
        supports_extracted_features=False,
        supports_cut3r_params=False,
    ),
}


# =============================================================================
# Config Helpers
# =============================================================================

def adjust_scene_id_for_dataset(scene_id: str, dataset_type: Optional[str]) -> str:
    """Adjust scene_id with prefix if needed (e.g., CO3D needs 'apple/' prefix)."""
    config = DATASET_CONFIGS.get(dataset_type, None)
    if config and config.scene_id_prefix:
        return f"{config.scene_id_prefix}/{scene_id}"
    return scene_id


def supports_extracted_features(dataset_type: Optional[str]) -> bool:
    """Check if dataset supports extracted features (ScanNet, ScanNet++, ARKitScenes)."""
    config = DATASET_CONFIGS.get(dataset_type, None)
    return config.supports_extracted_features if config else False


def supports_cut3r_params(dataset_type: Optional[str]) -> bool:
    """Check if dataset uses cut3r_params (not CO3D)."""
    config = DATASET_CONFIGS.get(dataset_type, None)
    return config.supports_cut3r_params if config else False


# =============================================================================
# Dataset Detection
# =============================================================================

def detect_dataset_type(video_file: str) -> Optional[str]:
    """
    Detect dataset type from file path.
    
    Returns:
        'scannet', 'scannetpp', 'arkitscenes', 'co3d', or None
    """
    if "scans_train" in video_file:
        return "scannet"
    elif "scannetpp" in video_file:
        return "scannetpp"
    elif "arkitscene" in video_file:
        return "arkitscenes"
    elif "co3d" in video_file:
        return "co3d"
    return None


def get_scene_directory(data_path: str, scene_id: str, dataset_type: str) -> str:
    """Get the full scene directory path for a dataset."""
    SCENE_DIR_TEMPLATES = {
        "scannet": ("processed_scannet_f", "scans_train"),
        "scannetpp": ("scannetpp",),
        "arkitscenes": ("arkitscenes", "Training"),
    }
    parts = SCENE_DIR_TEMPLATES.get(dataset_type)
    if parts:
        return os.path.join(data_path, *parts, scene_id)
    return ""



# =============================================================================
# Frame Sampling Functions
# =============================================================================

def blockwise_shuffle(x, block_size=None, rng=None):
    """Shuffle elements within blocks."""
    if rng is None:
        rng = np.random.RandomState()
    
    if block_size is None or block_size <= 0:
        return rng.permutation(x).tolist()
    
    blocks = [x[i:i + block_size] for i in range(0, len(x), block_size)]
    shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
    return [item for block in shuffled_blocks for item in block]


def get_seq_cut3r(num_views, start_idx, total_frames, 
                  video_prob=0.6, fix_interval_prob=0.6, 
                  max_interval=30, min_interval=1, allow_repeat=False,
                  block_shuffle_size=None):
    """
    Cut3R sampling with video/collection modes.
    
    Video mode: Temporal sequence with fixed or random intervals
    Collection mode: Random sampling with optional block shuffle
    """
    remaining_frames = total_frames - start_idx
    
    if remaining_frames >= num_views:
        video_mode_roll = random.random()
        
        if video_mode_roll < video_prob:
            # VIDEO MODE (temporal sequence)
            fix_interval_roll = random.random()
            
            if fix_interval_roll < fix_interval_prob:
                # Fixed interval sampling
                max_possible_interval = min(
                    (remaining_frames - 1) // (num_views - 1),
                    max_interval
                )
                
                if max_possible_interval >= min_interval:
                    fixed_interval = random.randint(min_interval, max_possible_interval)
                    indices = [start_idx + i * fixed_interval for i in range(num_views)]
                else:
                    fixed_interval = random.randint(1, max(1, max_possible_interval))
                    indices = list(range(start_idx, start_idx + num_views * fixed_interval, fixed_interval))
            else:
                # Random intervals within max_interval
                indices = [start_idx]
                current_pos = start_idx
                
                for _ in range(num_views - 1):
                    remaining_space = total_frames - current_pos - 1
                    remaining_needed = num_views - len(indices)
                    
                    if remaining_space <= 0 or remaining_needed <= 0:
                        break
                    
                    max_jump = min(max_interval, remaining_space // remaining_needed)
                    interval = random.randint(min_interval, max_jump) if max_jump >= min_interval else 1
                    current_pos += interval
                    indices.append(current_pos)
        else:
            # COLLECTION MODE (random sampling)
            available_indices = list(range(start_idx, total_frames))
            if len(available_indices) >= num_views:
                indices = sorted(random.sample(available_indices, num_views))
            else:
                indices = available_indices
                
            if block_shuffle_size is not None and block_shuffle_size > 0:
                indices = blockwise_shuffle(indices, block_size=block_shuffle_size)
    else:
        # Not enough remaining frames
        if allow_repeat:
            indices = sorted(random.choices(list(range(total_frames)), k=num_views))
        else:
            indices = list(range(max(0, total_frames - num_views), total_frames))
    
    # Ensure exactly num_views frames
    if len(indices) < num_views:
        indices.extend([indices[-1]] * (num_views - len(indices)))
    elif len(indices) > num_views:
        indices = indices[:num_views]
            
    return indices


def get_temporal_indices(num_views, start_idx, total_frames, 
                         max_interval=3, min_interval=1, allow_repeat=False):
    """Temporal sampling with fixed intervals."""
    remaining_frames = total_frames - start_idx
    
    if remaining_frames >= num_views:
        max_possible_interval = min(
            (remaining_frames - 1) // (num_views - 1) if num_views > 1 else remaining_frames,
            max_interval
        )
        
        if max_possible_interval >= min_interval:
            fixed_interval = random.randint(min_interval, max_possible_interval)
            indices = [start_idx + i * fixed_interval for i in range(num_views)]
        else:
            indices = list(range(start_idx, start_idx + num_views))
    else:
        if allow_repeat:
            indices = sorted(random.choices(list(range(total_frames)), k=num_views))
        else:
            indices = list(range(max(0, total_frames - num_views), total_frames))
    
    if len(indices) < num_views:
        indices.extend([indices[-1]] * (num_views - len(indices)))
    elif len(indices) > num_views:
        indices = indices[:num_views]
    
    return indices


def select_frame_indices(images_list, sample_mode, num_frames, cut3r_params=None,
                        temporal_params=None, data_args=None, allow_repeat=False):
    """
    Select frame indices based on sampling mode.
    
    Args:
        images_list: List of available images
        sample_mode: 'unified', 'temporal', 'cut3r', or 'random'
        num_frames: Number of frames to select
        cut3r_params: Parameters for cut3r mode
        temporal_params: Parameters for temporal mode
        data_args: Data arguments (for sample_jitter)
        allow_repeat: Whether to allow frame repetition
    
    Returns:
        List of selected frame indices
    """
    total_frames = len(images_list)
    
    if sample_mode == 'random':
        if total_frames >= num_frames and not allow_repeat:
            return sorted(random.sample(range(total_frames), num_frames))
        return sorted(random.choices(range(total_frames), k=num_frames))
    
    elif sample_mode == 'temporal':
        if temporal_params is None:
            raise ValueError("temporal_params required for temporal mode")
        
        max_start_pos = max(0, total_frames - num_frames)
        start_idx = random.randint(0, max_start_pos) if max_start_pos > 0 else 0
        
        return get_temporal_indices(
            num_views=num_frames,
            start_idx=start_idx,
            total_frames=total_frames,
            max_interval=temporal_params['max_interval'],
            min_interval=temporal_params['min_interval'],
            allow_repeat=allow_repeat
        )
        
    elif sample_mode == 'cut3r':
        if cut3r_params is None:
            raise ValueError("cut3r_params required for cut3r mode")
        
        max_start_pos = max(0, total_frames - num_frames)
        start_idx = random.randint(0, max_start_pos) if max_start_pos > 0 else 0
        
        return get_seq_cut3r(
            num_views=num_frames,
            start_idx=start_idx,
            total_frames=total_frames,
            video_prob=cut3r_params['video_prob'],
            fix_interval_prob=cut3r_params['fix_interval_prob'],
            max_interval=cut3r_params['max_interval'],
            min_interval=cut3r_params['min_interval'],
            allow_repeat=allow_repeat,
            block_shuffle_size=cut3r_params.get('block_shuffle_size')
        )
    
    else:  # unified sampling
        sampled_idx = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        if data_args and hasattr(data_args, 'sample_jitter') and data_args.sample_jitter > 0:
            max_jitter = int(total_frames * data_args.sample_jitter)
            for i in range(len(sampled_idx)):
                jitter = random.randint(-max_jitter, max_jitter)
                jittered = sampled_idx[i] + jitter
                max_val = sampled_idx[i+1] - 1 if i < len(sampled_idx) - 1 else total_frames - 1
                sampled_idx[i] = np.clip(jittered, 0, min(total_frames - 1, max_val))
                if i > 0 and sampled_idx[i] < sampled_idx[i-1]:
                    sampled_idx[i] = min(sampled_idx[i-1], total_frames - 1)
        
        return sampled_idx


# =============================================================================
# Frame Loading Helpers
# =============================================================================

def load_metadata(video_file: str, config: DatasetConfig) -> np.lib.npyio.NpzFile:
    """Load metadata file with fallback support."""
    metadata_path = osp.join(video_file, config.metadata_file)
    try:
        return np.load(metadata_path, allow_pickle=True)
    except FileNotFoundError:
        if config.fallback_metadata_file:
            fallback_path = osp.join(video_file, config.fallback_metadata_file)
            return np.load(fallback_path, allow_pickle=True)
        raise


def load_frames_from_disk(basenames, images_dir, extension, name_transform=None):
    """Load frames from disk given basenames."""
    video = []
    for basename in basenames:
        if name_transform:
            image_name = name_transform(basename)
        else:
            image_name = str(basename) + extension
        
        frame_path = os.path.join(images_dir, image_name)
        try:
            with Image.open(frame_path) as img:
                video.append(img.convert("RGB"))
        except IOError:
            continue
    return video


def load_cut3r_pairs_if_available(video_file, dataset_type, num_frames, cut3r_params):
    """
    Load cut3r metadata and select pairs if available.
    
    Only works for scannetpp and arkitscenes.
    
    Returns:
        (selected_basenames, metadata, mode) or (None, None, None) if not available
    """
    assert dataset_type in ['scannetpp', 'arkitscenes']
    
    cut3r_path = osp.join(video_file, "new_scene_metadata_cut3r_fixed.npz")
    if not osp.exists(cut3r_path):
        return None, None, None, None
    
    metadata = np.load(cut3r_path, allow_pickle=True)
    images_list = metadata['images']
    
    if 'image_collection' not in metadata:
        return None, metadata, None, None
    
    image_collection = metadata['image_collection'].item()
    
    valid_refs = [
        ref for ref, pairs in image_collection.items()
        if len(set(int(p[0]) for p in pairs)) >= num_frames - 1
    ]
    
    if not valid_refs:
        return None, metadata, None, None
    
    video_prob = cut3r_params.get('video_prob', 0.6)
    mode = 'video' if random.random() < video_prob else 'collection'
    
    ref_id = random.choice(valid_refs)
    pairs = image_collection[ref_id]
    
    if dataset_type == 'scannetpp':
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        selected_pairs = pairs_sorted[:num_frames-1]
    else:
        selected_pairs = random.sample(pairs, min(len(pairs), num_frames-1))
    
    selected_indices = sorted([ref_id] + [int(p[0]) for p in selected_pairs])[:num_frames]
    
    if mode == 'collection':
        random.shuffle(selected_indices)
        block_size = cut3r_params.get('block_shuffle_size')
        if block_size:
            selected_indices = blockwise_shuffle(selected_indices, block_size=block_size)
    
    selected_basenames = [images_list[idx] for idx in selected_indices]
    return selected_basenames, metadata, mode, selected_indices 


# =============================================================================
# Main Loading Functions
# =============================================================================

def load_scene_frames(
    video_file: str,
    dataset_type: str,
    sample_mode: str,
    num_frames: int,
    cut3r_params: Optional[Dict] = None,
    temporal_params: Optional[Dict] = None,
    data_args: Any = None,
    allow_repeat: bool = False,
) -> Tuple[Optional[List[Image.Image]], Optional[List[str]], Dict]:
    """
    Load frames from a scene directory (ScanNet, ScanNet++, ARKitScenes, CO3D).

    Args:
        video_file: Path to scene directory
        dataset_type: 'scannet', 'scannetpp', 'arkitscenes', or 'co3d'
        sample_mode: 'unified', 'temporal', 'cut3r', or 'random'
        num_frames: Number of frames to load
        cut3r_params: Parameters for cut3r sampling
        temporal_params: Parameters for temporal sampling
        data_args: Data arguments
        allow_repeat: Allow frame repetition

    Returns:
        (video_frames, basenames, interval_stats)
    """
    default_interval_stats = {"avg_interval": 0.0, "min_interval": 0, "max_interval": 0}
    
    # CO3D: dummy implementation
    if dataset_type == "co3d":
        dummy = Image.new('RGB', (384, 384), color='white')
        return [dummy] * num_frames, [f"frame_{i:06d}" for i in range(num_frames)], default_interval_stats
    
    config = DATASET_CONFIGS.get(dataset_type)
    if config is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    selected_basenames = None
    selected_indices = None  # Track indices for interval calculation
    
    # Try cut3r pairs first for supported datasets
    if sample_mode == 'cut3r' and config.supports_cut3r_pairs and cut3r_params:
        selected_basenames, _, _,selected_indices = load_cut3r_pairs_if_available(
            video_file, dataset_type, num_frames, cut3r_params
        )
        
        if selected_basenames is None:
            # Fallback to temporal
            metadata = load_metadata(video_file, config)
            images_list = metadata['images']
            selected_indices = select_frame_indices(
                images_list, 'temporal', num_frames,
                temporal_params=temporal_params,
                data_args=data_args,
                allow_repeat=allow_repeat
            )
            selected_basenames = [images_list[idx] for idx in selected_indices]
    
    # Standard selection
    if selected_basenames is None:
        metadata = load_metadata(video_file, config)
        images_list = metadata['images']
        selected_indices = select_frame_indices(
            images_list, sample_mode, num_frames,
            cut3r_params=cut3r_params,
            temporal_params=temporal_params,
            data_args=data_args,
            allow_repeat=allow_repeat
        )
        selected_basenames = [images_list[idx] for idx in selected_indices]
    
    # Compute interval statistics
    interval_stats = compute_frame_interval_stats(selected_indices)
    
    # Load frames from disk
    images_dir = os.path.join(video_file, config.images_subdir)
    video = load_frames_from_disk(
        selected_basenames, images_dir, 
        config.image_extension, config.name_transform
    )
    
    return video, selected_basenames, interval_stats



def build_cut3r_params(dataset_type: str, data_args: Any) -> Dict:
    """Build cut3r parameters from data_args for a dataset."""
    prefix = f'cut3r_{dataset_type}_'
    defaults = {
        'scannet': (30, 1, 0.6, 0.6),
        'scannetpp': (3, 1, 0.8, 0.5),
        'arkitscenes': (8, 1, 0.8, 0.5),
    }
    d = defaults.get(dataset_type, (30, 1, 0.6, 0.6))
    return {
        'max_interval': getattr(data_args, f'{prefix}max_interval', d[0]),
        'min_interval': getattr(data_args, f'{prefix}min_interval', d[1]),
        'video_prob': getattr(data_args, f'{prefix}video_prob', d[2]),
        'fix_interval_prob': getattr(data_args, f'{prefix}fix_interval_prob', d[3]),
        'block_shuffle_size': getattr(data_args, f'{prefix}block_shuffle_size', None),
    }


def build_temporal_params(dataset_type: str, data_args: Any) -> Dict:
    """Build temporal parameters from data_args for a dataset."""
    prefix = f'temporal_{dataset_type}_'
    defaults = {'scannet': (30, 1), 'scannetpp': (3, 1), 'arkitscenes': (8, 1)}
    d = defaults.get(dataset_type, (30, 1))
    return {
        'max_interval': getattr(data_args, f'{prefix}max_interval', d[0]),
        'min_interval': getattr(data_args, f'{prefix}min_interval', d[1]),
    }

def compute_frame_interval_stats(indices: list) -> dict:
    """
    Compute frame interval statistics from selected frame indices.
    
    Args:
        indices: List of frame indices (e.g., [0, 5, 10, 15, 20])
    
    Returns:
        Dict with avg_interval, min_interval, max_interval
    """
    if indices is None or len(indices) < 2:
        return {"avg_interval": 0.0, "min_interval": 0, "max_interval": 0}
    
    # Calculate intervals between consecutive frames
    intervals = []
    for i in range(1, len(indices)):
        interval = abs(indices[i] - indices[i-1])  # abs for shuffled cases
        if interval > 0:
            intervals.append(interval)
    
    if not intervals:
        return {"avg_interval": 0.0, "min_interval": 0, "max_interval": 0}
    
    return {
        "avg_interval": sum(intervals) / len(intervals),
        "min_interval": min(intervals),
        "max_interval": max(intervals),
    }