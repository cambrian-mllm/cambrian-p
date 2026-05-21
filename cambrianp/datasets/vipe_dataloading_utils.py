"""
ViPE Data Loading Utilities

Handles loading data from ViPE datasets:
- WSDG (Wild-SDG): 966K videos with RGB, Depth, Pose, Intrinsics
- W360 (Web360): 2,114 panoramic videos with default focal
- DPSP (Dynpose-100K): 99K videos with ViPE poses
- ViPE-Cambrians: ViPE pose/depth annotations on Cambrian-S videos

Actual ViPE-Cambrians structure (from scan):
    {results_root}/vipe/{source}/{uid}/
        ├── poses_pred.npy              # [N, 4, 4] cam2world poses
        ├── metrics.json
        ├── depth/
        │   └── {uid}.zip               # EXR frames in ZIP
        ├── intrinsics/
        │   ├── {uid}.npz
        │   └── {uid}_camera.txt
        ├── rgb/
        │   └── {uid}.mp4
        ├── pose/
        │   └── {uid}.npz
        └── mask/
            └── {uid}.zip
"""

import os
import os.path as osp
import json
import zipfile
import tempfile
import numpy as np
import cv2
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pickle


# =============================================================================
# Dataset Configurations
# =============================================================================

@dataclass 
class ViPEDatasetConfig:
    """Configuration for a ViPE dataset."""
    name: str
    image_size: Tuple[int, int]  # (H, W)
    default_focal: Optional[float]
    is_metric_scale: bool
    pose_convention: str  # 'w2c' or 'c2w'
    depth_scale: float  # Multiply raw depth by this
    covisibility_thres: float = 0.15  # For compatibility with MapAnything
    

VIPE_DATASET_CONFIGS = {
    "wsdg": ViPEDatasetConfig(
        name="wsdg",
        image_size=(720, 1280),
        default_focal=None,  # Has per-frame intrinsics
        is_metric_scale=False,  # ViPE depth is relative scale
        pose_convention="c2w",  # cam2world
        depth_scale=1.0,
    ),
    "w360": ViPEDatasetConfig(
        name="w360",
        image_size=(512, 1024),
        default_focal=512.0,  # Panoramic default
        is_metric_scale=False,
        pose_convention="c2w",
        depth_scale=1.0,
    ),
    "dpsp": ViPEDatasetConfig(
        name="dpsp",
        image_size=(720, 1280),
        default_focal=None,  # From dynpose PKL
        is_metric_scale=False,
        pose_convention="c2w",
        depth_scale=1.0,
    ),
    # ViPE-Cambrians uses c2w poses
    "vipe_cambrians": ViPEDatasetConfig(
        name="vipe_cambrians",
        image_size=(720, 1280),  # Will be overridden by actual video size
        default_focal=None,
        is_metric_scale=False,
        pose_convention="c2w",  # poses_pred.npy is cam2world
        depth_scale=1.0,
    ),
}


def get_vipe_config(dataset_name: str) -> ViPEDatasetConfig:
    """Get configuration for a ViPE dataset."""
    if dataset_name is None:
        dataset_name = "vipe_cambrians"
    
    key = dataset_name.lower().replace("-", "_").replace(" ", "_")
    
    # Check for cambrians prefix
    if "cambrians" in key or "cambrian" in key:
        return VIPE_DATASET_CONFIGS["vipe_cambrians"]
    
    if key in VIPE_DATASET_CONFIGS:
        return VIPE_DATASET_CONFIGS[key]
    
    # Default config
    return ViPEDatasetConfig(
        name=dataset_name,
        image_size=(720, 1280),
        default_focal=640.0,
        is_metric_scale=False,
        pose_convention="c2w",
        depth_scale=1.0,
    )


# =============================================================================
# Validation
# =============================================================================
def is_vipe_scene(scene_entry: Dict) -> bool:
    """
    Check if a scene entry is a raw ViPE scene (WSDG/DPSP).
    Note: W360 has been removed.
    """
    # Primary check: explicit data_source tag
    if scene_entry.get("data_source") == "vipe":
        return True
    # Fallback: source_dataset is one of ViPE's original datasets
    # (This catches augmented samples that don't have data_source but have source_dataset)
    if scene_entry.get("source_dataset") in ["wsdg", "dpsp"]:
        return True
    return False


def is_vipe_cambrians_scene(scene_entry: Dict) -> bool:
    """
    Check if a scene entry is a ViPE-annotated Cambrian-S scene.
    """
    # Primary check: explicit data_source tag
    if scene_entry.get("data_source") == "vipe_cambrians":
        return True
    
    source_dataset = scene_entry.get("source_dataset", "")
    if source_dataset.startswith("cambrians_"):
        return True
    
    # Fallback: check if it has ViPE-style fields AND is NOT original ViPE
    has_vipe_fields = all(k in scene_entry for k in ['rgb_path', 'pose_path', 'depth_path'])
    if has_vipe_fields:
        source = scene_entry.get("source_dataset", "")
        # If source_dataset is NOT original ViPE, assume it's ViPE-Cambrians
        if source and source not in ["wsdg", "dpsp"]:
            return True
    
    # Or check if video path contains cambrian-related paths
    video = scene_entry.get("video", "")
    if "cambrian" in video.lower() or "pose_results" in video.lower():
        return True
    
    return False


def validate_vipe_scene(scene_entry: Dict) -> Tuple[bool, str]:
    """
    Validate a ViPE scene entry has required fields and files exist.
    """
    required_fields = ["rgb_path", "depth_path", "pose_path"]
    
    for field in required_fields:
        if field not in scene_entry:
            return False, f"Missing field: {field}"
        if not osp.exists(scene_entry[field]):
            return False, f"File not found: {scene_entry[field]}"
    
    return True, ""


# =============================================================================
# Video Frame Loading
# =============================================================================

def load_video_frames(
    video_path: str,
    frame_indices: Optional[List[int]] = None,
    num_frames: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load frames from an MP4 video file.
    
    Args:
        video_path: Path to MP4 file
        frame_indices: Specific frame indices to load (if None, sample temporally)
        num_frames: Number of frames to load (used if frame_indices is None)
        
    Returns:
        frames: [N, H, W, 3] uint8 array
        total_frames: Total number of frames in video
    """
    import decord
    decord.bridge.set_bridge('native')
    
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    
    if frame_indices is None:
        if num_frames is None:
            num_frames = min(32, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    # Clamp indices to valid range
    frame_indices = [min(max(0, i), total_frames - 1) for i in frame_indices]
    
    frames = vr.get_batch(frame_indices).asnumpy()  # [N, H, W, 3]
    
    return frames, total_frames


# =============================================================================
# Depth Loading (from ZIP with EXR files)
# =============================================================================

def load_exr_from_bytes(exr_bytes: bytes) -> np.ndarray:
    """
    Load EXR depth from bytes using OpenEXR.
    
    ViPE uses HALF (float16) format.
    """
    import OpenEXR
    import Imath
    
    # Create temporary file (OpenEXR doesn't support memory buffers directly)
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp:
        tmp.write(exr_bytes)
        tmp_path = tmp.name
    
    try:
        exr_file = OpenEXR.InputFile(tmp_path)
        header = exr_file.header()
        
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Try common channel names
        for channel in ['Z', 'Y', 'R', 'depth', 'B']:
            if channel in header['channels']:
                # Try HALF first (ViPE uses this), then FLOAT
                try:
                    pt = Imath.PixelType(Imath.PixelType.HALF)
                    depth_str = exr_file.channel(channel, pt)
                    depth = np.frombuffer(depth_str, dtype=np.float16)
                except:
                    pt = Imath.PixelType(Imath.PixelType.FLOAT)
                    depth_str = exr_file.channel(channel, pt)
                    depth = np.frombuffer(depth_str, dtype=np.float32)
                
                depth = depth.reshape((height, width)).astype(np.float32)
                return depth
        
        # Fallback: use first available channel
        channels = list(header['channels'].keys())
        if channels:
            try:
                pt = Imath.PixelType(Imath.PixelType.HALF)
                depth_str = exr_file.channel(channels[0], pt)
                depth = np.frombuffer(depth_str, dtype=np.float16)
            except:
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                depth_str = exr_file.channel(channels[0], pt)
                depth = np.frombuffer(depth_str, dtype=np.float32)
            
            depth = depth.reshape((height, width)).astype(np.float32)
            return depth
            
    finally:
        os.unlink(tmp_path)
    
    raise ValueError("Could not load depth from EXR")


def load_depth_from_zip(
    zip_path: str,
    frame_indices: List[int],
) -> np.ndarray:
    """
    Load depth frames from a ZIP file containing EXR files.
    
    Args:
        zip_path: Path to ZIP file
        frame_indices: Frame indices to load
        
    Returns:
        depths: [N, H, W] float32 array
    """
    depths = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get sorted list of EXR files
        exr_files = sorted([f for f in zf.namelist() if f.endswith('.exr')])
        total_frames = len(exr_files)
        
        if total_frames == 0:
            raise ValueError(f"No EXR files found in {zip_path}")
        
        for idx in frame_indices:
            # Clamp to valid range
            idx = min(max(0, idx), total_frames - 1)
            exr_name = exr_files[idx]
            
            exr_bytes = zf.read(exr_name)
            depth = load_exr_from_bytes(exr_bytes)
            
            # Handle invalid values
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depths.append(depth)
    
    return np.stack(depths, axis=0)


def get_depth_frame_count_from_zip(zip_path: str) -> int:
    """Get the number of depth frames in a ZIP file."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        exr_files = [f for f in zf.namelist() if f.endswith('.exr')]
        return len(exr_files)


def get_depth_frame_count(depth_path: str, depth_format: str = "zip") -> int:
    """Get the number of depth frames from depth file/zip."""
    if depth_format == "zip":
        return get_depth_frame_count_from_zip(depth_path)
    else:
        # NPY format
        depths = np.load(depth_path)
        return depths.shape[0] if depths.ndim == 3 else 1


# =============================================================================
# Pose Loading
# =============================================================================

def load_vipe_poses(
    pose_path: str,
    frame_indices: List[int],
    pose_key: str = "data",
) -> np.ndarray:
    """
    Load poses from ViPE NPZ file.
    
    Returns:
        poses: [N, 4, 4] float32 array (world-to-camera)
    """
    data = np.load(pose_path, allow_pickle=True)
    poses = data[pose_key].astype(np.float32)  # [total_frames, 4, 4]
    
    total_frames = poses.shape[0]
    frame_indices = [min(max(0, i), total_frames - 1) for i in frame_indices]
    
    return poses[frame_indices]


def load_vipe_cambrians_poses(pose_path: str, frame_indices: List[int]) -> np.ndarray:
    data = np.load(pose_path, allow_pickle=True)
    
    if isinstance(data, np.lib.npyio.NpzFile):
        poses = None
        for key in ['data', 'poses', 'poses_pred', 'camera_poses']:
            if key in data:
                poses = data[key].astype(np.float32)
                break
        if poses is None:
            poses = data[list(data.keys())[0]].astype(np.float32)
        data.close()
    else:
        poses = data.astype(np.float32)
    
    # Handle (N, 3, 4) format
    if poses.ndim == 3 and poses.shape[1:] == (3, 4):
        p4 = np.zeros((len(poses), 4, 4), dtype=np.float32)
        p4[:, :3, :] = poses
        p4[:, 3, 3] = 1.0
        poses = p4
    
    total_frames = poses.shape[0]
    frame_indices = [min(max(0, i), total_frames - 1) for i in frame_indices]
    
    return poses[frame_indices]


def make_poses_canonical(poses: np.ndarray) -> np.ndarray:
    """
    Make poses canonical by setting first pose to identity.
    
    Args:
        poses: [N, 4, 4] cam2world poses
        
    Returns:
        canonical_poses: [N, 4, 4] with first pose = identity
    """
    first_pose_inv = np.linalg.inv(poses[0])
    return np.array([first_pose_inv @ p for p in poses], dtype=np.float32)


# =============================================================================
# Intrinsics Loading
# =============================================================================

def load_wsdg_intrinsics(
    intrinsics_path: str,
    frame_indices: List[int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Load WSDG intrinsics from NPZ file.
    
    WSDG stores intrinsics as (N, 4) array: [fx, fy, cx, cy]
    """
    data = np.load(intrinsics_path, allow_pickle=True)
    
    intri_key = None
    for key in data.files:
        arr = data[key]
        if arr.ndim == 2 and arr.shape[1] == 4:
            intri_key = key
            break
    
    if intri_key is None:
        raise ValueError(f"Could not find intrinsics in {intrinsics_path}")
    
    intri_array = data[intri_key].astype(np.float32)
    total_frames = intri_array.shape[0]
    
    frame_indices = [min(max(0, i), total_frames - 1) for i in frame_indices]
    
    intrinsics = []
    for idx in frame_indices:
        fx, fy, cx, cy = intri_array[idx]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        intrinsics.append(K)
    
    return np.stack(intrinsics, axis=0)


def load_dpsp_intrinsics(
    pkl_path: str,
    num_frames: int,
    image_size: Tuple[int, int] = None,
) -> np.ndarray:
    """
    Load DPSP intrinsics from Dynpose PKL file.
    
    Handles various PKL formats including numpy-saved files.
    """
    K = None
    
    try:
        # Try standard pickle first
        with open(pkl_path, 'rb') as f:
            cam_data = pickle.load(f)
        
        if isinstance(cam_data, dict):
            K = cam_data.get('intrinsics', cam_data.get('K', cam_data.get('camera_matrix', None)))
            # Also try focal length
            if K is None and 'focal' in cam_data:
                focal = cam_data['focal']
                if image_size:
                    H, W = image_size
                    cx, cy = W / 2, H / 2
                else:
                    cx, cy = 640, 360  # Default 720p center
                K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float32)
        elif isinstance(cam_data, np.ndarray):
            if cam_data.shape == (3, 3):
                K = cam_data
            elif cam_data.shape == (4,):
                # [fx, fy, cx, cy] format
                fx, fy, cx, cy = cam_data
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        else:
            K = cam_data
            
    except Exception as e:
        # Try numpy load as fallback (for .pkl files that are actually .npy)
        try:
            cam_data = np.load(pkl_path, allow_pickle=True)
            if isinstance(cam_data, np.ndarray):
                if cam_data.shape == (3, 3):
                    K = cam_data
                elif cam_data.ndim == 0:
                    # Scalar or dict wrapped in 0-d array
                    cam_data = cam_data.item()
                    if isinstance(cam_data, dict):
                        K = cam_data.get('intrinsics', cam_data.get('K', None))
        except Exception as e2:
            print(f"[ViPE] Warning: Failed to load DPSP intrinsics from {pkl_path}: {e}, {e2}")
    
    if K is None:
        # Return default intrinsics for 720p
        if image_size:
            H, W = image_size
        else:
            H, W = 720, 1280
        focal = max(H, W) * 1.2
        K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=np.float32)
        print(f"[ViPE] Warning: Using default intrinsics for {pkl_path}")
    
    K = np.array(K, dtype=np.float32)
    if K.shape != (3, 3):
        # Try to reshape or extract
        if K.size == 9:
            K = K.reshape(3, 3)
        elif K.size >= 4:
            # Assume [fx, fy, cx, cy]
            fx, fy, cx, cy = K.flat[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        else:
            raise ValueError(f"Invalid intrinsics shape: {K.shape}")
    
    return np.tile(K[None], (num_frames, 1, 1))


def load_vipe_cambrians_intrinsics(
    intrinsics_path: Optional[str], 
    image_size: Tuple[int, int], 
    num_frames: int
) -> np.ndarray:
    """
    Load intrinsics from ViPE Cambrian-S.
    
    Supports multiple TXT formats:
    - Single focal: "640.0"
    - With index prefix: "0: 640.0" or "0:640.0"  
    - Multiple lines with index: "0: 640.0\n1: 640.0\n..."
    - COLMAP PINHOLE format: "PINHOLE 1920 1080 fx fy cx cy" (first line header, values on second line or same line)
    
    Also supports NPZ files with intrinsics array.
    """
    H, W = image_size
    cx, cy = W / 2, H / 2
    
    # Default focal
    focal = max(H, W) * 1.2
    
    if intrinsics_path and osp.exists(intrinsics_path):
        try:
            if intrinsics_path.endswith('.txt'):
                with open(intrinsics_path, 'r') as f:
                    content = f.read().strip()
                    
                if content:
                    lines = content.split('\n')
                    first_line = lines[0].strip()
                    
                    # Handle COLMAP PINHOLE format: "PINHOLE width height fx fy cx cy"
                    # or multiline: first line "PINHOLE", second line has values
                    if first_line.upper().startswith('PINHOLE'):
                        parts = first_line.split()
                        if len(parts) >= 7:
                            # Single line: PINHOLE width height fx fy cx cy
                            try:
                                fx = float(parts[3])
                                fy = float(parts[4])
                                cx_val = float(parts[5])
                                cy_val = float(parts[6])
                                K = np.array([
                                    [fx, 0, cx_val],
                                    [0, fy, cy_val],
                                    [0, 0, 1]
                                ], dtype=np.float32)
                                return np.tile(K[None], (num_frames, 1, 1))
                            except (ValueError, IndexError):
                                pass
                        
                        # Try second line if first only says PINHOLE
                        if len(lines) > 1:
                            second_line = lines[1].strip()
                            parts = second_line.split()
                            if len(parts) >= 4:
                                try:
                                    # Format: fx fy cx cy
                                    fx = float(parts[0])
                                    fy = float(parts[1])
                                    cx_val = float(parts[2])
                                    cy_val = float(parts[3])
                                    K = np.array([
                                        [fx, 0, cx_val],
                                        [0, fy, cy_val],
                                        [0, 0, 1]
                                    ], dtype=np.float32)
                                    return np.tile(K[None], (num_frames, 1, 1))
                                except (ValueError, IndexError):
                                    pass
                            elif len(parts) >= 1:
                                # Just focal length
                                try:
                                    focal = float(parts[0])
                                except ValueError:
                                    pass
                    
                    # Handle "0: 640.0" or "0:640.0" format
                    elif ':' in first_line:
                        parts = first_line.split(':')
                        if len(parts) >= 2:
                            value_str = parts[1].strip()
                            value_parts = value_str.split()
                            if value_parts:
                                try:
                                    focal = float(value_parts[0])
                                except ValueError:
                                    pass
                    else:
                        # Simple format: just the focal length
                        parts = first_line.split()
                        if parts:
                            try:
                                focal = float(parts[0])
                            except ValueError:
                                pass
                            
            elif intrinsics_path.endswith('.npz'):
                data = np.load(intrinsics_path, allow_pickle=True)
                # Try common keys
                for key in ['intrinsics', 'K', 'focal', 'data']:
                    if key in data.files:
                        arr = data[key]
                        if arr.ndim == 0:
                            focal = float(arr)
                        elif arr.ndim == 1 and len(arr) >= 1:
                            focal = float(arr[0])
                        elif arr.ndim == 2 and arr.shape == (3, 3):
                            # Full intrinsic matrix
                            K = arr.astype(np.float32)
                            return np.tile(K[None], (num_frames, 1, 1))
                        elif arr.ndim == 2 and arr.shape[1] == 4:
                            fx, fy, cx_val, cy_val = abs(arr[0, 0]), abs(arr[0, 1]), arr[0, 2], arr[0, 3]
                            # Clamp insane focal lengths to prevent Inf in depth loss
                            max_dim = max(H, W)
                            max_focal = max_dim * 3.0  # FOV ~19° is already very telephoto
                            if fx > max_focal or fy > max_focal:
                                fx = min(fx, max_focal)
                                fy = min(fy, max_focal)
                            K = np.array([
                                [fx, 0, cx_val],
                                [0, fy, cy_val],
                                [0, 0, 1]
                            ], dtype=np.float32)
                            return np.tile(K[None], (num_frames, 1, 1))
                        break
        except Exception as e:
            print(f"[ViPE] Warning: Failed to load intrinsics from {intrinsics_path}: {e}")
    
    K = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return np.tile(K[None], (num_frames, 1, 1))


def build_default_intrinsics(
    focal_length: float,
    image_size: Tuple[int, int],
    num_frames: int,
) -> np.ndarray:
    """Build default intrinsics with given focal length."""
    H, W = image_size
    cx, cy = W / 2, H / 2
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return np.tile(K[None], (num_frames, 1, 1))


# =============================================================================
# Frame Sampling
# =============================================================================

def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    mode: str = "uniform",
    min_interval: int = 1,
    max_interval: int = 1,
) -> List[int]:
    """
    Sample frame indices from a video.
    
    For temporal mode: sample a contiguous window with random interval.
    If video is too short for the requested interval, shrink until it fits.
    Worst case falls back to unified sampling.
    """
    import random
    
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        return indices
    
    if mode == "uniform":
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    elif mode == "random":
        return sorted(random.sample(range(total_frames), num_frames))
    
    elif mode in ("temporal", "cut3r"):
        # Max feasible interval given video length and num_frames
        # Pick start first
        max_start = total_frames - num_frames  # worst case interval=1
        start = random.randint(0, max(0, max_start))
        
        # Now compute max feasible interval given start
        remaining = total_frames - 1 - start
        max_feasible = remaining // (num_frames - 1)
        actual_max = min(max_interval, max_feasible)
        actual_min = min(min_interval, actual_max)
        
        if actual_max < 1:
            return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        interval = random.randint(actual_min, actual_max)
        return [start + i * interval for i in range(num_frames)]
    
    else:
        raise ValueError(f"[ViPE] Unknown sample mode {mode}")


# =============================================================================
# 3D Point Computation (same as MapAnything)
# =============================================================================

def compute_cam_points(
    depths: np.ndarray,  # [N, H, W]
    intrinsics: np.ndarray,  # [N, 3, 3]
) -> torch.Tensor:
    """
    Compute 3D points in camera coordinates from depth maps.
    """
    N, H, W = depths.shape
    
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    
    cam_points = np.zeros((N, H, W, 3), dtype=np.float32)
    
    for i in range(N):
        depth = depths[i]
        K = intrinsics[i]
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x = (u - cx) * depth / (fx + 1e-8)
        y = (v - cy) * depth / (fy + 1e-8)
        z = depth
        
        cam_points[i] = np.stack([x, y, z], axis=-1)
    
    return torch.from_numpy(cam_points)


def compute_world_points(
    cam_points: torch.Tensor,  # [N, H, W, 3]
    cam2world: torch.Tensor,   # [N, 4, 4]
) -> torch.Tensor:
    """
    Transform camera-space points to world coordinates.
    """
    N, H, W, _ = cam_points.shape
    
    pts_flat = cam_points.view(N, -1, 3)
    
    R = cam2world[:, :3, :3]
    t = cam2world[:, :3, 3:4]
    
    world_flat = torch.bmm(pts_flat, R.transpose(1, 2)) + t.transpose(1, 2)
    
    return world_flat.view(N, H, W, 3)


# =============================================================================
# ViPE Cambrian-S Scene Loading
# =============================================================================

def load_vipe_cambrians_depths(
    depth_path: str, 
    frame_indices: List[int],
    depth_format: str = "zip"
) -> np.ndarray:
    """
    Load depths from ViPE Cambrian-S file (NPY or ZIP).
    
    Args:
        depth_path: Path to depth/{uid}.zip or depths_pred.npy
        frame_indices: Frame indices to load
        depth_format: 'npy' or 'zip'
    
    Returns:
        depths: [N, H, W] float32
    """
    if depth_format == "zip":
        return load_depth_from_zip(depth_path, frame_indices)
    else:
        # NPY format
        depths = np.load(depth_path).astype(np.float32)
        total_frames = depths.shape[0]
        frame_indices = [min(max(0, i), total_frames - 1) for i in frame_indices]
        return depths[frame_indices]


def _resolve_vipe_cambrians_paths(scene_entry: Dict) -> Dict[str, Any]:
    """
    Resolve paths for ViPE-Cambrians scene, handling augmented samples.
    
    Supports two structures:
    
    1. Sample/old format:
        {results_root}/vipe/{source}/{uid}/
            ├── poses_pred.npy
            ├── depths_pred.npy
            ├── rgb/{uid}.mp4
            └── intrinsics/{uid}_camera.txt
    
    2. Stage3 format (actual ViPE output):
        {results_root}/results/result_X/{source}/{uid}/
            ├── pose/{uid}.npz
            ├── depth/{uid}.zip
            ├── rgb/{uid}.mp4
            └── intrinsics/{uid}.npz
    
    Returns dict with resolved: rgb_path, pose_path, depth_path, intrinsics_path, depth_format, uid
    """
    rgb_path = scene_entry.get("rgb_path")
    pose_path = scene_entry.get("pose_path")
    depth_path = scene_entry.get("depth_path")
    intrinsics_path = scene_entry.get("intrinsics_path")
    depth_format = scene_entry.get("depth_format", "zip")
    
    if rgb_path is None or pose_path is None or depth_path is None:
        video_path = scene_entry.get("video", "")
        if not video_path:
            raise ValueError(f"Cannot determine paths: no 'video' field in scene_entry")
        
        if "/rgb/" in video_path:
            rgb_dir = osp.dirname(video_path)
            base_dir = osp.dirname(rgb_dir)
        else:
            base_dir = osp.dirname(video_path)
            
        uid = osp.splitext(osp.basename(video_path))[0]
        
        if rgb_path is None:
            rgb_path = video_path
        
        if pose_path is None:
            # Try multiple formats in priority order
            candidates = [
                osp.join(base_dir, "pose", f"{uid}.npz"),      # stage3 format
                osp.join(base_dir, "poses_pred.npy"),            # old format
                osp.join(base_dir, "pose", f"{uid}_pose.npz"),  # alternative
            ]
            pose_path = next((c for c in candidates if osp.exists(c)), candidates[0])
        
        if depth_path is None:
            candidates = [
                (osp.join(base_dir, "depth", f"{uid}.zip"), "zip"),    # stage3 format
                (osp.join(base_dir, "depths_pred.npy"), "npy"),         # old format
            ]
            for cand_path, cand_fmt in candidates:
                if osp.exists(cand_path):
                    depth_path = cand_path
                    depth_format = cand_fmt
                    break
            if depth_path is None:
                depth_path = candidates[0][0]
                depth_format = candidates[0][1]
        
        if intrinsics_path is None:
            candidates = [
                osp.join(base_dir, "intrinsics", f"{uid}.npz"),          # npz (has actual values)
                osp.join(base_dir, "intrinsics", f"{uid}_camera.txt"),   # txt (may only say PINHOLE)
            ]
            intrinsics_path = next((c for c in candidates if osp.exists(c)), None)
    else:
        uid = osp.splitext(osp.basename(rgb_path or pose_path or "unknown"))[0]
        
        if depth_path and depth_format is None:
            depth_format = "zip" if depth_path.endswith(".zip") else "npy"
    
    return {
        "rgb_path": rgb_path,
        "pose_path": pose_path,
        "depth_path": depth_path,
        "intrinsics_path": intrinsics_path,
        "depth_format": depth_format,
        "uid": uid,
    }

def load_vipe_cambrians_scene(
    scene_entry: Dict,
    num_frames: int = 32,
    target_size: Tuple[int, int] = (192, 192),
    target_size_llava: Tuple[int, int] = (384, 384),
    sample_mode: str = "uniform",
    make_canonical: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
    vipe_cambrians_data_root: str = None,
    vipe_cambrians_results_root: str = None,
    min_interval: int = 1,
    max_interval: int = 100,
) -> Tuple[List[np.ndarray], List[str], Dict]:
    """
    Load a ViPE-annotated Cambrian-S scene for training.
    
    Handles actual ViPE output structure:
        {results_root}/vipe/{source}/{uid}/
            ├── poses_pred.npy              # cam2world poses
            ├── depth/{uid}.zip             # Depth in ZIP (EXR files)
            ├── rgb/{uid}.mp4               # Video
            └── intrinsics/{uid}_camera.txt # Intrinsics
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    # Resolve paths (handles augmented samples with only 'video' field)
    resolved = _resolve_vipe_cambrians_paths(scene_entry)
    rgb_path = resolved["rgb_path"]
    pose_path = resolved["pose_path"]
    depth_path = resolved["depth_path"]
    intrinsics_path = resolved["intrinsics_path"]
    depth_format = resolved["depth_format"]
    uid = resolved["uid"]
    
    # Resolve relative paths if roots provided
    if vipe_cambrians_results_root:
        if pose_path and not osp.isabs(pose_path):
            pose_path = osp.join(vipe_cambrians_results_root, pose_path)
        if depth_path and not osp.isabs(depth_path):
            depth_path = osp.join(vipe_cambrians_results_root, depth_path)
        if intrinsics_path and not osp.isabs(intrinsics_path):
            intrinsics_path = osp.join(vipe_cambrians_results_root, intrinsics_path)
    
    rgb_relative_to = scene_entry.get("rgb_relative_to", "results")
    if rgb_path and not osp.isabs(rgb_path):
        if rgb_relative_to == "results" and vipe_cambrians_results_root:
            rgb_path = osp.join(vipe_cambrians_results_root, rgb_path)
        elif rgb_relative_to == "data" and vipe_cambrians_data_root:
            rgb_path = osp.join(vipe_cambrians_data_root, rgb_path)
        elif vipe_cambrians_results_root:
            rgb_path = osp.join(vipe_cambrians_results_root, rgb_path)
    
    source_dataset = scene_entry.get("source_dataset", "vipe_cambrians")
    config = get_vipe_config(source_dataset)
    
    if verbose:
        print(f"[ViPE-Cambrians] Loading {source_dataset}/{uid}")
        print(f"  RGB: {rgb_path}")
        print(f"  Pose: {pose_path}")
        print(f"  Depth: {depth_path} (format={depth_format})")
    
    # Verify files exist. Pose is the only hard requirement; RGB is required to read
    # frames. Depth is optional — the public Cambrian-P-Data release ships pose only,
    # in which case we fabricate zero depths + zero point_masks and the rec depth loss
    # short-circuits via the `gt_depth_mask.sum() < 100` gate in vggt loss.py.
    if not osp.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    if not osp.exists(rgb_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    has_depth = bool(depth_path) and osp.exists(depth_path)

    # Get frame count from depth when available (most reliable for ViPE); otherwise
    # from RGB via decord (cheap — does not decode frames).
    if has_depth:
        total_frames = get_depth_frame_count(depth_path, depth_format)
    else:
        import decord
        total_frames = len(decord.VideoReader(rgb_path))

    # Sample frame indices
    frame_indices = sample_frame_indices(total_frames, num_frames, sample_mode, min_interval, max_interval)

    if verbose:
        depth_tag = f"depth={depth_path}" if has_depth else "depth=NONE (pose-only)"
        print(f"[ViPE-Cambrians] Sampled {len(frame_indices)} frames from {total_frames}  ({depth_tag})")

    # Load RGB frames
    rgb_frames, rgb_total = load_video_frames(rgb_path, frame_indices)

    # Handle frame count mismatch (RGB may have different count than depth)
    if rgb_total != total_frames:
        # Resample RGB indices to match depth
        rgb_indices = [int(i * (rgb_total - 1) / (total_frames - 1)) for i in frame_indices]
        rgb_frames, _ = load_video_frames(rgb_path, rgb_indices)
        if verbose:
            print(f"[ViPE-Cambrians] Resampled RGB: {rgb_total} -> {total_frames}")

    # Get original size from first RGB frame
    original_size = (rgb_frames.shape[2], rgb_frames.shape[1])  # (W, H)

    # Load depths (or fabricate zeros when pose-only).
    if has_depth:
        depths = load_vipe_cambrians_depths(depth_path, frame_indices, depth_format)
        depths = np.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)
        depths = np.clip(depths, 0.0, 1000.0)  # Cap at 1km
    else:
        # Zero depths at RGB resolution; downstream resize to target_size + the
        # `valid_mask = depths_192 > 0` check naturally produces an all-False
        # point_masks tensor that gates depth loss off in vggt/loss/loss.py.
        depths = np.zeros((len(frame_indices), rgb_frames.shape[1], rgb_frames.shape[2]), dtype=np.float32)

    # Load poses (cam2world format)
    poses_c2w = load_vipe_cambrians_poses(pose_path, frame_indices)
    if poses_c2w is None or len(poses_c2w) == 0:
        raise ValueError(f"Failed to load poses from {pose_path}")

    # Make canonical if requested
    if make_canonical:
        poses_c2w = make_poses_canonical(poses_c2w)

    # Compute world2cam for extrinsics
    poses_w2c = np.linalg.inv(poses_c2w)

    # Load intrinsics — load_vipe_cambrians_intrinsics already falls back to a default
    # focal (max(H,W)*1.2) when intrinsics_path is missing/None, so pose-only works here.
    depth_size = (depths.shape[1], depths.shape[2])  # (H, W)
    intrinsics = load_vipe_cambrians_intrinsics(intrinsics_path, depth_size, len(frame_indices))
    
    # Process frames
    N = len(frame_indices)
    images_192 = []
    images_384 = []
    depths_192 = []
    video_frames_out = []
    
    for i in range(N):
        img = rgb_frames[i]  # [H, W, 3] uint8
        
        # Resize for VLM
        img_384 = cv2.resize(img, (target_size_llava[1], target_size_llava[0]))
        images_384.append(np.transpose(img_384.astype(np.float32) / 255.0, (2, 0, 1)))
        video_frames_out.append(img_384)
        
        # Resize for reconstruction
        img_192 = cv2.resize(img, (target_size[1], target_size[0]))
        images_192.append(np.transpose(img_192.astype(np.float32) / 255.0, (2, 0, 1)))
        
        # Depth - resize from depth_size to target_size
        depth = depths[i]
        depth_192 = cv2.resize(depth, (target_size[1], target_size[0]), 
                               interpolation=cv2.INTER_NEAREST)
        depths_192.append(depth_192)
    
    # Scale intrinsics for target_size (192x192)
    # Original intrinsics are for depth_size
    scale_x = target_size[1] / depth_size[1]
    scale_y = target_size[0] / depth_size[0]
    
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[:, 0, 0] *= scale_x
    scaled_intrinsics[:, 1, 1] *= scale_y
    scaled_intrinsics[:, 0, 2] *= scale_x
    scaled_intrinsics[:, 1, 2] *= scale_y
    
    # Stack arrays
    images_192 = np.stack(images_192, axis=0)
    images_384 = np.stack(images_384, axis=0)
    depths_192 = np.stack(depths_192, axis=0)
    
    # Compute 3D points
    c2w_tensor = torch.from_numpy(poses_c2w.astype(np.float32))
    cam_points = compute_cam_points(depths_192, scaled_intrinsics)
    world_points = compute_world_points(cam_points, c2w_tensor)
    
    # Valid mask
    valid_mask = depths_192 > 0
    
    # Extrinsics (world2cam, 3x4)
    extrinsics = poses_w2c[:, :3, :]
    
    # Generate basenames
    basenames = [f"frame_{frame_indices[i]:06d}" for i in range(N)]
    
    # Build rec_views (matching MapAnything structure exactly)
    rec_views = {
        # Core image/depth data
        "images": torch.from_numpy(images_192),  # [N, 3, H, W]
        "images_for_llava": torch.from_numpy(images_384),  # [N, 3, H, W]
        "depths": torch.from_numpy(depths_192),  # [N, H, W]
        
        # Camera parameters
        "extrinsics": torch.from_numpy(extrinsics.astype(np.float32)),  # [N, 3, 4]
        "intrinsics": torch.from_numpy(scaled_intrinsics),  # [N, 3, 3]
        
        # 3D points
        "cam_points": cam_points,  # [N, H, W, 3]
        "world_points": world_points,  # [N, H, W, 3]
        
        # Masks
        "point_masks": torch.from_numpy(valid_mask),  # [N, H, W] bool
        
        # Normalization control
        "scale_by_points": torch.tensor(True),  # Non-metric
        "is_metric_scale": torch.tensor(False, dtype=torch.bool),
        
        # Size info
        "original_size": np.array([list(original_size)] * N, dtype=np.int32),
        "ids": np.arange(N, dtype=np.int64),
        "image_paths": [rgb_path] * N,
        "is_video": True,
        "num_views": N,
        
        # Metadata
        "basenames": basenames,
        "seq_name": [f"{source_dataset}_{uid}"],
        "sample_mode": sample_mode,
    }
    
    return video_frames_out, basenames, rec_views


# =============================================================================
# Training Integration
# =============================================================================

def load_vlm_filter(vlm_filter_path: str) -> Dict[str, set]:
    """
    Load VLM quality flags JSON: {rel_path: [discard_reasons], ...}.

    Returns dict mapping video rel_path -> set of discard reasons.
    Only flagged (bad) videos are stored; absence means "keep".
    """
    with open(vlm_filter_path, 'r') as f:
        data = json.load(f)

    index = {}
    for rel_path, reasons in data.items():
        reason_set = set(reasons)
        index[rel_path] = reason_set
        
        # Also index by basename to handle nested paths
        basename = osp.basename(rel_path)
        if basename not in index:
            index[basename] = reason_set
            
    return index


def should_keep_scene(scene: Dict, discard_reasons: Dict[str, set], categories_to_exclude: set) -> bool:
    """Return True if scene should be kept, False if it should be discarded."""
    rgb_path = scene.get("rgb_path", scene.get("video", ""))
    source = scene.get("source", scene.get("source_dataset", ""))
    basename = osp.basename(rgb_path) if rgb_path else ""

    # check if any reason to discard this scene
    # (basename fallback handles nested filter paths like "guiworld/android/video.mp4")
    reasons = (discard_reasons.get(f"{source}/{basename}")
               or discard_reasons.get(basename))

    if reasons is None:
        return True  # not flagged → keep
    return not (reasons & categories_to_exclude)


def load_vipe_cambrians_scenes_for_training(
    vipe_cambrians_path: str,
    thorough_validation: bool = False,
    vipe_cambrians_data_root: str = None,
    vipe_cambrians_results_root: str = None,
    vlm_filter_path: str = None,
    categories_to_exclude: List[str] = None,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    """
    Load ViPE Cambrian-S scenes JSON and format for training.

    Args:
        vipe_cambrians_path: Path to JSON file from generate_vipe_cambrians_json.py
        thorough_validation: Whether to validate files
        vipe_cambrians_data_root: Root for RGB videos (if rgb_relative_to='data')
        vipe_cambrians_results_root: Root for pose/depth results
        vlm_filter_path: Path to VLM quality flags JSON ({rel_path: [reasons]}).
        categories_to_exclude: Categories to discard, e.g.
            ['text_overlay', 'synthetic', 'screen_recording'].
            None or empty = no quality filtering.

    Returns:
        (formatted_entries, skipped_count, skipped_reasons)
    """
    # Load VLM quality flags if provided
    discard_reasons = None
    exclude_set = None
    if vlm_filter_path and categories_to_exclude:
        print(f"[VLM Filter] Loading {vlm_filter_path}...")
        print(f"[VLM Filter] Excluding categories: {categories_to_exclude}")
        discard_reasons = load_vlm_filter(vlm_filter_path)
        exclude_set = set(categories_to_exclude)
        print(f"[VLM Filter] Indexed {len(discard_reasons)} entries")

    with open(vipe_cambrians_path, 'r') as f:
        raw_scenes = json.load(f)

    formatted_entries = []
    skipped_count = 0
    skipped_reasons = {}

    vlm_discarded = 0
    for scene in raw_scenes:
        if discard_reasons is not None:
            if not should_keep_scene(scene, discard_reasons, exclude_set):
                vlm_discarded += 1
                skipped_count += 1
                skipped_reasons["vlm_filtered"] = skipped_reasons.get("vlm_filtered", 0) + 1
                continue
        source_dataset = scene.get("source_dataset", "vipe_cambrians")
        rgb_relative_to = scene.get("rgb_relative_to", "results")
        
        # Get paths
        rgb_path = scene.get("rgb_path", scene.get("video"))
        pose_path = scene.get("pose_path")
        depth_path = scene.get("depth_path")
        intrinsics_path = scene.get("intrinsics_path")
        
        # Auto-detect depth format (default to ZIP for ViPE-Cambrians)
        depth_format = scene.get("depth_format")
        if depth_format is None:
            if depth_path and depth_path.endswith(".zip"):
                depth_format = "zip"
            elif depth_path and depth_path.endswith(".npy"):
                depth_format = "npy"
            else:
                depth_format = "zip"  # Default!
        
        # Resolve paths to absolute
        resolved_rgb = rgb_path
        resolved_pose = pose_path
        resolved_depth = depth_path
        
        if vipe_cambrians_results_root:
            if resolved_rgb and not osp.isabs(resolved_rgb):
                if rgb_relative_to == "data" and vipe_cambrians_data_root:
                    resolved_rgb = osp.join(vipe_cambrians_data_root, resolved_rgb)
                else:
                    resolved_rgb = osp.join(vipe_cambrians_results_root, resolved_rgb)
            if resolved_pose and not osp.isabs(resolved_pose):
                resolved_pose = osp.join(vipe_cambrians_results_root, resolved_pose)
            if resolved_depth and not osp.isabs(resolved_depth):
                resolved_depth = osp.join(vipe_cambrians_results_root, resolved_depth)
        
        # Validation
        if thorough_validation:
            if not resolved_rgb or not osp.exists(resolved_rgb):
                skipped_count += 1
                skipped_reasons["no_rgb"] = skipped_reasons.get("no_rgb", 0) + 1
                continue
            if not resolved_pose or not osp.exists(resolved_pose):
                skipped_count += 1
                skipped_reasons["no_pose"] = skipped_reasons.get("no_pose", 0) + 1
                continue
            if not resolved_depth or not osp.exists(resolved_depth):
                skipped_count += 1
                skipped_reasons["no_depth"] = skipped_reasons.get("no_depth", 0) + 1
                continue
        
        entry = {
            "video": resolved_rgb,  # Absolute path for video loading
            "loading_type": "rec",
            "data_source": "vipe_cambrians",
            
            "source_dataset": source_dataset,
            "is_metric_scale": False,
            
            # Store original relative paths (for path resolution in loader)
            "rgb_path": scene.get("rgb_path"),
            "pose_path": scene.get("pose_path"),
            "depth_path": scene.get("depth_path"),
            "intrinsics_path": scene.get("intrinsics_path"),
            "depth_format": depth_format,
            "rgb_relative_to": rgb_relative_to,
            
            # Metadata
            "image_size": scene.get("image_size"),
            "focal_length": scene.get("focal_length"),
            
            "conversations": scene.get("conversations", [
                {"from": "human", "value": "<video>\nReconstruct the 3D scene."},
                {"from": "gpt", "value": ""}
            ]),
            "max_velocity": scene.get("max_velocity", None),
            "max_rotation_rate": scene.get("max_rotation_rate", None),
            "max_delta_velocity": scene.get("max_delta_velocity", None),
            "max_delta_rotation": scene.get("max_delta_rotation", None),
            "velocity_std": scene.get("velocity_std", None),
            "rotation_std": scene.get("rotation_std", None),
        }
        formatted_entries.append(entry)

    if discard_reasons is not None:
        print(f"[VLM Filter] exclude={categories_to_exclude}: {len(raw_scenes)} total → "
              f"{vlm_discarded} discarded → {len(formatted_entries)} kept")

    return formatted_entries, skipped_count, skipped_reasons


# Rec metadata fields used for attaching ViPE rec data to VQA entries
# and for creating interleaved dummy samples.
REC_META_FIELDS = [
    "data_source", "source_dataset", "loading_type", "is_metric_scale",
    "rgb_path", "pose_path", "depth_path", "depth_format",
    "intrinsics_path", "rgb_relative_to", "image_size", "focal_length",
]


def attach_vipe_rec_to_vqa_entries(vipe_cambrians_entries, list_data_dict):
    """
    "attach" mode — add rec metadata to matching VQA entries in-place.

    Steps:
        1. Build a lookup from video basename -> rec metadata fields
           (data_source, loading_type, rgb_path, pose_path, etc.)
           extracted from vipe_cambrians_entries.
        2. Walk through list_data_dict; for each VQA entry whose video
           basename matches a ViPE scene, attach rec fields via
           sample.update(). This makes the entry loadable by
           load_vipe_cambrians_scene() in _get_item.

    No entries are added or removed. Unmatched VQA entries stay as
    pure VQA (no rec supervision). This preserves the original jsonl
    VQA data composition exactly.

    Args:
        vipe_cambrians_entries: Filtered ViPE-Cambrians entries
            (from load_vipe_cambrians_scenes_for_training + filtering).
        list_data_dict: Main dataset list (modified in-place).

    Returns:
        attached_count: Number of VQA entries that received rec fields.
        attached_scenes: Set of video basenames that were matched.
        total_vipe_scenes: Total number of ViPE scenes in the lookup.
    """
    # build lookup from video basename -> rec metadata. This saves lookup time when matching VQA entries to ViPE scenes.
    vipe_rec_lookup = {}
    for e in vipe_cambrians_entries:
        vid_name = osp.splitext(osp.basename(e.get("video", "")))[0]
        vipe_rec_lookup[vid_name] = {k: e[k] for k in REC_META_FIELDS if k in e}

    # walk through VQA entries and attach rec fields to matches
    attached_count = 0
    attached_scenes = set()
    for sample in list_data_dict:
        vid_name = osp.splitext(osp.basename(sample.get("video", "")))[0]
        if vid_name in vipe_rec_lookup:
            # After update, sample gains data_source='vipe_cambrians',
            # loading_type='rec', rgb_path, pose_path, etc.
            sample.update(vipe_rec_lookup[vid_name])
            attached_count += 1
            attached_scenes.add(vid_name)

    return attached_count, attached_scenes, len(vipe_rec_lookup)


def replace_vipe_vqa_entries(vipe_cambrians_entries, list_data_dict):
    """
    "replace" mode: Append ViPE entries and remove overlapping VQA entries.

    ViPE-Cambrians entries are appended to list_data_dict. Then, any jsonl
    VQA entries whose video basename matches a ViPE scene are removed (deduped)
    so the model doesn't see the same video twice. The ViPE version is kept
    because it has reconstruction supervision.

    Note: This changes VQA data composition — the number of VQA entries per
    video may differ between the jsonl and the ViPE json.

    Args:
        vipe_cambrians_entries: Filtered ViPE-Cambrians entries.
        list_data_dict: The main dataset list (modified in-place, entries
            may be removed). Returns the new list since filtering creates
            a new list object.

    Returns:
        deduped_count: Number of jsonl VQA entries removed.
        vipe_scene_basenames: Set of video basenames from ViPE entries.
        deduped_list: New list with ViPE entries added and overlapping
            jsonl VQA entries removed.
    """
    # collect video basenames from ViPE-Cambrians for dedup matching
    vipe_scene_basenames = set()
    for entry in vipe_cambrians_entries:
        basename = osp.splitext(osp.basename(entry.get("video", "")))[0]
        vipe_scene_basenames.add(basename)

    # extend the dataset with ViPE entries
    list_data_dict.extend(vipe_cambrians_entries)

    # remove jsonl VQA entries whose video overlaps with a ViPE scene
    # Keep: ViPE entries (data_source == "vipe_cambrians") + non-overlapping jsonl entries
    # Note: the number of VQA entries per video may differ between the vqa jsonl and the ViPE json due to subsample
    if vipe_scene_basenames:
        total_num_before_dedup = len(list_data_dict)
        deduped_list = [
            sample for sample in list_data_dict
            if sample.get("data_source") == "vipe_cambrians"
            or osp.splitext(osp.basename(sample.get("video", "")))[0] not in vipe_scene_basenames
        ]
        deduped_count = total_num_before_dedup - len(deduped_list)
    else:
        deduped_list = list_data_dict
        deduped_count = 0

    return deduped_count, vipe_scene_basenames, deduped_list


# =============================================================================
# Original ViPE Scene Loading (WSDG/W360/DPSP)
# =============================================================================

def _resolve_vipe_paths(scene_entry: Dict, vipe_data_root: str = None, debug: bool = False) -> Dict[str, Any]:
    """
    Resolve paths for original ViPE scene (WSDG/DPSP), handling augmented samples.
    
    WSDG actual structure (confirmed from scan):
        {vipe_root}/wsdg/payload/wsdg-{shard}/
            ├── rgb/
            │   └── {uid}.mp4
            ├── depth/
            │   └── {uid}.zip
            ├── pose/
            │   └── {uid}.npz or {uid}_pose.npz
            └── intrinsics/
                └── {uid}_intri.npz or {uid}_camera.txt
    
    DPSP structure:
        {vipe_root}/dpsp/rgb_720p/{uid}.mp4           # RGB is separate!
        {vipe_root}/dpsp/payload/dpsp-{shard}/
            ├── depth/
            │   └── {uid}.zip
            ├── pose/
            │   └── {uid}.npz
            └── intrinsics/
                └── {uid}_intri.npz
        OR flat structure:
            ├── {uid}.zip                             # depth directly in shard
            ├── {uid}_pose.npz                        # pose directly in shard
            └── {uid}_intri.npz
    
    Returns dict with resolved: rgb_path, pose_path, depth_path, intrinsics_path
    """
    rgb_path = scene_entry.get("rgb_path")
    pose_path = scene_entry.get("pose_path")
    depth_path = scene_entry.get("depth_path")
    intrinsics_path = scene_entry.get("intrinsics_path")
    
    # If all paths are present, just resolve them
    if rgb_path and pose_path and depth_path:
        if vipe_data_root:
            if not osp.isabs(rgb_path):
                rgb_path = osp.join(vipe_data_root, rgb_path)
            if not osp.isabs(depth_path):
                depth_path = osp.join(vipe_data_root, depth_path)
            if not osp.isabs(pose_path):
                pose_path = osp.join(vipe_data_root, pose_path)
            if intrinsics_path and not osp.isabs(intrinsics_path):
                intrinsics_path = osp.join(vipe_data_root, intrinsics_path)
        
        uid = osp.splitext(osp.basename(rgb_path))[0]
        return {
            "rgb_path": rgb_path,
            "pose_path": pose_path,
            "depth_path": depth_path,
            "intrinsics_path": intrinsics_path,
            "uid": uid,
        }
    
    # Derive paths from video field
    video_path = scene_entry.get("video", "")
    if not video_path:
        raise ValueError(f"Cannot determine paths: no 'video' field in scene_entry")
    
    uid = osp.splitext(osp.basename(video_path))[0]
    source_dataset = scene_entry.get("source_dataset", "vipe").lower()
    
    if rgb_path is None:
        rgb_path = video_path
    
    if debug:
        print(f"[DEBUG] Resolving paths for uid={uid}, source={source_dataset}")
        print(f"[DEBUG] video_path={video_path}")
    
    # =========================================
    # DPSP: RGB is in rgb_720p/, payload has depth/pose
    # Pose is FLAT in shard, depth is in depth/ subdirectory
    # Note: DPSP video field might be just UID, not full path!
    # =========================================
    if source_dataset == "dpsp" or "/rgb_720p/" in video_path or "/dpsp/" in video_path:
        # Find the dpsp root
        dpsp_root = None
        if "/dpsp/" in video_path:
            # Extract dpsp root from path
            idx = video_path.find("/dpsp/")
            dpsp_root = video_path[:idx + 5]  # .../dpsp
        elif vipe_data_root:
            dpsp_root = osp.join(vipe_data_root, "dpsp")
        
        # Handle case where video is just UID (not full path)
        # DPSP JSON has: "video": "3847c82a-c63c-4afb-b6de-3dea8874cef1"
        if "/" not in video_path and dpsp_root:
            # video_path is just the UID, construct full rgb path
            rgb_path = osp.join(dpsp_root, "rgb_720p", f"{uid}.mp4")
        
        if debug:
            print(f"[DEBUG] DPSP detected, dpsp_root={dpsp_root}, rgb_path={rgb_path}")
        
        if dpsp_root and osp.exists(dpsp_root):
            payload_dir = osp.join(dpsp_root, "payload")
            
            if osp.isdir(payload_dir):
                shards = os.listdir(payload_dir)
                if debug:
                    print(f"[DEBUG] Found {len(shards)} shards in {payload_dir}")
                
                # Search through shards for this uid
                for shard in shards:
                    shard_path = osp.join(payload_dir, shard)
                    if not osp.isdir(shard_path):
                        continue
                    
                    # Check for depth/pose in subdirectories first
                    depth_dir = osp.join(shard_path, "depth")
                    pose_dir = osp.join(shard_path, "pose")
                    intri_dir = osp.join(shard_path, "intrinsics")
                    
                    # Try depth - subdirectory first, then flat
                    if depth_path is None:
                        if osp.isdir(depth_dir):
                            depth_zip = osp.join(depth_dir, f"{uid}.zip")
                            if osp.exists(depth_zip):
                                depth_path = depth_zip
                        # Flat structure fallback
                        if depth_path is None:
                            depth_zip = osp.join(shard_path, f"{uid}.zip")
                            if osp.exists(depth_zip):
                                depth_path = depth_zip
                    
                    # Try pose - FLAT FIRST for DPSP (actual structure)
                    if pose_path is None:
                        # Flat structure first (DPSP actual structure)
                        for pose_name in [f"{uid}.npz", f"{uid}_pose.npz"]:
                            pose_npz = osp.join(shard_path, pose_name)
                            if osp.exists(pose_npz):
                                pose_path = pose_npz
                                break
                        # Subdirectory fallback
                        if pose_path is None and osp.isdir(pose_dir):
                            for pose_name in [f"{uid}.npz", f"{uid}_pose.npz"]:
                                pose_npz = osp.join(pose_dir, pose_name)
                                if osp.exists(pose_npz):
                                    pose_path = pose_npz
                                    break
                    
                    # Try intrinsics - subdirectory first, then flat
                    if intrinsics_path is None:
                        if osp.isdir(intri_dir):
                            for intri_name in [f"{uid}_intri.npz", f"{uid}.npz", f"{uid}_camera.txt"]:
                                intri_file = osp.join(intri_dir, intri_name)
                                if osp.exists(intri_file):
                                    intrinsics_path = intri_file
                                    break
                        # Flat structure fallback
                        if intrinsics_path is None:
                            for intri_name in [f"{uid}_intri.npz", f"{uid}.npz"]:
                                intri_file = osp.join(shard_path, intri_name)
                                if osp.exists(intri_file):
                                    intrinsics_path = intri_file
                                    break
                    
                    # If we found depth and pose, we're done
                    if depth_path and pose_path:
                        if debug:
                            print(f"[DEBUG] Found in shard {shard}: depth={depth_path}, pose={pose_path}")
                        break
        
        if debug and (depth_path is None or pose_path is None):
            print(f"[DEBUG] DPSP search failed: depth={depth_path}, pose={pose_path}")
        
        # Also check dynpose cameras directory for intrinsics
        if intrinsics_path is None and dpsp_root:
            dynpose_dir = osp.join(dpsp_root, "dynpose-100k", "dynpose_100k", "cameras")
            if osp.isdir(dynpose_dir):
                pkl_file = osp.join(dynpose_dir, f"{uid}.pkl")
                if osp.exists(pkl_file):
                    intrinsics_path = pkl_file
        
        return {
            "rgb_path": rgb_path,
            "pose_path": pose_path,
            "depth_path": depth_path,
            "intrinsics_path": intrinsics_path,
            "uid": uid,
        }
    
    # =========================================
    # WSDG: Everything in shard subdirectories
    # =========================================
    shard_dir = None
    if "/rgb/" in video_path:
        rgb_dir = osp.dirname(video_path)  # .../rgb
        shard_dir = osp.dirname(rgb_dir)   # .../wsdg-xxx
    else:
        # video_path: .../payload/wsdg-xxx/{uid}.mp4 (flat structure)
        shard_dir = osp.dirname(video_path)
    
    # Try to find pose file - check pose/ subdirectory first
    if pose_path is None and shard_dir and osp.exists(shard_dir):
        pose_dir = osp.join(shard_dir, "pose")
        if osp.isdir(pose_dir):
            for pose_name in [f"{uid}.npz", f"{uid}_pose.npz"]:
                pose_npz = osp.join(pose_dir, pose_name)
                if osp.exists(pose_npz):
                    pose_path = pose_npz
                    break
        
        # Fallback: flat structure (old format)
        if pose_path is None:
            for pose_name in [f"{uid}_pose.npz", f"{uid}.npz"]:
                pose_npz = osp.join(shard_dir, pose_name)
                if osp.exists(pose_npz):
                    pose_path = pose_npz
                    break
    
    # Try to find depth file - check depth/ subdirectory first
    if depth_path is None and shard_dir and osp.exists(shard_dir):
        depth_dir = osp.join(shard_dir, "depth")
        if osp.isdir(depth_dir):
            depth_zip = osp.join(depth_dir, f"{uid}.zip")
            if osp.exists(depth_zip):
                depth_path = depth_zip
        
        # Fallback: flat structure
        if depth_path is None:
            depth_zip = osp.join(shard_dir, f"{uid}.zip")
            if osp.exists(depth_zip):
                depth_path = depth_zip
    
    # Try to find intrinsics file - check intrinsics/ subdirectory first  
    if intrinsics_path is None and shard_dir and osp.exists(shard_dir):
        intri_dir = osp.join(shard_dir, "intrinsics")
        if osp.isdir(intri_dir):
            for intri_name in [f"{uid}_intri.npz", f"{uid}.npz", f"{uid}_camera.txt"]:
                intri_file = osp.join(intri_dir, intri_name)
                if osp.exists(intri_file):
                    intrinsics_path = intri_file
                    break
        
        # Fallback: flat structure
        if intrinsics_path is None:
            for intri_name in [f"{uid}_intri.npz", f"{uid}.npz"]:
                intri_file = osp.join(shard_dir, intri_name)
                if osp.exists(intri_file):
                    intrinsics_path = intri_file
                    break
    
    return {
        "rgb_path": rgb_path,
        "pose_path": pose_path,
        "depth_path": depth_path,
        "intrinsics_path": intrinsics_path,
        "uid": uid,
    }


def load_vipe_scene(
    scene_entry: Dict,
    num_frames: int = 32,
    target_size: Tuple[int, int] = (192, 192),
    target_size_llava: Tuple[int, int] = (384, 384),
    sample_mode: str = "uniform",
    make_canonical: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
    vipe_data_root: str = None,
) -> Tuple[List[np.ndarray], List[str], Dict]:
    """
    Load a ViPE scene (WSDG/W360/DPSP) for Cambrian-P training.
    
    This is for the original ViPE datasets, not Cambrian-S.
    Handles both full scene entries and augmented samples with only 'video' field.
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    config = get_vipe_config(scene_entry.get("source_dataset", "vipe"))
    
    # Resolve paths (handles augmented samples with only 'video' field)
    resolved = _resolve_vipe_paths(scene_entry, vipe_data_root)
    rgb_path = resolved["rgb_path"]
    depth_path = resolved["depth_path"]
    pose_path = resolved["pose_path"]
    intrinsics_path = resolved["intrinsics_path"]
    uid = resolved["uid"]
    
    # Validate required paths
    if not rgb_path or not osp.exists(rgb_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    if not depth_path or not osp.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    if not pose_path or not osp.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    
    if verbose:
        print(f"[ViPE] Loading {config.name}/{uid}")
    
    # Get frame count from depth ZIP
    total_frames = get_depth_frame_count_from_zip(depth_path)
    frame_indices = sample_frame_indices(total_frames, num_frames, sample_mode)
    
    if verbose:
        print(f"[ViPE] Sampled {len(frame_indices)} frames from {total_frames} total")
    
    # Load RGB
    rgb_frames, rgb_total = load_video_frames(rgb_path, frame_indices)
    
    if rgb_total != total_frames:
        rgb_indices = [int(i * (rgb_total - 1) / (total_frames - 1)) for i in frame_indices]
        rgb_frames, _ = load_video_frames(rgb_path, rgb_indices)
        if verbose:
            print(f"[ViPE] Resampled RGB: {rgb_total} -> {total_frames}")
    
    # Load depth
    depths = load_depth_from_zip(depth_path, frame_indices)
    depths *= config.depth_scale
    
    # Load poses
    poses_c2w = load_vipe_poses(pose_path, frame_indices)

    if make_canonical:
        poses_c2w = make_poses_canonical(poses_c2w)

    poses_w2c = np.linalg.inv(poses_c2w)
    
    # Load intrinsics
    original_size = config.image_size
    source_dataset = scene_entry.get("source_dataset", "vipe")
    
    if source_dataset == "wsdg" and intrinsics_path and osp.exists(intrinsics_path):
        intrinsics = load_wsdg_intrinsics(intrinsics_path, frame_indices, original_size)
    elif source_dataset == "dpsp" and intrinsics_path and osp.exists(intrinsics_path):
        intrinsics = load_dpsp_intrinsics(intrinsics_path, num_frames)
    elif config.default_focal is not None:
        intrinsics = build_default_intrinsics(config.default_focal, original_size, num_frames)
    else:
        intrinsics = build_default_intrinsics(float(original_size[1]), original_size, num_frames)
    
    # Process frames
    N = len(frame_indices)
    images_192 = []
    images_384 = []
    depths_192 = []
    video_frames_out = []
    
    for i in range(N):
        img = rgb_frames[i]
        
        img_384 = cv2.resize(img, (target_size_llava[1], target_size_llava[0]))
        images_384.append(np.transpose(img_384.astype(np.float32) / 255.0, (2, 0, 1)))
        video_frames_out.append(img_384)
        
        img_192 = cv2.resize(img, (target_size[1], target_size[0]))
        images_192.append(np.transpose(img_192.astype(np.float32) / 255.0, (2, 0, 1)))
        
        depth = depths[i]
        depth_192 = cv2.resize(depth, (target_size[1], target_size[0]), 
                               interpolation=cv2.INTER_NEAREST)
        depths_192.append(depth_192)
    
    scale_x = target_size[1] / original_size[1]
    scale_y = target_size[0] / original_size[0]
    
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[:, 0, 0] *= scale_x
    scaled_intrinsics[:, 1, 1] *= scale_y
    scaled_intrinsics[:, 0, 2] *= scale_x
    scaled_intrinsics[:, 1, 2] *= scale_y
    
    images_192 = np.stack(images_192, axis=0)
    images_384 = np.stack(images_384, axis=0)
    depths_192 = np.stack(depths_192, axis=0)
    
    c2w_tensor = torch.from_numpy(poses_c2w.astype(np.float32))
    cam_points = compute_cam_points(depths_192, scaled_intrinsics)
    world_points = compute_world_points(cam_points, c2w_tensor)
    
    valid_mask = depths_192 > 0
    extrinsics = poses_w2c[:, :3, :]
    basenames = [f"frame_{frame_indices[i]:06d}" for i in range(N)]
    
    rec_views = {
        "images": torch.from_numpy(images_192),
        "images_for_llava": torch.from_numpy(images_384),
        "depths": torch.from_numpy(depths_192),
        
        "extrinsics": torch.from_numpy(extrinsics.astype(np.float32)),
        "intrinsics": torch.from_numpy(scaled_intrinsics),
        
        "cam_points": cam_points,
        "world_points": world_points,
        
        "point_masks": torch.from_numpy(valid_mask),
        
        "scale_by_points": torch.tensor(True),
        "is_metric_scale": torch.tensor(config.is_metric_scale, dtype=torch.bool),
        
        "original_size": np.array([list(original_size)] * N, dtype=np.int32),
        "ids": np.arange(N, dtype=np.int64),
        "image_paths": [rgb_path] * N,
        "is_video": True,
        "num_views": N,
        "basenames": basenames,
        "seq_name": [f"{config.name}_{uid}"],
        "sample_mode": sample_mode,
    }
    
    return video_frames_out, basenames, rec_views


def load_vipe_scenes_for_training(
    vipe_data_path: str,
    thorough_validation: bool = False,
    vipe_data_root: str = None,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    """
    Load ViPE scenes JSON (WSDG/W360/DPSP) and format entries for list_data_dict.
    """
    with open(vipe_data_path, 'r') as f:
        raw_scenes = json.load(f)
    
    formatted_entries = []
    skipped_count = 0
    skipped_reasons = {}
    
    for scene in raw_scenes:
        source_dataset = scene.get("source_dataset", "vipe")
        
        # Use .get() for robustness - some JSON may have missing fields
        rgb_path = scene.get("rgb_path")
        depth_path = scene.get("depth_path")
        pose_path = scene.get("pose_path")
        intrinsics_path = scene.get("intrinsics_path")
        
        # Skip entries with missing required fields
        if not rgb_path or not depth_path or not pose_path:
            skipped_count += 1
            skipped_reasons["missing_paths"] = skipped_reasons.get("missing_paths", 0) + 1
            continue
        
        if vipe_data_root:
            if not osp.isabs(rgb_path):
                rgb_path = osp.join(vipe_data_root, rgb_path)
            if not osp.isabs(depth_path):
                depth_path = osp.join(vipe_data_root, depth_path)
            if not osp.isabs(pose_path):
                pose_path = osp.join(vipe_data_root, pose_path)
            if intrinsics_path and not osp.isabs(intrinsics_path):
                intrinsics_path = osp.join(vipe_data_root, intrinsics_path)
        
        if thorough_validation:
            if not osp.exists(rgb_path):
                skipped_count += 1
                skipped_reasons["no_rgb"] = skipped_reasons.get("no_rgb", 0) + 1
                continue
            if not osp.exists(depth_path):
                skipped_count += 1
                skipped_reasons["no_depth"] = skipped_reasons.get("no_depth", 0) + 1
                continue
            if not osp.exists(pose_path):
                skipped_count += 1
                skipped_reasons["no_pose"] = skipped_reasons.get("no_pose", 0) + 1
                continue
        
        entry = {
            "video": rgb_path,  # Now absolute!
            "loading_type": "rec",
            "data_source": "vipe",  # CRITICAL for is_vipe_scene()
            
            "source_dataset": source_dataset,
            "is_metric_scale": False,
            
            # Store absolute paths
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "pose_path": pose_path,
            "intrinsics_path": intrinsics_path,
            "image_size": scene.get("image_size", [720, 1280]),
            "default_focal": scene.get("default_focal"),
            
            "conversations": [
                {"from": "human", "value": "<video>\nReconstruct the 3D scene."},
                {"from": "gpt", "value": ""}
            ],
        }
        formatted_entries.append(entry)
    
    return formatted_entries, skipped_count, skipped_reasons


# =============================================================================
# Export for Sanity Check (call manually if needed)
# =============================================================================


def export_vipe_rec_views(
    rec_views: Dict,
    output_dir: str,
    scene_name: str = None,
    sample_idx: int = 0,
) -> str:
    """Export rec_views data for point cloud sanity check."""
    if scene_name is None:
        scene_name = rec_views.get("seq_name", ["unknown"])[0]
    
    scene_name = scene_name.replace("/", "_").replace(" ", "_")
    scene_dir = osp.join(output_dir, f"{sample_idx:04d}_{scene_name}")
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(osp.join(scene_dir, "images"), exist_ok=True)
    os.makedirs(osp.join(scene_dir, "depths"), exist_ok=True)
    
    images = rec_views["images"]
    depths = rec_views["depths"]
    basenames = rec_views["basenames"]
    
    extrinsics = rec_views["extrinsics"].numpy()
    cam2world = []
    for i in range(extrinsics.shape[0]):
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = extrinsics[i]
        c2w = np.linalg.inv(w2c)
        cam2world.append(c2w)
    cam2world = np.stack(cam2world, axis=0)
    
    intrinsics = rec_views["intrinsics"].numpy()
    
    for i, basename in enumerate(basenames):
        img = images[i].numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join(scene_dir, "images", f"{basename}.png"), img_bgr)
        
        depth = depths[i].numpy()
        np.save(osp.join(scene_dir, "depths", f"{basename}.npy"), depth)
    
    poses_data = {"frames": []}
    for i, basename in enumerate(basenames):
        poses_data["frames"].append({
            "frame_name": basename,
            "transform_matrix": cam2world[i].tolist(),
        })
    
    with open(osp.join(scene_dir, "poses.json"), "w") as f:
        json.dump(poses_data, f, indent=2)
    
    intrinsics_data = {"frames": []}
    for i, basename in enumerate(basenames):
        K = intrinsics[i]
        intrinsics_data["frames"].append({
            "frame_name": basename,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        })
    
    with open(osp.join(scene_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics_data, f, indent=2)
    
    def to_bool(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return bool(x.item()) if x.numel() == 1 else bool(x.any().item())
        return bool(x)
    
    metadata = {
        "scene_name": scene_name,
        "num_frames": len(basenames),
        "basenames": basenames,
        "is_metric_scale": to_bool(rec_views.get("is_metric_scale", False)),
        "scale_by_points": to_bool(rec_views.get("scale_by_points", True)),
        "image_size": list(images.shape[2:]),
        "sample_mode": rec_views.get("sample_mode", "uniform"),
    }
    
    with open(osp.join(scene_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[ViPE] Exported {len(basenames)} frames to {scene_dir}")
    return scene_dir