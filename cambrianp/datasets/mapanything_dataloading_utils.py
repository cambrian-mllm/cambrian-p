"""
MapAnything Data Loading Utilities 

Complete version with covisibility-based sampling (exact MapAnything algorithm).
"""

import os
import os.path as osp
import json
import numpy as np
import cv2
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

@dataclass
class MapAnythingDatasetConfig:
    """Configuration for a MapAnything dataset."""
    name: str
    is_metric_scale: bool
    is_synthetic: bool
    default_depth_source: Optional[str] = None
    image_dir_priority: List[str] = field(default_factory=lambda: ["images", "images_distorted"])
    depth_extensions: List[str] = field(default_factory=lambda: [".npy", ".png"])
    needs_exr: bool = False
    covisibility_thres: float = 0.15 

# MAPANYTHING_DATASET_CONFIGS = {
#     # Metric scale datasets (real-world metric units)
#     "paralleldomain4d": MapAnythingDatasetConfig("paralleldomain4d", True, True, "depth", ["images"], [".npy", ".exr"], False, 0.15),
#     "tav2_wb": MapAnythingDatasetConfig("tav2_wb", True, True, "depth", ["images"], [".npy"], False, 0.15),
#     "mvs_synth": MapAnythingDatasetConfig("mvs_synth", True, True, "depth", ["images"], [".exr", ".npy"], True, 0.15),
#     "spring": MapAnythingDatasetConfig("spring", True, True, "depth", ["images"], [".exr", ".npy"], True, 0.15),
#     "sailvos3d": MapAnythingDatasetConfig("sailvos3d", True, True, "depth", ["images"], [".npy", ".exr"], False, 0.15),
#     "eth3d": MapAnythingDatasetConfig("eth3d", True, False, "depth", ["images"], [".npy", ".png"], False, 0.3),
#     "dynamicreplica": MapAnythingDatasetConfig("dynamicreplica", True, True, "depth", ["images"], [".npy"], False, 0.3),
#     "mpsd": MapAnythingDatasetConfig("mpsd", True, False, "depth", ["images"], [".npy", ".exr"], False, 0.15), 
#     "unrealstereo4k": MapAnythingDatasetConfig("unrealstereo4k", True, True,"depth", ["images"], [".exr", ".npy"], True, 0.25), 
        
#     # NON-METRIC scale datasets (relative scale only, need alignment)
#     "megadepth": MapAnythingDatasetConfig("megadepth", False, False, "depth", ["images"], [".npy", ".png"], False, 0.15),
#     "dl3dv-960p": MapAnythingDatasetConfig("dl3dv-960p", False, False, "moge", ["images"], [".exr", ".npy"], True, 0.15),
#     "blendedmvs": MapAnythingDatasetConfig("blendedmvs", False, False, "depth", ["images"], [".exr", ".npy", ".png"], True, 0.15),  
# }


MAPANYTHING_DATASET_CONFIGS = {
    # Metric scale datasets (real-world metric units)
    "paralleldomain4d": MapAnythingDatasetConfig("paralleldomain4d", True, True, "depth", ["images"], [".npy", ".exr"], False, 0.15),
    "tartanairv2-wb": MapAnythingDatasetConfig("tartanairv2-wb", True, True, "depth", ["images"], [".npy"], False, 0.15),
    "tav2_wb": MapAnythingDatasetConfig("tav2_wb", True, True, "depth", ["images"], [".npy"], False, 0.15),
    "mvs-synth": MapAnythingDatasetConfig("mvs-synth", True, True, "depth", ["images"], [".exr", ".npy"], True, 0.15),
    "mvs_synth": MapAnythingDatasetConfig("mvs_synth", True, True, "depth", ["images"], [".exr", ".npy"], True, 0.15),
    "spring": MapAnythingDatasetConfig("spring", True, True, "depth", ["images"], [".exr", ".npy"], True, 0.15),
    "sailvos3d": MapAnythingDatasetConfig("sailvos3d", True, True, "depth", ["images"], [".npy", ".exr"], False, 0.15),
    "eth3d": MapAnythingDatasetConfig("eth3d", True, False, "depth", ["images"], [".npy", ".png"], False, 0.3),
    "dynamic_replica": MapAnythingDatasetConfig("dynamic_replica", True, True, "depth", ["images"], [".npy"], False, 0.3),
    "dynamicreplica": MapAnythingDatasetConfig("dynamicreplica", True, True, "depth", ["images"], [".npy"], False, 0.3),
    "mpsd": MapAnythingDatasetConfig("mpsd", True, False, "depth", ["images"], [".npy", ".exr"], False, 0.15), 
    "unrealstereo4k": MapAnythingDatasetConfig("unrealstereo4k", True, True, "depth", ["images"], [".exr", ".npy"], True, 0.25),
        
    # NON-METRIC scale datasets (relative scale only, need alignment)
    "megadepth": MapAnythingDatasetConfig("megadepth", False, False, "depth", ["images"], [".npy", ".png"], False, 0.15),
    "dl3dv-960p": MapAnythingDatasetConfig("dl3dv-960p", False, False, "moge", ["images"], [".exr", ".npy"], True, 0.15),
    "blendedmvs": MapAnythingDatasetConfig("blendedmvs", False, False, "depth", ["images"], [".exr", ".npy", ".png"], True, 0.15),
}

def get_mapanything_config(dataset_name: str) -> MapAnythingDatasetConfig:
    """Get configuration for a MapAnything dataset."""
    if dataset_name is None:
        dataset_name = "unknown"
    
    key = dataset_name.lower().replace("-", "_").replace(" ", "_")
    
    if key in MAPANYTHING_DATASET_CONFIGS:
        return MAPANYTHING_DATASET_CONFIGS[key]
    
    for k, v in MAPANYTHING_DATASET_CONFIGS.items():
        k_norm = k.replace("-", "_")
        if k_norm in key or key in k_norm:
            return v
    
    return MapAnythingDatasetConfig(name=dataset_name, is_metric_scale=False, is_synthetic=False)


# ============================================================================
# PATH RESOLUTION
# ============================================================================

def resolve_mapanything_scene_path(scene_path: str) -> str:
    """Resolve nested scene structure (scene_name/scene_name/)."""
    if not osp.exists(scene_path):
        return scene_path
    
    for meta_file in ["scene_meta.json", "scene_meta_distorted.json"]:
        if osp.exists(osp.join(scene_path, meta_file)):
            return scene_path
    
    scene_name = osp.basename(scene_path)
    nested_path = osp.join(scene_path, scene_name)
    
    if osp.exists(nested_path):
        for meta_file in ["scene_meta.json", "scene_meta_distorted.json"]:
            if osp.exists(osp.join(nested_path, meta_file)):
                return nested_path
    
    return scene_path

def validate_mapanything_scene(scene_path: str, thorough: bool = False) -> Tuple[bool, str]:
    try:
        resolved = resolve_mapanything_scene_path(scene_path)
        
        # Check 1: scene_meta.json exists
        meta_path = None
        for meta_file in ["scene_meta.json", "scene_meta_distorted.json"]:
            candidate = osp.join(resolved, meta_file)
            if osp.exists(candidate):
                meta_path = candidate
                break
        
        if meta_path is None:
            return False, f"No scene_meta.json found"
        
        # Check 2: Can load metadata
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            return False, f"Failed to parse metadata: {e}"
        
        # Check 3: Has valid frames
        frames = meta.get("frames", [])
        valid_frames = [f for f in frames if not f.get("is_bad", False)]
        if not valid_frames:
            return False, f"No valid frames"
        
        # Check 4: Images directory exists with files
        images_dir = None
        for dir_name in ["images_distorted", "images"]:
            full_path = osp.join(resolved, dir_name)
            if osp.exists(full_path) and osp.isdir(full_path):
                files = [f for f in os.listdir(full_path) 
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if files:
                    images_dir = full_path
                    break
        
        if images_dir is None:
            return False, f"No images directory"
        
        # Check 4b: Thorough check - verify frame images actually exist
        if thorough and images_dir:
            image_files = set(os.listdir(images_dir))
            missing_count = 0
            check_frames = valid_frames[:min(10, len(valid_frames))]  # Check first 10
            
            for frame in check_frames:
                name = frame.get("frame_name", frame.get("file_path", ""))
                if "/" in name:
                    name = osp.splitext(osp.basename(name))[0]
                name = osp.splitext(name)[0]
                
                found = any(
                    f"{name}{ext}" in image_files 
                    for ext in [".png", ".jpg", ".JPG", ".jpeg"]
                )
                if not found:
                    missing_count += 1
            
            if missing_count > len(check_frames) // 2:
                return False, f"Frame images missing ({missing_count}/{len(check_frames)} checked)"
        
        # Check 5: Covisibility exists
        covis_dir = osp.join(resolved, "covisibility", "v0")
        if not osp.exists(covis_dir):
            return False, f"No covisibility directory"
        
        npy_files = [f for f in os.listdir(covis_dir) if f.endswith(".npy")]
        if not npy_files:
            return False, f"No covisibility .npy file"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {e}"


def is_mapanything_scene(path: str) -> bool:
    """Check if path contains valid MapAnything WAI data."""
    resolved = resolve_mapanything_scene_path(path)
    
    for meta_file in ["scene_meta.json", "scene_meta_distorted.json"]:
        meta_path = osp.join(resolved, meta_file)
        if osp.exists(meta_path):
            return True
    return False


def detect_mapanything_dataset_type(video_file: str) -> Optional[str]:
    """Detect MapAnything dataset type from path or metadata."""
    if not is_mapanything_scene(video_file):
        return None
    
    resolved = resolve_mapanything_scene_path(video_file)
    
    for meta_file in ["scene_meta.json", "scene_meta_distorted.json"]:
        meta_path = osp.join(resolved, meta_file)
        if osp.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                return meta.get("dataset_name", "mapanything")
            except:
                pass
    
    path_lower = video_file.lower()
    for dataset_name in MAPANYTHING_DATASET_CONFIGS.keys():
        if dataset_name.replace("_", "-") in path_lower or dataset_name in path_lower:
            return dataset_name
    
    return "mapanything"


# ============================================================================
# COVISIBILITY-BASED SAMPLING 
# ============================================================================

def load_covisibility_matrix(scene_path: str, version: str = "v0") -> np.ndarray:
    """
    Load pairwise covisibility matrix for a scene.
    
    MapAnything stores covisibility in: {scene}/covisibility/{version}/*.npy
    The matrix is loaded as memory-mapped for efficiency with large scenes.
    
    Args:
        scene_path: Path to scene directory
        version: Covisibility version key (default "v0")
    
    Returns:
        N x N covisibility matrix
    
    Raises:
        FileNotFoundError: If covisibility matrix not found (REQUIRED in MapAnything)
    """
    resolved_path = resolve_mapanything_scene_path(scene_path)
    covisibility_dir = osp.join(resolved_path, "covisibility", version)
    
    if not osp.isdir(covisibility_dir):
        raise FileNotFoundError(
            f"Covisibility directory not found: {covisibility_dir}\n"
            "MapAnything requires covisibility data for sampling. "
            "There is NO temporal fallback - this is by design."
        )
    
    # Find the .npy file in the directory
    npy_files = [f for f in os.listdir(covisibility_dir) if f.endswith(".npy")]
    if not npy_files:
        raise FileNotFoundError(
            f"No .npy file found in {covisibility_dir}\n"
            "MapAnything requires covisibility data for sampling."
        )
    
    covisibility_path = osp.join(covisibility_dir, npy_files[0])
    
    # Load as memory-mapped array for efficiency (matching MapAnything)
    return np.load(covisibility_path, mmap_mode='r')


def _random_walk_sampling(
    scene_pairwise_covisibility: np.ndarray,
    num_of_samples: int,
    covisibility_thres: float,
    rng: np.random.Generator,
    max_retries: int = 4,
    use_bidirectional_covis: bool = True,
) -> np.ndarray:
    """
    Randomly samples S indices from an N x N covisibility matrix by forming adjacency edges such that the resulting subgraph (given by the indices) is connected.
    If the current node has no new unvisited neighbors, backtracking occurs.
    Retries with different starting indices if the desired number of samples is not reached, excluding previously visited components.
    
    Args:
        scene_pairwise_covisibility: N x N covisibility matrix for the scene
        num_of_samples: The desired number of nodes to sample
        covisibility_thres: Threshold for adjacency (normalized covisibility > threshold)
        rng: Random number generator
        max_retries: Maximum number of retries with different starting indices
        use_bidirectional_covis: Whether to compute bidirectional covisibility
    
    Returns:
        Array of sampled indices forming a connected subgraph
    """
    excluded_nodes = set()
    best_walk = []  # To keep track of the best walk found
    
    for _ in range(max_retries):
        visited = set()
        walk = []  # List to store the random walk sampling order
        stack = []  # Stack for backtracking
        
        # Choose a random starting index that is not in the excluded set
        all_nodes = set(range(len(scene_pairwise_covisibility)))
        available_nodes = list(all_nodes - excluded_nodes)
        if not available_nodes:
            break  # No more nodes to try
        
        start = rng.choice(available_nodes)
        walk.append(start)
        visited.add(start)
        stack.append(start)
        
        # Continue until we have sampled S indices or all expandable nodes are exhausted
        while len(walk) < num_of_samples and stack:
            current = stack[-1]
            
            # Get the pairwise covisibility for the current node
            if use_bidirectional_covis:
                # Use bidirectional covisibility (slower for large memory-mapped arrays)
                pairwise_covisibility = (
                    scene_pairwise_covisibility[current, :]
                    + scene_pairwise_covisibility[:, current].T
                ) / 2
            else:
                # Use only row access (faster for large memory-mapped arrays)
                pairwise_covisibility = scene_pairwise_covisibility[current, :]
            
            # Normalize the covisibility using self covisibility
            # EXACT SAME AS MAPANYTHING: divide by (self + 1e-8)
            pairwise_covisibility = pairwise_covisibility / (
                pairwise_covisibility[current] + 1e-8
            )
            
            # Assign overlap score of zero to self-pairs
            pairwise_covisibility[current] = 0
            
            # Threshold the covisibility to get adjacency list for the current node
            adjacency_list_for_current = (
                pairwise_covisibility > covisibility_thres
            ).astype(int)
            adjacency_list_for_current = np.flatnonzero(adjacency_list_for_current)
            
            # Get all unvisited neighbors
            candidates = [
                idx for idx in adjacency_list_for_current if idx not in visited
            ]
            
            if candidates:
                # Randomly select one of the unvisited overlapping neighbors
                next_node = rng.choice(candidates)
                walk.append(next_node)
                visited.add(next_node)
                stack.append(next_node)
            else:
                # If no unvisited neighbor is available, backtrack
                stack.pop()
        
        # Update the best walk if the current walk is larger
        if len(walk) > len(best_walk):
            best_walk = walk
        
        # If we have enough samples, return the result
        if len(walk) >= num_of_samples:
            return np.array(walk)
        
        # Add all visited nodes to the excluded set
        excluded_nodes.update(visited)
    
    # If all retries are exhausted and we still don't have enough samples, return the best walk found
    return np.array(best_walk)


def _sample_view_indices(
    num_views_to_sample: int,
    num_views_in_scene: int,
    scene_pairwise_covisibility: np.ndarray,
    covisibility_thres: float,
    rng: np.random.Generator,
    use_bidirectional_covis: bool = True,
) -> np.ndarray:
    """
    Sample view indices from a scene based on covisibility matrix.
    
    from MapAnything's BaseDataset._sample_view_indices()
    
    Args:
        num_views_to_sample: Number of views to sample
        num_views_in_scene: Total number of views available in the scene
        scene_pairwise_covisibility: N x N covisibility matrix
        covisibility_thres: Threshold for adjacency
        rng: Random number generator
        use_bidirectional_covis: Whether to use bidirectional covisibility
    
    Returns:
        Array of sampled view indices
    """
    if num_views_to_sample == num_views_in_scene:
        # Select all views in the scene
        view_indices = rng.permutation(num_views_in_scene)
    elif num_views_to_sample > num_views_in_scene:
        # Select all views in the scene and repeat them to get the desired number of views
        view_indices = rng.choice(
            num_views_in_scene, size=num_views_to_sample, replace=True
        )
    else:
        # Select a subset of single component connected views in the scene using random walk sampling
        view_indices = _random_walk_sampling(
            scene_pairwise_covisibility,
            num_views_to_sample,
            covisibility_thres,
            rng,
            use_bidirectional_covis=use_bidirectional_covis,
        )
        # If the required num of views can't be obtained even with 4 retries, repeat existing indices
        # THIS IS EXACTLY WHAT MAPANYTHING DOES - NO TEMPORAL FALLBACK
        if len(view_indices) < num_views_to_sample:
            view_indices = rng.choice(
                view_indices, size=num_views_to_sample, replace=True
            )
    
    return view_indices


# ============================================================================
# DEPTH LOADING
# ============================================================================

def load_exr_depth(path: str) -> Optional[np.ndarray]:
    """Load EXR depth file with multiple backends."""
    # Try OpenEXR library first
    try:
        import OpenEXR
        import Imath
        
        exr_file = OpenEXR.InputFile(path)
        header = exr_file.header()
        
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        for channel in ['Z', 'Y', 'R', 'depth', 'B']:
            if channel in header['channels']:
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                depth_str = exr_file.channel(channel, pt)
                depth = np.frombuffer(depth_str, dtype=np.float32)
                depth = depth.reshape((height, width))
                return depth
        
        channels = list(header['channels'].keys())
        if channels:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr_file.channel(channels[0], pt)
            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth = depth.reshape((height, width))
            return depth
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try OpenCV with EXR support
    try:
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is not None:
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            return depth.astype(np.float32)
    except Exception:
        pass
    
    # Try imageio
    try:
        import imageio
        depth = imageio.imread(path)
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        return depth.astype(np.float32)
    except:
        pass
    
    return None


def load_mapanything_depth_file(path: str) -> Optional[np.ndarray]:
    """Load depth from various formats."""
    if not osp.exists(path):
        return None
    
    ext = osp.splitext(path)[1].lower()
    
    if ext == '.npy':
        try:
            return np.load(path).astype(np.float32)
        except Exception as e:
            print(f"[MapAnything] Failed to load depth {path}: {e}")
            return None
    
    elif ext == '.exr':
        return load_exr_depth(path)
    
    elif ext in ['.png']:
        try:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                return depth.astype(np.float32) / 1000.0
        except Exception as e:
            print(f"[MapAnything] Failed to load depth {path}: {e}")
            return None
    
    return None


# ============================================================================
# DEPTH DIRECTORY FINDING (handles nested structures like dl3dv)
# ============================================================================

def find_depth_directory(
    resolved_path: str, 
    config: MapAnythingDatasetConfig,
    is_distorted: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find depth directory with proper handling of nested structures.
    
    Handles:
    - Standard: depth/, rendered_depth/
    - Distorted variants: depth_distorted/, moge_distorted/
    - Nested: moge/v0/depth/, metric3dv2/v0/depth/
    
    Args:
        resolved_path: Scene directory path
        config: Dataset configuration
        is_distorted: If True, prioritize distorted depth directories
    
    Returns: (depth_dir_full, depth_dir_rel) or (None, None)
    """
    depth_priority = []
    
    if config.default_depth_source:
        depth_priority.append(config.default_depth_source)
    
    depth_priority.extend(["depth", "rendered_depth"])
    depth_priority.extend(["moge", "metric3dv2", "mvsanywhere"])
    
    # Remove duplicates while preserving order
    seen = set()
    depth_priority = [x for x in depth_priority if not (x in seen or seen.add(x))]
    
    # If distorted, prepend distorted variants to priority list
    if is_distorted:
        distorted_priority = []
        for dir_name in depth_priority:
            distorted_priority.append(f"{dir_name}_distorted")
        # Also check for nested distorted (e.g., moge/v0/depth_distorted/)
        depth_priority = distorted_priority + depth_priority
    
    for dir_name in depth_priority:
        base_path = osp.join(resolved_path, dir_name)
        
        if not osp.exists(base_path) or not osp.isdir(base_path):
            continue
        
        # Case 1: Direct depth files (depth/, rendered_depth/, depth_distorted/)
        if any(x in dir_name for x in ["depth", "rendered_depth"]):
            files = [f for f in os.listdir(base_path) 
                    if f.lower().endswith(('.npy', '.exr', '.png'))]
            if files:
                return base_path, dir_name
        
        # Case 2: Nested v0/depth structure (moge/v0/depth/ or moge_distorted/v0/depth/)
        # Also check for moge/v0/depth_distorted/
        nested_paths_to_check = [
            osp.join(base_path, "v0", "depth"),
        ]
        if is_distorted:
            nested_paths_to_check.insert(0, osp.join(base_path, "v0", "depth_distorted"))
        
        for nested_v0_depth in nested_paths_to_check:
            if not osp.exists(nested_v0_depth) or not osp.isdir(nested_v0_depth):
                continue
                
            items = os.listdir(nested_v0_depth)
            rel_suffix = nested_v0_depth.replace(resolved_path + os.sep, "")
            
            # Check if v0/depth contains depth files directly
            depth_files = [f for f in items if f.lower().endswith(('.npy', '.exr', '.png'))]
            if depth_files:
                return nested_v0_depth, rel_suffix
            
            # Case 3: Extra subdirectory (moge/v0/depth/moge2/)
            # For distorted, check moge2_distorted first
            subdirs = [d for d in items if osp.isdir(osp.join(nested_v0_depth, d))]
            
            if is_distorted:
                # Prioritize distorted subdirs
                distorted_subdirs = [d for d in subdirs if "distorted" in d.lower()]
                other_subdirs = [d for d in subdirs if "distorted" not in d.lower()]
                subdirs = distorted_subdirs + other_subdirs
            
            for subdir in subdirs:
                subdir_path = osp.join(nested_v0_depth, subdir)
                subdir_files = [f for f in os.listdir(subdir_path) 
                               if f.lower().endswith(('.npy', '.exr', '.png'))]
                if subdir_files:
                    return subdir_path, f"{rel_suffix}/{subdir}"
    
    return None, None


# ============================================================================
# METADATA LOADING
# ============================================================================

def load_mapanything_metadata(scene_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Load WAI format metadata from a MapAnything scene."""
    resolved_path = resolve_mapanything_scene_path(scene_path)
    
    meta_path = None
    is_distorted = False
    for meta_file in ["scene_meta.json", "scene_meta_distorted.json"]:
        candidate = osp.join(resolved_path, meta_file)
        if osp.exists(candidate):
            meta_path = candidate
            is_distorted = "distorted" in meta_file
            break
    
    if meta_path is None:
        raise FileNotFoundError(f"[MapAnything] No metadata in {scene_path}")
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    dataset_name = meta.get("dataset_name", "unknown")
    config = get_mapanything_config(dataset_name)
    
    # Find images directory
    images_dir_full = None
    images_dir_rel = None
    priority = ["images_distorted", "images"] if is_distorted else config.image_dir_priority
    
    for dir_name in priority:
        full_path = osp.join(resolved_path, dir_name)
        if osp.exists(full_path) and osp.isdir(full_path):
            files = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                images_dir_full = full_path
                images_dir_rel = dir_name
                break
    
    if images_dir_full is None:
        raise FileNotFoundError(f"[MapAnything] No images in {resolved_path}")
    
    # IMPORTANT: Update is_distorted based on actual image directory selected
    is_distorted = "distorted" in images_dir_rel.lower()
    
    # Find depth directory using new function
    depth_dir_full, depth_dir_rel = find_depth_directory(resolved_path, config, is_distorted=is_distorted)
    
    # Get valid frames
    frames = [f for f in meta.get("frames", []) if not f.get("is_bad", False)]
    if not frames:
        raise ValueError(f"[MapAnything] No valid frames in {meta_path}")
    
    # Build basenames
    images = []
    for frame in frames:
        name = frame.get("frame_name", frame.get("file_path", ""))
        if "/" in name:
            name = osp.splitext(osp.basename(name))[0]
        name = osp.splitext(name)[0]
        images.append(name)
    
    # Build intrinsics
    shared = meta.get("shared_intrinsics", "fl_x" in meta)
    if shared and "fl_x" in meta:
        fx, fy = meta["fl_x"], meta["fl_y"]
        cx, cy = meta["cx"], meta["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics = np.tile(K[None], (len(frames), 1, 1))
    else:
        intrinsics = []
        for frame in frames:
            fx = frame.get("fl_x", frame.get("fx", meta.get("fl_x", 500)))
            fy = frame.get("fl_y", frame.get("fy", meta.get("fl_y", 500)))
            cx = frame.get("cx", meta.get("cx", 320))
            cy = frame.get("cy", meta.get("cy", 240))
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            intrinsics.append(K)
        intrinsics = np.stack(intrinsics, axis=0)
    
    # Build cam2world poses
    trajectories = []
    for frame in frames:
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        if c2w.shape == (3, 4):
            c2w_full = np.eye(4, dtype=np.float32)
            c2w_full[:3, :] = c2w
            c2w = c2w_full
        trajectories.append(c2w)
    trajectories = np.stack(trajectories, axis=0)
    
    if verbose:
        print(f"[MapAnything] Loaded {dataset_name}: {len(frames)} frames, images={images_dir_rel}, depth={depth_dir_rel}")
    
    return {
        "images": np.array(images),
        "intrinsics": intrinsics,
        "trajectories": trajectories,
        "_images_dir_full": images_dir_full,
        "_images_dir_rel": images_dir_rel,
        "_depth_dir_full": depth_dir_full,
        "_depth_dir_rel": depth_dir_rel,
        "_scene_path": resolved_path,
        "_config": config,
        "_meta": meta,
        "_is_distorted": is_distorted,
        "_original_size": (meta.get("w", 640), meta.get("h", 480)),
    }


# ============================================================================
# 3D POINT COMPUTATION
# ============================================================================

def compute_cam_points(
    depths: np.ndarray,  # [N, H, W]
    intrinsics: np.ndarray,  # [N, 3, 3]
) -> torch.Tensor:
    """
    Compute 3D points in camera coordinates from depth maps.
    
    Returns:
        cam_points: [N, H, W, 3] - 3D points in camera space
    """
    N, H, W = depths.shape
    
    # Create pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)  # [H, W]
    
    cam_points = np.zeros((N, H, W, 3), dtype=np.float32)
    
    for i in range(N):
        depth = depths[i]  # [H, W]
        K = intrinsics[i]  # [3, 3]
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Unproject to camera coordinates
        x = (u - cx) * depth / (fx + 1e-8)
        y = (v - cy) * depth / (fy + 1e-8)
        z = depth
        
        cam_points[i] = np.stack([x, y, z], axis=-1)  # [H, W, 3]
    
    return torch.from_numpy(cam_points)


def compute_world_points(
    cam_points: torch.Tensor,  # [N, H, W, 3]
    cam2world: torch.Tensor,   # [N, 4, 4]
) -> torch.Tensor:
    """
    Transform camera-space points to world coordinates.
    
    Returns:
        world_points: [N, H, W, 3] - 3D points in world space
    """
    N, H, W, _ = cam_points.shape
    
    # Reshape for batch matmul: [N, H*W, 3]
    pts_flat = cam_points.view(N, -1, 3)
    
    # Extract rotation and translation
    R = cam2world[:, :3, :3]  # [N, 3, 3]
    t = cam2world[:, :3, 3:4]  # [N, 3, 1]
    
    # Transform: world = R @ cam + t
    # pts_flat: [N, H*W, 3], R: [N, 3, 3]
    # We want: [N, H*W, 3] @ [N, 3, 3]^T + [N, 1, 3]
    world_flat = torch.bmm(pts_flat, R.transpose(1, 2)) + t.transpose(1, 2)  # [N, H*W, 3]
    
    world_points = world_flat.view(N, H, W, 3)
    
    return world_points


# ============================================================================
# MAIN LOADING FUNCTION
# ============================================================================

def load_mapanything_scene(
    video_file: str,
    num_frames: int = 32,
    target_size: Tuple[int, int] = (192, 192),
    target_size_llava: Tuple[int, int] = (384, 384),
    covisibility_thres: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    sort_mapanything_frames: bool = False,
) -> Tuple[List[np.ndarray], List[str], Dict]:
    """
    Load MapAnything scene for Cambrian-P training.
    
    Args:
        video_file: Path to MapAnything scene directory
        num_frames: Number of frames to sample
        target_size: (H, W) for reconstruction images
        target_size_llava: (H, W) for VLM images (default 384x384)
        covisibility_thres: Covisibility threshold (uses dataset default if None)
        seed: Random seed for reproducibility
        verbose: Print debug info
        
    Returns:
        video_frames: List of RGB frames [H, W, 3] uint8 for VQA
        basenames: List of frame names
        rec_views: Dict with tensors for reconstruction (includes all required fields)
    """
    # Load metadata
    metadata = load_mapanything_metadata(video_file, verbose=verbose)
    config = metadata["_config"]
    resolved_path = metadata["_scene_path"]
    
    # Get covisibility threshold
    if covisibility_thres is None:
        covisibility_thres = config.covisibility_thres
    
    total = len(metadata["images"])
    
    # Initialize RNG
    rng = np.random.default_rng(seed)
    
    # Load covisibility matrix
    pairwise_covisibility = load_covisibility_matrix(resolved_path)
    covis_matrix_size = len(pairwise_covisibility)
 
    # MapAnything original data preprocessing:
    #   1. Extract all frames from video (e.g., 161 frames)
    #   2. Run SfM, mark low-quality frames as is_bad=True (e.g., 70 frames)
    #      Note: Only some samples have the is_bad field
    #   3. Compute covisibility matrix on all 161 frames → 161x161
    #   4. Save scene_meta.json with is_bad flags
    #
    # Because some samples have is_bad frames, the covisibility matrix and 
    # metadata["images"] can become misaligned.
    #
    # Solution: When mismatch detected (covis_matrix_size != total_valid_frames):
    #   - Re-read metadata to find valid_indices mapping: [0,1,3,4,6,...] (skip is_bad)
    #   - Subset covisibility so covis[i,j] represents covisibility between 
    #     valid_frame[i] and valid_frame[j]
    #   - Sampling now returns indices 0-90, aligned with metadata["images"]
    
    # Handle mismatch between covisibility matrix and valid frames
    if covis_matrix_size != total:

        # Re-read metadata to get valid frame indices mapping
        meta_path = osp.join(resolved_path, "scene_meta.json")
        if not osp.exists(meta_path):
            meta_path = osp.join(resolved_path, "scene_meta_distorted.json")
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        # Find which original indices are valid
        all_frames = meta.get("frames", [])
        valid_indices = np.array([i for i, f in enumerate(all_frames) if not f.get("is_bad", False)])
        
        # Subset covisibility matrix to valid frames only
        pairwise_covisibility = pairwise_covisibility[np.ix_(valid_indices, valid_indices)]
        
        print(f"[MapAnything] Subsetted covisibility: {covis_matrix_size}x{covis_matrix_size} -> {len(valid_indices)}x{len(valid_indices)}")

    indices = _sample_view_indices(
        num_views_to_sample=num_frames,
        num_views_in_scene=total,
        scene_pairwise_covisibility=pairwise_covisibility,  # Now correctly sized
        covisibility_thres=covisibility_thres,
        rng=rng,
    )

    if sort_mapanything_frames:
        indices = np.sort(indices)

    if verbose:
        print(f"[MapAnything] Sampled {len(indices)} views using covisibility (thres={covisibility_thres})")

    # Get selected data
    basenames = [metadata["images"][i] for i in indices]
    intrinsics = metadata["intrinsics"][indices]  # [N, 3, 3]
    c2w_poses = metadata["trajectories"][indices]  # [N, 4, 4] cam2world
    
    # Convert to world2cam (extrinsics)
    extrinsics = []
    for c2w in c2w_poses:
        w2c = np.linalg.inv(c2w)
        extrinsics.append(w2c[:3, :4])
    extrinsics = np.stack(extrinsics, axis=0)  # [N, 3, 4]
    
    # Load images
    images_dir = metadata["_images_dir_full"]
    depth_dir = metadata.get("_depth_dir_full")
    
    if verbose and depth_dir:
        print(f"[MapAnything] Using depth from: {metadata.get('_depth_dir_rel')}")
    
    # Depth extensions to try
    depth_extensions = [".exr", ".npy", ".png"]
    
    images = []
    images_for_llava = []
    video_frames = []
    depths = []
    original_size = None
    has_depth = False
    
    for basename in basenames:
        # Find image
        img_path = None
        for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
            candidate = osp.join(images_dir, basename + ext)
            if osp.exists(candidate):
                img_path = candidate
                break
        
        if img_path is None:
            raise FileNotFoundError(f"[MapAnything] Image not found: {basename} in {images_dir}")
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"[MapAnything] Failed to load: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if original_size is None:
            original_size = (img.shape[1], img.shape[0])  # W, H
        
        # Resize for reconstruction
        img_rec = cv2.resize(img, (target_size[1], target_size[0]))
        img_rec = img_rec.astype(np.float32) / 255.0
        img_rec = np.transpose(img_rec, (2, 0, 1))  # [3, H, W]
        images.append(img_rec)
        
        # Resize for LLaVA
        img_llava = cv2.resize(img, (target_size_llava[1], target_size_llava[0]))
        images_for_llava.append(np.transpose(img_llava.astype(np.float32) / 255.0, (2, 0, 1)))
        video_frames.append(img_llava)
        
        # Load depth
        if depth_dir:
            depth = None
            for ext in depth_extensions:
                depth_path = osp.join(depth_dir, basename + ext)
                if osp.exists(depth_path):
                    depth = load_mapanything_depth_file(depth_path)
                    if depth is not None:
                        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        depth = cv2.resize(depth, (target_size[1], target_size[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                        has_depth = True
                        break
            
            if depth is None:
                depth = np.zeros(target_size, dtype=np.float32)
            depths.append(depth)
        else:
            depths.append(np.zeros(target_size, dtype=np.float32))
    
    # Scale intrinsics
    scale_x = target_size[1] / original_size[0]
    scale_y = target_size[0] / original_size[1]
    
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[:, 0, 0] *= scale_x
    scaled_intrinsics[:, 1, 1] *= scale_y
    scaled_intrinsics[:, 0, 2] *= scale_x
    scaled_intrinsics[:, 1, 2] *= scale_y
    
    # Build tensors
    scene_name = osp.basename(metadata["_scene_path"])
    
    images_np = np.stack(images, axis=0)  # [N, 3, H, W]
    images_llava_np = np.stack(images_for_llava, axis=0)  # [N, 3, H, W]
    depths_np = np.stack(depths, axis=0)  # [N, H, W]
    c2w_tensor = torch.from_numpy(c2w_poses.astype(np.float32))  # [N, 4, 4]
    
    # Compute cam_points (3D points in camera space)
    cam_points = compute_cam_points(depths_np, scaled_intrinsics)  # [N, H, W, 3]
    
    # Compute world_points (3D points in world space)
    world_points = compute_world_points(cam_points, c2w_tensor)  # [N, H, W, 3]
    
    valid_mask = depths_np > 0  # [N, H, W]
    
    # Create mask tensor
    valid_mask_tensor = torch.from_numpy(valid_mask)  # [N, H, W] bool
    
    # scale_by_points: 
    # - True for non-metric datasets (normalize by point cloud extent)
    # - False for metric datasets (normalize by camera trajectory)
    scale_by_points = not config.is_metric_scale
    
    # Build rec_views dict with ALL required fields
    rec_views = {
        # Core image/depth data
        "images": torch.from_numpy(images_np),  # [N, 3, H, W]
        "images_for_llava": torch.from_numpy(images_llava_np),  # [N, 3, H, W]
        "depths": torch.from_numpy(depths_np),  # [N, H, W]
        
        # Camera parameters
        "extrinsics": torch.from_numpy(extrinsics.astype(np.float32)),  # [N, 3, 4] world2cam
        "intrinsics": torch.from_numpy(scaled_intrinsics.astype(np.float32)),  # [N, 3, 3]
        
        # 3D points (required by _process_batch_vggt)
        "cam_points": cam_points,  # [N, H, W, 3] - 3D points in camera space
        "world_points": world_points,  # [N, H, W, 3] - 3D points in world space
        
        # Masks (point_masks is the expected name in _process_batch_vggt)
        "point_masks": valid_mask_tensor,  # [N, H, W] bool - valid depth mask

        # Normalization control
        "scale_by_points": torch.tensor(scale_by_points),  # bool tensor
        # make sure to convert to booltensor here, otherwise can't be stacked later
        "is_metric_scale": torch.tensor(config.is_metric_scale, dtype=torch.bool),  
        
        "original_size": np.array([original_size] * len(basenames), dtype=np.int32),  # [N, 2] (W, H)
        "ids": np.arange(len(basenames), dtype=np.int64),  # [N]
        "image_paths": [osp.join(images_dir, f"{bn}.png") for bn in basenames],  # List[str]
        "is_video": True,
        "num_views": len(basenames),
        
        # Metadata
        "basenames": basenames,
        "seq_name": [f"{config.name}_{scene_name}"],
        "sample_mode": "covisibility",
    }
    
    return video_frames, basenames, rec_views


# ============================================================================
# EXPORT FOR SANITY CHECK
# ============================================================================

def export_mapanything_rec_views(
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
    extrinsics = rec_views["extrinsics"]
    intrinsics = rec_views["intrinsics"]
    basenames = rec_views["basenames"]
    
    if "_cam2world" in rec_views:
        cam2world = rec_views["_cam2world"].numpy()
    else:
        cam2world = []
        for i in range(extrinsics.shape[0]):
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = extrinsics[i].numpy()
            c2w = np.linalg.inv(w2c)
            cam2world.append(c2w)
        cam2world = np.stack(cam2world, axis=0)
    
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
        K = intrinsics[i].numpy()
        intrinsics_data["frames"].append({
            "frame_name": basename,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        })
    
    with open(osp.join(scene_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics_data, f, indent=2)
    
    # Helper to safely convert tensor/bool to Python bool for JSON serialization
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
        "scale_by_points": to_bool(rec_views.get("scale_by_points", False)),
        "has_depth": rec_views.get("has_depth", False),
        "image_size": list(images.shape[2:]),
        "sample_mode": rec_views.get("sample_mode", "covisibility"),
    }
    
    with open(osp.join(scene_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[MapAnything] Exported {len(basenames)} frames to {scene_dir}")
    return scene_dir

MAPANYTHING_METRIC_SCALE_LOOKUP = {
    # Metric scale datasets (real-world units)
    "paralleldomain4d": True,
    "tartanairv2-wb": True,
    "tav2_wb": True,
    "mvs-synth": True,
    "mvs_synth": True,
    "spring": True,
    "sailvos3d": True,
    "eth3d": True,
    "dynamic_replica": True,
    "dynamicreplica": True,
    "mpsd": True,  
    "unrealstereo4k": True,
    
    # Non-metric scale datasets (relative scale only)
    "megadepth": False,
    "dl3dv-960p": False,
    "blendedmvs": False,
    
    # Default for unknown
    "mapanything": True,
}


def is_mapanything_source_dataset(source_dataset: Optional[str]) -> bool:
    """Return True if source_dataset belongs to MapAnything datasets."""
    if source_dataset is None:
        return False

    # Normalize separators so checks are robust to "-" vs "_" naming variants.
    normalized = source_dataset.lower().replace("-", "_").replace(" ", "_")
    for key in MAPANYTHING_METRIC_SCALE_LOOKUP.keys():
        key_norm = key.replace("-", "_")
        if key_norm in normalized or normalized in key_norm:
            return True
    return False


def get_is_metric_scale(source_dataset: str) -> bool:
    """Get whether a dataset has metric scale."""
    normalized = source_dataset.lower().replace("-", "_").replace(" ", "_")
    
    for key, value in MAPANYTHING_METRIC_SCALE_LOOKUP.items():
        if key.replace("-", "_") in normalized or normalized in key.replace("-", "_"):
            return value
    
    return True  # Default to metric


def load_mapanything_scenes_for_training(
    mapanything_data_path: str,
    mapanything_data_root: Optional[str] = None,
    use_metric_only: bool = False,   
    thorough_validation: bool = False,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    """
    Load MapAnything scenes JSON and format entries for list_data_dict.
    
    Args:
        mapanything_data_path: Path to JSON file with MapAnything scenes.
        mapanything_data_root: Optional root directory for relative paths.
        use_metric_only: If True, only include metric-scale datasets.
        thorough_validation: If True, validate each scene before adding (slower but safer).
        
    Returns:
        Tuple of (formatted_entries, skipped_count, skipped_reasons)
    """
    from tqdm import tqdm
    
    with open(mapanything_data_path, 'r') as f:
        raw_scenes = json.load(f)
    
    formatted_entries = []
    skipped_count = 0  # Track skipped non-metric scenes
    skipped_reasons = {}
    skipped_scenes = []  # Track problematic scenes
    
    desc = "[MapAnything] Validating scenes" if thorough_validation else "[MapAnything] Loading scenes"
    for scene in tqdm(raw_scenes, desc=desc):
        video_path = scene['video']
        source_dataset = scene.get('source_dataset', 'mapanything')
        
        # Resolve path if root is provided
        if mapanything_data_root and not osp.isabs(video_path):
            video_path = osp.join(mapanything_data_root, video_path)
         
        # Get metric scale flag
        is_metric = get_is_metric_scale(source_dataset)
        
        # Filter non-metric datasets if use_metric_only is True
        if use_metric_only and not is_metric:
            skipped_count += 1
            skipped_reasons["non_metric"] = skipped_reasons.get("non_metric", 0) + 1
            continue
        
        # Only validate if thorough_validation is enabled
        if thorough_validation:
            is_valid, error_msg = validate_mapanything_scene(video_path)
            if not is_valid:
                skipped_count += 1
                if "images" in error_msg.lower():
                    reason = "missing_images"
                elif "covisibility" in error_msg.lower():
                    reason = "missing_covisibility"
                elif "meta" in error_msg.lower():
                    reason = "missing_metadata"
                elif "frame" in error_msg.lower():
                    reason = "missing_frames"
                else:
                    reason = "other"
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                skipped_scenes.append(f"{video_path}\t{reason}\t{error_msg}")
                continue
        
        # Format entry to match existing list_data_dict structure
        entry = {
            # Core fields for data loading
            "video": video_path,
            "loading_type": "rec",
            
            "source_dataset": source_dataset,
            # Metric scale flag for loss computation
            "is_metric_scale": is_metric,
            
            # Placeholder conversations (MapAnything uses rec-style loading)
            "conversations": [
                {"from": "human", "value": "<video>\nReconstruct the 3D scene."},
                {"from": "gpt", "value": ""}
            ],
        }
        formatted_entries.append(entry)
    
    # Save skipped scenes to txt file
    if skipped_scenes:
        skipped_txt_path = mapanything_data_path.replace('.json', 'skipped.txt')
        with open(skipped_txt_path, 'w') as f:
            f.write("path\treason\terror_msg\n")
            f.write("\n".join(skipped_scenes))
        print(f"[MapAnything] Saved {len(skipped_scenes)} skipped scenes to {skipped_txt_path}")
    
    return formatted_entries, skipped_count, skipped_reasons
