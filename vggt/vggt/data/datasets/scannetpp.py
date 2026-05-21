# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import os
import logging
import cv2
import numpy as np

from vggt.data.dataset_util import *
from vggt.data.base_dataset import BaseDataset

logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('asyncio').setLevel(logging.WARNING)
from cambrianp.datasets.utils.image import imread_cv2

class ScanNetppDataset(BaseDataset):
    def __init__(
        self,
        data_args,
        split: str = "train",
        SCANNETPP_DIR: str = None,
        scene_ids_file: str = None,
        len_train: int = 10000,
        len_test: int = 10000,
        sample_mode: str = 'unified',
        input_use_augs: bool = None,   
        rec_use_augs: bool = None,  
    ):
        """
        Initialize the ScanNetppDataset.

        Args:
            data_args: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            SCANNETPP_DIR (str): Directory path to ScanNet++ data.
            scene_ids_file (str): Path to file containing scene IDs to load.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            sample_mode (str): Sampling mode, either 'unified', 'interleaved', 'cut3r', or 'random'.
        """
        super().__init__(data_args=data_args, 
                        input_use_augs=input_use_augs, 
                        rec_use_augs=rec_use_augs)

        self.target_image_shape = np.array([384, 384])
        self.load_depth_data = data_args.load_depth_data
        self.scale_by_points = data_args.scale_by_points
        
        if SCANNETPP_DIR is None:
            raise ValueError("SCANNETPP_DIR must be specified.")

        self.SCANNETPP_DIR = SCANNETPP_DIR
        self.sample_mode = sample_mode

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        if scene_ids_file is not None:
            self.scenes = [scene_ids_file]
        else:
            # Load all available scenes
            self.scenes = [d for d in os.listdir(SCANNETPP_DIR) 
                          if osp.isdir(osp.join(SCANNETPP_DIR, d))]

        self.data_store = {}
        self.total_frame_num = 0

        logging.info(f"SCANNETPP_DIR is {SCANNETPP_DIR}")
        logging.info(f"Loading ScanNet++ scenes")

        # Load metadata for each scene
        for scene in self.scenes:
            scene_dir = osp.join(SCANNETPP_DIR, scene)
            
            try:
                metadata_path = osp.join(scene_dir, "scene_metadata_all.npz")
                with np.load(metadata_path, allow_pickle=True) as data:
                    images = data["images"]
                    intrinsics = data["intrinsics"]
                    trajectories = data["trajectories"]
                
                num_imgs = len(images)
                
                scene_data = []
                for idx in range(num_imgs):
                    frame_data = {
                        "basename": images[idx],
                        "index": idx,
                        "scene": scene,
                        "intrinsics": intrinsics[idx],
                        "pose": trajectories[idx],
                    }
                    scene_data.append(frame_data)
                
                self.data_store[scene] = scene_data
                self.total_frame_num += len(scene_data)
                
            except FileNotFoundError:
                logging.error(f"Metadata file not found for scene: {scene}")
                continue
            except Exception as e:
                logging.error(f"Error loading scene {scene}: {str(e)}")
                continue

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ScanNet++ Data size: {self.sequence_list_len} scenes")
        logging.info(f"{status}: ScanNet++ Total frames: {self.total_frame_num}")
        logging.info(f"{status}: ScanNet++ Dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        basenames: list = None,   
        seq_name: str = None,
        img_per_seq: int = 32,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.
        Process at 384x384, then downsample to 192x192 for reconstruction.
        
        Args:
            seq_index: Index of the sequence (unused when basenames provided)
            ids: List of image basenames to load (not indices!)
            seq_name: Name of the sequence
            img_per_seq: Number of images per sequence
            aspect_ratio: Aspect ratio for image processing
        """
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]
        scene_dir = osp.join(self.SCANNETPP_DIR, seq_name)

        assert basenames is not None, "basenames must be provided by the dataloader!"
        assert len(basenames) == img_per_seq, f"Expected {img_per_seq} basenames, got {len(basenames)}"

        basename_to_frame = {frame["basename"]: frame for frame in metadata}
        frames = []
        for basename in basenames:
            if basename in basename_to_frame:
                frames.append(basename_to_frame[basename])
            else:
                print(f"[WARNING] Basename {basename} not found in metadata, using fallback")
                frames.append({
                    "basename": basename,
                    "index": -1,
                    "scene": seq_name,
                    "intrinsics": metadata[0]["intrinsics"],  # Use first frame's intrinsics as fallback
                    "pose": metadata[0]["pose"],  # Use first frame's pose as fallback
                })
        
        # Initialize lists for collected data
        images_384 = []  # 384x384 images for VLM
        images_192 = []  # 192x192 images for reconstruction
        depths_192 = []  # Downsampled depths for reconstruction
        extrinsics_list = []
        intrinsics_192 = []  # Downsampled intrinsics for reconstruction
        cam_points_192 = []
        world_points_192 = []
        point_masks_192 = []
        original_sizes_list = []
        basenames_list = []
        image_paths = []

        rgb_dir = osp.join(scene_dir, "images")
        depth_dir = osp.join(scene_dir, "depth")

        for idx, frame in enumerate(frames):
            basename = frame["basename"]
            
            # Load RGB image
            image_path = osp.join(rgb_dir, basename + ".jpg")
            image_orig = read_image_cv2(image_path)
            
            # Load depth map
            depth_path = osp.join(depth_dir, basename + ".png")
            if self.load_depth_data:
                depth_map_orig = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
                depth_map_orig = depth_map_orig.astype(np.float32) / 1000.0  # Convert mm to meters
                depth_map_orig[~np.isfinite(depth_map_orig)] = 0  # Handle invalid depths
                # Threshold depth
                depth_map_orig = threshold_depth_map(depth_map_orig, min_percentile=-1, max_percentile=99)
            else:
                depth_map_orig = np.zeros(image_orig.shape[:2], dtype=np.float32)

            original_size = np.array(image_orig.shape[:2])
            
            # Get camera parameters
            extri_cam2world = frame["pose"].astype(np.float32)  # 4x4
            # Convert cam2world to world2cam
            extri_world2cam = np.linalg.inv(extri_cam2world)
            extri_opencv = extri_world2cam[:3, :]  # 3x4
            
            intri_opencv = frame["intrinsics"].astype(np.float32)

            # Use the unified processing method from base class
            processed = self.process_frame_with_augmentation(
                image_orig, depth_map_orig, extri_opencv, intri_opencv,
                original_size, self.target_image_shape, image_path
            )
            
            # Convert images from HWC to CHW format
            image_384_chw = np.transpose(processed['image_384_vlm'], (2,0,1)).astype(np.float32)
            image_192_chw = np.transpose(processed['image_192'], (2,0,1)).astype(np.float32) / 255.0
            
            images_384.append(image_384_chw)
            images_192.append(image_192_chw)
            depths_192.append(processed['depth_192'])
            extrinsics_list.append(processed['extri_opencv'])
            intrinsics_192.append(processed['intri_192'])
            cam_points_192.append(processed['cam_coords_192'])
            world_points_192.append(processed['world_coords_192'])
            point_masks_192.append(processed['point_mask_192'])
            original_sizes_list.append(original_size)
            basenames_list.append(basename)
            image_paths.append(image_path)

        # Stack all arrays
        views = {
            # Image data - [L, C, H, W] at 192x192 for reconstruction
            "images": np.stack(images_192, axis=0).astype(np.float32),
            
            # Image data - [L, C, H, W] at 384x384 for VLM
            "images_for_llava": np.stack(images_384, axis=0).astype(np.float32),
            
            # Depth maps - [L, H, W] at 192x192 for reconstruction
            "depths": np.stack(depths_192, axis=0).astype(np.float32),
            
            # Camera parameters - [L, 3, 4] and [L, 3, 3]
            "extrinsics": np.stack(extrinsics_list, axis=0).astype(np.float32),
            "intrinsics": np.stack(intrinsics_192, axis=0).astype(np.float32),
            
            # Point clouds - [L, H, W, 3] at 192x192
            "cam_points": np.stack(cam_points_192, axis=0).astype(np.float32),
            "world_points": np.stack(world_points_192, axis=0).astype(np.float32),
            
            # Masks - [L, H, W] at 192x192
            "point_masks": np.stack(point_masks_192, axis=0).astype(bool),
            
            # Sizes - [L, 2]
            "original_size": np.stack(original_sizes_list, axis=0).astype(np.int32),
            
            "seq_name": [f"scannetpp_{seq_name}"],
            "ids": np.array([frame.get("index", i) for i, frame in enumerate(frames)], dtype=np.int64),
            "basenames": basenames_list,
            "image_paths": image_paths,
            "is_video": True,
            "sample_mode": self.sample_mode,
            "num_views": len(frames),
            "scale_by_points": self.scale_by_points,
            "is_metric_scale": True,
        }
        
        return views