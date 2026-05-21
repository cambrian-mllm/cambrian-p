# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('asyncio').setLevel(logging.WARNING)

import cv2
import random
import numpy as np

from vggt.data.dataset_util import *
from vggt.data.base_dataset import BaseDataset


class Co3dDataset(BaseDataset):
    def __init__(
        self,
        data_args,
        split: str = "train",
        CO3D_DIR: str = None,
        CO3D_ANNOTATION_DIR: str = None,
        scene_ids_file: str = None,
        sample_mode: str = 'unified',
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the Co3dDataset.

        Args:
            data_args: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            CO3D_DIR (str): Directory path to CO3D data.
            CO3D_ANNOTATION_DIR (str): Directory path to CO3D annotations.
            scene_ids_file (str): Full path to scene directory (e.g., "apple/290_30758_58511").
            sample_mode (str): Sampling mode ('unified' or 'random'). Default is 'unified'.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        
        Raises:
            ValueError: If scene_ids_file is not specified.
        """
        super().__init__(data_args=data_args)

        self.training = getattr(data_args, 'training', True)
        self.force_square = True  # For [384,384] like ScanNet
        self.use_augs = data_args.use_augs
        self.sample_mode = sample_mode
        
        # Co3D specific parameters
        self.load_depth = getattr(data_args, 'load_depth', True)
        self.allow_duplicate_img = getattr(data_args, 'allow_duplicate_img', False)

        if scene_ids_file is None:
            raise ValueError("scene_ids_file must be specified for Co3D dataset")

        # Parse category and sequence from scene_ids_file
        # scene_ids_file format: "apple/290_30758_58511"
        parts = scene_ids_file.split("/")
        if len(parts) != 2:
            raise ValueError(f"scene_ids_file must be in format 'category/sequence', got: {scene_ids_file}")
        
        self.category = parts[0]
        self.sequence = parts[1]
        self.scene_path = scene_ids_file

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test

        # Load annotation for this specific sequence
        split_name = "train" if split == "train" else "test"
        annotation_file = osp.join(
            self.CO3D_ANNOTATION_DIR, f"{self.category}_{split_name}.jgz"
        )

        if not osp.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        with gzip.open(annotation_file, "r") as fin:
            all_annotations = json.loads(fin.read())

        if self.sequence not in all_annotations:
            raise ValueError(f"Sequence {self.sequence} not found in annotation file")

        self.metadata = all_annotations[self.sequence]
        self.total_frame_num = len(self.metadata)

        if self.total_frame_num < min_num_images:
            raise ValueError(f"Sequence has only {self.total_frame_num} frames, less than minimum {min_num_images}")

        # Store single sequence
        self.sequence_list = [self.scene_path]
        self.sequence_list_len = 1

        logging.info(f"Co3D Dataset initialized for scene: {scene_ids_file}")
        logging.info(f"Category: {self.category}, Sequence: {self.sequence}")
        logging.info(f"Total frames: {self.total_frame_num}")
        logging.info(f"Sample mode: {self.sample_mode}")

    def get_target_shape(self, aspect_ratio):
        """
        Override to always return square shape for reconstruction
        """
        if self.force_square:
            return np.array([self.img_size, self.img_size])
        else:
            return super().get_target_shape(aspect_ratio)

    def get_data(
        self,
        seq_index: int = None,
        ids: list = None,
        seq_name: str = None,
        img_per_seq: int = 32,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for the loaded sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve (ignored, only one sequence).
            ids (list): Specific IDs to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence (ignored, only one sequence).
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A dictionary with arrays of shape [L, ...] where L is the number of views
        """
        # Only one sequence loaded
        metadata = self.metadata

        if ids is None:
            if self.sample_mode == 'unified':
                # Unified sampling like ScanNet
                total_frames = len(metadata)
                ids = np.linspace(0, total_frames - 1, img_per_seq, dtype=int).tolist()
            else:
                # Random sampling (Co3D default)
                ids = np.random.choice(
                    len(metadata), img_per_seq, replace=self.allow_duplicate_img
                )

        frames = [metadata[i] for i in ids]
        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        extrinsics_list = []
        intrinsics_list = []
        cam_points_list = []
        world_points_list = []
        point_masks = []
        original_sizes_list = []
        image_paths = []

        for idx, frame in enumerate(frames):
            filepath = frame["filepath"]
            image_path = osp.join(self.CO3D_DIR, filepath)
            
            # Load RGB image
            image = read_image_cv2(image_path)

            # Load depth if enabled
            if self.load_depth:
                depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
                if osp.exists(depth_path):
                    depth_map = read_depth(depth_path, 1.0) #weird np.frombuffer in dataloading
                    # Apply MVS mask if available
                    mvs_mask_path = image_path.replace("/images", "/depth_masks").replace(".jpg", ".png")
                    if osp.exists(mvs_mask_path):
                        im = Image.open(mvs_mask_path).convert("L") 
                        mvs_mask = np.array(im) > 128  # dtype=bool
                        # mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                        depth_map[~mvs_mask] = 0

                    depth_map = threshold_depth_map(
                        depth_map, min_percentile=-1, max_percentile=98
                    )
                else:
                    # No depth file found
                    depth_map = np.zeros(image.shape[:2], dtype=np.float32)
            else:
                # Create dummy depth map
                depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            original_size = np.array(image.shape[:2])
            
            extri_opencv = np.array(frame["extri"], dtype=np.float32)
            intri_opencv = np.array(frame["intri"], dtype=np.float32)

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )
            # Convert image from HWC to CHW format
            image = np.transpose(image, (2,0,1)).astype(np.float32) / 255.0

            images.append(image)
            depths.append(depth_map)
            extrinsics_list.append(extri_opencv)
            intrinsics_list.append(intri_opencv)
            cam_points_list.append(cam_coords_points)
            world_points_list.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes_list.append(original_size)
            image_paths.append(image_path)

        # Stack all arrays to match ScanNet format
        views = {
            # Image data - [L, C, H, W]
            "images": np.stack(images, axis=0).astype(np.float32),
            
            # Depth maps - [L, H, W]
            "depths": np.stack(depths, axis=0).astype(np.float32),
            
            # Camera parameters - [L, 3, 4] and [L, 3, 3]
            "extrinsics": np.stack(extrinsics_list, axis=0).astype(np.float32),
            "intrinsics": np.stack(intrinsics_list, axis=0).astype(np.float32),
            
            # Point clouds - [L, H, W, 3]
            "cam_points": np.stack(cam_points_list, axis=0).astype(np.float32),
            "world_points": np.stack(world_points_list, axis=0).astype(np.float32),
            
            # Masks - [L, H, W]
            "point_masks": np.stack(point_masks, axis=0).astype(bool),
            
            # Sizes - [L, 2]
            "original_size": np.stack(original_sizes_list, axis=0).astype(np.int32),
            
            # Metadata
            "seq_name": [f"co3d_{self.scene_path}"],
            "ids": np.array(ids, dtype=np.int64),
            "image_paths": image_paths,
            "category": self.category,
            "is_video": self.sample_mode == 'unified',
            "sample_mode": self.sample_mode,
            "num_views": len(frames),
        }
        return views