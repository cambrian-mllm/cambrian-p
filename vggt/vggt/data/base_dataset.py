# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *
from .augmentation_strategies import StrategyFactory

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(Dataset):
    """
    Base dataset class for VGGT and VGGSfM training.

    This abstract class handles common operations like image resizing,
    augmentation, and coordinate transformations. Concrete dataset
    implementations should inherit from this class.
    
    AUGMENTATION STRATEGY OVERVIEW:
    The class supports multiple augmentation modes based on training configuration:
    
    1. Standard VQA Training (default mode):
       - VLM gets clean images for accurate question answering
       - Reconstruction gets augmented images for robust 3D learning
       
    2. Interleaved Training - VQA samples:
       - When mixing VQA and reconstruction in same batch
       - VLM still gets clean images for QA tasks
       - Reconstruction uses augmentation for robustness
       
    3. Interleaved Training - Pure reconstruction:
       - No QA tasks, only 3D reconstruction
       - Both VLM and reconstruction can use augmented images
       - Maximizes data diversity since no QA accuracy needed
       
    4. Test/Evaluation mode:
       - No augmentations for deterministic results
       - Ensures reproducible evaluation metrics
    """
    def __init__(
        self,
        data_args,
        input_use_augs=None,
        rec_use_augs=None,
    ):
        """
        Initialize the base dataset with data arguments.

        Args:
            data_args: Data arguments object containing:
                - rec_resolution: Target image size
                - patch_size: Size of patches for ViT
                - aug_scales: Scale range for data augmentation [min, max]
                - rescale: Whether to rescale images
                - rescale_aug: Whether to apply augmentation during rescaling
                - landscape_check: Whether to handle landscape vs portrait orientation
                - use_augs: Master flag to control all augmentations
                - training: Whether in training mode
        """
        super().__init__()
        self.data_args = data_args
        self.training = getattr(data_args, 'training', True)
           
        if input_use_augs is not None and rec_use_augs is not None:
            # Case 1: Explicit specification (backward compatibility)
            # Legacy code can still pass explicit augmentation settings
            self.strategy = StrategyFactory.from_explicit_settings(
                input_use_augs, rec_use_augs
            )
            self.input_use_augs = input_use_augs
            self.rec_use_augs = rec_use_augs
        
        elif input_use_augs is not None or rec_use_augs is not None:
            # Case 2: Partial specification (backward compatibility with defaults)
            # If only one parameter is specified, use strategy defaults for the other
            default_strategy = self._determine_strategy()
            self.input_use_augs = (
                input_use_augs if input_use_augs is not None 
                else default_strategy.get_input_augs(data_args)
            )
            self.rec_use_augs = (
                rec_use_augs if rec_use_augs is not None
                else default_strategy.get_rec_augs(data_args)
            )
            self.strategy = StrategyFactory.from_explicit_settings(
                self.input_use_augs, self.rec_use_augs
            )
        
        else:
            # Case 3: Strategy pattern (modern approach)
            # Automatically determine strategy based on training configuration
            self.strategy = self._determine_strategy()
            self.input_use_augs = self.strategy.get_input_augs(data_args)
            self.rec_use_augs = self.strategy.get_rec_augs(data_args)
        
        # Set all attributes from strategy configuration
        self._setup_attributes()

    def __len__(self):
        return self.len_train
    
    def _setup_attributes(self):
        config = self.strategy.get_full_config(self.data_args)
        for key, value in config.items():
            setattr(self, key, value)
        
        # Ensure target_image_shape exists
        if not hasattr(self, 'target_image_shape'):
            self.target_image_shape = np.array([384, 384])
    
    def _determine_strategy(self):
        
        return StrategyFactory.get_strategy(
            training=self.training,
            interleaved=getattr(self.data_args, 'interleaved_training', False),
            has_vqa=True  # default assumption, subclasses can override
        )

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Tuple containing (seq_index, img_per_seq, aspect_ratio)

        Returns:
            Dataset item as returned by get_data()
        """
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve data for a given sequence.

        Args:
            seq_index (int, optional): Index of the sequence
            seq_name (str, optional): Name of the sequence
            ids (list, optional): List of frame IDs
            aspect_ratio (float, optional): Target aspect ratio.

        Returns:
            Dataset-specific data

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.

        Args:
            aspect_ratio: Target aspect ratio

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        if self.force_square:
            return np.array([self.img_size, self.img_size])
        
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size
        
        # Ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size
        
        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
        use_augs=None,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.
        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.
            use_augs (bool, optional): Override augmentation setting. Critical for strategy pattern.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Check if augmentations are enabled (via use_augs)
        if use_augs is None:
            use_augs = getattr(self, 'use_augs', False)
    
        # Apply random scale augmentation only if use_augs is True
        if self.training and self.aug_scale and use_augs:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # Avoid random padding by capping at 1.0
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size

        # Move principal point to the image center and crop if necessary
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
        )

        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape

        # Handle landscape vs. portrait orientation (rotation augmentation)
        rotate_to_portrait = False
        if self.landscape_check and use_augs:  # Only if augmentations are enabled
            # Switch between landscape and portrait if necessary
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        # Resize images and update intrinsics
        if self.rescale:
            # Apply rescale_aug only if use_augs is True
            apply_rescale_aug = self.rescale_aug and use_augs
            
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=apply_rescale_aug
            )
        else:
            print("Not rescaling the images")

        # Ensure final crop to target shape
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
        )

        # Apply 90-degree rotation only if augmentations are enabled
        if rotate_to_portrait and use_augs:
            assert self.landscape_check
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )

        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def apply_image_only_augmentations(self, image):
        """
        Apply augmentations only to the image without affecting depth/camera params.
        
        This is used in special cases where VLM needs augmented images but
        reconstruction supervision doesn't (rare case, but supported for flexibility).
        
        Only applies:
        - Color jitter (brightness, contrast, saturation changes)
        - Horizontal flip (doesn't affect depth consistency)
        
        Does NOT apply:
        - Geometric transformations that would require depth/camera updates
        """
        if hasattr(self.data_args, 'color_jitter') and self.data_args.color_jitter is not None:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            pil_img = self.data_args.color_jitter(pil_img)
            image = np.array(pil_img)
        
        # Apply random horizontal flip (50% chance)
        if self.training and np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
        
        return image

    def process_frame_with_augmentation(
        self,
        image_orig,
        depth_map_orig,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        image_path,
    ):
        """
        Process a frame with strategy-based augmentation.
        
        This is the main entry point that handles the dual processing pipeline:
        1. Process for reconstruction supervision (may use augmentation)
        2. Process for VLM input (may use different augmentation settings)
        
        The key insight: VQA needs clean images for understanding, while
        reconstruction benefits from augmented data for robustness.
        """
        # Step 1: Process reconstruction supervision image
        # This uses rec_use_augs setting from the strategy
        rec_result = self._process_for_reconstruction(
            image_orig, depth_map_orig, extri_opencv, intri_opencv,
            original_size, target_image_shape, image_path
        )
        
        # Step 2: Process VLM input image
        # This uses input_use_augs setting from the strategy
        # May reuse reconstruction image if settings match (efficiency)
        vlm_image = self._process_for_vlm(
            image_orig, depth_map_orig, extri_opencv, intri_opencv,
            original_size, target_image_shape, image_path,
            rec_image=rec_result[0]  # Pass reconstruction image for potential reuse
        )
        
        # Step 3: Downsample to 192x192 for reconstruction # For now 192 is hardcoded
        image_192, depth_192, intri_192 = precise_downsample(
            rec_result[0], rec_result[1], rec_result[3], target_size=192
        )
        
        # Step 4: Compute 3D points at 192x192 resolution
        world_coords_192, cam_coords_192, point_mask_192 = (
            depth_to_world_coords_points(depth_192, rec_result[2], intri_192)
        )
        
        return {
            'image_384_vlm': vlm_image,      # VLM input at full resolution
            'image_192': image_192,          # Reconstruction at lower resolution
            'depth_192': depth_192,
            'extri_opencv': rec_result[2],
            'intri_192': intri_192,
            'cam_coords_192': cam_coords_192,
            'world_coords_192': world_coords_192,
            'point_mask_192': point_mask_192,
        }

    def _process_for_reconstruction(
        self,
        image_orig, depth_map_orig, extri_opencv, intri_opencv,
        original_size, target_image_shape, image_path
    ):
        """
        Uses rec_use_augs setting from the strategy.
        """
        return self.process_one_image(
            image_orig.copy(),
            depth_map_orig.copy(),
            extri_opencv.copy(),
            intri_opencv.copy(),
            original_size,
            target_image_shape,
            filepath=image_path,
            use_augs=self.rec_use_augs  # Use reconstruction augmentation setting
        )

    def _process_for_vlm(
        self,
        image_orig, depth_map_orig, extri_opencv, intri_opencv,
        original_size, target_image_shape, image_path,
        rec_image=None
    ):
        """
        Process VLM input data with appropriate augmentation strategy.
        
        Three cases handled:
        1. Same settings as reconstruction - reuse processed image (efficiency)
        2. VLM needs light augmentation only - apply image-only transforms
        3. Different settings - process separately
        
        Args:
            rec_image: Reconstruction image that may be reused if settings match
        """
        # Case 1: Same augmentation settings - reuse reconstruction image
        if self.input_use_augs == self.rec_use_augs and rec_image is not None:
            return rec_image
        
        # Case 2: VLM rgb input needs augmentation but rec does not (probably not used)
        if self.input_use_augs and not self.rec_use_augs:
            # Get base image without augmentation
            base_result = self.process_one_image(
                image_orig.copy(),
                depth_map_orig.copy(),
                extri_opencv.copy(),
                intri_opencv.copy(),
                original_size,
                target_image_shape,
                filepath=image_path,
                use_augs=False
            )
            # Apply only image-level augmentations
            return self.apply_image_only_augmentations(base_result[0])
        
        # Case 3: VLM needs different augmentation settings
        # VLM needs clean, reconstruction needs augmented
        vlm_result = self.process_one_image(
            image_orig.copy(),
            depth_map_orig.copy(),
            extri_opencv.copy(),
            intri_opencv.copy(),
            original_size,
            target_image_shape,
            filepath=image_path,
            use_augs=self.input_use_augs  # Use VLM augmentation setting
        )
        return vlm_result[0]

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        Sample a set of IDs from a sequence close to a given start index.
        You can specify the range either as a ratio of the number of input IDs
        or as a fixed integer window.

        Args:
            ids (list): Initial list of IDs. The first element is used as the anchor.
            full_seq_num (int): Total number of items in the full sequence.
            expand_ratio (float, optional): Factor by which the number of IDs expands
                around the start index. Default is 2.0 if neither expand_ratio nor
                expand_range is provided.
            expand_range (int, optional): Fixed number of items to expand around the
                start index. If provided, expand_ratio is ignored.

        Returns:
            numpy.ndarray: Array of sampled IDs, with the first element being the
                original start index.

        Examples:
            # Using expand_ratio (default behavior)
            # If ids=[100,101,102] and full_seq_num=200, with expand_ratio=2.0,
            # expand_range = int(3 * 2.0) = 6, so IDs sampled from [94...106] (if boundaries allow).

            # Using expand_range directly
            # If ids=[100,101,102] and full_seq_num=200, with expand_range=10,
            # IDs are sampled from [90...110] (if boundaries allow).

        Raises:
            ValueError: If no IDs are provided.
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0  # Default behavior

        total_ids = len(ids)
        start_idx = ids[0]

        # Determine the actual expand_range
        if expand_range is None:
            # Use ratio to determine range
            expand_range = int(total_ids * expand_ratio)

        # Calculate valid boundaries
        low_bound = max(0, start_idx - expand_range)
        high_bound = min(full_seq_num, start_idx + expand_range)

        # Create the valid range of indices
        valid_range = np.arange(low_bound, high_bound)

        # Sample 'total_ids - 1' items, because we already have the start_idx
        sampled_ids = np.random.choice(
            valid_range,
            size=(total_ids - 1),
            replace=True,   # Accept duplicate samples
        )

        # Insert the start_idx at the beginning
        result_ids = np.insert(sampled_ids, 0, start_idx)

        return result_ids