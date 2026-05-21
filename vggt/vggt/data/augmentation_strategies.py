from abc import ABC, abstractmethod

class AugmentationStrategy(ABC):
    """
    Base class for augmentation strategies.
    
    This class defines the interface for different augmentation strategies used in
    training. Each strategy determines whether to apply augmentations to:
    1. Input images (sent to VLM for understanding)
    2. Reconstruction supervision images (used for 3D reconstruction loss)
    """
    
    @abstractmethod
    def get_input_augs(self, data_args) -> bool:
        """Whether to apply augmentations to VLM input images"""
        pass
    
    @abstractmethod
    def get_rec_augs(self, data_args) -> bool:
        """Whether to apply augmentations to reconstruction supervision"""
        pass
    
    def get_full_config(self, data_args) -> dict:
        """
        Get complete configuration including all hyperparameters.
        
        Returns a dictionary with all settings needed for image processing:
        - img_size: Target image resolution
        - patch_size: Vision transformer patch size
        - force_square: Whether to force square aspect ratio
        - use_augs: Legacy augmentation flag (for backward compatibility)
        - rescale: Whether to resize images
        - aug_scale: Range for random scaling augmentation
        - rescale_aug: Whether to apply augmentation during rescaling
        - landscape_check: Whether to handle landscape/portrait orientation
        """
        base_config = {
            'img_size': data_args.rec_resolution,
            'patch_size': getattr(data_args, 'patch_size', 14),
            'force_square': True,
            'use_augs': getattr(data_args, 'use_augs', False),
            'rescale': getattr(data_args, 'rescale', True),
        }
        
        strategy_config = self._get_strategy_specific_config(data_args)
        base_config.update(strategy_config)
        
        return base_config
    
    @abstractmethod
    def _get_strategy_specific_config(self, data_args) -> dict:
        """Get strategy-specific configuration parameters"""
        pass
    
    def get_description(self) -> str:
        """Get human-readable description of the strategy"""
        return self.__class__.__name__


class CustomStrategy(AugmentationStrategy):
    # Normally not used, but kept for backward compatibility
    def __init__(self, input_use_augs, rec_use_augs):
        self._input_use_augs = input_use_augs
        self._rec_use_augs = rec_use_augs
    
    def get_input_augs(self, data_args) -> bool:
        return self._input_use_augs
    
    def get_rec_augs(self, data_args) -> bool:
        return self._rec_use_augs
    
    def _get_strategy_specific_config(self, data_args) -> dict:
        any_aug = self._input_use_augs or self._rec_use_augs
        
        # Maintain original logic for backward compatibility
        if getattr(data_args, 'use_augs', False) or any_aug:
            aug_scale = getattr(data_args, 'aug_scales', [0.8, 1.2])
        else:
            aug_scale = [1.0, 1.0]
        
        return {
            'aug_scale': aug_scale,
            'rescale_aug': getattr(data_args, 'rescale_aug', False) and any_aug,
            'landscape_check': getattr(data_args, 'landscape_check', False) and any_aug,
        }


class StandardVQAStrategy(AugmentationStrategy):
    """
    Standard VQA training strategy.
    """
    
    def get_input_augs(self, data_args) -> bool:
        # VQA images should be clean for accurate understanding
        return getattr(data_args, 'vqa_image_augs', False)
    
    def get_rec_augs(self, data_args) -> bool:
        # Reconstruction benefits from augmentation
        return getattr(data_args, 'use_augs', True)
    
    def _get_strategy_specific_config(self, data_args) -> dict:
        rec_aug = self.get_rec_augs(data_args)
        
        # Enable augmentation parameters only if reconstruction uses augmentation
        if getattr(data_args, 'use_augs', False) or rec_aug:
            aug_scale = getattr(data_args, 'aug_scales', [0.8, 1.2])
        else:
            aug_scale = [1.0, 1.0]
        
        return {
            'aug_scale': aug_scale,
            'rescale_aug': getattr(data_args, 'rescale_aug', False) and rec_aug,
            'landscape_check': getattr(data_args, 'landscape_check', False) and rec_aug,
        }


class InterleavedVQAStrategy(AugmentationStrategy):
    """
    Interleaved training - VQA mode.
    """
    
    def get_input_augs(self, data_args) -> bool:
        return getattr(data_args, 'vqa_image_augs', False)
    
    def get_rec_augs(self, data_args) -> bool:
        return getattr(data_args, 'vqa_rec_sup_augs', True)
    
    def _get_strategy_specific_config(self, data_args) -> dict:
        rec_aug = self.get_rec_augs(data_args)
        
        if getattr(data_args, 'use_augs', False) or rec_aug:
            aug_scale = getattr(data_args, 'aug_scales', [0.8, 1.2])
        else:
            aug_scale = [1.0, 1.0]
        
        return {
            'aug_scale': aug_scale,
            'rescale_aug': getattr(data_args, 'rescale_aug', False) and rec_aug,
            'landscape_check': getattr(data_args, 'landscape_check', False) and rec_aug,
        }


class InterleavedRecStrategy(AugmentationStrategy):
    """
    Interleaved training - Reconstruction mode.
    """
    
    def get_input_augs(self, data_args) -> bool:
        return getattr(data_args, 'rec_image_augs', True)
    
    def get_rec_augs(self, data_args) -> bool:
        return getattr(data_args, 'rec_rec_sup_augs', True)
    
    def _get_strategy_specific_config(self, data_args) -> dict:
        any_aug = self.get_input_augs(data_args) or self.get_rec_augs(data_args)
        
        if getattr(data_args, 'use_augs', False) or any_aug:
            aug_scale = getattr(data_args, 'aug_scales', [0.8, 1.2])
        else:
            aug_scale = [1.0, 1.0]
        
        return {
            'aug_scale': aug_scale,
            'rescale_aug': getattr(data_args, 'rescale_aug', False) and any_aug,
            'landscape_check': getattr(data_args, 'landscape_check', False) and any_aug,
        }


class NoAugmentationStrategy(AugmentationStrategy):
    
    def get_input_augs(self, data_args) -> bool:
        return False
    
    def get_rec_augs(self, data_args) -> bool:
        return False
    
    def _get_strategy_specific_config(self, data_args) -> dict:
        return {
            'aug_scale': [1.0, 1.0],  # No random scaling
            'rescale_aug': False,      # No rescale augmentation
            'landscape_check': False,  # No random rotation
        }


class StrategyFactory:
    
    @staticmethod
    def get_strategy(training, interleaved, has_vqa):
        """    
        Decision flow:
        1. If not training -> NoAugmentationStrategy (deterministic evaluation)
        2. If not interleaved -> StandardVQAStrategy (regular VQA training)
        3. If interleaved and has VQA -> InterleavedVQAStrategy (VQA in mixed batch)
        4. If interleaved and no VQA -> InterleavedRecStrategy (Rec in mixed batch)
        """
        if not training:
            return NoAugmentationStrategy()
        
        if not interleaved:
            return StandardVQAStrategy()
        
        if has_vqa:
            return InterleavedVQAStrategy()
        else:
            return InterleavedRecStrategy()
    
    @staticmethod
    def from_explicit_settings(input_use_augs, rec_use_augs):
        return CustomStrategy(input_use_augs, rec_use_augs)