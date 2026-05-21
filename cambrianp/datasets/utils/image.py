import cv2
from PIL import Image
import numpy as np


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with PIL, maintaining behavior compatible with opencv."""
    
    # Special handling for EXR files still using OpenCV
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
        img = cv2.imread(path, options)
        if img is None:
            raise IOError(f"Could not load image={path} with {options=}")
        return img
    
    # For other file types, use PIL but match OpenCV's behavior
    try:
        pil_img = Image.open(path)
        
        # Handle different image types
        if options == cv2.IMREAD_GRAYSCALE:
            # Convert to grayscale if requested
            if pil_img.mode != 'L':
                pil_img = pil_img.convert('L')
            img = np.array(pil_img)
        
        elif options == cv2.IMREAD_UNCHANGED or options == -1:
            # Preserve original format (for depth images)
            if pil_img.mode in ['I', 'F', 'L']:  # 32-bit integer, 32-bit float, 8-bit
                img = np.array(pil_img)
            elif pil_img.mode in ['I;16', 'I;16L', 'I;16B']:
                # 16-bit - convert to uint16 as OpenCV would
                img = np.array(pil_img).astype(np.uint16)
            else:
                # Regular RGB image - convert to BGR to match OpenCV
                img = np.array(pil_img)
                if img.ndim == 3 and img.shape[2] == 3:
                    # Convert RGB to BGR to match OpenCV behavior
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        else:  # Default color handling (IMREAD_COLOR)
            # Ensure we have a color image
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            # Convert RGB to BGR to match OpenCV behavior
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if img is None:
            raise IOError(f"Could not load image={path} with PIL")
        
        return img
        
    except Exception as e:
        # Log the error
        print(f"PIL loading failed with error: {e}")
        # Try OpenCV as fallback (might still fail with libpng error)
        img = cv2.imread(path, options)
        if img is None:
            raise IOError(f"Could not load image={path} with {options=}")
        return img