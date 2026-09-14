import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

def polygons2mask(img_shape, polygons, scale=1):
    """
    Creates a binary (0 or 1) segmentation mask with smoothed edges 
    from polygon coordinates.

    Parameters:
    - img_shape (tuple): The (height, width) of the original image.
    - polygons (list): A list of polygon coordinate lists.

    Returns:
    - np.ndarray: A NumPy array (H, W) containing the binary mask (dtype=np.uint8, 
                  values are 0 or 1).
    """
    h, w = img_shape

    # 1. Create a blank binary mask (using PIL)
    mask = Image.new("L", (w, h), 0) # "L" (Luminance) - 8 bit unsigned int 0 - 255
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        # Flatten the list of [x, y] points into a tuple of (x1, y1, x2, y2, ...)
        flat_poly = [int(coord * scale)
                     for point in poly
                     	for coord in point]
        
        if len(flat_poly) >= 6:
            # Fill with max brightness (255)
            draw.polygon(flat_poly, fill=255, outline=255)

    # Convert PIL Image (0 or 255) directly to NumPy's uint8 (0 or 255).
    mask_np = np.array(mask, dtype=np.uint8)
    
    # (0 or 255) -> (0 or 1)
    binary_mask = (mask_np > 0).astype(np.uint8)
    
    return binary_mask

def create_binary_smoothed_mask(img_shape, polygons, scale=1, sigma=3, threshold=0.5):
    """
    Creates a binary (0 or 1) segmentation mask with smoothed edges 
    from polygon coordinates.

    Parameters:
    - img_shape (tuple): The (height, width) of the original image.
    - polygons (list): A list of polygon coordinate lists.
    - sigma (float): Standard deviation for the Gaussian kernel (controls blur strength).
    - threshold (float): The cutoff value (0.0 to 1.0) to convert the 
                         smoothed float mask into a binary mask (0 or 1).

    Returns:
    - np.ndarray: A NumPy array (H, W) containing the binary mask (dtype=np.uint8, 
                  values are 0 or 1).
    """
    h, w = img_shape

    # 1. Create a blank binary mask (using PIL)
    mask = Image.new("L", (w, h), 0) # "L" (Luminance) - 8 bit unsigned int 0 - 255
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        # Flatten the list of [x, y] points into a tuple of (x1, y1, x2, y2, ...)
        flat_poly = [coord * scale
                     for point in poly
                     	for coord in point]
        
        if len(flat_poly) >= 6:
            # Fill with max brightness (255)
            draw.polygon(flat_poly, fill=255, outline=255)

    # Convert (0-255) -> (0.0 or 1.0)
    mask_np = np.array(mask, dtype=np.float32) / 255.0

    # 2. Apply Gaussian Blur (results in soft-boundary float values)
    smoothed_float_mask = gaussian_filter(mask_np, sigma=sigma)
    
    # 3. Apply Thresholding to enforce binary values
    # Pixels where the blurred value is >= threshold are set to 1, others to 0.
    binary_mask = (smoothed_float_mask >= threshold).astype(np.uint8)
    
    return binary_mask