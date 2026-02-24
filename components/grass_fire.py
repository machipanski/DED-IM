"""
Grass Fire segmentation algorithms for labeled images.
Implements different strategies for iterative region expansion with boundary detection.
"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import disk
from components import morphology_tools as mt


def grass_fire_simple(labeled_img: np.ndarray, max_iterations: int = None) -> np.ndarray:
    """
    Simple grass fire algorithm: iteratively dilates each label and stops 
    when regions meet.
    
    Args:
        labeled_img: Input labeled image (0 = background, >0 = region labels)
        max_iterations: Maximum dilation iterations (None = until convergence)
    
    Returns:
        Segmented image where each pixel belongs to nearest label
    """
    result = labeled_img.copy()
    if max_iterations is None:
        max_iterations = int(np.max(labeled_img.shape) * 1.5)
    
    for iteration in range(max_iterations):
        new_result = result.copy()
        
        # For each unique label, try to expand
        for label in np.unique(labeled_img):
            if label == 0:  # Skip background
                continue
            
            # Get current region
            current_region = result == label
            
            # Dilate by 1 pixel
            dilated = binary_dilation(current_region, iterations=1)
            
            # Only expand into background or unlabeled regions
            expansion_mask = dilated & (result == 0)
            new_result[expansion_mask] = label
        
        # Check for convergence
        if np.array_equal(new_result, result):
            break
        
        result = new_result
    
    return result


def grass_fire_with_priority(labeled_img: np.ndarray, 
                             max_iterations: int = None,
                             priority_by_size: bool = True) -> np.ndarray:
    """
    Grass fire with priority: larger regions expand first (or vice versa).
    This prevents small regions from being completely overtaken.
    
    Args:
        labeled_img: Input labeled image
        max_iterations: Maximum dilation iterations
        priority_by_size: If True, larger regions expand first
    
    Returns:
        Segmented image
    """
    result = labeled_img.copy()
    if max_iterations is None:
        max_iterations = int(np.max(labeled_img.shape) * 1.5)
    
    # Calculate region sizes
    unique_labels = np.unique(labeled_img)
    unique_labels = unique_labels[unique_labels != 0]
    
    if priority_by_size:
        sizes = [(np.sum(labeled_img == label), label) for label in unique_labels]
        sizes.sort(reverse=True)
        expansion_order = [label for _, label in sizes]
    else:
        expansion_order = list(unique_labels)
    
    for iteration in range(max_iterations):
        new_result = result.copy()
        expanded = False
        
        for label in expansion_order:
            current_region = result == label
            dilated = binary_dilation(current_region, iterations=1)
            expansion_mask = dilated & (result == 0)
            
            if np.any(expansion_mask):
                expanded = True
                new_result[expansion_mask] = label
        
        if not expanded:
            break
        
        result = new_result
    
    return result


def grass_fire_with_distance_weighting(labeled_img: np.ndarray,
                                      kernel_size: int = 3) -> np.ndarray:
    """
    Grass fire using distance transform: faster convergence.
    Each pixel is assigned to the nearest labeled region.
    
    Args:
        labeled_img: Input labeled image
        kernel_size: Structuring element size for dilation
    
    Returns:
        Segmented image
    """
    # Initialize result
    result = labeled_img.copy()
    background_mask = labeled_img == 0
    
    if not np.any(background_mask):
        return result
    
    kernel = disk(kernel_size // 2) if kernel_size > 1 else np.array([[1]])
    
    # Iteratively dilate until background is filled
    while np.any(result == 0):
        new_result = result.copy()
        
        for label in np.unique(labeled_img):
            if label == 0:
                continue
            
            current_region = new_result == label
            dilated = binary_dilation(current_region, footprint=kernel)
            expansion_mask = dilated & (new_result == 0)
            new_result[expansion_mask] = label
        
        if np.array_equal(new_result, result):
            break
        
        result = new_result
    
    return result


def grass_fire_watershed_alternative(labeled_img: np.ndarray) -> np.ndarray:
    """
    Use watershed as alternative: often faster for grass fire segmentation.
    The labeled regions act as markers and expand via watershed.
    
    Args:
        labeled_img: Input labeled image (markers)
    
    Returns:
        Segmented image via watershed
    """
    # Create an image to segment: 0 where labeled, high values where background
    segmentation_input = np.where(labeled_img == 0, 1, 0).astype(float)
    
    # Apply watershed with labeled_img as markers
    result = watershed(segmentation_input, markers=labeled_img, mask=(labeled_img != 0) | (segmentation_input > 0))
    
    return result


def grass_fire_level_set(labeled_img: np.ndarray, 
                        max_iterations: int = None,
                        dilation_rate: int = 1) -> np.ndarray:
    """
    Level-set style grass fire: smoother expansion with configurable rate.
    
    Args:
        labeled_img: Input labeled image
        max_iterations: Maximum iterations
        dilation_rate: Pixels to expand per iteration
    
    Returns:
        Segmented image
    """
    result = labeled_img.copy()
    if max_iterations is None:
        max_iterations = int(np.max(labeled_img.shape) * 1.5)
    
    kernel = disk(dilation_rate) if dilation_rate > 1 else np.array([[1]])
    
    for iteration in range(max_iterations):
        new_result = result.copy()
        expanded = False
        
        for label in np.unique(labeled_img):
            if label == 0:
                continue
            
            current_region = result == label
            dilated = binary_dilation(current_region, footprint=kernel, iterations=1)
            expansion_mask = dilated & (result == 0)
            
            if np.any(expansion_mask):
                expanded = True
                new_result[expansion_mask] = label
        
        if not expanded:
            break
        
        result = new_result
    
    return result


def grass_fire_with_barrier(labeled_img: np.ndarray,
                           barrier_img: np.ndarray,
                           max_iterations: int = None) -> np.ndarray:
    """
    Grass fire that respects barrier regions (e.g., walls, edges).
    Regions cannot expand through barriers.
    
    Args:
        labeled_img: Input labeled image
        barrier_img: Binary image where 1 = barrier, 0 = passable
        max_iterations: Maximum iterations
    
    Returns:
        Segmented image respecting barriers
    """
    result = labeled_img.copy()
    if max_iterations is None:
        max_iterations = int(np.max(labeled_img.shape) * 1.5)
    
    for iteration in range(max_iterations):
        new_result = result.copy()
        expanded = False
        
        for label in np.unique(labeled_img):
            if label == 0:
                continue
            
            current_region = result == label
            dilated = binary_dilation(current_region, iterations=1)
            
            # Only expand where: dilated AND no barrier AND no label
            expansion_mask = dilated & (~barrier_img) & (result == 0)
            
            if np.any(expansion_mask):
                expanded = True
                new_result[expansion_mask] = label
        
        if not expanded:
            break
        
        result = new_result
    
    return result


def grass_fire_with_morphology(labeled_img: np.ndarray,
                               max_iterations: int = None,
                               kernel_size: int = 3) -> np.ndarray:
    """
    Grass fire using morphological operations for smoother results.
    Uses erosion/dilation based reconstruction.
    
    Args:
        labeled_img: Input labeled image
        max_iterations: Maximum iterations
        kernel_size: Size of morphological kernel
    
    Returns:
        Segmented image
    """
    result = labeled_img.copy()
    if max_iterations is None:
        max_iterations = int(np.max(labeled_img.shape) * 1.5)
    
    kernel = disk(kernel_size // 2) if kernel_size > 1 else np.array([[1]])
    
    for iteration in range(max_iterations):
        new_result = result.copy()
        
        # Dilate each label
        for label in np.unique(labeled_img):
            if label == 0:
                continue
            
            current_region = result == label
            dilated = mt.dilation(current_region.astype(np.uint8), kernel_size=kernel_size)
            expansion_mask = (dilated > 0) & (result == 0)
            new_result[expansion_mask] = label
        
        if np.array_equal(new_result, result):
            break
        
        result = new_result
    
    return result


# Example usage comparison
if __name__ == "__main__":
    # Create a simple labeled image for testing
    test_img = np.zeros((100, 100), dtype=np.int32)
    test_img[20:30, 20:30] = 1  # Region 1
    test_img[70:80, 70:80] = 2  # Region 2
    test_img[20:30, 70:80] = 3  # Region 3
    test_img[70:80, 20:30] = 4  # Region 4
    
    print("Testing different grass fire algorithms...")
    print(f"Original labeled regions: {np.unique(test_img)}")
    
    # Test each method
    result1 = grass_fire_simple(test_img)
    print(f"Simple grass fire - filled pixels: {np.sum(result1 > 0)}")
    
    result2 = grass_fire_with_priority(test_img, priority_by_size=True)
    print(f"Priority-based grass fire - filled pixels: {np.sum(result2 > 0)}")
    
    result3 = grass_fire_with_distance_weighting(test_img)
    print(f"Distance-weighted grass fire - filled pixels: {np.sum(result3 > 0)}")
    
    result4 = grass_fire_level_set(test_img, dilation_rate=2)
    print(f"Level-set grass fire - filled pixels: {np.sum(result4 > 0)}")
