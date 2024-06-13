import cv2
import numpy as np
import ploters
from components import morphology_tools as mt
from skimage.morphology import disk
from scipy.ndimage import distance_transform_edt
from skimage import morphology as skmorph
from components import images_tools as it


def create_prune_divide_skel(original_img: np.ndarray, size_prune):
    skel = skmorph.skeletonize(original_img.astype(bool))
    skel = skel.astype(np.uint16)
    dist = distance_transform_edt(original_img)
    sem_galhos, segmented_img, segment_objects = prune(
        skel_img=skel.astype(np.uint16),
        size=size_prune
    )
    if np.sum(sem_galhos) == 0:
        sem_galhos = skel
        segmented_img, segment_objects = segment_skeleton(skel)
    return sem_galhos, dist, segment_objects


def prune(skel_img: np.ndarray, size=0, mask=None):
    """Prune the ends of skeletonized segments.
    The pruning algorithm proposed by https://github.com/karnoldbio
    Segments a skeleton into discrete pieces, prunes off all segments less than or
    equal to user specified size. Returns the remaining objects as a list and the
    pruned skeleton.
    Inputs:
    skel_img    = Skeletonized image
    size        = Size to get pruned off each branch
    mask        = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
    Returns:
    pruned_img      = Pruned image
    segmented_img   = Segmented debugging image
    segment_objects = List of contours
    :param skel_img: numpy.ndarray
    :param size: int
    :param mask: numpy.ndarray
    :return pruned_img: numpy.ndarray
    :return segmented_img: numpy.ndarray
    :return segment_objects: list
    """
    pruned_img = skel_img.copy()
    cleaned_img = _iterative_prune(pruned_img,2)
    _, objects = segment_skeleton(cleaned_img)
    kept_segments = []
    removed_segments = []
    if size > 0:
        # If size>0 then check for segments that are smaller than size pixels long
        # Sort through segments since we don't want to remove primary segments
        secondary_objects, _, BBB = segment_sort(cleaned_img, objects)
        # Keep segments longer than specified size
        for i in range(0, len(secondary_objects)):
            if len(secondary_objects[i]) > size:
                kept_segments.append(secondary_objects[i])
            else:
                removed_segments.append(secondary_objects[i])
        # Draw the contours that got removed
        removed_barbs = np.zeros(cleaned_img.shape[:2], np.uint16)
        cv2.drawContours(removed_barbs, removed_segments, -1, 1, 1, lineType=8)
        # Subtract all short segments from the skeleton image
        pruned_img = it.image_subtract(cleaned_img, removed_barbs)
        pruned_img = _iterative_prune(pruned_img, 3)
    # cleaned_img = _iterative_prune(pruned_img,2)
    segmented_img, segment_objects = segment_skeleton(pruned_img, mask)
    return pruned_img, segmented_img, segment_objects

def _iterative_prune(skel_img: np.ndarray, size):
    """Iteratively remove endpoints (tips) from a skeletonized image.
    The pruning algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699
    "Prunes" barbs off a skeleton.
    Inputs:
    skel_img    = Skeletonized image
    size        = Size to get pruned off each branch
    Returns:
    pruned_img  = Pruned image
    :param skel_img: numpy.ndarray
    :param size: int
    :return pruned_img: numpy.ndarray
    """
    pruned_img = skel_img.copy()
    # Iteratively remove endpoints (tips) from a skeleton
    for _ in range(0, size):
        endpoints = find_tips(pruned_img)
        pruned_img = it.image_subtract(pruned_img, endpoints)
    return pruned_img
    
def segment_skeleton(skel_img: np.ndarray, mask=None):
    """Segment a skeleton image into pieces.
    Inputs:
    skel_img         = Skeletonized image
    mask             = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
    Returns:
    segmented_img       = Segmented debugging image
    segment_objects     = list of contours
    :param skel_img: numpy.ndarray
    :param mask: numpy.ndarray
    :return segmented_img: numpy.ndarray
    :return segment_objects: list
    """
    # Find branch points
    bp = find_branch_pts(skel_img)
    bp = mt.dilation(bp, kernel_size=3)
    # Subtract from the skeleton so that leaves are no longer connected
    segments = it.image_subtract(skel_img, bp)
    # Gather contours of leaves
    # segment_objects, _ = cv2.findContours(bin_img=segments)<e
    segment_objects = mt.detect_contours(segments)
    # Color each segment a different color, do not used a previously saved color scale
    rand_color = ploters.color_palette(num=len(segment_objects), saved=False)
    if mask is None:
        segmented_img = skel_img.copy()
    else:
        segmented_img = mask.copy()
    segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2RGB)
    for i, _ in enumerate(segment_objects):
        cv2.drawContours(
            segmented_img, segment_objects, i, rand_color[i], 4, lineType=8
        )
    return segmented_img, segment_objects


def find_tips(skel_img: np.ndarray, mask=None, label=None):
    """Find tips in skeletonized image.
    The endpoints algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699
    Inputs:
    skel_img    = Skeletonized image
    mask        = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
    label       = Optional label parameter, modifies the variable name of
                  observations recorded (default = pcv.params.sample_label).
    Returns:
    tip_img   = Image with just tips, rest 0
    :param skel_img: numpy.ndarray
    :param mask: numpy.ndarray
    :param label: str
    :return tip_img: numpy.ndarray
    """
    # In a kernel: 1 values line up with 255s, -1s line up with 0s, and 0s correspond to dont care
    endpoint1 = np.array([[-1, -1, -1], [-1, 1, -1], [0, 1, 0]])
    endpoint2 = np.array([[-1, -1, -1], [-1, 1, 0], [-1, 0, 1]])
    endpoint3 = np.rot90(endpoint1)
    endpoint4 = np.rot90(endpoint2)
    endpoint5 = np.rot90(endpoint3)
    endpoint6 = np.rot90(endpoint4)
    endpoint7 = np.rot90(endpoint5)
    endpoint8 = np.rot90(endpoint6)
    endpoints = [
        endpoint1,
        endpoint2,
        endpoint3,
        endpoint4,
        endpoint5,
        endpoint6,
        endpoint7,
        endpoint8,
    ]
    tip_img = np.zeros(skel_img.shape[:2], dtype=int)
    for endpoint in endpoints:
        tip_img = np.logical_or(
            cv2.morphologyEx(
                skel_img.astype(np.uint8),
                op=cv2.MORPH_HITMISS,
                kernel=endpoint,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            ),
            tip_img,
        )
    tip_img = tip_img.astype(np.uint16)
    return tip_img


def find_branch_pts(skel_img: np.ndarray):
    """Find branch points in a skeletonized image.
    The branching algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699
    Inputs:
    skel_img    = Skeletonized image
    mask        = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
    label        = Optional label parameter, modifies the variable name of
                   observations recorded (default = pcv.params.sample_label).
    Returns:
    branch_pts_img = Image with just branch points, rest 0
    :param skel_img: numpy.ndarray
    :param mask: np.ndarray
    :param label: str
    :return branch_pts_img: numpy.ndarray
    """
    # In a kernel: 1 values line up with 255s, -1s line up with 0s, and 0s correspond to don't care
    # T like branch points
    t1 = np.array([[-1, 1, -1], [1, 1, 1], [-1, -1, -1]])
    t2 = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, -1]])
    t3 = np.rot90(t1)
    t4 = np.rot90(t2)
    t5 = np.rot90(t3)
    t6 = np.rot90(t4)
    t7 = np.rot90(t5)
    t8 = np.rot90(t6)
    # Y like branch points
    y1 = np.array([[1, -1, 1], [0, 1, 0], [0, 1, 0]])
    y2 = np.array([[-1, 1, -1], [1, 1, 0], [-1, 0, 1]])
    y3 = np.rot90(y1)
    y4 = np.rot90(y2)
    y5 = np.rot90(y3)
    y6 = np.rot90(y4)
    y7 = np.rot90(y5)
    y8 = np.rot90(y6)
    kernels = [t1, t2, t3, t4, t5, t6, t7, t8, y1, y2, y3, y4, y5, y6, y7, y8]
    branch_pts_img = np.zeros(skel_img.shape[:2], dtype=int)
    # Store branch points
    for kernel in kernels:
        branch_pts_img = np.logical_or(
            cv2.morphologyEx(
                skel_img.astype(np.uint8),
                op=cv2.MORPH_HITMISS,
                kernel=kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            ),
            branch_pts_img,
        )
    # Switch type to uint8 rather than bool
    branch_pts_img = branch_pts_img.astype(np.uint16)
    return branch_pts_img


def segment_sort(skel_img: np.ndarray, objects, mask=None, first_stem=True):
    """ MODIFICADO DE PLANT CV PARA DETERMINAR PRIMARI COMO CONECTADO PELAS DUAS PONTAS
    Sort segments from a skeletonized image into two categories: leaf objects and other objects.
    Inputs:
    skel_img          = Skeletonized image
    objects           = List of contours
    mask              = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
    first_stem        = (Optional) if True, then the first (bottom) segment always gets classified as stem
    Returns:
    labeled_img       = Segmented debugging image with lengths labeled
    secondary_objects = List of secondary segments (leaf)
    primary_objects   = List of primary objects (stem)
    :param skel_img: numpy.ndarray
    :param objects: list
    :param mask: numpy.ndarray
    :param first_stem: bool
    :return secondary_objects: list
    :return other_objects: list
    """
    secondary_objects = []
    primary_objects = []
    labeled_img = np.zeros(skel_img.shape[:2], np.uint16)
    tips_img = find_tips(skel_img)
    tips_img = mt.dilation(tips_img, kernel_size=3)
    # Loop through segment contours
    for i, cnt in enumerate(objects):
        segment_plot = np.zeros(skel_img.shape[:2], np.uint8)
        cv2.drawContours(segment_plot, objects, i, 255, 1, lineType=8)
        overlap_img = np.logical_and(segment_plot, tips_img)
        # The first contour is the base, and while it contains a tip, it isn't a leaf
        if np.sum(overlap_img) == 0 :
            primary_objects.append(cnt)
        # Sort segments
        else:
            secondary_objects.append(cnt)
    # Plot segments where green segments are leaf objects and fuschia are other objects
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2RGB)
    for i, cnt in enumerate(primary_objects):
        cv2.drawContours(labeled_img, primary_objects, i, (255, 0, 255), 4, lineType=8)
    for i, cnt in enumerate(secondary_objects):
        cv2.drawContours(labeled_img, secondary_objects, i, (0, 255, 0), 4, lineType=8)
    return secondary_objects, primary_objects, labeled_img
