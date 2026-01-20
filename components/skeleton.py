from __future__ import annotations
from email.headerregistry import Group
from typing import TYPE_CHECKING

import cv2
import numpy as np
import components.ploters as ploters
import random
from components import morphology_tools as mt, path_tools
from scipy.ndimage import distance_transform_edt
from skimage import morphology as skmorph
from components import images_tools as it
from components import points_tools as pt

if TYPE_CHECKING:
    from typing import List


def create_prune_skel(original_img: np.ndarray, size_prune=0, distance=0):
    """
    :param original_img: np.ndarray: binary image to be skeletonized
    :param size_prune: int: size to prune the skeleton branches
    :param distance: int: distance threshold to prune the skeleton branches
    :return sem_galhos: np.ndarray: pruned skeleton image
    :return dist: np.ndarray: distance transform of the original image
    :return segment_objects: list: list of contours of the pruned skeleton segments

    Creates, prunes and divides the skeleton of the original image
    """
    skel = skmorph.skeletonize(original_img.astype(bool))
    skel = skel.astype(np.uint16)
    dist = distance_transform_edt(original_img)
    sem_galhos, segmented_img, segment_objects = prune(
        skel_img=skel.astype(np.uint16),
        size=size_prune,
        distance=distance,
        dist_map=dist,
    )
    return sem_galhos, dist, segment_objects


def create_prune_divide_skel(original_img: np.ndarray, size_prune):
    """
    :param original_img: np.ndarray: binary image to be skeletonized
    :param size_prune: int: size to prune the skeleton branches
    :return sem_galhos: np.ndarray: pruned skeleton image
    :return dist: np.ndarray: distance transform of the original image
    :return segment_objects: list: list of contours of the pruned skeleton segments
    
    Creates, prunes and divides the skeleton of the original image
    """
    skel = skmorph.skeletonize(original_img.astype(bool))
    skel = skel.astype(np.uint16)
    dist = cv2.distanceTransform(original_img.astype(np.uint8), cv2.DIST_L2, 5)
    sem_galhos, segmented_img, segment_objects = prune(
        skel_img=skel.astype(np.uint16), size=size_prune
    )
    if np.sum(sem_galhos) == 0:
        sem_galhos, segmented_img, segment_objects = prune(
            skel_img=skel.astype(np.uint16), size=round(size_prune / 2)
        )
    if np.sum(sem_galhos) == 0:
        sem_galhos = skel
        segment_objects = segment_skeleton(skel, mask=None)
    skeleton_graph, trunks_img, segment_objects = path_tools.skel_to_graph(
        sem_galhos, 2
    )
    segment_objects = [pt.invert_x_y(list(seg)) for seg in segment_objects]
    return sem_galhos, dist, segment_objects


def create_prune_divide_normalize_skel(rest_of_picture: np.ndarray, path_radius: int):
    """
    :param rest_of_picture: np.ndarray: binary image to be skeletonized
    :param path_radius: int: radius of the path to normalize the skeleton
    :return norm_trunks: np.ndarray: list of normalized trunk images
    :return norm_dist_map: np.ndarray: normalized distance map
    :return medial_transform: np.ndarray: medial transform image

    Breaks the image into its MAT components - already normalized to the number
    of routes that fit in the parallel direction
    """
    sem_galhos, sem_galhos_dist, trunks = create_prune_divide_skel(
        rest_of_picture.astype(np.uint8), path_radius
    )
    medial_transform = sem_galhos * sem_galhos_dist
    trunks = [it.points_to_img(x, np.zeros_like(rest_of_picture)) for x in trunks]
    trunks = it.eliminate_duplicates(trunks)
    normalized_dist_map = sem_galhos_dist / path_radius
    normalized_trunks = [trunk * normalized_dist_map for trunk in trunks]
    return normalized_trunks, normalized_dist_map, medial_transform


def break_too_big_parts(
    normalized_trunks: List[np.ndarray], 
    normalized_dist_map: np.ndarray, 
    necks_max_paths: int,
):
    """
    :param normalized_trunks: List[np.ndarray]: list of normalized trunk images
    :param normalized_dist_map: np.ndarray: normalized distance map
    :param necks_max_paths: int: maximum width to be considered a neck
    :return minus_bigger_than_limmit: List[np.ndarray]: list of trunk images further divided

    Further divides the trunks to avoid overly large regions
    """
    minus_bigger_than_limmit = []
    for trunk in normalized_trunks:
        less_than_limmit = np.logical_and(trunk > 0, trunk <= necks_max_paths)
        divided, _, num = it.divide_by_connected(less_than_limmit)
        if divided != []:
            minus_bigger_than_limmit = minus_bigger_than_limmit + divided
    minus_bigger_than_limmit = [
        np.multiply(x, normalized_dist_map, dtype=np.float32)
        for x in minus_bigger_than_limmit
    ]
    ccc = minus_bigger_than_limmit[0]
    ddd = it.sum_imgs(minus_bigger_than_limmit)
    return minus_bigger_than_limmit


def filter_trunks_with_smaller_than(minus_bigger_than_limmit, necks_max_paths):
    """Returns only the trunks that have bottleneck sections"""
    n_trilhas_minima_by_trunk = [
        (np.unique(trunk))[1] for trunk in minus_bigger_than_limmit
    ]
    return [
        minus_bigger_than_limmit[i]
        for i, x in enumerate(n_trilhas_minima_by_trunk)
        if x < necks_max_paths
    ]


def reduce_origin(
    candidate: np.ndarray, necks_max_paths: int, norm_dist_map: np.ndarray
):
    """
    :param candidate: np.ndarray: image of the trunk
    :param necks_max_paths: int: maximum width to be considered a neck
    :param norm_dist_map: np.ndarray: image of the normalized distance map
    :return: np.ndarray: reduced trunk image

    Determines the start and end for each trunk, then reduces
    the margins until all its bottlenecks are encompassed, sometimes
    this process results in more than one segnet for esch trunk"""

    t_ends = pt.img_to_points(find_tips(candidate.astype(bool)))
    if len(t_ends) == 0:  # se for um ciclo fechado
        start_pnt = random.choice(pt.x_y_para_pontos(np.nonzero(candidate)))
        origin_chain = pt.invert_x_y(
            path_tools.make_a_chain(candidate.astype(bool), start_pnt)
        )
        origin_chain_img = it.points_to_img(origin_chain, np.zeros_like(candidate))
        new_ends = pt.img_to_points(find_tips(origin_chain_img.astype(bool)))
        origin_chain = path_tools.set_first_pt_in_seq(origin_chain, new_ends[0])
        candidate = np.multiply(candidate, origin_chain_img)
    else:  # se for um trunk aberto
        start_pnt = []
        origin_chain = pt.invert_x_y(
            path_tools.make_a_chain_open_segment(candidate.astype(bool), t_ends)
        )
    reduced_origin = np.logical_and(candidate != 0, candidate < necks_max_paths)
    if np.sum(reduced_origin) == 0:
        return np.zeros_like(candidate), []
    ends = pt.img_to_points(find_tips(reduced_origin.astype(bool)))
    new_origin = origin_chain.copy()
    count_up = 0
    count_down = -1
    start_flag = 0
    end_flag = 0
    while not (start_flag and end_flag):
        current_pt_1 = origin_chain[count_up]
        current_pt_2 = origin_chain[count_down]
        if current_pt_1 in ends:
            start_flag = 1
        else:
            new_origin.remove(current_pt_1)
            count_up += 1
        if current_pt_2 in ends:
            end_flag = 1
        else:
            new_origin.remove(current_pt_2)
            count_down -= 1
    reduced_origin = it.points_to_img(new_origin, np.zeros_like(candidate))
    separated, _, n = it.divide_by_connected(reduced_origin)
    if n > 1:
        reduced_origin = it.take_the_bigger_area(reduced_origin)
        print("   ERROR")
    return reduced_origin * norm_dist_map, origin_chain[0]


def prune(skel_img: np.ndarray, size=0, distance=0, dist_map=[], mask=None, it_prune=0):
    """Inputs:
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

    Prune the ends of skeletonized segments.
    The pruning algorithm proposed by https://github.com/karnoldbio
    Segments a skeleton into discrete pieces, prunes off all segments less than or
    equal to user specified size. Returns the remaining objects as a list and the
    pruned skeleton.
    """
    pruned_img = skel_img.copy()
    cleaned_img = pruned_img
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
    segmented_img, segment_objects = segment_skeleton(pruned_img, mask)
    if distance > 0:
        # If size>0 then check for segments that are smaller than size pixels long
        # Sort through segments since we don't want to remove primary segments
        secondary_objects, _, BBB = segment_sort(pruned_img, segment_objects)
        # Keep segments longer than specified size
        for i in range(0, len(secondary_objects)):
            pontos_obj = [list(x[0]) for x in secondary_objects[i]]
            obj_img = it.points_to_img(
                pt.invert_x_y(pontos_obj), np.zeros_like(pruned_img)
            )
            transform_secondary = obj_img * dist_map
            over_distance = transform_secondary > distance
            if np.sum(over_distance) > 0:
                removed_segments.append(secondary_objects[i])
            else:
                kept_segments.append(secondary_objects[i])
        # Draw the contours that got removed
        removed_barbs = np.zeros(cleaned_img.shape[:2], np.uint16)
        cv2.drawContours(removed_barbs, removed_segments, -1, 1, 1, lineType=8)
        # Subtract all short segments from the skeleton image
        pruned_img = it.image_subtract(pruned_img, removed_barbs)
        # pruned_img = _iterative_prune(pruned_img, 3)
    if it_prune > 0:
        pruned_img = _iterative_prune(pruned_img, it_prune)
    segmented_img, segment_objects = segment_skeleton(pruned_img, mask)
    return pruned_img, segmented_img, segment_objects


def _iterative_prune(skel_img: np.ndarray, size):
    """Iteratively remove endpoints (tips) from a skeletonized image.
    The pruning algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699
    Iteratively remove endpoints (tips) from a skeleton
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
    for _ in range(0, size):
        endpoints = find_tips(pruned_img)
        pruned_img = it.image_subtract(pruned_img, endpoints)
    return pruned_img


def segment_skeleton(skel_img: np.ndarray, mask=None):
    """Segment a skeleton image into pieces.

    Inputs:
    :param skel_img: numpy.ndarray
    :param mask: numpy.ndarray
    Returns:
    :return segmented_img: numpy.ndarray
    :return segment_objects: list of contours
    """
    # Find branch points
    bp = find_branch_pts(skel_img)
    bp = mt.dilation(bp, kernel_size=3)
    # Subtract from the skeleton so that leaves are no longer connected
    segments = it.image_subtract(skel_img, bp)
    # Gather contours of leaves
    segment_objects = mt.detect_contours(segments)
    # Color each segment a different color, do not used a previously saved color scale
    rand_color = ploters.color_palette(num=len(segment_objects), saved=False)
    if mask is None:
        segmented_img = skel_img.copy()
    else:
        segmented_img = mask.copy()
    segmented_img = cv2.cvtColor(segmented_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for i, _ in enumerate(segment_objects):
        cv2.drawContours(
            segmented_img, segment_objects, i, rand_color[i], 4, lineType=8
        )
    return segmented_img, segment_objects


def find_tips(skel_img: np.ndarray):
    """
    :param skel_img: numpy.ndarray = Skeletonized image
    :return tip_img: numpy.ndarray = Image with just tips, rest 0

    Find tips in skeletonized image.
    The endpoints algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699
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
    endpoint9 = np.array(
        [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]
    )  # TODO se der problema tira aqui
    endpoints = [
        endpoint1,
        endpoint2,
        endpoint3,
        endpoint4,
        endpoint5,
        endpoint6,
        endpoint7,
        endpoint8,
        endpoint9,
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
    """
    :param skel_img: numpy.ndarray = Skeletonized image
    :return branch_pts_img: numpy.ndarray = Image with just branch points
    Find branch points in a skeletonized image.
    The branching algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699
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


def segment_sort(skel_img: np.ndarray, objects: List):
    """
    :param skel_img: numpy.ndarray = Skeletonized image
    :param objects: list
    :return primary_objects: list = List of primary objects (stem)
    :return secondary_objects: list = List of secondary segments (leaf)
    :return labeled_img: numpy.ndarray = Segmented debugging image
    
    Modified from PlantCV (https://plantcv.readthedocs.io/en/stable/)
    Sort segments from a skeletonized image into two categories: leaf objects and other objects.
    """
    secondary_objects = []
    primary_objects = []
    labeled_img = np.zeros(skel_img.shape[:2], np.uint16)
    tips_img = find_tips(skel_img)
    tips_img = mt.dilation(tips_img, kernel_size=3)
    # Loop through segment contours
    for i, cnt in enumerate(objects):
        segment_plot = np.zeros(skel_img.shape[:2], np.uint8)
        if isinstance(objects, tuple):
            cv2.drawContours(segment_plot, objects, i, 255, 1, lineType=8)
        else:
            segment_plot = it.points_to_img(cnt, segment_plot)
        overlap_img = np.logical_and(segment_plot, tips_img)
        # The first contour is the base, and while it contains a tip, it isn't a leaf
        if np.sum(overlap_img) == 0:
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


def reconstruct_img_from_skeleton(medial_img):
    """
    Receives an image (2D numpy array) where each pixel indicates the radius of the circle.
    Returns a binary image with all circles filled.
    """
    output = np.zeros_like(medial_img, dtype=np.uint8)
    coords = np.argwhere(medial_img > 0)
    for y, x in coords:
        radius = int(round(medial_img[y, x]))
        if radius > 0:
            cv2.circle(output, (x, y), radius, 1, thickness=-1)
    return output


def close_contour_TW(
    reduced_continuous_origin: np.ndarray,
    initial_point: np.ndarray,
    trunk_number: int,
    island_img: np.ndarray,
    path_radius: int,
    base_frame: np.ndarray,
    n_trilhas_max: float,
):
    max_width = 2
    try:
        bridge_img, contour_elements, contour, extreme_points = close_bridge_contour(
            reduced_continuous_origin,
            max_width,
            island_img,
            path_radius,
            initial_point,
            base_frame,
        )
        if np.sum(bridge_img) > 0:
            y_mark = np.where(reduced_continuous_origin)[1][
                np.round(len(np.where(reduced_continuous_origin)))
            ]
            x_mark = np.where(reduced_continuous_origin)[0][
                np.round(len(np.where(reduced_continuous_origin)))
            ]
            origin_mark = [y_mark, x_mark, str(n_trilhas_max)]
            region = [
                f"TW_{trunk_number:03d}",
                bridge_img,
                reduced_continuous_origin,
                reduced_continuous_origin,
                n_trilhas_max,
                origin_mark,
                contour_elements,
                extreme_points,
            ]
            print("OK: closed contour")
        else:
            region = []
    except Exception:
        print("\033[3#m" + "Error: didn´t closed one contour" + "\033[0m")
        region = []
    return region


def close_contour_ZZB(
    reduced_continuous_origin: np.ndarray,
    initial_point: np.ndarray,
    trunk_number: int,
    rest_of_picture: np.ndarray,
    path_radius_bridg: int,
    base_frame: np.ndarray,
    necks_max_paths: float,
):
    try:
        bridge_img, contour_elements, contour, extreme_points = close_bridge_contour(
            reduced_continuous_origin,
            1.5 * necks_max_paths,  # TODO check this value
            rest_of_picture,
            path_radius_bridg,
            initial_point,
            base_frame,
        )
        if np.sum(bridge_img) > 0:
            y_mark = np.where(reduced_continuous_origin)[1][
                np.round(len(np.where(reduced_continuous_origin)))
            ]
            x_mark = np.where(reduced_continuous_origin)[0][
                np.round(len(np.where(reduced_continuous_origin)))
            ]
            origin_mark = [y_mark, x_mark, str("not used")]
            region = [
                f"ZB_{trunk_number:03d}",
                bridge_img,
                reduced_continuous_origin,
                reduced_continuous_origin,
                [],
                origin_mark,
                contour_elements,
                extreme_points,
            ]
            # bridge_obj.zigzag_bridges[-1].get_linked_offsets(offset_regions)
        print("   Closed a bridge")
    except Exception:
        print("   Bridge failed")
        region = []
    return region


def close_bridge_contour(
    trunk,
    max_accepted,
    rest_of_picture,
    path_radius_bridg,
    inicial_pnt,
    base_frame,
):
    def find_contours_around_origin(
        rest_of_picture, max_accepted, path_radius_bridg, trunk
    ):
        all_borders, all_borders_img = mt.detect_contours(
            rest_of_picture, return_img=True
        )
        area_pescocal = mt.dilation(
            trunk.astype(bool), kernel_size=(max_accepted * path_radius_bridg + 8)
        )
        overlap = np.add(area_pescocal, all_borders_img)
        lines_do_limite = overlap == 2
        _, labeled, labeled_n = it.divide_by_connected(lines_do_limite)
        if labeled_n == 1:
            print("   Special case: Only one line around origin")
            possible_c1_c2, counter_accepted, curvature_points = (
                path_tools.decompose_pol_cont_by_corners(
                    lines_do_limite, trunk, path_radius_bridg
                )
            )
            labeled = possible_c1_c2
            labeled_n = counter_accepted
        if labeled_n > 2:
            dists = []
            trunk_pts = pt.x_y_para_pontos(np.nonzero(trunk))
            trunk_center = pt.points_center(trunk_pts)
            trunk_center_pt, _ = pt.closest_point(trunk_center, trunk_pts)
            for l in np.arange(1, labeled_n + 1):
                line_pts = pt.x_y_para_pontos(np.nonzero(labeled == l))
                _, dist = pt.closest_point(trunk_center_pt, line_pts)
                dists.append(dist)
            lista_dist = dists.copy()
            idx1 = np.argmin(lista_dist)
            lista_dist[idx1] = 999999
            idx2 = np.argmin(lista_dist)
            line1 = labeled == idx1 + 1
            # ATENçÂO PARA NOVOS CASOS AQUI!!!!!!!
            line2 = labeled == idx2 + 1
        elif labeled_n == 2:
            line1 = labeled == 1
            line2 = labeled == 2
            return line1, line2
        else:
            print("   ERROR: No lines around origin")
            return np.zeros_like(trunk), np.zeros_like(trunk)
        points_trunk = pt.img_to_points(mt.hitmiss_ends_v2(trunk.astype(bool)))
        points_line1 = pt.img_to_points(mt.hitmiss_ends_v2(line1.astype(bool)))
        if len(points_line1) > 0:
            line1 = reduce_lines_overshoot(line1, points_trunk)
        points_line2 = pt.img_to_points(mt.hitmiss_ends_v2(line2.astype(bool)))
        if len(points_line2) > 0:
            line2 = reduce_lines_overshoot(line2, points_trunk)
        return line1, line2

    def close_area_from_lines(
        line1: np.ndarray,
        line2: np.ndarray,
        base_frame: np.ndarray,
        inicial_pnt: np.ndarray,
    ):
        """Given two lines, tries to close the area between them."""
        starts_and_ends1 = pt.x_y_para_pontos(
            np.where(find_tips(line1.astype(np.uint8)))
        )
        starts_and_ends2 = pt.x_y_para_pontos(
            np.where(find_tips(line2.astype(np.uint8)))
        )
        # Se alguma delas é um circulo fechado, interrompe perto da origem indicada poe Inicial_pt
        if len(starts_and_ends1) == 0:
            line_pts = pt.img_to_points(line1.astype(np.uint8))
            break_point, _ = pt.closest_point(inicial_pnt, line_pts)
            origin_chain = pt.invert_x_y(path_tools.make_a_chain(line1, break_point))
            line1 = it.points_to_img(origin_chain, np.zeros_like(line1))
            starts_and_ends1 = pt.x_y_para_pontos(
                np.where(find_tips(line1.astype(np.uint8)))
            )
        if len(starts_and_ends2) == 0:
            line_pts = pt.img_to_points(line2.astype(np.uint8))
            break_point, _ = pt.closest_point(inicial_pnt, line_pts)
            origin_chain = pt.invert_x_y(path_tools.make_a_chain(line2, break_point))
            line2 = it.points_to_img(origin_chain, np.zeros_like(line2))
            starts_and_ends2 = pt.x_y_para_pontos(
                np.where(find_tips(line2.astype(np.uint8)))
            )
        # Se houverem pontos demais por causa dos contours, faz uma poda
        if len(starts_and_ends1) > 2:
            line1, _, _ = prune(skel_img=line1, size=2)
            starts_and_ends1 = pt.x_y_para_pontos(
                np.where(find_tips(line1.astype(np.uint8)))
            )
        if len(starts_and_ends2) > 2:
            line2, _, _ = prune(skel_img=line2, size=2)
            starts_and_ends2 = pt.x_y_para_pontos(
                np.where(find_tips(line2.astype(np.uint8)))
            )
        # Se as duas lines coincidem o final e o começo
        if starts_and_ends1 == starts_and_ends2:
            line1 = line2
            pontos_fins = mt.hitmiss_ends_v2(line1)
            pontos_fins = pt.img_to_points(pontos_fins)
            if len(pontos_fins) == 2:
                linebaixo = linetopo = it.draw_line(
                    np.zeros(base_frame), starts_and_ends1[0], starts_and_ends1[1]
                )
                bridge_border = it.sum_imgs([line1, linetopo, line2, linebaixo]) >= 1
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(pontos_fins) > 2:
                bridge_border = it.draw_polyline(
                    np.zeros(base_frame), pontos_fins, closed=True
                )
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
                linetopo = linebaixo = np.zeros_like(line1)
            else:
                print("   Special case: no solution yet")
                # TODO: still need to find a workaround here
        else:
            unique_points = []
            for p in starts_and_ends1 + starts_and_ends2:
                if p not in unique_points:
                    unique_points.append(p)
            if len(unique_points) == 4:
                extreme_points = [
                    starts_and_ends1[0],
                    starts_and_ends1[1],
                    starts_and_ends2[1],
                    starts_and_ends2[0],
                ]
                if pt.intersects(
                    [extreme_points[0], extreme_points[2]],
                    [extreme_points[1], extreme_points[3]],
                ):
                    extreme_points = [
                        starts_and_ends1[0],
                        starts_and_ends1[1],
                        starts_and_ends2[0],
                        starts_and_ends2[1],
                    ]
                linetopo = it.draw_line(
                    np.zeros(base_frame), extreme_points[0], extreme_points[2]
                )
                linebaixo = it.draw_line(
                    np.zeros(base_frame), extreme_points[1], extreme_points[3]
                )
                bridge_border = it.sum_imgs([line1, linetopo, line2, linebaixo])
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(unique_points) == 2:
                fechamento1_pts = unique_points
                linetopo = it.draw_line(
                    np.zeros(base_frame), fechamento1_pts[0], fechamento1_pts[1]
                )
                linebaixo = np.zeros_like(linetopo)
                bridge_border = it.sum_imgs([line1, linetopo])
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(unique_points) == 0:
                if np.sum(line1) > 0:
                    fechamento1_pts = unique_points
                    linetopo = np.zeros_like(line1)
                    linebaixo = np.zeros_like(line1)
                    bridge_border = line1.copy()
                    bridge_img = it.fill_internal_area(
                        bridge_border, np.ones(base_frame)
                    )
                    bridge_img = np.logical_and(bridge_img, rest_of_picture)
        return bridge_img, linetopo, linebaixo, bridge_border, line1, line2

    line1, line2 = find_contours_around_origin(
        rest_of_picture, max_accepted, path_radius_bridg, trunk
    )
    try:
        bridge_img, linetopo, linebaixo, bridge_border, line1, line2 = (
            close_area_from_lines(line1, line2, base_frame, inicial_pnt)
        )
        bridge_border_seq = path_tools.img_to_chain(bridge_border)
    except:
        return [[], [], [], []]
    while np.sum(bridge_border == 2) > 4 and len(bridge_border_seq) > 1:
        opened = mt.opening(bridge_img, kernel_size=1)
        line1b = np.logical_and(line1, opened)
        line2b = np.logical_and(line2, opened)
        if line1b.any() and line2b.any():
            line1c = it.restore_continuous(line1b)
            line2c = it.restore_continuous(line2b)
            bridge_img, linetopo, linebaixo, bridge_border, line1, line2 = (
                close_area_from_lines(line1c, line2c, base_frame, inicial_pnt)
            )
            bridge_border_seq = path_tools.img_to_chain(bridge_border)
        else:
            extreme_external_points = [[], [], [], []]
            break
    lens = [len(x) for x in bridge_border_seq]
    bridge_border_seq = bridge_border_seq[np.argmax(lens)]
    ends_topo = pt.img_to_points(find_tips(linetopo))
    ends_baixo = pt.img_to_points(find_tips(linebaixo))
    bridge_border_seq = path_tools.set_first_pt_in_seq(bridge_border_seq, ends_topo[0])
    counter = 0
    flag = 0
    extreme_external_points = [ends_topo[0], [], [], []]
    for p in bridge_border_seq:
        if p in ends_topo + ends_baixo:
            if (counter == 1) and (p in ends_topo):
                flag = 1
            if flag and counter > 0:
                extreme_external_points[4 - counter] = p
            else:
                extreme_external_points[counter] = p
            counter += 1
    return (
        bridge_img,
        [line1, line2, linetopo, linebaixo],
        bridge_border,
        extreme_external_points,
    )


def reduce_lines_overshoot(candidate: np.ndarray, origin_points: List[np.ndarray]):
    """Determines the start and end for each trunk, then reduces the margins until all its bottlenecks are encompassed"""
    t_ends = pt.img_to_points(find_tips(candidate.astype(bool)))
    origin_chain = pt.invert_x_y(
        path_tools.make_a_chain_open_segment(candidate.astype(bool), t_ends)
    )
    new_ends = [pt.closest_point(x, origin_chain)[0] for x in origin_points]
    new_origin = origin_chain.copy()
    count_up = 0
    count_down = -1
    start_flag = 0
    end_flag = 0
    while not (start_flag and end_flag):
        current_pt_1 = origin_chain[count_up]
        current_pt_2 = origin_chain[count_down]
        if current_pt_1 in new_ends:
            start_flag = 1
        else:
            new_origin.remove(current_pt_1)
            count_up += 1
        if current_pt_2 in new_ends:
            end_flag = 1
        else:
            new_origin.remove(current_pt_2)
            count_down -= 1
    reduced_origin = it.points_to_img(new_origin, np.zeros_like(candidate))
    return reduced_origin
