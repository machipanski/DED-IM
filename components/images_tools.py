from __future__ import annotations
import copy
import itertools
from math import e
import random
import cv2
import numpy as np
from components import morphology_tools as mt
from skimage.measure import label
from skimage.segmentation import flood_fill
from components import points_tools as pt
from typing import TYPE_CHECKING
import subprocess
import tempfile
import shutil
import os
import logging

if TYPE_CHECKING:
    from components.layer import Layer
    from typing import List
    from components.files import System_Paths


def chain_to_lines(final_chain, canvas, color=999):
    """recieves a sequence of points (y,x) and draws it line by line on the canvas"""
    if color == 999:
        color = 1
        by_seg_color = True
    else:
        by_seg_color = False
    count = 0
    chain = final_chain.copy()
    chain = pt.invert_x_y(chain)
    first = chain[0]
    last = chain[-1]
    end_p = chain.pop()
    while len(chain) > 0:
        start_p = end_p
        if chain:
            end_p = chain.pop()
            cv2.line(canvas, tuple(np.int32(start_p)), tuple(np.int32(end_p)), color, 1)
        else:
            end_p = last
            cv2.line(canvas, tuple(np.int32(start_p)), tuple(np.int32(end_p)), color, 1)
        count += 1
        if by_seg_color:
            color = count % 5 + 1
    return canvas


def closest_points_btwn_imgs(img1, img2):
    # Encontra os contours das duas imagens
    contours1, _ = cv2.findContours(
        img1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours2, _ = cv2.findContours(
        img2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Inicializa as variáveis para armazenar a menor distância e os pontos correspondentes
    menor_distancia = float("inf")
    ponto1_mais_proximo = None
    ponto2_mais_proximo = None
    # Compara cada ponto do contour da primeira imagem com cada ponto do contour da segunda imagem
    for contour1 in contours1:
        for ponto1 in contour1:
            for contour2 in contours2:
                for ponto2 in contour2:
                    distancia = np.linalg.norm(ponto1 - ponto2)
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        ponto1_mais_proximo = ponto1
                        ponto2_mais_proximo = ponto2

    return list(ponto1_mais_proximo[0]), list(ponto2_mais_proximo[0])


def longer_than(area, comp):
    area_pts = pt.x_y_para_pontos(np.nonzero(area))
    area_xs = [a[1] for a in area_pts]
    if area_xs:
        area_comp = np.max(area_xs) - np.min(area_xs)
        if area_comp >= comp:
            return True
    return False


def divide_by_connected(img, connectivity=2) -> List[List[np.ndarray], np.ndarray, int]:
    """returns separated_imgs, labels, num"""
    separated_imgs = []
    labels, num = label(img, connectivity=connectivity, return_num=True)
    # divide a área em regiões desconexas
    for i in np.arange(0, num):
        separated_imgs.append(labels == i + 1)  # cria a imagem da area
    return separated_imgs, labels, num


def filter_connected_by_points(binary_img: np.ndarray, points: list) -> np.ndarray:
    """
    Receives a binary image and a list of (row, col) points.
    Returns a binary image with only the connected components that contain at least one of the points.
    """
    labeled = label(binary_img, connectivity=1)
    labels_to_keep = set()
    for y, x in points:
        if 0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]:
            lbl = labeled[y, x]
            if lbl > 0:
                labels_to_keep.add(lbl)
    result = np.isin(labeled, list(labels_to_keep))
    return result.astype(binary_img.dtype)


def draw_line(img, a, b, color=1):
    af = tuple(np.flip(a))
    bf = tuple(np.flip(b))
    return cv2.line(img.astype(np.uint8), af, bf, color, 1)


def draw_circle(img, center, radius, fill=-1):
    af = tuple(np.flip(center))
    return cv2.circle(img.astype(np.uint8), af, radius, 1, fill)


def draw_polyline(img, pts_list, closed):
    points = copy.deepcopy(pts_list)
    for p in points:
        p.reverse()
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(img.astype(np.uint8), [pts], closed, 1, 1)


def extend_tangent(last_point, second_last, slope, length):
    """Extend the tangent line from the last point."""
    if slope is None:  # Vertical line
        if last_point[0] > second_last[0]:
            return [
                (last_point[0] + length, last_point[1]),
                (last_point[0], last_point[1]),
            ]
        else:
            return [
                (last_point[0] - length, last_point[1]),
                (last_point[0], last_point[1]),
            ]
        # return [(last_point[0] - length, last_point[1]), (last_point[0] + length, last_point[1])]
    # Calculate the angle of the slope
    angle = np.arctan(slope)
    # Calculate the end points of the tangent line
    if second_last[1] < last_point[1]:
        x2 = last_point[1] + length * np.cos(angle)
        y2 = last_point[0] + length * np.sin(angle)
    elif second_last[1] > last_point[1]:
        x2 = last_point[1] - length * np.cos(angle)
        y2 = last_point[0] - length * np.sin(angle)
    else:
        if second_last[0] > last_point[0]:
            x2 = last_point[1] + length * np.cos(angle)
            y2 = last_point[0] + length * np.sin(angle)
        elif second_last[0] < last_point[0]:
            x2 = last_point[1] - length * np.cos(angle)
            y2 = last_point[0] - length * np.sin(angle)
    # return [(y1, x1), (y2, x2)
    if y2 < 0:
        y2 = 0
    if x2 < 0:
        x2 = 0
    return [(last_point[0], last_point[1]), (y2, x2)]


def esta_contido(a, b):
    """Analyzes if area 'a' has all its pixels within 'b',
    *a maximum pixel limit is established to consider it inside,
     but if possible, I want to return to the previous form"""
    return not (np.logical_and(a, np.logical_not(b)).any())


def eliminate_duplicates(lista: List[np.ndarray]):
    list_points = [pt.img_to_points(x) for x in lista]
    # hashes = [hash(str(x)) for x in lista]
    included = []
    no_repetition = []
    for i, t in enumerate(lista):
        if not (list_points[i] in included):
            no_repetition.append(t)
            included.append(list_points[i])
    return no_repetition


def fill_internal_area(
    contour_img: np.ndarray, original_img: np.ndarray, old_method=False
) -> np.ndarray:
    if old_method:
        filled_img = flood_fill(np.logical_not(contour_img), (0, 0), 0, connectivity=1)
        filled_img = np.logical_or(filled_img, contour_img)  # OR reinsere a trilha
        filled_img = np.logical_and(filled_img, original_img)
        return filled_img
    else:
        contours, hierarchy = cv2.findContours(
            contour_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        filled_img = np.zeros(contour_img.shape[:2], dtype=np.uint8)
        all_contours = np.zeros(contour_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(all_contours, contours, -1, 1, 1)
        cv2.drawContours(filled_img, contours, 0, 1, cv2.FILLED)
        no_external = np.logical_and(
            all_contours,
            np.logical_not(
                cv2.drawContours(
                    np.zeros(contour_img.shape[:2], dtype=np.uint8), contours, 0, 1, 1
                )
            ),
        )
        if no_external.any():
            contours, hierarchy = cv2.findContours(
                no_external.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            for i, contour in enumerate(contours):
                # Check if the contour has no child
                if hierarchy[0][i][2] < 0:
                    # If contour has no child, fill the contour with black color
                    cv2.drawContours(filled_img, [contour], -1, 0, cv2.FILLED)
        return filled_img


def filter_first_unique_images(image_list):
    """
    Returns only the first occurrence of each unique binary image in the list.
    Args:
        image_list (list of np.ndarray): List of binary images (same shape).
    Returns:
        list of np.ndarray: Filtered list with only the first occurrence of each unique image.
    """
    seen = set()
    result = []
    for img in image_list:
        # Convert image to a hashable type (bytes)
        img_bytes = img.tobytes()
        if img_bytes not in seen:
            seen.add(img_bytes)
            result.append(img)
    return result


def filter_segments_img_by_length(
    imgs: List[np.ndarray], min_length: int
) -> List[np.ndarray]:
    """
    :param imgs: List[np.ndarray] = list of segment images
    :param min_length: int = minimum length to keep the segment
    :return: List [np.ndarray] = filtered list of segment images
    """
    filtered = []
    for i in imgs:
        if np.sum(i.astype(bool)) >= min_length:
            filtered.append(i)
    return filtered


def filter_img_elements_by_openess(
    img: np.ndarray, radius: int = None, kernell: np.ndarray = None
) -> np.ndarray:
    """
    :param img: np.ndarray = image to be filtered
    :param element: np.ndarray = structuring element for opening operation
    :return: np.ndarray = filtered image
    """
    filtered = np.zeros_like(img)
    if kernell is None:
        kernell = mt.disk(radius)
    elements = divide_by_connected(img)[0]
    for i in elements:
        if np.sum(mt.opening(i, kernel_img=kernell)) > 0:  # minimum area threshold
            filtered = np.logical_or(filtered, i)
    return filtered


def final_mapping(layer: Layer, folders: System_Paths):
    isl_final_map = np.zeros(layer.base_frame)
    regions_imgs = []
    for isl in layer.islands:
        folders.load_zigzags_hdf5(layer.name, isl)
        if hasattr(isl, "zigzags"):
            if (
                hasattr(isl.zigzags, "internal_islands")
                and len(isl.zigzags.internal_islands) > 0
            ):
                for int_isl in isl.zigzags.internal_islands:
                    if hasattr(int_isl, "w_regions") and len(int_isl.w_regions) > 0:
                        regions_imgs.append(
                            sum_imgs_colored(
                                [reg.img for reg in int_isl.w_regions], limited=True
                            ).astype(np.uint16)
                        )
                    if hasattr(int_isl, "l_regions") and len(int_isl.l_regions) > 0:
                        regions_imgs.append(
                            sum_imgs_colored(
                                [reg.img for reg in int_isl.l_regions], limited=True
                            ).astype(np.uint16)
                        )
        folders.load_thin_walls_hdf5(layer.name, isl)
        if hasattr(isl, "thin_walls"):
            if hasattr(isl.thin_walls, "regions") and len(isl.thin_walls.regions) > 0:
                regions_imgs.append(
                    sum_imgs([reg.img for reg in isl.thin_walls.regions]).astype(
                        np.uint16
                    )
                    * 5
                )
        folders.load_offsets_hdf5(layer.name, isl)
        if hasattr(isl, "offsets"):
            if hasattr(isl.offsets, "regions") and len(isl.offsets.regions) > 0:
                regions_imgs.append(
                    sum_imgs([reg.img for reg in isl.offsets.regions]).astype(np.uint16)
                    * 6
                )
        folders.load_bridges_hdf5(layer.name, isl)
        if hasattr(isl, "bridges"):
            if hasattr(isl.bridges, "zigzag_bridges"):
                if len(isl.bridges.zigzag_bridges) > 0:
                    regions_imgs.append(
                        sum_imgs(
                            [reg.img for reg in isl.bridges.zigzag_bridges]
                        ).astype(np.uint16)
                        * 7
                    )
            if hasattr(isl.bridges, "offset_bridges"):
                if len(isl.bridges.offset_bridges) > 0:
                    regions_imgs.append(
                        sum_imgs(
                            [reg.img for reg in isl.bridges.offset_bridges]
                        ).astype(np.uint16)
                        * 8
                    )
            if hasattr(isl.bridges, "cross_over_bridges"):
                if len(isl.bridges.cross_over_bridges) > 0:
                    regions_imgs.append(
                        sum_imgs(
                            [reg.img for reg in isl.bridges.cross_over_bridges]
                        ).astype(np.uint16)
                        * 9
                    )
        isl_final_map = sum_imgs(regions_imgs)
    return isl_final_map


def areas_interfaces(layer: Layer, folders: System_Paths):
    isl_final_map = np.zeros(layer.base_frame)
    tw = np.zeros(layer.base_frame)
    off = np.zeros(layer.base_frame)
    zz = np.zeros(layer.base_frame)
    ob = np.zeros(layer.base_frame)
    zb = np.zeros(layer.base_frame)
    cob = np.zeros(layer.base_frame)

    regions_imgs = []
    for isl in layer.islands:
        folders.load_thin_walls_hdf5(layer.name, isl)
        if hasattr(isl, "thin_walls"):
            if hasattr(isl.thin_walls, "regions") and len(isl.thin_walls.regions) > 0:
                regions_imgs.append(
                    sum_imgs([reg.img for reg in isl.thin_walls.regions]).astype(
                        np.uint16
                    )
                    * 1
                )
        folders.load_zigzags_hdf5(layer.name, isl)
        if hasattr(isl, "zigzags"):
            if hasattr(isl.zigzags, "regions") and len(isl.zigzags.regions) > 0:
                regions_imgs.append(
                    sum_imgs_colored(
                        [reg.img for reg in isl.zigzags.regions], limited=True
                    ).astype(np.uint16)
                    * 2
                )
        folders.load_offsets_hdf5(layer.name, isl)
        if hasattr(isl, "offsets"):
            if hasattr(isl.offsets, "regions") and len(isl.offsets.regions) > 0:
                regions_imgs.append(
                    sum_imgs([reg.img for reg in isl.offsets.regions]).astype(np.uint16)
                    * 4
                )
        folders.load_bridges_hdf5(layer.name, isl)
        if hasattr(isl, "bridges"):
            if hasattr(isl.bridges, "zigzag_bridges"):
                if len(isl.bridges.zigzag_bridges) > 0:
                    regions_imgs.append(
                        sum_imgs(
                            [reg.img for reg in isl.bridges.zigzag_bridges]
                        ).astype(np.uint16)
                        * 8
                    )
            if hasattr(isl.bridges, "offset_bridges"):
                if len(isl.bridges.offset_bridges) > 0:
                    regions_imgs.append(
                        sum_imgs(
                            [reg.img for reg in isl.bridges.offset_bridges]
                        ).astype(np.uint16)
                        * 16
                    )
            if hasattr(isl.bridges, "cross_over_bridges"):
                if len(isl.bridges.cross_over_bridges) > 0:
                    regions_imgs.append(
                        sum_imgs(
                            [reg.img for reg in isl.bridges.cross_over_bridges]
                        ).astype(np.uint16)
                        * 32
                    )
        isl_final_map = sum_imgs(regions_imgs)
    return isl_final_map


def individual_routes(layer: Layer, folders: System_Paths):
    isl_ind_routes = np.zeros(layer.base_frame)
    regions_imgs = []
    for isl in layer.islands:
        folders.load_thin_walls_hdf5(layer.name, isl)
        if hasattr(isl, "thin_walls"):
            if hasattr(isl.thin_walls, "regions") and len(isl.thin_walls.regions) > 0:
                if (
                    hasattr(isl.thin_walls.regions[0], "route")
                    and len(isl.thin_walls.regions[0].route) > 0
                ):
                    regions_imgs.append(
                        sum_imgs([reg.route for reg in isl.thin_walls.regions]).astype(
                            np.uint16
                        )
                        * 501
                    )
        folders.load_zigzags_hdf5(layer.name, isl)
        if hasattr(isl, "zigzags"):
            if (
                hasattr(isl.zigzags, "internal_islands")
                and len(isl.zigzags.internal_islands) > 0
            ):
                for intisl in isl.zigzags.internal_islands:
                    if hasattr(intisl, "l_regions") and len(intisl.l_regions) > 0:
                        for l_reg in intisl.l_regions:
                            if hasattr(l_reg, "route"):
                                aaaa = l_reg.route
                                regions_imgs.append(aaaa * 101)
                    if hasattr(intisl, "w_regions") and len(intisl.w_regions) > 0:
                        for w_reg in intisl.w_regions:
                            if hasattr(w_reg, "route"):
                                aaaa = w_reg.route
                                regions_imgs.append(aaaa * 101)
        folders.load_offsets_hdf5(layer.name, isl)
        if hasattr(isl, "offsets"):
            if hasattr(isl.offsets, "regions") and len(isl.offsets.regions) > 0:
                if (
                    hasattr(isl.offsets.regions[0], "route")
                    and len(isl.offsets.regions[0].route) > 0
                ):
                    regions_imgs.append(
                        sum_imgs([reg.route for reg in isl.offsets.regions]).astype(
                            np.uint16
                        )
                        * 601
                    )
        folders.load_bridges_hdf5(layer.name, isl)
        if hasattr(isl, "bridges"):
            if hasattr(isl.bridges, "zigzag_bridges"):
                if len(isl.bridges.zigzag_bridges) > 0:
                    if (
                        hasattr(isl.bridges.zigzag_bridges[0], "route")
                        and len(isl.bridges.zigzag_bridges[0].route) > 0
                    ):
                        regions_imgs.append(
                            sum_imgs(
                                [reg.route for reg in isl.bridges.zigzag_bridges]
                            ).astype(np.uint16)
                            * 701
                        )
            if hasattr(isl.bridges, "offset_bridges"):
                if len(isl.bridges.offset_bridges) > 0:
                    for bridg in isl.bridges.offset_bridges:
                        if np.sum(bridg.route) > 0:
                            regions_imgs.append(bridg.route.astype(np.uint16) * 801)
            if hasattr(isl.bridges, "cross_over_bridges"):
                if (
                    hasattr(isl.bridges.cross_over_bridges[0], "route")
                    and len(isl.bridges.cross_over_bridges[0].route) > 0
                ):
                    if len(isl.bridges.cross_over_bridges) > 0:
                        regions_imgs.append(
                            sum_imgs(
                                [reg.route for reg in isl.bridges.cross_over_bridges]
                            ).astype(np.uint16)
                            * 901
                        )
        isl_ind_routes = sum_imgs(regions_imgs)
    aaaaaa = sum_imgs_colored(
        [
            isl.zigzags.internal_islands[1].w_regions[1].route,
            isl.zigzags.internal_islands[1].w_regions[0].route,
            isl.bridges.zigzag_bridges[1].route_b,
            isl.zigzags.internal_islands[0].w_regions[1].route,
            isl.zigzags.internal_islands[0].w_regions[0].route_b,
            isl.zigzags.internal_islands[0].l_regions[0].route,
            isl.zigzags.internal_islands[0].w_regions[5].route,
            isl.zigzags.internal_islands[0].w_regions[4].route_b,
            isl.bridges.zigzag_bridges[0].route,
            isl.zigzags.internal_islands[2].w_regions[2].route,
            isl.zigzags.internal_islands[2].w_regions[1].route,
            isl.zigzags.internal_islands[2].w_regions[0].route_b,
            isl.zigzags.internal_islands[0].w_regions[2].route,
            isl.zigzags.internal_islands[0].w_regions[3].route_b,
        ]
    )
    folders.save_img(aaaaaa, "sequence.png", layer.name, isl.name)
    return isl_ind_routes


def individual_routes_b(layer: Layer, folders: System_Paths):
    isl_ind_routes_b = np.zeros(layer.base_frame)
    regions_imgs = []
    for isl in layer.islands:
        folders.load_thin_walls_hdf5(layer.name, isl)
        if hasattr(isl, "thin_walls"):
            if hasattr(isl.thin_walls, "regions") and len(isl.thin_walls.regions) > 0:
                if (
                    hasattr(isl.thin_walls.regions[0], "route")
                    and len(isl.thin_walls.regions[0].route) > 0
                ):
                    regions_imgs.append(
                        sum_imgs(
                            [reg.route_b for reg in isl.thin_walls.regions]
                        ).astype(np.uint16)
                        * 501
                    )
        folders.load_zigzags_hdf5(layer.name, isl)
        if hasattr(isl, "zigzags"):
            if (
                hasattr(isl.zigzags, "internal_islands")
                and len(isl.zigzags.internal_islands) > 0
            ):
                for intisl in isl.zigzags.internal_islands:
                    if hasattr(intisl, "l_regions") and len(intisl.l_regions) > 0:
                        for l_reg in intisl.l_regions:
                            if hasattr(l_reg, "route_b"):
                                aaaa = l_reg.route_b
                                regions_imgs.append(aaaa * 101)
                    if hasattr(intisl, "w_regions") and len(intisl.w_regions) > 0:
                        for w_reg in intisl.w_regions:
                            if hasattr(w_reg, "route_b"):
                                aaaa = w_reg.route_b
                                regions_imgs.append(aaaa * 101)
        folders.load_offsets_hdf5(layer.name, isl)
        if hasattr(isl, "offsets"):
            if hasattr(isl.offsets, "regions") and len(isl.offsets.regions) > 0:
                if (
                    hasattr(isl.offsets.regions[0], "route")
                    and len(isl.offsets.regions[0].route) > 0
                ):
                    regions_imgs.append(
                        sum_imgs([reg.route for reg in isl.offsets.regions]).astype(
                            np.uint16
                        )
                        * 601
                    )
        folders.load_bridges_hdf5(layer.name, isl)
        if hasattr(isl, "bridges"):
            if hasattr(isl.bridges, "zigzag_bridges"):
                if len(isl.bridges.zigzag_bridges) > 0:
                    if (
                        hasattr(isl.bridges.zigzag_bridges[0], "route_b")
                        and len(isl.bridges.zigzag_bridges[0].route_b) > 0
                    ):
                        regions_imgs.append(
                            sum_imgs(
                                [reg.route_b for reg in isl.bridges.zigzag_bridges]
                            ).astype(np.uint16)
                            * 701
                        )
            if hasattr(isl.bridges, "offset_bridges"):
                if len(isl.bridges.offset_bridges) > 0:
                    for bridg in isl.bridges.offset_bridges:
                        if np.sum(bridg.route) > 0:
                            regions_imgs.append(bridg.route.astype(np.uint16) * 801)
            if hasattr(isl.bridges, "cross_over_bridges"):
                if (
                    hasattr(isl.bridges.cross_over_bridges[0], "route_b")
                    and len(isl.bridges.cross_over_bridges[0].route_b) > 0
                ):
                    if len(isl.bridges.cross_over_bridges) > 0:
                        regions_imgs.append(
                            sum_imgs(
                                [reg.route_b for reg in isl.bridges.cross_over_bridges]
                            ).astype(np.uint16)
                            * 901
                        )
        isl_ind_routes_b = sum_imgs(regions_imgs)
    return isl_ind_routes_b


def individual_trails(layer: Layer, folders: System_Paths):
    isl_ind_trails = np.zeros(layer.base_frame)
    regions_imgs = []
    for isl in layer.islands:
        folders.load_thin_walls_hdf5(layer.name, isl)
        if hasattr(isl, "thin_walls"):
            if hasattr(isl.thin_walls, "regions") and len(isl.thin_walls.regions) > 0:
                if (
                    hasattr(isl.thin_walls.regions[0], "trail")
                    and len(isl.thin_walls.regions[0].trail) > 0
                ):
                    regions_imgs.append(
                        sum_imgs([reg.trail for reg in isl.thin_walls.regions]).astype(
                            np.uint16
                        )
                        * 501
                    )
        folders.load_zigzags_hdf5(layer.name, isl)
        if hasattr(isl, "zigzags"):
            if (
                hasattr(isl.zigzags, "internal_islands")
                and len(isl.zigzags.internal_islands) > 0
            ):
                for intisl in isl.zigzags.internal_islands:
                    if hasattr(intisl, "l_regions") and len(intisl.l_regions) > 0:
                        for l_reg in intisl.l_regions:
                            if hasattr(l_reg, "trail"):
                                aaaa = l_reg.trail
                                regions_imgs.append(aaaa * 101)
                    if hasattr(intisl, "w_regions") and len(intisl.w_regions) > 0:
                        for w_reg in intisl.w_regions:
                            if hasattr(w_reg, "trail"):
                                aaaa = w_reg.trail
                                regions_imgs.append(aaaa * 101)
        folders.load_offsets_hdf5(layer.name, isl)
        if hasattr(isl, "offsets"):
            if hasattr(isl.offsets, "regions") and len(isl.offsets.regions) > 0:
                if (
                    hasattr(isl.offsets.regions[0], "trail")
                    and len(isl.offsets.regions[0].trail) > 0
                ):
                    regions_imgs.append(
                        sum_imgs([reg.trail for reg in isl.offsets.regions]).astype(
                            np.uint16
                        )
                        * 601
                    )
        folders.load_bridges_hdf5(layer.name, isl)
        if hasattr(isl, "bridges"):
            if hasattr(isl.bridges, "zigzag_bridges"):
                if len(isl.bridges.zigzag_bridges) > 0:
                    if (
                        hasattr(isl.bridges.zigzag_bridges[0], "trail")
                        and len(isl.bridges.zigzag_bridges[0].trail) > 0
                    ):
                        regions_imgs.append(
                            sum_imgs(
                                [reg.trail for reg in isl.bridges.zigzag_bridges]
                            ).astype(np.uint16)
                            * 701
                        )
            if hasattr(isl.bridges, "offset_bridges"):
                if len(isl.bridges.offset_bridges) > 0:
                    for bridg in isl.bridges.offset_bridges:
                        if np.sum(bridg.trail) > 0:
                            regions_imgs.append(bridg.trail.astype(np.uint16) * 801)
            if hasattr(isl.bridges, "cross_over_bridges"):
                if (
                    hasattr(isl.bridges.cross_over_bridges[0], "trail")
                    and len(isl.bridges.cross_over_bridges[0].trail) > 0
                ):
                    if len(isl.bridges.cross_over_bridges) > 0:
                        regions_imgs.append(
                            sum_imgs(
                                [reg.trail for reg in isl.bridges.cross_over_bridges]
                            ).astype(np.uint16)
                            * 901
                        )
        isl_ind_trails = sum_imgs(regions_imgs)
    return isl_ind_trails


def has_contact(fail, new_zigzag):
    connection = np.add(fail.astype(np.uint8), new_zigzag.astype(np.uint8))
    return (connection == 2).any()


def image_subtract(gray_img1: np.ndarray, gray_img2: np.ndarray) -> np.ndarray:
    """
    :param gray_img1: numpy.ndarray = Grayscale image data from which gray_img2 will be subtracted
    :param gray_img2: numpy.ndarray = Grayscale image data which will be subtracted from gray_img1
    :return new_img: numpy.ndarray = subtracted image
    This is a function used to subtract values of one
    gray-scale image array from another gray-scale image array. The
    resulting gray-scale image array has a minimum element value of zero.
    That is all negative values resulting from the
    subtraction are forced to zero.
    """
    new_img = gray_img1.astype(np.float64) - gray_img2.astype(
        np.float64
    )  # subtract values
    new_img[np.where(new_img < 0)] = 0  # force negative array values to zero
    new_img = new_img.astype(np.uint8)  # typecast image to 8-bit image
    return new_img


def points_to_img(pts_list, img):
    for p in pts_list:
        img[p[0], p[1]] = 1
    return img


def neighborhood(group1, group2=[], ends=False, path_radius=10):
    neighbor_areas_g1 = []
    for area_a, area_b in itertools.combinations(group1, 2):
        if ends:
            atual = np.add(
                mt.dilation(
                    np.logical_or(
                        mt.hitmiss_ends_v2(area_a.route),
                        mt.hitmiss_ends_v2(area_a.route_b),
                    ),
                    kernel_size=2 * path_radius,
                ),
                mt.dilation(
                    np.logical_or(
                        mt.hitmiss_ends_v2(area_b.route),
                        mt.hitmiss_ends_v2(area_b.route_b),
                    ),
                    kernel_size=2 * path_radius,
                ),
            )
        else:
            atual = np.logical_or(area_a.img, area_b.img)
        _, n_labels = label(atual, return_num=True, connectivity=2)
        # print(area_a.name, area_b.name)
        if n_labels <= 1 or (ends and (atual == 2).any()):
            neighbor_areas_g1.append([area_a.name, area_b.name])
    if not group2:
        return neighbor_areas_g1
    else:
        neighbor_areas_g2 = []
        for area_a, area_b in itertools.combinations(group2, 2):
            if ends:
                atual = np.add(
                    mt.dilation(
                        np.logical_or(
                            mt.hitmiss_ends_v2(area_a.route),
                            mt.hitmiss_ends_v2(area_a.route_b),
                        ),
                        kernel_size=2 * path_radius,
                    ),
                    mt.dilation(
                        np.logical_or(
                            mt.hitmiss_ends_v2(area_b.route),
                            mt.hitmiss_ends_v2(area_b.route_b),
                        ),
                        kernel_size=2 * path_radius,
                    ),
                )
            else:
                atual = np.logical_or(area_a.img, area_b.img)
            _, n_labels = label(atual, return_num=True, connectivity=2)
            if n_labels == 1 or (ends and (atual == 2).any()):
                neighbor_areas_g2.append([area_a.name, area_b.name])
        neighbor_areas_g1xg2 = []
        for area_a, area_b in itertools.product(group1, group2):
            if ends:
                atual = np.add(
                    mt.dilation(
                        np.logical_or(
                            mt.hitmiss_ends_v2(area_a.route),
                            mt.hitmiss_ends_v2(area_a.route_b),
                        ),
                        kernel_size=2 * path_radius,
                    ),
                    mt.dilation(
                        np.logical_or(
                            mt.hitmiss_ends_v2(area_b.route),
                            mt.hitmiss_ends_v2(area_b.route_b),
                        ),
                        kernel_size=2 * path_radius,
                    ),
                )
            else:
                atual = np.logical_or(area_a.img, area_b.img)
            _, n_labels = label(atual, return_num=True, connectivity=2)
            if n_labels == 1 or (ends and (atual == 2).any()):
                neighbor_areas_g1xg2.append([area_a.name, area_b.name])
        return neighbor_areas_g1, neighbor_areas_g2, neighbor_areas_g1xg2


def neighborhood_routes(group1, group2=[], path_radius=10, apendix1="", apendix2=""):

    def check_neighbors(g1, g2, n_list):
        if not g2:
            iteration = list(itertools.combinations(g1, 2))
        else:
            iteration = list(itertools.product(g1, g2))
        for area_a, area_b in iteration:
            aa = np.add(
                mt.dilation(
                    mt.hitmiss_ends_v2(area_a.route),
                    kernel_size=path_radius * 3,
                ),
                mt.dilation(
                    mt.hitmiss_ends_v2(area_b.route),
                    kernel_size=path_radius * 3,
                ),
            )
            if (aa == 2).any():
                n_list.append(
                    [
                        apendix1 + area_a.name + "_route",
                        apendix2 + area_b.name + "_route",
                    ]
                )
            ab = np.add(
                mt.dilation(
                    mt.hitmiss_ends_v2(area_a.route),
                    kernel_size=path_radius * 3,
                ),
                mt.dilation(
                    mt.hitmiss_ends_v2(area_b.route_b),
                    kernel_size=path_radius * 3,
                ),
            )
            if (ab == 2).any():
                n_list.append(
                    [
                        apendix1 + area_a.name + "_route",
                        apendix2 + area_b.name + "_route_b",
                    ]
                )
            ba = np.add(
                mt.dilation(
                    mt.hitmiss_ends_v2(area_a.route_b),
                    kernel_size=path_radius * 3,
                ),
                mt.dilation(
                    mt.hitmiss_ends_v2(area_b.route),
                    kernel_size=path_radius * 3,
                ),
            )
            if (ba == 2).any():
                n_list.append(
                    [
                        apendix1 + area_a.name + "_route_b",
                        apendix2 + area_b.name + "_route",
                    ]
                )
            bb = np.add(
                mt.dilation(
                    mt.hitmiss_ends_v2(area_a.route_b),
                    kernel_size=path_radius * 3,
                ),
                mt.dilation(
                    mt.hitmiss_ends_v2(area_b.route_b),
                    kernel_size=path_radius * 3,
                ),
            )
            if (bb == 2).any():
                n_list.append(
                    [
                        apendix1 + area_a.name + "_route_b",
                        apendix2 + area_b.name + "_route_b",
                    ]
                )
        return n_list

    if not group2:
        apendix2 = apendix1
        neighbor_areas_g1 = check_neighbors(group1, [], [])
        return neighbor_areas_g1
    else:
        neighbor_areas_g1 = check_neighbors(group1, [], [])
        neighbor_areas_g2 = check_neighbors(group2, [], [])
        neighbor_areas_g1xg2 = check_neighbors(group1, group2, [])
        return neighbor_areas_g1, neighbor_areas_g2, neighbor_areas_g1xg2


def dwg_to_binary_image_external(input_dwg, output_png, dpi=300):
    """
    Use external tools like QCAD or LibreCAD for conversion.
    Install QCAD: https://www.qcad.org/
    """

    qcad_path = r"/path/to/qcad"  # Adjust to your installation

    # Convert DWG to PNG using QCAD command line
    subprocess.run(
        [
            os.path.join(qcad_path, "qcad"),
            "-r",
            "-o",
            output_png,
            "-l",
            "en",
            "-t",
            "png",
            "-s",
            f"{dpi}x{dpi}",  # Resolution
            input_dwg,
        ],
        check=True,
    )

    # Convert to binary if needed
    img = Image.open(output_png)
    binary = img.convert("1")  # Convert to 1-bit
    binary.save(output_png)

    return output_png


def neighborhood_imgs(areas):
    areas_down = [x.img[:-1].astype(int) for x in areas]
    areas_up = [y.img[1:].astype(int) for y in areas]
    for a in areas:
        a.viz_down = []
        a.viz_up = []
    for area_a, area_b in itertools.permutations([x.name for x in areas], 2):
        if (
            areas_down[area_a] & areas_up[area_b]
        ).any():  # vertical edge start positions
            areas[area_a].viz_down.append(area_b)
        if (
            areas_up[area_a] & areas_down[area_b]
        ).any():  # vertical edge start positions
            areas[area_a].viz_up.append(area_b)
    return areas


def img_add_border(img: np.ndarray):
    """There are moments when some morphological operations undergo changes
    when the pixels are at the edge of the image.
    To avoid these distortions, some pixels are added to the image."""
    # print(os.chdir())
    img_w_border = np.zeros(np.add(img.shape, [int(20) * 4, int(20) * 4]))
    x_offset = y_offset = int(20) * 2
    img_w_border[
        y_offset : y_offset + int(img.shape[0]),
        x_offset : x_offset + int(img.shape[1]),
    ] = img
    img_layer = img_w_border.astype(np.uint16)
    _, img_bin = cv2.threshold(
        img_layer, 100, 255, cv2.THRESH_BINARY
    )  # aqui a sensibilidade do filtro é alterada
    img_bin[img_bin > 50] = 1
    img_bin = mt.closing(img_bin, kernel_size=1)
    return img_bin.astype(np.uint16)


def remove_border(img, nozzle_diam_pxl) -> np.ndarray:
    """Adding pixels to the image shifts the final image.
    To prevent this from happening, we can remove them with this function."""
    x_offset = y_offset = int(nozzle_diam_pxl) * 2
    img_no_border = img[
        y_offset : int(img.shape[0] - y_offset), x_offset : int(img.shape[1] - x_offset)
    ]
    return img_no_border.astype(np.uint8)


def restore_continuous(line_img):
    newline = np.add(line_img, mt.find_failures(line_img, np.zeros_like(line_img)))
    newline = np.add(newline, mt.find_crosses(newline, np.zeros_like(newline)))
    newline = take_the_bigger_area(newline)
    return mt.thinning(newline)


def rotate_img_cw(img: np.ndarray) -> np.ndarray:
    """Rotates 90 degrees clockwise"""
    return cv2.rotate(img.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)


def rotate_img_ccw(img: np.ndarray) -> np.ndarray:
    """Rotates 90 degrees counterclockwise"""
    return cv2.rotate(img.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)


def sum_imgs_colored(imgs_list, limited=False, start_color=1) -> np.ndarray:
    """recieves a list of images and add returns a lebeled version of them"""
    filtered = [img for img in imgs_list if np.any(img)]
    if filtered == []:
        return []
    all = np.zeros_like(filtered[0], np.uint16)
    color = start_color
    for img in filtered:
        all = np.add(img * color, all)
        if limited and color == 4:
            color = 1
        else:
            color += 1
    return all


def sum_imgs(imgs_list: List[np.ndarray]) -> np.ndarray:
    """recieves a list of images and add them up"""
    filtered = [img for img in imgs_list if np.any(img)]
    if filtered == []:
        return []
    all = np.zeros_like(filtered[0], np.uint16)
    for img in filtered:
        all = np.add(img, all)
    return all


def take_the_bigger_area(img: np.ndarray):
    new_img, areas_n = label(img, return_num=True)
    separated_areas = []
    for idx in np.arange(1, areas_n + 1):
        area = new_img == idx
        separated_areas.append(area)
    area_sums = list(map(lambda x: np.sum(x), separated_areas))
    return separated_areas[np.argmax(area_sums)]


def extend_line_random_to_touch(
    image, origin, minimum=2, touches=1, pre_dettermined=9999, print_from_first=False
):
    directions = [
        (0, -1),  # Left
        (-1, 0),  # Up
        (0, 1),  # Right
        (1, 0),  # Down
        # (-1, -1),  # Up-Left
        # (-1, 1),  # Up-Right
        # (1, -1),  # Down-Left
        # (1, 1),  # Down-Right
    ]
    touches_counter = 0
    flag_touch = False
    if pre_dettermined > 5:
        direction = random.choice(directions)
        direction_index = directions.index(direction)
    else:
        direction_index = pre_dettermined
        direction = directions[direction_index]
    extended_line = np.zeros_like(image)
    y, x = origin
    while 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
        if touches_counter == touches:
            flag_touch = True
            break
        extended_line[y, x] = image[y, x] + 1
        if extended_line[y, x] >= minimum and (y, x) != origin:
            touches_counter += 1
        if print_from_first and touches_counter == 0:
            extended_line[y, x] = 0
        y += direction[0]
        x += direction[1]
    if flag_touch:
        # print("   Touch found")
        return extended_line, flag_touch, direction_index
    else:
        print("   No touch found")
        return extend_line_random_to_touch(
            image,
            origin,
            minimum=minimum,
            touches=touches,
            print_from_first=print_from_first,
        )


def rectangle_middle_and_corner_points_expanded(shape):
    """
    Recebe o shape (altura, largura) de uma imagem e retorna uma lista com:
    - os pontos médios das arestas,
    - os quatro vértices do retângulo expandido com o dobro da área e mesmo centroide.
    """
    h, w = shape[:2]
    area = h * w
    new_area = area * 5
    aspect = w / h

    # Calcula novo h e w mantendo o centroide
    new_h = int(round((new_area / aspect) ** 0.5))
    new_w = int(round(new_h * aspect))

    # Garante que a área seja pelo menos o dobro
    if new_h * new_w < new_area:
        new_w += 1

    # Centro do retângulo original
    cy = h // 2
    cx = w // 2

    # Calcula os limites do novo retângulo
    top = cy - new_h // 2
    bottom = top + new_h - 1
    left = cx - new_w // 2
    right = left + new_w - 1

    # Pontos médios das arestas
    top_middle = (top, (left + right) // 2)
    bottom_middle = (bottom, (left + right) // 2)
    left_middle = ((top + bottom) // 2, left)
    right_middle = ((top + bottom) // 2, right)

    # Vértices
    corners = [
        (top, left),  # canto superior esquerdo
        (top, right),  # canto superior direito
        (bottom, left),  # canto inferior esquerdo
        (bottom, right),  # canto inferior direito
    ]
    middles = [top_middle, bottom_middle, left_middle, right_middle]
    return middles + corners
