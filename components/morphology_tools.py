from __future__ import annotations
import cv2
import numpy as np
from components import skeleton as sk
from skimage.morphology import disk, thin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from components.layer import Layer
from components.elements import (
    NORMAL_SKELETON_POINT,
    LOOSE_ENDS,
    INTERSECTIONS,
    CUTTEN,
    INTERRUPTED,
    EXCESSIVE_DIAGONALS,
    CROSSES,
)


def closing(img: np.ndarray, kernel_img=None, kernel_size=None) -> np.ndarray:
    if kernel_img is None:
        closed = cv2.morphologyEx(
            img.astype(np.uint8), cv2.MORPH_CLOSE, disk(kernel_size)
        )
    if kernel_size is None:
        closed = cv2.morphologyEx(
            img.astype(np.uint8), cv2.MORPH_CLOSE, kernel_img.astype(np.uint8)
        )
    return closed


def detect_contours(
    img: np.ndarray, return_img=False, return_hierarchy=False, only_external=False
) -> np.ndarray:
    """Returns the contour and, if asked nicely, the image and the hierarchy"""
    retrieve = cv2.RETR_TREE
    if only_external:
        retrieve = cv2.RETR_EXTERNAL
    area_contour, hierarchy = cv2.findContours(
        img.astype(np.uint8), retrieve, cv2.CHAIN_APPROX_NONE
    )
    if return_img:
        area_contour_img = cv2.drawContours(
            np.zeros_like(img).astype(np.uint8), area_contour, -1, 1
        )
        if return_hierarchy:
            return area_contour, area_contour_img, hierarchy
        return area_contour, area_contour_img
    else:
        if return_hierarchy:
            return area_contour, hierarchy
        return area_contour


def dilation(img: np.ndarray, kernel_img=None, kernel_size=None) -> np.ndarray:
    if kernel_img is None:
        kernel_img = []
        dilated = cv2.dilate(img.astype(np.uint8), disk(kernel_size))
    if kernel_size is None:
        kernel_size = []
        dilated = cv2.dilate(img.astype(np.uint8), kernel_img.astype(np.uint8))
    return dilated


def erosion(img: np.ndarray, kernel_img=None, kernel_size=None) -> np.ndarray:
    if kernel_img is None:
        kernel_img = []
        dilated = cv2.erode(img.astype(np.uint8), disk(kernel_size))
    if kernel_size is None:
        kernel_size = []
        dilated = cv2.erode(img.astype(np.uint8), kernel_img.astype(np.uint8))
    return dilated


def find_crosses(img: np.ndarray, base) -> np.ndarray:
    result = base
    for c in CROSSES:
        result = np.logical_or(
            result.astype(np.uint8),
            cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, c),
        )
    return result


def find_failures(img, base):
    result = base
    for k in np.arange(0, len(INTERRUPTED)):
        result = np.logical_or(
            result.astype(np.uint8),
            cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, INTERRUPTED[k]),
        )
    result[:2] = 0
    result[-2:] = 0
    result[:, :2] = 0
    result[:, -2:] = 0
    return result


def gradient(img, kernel_img=None, kernel_size=None):
    if kernel_img is None:
        grad = cv2.morphologyEx(
            img.astype(np.uint8), cv2.MORPH_GRADIENT, disk(kernel_size)
        )
    if kernel_size is None:
        grad = cv2.morphologyEx(
            img.astype(np.uint8), cv2.MORPH_GRADIENT, kernel_img.astype(np.uint8)
        )
    return grad


def blackhat(img: np.ndarray, kernel_img=None, kernel_size=None) -> np.ndarray:
    if kernel_img is None:
        blackhat = cv2.morphologyEx(
            img.astype(np.uint8), cv2.MORPH_BLACKHAT, disk(kernel_size)
        )
    if kernel_size is None:
        blackhat = cv2.morphologyEx(
            img.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel_img.astype(np.uint8)
        )
    return blackhat


def hitmiss_ends_v2(img):
    # return pcv.morphology.find_tips(img.astype(np.uint8))
    return sk.find_tips(img)


def make_parabola_kernel(size=7, center_value=3, edge_value=0.1):
    assert size % 2 == 1, "O tamanho deve ser ímpar"
    c = size // 2
    y, x = np.ogrid[-c : size - c, -c : size - c]
    dist2 = (x**2 + y**2) / (c**2)
    a = edge_value - center_value
    b = center_value
    kernel = a * dist2 + b
    kernel = np.maximum(kernel, 0)  # Zera valores negativos
    return kernel


def make_mask(layer: Layer, size: str) -> np.ndarray:
    """Creates a mask element for morphological operations"""
    if size == "full_tw":
        mask = disk(round(layer.path_radius_tw))
    if size == "half_tw":
        mask = disk(round(layer.path_radius_tw * 0.5))
    if size == "3_4_tw":
        mask = disk(round(layer.path_radius_tw * 0.75))
    if size == "3_2_tw":
        mask = disk(round(layer.path_radius_tw * 1.5))
    if size == "double_tw":
        mask = disk(round(layer.path_radius_tw * 2))
    if size == "full_cont":
        mask = disk(round(layer.path_radius_cont))
    if size == "half_cont":
        mask = disk(round(layer.path_radius_cont * 0.5))
    if size == "3_4_cont":
        mask = disk(round(layer.path_radius_cont * 0.75))
    if size == "3_2_cont":
        mask = disk(round(layer.path_radius_cont * 1.5))
    if size == "double_cont":
        mask = disk(round(layer.path_radius_cont * 2))
    if size == "full_bridg":
        mask = disk(round(layer.path_radius_bridg))
    if size == "half_bridg":
        mask = disk(round(layer.path_radius_bridg * 0.5))
    if size == "3_4_bridg":
        mask = disk(round(layer.path_radius_bridg * 0.75))
    if size == "3_2_bridg":
        mask = disk(round(layer.path_radius_bridg * 1.5))
    if size == "double_bridg":
        mask = disk(round(layer.path_radius_bridg * 2))
    if size == "full_larg":
        mask = disk(round(layer.path_radius_larg))
    if size == "half_larg":
        mask = disk(round(layer.path_radius_larg * 0.5))
    if size == "3_4_larg":
        mask = disk(round(layer.path_radius_larg * 0.75))
    if size == "3_2_larg":
        mask = disk(round(layer.path_radius_larg * 1.5))
    if size == "double_larg":
        mask = disk(round(layer.path_radius_larg * 2))
    return mask


def make_distancer(layer: Layer, region: str, percentage: float = 50) -> np.ndarray:
    """Creates a mask element for morphological operations
    when a spacer is made for overlap between welding tracks
    in relation to their total diameter. Therefore,
    50% (standard) returns the element the size of the solitary track
    (real diameter of the welding program)"""
    if region == "tw":
        orig_diam_mm = layer.diam_tw_real
    if region == "cont":
        orig_diam_mm = layer.diam_cont_real
    if region == "bridg":
        orig_diam_mm = layer.diam_bridg_real
    if region == "larg":
        orig_diam_mm = layer.diam_larg_real
    if region == "int_ext":
        orig_diam_mm = layer.diam_larg_real
    displacement = orig_diam_mm * (
        (100 - percentage) / 100
    )  # para isolar o diametro real da trilha
    mask = disk(round(displacement * layer.pxl_per_mm))
    return mask


def opening(img: np.ndarray, kernel_img=None, kernel_size=None) -> np.ndarray:
    if kernel_img is None:
        opened = cv2.morphologyEx(
            img.astype(np.uint8), cv2.MORPH_OPEN, disk(kernel_size)
        )
    if kernel_size is None:
        opened = cv2.morphologyEx(
            img.astype(np.uint8), cv2.MORPH_OPEN, kernel_img.astype(np.uint8)
        )
    return opened


def thinning(img):
    return thin(img, max_num_iter=None)


def colored_dilation(image, kernel):
    output = cv2.filter2D(src=image, ddepth=2, kernel=kernel)
    return output
