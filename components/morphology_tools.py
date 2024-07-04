from __future__ import annotations
from typing import TYPE_CHECKING
from elements import MEIO_DO_CAMINHO, PONTAS_SOLTAS, INTERSECCOES, CORTADO, INTERROMPIDO, EXCESSIVE_DIAGONALS, CROSSES
from skimage.morphology import disk
import numpy as np
import cv2
from skimage.morphology import thin
if TYPE_CHECKING:
    from components.layer import Layer

def find_failures(img, base):
    result = base
    for k in np.arange(0, len(INTERROMPIDO)):
        result = np.logical_or(result.astype(np.uint8),
                               cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, INTERROMPIDO[k]))
    result[:2] = 0
    result[-2:] = 0
    result[:, :2] = 0
    result[:, -2:] = 0
    return result

def make_mask(layer: Layer, size:str) -> np.ndarray:
    if size=="full_ext":
        mask = disk(layer.path_radius_external)
    if size=="half_ext":
        mask = disk(int(layer.path_radius_external * 0.5))
    if size=="3_4_ext":
        mask = disk(int(layer.path_radius_external * 0.75))
    if size=="3_2_ext":
        mask = disk(int(layer.path_radius_external * 1.5))
    if size=="double_ext":
        mask = disk(layer.path_radius_external * 2)
    if size=="full_int":
        mask = disk(layer.path_radius_internal)
    if size=="half_int":
        mask = disk(int(layer.path_radius_internal * 0.5))
    if size=="3_4_int":
        mask = disk(int(layer.path_radius_internal * 0.75))
    if size=="3_2_int":
        mask = disk(int(layer.path_radius_internal * 1.5))
    if size=="double_int":
        mask = disk(layer.path_radius_internal * 2)
    return mask


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

def find_crosses(img: np.ndarray, base) -> np.ndarray:
    result = base
    for c in CROSSES:
        result = np.logical_or(result.astype(np.uint8), cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, c))
    return result

def detect_contours(img: np.ndarray, return_img=False, return_hierarchy=False, only_external=False) -> np.ndarray:
    """Retorna o contorno e, se pedir com jeitinho, a imagem e a hierarquia"""
    retrieve = cv2.RETR_TREE
    if only_external:
        retrieve = cv2.RETR_EXTERNAL
    area_contour, hierarchy = cv2.findContours(img.astype(np.uint8), retrieve, cv2.CHAIN_APPROX_NONE)
    if return_img:
        area_contour_img = cv2.drawContours(np.zeros_like(img).astype(np.uint8), area_contour, -1, 1)
        if return_hierarchy:
            return area_contour, area_contour_img, hierarchy
        return area_contour, area_contour_img
    else:
        if return_hierarchy:
            return area_contour, hierarchy
        return area_contour
    
def thinning(img):
    return thin(img, max_num_iter=None)
