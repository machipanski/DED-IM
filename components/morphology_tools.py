from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from components.layer import Layer
import cv2
import numpy as np
from components import skeleton as sk
from skimage.morphology import disk, thin
from elements import (
    MEIO_DO_CAMINHO,
    PONTAS_SOLTAS,
    INTERSECCOES,
    CORTADO,
    INTERROMPIDO,
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
    """Retorna o contorno e, se pedir com jeitinho, a imagem e a hierarquia"""
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
    for k in np.arange(0, len(INTERROMPIDO)):
        result = np.logical_or(
            result.astype(np.uint8),
            cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, INTERROMPIDO[k]),
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


def hitmiss_ends_v2(img):
    # return pcv.morphology.find_tips(img.astype(np.uint8))
    return sk.find_tips(img)


def make_mask(layer: Layer, size: str) -> np.ndarray:
    """Cria um elemento de máscara para operações morfológicas"""
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


def make_distancer(layer: Layer, region: str, percentage:float=50) -> np.ndarray:
    """Cria um elemento de máscara para operções morfológicas
        quando é feito um distanciador para sobreposição entre trilhas 
        de solda em relação ao seu diâmetro total. Portanto,
        50%(standart) retorna o elemento do tamanho da trilha
        solitária (diam real do programa de solda)"""
    if region == "tw":
        orig_diam_mm = layer.diam_tw_real
    if region == "cont":
        orig_diam_mm = layer.diam_cont_real
    if region == "bridg":
        orig_diam_mm = layer.diam_bridg_real
    if region == "larg":
        orig_diam_mm = layer.diam_larg_real
    deslocamento = orig_diam_mm*((100-percentage)/100) #para isolar o diametro real da trilha
    mask = disk(round(deslocamento*layer.pxl_per_mm))
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
