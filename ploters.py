from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from files import Paths
# from __future__ import annotations
from typing import TYPE_CHECKING
from matplotlib import pyplot as plt
from components import morphology_tools as mt
import numpy as np

if TYPE_CHECKING:
    from components.layer import Layer


def color_palette(num, saved=False):
    """color_palette: Returns a list of colors length num
    Inputs:
    num        = number of colors to return.
    saved      = use the previously stored color scale, if any (default = False).
    Returns:
    colors     = a list of color lists (RGB values)
    :param num: int
    :return colors: list
    """
    cmap = plt.get_cmap("gist_rainbow")
    colors = cmap(np.linspace(0, 1, num), bytes=True)
    colors = colors[:, 0:3].tolist()
    return colors


def mapping_thin_walls(layer: Layer, folders: Paths):
    # thin_img = layer.original_img.copy()
    # thin_img = folders.load_layer_orig_img(layer.original_img)
    thin_img = np.zeros(layer.base_frame, np.uint8)
    # thin_img = thin_img.astype(np.uint8)
    for isl in layer.islands:
        isl_img = folders.load_island_img(isl)
        thin_img = np.add(thin_img, isl_img.astype(np.uint8))
        if len(isl.thin_walls.regions) > 0:
            for tw in isl.thin_walls.regions:
                reg_img = folders.load_thin_wall_img(tw)
                thin_img = np.add(thin_img, reg_img.astype(np.uint8))
    return thin_img

def mapping_thin_walls_medialAxis(layer: Layer) -> np.ndarray:
    all_medial = np.zeros(layer.base_frame)
    for isl in layer.islands:
        all_medial = np.add(all_medial, mt.dilation(isl.thin_walls.medial_transform, kernel_size=8))
    return all_medial

