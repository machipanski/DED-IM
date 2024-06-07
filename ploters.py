from __future__ import annotations
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


def mapping_thin_walls(layer: Layer):
    thin_img = layer.original_img.copy()
    thin_img = thin_img.astype(np.uint8)
    if len(layer.thin_walls.regions) > 0:
        for i in layer.thin_walls.regions:
            thin_img = np.add(thin_img, i.img.astype(np.uint8))
    return thin_img

def mapping_thin_walls_medialAxis(layer: Layer):
    return mt.dilation(layer.thin_walls.medial_transform, kernel_size=8)

