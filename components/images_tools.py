from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from components.layer import Layer
    from typing import List
    from files import System_Paths
import itertools
import cv2
import numpy as np
from components import morphology_tools as mt
from skimage.measure import label
from skimage.segmentation import flood_fill
import copy
from components import points_tools as pt


def chain_to_lines(final_chain, canvas):
    """recieves a sequence of points (y,x) and draws it line by line on the canvas"""
    color = 1
    count = 0
    chain = final_chain.copy()
    first = chain[0]
    last = chain[-1]
    end_p = chain.pop()
    while len(chain) > 0:
        start_p = end_p
        if chain:
            end_p = chain.pop()
            cv2.line(canvas, tuple(start_p), tuple(end_p), color, 1)
        else:
            end_p = last
            cv2.line(canvas, tuple(start_p), tuple(end_p), color, 1)
        count += 1
        color = count % 5 + 1
    return canvas


def comprimento_maior_que(area, comp):
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


def draw_line(img, a, b):
    af = tuple(np.flip(a))
    bf = tuple(np.flip(b))
    return cv2.line(img.astype(np.uint8), af, bf, 1, 1)


def draw_circle(img, center, radius):
    af = tuple(np.flip(center))
    return cv2.circle(img.astype(np.uint8), af, radius, 1, -1)


def draw_polyline(img, pts_list, closed):
    points = copy.deepcopy(pts_list)
    for p in points:
        p.reverse()
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(img.astype(np.uint8), [pts], closed, 1, 1)


def esta_contido(a, b):
    """Analisa se a área a tem todos os pixels dentro de b,
    *está estabelecido um limite máximo de pixels para considerar
     dentro, mas se possível quero voltar à forma anterior"""
    return not (np.logical_and(a, np.logical_not(b)).any())


def fill_internal_area(contour_img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    internal_area = flood_fill(np.logical_not(contour_img), (0, 0), 0, connectivity=1)
    internal_area = np.logical_or(internal_area, contour_img)  # OR reinsere a trilha
    internal_area = np.logical_and(internal_area, original_img)
    # AND para garantir os buracos
    return internal_area


def final_mapping(layer: Layer, folders: System_Paths):
    isl_final_map = np.zeros(layer.base_frame)
    regions_imgs = []
    for isl in layer.islands:
        folders.load_thin_walls_hdf5(layer.name, isl)
        if hasattr(isl.thin_walls, "regions") and len(isl.thin_walls.regions) > 0:
            regions_imgs.append(
                sum_imgs([reg.img for reg in isl.thin_walls.regions]).astype(np.uint16)
                * 501
            )
        folders.load_zigzags_hdf5(layer.name, isl)
        if hasattr(isl.zigzags, "regions") and len(isl.zigzags.regions) > 0:
            regions_imgs.append(
                sum_imgs_colored([reg.img for reg in isl.zigzags.regions]).astype(
                    np.uint16
                )
                * 101
            )
        folders.load_offsets_hdf5(layer.name, isl)
        if hasattr(isl.offsets, "regions") and len(isl.offsets.regions) > 0:
            regions_imgs.append(
                sum_imgs([reg.img for reg in isl.offsets.regions]).astype(np.uint16)
                * 601
            )
        folders.load_bridges_hdf5(layer.name, isl)
        if hasattr(isl.bridges, "zigzag_bridges"):
            if len(isl.bridges.zigzag_bridges) > 0:
                regions_imgs.append(
                    sum_imgs([reg.img for reg in isl.bridges.zigzag_bridges]).astype(
                        np.uint16
                    )
                    * 701
                )
        if hasattr(isl.bridges, "offset_bridges"):
            if len(isl.bridges.offset_bridges) > 0:
                regions_imgs.append(
                    sum_imgs([reg.img for reg in isl.bridges.offset_bridges]).astype(
                        np.uint16
                    )
                    * 801
                )
        if hasattr(isl.bridges, "cross_over_bridges"):
            if len(isl.bridges.cross_over_bridges) > 0:
                regions_imgs.append(
                    sum_imgs(
                        [reg.img for reg in isl.bridges.cross_over_bridges]
                    ).astype(np.uint16)
                    * 901
                )
        isl_final_map = sum_imgs(regions_imgs)
    return isl_final_map


def has_contact(fail, new_zigzag):
    connection = np.add(fail.astype(np.uint8), new_zigzag.astype(np.uint8))
    return (connection == 2).any()


def image_subtract(gray_img1: np.ndarray, gray_img2: np.ndarray) -> np.ndarray:
    """
    This is a function used to subtract values of one
    gray-scale image array from another gray-scale image array. The
    resulting gray-scale image array has a minimum element value of zero.
    That is all negative values resulting from the
    subtraction are forced to zero.
    Inputs:
    gray_img1   = Grayscale image data from which gray_img2 will be subtracted
    gray_img2   = Grayscale image data which will be subtracted from gray_img1
    Returns:
    new_img = subtracted image
    :param gray_img1: numpy.ndarray
    :param gray_img2: numpy.ndarray
    :return new_img: numpy.ndarray
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


def neighborhood(group1, group2=[]):
    areas_vizinhas_g1 = []
    for area_a, area_b in itertools.combinations(group1, 2):
        atual = np.logical_or(area_a.img, area_b.img)
        _, n_labels = label(atual, return_num=True, connectivity=2)
        # print(area_a.name, area_b.name)
        if n_labels <= 1:
            areas_vizinhas_g1.append([area_a.name, area_b.name])
    if not group2:
        return areas_vizinhas_g1
    else:
        areas_vizinhas_g2 = []
        for area_a, area_b in itertools.combinations(group2, 2):
            atual = np.logical_or(area_a.img, area_b.img)
            _, n_labels = label(atual, return_num=True, connectivity=2)
            if n_labels == 1:
                areas_vizinhas_g2.append([area_a.name, area_b.name])
        areas_vizinhas_g1xg2 = []
        for area_a, area_b in itertools.product(group1, group2):
            atual = np.logical_or(area_a.img, area_b.img)
            _, n_labels = label(atual, return_num=True, connectivity=2)
            if n_labels == 1:
                areas_vizinhas_g1xg2.append([area_a.name, area_b.name])
        return areas_vizinhas_g1, areas_vizinhas_g2, areas_vizinhas_g1xg2


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


def read_img_add_border(img_name: str) -> np.ndarray:
    """Há momentos em que algumas operações morfológicas sofrem alterações
    quando os pixels estão no limite da imagem
    para evitar essas distorções, são adicionados alguns pixels no imagem"""
    # print(os.chdir())
    img = cv2.imread(img_name, 0)
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
    """Adicionar pixels na imagem desloca a imagem final.
    para que isso não aconteça podemos remove-los com essa função"""
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
    """Gira 90 graus no sentido horario"""
    return cv2.rotate(img.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)


def rotate_img_ccw(img: np.ndarray) -> np.ndarray:
    """Gira 90 graus no sentido anti-horario"""
    return cv2.rotate(img.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)


def sum_imgs_colored(imgs_list):
    """recieves a list of images and add returns a lebeled version of them"""
    all = np.zeros_like(imgs_list[0], np.uint16)
    color = 1
    for img in imgs_list:
        all = np.add(img.astype(np.uint16) * color, all)
        if color == 3:
            color = 1
        else:
            color += 1
    return all


def sum_imgs(imgs_list: List[np.ndarray]) -> np.ndarray:
    """recieves a list of images and add them up"""
    all = np.zeros_like(imgs_list[0], np.uint16)
    for img in imgs_list:
        all = np.add(img.astype(np.uint16), all)
    return all


def take_the_bigger_area(img: np.ndarray):
    new_img, areas_n = label(img, return_num=True)
    separated_areas = []
    for idx in np.arange(1, areas_n + 1):
        area = new_img == idx
        separated_areas.append(area)
    area_sums = list(map(lambda x: np.sum(x), separated_areas))
    return separated_areas[np.argmax(area_sums)]
