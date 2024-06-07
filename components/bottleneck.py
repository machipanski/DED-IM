from cv2 import getStructuringElement, MORPH_RECT
import numpy as np
from components import morphology_tools as mt
from components import images_tools as it
from components import points_tools as pt
from components import skeleton as sk
from components import path_tools


class Bottleneck:

    def __init__(
        self,
        name,
        img,
        origin,
        trunk,
        n_paths,
        origin_marks,
        elementos_contorno=None,
        pontos_extremos=None,
        linked_offset_regions=None,
        linked_zigzag_regions=None,
    ):
        if elementos_contorno is None:
            elementos_contorno = []
        if pontos_extremos is None:
            pontos_extremos = []
        if linked_offset_regions is None:
            linked_offset_regions = []
        if linked_zigzag_regions is None:
            linked_zigzag_regions = []
        self.name = name
        self.img = img
        self.origin = origin
        self.destiny = 0
        self.n_paths = n_paths
        self.origin_mark = origin_marks
        self.trunk = trunk
        self.contorno = elementos_contorno
        self.pontos_extremos = pontos_extremos
        self.origin_coords = []
        self.destiny_coords = []
        self.route = []
        self.trail = []
        self.center = []
        self.interruption_points = []
        self.reference_points = []
        self.linked_offset_regions = linked_offset_regions
        self.linked_zigzag_regions = linked_zigzag_regions
        return

    def make_offset_bridge_route(
        self, offsets_regions, path_radius, base_frame, rest_of_picture
    ):
        all_offsets = it.sum_imgs([x.img for x in offsets_regions])
        _, outer = mt.detect_contours(self.img, return_img=True)
        objective_lines = np.logical_and(
            outer.astype(np.uint8),
            np.logical_not(mt.dilation(all_offsets, kernel_size=1).astype(np.uint8)),
        )
        square_mask = getStructuringElement(
            MORPH_RECT, (int(path_radius * 2), int(path_radius * 2))
        )
        objective_lines_dilated = mt.dilation(objective_lines, kernel_img=square_mask)
        outer_offseted = np.logical_and(
            self.img, np.logical_not(objective_lines_dilated)
        )
        outer_offseted = it.take_the_bigger_area(outer_offseted)
        _, outer_new = mt.detect_contours(outer_offseted, return_img=True)
        objective_lines_new = np.logical_and(
            outer_new.astype(np.uint8),
            np.logical_not(mt.dilation(all_offsets, kernel_size=1).astype(np.uint8)),
        )
        self.route = objective_lines_new
        self.trail = mt.dilation(self.route, kernel_size=path_radius)
        return


def close_bridge_contour_v2(
    trunk, base_frame, dist, rest_of_picture, mask, path_radius
):
    def find_contours_around_origin(rest_of_picture, base_frame, dist, path_radius, trunk):
        all_borders, all_borders_img = mt.detect_contours(rest_of_picture, return_img=True)
        area_pescocal = mt.dilation(trunk, kernel_size=(dist + 1.5 * path_radius))
        overlap = np.add(area_pescocal, all_borders_img)
        linhas_do_limite = overlap == 2
        _, labeled, labeled_n = it.divide_by_connected(linhas_do_limite)
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
            linha1 = labeled == idx1 + 1
            linha2 = labeled == idx2 + 1
        elif labeled_n == 2:
            linha1 = labeled == 1
            linha2 = labeled == 2
        else:
            return [], []
        return linha1, linha2
    
    def close_area_from_lines(linha1: np.ndarray, linha2: np.ndarray, base_frame, new_base):
        # ORGANIZAR LINHAS E PONTOS EXTREMOS
        inicios_e_fins1 = pt.x_y_para_pontos(
            np.where(sk.find_tips(linha1.astype(np.uint8)))
        )
        inicios_e_fins2 = pt.x_y_para_pontos(
            np.where(sk.find_tips(linha2.astype(np.uint8)))
        )
        dist_1a_2 = list(
            map(lambda x: pt.distance_pts(inicios_e_fins1[0], x), inicios_e_fins2)
        )
        dist_1b_2 = list(
            map(lambda x: pt.distance_pts(inicios_e_fins1[1], x), inicios_e_fins2)
        )
        ponto_destino_1 = inicios_e_fins2[np.argmin(dist_1a_2)]
        ponto_destino_2 = inicios_e_fins2[np.argmin(dist_1b_2)]
        fechamento1_pts = [inicios_e_fins1[0], ponto_destino_1]
        fechamento2_pts = [inicios_e_fins1[1], ponto_destino_2]
        linhatopo = it.draw_line(
            np.zeros(base_frame), fechamento1_pts[0], fechamento1_pts[1]
        )
        linhabaixo = it.draw_line(
            np.zeros(base_frame), fechamento2_pts[0], fechamento2_pts[1]
        )
        bridge_border = it.sum_imgs([linha1, linhatopo, linha2, linhabaixo])
        bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
        return bridge_img, linhatopo, linhabaixo, bridge_border


    new_base = np.zeros(base_frame)
    linha1, linha2 = find_contours_around_origin(
        rest_of_picture, base_frame, dist, path_radius, trunk
    )
    bridge_img, linhatopo, linhabaixo, bridge_border = close_area_from_lines(
        linha1, linha2, base_frame, new_base
    )
    bridge_border_seq = path_tools.img_to_chain(bridge_border)
    while np.sum(bridge_border == 2) > 4 or len(bridge_border_seq) > 1:
        opened = mt.opening(bridge_img, kernel_size=1)
        linha1b = np.logical_and(linha1, opened)
        linha2b = np.logical_and(linha2, opened)
        linha1c = it.restore_continuous(linha1b)
        linha2c = it.restore_continuous(linha2b)
        bridge_img, linhatopo, linhabaixo, bridge_border = close_area_from_lines(
            linha1c, linha2c, base_frame, new_base
        )
        bridge_border_seq = path_tools.img_to_chain(bridge_border)
    bridge_border_seq = bridge_border_seq[0]
    ends_topo = pt.img_to_points(
        sk.find_tips(linhatopo)
    )
    ends_baixo = pt.img_to_points(
        sk.find_tips(linhabaixo)
    )
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
        [linha1, linha2, linhatopo, linhabaixo],
        bridge_border,
        extreme_external_points,
    )