# import concurrent.futures
import itertools

import math
import copy
import re

# from stat import FILE_ATTRIBUTE_ARCHIVE
# from keyring import set_keyring
# from networkx import reconstruct_path
# from components.layer import Island
from scipy.ndimage import distance_transform_edt
import numpy as np
from components import bottleneck, path_tools
from components import images_tools as it
from components import skeleton as sk
from components import points_tools as pt
from components import morphology_tools as mt
from skimage.measure import label
import networkx as nx
from skimage.feature import corner_harris, corner_peaks

# from skimage.morphology import disk
from cv2 import getStructuringElement, MORPH_RECT

# from scipy.spatial import distance_matrix, distance
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import flood_fill, watershed
from components.timer import Timer
from typing import TYPE_CHECKING
from typing import List

# if TYPE_CHECKING:


class DivisionLine:
    def __init__(self, name, img, procedence, y_line, xs_line):
        self.name = name
        self.img = img
        self.procedence = procedence
        self.y = y_line
        self.xs = xs_line


class ShadowArea:
    def __init__(self, name, img):
        self.name = name
        self.img = img
        self.viz_up = []
        self.viz_down = []
        self.remove = 0
        self.unite_with = []


class Subregion:
    def __init__(self, name, img):
        self.name = name
        self.img = img
        self.routes = []
        self.areas_somadas = []
        self.corte_inicial = []
        self.all_lines_corte = []
        self.lines_corte = []
        self.labeled_monotonic_regions = []
        self.regions = []
        self.evento_limite = []

    def create_shadow_img(self, path_radius):
        """O conceito da imagem de sombras é uma operação de labeling para cada line dos dois lados
        dessa forma conseguimeos ver as "sombras" se iluminarmos a mesma imagem nos dois sentidos
        """
        img = copy.deepcopy(self.img)
        shadows = []
        while np.sum(img) > 0:
            shadow_img_dir = np.zeros_like(img, int)
            shadow_img_esq = np.zeros_like(img, int)
            for index, line in enumerate(img[0:]):
                labeled_line_dir = label(line, connectivity=1)
                labeled_line_esq = label(np.flip(line), connectivity=1)
                shadow_img_dir[index] = labeled_line_dir
                shadow_img_esq[index] = np.flip(labeled_line_esq)
            shadow_img_dir_dd = shadow_img_dir == 1
            shadow_img_esq_dd = shadow_img_esq == 1
            candidates_d, _, num = it.divide_by_connected(shadow_img_dir_dd)
            candidates_e, _, num = it.divide_by_connected(shadow_img_esq_dd)
            candidates = candidates_d + candidates_e
            sums = [np.sum(x) for x in candidates]
            all_ones = candidates[np.argmax(sums)]
            shadows.append(all_ones)
            img = np.logical_and(img, np.logical_not(all_ones))
        new_shadow_img = it.sum_imgs_colored(shadows)
        shadows_after_rejoined, labeled_shadows, num = it.divide_by_connected(
            new_shadow_img
        )
        final_areas = []
        for i, area in enumerate(shadows_after_rejoined):
            final_areas.append(ShadowArea(i, area))
            shadow_img = it.sum_imgs_colored([x.img for x in final_areas])
        return new_shadow_img, final_areas

    def unite_monotonic_shadow_areas(self, areas):
        """Toda área com apenas um vizinho acima e abaixo é listada
        depois estas são eliminadas e a sua imagem usada em um novo mapeamento
        passando um label nele é possível re-separar as áreas que estavam quebradas"""
        areas_cleaned = []
        to_remove_areas = np.zeros_like(areas[0].img)
        for a in areas:
            if len(a.viz_up) <= 1 and len(a.viz_down) <= 1:
                to_remove_areas = np.logical_or(to_remove_areas, a.img)
                a.remove = 0
            else:
                areas_cleaned.append(a)
        separated_imgs, _, _ = it.divide_by_connected(to_remove_areas)
        for i in separated_imgs:
            areas_cleaned.append(ShadowArea("teste", i))
        for i, a in enumerate(areas_cleaned):
            a.name = i
        return areas_cleaned, it.sum_imgs_colored([x.img for x in areas_cleaned])

    def divide_small_shadow_areas(self, areas, path_radius):
        lista_de_pequenas = []
        new_list_areas = []
        for a in areas:
            erodida = mt.erosion(a.img, kernel_size=path_radius)
            if np.sum(erodida) < 1:
                lista_de_pequenas.append(a.name)
            else:
                new_list_areas.append(a)
        if len(lista_de_pequenas) == 0:
            pass
        else:
            subareas = np.zeros_like(areas[0].img, int)
            for p in lista_de_pequenas:
                if len(areas[p].viz_up) > 1:
                    for viz in areas[p].viz_up:
                        contato = areas[viz].img[:-1].astype(int) & areas[p].img[
                            1:
                        ].astype(int)
                        max_c, min_c = pt.max_e_min_coords_img(contato, 1)
                        B = np.zeros_like(areas[0].img)
                        B[:, min_c : max_c + 1] = 1
                        subareas = np.add(
                            subareas, np.logical_and(B, areas[p].img).astype(int)
                        )
                if len(areas[p].viz_down) > 1:
                    for vizD in areas[p].viz_down:
                        contato = areas[p].img[:-1].astype(int) & areas[vizD].img[
                            1:
                        ].astype(int)
                        max_c, min_c = pt.max_e_min_coords_img(contato, 1)
                        B = np.zeros_like(areas[0].img)
                        B[:, min_c : max_c + 1] = 1
                        subareas = np.add(
                            subareas, np.logical_and(B, areas[p].img).astype(int) * 10
                        )
            divs = []
            shadows = np.unique(subareas)
            shadows = shadows[1:]
            for s in shadows:
                divs.append(subareas == s)
            for d in divs:
                labeled_divs, _, _ = it.divide_by_connected(d)
                for ld in labeled_divs:
                    new_list_areas.append(ShadowArea("teste", ld))
            for i, a in enumerate(new_list_areas):
                a.name = i
        return new_list_areas

    def analyze_images_with_erosion(self, areas, path_radius):
        """
        Analyzes a list of binary images. For each image, computes the difference between
        the image and its erosion by a structuring element of path_radius diameter.
        For each connected body in the difference, if the number of pixels is greater
        than the size of the structuring element, appends the body to a list.

        Args:
            images (list of numpy.ndarray): List of binary images (values 0 and 1).
            path_radius (int): Radius of the structuring element used for erosion.

        Returns:
            list of numpy.ndarray: List of connected bodies that meet the size criteria.
        """
        result_bodies = []
        images = [x.img for x in areas]
        structuring_element = mt.disk(path_radius)
        structuring_element_size = np.sum(structuring_element)
        for image in images:
            eroded_image = mt.erosion(image, kernel_size=1.25 * path_radius)
            if np.sum(eroded_image) > 0:
                reconstructed_image = mt.dilation(
                    eroded_image, kernel_size=1.25 * path_radius
                )
                difference = np.logical_and(image, np.logical_not(reconstructed_image))
                aaa, labeled_image, num_features = it.divide_by_connected(difference)
                for region_id in range(1, num_features + 1):
                    # connected_body = it.sum_imgs(
                    #     [labeled_image == region_id, reconstructed_image]
                    # )
                    connected_body = labeled_image == region_id
                    connected_body = mt.opening(
                        connected_body, kernel_size=path_radius * 0.5
                    )
                    # y_coords_orig, x_coords_orig = np.nonzero(image)
                    # x_size_orig = np.max(x_coords_orig) - np.min(x_coords_orig)
                    if np.sum(connected_body) > 0:
                        y_coords_rec, x_coords_rec = np.nonzero(connected_body)
                        x_size_rec = np.max(x_coords_rec) - np.min(x_coords_rec)
                        # difference_x = x_size_orig - x_size_rec
                        if (
                            # np.sum(connected_body) > structuring_element_size
                            # or
                            # difference_x
                            np.sum(connected_body) > 2 * structuring_element_size
                            and x_size_rec > path_radius * 4
                        ):
                            result_bodies.append(labeled_image == region_id)
        if len(result_bodies) > 0:
            aaaaa = it.sum_imgs_colored(result_bodies)
            counter = len(areas)
            for i, j in itertools.product(areas, result_bodies):
                if np.sum(np.logical_and(i.img, j)) > 0:
                    indx = areas.index(i)
                    areas[indx].img = np.logical_and(i.img, np.logical_not(j))
                    areas.append(ShadowArea(counter, j))
        return areas

    def scan_monotonic(self, path_radius, base_frame, ideal_sum):
        shadow_img, areas = self.create_shadow_img(path_radius)
        areas = self.analyze_images_with_erosion(areas, path_radius)
        divided_small_img = it.sum_imgs_colored([x.img for x in areas])
        areas = it.neighborhood_imgs(areas)
        if len(areas) > 1:
            monotonic_regions, self.labeled_monotonic_regions, self.areas_somadas = (
                self.unite_small_monotonic_areas(
                    areas, path_radius, base_frame, ideal_sum
                )
            )
            aaaaaaaaaa = self.labeled_monotonic_regions
            for i, mr in enumerate(monotonic_regions):
                if np.sum(mt.opening(mr.img, kernel_size=path_radius)) > 0:
                    self.regions.append(ZigZag(i, mr.img))
        else:
            if np.sum(mt.opening(self.img, kernel_size=path_radius * 2)) > 0:
                self.labeled_monotonic_regions = self.img
                self.areas_somadas = self.img
                self.regions.append(ZigZag(0, self.labeled_monotonic_regions))
        # aaaaaaa = it.sum_imgs_colored([x.img for x in self.regions])
        return

    def scan_monotonic_v2(
        self,
        path_radius,
        base_frame,
        sem_galhos_nobridges,
        sem_galhos_dist_nobridges,
        trunks_nobridges,
        zigzags_bridges,
        rest_of_picture_f3,
    ):
        internal_area = rest_of_picture_f3
        # prune_divisor = 6 * path_radius
        # prune_internal = 6 * path_radius
        prune_divisor = 0
        prune_internal = path_radius * 2
        if len(zigzags_bridges) > 0:
            internal_area = it.sum_imgs(
                [x.img for x in zigzags_bridges] + [rest_of_picture_f3]
            )
        # internal_area_opened = mt.opening(internal_area, kernel_size=path_radius * 3)
        internal_area_opened = it.take_the_bigger_area(internal_area)
        # sem_galhos = skmorph.skeletonize(internal_area_opened.astype(bool))
        sem_galhos, dist, segment_objects = sk.create_prune_skel(
            internal_area_opened, prune_divisor, distance=path_radius * 10
        )
        translated_lines = []
        for i in segment_objects:
            line = [list(x[0]) for x in i]
            translated_lines.append(pt.invert_x_y(line))
        segment_objects = translated_lines
        seg_imgs = [
            it.points_to_img(line, np.zeros_like(self.img)) for line in segment_objects
        ]
        new_divisor = rest_of_picture_f3
        if len(zigzags_bridges) > 0:
            new_divisor = it.sum_imgs(
                rest_of_picture_f3.astype(np.uint8) + [x.img for x in zigzags_bridges]
            )
            new_divisor = np.logical_and(
                new_divisor,
                np.logical_not(
                    mt.dilation(
                        it.sum_imgs(sem_galhos + [x.origin for x in zigzags_bridges]),
                        kernel_size=1,
                    )
                ),
            )
        else:
            new_divisor = np.logical_and(
                new_divisor,
                np.logical_not(mt.dilation(sem_galhos, kernel_size=1)),
            )
        sem_galhos_new, dist_new, segment_objects_new = sk.create_prune_skel(
            new_divisor.astype(bool), prune_internal
        )
        remade = []
        for line in segment_objects_new:
            x = line.tolist()
            y = [a[0] for a in x]
            remade.append(y)
        reconstruct_path = sk.reconstruct_img_from_skeleton(sem_galhos_new * dist_new)
        tree_deposition_diameter = path_radius * 6
        filtered_segments = list(
            filter(lambda x: len(x) > tree_deposition_diameter, remade)
        )
        seg_imgs = [
            it.points_to_img(pt.invert_x_y(line), np.zeros_like(self.img))
            for line in filtered_segments
        ]
        segs_labels = it.sum_imgs_colored(seg_imgs)
        island_divisor = np.logical_and(new_divisor, self.img.astype(bool))
        labels = watershed(
            island_divisor, segs_labels, mask=island_divisor.astype(np.uint8)
        )
        a_filtrado_ecortado = it.sum_imgs([labels, sem_galhos_new * 10])

        labels_ids = np.unique(labels)
        labels_ids = labels_ids[labels_ids != 0]  # Ignora o fundo
        separated = [(labels == lbl).astype(np.uint8) for lbl in labels_ids]
        labeled_monotonic_regions = labels
        areas_somadas = np.zeros_like(self.img)
        self.regions = []
        for i, label_img in enumerate(separated):
            region_origin = np.logical_and(label_img.astype(bool), sem_galhos_new)
            self.regions.append(ZigZag(i, label_img, origin=region_origin))
            areas_somadas = np.logical_or(areas_somadas, label_img.astype(bool))

        self.labeled_monotonic_regions = labeled_monotonic_regions
        aaa = it.sum_imgs([self.labeled_monotonic_regions, sem_galhos_new])
        aaaa = it.sum_imgs(
            [
                self.labeled_monotonic_regions,
                mt.dilation(sem_galhos_new, kernel_size=path_radius) * 20,
            ]
        )
        self.areas_somadas = areas_somadas
        # ab, ac, aaaassss = sk.segment_sort(sem_galhos_new, segment_objects_new)
        # abc, acc, aaaasssss = sk.segment_sort(sem_galhos, segment_objects)
        return

    def trace_divisions(self, rest_of_picture_f2, base_frame, limites):
        mudancas_line = []
        labeled_img = []
        for i in np.arange(0, limites[1]):
            line = rest_of_picture_f2[i]
            _, labeled_line, changes = it.divide_by_connected(line)
            mudancas_line.append(changes)
            labeled_img.append(labeled_line)
        evento_limite = []
        line_ant = 0
        for i in np.arange(0, len(mudancas_line)):
            line_now = mudancas_line[i]
            if line_now > line_ant:
                evento_limite[-1] = 1
                evento_limite.append(0)
            elif line_now < line_ant:
                evento_limite.append(-1)
            elif line_now == line_ant and i > 0:
                flag = 0
                for j in np.arange(0, limites[0]):
                    if labeled_img[i][j] != 0:
                        if (
                            labeled_img[i][j] != labeled_img[i][j - 1]
                            and labeled_img[i][j - 1] != 0
                        ):
                            flag = 1
                if flag:
                    evento_limite[-1] = 1
                    evento_limite.append(0)
                else:
                    evento_limite.append(0)
            else:
                evento_limite.append(0)
            line_ant = line_now
        all_div_lines = np.zeros(base_frame)
        for i in np.arange(0, limites[1]):
            if evento_limite[i] != 0:
                all_div_lines[i] = 1
        all_div_lines = np.logical_and(all_div_lines, rest_of_picture_f2)
        evento_limite_reduct = []
        for l in np.arange(0, len(evento_limite)):
            if evento_limite[l] != 0:
                evento_limite_reduct.append([l, evento_limite[l]])
        evento_limite_reduct = np.unique(evento_limite_reduct, axis=0)
        lines, _, n_lines_corte = it.divide_by_connected(all_div_lines)
        procedencia = 0
        for i in np.arange(0, n_lines_corte):
            line_points = np.nonzero(lines[i])
            y_line = np.unique(line_points[0])
            if len(y_line) > 1:
                coords = pt.x_y_para_pontos(line_points)
                n_points = []
                for y in y_line:
                    n_points.append(len(list(filter(lambda x: x[0] == y, coords))))
                y_line = y_line[np.argmax(n_points)]
                line_points = list(filter(lambda x: x[0] == y_line, coords))
                line_points = [
                    [x[0] for x in line_points],
                    [x[1] for x in line_points],
                ]
            xs_line = [
                np.min(np.unique(line_points[1])),
                np.max(np.unique(line_points[1])),
            ]
            for evento in evento_limite_reduct:
                if y_line == evento[0]:
                    procedencia = evento[1]
            self.lines_corte.append(
                DivisionLine(i, lines[i], procedencia, y_line, xs_line)
            )
        return all_div_lines

    def unite_small_monotonic_areas(
        self, areas: List[ShadowArea], path_radius, base_frame, ideal_sum
    ):
        def max_fit_inside(area, path_radius, ideal_sum):
            # from skimage.morphology import disk
            max_radius = np.max(distance_transform_edt(area))
            sum_area = np.sum(area)
            # ideal_sum = np.sum(mt.make_mask())
            return max_radius / path_radius, sum_area / ideal_sum

        divided_small_img = it.sum_imgs_colored([x.img for x in areas])
        monotonic_regions = copy.deepcopy(areas)
        for i in np.arange(0, len(monotonic_regions)):
            # radius_pctg, _ = max_fit_inside(
            #     monotonic_regions[i].img, path_radius, ideal_sum
            # )
            radius_pctg, _ = max_fit_inside(
                monotonic_regions[i].img, path_radius, ideal_sum
            )
            if radius_pctg < 2:
                monotonic_regions[i].remove = True
        for region in monotonic_regions:
            if region.remove:
                vizinhos = region.viz_down + region.viz_up
                if not vizinhos == []:
                    interface_sizes = []
                    for v in vizinhos:
                        composed_image = np.add(
                            region.img, monotonic_regions[v].img * 2
                        )
                        interface = path_tools.draw_interface(
                            composed_image, base_frame, 1
                        )
                        interface_sizes.append(np.sum(interface))
                    # vizinho_escolhido = vizinhos[np.argmax(interface_sizes)]
                    vizinho_escolhido = vizinhos[np.argmax(interface_sizes)]
                    region.unite_with = vizinho_escolhido
                    if monotonic_regions[vizinho_escolhido].unite_with:
                        destiny = monotonic_regions[vizinho_escolhido].unite_with
                    else:
                        destiny = vizinho_escolhido
                    new_img = np.logical_or(region.img, monotonic_regions[destiny].img)
                    monotonic_regions[destiny].img = new_img
                    region.img = new_img
                    if vizinho_escolhido in region.viz_down:
                        region.viz_down == monotonic_regions[vizinho_escolhido].viz_down
                    if vizinho_escolhido in region.viz_up:
                        region.viz_up == monotonic_regions[vizinho_escolhido].viz_up
        monotonic_regions = list(filter(lambda x: x.remove == False, monotonic_regions))
        new_labeled = np.zeros(base_frame)
        for i in np.arange(0, len(monotonic_regions)):
            fragmento = monotonic_regions[i].img.astype(np.uint) * (i + 1)
            new_labeled = np.add(new_labeled, fragmento)
        labeled_monotonic_regions = new_labeled
        areas_somadas = np.zeros(base_frame)
        for a in monotonic_regions:
            areas_somadas = np.add(areas_somadas, a.img)
        return monotonic_regions, labeled_monotonic_regions, areas_somadas


class ZigZag:
    def __init__(self, name, img, **kwargs):
        self.name = name
        self.img = img
        self.route = []
        self.trail = []
        self.center = []
        self.remove = False
        self.region_path_radius = 0
        self.origin = []
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        return

    def make_internal_border(
        self, internal, filled_external_borders, eroded, origin_axis, path_radius_bridg
    ):
        def make_closest_path_to(
            internal_borders, original_line, start_orig, end_orig, internal_extreme
        ):
            distance_map = np.multiply(
                internal_borders, distance_transform_edt(np.logical_not(original_line))
            )
            # internal_borders_pts = pt.img_to_points(internal_borders)
            start_ci, _ = pt.closest_point(start_orig, internal_extreme)
            internal_extreme.remove(start_ci)
            end_ci, _ = pt.closest_point(end_orig, internal_extreme)
            internal_extreme.remove(end_ci)
            G = path_tools.img_to_graph_com_distancias(distance_map)
            _, path1 = nx.bidirectional_dijkstra(G, tuple(start_ci), tuple(end_ci))
            line_ci = it.points_to_img(path1, np.zeros_like(internal_borders))
            return line_ci

        internal = np.logical_or(internal, origin_axis)
        _, internal_borders = mt.detect_contours(internal, return_img=True)
        internal_borders = np.logical_and(internal_borders, filled_external_borders)
        _, labeled, labeled_n = it.divide_by_connected(internal_borders)
        divided_contour = it.image_subtract(
            internal_borders, mt.dilation(origin_axis, kernel_size=1)
        )
        # possible_c1_c2, counter_accepted, internal_extreme = (
        #     decompose_pol_cont_by_corners(
        #         internal_borders, origin_axis, path_radius_bridg
        #     )
        # )
        # [line_ci1, line_ci2], labeled_divided, num = it.divide_by_connected(
        #     possible_c1_c2 != 0
        # )
        [line_ci1, line_ci2], labeled_divided, num = it.divide_by_connected(
            divided_contour != 0
        )
        # line_ci1 = np.zeros_like(self.img)
        # line_ci2 = np.zeros_like(self.img)
        # if labeled_n == 1:
        #     internal_extreme = pt.img_to_points(mt.hitmiss_ends_v2(internal_borders))
        #     if len(internal_extreme) == 0:
        #         possible_c1_c2, counter_accepted, internal_extreme = (
        #             decompose_pol_cont_by_corners(
        #                 internal_borders, origin_axis, path_radius_bridg
        #             )
        #         )
        #         # bbb = it.sum_imgs([tng_end,tng_start,all_borders_img])
        #         labeled = possible_c1_c2
        #         labeled_n = counter_accepted
        #     line_ci1 = make_closest_path_to(
        #         internal_borders,
        #         self.contour[0],
        #         self.extreme_points[0],
        #         self.extreme_points[1],
        #         internal_extreme,
        #     )
        #     if len(internal_extreme) >= 2:
        #         line_ci2 = make_closest_path_to(
        #             internal_borders,
        #             self.contour[1],
        #             self.extreme_points[2],
        #             self.extreme_points[3],
        #             internal_extreme,
        #         )
        #     else:
        #         line_ci2 = line_ci1
        # elif labeled_n > 2:
        #     sums = []
        #     for l in np.arange(0, labeled_n):
        #         sums.append(np.sum(labeled == l + 1))
        #     idx = [sums.index(i) for i in sorted(sums, reverse=True)][:2]
        #     line_ci1 = labeled == idx[0] + 1
        #     line_ci2 = labeled == idx[1] + 1
        # elif labeled_n == 2:
        #     line_ci1 = labeled == 1
        #     line_ci2 = labeled == 2
        # internal_borders_closed = np.add(
        #     internal_borders, np.logical_or(self.contour[2], self.contour[3])
        # )
        # internal_borders_closed = np.logical_and(internal, internal_borders_closed)
        # _, labeled, labeled_n = it.divide_by_connected(internal_borders_closed)
        # if labeled_n > 1:
        #     tri_1 = labeled == 1
        #     tri_1 = it.fill_internal_area(tri_1.astype(np.uint8), np.ones_like(tri_1))
        #     tri_2 = labeled == 2
        #     tri_2 = it.fill_internal_area(tri_2.astype(np.uint8), np.ones_like(tri_2))
        #     new_fig = np.logical_or(tri_1, tri_2)
        #     new_fig = np.logical_or(new_fig, self.origin.astype(bool))
        #     new_fig = np.logical_and(new_fig, filled_external_borders)
        #     _, internal_borders_closed = mt.detect_contours(new_fig, return_img=True)
        # line_ci1 = np.logical_and(eroded, line_ci1)
        # line_ci1, _, _ = sk.create_prune_skel(line_ci1, 1)
        # line_ci2 = np.logical_and(eroded, line_ci2)
        # line_ci2, _, _ = sk.create_prune_skel(line_ci2, 1)
        # if np.sum(line_ci1) == 0:
        #     print("Error: no line 1")
        # if np.sum(line_ci2) == 0:
        #     print("Error: no line 2")
        return line_ci1, line_ci2

    def find_center(self):
        contour = mt.detect_contours(self.img)
        contour = pt.contour_to_list(contour)
        pt.points_center(contour)


class ZigZagRegions:
    """Caminho fechado individual por ZigueZague"""

    def __init__(self):
        self.regions = []
        self.all_zigzags = []
        self.macro_areas = []
        self.zigzags_graph = []
        self.zigzags_mst = []
        self.pos_zigzag_nodes = []
        self.macro_areas = []
        self.all_zigzags = []

    def create_oscilatory_inner(
        self,
        macro_areas,
        original_img,
        base_frame,
        path_radius_larg,
        mask_full_larg,
        zigzags,
        bridges,
        offsets,
        thin_walls,
        internal_weaving,
    ):
        if internal_weaving:
            mask_line = np.zeros(np.add(mask_full_larg.shape, [4, 4]))
            mask_line[:, int(mask_full_larg.shape[0] / 2)] = 1
            old_zigzag = all_internal_routes(macro_areas, base_frame)
            with Timer("   Finding internal voids"):
                separated_fail_imgs = find_internal_fails(
                    original_img,
                    base_frame,
                    bridges,
                    macro_areas,
                    mask_full_larg,
                    path_radius_larg,
                    offsets,
                    thin_walls,
                )
            with Timer("   Searching contacts"):
                connected_fails, interface_lines = connect_fails_to_zigzags(
                    old_zigzag, separated_fail_imgs, path_radius_larg
                )

            with Timer("   Creating weavings"):
                # aaaa = it.sum_imgs(separated_fail_imgs+separated_connected_fails)
                fail_internal_zigzags = []
                succesfuly_weaved = []
                for i, fail in enumerate(connected_fails):
                    try:
                        weaves = internal_weaving_cut(
                            interface_lines[i], path_radius_larg, fail
                        )
                        fail_internal_zigzags.append([weaves])
                        succesfuly_weaved.append(interface_lines[i])
                    except:
                        print("Weaving failed")
                        pass
            if len(succesfuly_weaved) > 0:
                all_new_zigzags = copy.deepcopy(old_zigzag)
                for i, weave in enumerate(succesfuly_weaved):
                    all_new_zigzags = it.image_subtract(
                        all_new_zigzags, interface_lines[i]
                    )
                    all_new_zigzags_ver_a = it.sum_imgs(
                        [all_new_zigzags, fail_internal_zigzags[i][0][0]]
                    ).astype(bool)
                    ends_a = mt.hitmiss_ends_v2(all_new_zigzags_ver_a)
                    all_new_zigzags_ver_b = it.sum_imgs(
                        [all_new_zigzags, fail_internal_zigzags[i][0][1]]
                    ).astype(bool)
                    ends_b = mt.hitmiss_ends_v2(all_new_zigzags_ver_b)
                    if np.sum(ends_a) > np.sum(ends_b):
                        all_new_zigzags = all_new_zigzags_ver_b
                    else:
                        all_new_zigzags = all_new_zigzags_ver_a
            else:
                all_new_zigzags = old_zigzag
            new_macro_areas, _, _ = it.divide_by_connected(all_new_zigzags)
        else:
            all_new_zigzags = np.zeros(base_frame)
            for r in macro_areas:
                all_new_zigzags = np.logical_or(all_new_zigzags, r)
            new_macro_areas = macro_areas
        all_new_zigzags, _, _ = sk.create_prune_skel(all_new_zigzags, path_radius_larg)
        return new_macro_areas, all_new_zigzags

    def connect_island_zigzags(self, path_radius_larg, mask_full_larg, base_frame):
        interfaces, centers, interface_types = path_tools.find_points_of_contact(
            list(self.zigzags_mst.edges),
            path_radius_larg,
            mask_full_larg,
            self.regions,
        )
        unified_zigzags = path_tools.draw_the_links(
            self,
            self.zigzags_mst,
            base_frame,
            interfaces,
            centers,
            path_radius_larg,
        )
        macro_area_list, _, _ = it.divide_by_connected(unified_zigzags)
        self.all_zigzags = unified_zigzags
        self.macro_areas = macro_area_list
        return

    def find_monotonic(
        self,
        rest_of_picture_f3,
        base_frame,
        path_radius,
        sem_galhos,
        sem_galhos_dist,
        trunks,
        zigzags_bridges,
        w_style,
        ideal_sum,
    ):
        sub_regions: List[Subregion] = []
        separated_imgs, labeled, num = it.divide_by_connected(rest_of_picture_f3)
        # for i in np.arange(0, num):
        for i, subregion in enumerate(separated_imgs):
            sub_regions.append(Subregion(i, subregion))
            if w_style == 1:
                sub_regions[-1].scan_monotonic_v2(
                    path_radius,
                    base_frame,
                    sem_galhos,
                    sem_galhos_dist,
                    trunks,
                    zigzags_bridges,
                    rest_of_picture_f3,
                )
            else:
                sub_regions[-1].scan_monotonic(path_radius, base_frame, ideal_sum)
        regs_counter = 0
        for sub_region in sub_regions:
            for region in sub_region.regions:
                region.name = regs_counter
                self.regions.append(region)
                regs_counter += 1
        return

    def make_graph(self, zigzags_bridges, base_frame):
        self.zigzags_graph, self.pos_zigzag_nodes = path_tools.make_zigzag_graph(
            self.regions, zigzags_bridges, base_frame
        )
        self.zigzags_mst, zigzags_mst_sequence = path_tools.regions_mst(
            self.zigzags_graph
        )
        return

    def make_routes_z(self, base_frame, path_radius, mask_distancer):
        for region in self.regions:
            region.center = pt.points_center(
                pt.contour_to_list(mt.detect_contours(region.img))
            )
            if len(region.origin) == 0:
                zig_options = []
                lines, n_lines, internal_border_img, contours, new_path_radius = (
                    cut_in_lines(
                        region.img, path_radius, mask_distancer, var_path_width=0
                    )
                )
                filled = it.fill_internal_area(
                    internal_border_img.astype(np.uint8),
                    np.ones_like(internal_border_img),
                    True,
                )
                opened = mt.opening(filled, kernel_size=path_radius)
                with Timer("   Creating the three possible options:"):
                    if np.sum(opened) > 0:
                        [new_zigzag_a, new_zigzag_b] = zig_zag_two_options(
                            internal_border_img,
                            lines,
                            n_lines,
                            new_path_radius,
                            contours,
                            base_frame,
                            False,
                        )
                        [new_zigzag_d, new_zigzag_e] = zig_zag_two_options(
                            internal_border_img,
                            lines,
                            n_lines,
                            new_path_radius,
                            contours,
                            base_frame,
                            True,
                        )
                        zig_options.append(new_zigzag_a)
                        zig_options.append(new_zigzag_b)
                        zig_options.append(new_zigzag_d)
                        zig_options.append(new_zigzag_e)
                    [new_zigzag_c] = zig_zag_third_option(
                        region.img,
                        lines,
                        n_lines,
                        new_path_radius,
                        contours,
                        base_frame,
                    )
                    zig_options.append(new_zigzag_c)
            else:
                new_zigzag_a, new_zigzag_b = make_weaving_wide_route(
                    region,
                    path_radius,
                    mask_distancer,
                    mask_distancer,
                    region.img,
                )
                zig_options = [new_zigzag_a, new_zigzag_b]
            with Timer("   Calculating best route:"):
                zig_fills = [
                    mt.dilation(x.astype(np.uint8), kernel_size=path_radius)
                    for x in zig_options
                ]
                zig_sums = [np.sum(x) for x in zig_fills]
                new_zigzag = zig_options[np.argmax(zig_sums)]
                new_trail = mt.dilation(
                    new_zigzag.astype(np.uint8), kernel_size=path_radius
                )
            region.route = new_zigzag
            region.trail = new_trail
        aaaa = it.sum_imgs_colored([x.route for x in self.regions])
        aaaaa = it.sum_imgs_colored([x.trail for x in self.regions])
        return


def border_cut(contours, lines, points, n_lines, base_frame, zag_zig=0):
    fila = pt.contour_to_list(contours)
    rotations = fila.index(points[0])
    fila = fila[rotations:] + fila[:rotations]  # garante que a fila começa pelo ponto A
    borda_cortada = np.zeros(base_frame)
    borda_normal = np.zeros(base_frame)
    counter = 0
    counter_pixels = 0
    last_y_change = 0
    for i in np.arange(0, len(fila)):
        borda_normal[fila[i][0]][fila[i][1]] = 1
        counter_pixels += 1
        y = fila[i][0]
        x = fila[i][1]
        pixel_lines = lines[y][x]
        ca = [y, x] == points[0]
        cb = [y, x] == points[1]
        cc = [y, x] == points[2]
        cd = [y, x] == points[3]
        ce = pixel_lines == 1
        cf = y != last_y_change
        cg = n_lines % 2
        if zag_zig:
            if cg:
                if (ca or cb or cd or ce) and cf:
                    counter += 1
                    last_y_change = y
            else:
                if (ca or cc or cd or ce) and cf:
                    counter += 1
                    last_y_change = y
        else:
            if cg:
                if (cc or ce) and cf:
                    counter += 1
                    last_y_change = y
            else:
                if (cb or ce) and cf:
                    counter += 1
                    last_y_change = y
        if counter % 2 != 0:
            borda_cortada[fila[i][0]][fila[i][1]] = 1
    return borda_cortada


def clean_zigzag_over_extrusion(contours_img, new_path_radius, base_frame):
    square_mask = getStructuringElement(
        MORPH_RECT, (new_path_radius * 2 - 2, new_path_radius * 2 - 2)
    )
    no_failure = mt.gradient(contours_img, kernel_img=square_mask)
    no_failure_axis_img, _, _ = sk.create_prune_skel(no_failure, new_path_radius)
    no_failure_axis_path, no_failure_axis_path_img = mt.detect_contours(
        no_failure_axis_img, return_img=True, only_external=True
    )
    path_candidates, _, _ = it.divide_by_connected(no_failure_axis_path_img)
    path = path_candidates[0]
    return path


def cut_in_lines(img, path_radius, mask_distancer, var_path_width=0):
    # img2 = mt.opening(img, kernel_size=(path_radius * 2))
    img2 = mt.opening(img, kernel_size=(path_radius))
    considered = np.where(img2 != 0)
    if np.sum(considered[0]) == 0:
        print("pulei um!")
        return [], 0, [], [], []
    top = np.min(considered[0])
    bottom = np.max(considered[0])
    new_path_radius = path_radius
    # region_mask_full = disk(new_path_radius * 2)
    if var_path_width:
        considered_height = bottom - top
        n_lines = considered_height / (path_radius * 2)
        resto, divs = math.modf(n_lines / 2)
        new_path_radius = (considered_height / divs) / 4
        # region_mask_full = disk(new_path_radius * 2)
    # internal_border = mt.erosion(img, kernel_size=path_radius + path_radius_int_ext)
    # internal_border = mt.erosion(img, kernel_size=path_radius)
    internal_border = mt.erosion(img, kernel_img=mask_distancer)
    contours, internal_border_img = mt.detect_contours(
        internal_border, return_img=True, only_external=True
    )
    border_coords = np.where(internal_border_img != 0)
    new_y = np.min(border_coords[0])
    y_list = []
    while new_y < bottom:
        y_list.append(new_y)
        new_y += 4 * new_path_radius
    lines = np.zeros_like(img2)
    y_list = list(map(lambda a: int(round(a)), y_list))
    n_lines = len(y_list)
    if len(y_list) > 2:
        y_list.pop(0)
        y_list.pop(-1)
    for y in y_list:
        line = internal_border_img[y, :]
        if line.any():
            min_x = np.min(np.where(line != 0)[0])
            max_x = np.max(np.where(line != 0)[0])
            for x in np.arange(0, len(line)):
                if min_x <= x <= max_x:
                    lines[y][x] = 1
    return lines, n_lines, internal_border_img, contours, new_path_radius


def internal_weaving_cut(interface_line, path_radius_larg, fail):
    def divide_in_pairs(interface_line, path_radius):
        line_points = pt.img_to_points(mt.hitmiss_ends_v2(interface_line))
        [origin_point, end_point] = line_points
        n_origens = 0
        adjust = 0
        pontos_org = path_tools.line_img_to_freeman_chain(interface_line, origin_point)
        if pt.distance_pts(pontos_org[0], pontos_org[1]) > 3:
            pontos_org.reverse()
            pontos_org = [pontos_org[-1]] + pontos_org[:-1]
        counter_tries = 0
        counter_tries_2 = 0
        while n_origens % 2 == 1 or n_origens == 0 or counter_tries < 5:
            origens_pontos = [pontos_org[0]]
            division_distance = (path_radius * 2) - adjust
            copied_origin = interface_line.copy()
            counter_tries += 1
            while np.sum(copied_origin.astype(np.uint8)) > 0 or counter_tries_2 < 5:
                canvas = np.zeros_like(interface_line, np.uint8)
                centro = origens_pontos[-1]
                area_distance = it.draw_circle(canvas, (centro), division_distance)
                candidates = np.logical_and(area_distance, copied_origin)
                candidates = pt.x_y_para_pontos(np.nonzero(candidates))
                if candidates:
                    new_point = pt.most_distant_from(origens_pontos[-1], candidates)
                    dist = pt.distance_pts(new_point, origens_pontos[-1])
                    dist_from_end = pt.distance_pts(new_point, end_point)
                    if (
                        dist > (3 * division_distance) / 4
                        and dist_from_end >= division_distance
                    ):
                        origens_pontos.append(new_point)
                        copied_origin = np.logical_and(
                            copied_origin, np.logical_not(area_distance)
                        )
                    else:
                        origens_pontos.append(end_point)
                        copied_origin = np.zeros_like(interface_line, np.uint8)
                        break
                    counter_tries_2 += 1
            n_origens = len(origens_pontos)
            adjust += 1
        return origens_pontos, line_points

    div_points, line_points = divide_in_pairs(interface_line, path_radius_larg)
    div_lines = np.zeros_like(fail)
    crossings = []
    extreme_points = [[], [], [], []]
    last_iter = len(div_points) - 1
    for i, div in enumerate(div_points):
        thisdiv = np.zeros_like(fail)
        thisdiv[:, div[1]] = 1
        thisdiv = np.logical_and(thisdiv, fail)
        crossings_img = mt.hitmiss_ends_v2(thisdiv)
        crossings_pts = pt.img_to_points(crossings_img)
        if i == 0:
            div_min = div[1]
            if len(list(filter(lambda x: x in line_points, crossings_pts))) > 0:
                extreme_points[0] = list(
                    filter(lambda x: x in line_points, crossings_pts)
                )[0]
                # extreme_points[3] = list(filter(lambda x: not (x in line_points), crossings_pts))[0]
                extreme_points[3] = pt.most_distant_from(
                    extreme_points[0], crossings_pts
                )
            else:
                break
        elif i == last_iter:
            div_max = div[1]
            fail_inside_divs = fail.copy()
            fail_inside_divs[:, :div_min] = 0
            fail_inside_divs[:, (div_max + 1) :] = 0
            if np.sum(fail_inside_divs) == 0:
                fail_inside_divs = fail.copy()
                fail_inside_divs[:, :div_max] = 0
                fail_inside_divs[:, (div_min + 1) :] = 0
            if len(list(filter(lambda x: x in line_points, crossings_pts))) > 0:
                extreme_points[1] = list(
                    filter(lambda x: x in line_points, crossings_pts)
                )[0]
                # extreme_points[2] = list(filter(lambda x: not (x in line_points), crossings_pts))[0]
                extreme_points[2] = pt.most_distant_from(
                    extreme_points[1], crossings_pts
                )
            else:
                last_point = list(
                    filter(lambda x: not (x in extreme_points), line_points)
                )[0]
                fail_ctr, fail_contour_img = mt.detect_contours(
                    fail_inside_divs, return_img=True, only_external=True
                )
                ctr_list = pt.contour_to_list(fail_ctr)
                if last_point not in ctr_list:
                    included_interface = pt.img_to_points(
                        np.logical_and(fail_contour_img, interface_line)
                    )
                    last_point, _ = pt.closest_point(last_point, included_interface)
                extreme_points[1] = last_point
                extreme_points[2] = extreme_points[1]
        else:
            crossings = crossings + crossings_pts
        div_lines = np.logical_or(div_lines, thisdiv)
    new_zigzags = []
    ends_distances = []
    new_zigzag = np.zeros_like(interface_line)
    if not ([] in extreme_points):
        all_bordacortada_medicao = []
        fail_ctr, fail_contour_img = mt.detect_contours(
            fail_inside_divs, return_img=True, only_external=True
        )
        fail_ctr = pt.contour_to_list(fail_ctr)
        if extreme_points[0] in crossings:
            crossings.remove(extreme_points[0])
        if extreme_points[1] in crossings:
            crossings.remove(extreme_points[1])
        if extreme_points[2] in crossings:
            crossings.remove(extreme_points[2])
        if extreme_points[3] in crossings:
            crossings.remove(extreme_points[3])
        for zig_zag_zag_zig in [0, 1]:
            bordacortada = internal_oscilatory_cut(
                fail_ctr, crossings, extreme_points, zig_zag_zag_zig, fail
            )
            zigzag_candidate = np.logical_or(bordacortada, div_lines)
            _, _, n = it.divide_by_connected(zigzag_candidate)
            if n > 1:
                extreme_points = [
                    extreme_points[0],
                    extreme_points[2],
                    extreme_points[1],
                    extreme_points[3],
                ]
                print("Corrected sequence: C and B inverted")
                bordacortada = internal_oscilatory_cut(
                    fail_ctr, crossings, extreme_points, zig_zag_zag_zig, fail
                )
                zigzag_candidate = np.logical_or(bordacortada, div_lines)
            ends = mt.hitmiss_ends_v2(zigzag_candidate)
            aaaaaa = it.sum_imgs([ends, interface_line])
            aaaaaa_ends = mt.hitmiss_ends_v2(aaaaaa)
            # if np.sum(aaaaaa >= 2) > 0:
            new_zigzags.append(zigzag_candidate)
        # sums = [np.sum(x) for x in new_zigzags]
        # new_zigzag = np.add(new_zigzag, new_zigzags[np.argmax(sums)])

        # invertida = it.image_subtract(fail_contour_img, bordacortada)
        # zigzag_candidate = np.logical_or(invertida, div_lines)
        # ends = mt.hitmiss_ends_v2(zigzag_candidate)
        # aaaaaa = it.sum_imgs([ends, interface_line])
    return new_zigzags


def chamfer_smaller_corners(original_border, origin, region_img, path_radius):
    orig_bord_seq = path_tools.img_to_chain(original_border)
    # limpa se houver contornos extras
    if len(orig_bord_seq) < 4 and len(orig_bord_seq) > 0:
        lens = [len(x) for x in orig_bord_seq]
        orig_bord_seq = orig_bord_seq[lens.index(max(lens))]
    pontas = path_tools.find_curvature_pts(orig_bord_seq)
    # se achar, remove areas proximas as pontas
    if len(pontas) > 0:
        pontas_img = mt.dilation(
            it.points_to_img(pontas, np.zeros_like(region_img)), kernel_size=1
        )
        detected_area_spikes = it.sum_imgs(
            [mt.dilation(origin, kernel_size=1), pontas_img]
        )
        dist_map = distance_transform_edt(region_img)
        norm_dist_map = dist_map / path_radius
        norm_MAT = norm_dist_map * origin
        minimal_areas = np.logical_and(0 < norm_MAT, norm_MAT <= 1)
        labeled_minimals, _, _ = it.divide_by_connected(minimal_areas)
        spikes_exclusion = np.zeros_like(region_img)
        chamfered_border = original_border.copy()
        cutted_corners = np.zeros_like(region_img)
        for mini in labeled_minimals:
            if np.sum(np.logical_and(detected_area_spikes == 2, mini)) > 0:
                spikes_exclusion = mt.dilation(mini, kernel_size=path_radius)
                # ja corta essa ponta
                chamfered_border = it.image_subtract(chamfered_border, spikes_exclusion)
                chamfered_border = it.take_the_bigger_area(chamfered_border)
                cutted_corners = np.logical_or(cutted_corners, mini)
                # e ja costura de volta
                ends_depois_de_aparar = mt.hitmiss_ends_v2(chamfered_border)
                ends_depois_de_aparar_pts = pt.img_to_points(ends_depois_de_aparar)
                if len(ends_depois_de_aparar_pts) > 2:
                    ends_depois_de_aparar_pts = pt.closest_points(
                        ends_depois_de_aparar_pts
                    )
                chamfered_border = np.logical_or(
                    chamfered_border,
                    it.draw_polyline(
                        np.zeros_like(region_img),
                        ends_depois_de_aparar_pts,
                        False,
                    ),
                )
                chamfered_border = it.take_the_bigger_area(chamfered_border)
    return chamfered_border, cutted_corners


def decompose_pol_cont_by_corners(lines_do_limite, min_distance=5, threshold_rel=0.02):
    pontos = path_tools.img_to_chain(lines_do_limite, minimal_seq=8)[0]
    harris_pts = corner_peaks(
        corner_harris(lines_do_limite),
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )
    harris_pts = [list(x) for x in harris_pts]
    pontos_corrigidos = []
    for point in harris_pts:
        if not (point in pontos):
            pnt, _ = pt.closest_point(point, pontos)
            pontos_corrigidos.append(pnt)
        else:
            pontos_corrigidos.append(point)
    pontos_curvatura = pontos_corrigidos
    pontos = path_tools.set_first_pt_in_seq(pontos, pontos_curvatura[0])
    labeled_segments_img = path_tools.colorbyevent(
        pontos, pontos_curvatura, np.zeros_like(lines_do_limite)
    )
    segments_n = max(np.unique(labeled_segments_img))
    segments = []
    for labl in np.add(list(range(segments_n)), 1):
        segment = np.zeros_like(lines_do_limite)
        segment[labeled_segments_img == labl] = 1
        segments.append(segment)
    return segments, labeled_segments_img


def quadrilateralize_segmented_contour(all_segments, origin):
    possible_c1_c2 = np.zeros_like(origin)
    possible_opening = np.zeros_like(origin)
    for segment in all_segments:
        bbbbbb = it.sum_imgs([segment, mt.dilation(origin, kernel_size=2)])
        if np.logical_and(segment, mt.dilation(origin, kernel_size=2)).any():
            possible_opening = np.logical_or(possible_opening, segment)
        else:
            possible_c1_c2 = np.logical_or(possible_c1_c2, segment)
    # o que não é abertura é beirada: c1 e c2
    possible_c1_c2 = it.image_subtract(possible_c1_c2, possible_opening)
    [line_ci1, line_ci2], _, _ = it.divide_by_connected(possible_c1_c2)
    # limitadores_de_ponta = np.zeros_like(region.img
    # o resto são as aberturas
    [closing_tip_1, closing_tip_2], _, _ = it.divide_by_connected(possible_opening)
    internal_pol = it.fill_internal_area(
        it.sum_imgs([line_ci1, line_ci2, closing_tip_1, closing_tip_2]) > 0,
        np.ones_like(origin),
    )
    return line_ci1, line_ci2, closing_tip_1, closing_tip_2, internal_pol


def make_weaving_wide_route(
    region: ZigZag, path_radius, mask_distancer, internal_mask_dist, rest_of_picture
):
    origin = region.origin
    _, _, n = it.divide_by_connected(origin)
    if n > 1:
        origin = it.take_the_bigger_area(origin)
    _, original_border = mt.detect_contours(
        region.img, return_img=True, only_external=True
    )
    # corta as pontas menores que o path_radius
    chamfered_border, cutted_corners = chamfer_smaller_corners(
        original_border, origin, region.img, path_radius
    )
    # agora divide o poligono em segmentos baseados nos angulos
    all_segments_orig, labeled_segmented_orig = decompose_pol_cont_by_corners(
        chamfered_border
    )
    # hora de encontrar a quadrilateralização do nosso contorno:
    line_ci1, line_ci2, closing_tip_1, closing_tip_2, internal_pol = (
        quadrilateralize_segmented_contour(all_segments_orig, origin)
    )
    labeled_quadrilateralization = it.sum_imgs_colored(
        [line_ci1, line_ci2, closing_tip_1, closing_tip_2]
    )

    # dilata para garantir sobreposicao depois erode para fazer as rotas do limite do offset
    total_sobreposition = mt.dilation(internal_pol, kernel_img=mask_distancer)

    A = np.zeros_like(origin)
    distance_map = distance_transform_edt(total_sobreposition)
    markers = labeled_quadrilateralization
    labels = watershed(-distance_map, markers, mask=total_sobreposition)
    internal_divisory = mt.blackhat(rest_of_picture, kernel_size=path_radius)
    internal_divisory = mt.opening(internal_divisory, kernel_size=1)
    eroded = mt.erosion(total_sobreposition, kernel_size=path_radius)
    eroded = it.image_subtract(eroded, internal_divisory)
    _, eroded_border = mt.detect_contours(eroded, return_img=True, only_external=True)
    labeled_eroded = np.multiply(labels, eroded_border)
    line_ci1 = labeled_eroded == 1
    line_ci2 = labeled_eroded == 2
    closing_tip_1 = labeled_eroded == 3
    closing_tip_2 = labeled_eroded == 4
    internal_pol = it.fill_internal_area(
        it.sum_imgs([line_ci1, line_ci2, closing_tip_1, closing_tip_2]) > 0,
        np.ones_like(origin),
    )
    # eroded_seq = path_tools.img_to_chain(eroded)

    # Evitando apagar as divisoes dentro da area

    nova_origin = np.logical_and(origin, internal_pol)
    limitadores_de_ponta = it.sum_imgs([closing_tip_1, closing_tip_2])
    ends_line_ci1 = pt.img_to_points(mt.hitmiss_ends_v2(line_ci1))
    ends_line_ci2 = pt.img_to_points(mt.hitmiss_ends_v2(line_ci2))
    pts_trns_origin = equidistant_in_seq(nova_origin, path_radius, internal_mask_dist)
    pts_trns_ci1 = equidistant_by_proximity(
        line_ci1, pts_trns_origin, path_radius, total_sobreposition
    )
    pts_trns_ci2 = equidistant_by_proximity(
        line_ci2, pts_trns_origin, path_radius, total_sobreposition
    )
    lines_transversais = np.zeros_like(region.img)
    lines_limitrofes = np.zeros_like(region.img)
    for i, point in enumerate(pts_trns_origin):
        if i == 0 or i == len(pts_trns_origin) - 1:
            thisline = it.draw_polyline(
                lines_transversais,
                [pts_trns_ci1[i], pts_trns_origin[i], pts_trns_ci2[i]],
                False,
            )
            if not (np.logical_and(thisline, limitadores_de_ponta).any()):
                lines_limitrofes = np.logical_or(lines_limitrofes, thisline)
        else:
            thisline = it.draw_polyline(
                lines_transversais,
                [pts_trns_ci1[i], pts_trns_origin[i], pts_trns_ci2[i]],
                False,
            )
            lines_transversais = np.logical_or(lines_transversais, thisline)
    new_contour, new_contour_img = mt.detect_contours(
        it.sum_imgs([lines_limitrofes, limitadores_de_ponta, line_ci1, line_ci2]),
        return_img=True,
        only_external=True,
    )
    new_contour = pt.contour_to_list(new_contour)
    organized_points = path_tools.organize_points_cw(ends_line_ci1 + ends_line_ci2)
    ia, _ = pt.closest_point(organized_points[0], ends_line_ci1)
    ib = [x for x in ends_line_ci1 if x != ia][0]
    ic, _ = pt.closest_point(organized_points[2], ends_line_ci2)
    id = [x for x in ends_line_ci2 if x != ic][0]
    extr_int_pts = [ia, ib, ic, id]
    new_contour = path_tools.set_first_pt_in_seq(new_contour, extr_int_pts[0])

    new_zigzag = weaving_zigzag(
        new_contour,
        new_contour_img,
        lines_transversais,
        lines_limitrofes,
        extr_int_pts,
        0,
    )
    new_zigzag = np.logical_or(new_zigzag, lines_transversais)
    new_zigzag = np.logical_or(new_zigzag, cutted_corners)
    new_zigzag, _, _ = sk.create_prune_skel(new_zigzag, 2)
    new_zigzag_b = weaving_zigzag(
        new_contour,
        new_contour_img,
        lines_transversais,
        lines_limitrofes,
        extr_int_pts,
        1,
    )
    new_zigzag_b = np.logical_or(new_zigzag_b, lines_transversais)
    new_zigzag_b = np.logical_or(new_zigzag_b, cutted_corners)
    new_zigzag_b, _, _ = sk.create_prune_skel(new_zigzag_b, 2)
    aaaaaaaa = it.sum_imgs(
        [total_sobreposition, eroded, mt.dilation(new_zigzag, kernel_size=path_radius)]
    )
    region.route = new_zigzag
    region.route_b = new_zigzag_b
    region.trail = mt.dilation(region.route, kernel_size=path_radius)
    region.trail_b = mt.dilation(region.route_b, kernel_size=path_radius)
    # aaaa = it.sum_imgs([region.route, line_ci1,line_ci2,eroded, it.points_to_img(pts_trns_ci1, np.zeros_like(eroded)), region.img, lines_limitrofes, origin_axis, region.origin])
    return new_zigzag, new_zigzag_b


def zig_zag_two_options(
    internal_border_img,
    lines,
    n_lines,
    new_path_radius,
    contours,
    base_frame,
    force_top,
):
    points_external = pt.extreme_points(internal_border_img, force_top=force_top)
    points_internal = pt.extreme_points(lines)
    points_external_img = it.points_to_img(points_external, np.zeros(base_frame))
    new_zigzags = []
    extreme_points = separate_extreme_points(
        points_external, points_internal, internal_border_img, new_path_radius
    )
    for zig_zag_zag_zig in [0, 1]:
        bordacortada = border_cut(
            contours, lines, extreme_points, n_lines, base_frame, zig_zag_zag_zig
        )
        square_mask = getStructuringElement(
            MORPH_RECT, (new_path_radius * 2, new_path_radius * 2)
        )
        new_zigzag = mt.dilation(
            np.logical_or(bordacortada, lines), kernel_img=square_mask
        )
        _, contours2_img = mt.detect_contours(new_zigzag, return_img=True)
        contours2_img = clean_zigzag_over_extrusion(
            contours2_img, new_path_radius, base_frame
        )
        new_zigzags.append(contours2_img)
    return new_zigzags


def zig_zag_third_option(
    self_img, lines, n_lines, new_path_radius, contours, base_frame
):
    new_zigzags = []
    eroded = mt.erosion(self_img, kernel_size=new_path_radius)
    _, bordacortada = mt.detect_contours(eroded, return_img=True)
    new_zigzag = bordacortada
    _, bordacortada_img = mt.detect_contours(new_zigzag, return_img=True)
    new_zigzags.append(bordacortada_img)
    return new_zigzags


def separate_extreme_points(
    points_external, points_internal, internal_border_img, new_path_radius
):
    def too_close(pt1, pt2):
        dist = pt.distance_pts(pt1, pt2)
        if dist < new_path_radius * 4:
            return True
        return False

    def dislocated_pt(idx):
        candidates_area = np.zeros_like(internal_border_img, np.uint8)
        candidates_area[points_internal[idx][0], points_internal[idx][1]] = 1
        candidates_area = mt.dilation(
            candidates_area, kernel_size=(new_path_radius * 5)
        )
        candidates = pt.x_y_para_pontos(
            np.nonzero(np.logical_and(candidates_area, internal_border_img))
        )
        if idx == 0 or idx == 3:
            new_point = candidates[np.argmin(list(map(lambda x: x[0], candidates)))]
        else:
            new_point = candidates[np.argmax(list(map(lambda x: x[0], candidates)))]
        return new_point

    extreme_points = points_external.copy()
    if too_close(points_external[0], points_external[3]):
        extreme_points[0] = dislocated_pt(0)
        extreme_points[3] = dislocated_pt(3)
    if too_close(points_external[1], points_external[2]):
        extreme_points[1] = dislocated_pt(1)
        extreme_points[2] = dislocated_pt(2)
    return extreme_points


def internal_oscilatory_cut(
    new_contour, cruzamentos, extreme_internal_points, sentido, img
):
    borda_cortada = np.zeros_like(img)
    borda_normal = np.zeros_like(img)
    fila = new_contour.copy()
    fila = path_tools.set_first_pt_in_seq(fila, extreme_internal_points[0])
    if sentido:
        fila.reverse()
    counter = 0
    counter_pixels = 0
    last_change = 0
    counter_debug = 0
    for i in np.arange(0, len(fila)):
        borda_normal[fila[i][0]][fila[i][1]] = 1
        counter_pixels += 1
        y = fila[i][0]
        x = fila[i][1]
        ca = [y, x] == extreme_internal_points[0]
        cb = [y, x] == extreme_internal_points[1]
        cc = [y, x] == extreme_internal_points[2]
        cd = [y, x] == extreme_internal_points[3]
        ce = [y, x] in cruzamentos
        cn = False
        if i > 0:
            cn = last_change == fila[i - 1]
        if ce or cb:  # cq
            if cn:
                print(fila[i])
            if not cn:
                counter += 1
            last_change = fila[i]
        if counter % 2 == 0:
            borda_cortada[fila[i][0]][fila[i][1]] = 1
        counter_debug += 1
    return borda_cortada


def all_internal_routes(macro_areas, base_frame):
    rotas = np.zeros(base_frame)
    for r in macro_areas:
        rotas = np.logical_or(rotas, r)
    return rotas


def all_external_trails(bridges, offsets, thin_walls, base_frame):
    rotas = np.zeros(base_frame)
    for r in bridges.cross_over_bridges:
        rotas = np.logical_or(rotas, r.img)
    for r in bridges.offset_bridges:
        rotas = np.logical_or(rotas, r.img)
    for r in offsets:
        rotas = np.logical_or(rotas, r.img)
    for r in thin_walls:
        rotas = np.logical_or(rotas, r.img)
    return rotas


def find_internal_fails(
    original_img,
    base_frame,
    bridges,
    macro_areas,
    mask_full_larg,
    path_radius_larg,
    offsets,
    thin_walls,
):  # calcula e separa em imagens cada area nao coberta pelo ziguezague interno
    from cv2 import boundingRect

    internal_area = np.logical_and(
        original_img,
        np.logical_not(all_external_trails(bridges, offsets, thin_walls, base_frame)),
    )
    limits = boundingRect(all_internal_routes(macro_areas, base_frame).astype(np.uint8))
    internal_trails = np.zeros(base_frame)
    for z in bridges.zigzag_bridges:
        if len(z.route) > 0:
            internal_trails = np.logical_or(internal_trails, z.trail)
    for r in macro_areas:
        trail = mt.dilation(r, kernel_img=mask_full_larg)
        internal_trails = np.logical_or(internal_trails, trail)
    internal_fails = np.logical_and(internal_area, np.logical_not(internal_trails))
    internal_fails = mt.opening(internal_fails, kernel_size=2)
    internal_fails_in_limmits = np.zeros_like(internal_fails)
    [
        internal_fails_in_limmits.__setitem__((y, x), internal_fails[y, x])
        for x in range(limits[0], limits[0] + limits[2])
        for y in range(limits[1], limits[1] + limits[3])
    ]
    separated_imgs, _, _ = it.divide_by_connected(internal_fails_in_limmits)
    separated_imgs = list(
        filter(
            lambda x: it.longer_than(x, (path_radius_larg * 4)),
            separated_imgs,
        )
    )
    return separated_imgs


def zigzag_region_next2fail(separated_fail_imgs, macro_areas, mask_line):
    # determina a qual zzarea deve ser conectada a falha, o criterio eh a maior area de conexao
    fail_reg = {}
    for i, fail in enumerate(separated_fail_imgs):
        reg_list = []
        sums = []
        for j, reg in enumerate(macro_areas):
            vertical_trail = mt.dilation(reg, kernel_img=mask_line)
            area_conjunta = np.logical_and(fail, vertical_trail)
            if np.sum(area_conjunta) > 0:
                reg_list.append(j)
                sums.append(np.sum(area_conjunta))
            if sums:
                fail_reg.update({i: reg_list[np.argmax(sums)]})
    return fail_reg


def connect_fails_to_zigzags(old_zigzag, separated_fail_imgs, path_radius_larg):
    def extend_until_it_touches(fail_img, sentido):
        # fail_img = separated_fail_imgs[j]
        num_parts = 99
        extension = 0
        counter_tries = 0
        if path_radius_larg % 2 == 0:
            extension = extension + 1
        interface_line = np.zeros_like(fail_img)
        while (
            (num_parts > 1 or np.sum(interface_line) == 0)
            and extension < path_radius_larg * 2.5
            or counter_tries < 5
        ):
            mask_line = np.zeros(
                [path_radius_larg + extension, path_radius_larg + extension]
            )
            mask_line[:, int((path_radius_larg + extension) / 2)] = 1
            selective_kernel = mask_line.copy()
            if sentido == "down":
                selective_kernel[int((path_radius_larg + extension) / 2) + 1 :] = 0
            elif sentido == "up":
                selective_kernel[: int((path_radius_larg + extension) / 2)] = 0
            interface_line_a = np.add(
                mt.dilation(fail_img.astype(np.uint8), kernel_img=selective_kernel),
                old_zigzag,
            )
            interface_line = interface_line_a == 2
            _, labeled, num_parts = it.divide_by_connected(interface_line)
            extension = extension + 2
            counter_tries += 1
        # line_points = pt.img_to_points(mt.hitmiss_ends_v2(interface_line))
        # dilated_route = mt.dilation(fail_img, kernel_img=mask_line)
        # fail_contact = np.add(dilated_route, fail_img.astype(np.uint8)) == 2
        if num_parts > 1:
            if extension >= path_radius_larg * 2:
                interface_line = it.take_the_bigger_area(labeled.astype(bool))
            else:
                interface_line = np.zeros_like(fail_img)
        return interface_line, extension

    contacts_pts = []
    all_connected_fails = np.zeros_like(old_zigzag)
    connected_fails = []
    for j, fail_img in enumerate(separated_fail_imgs):  # aqui comeca uma operacao nova
        contact_down, extensions_down = extend_until_it_touches(fail_img, "up")
        contact_up, extensions_up = extend_until_it_touches(fail_img, "down")
        lens = [len(pt.img_to_points(contact_down)), len(pt.img_to_points(contact_up))]
        zigzag_contact = [contact_down, contact_up][(np.argmax(lens))]
        if np.sum(zigzag_contact) > 0:
            _, contact_xs = np.nonzero(zigzag_contact)
            contact_xs = [np.min(contact_xs), np.max(contact_xs)]
            pts_zigzag_contact = pt.img_to_points(zigzag_contact)
            pts_zigzag_contact = sorted(pts_zigzag_contact, key=lambda x: x[1])
            pts_zigzag_contact_extremes = [
                pts_zigzag_contact[0],
                pts_zigzag_contact[-1],
            ]
            contacts_pts.append(pts_zigzag_contact_extremes)
            new_fail = copy.deepcopy(fail_img)
            new_fail[:, : contact_xs[0]] = 0  # zera tudo antes
            new_fail[:, (contact_xs[1] + 1) :] = 0  # zera tudo depois
            line_kernel = disk(path_radius_larg)
            line_image = np.zeros_like(line_kernel)
            center_row = path_radius_larg  # line do centro
            line_image[center_row, :] = 1  # Preenche a line do centro
            new_fail = mt.opening(new_fail, kernel_img=line_image)
            if np.sum(new_fail) > 0:
                pts_fail_contact = pt.img_to_points(new_fail)
                pts_fail_contact = sorted(pts_fail_contact, key=lambda x: x[1])
                pts_fail_contact_extremes = [pts_fail_contact[0], pts_fail_contact[-1]]
                canvas = np.zeros_like(old_zigzag)
                canvas = it.draw_line(zigzag_contact, *pts_fail_contact_extremes)
                canvas = it.draw_line(
                    canvas, pts_zigzag_contact_extremes[0], pts_fail_contact_extremes[0]
                )
                canvas = it.draw_line(
                    canvas, pts_zigzag_contact_extremes[1], pts_fail_contact_extremes[1]
                )
                _, connected_fail = mt.detect_contours(
                    np.logical_or(new_fail, canvas), return_img=True, only_external=True
                )
                connected_fail = it.fill_internal_area(
                    connected_fail, np.ones_like(canvas), True
                )
                connected_fails.append(it.sum_imgs([connected_fail, zigzag_contact]))
                all_connected_fails = np.logical_or(all_connected_fails, connected_fail)
        else:
            pass
    eroded_connected_fails = []
    all_eroded_connected_fails = np.zeros_like(old_zigzag)
    for cf in connected_fails:
        area_center = pt.points_center(pt.img_to_points(cf > 0))
        line_ys = [point[0] for point in pt.img_to_points(cf > 1)]
        extension = 0
        if path_radius_larg % 2 == 0:
            extension = extension + 1
        mask_line = np.zeros(
            [path_radius_larg + extension, path_radius_larg + extension]
        )
        mask_line[:, int((path_radius_larg + extension) / 2)] = 1
        mask_circ = disk(path_radius_larg)
        selective_kernel = mask_circ.copy()
        the_other = mask_line.copy()
        if area_center[0] > (max(line_ys) + min(line_ys)) / 2:
            selective_kernel[: int((path_radius_larg))] = 0
            the_other[int((path_radius_larg / 2)) + 1 :] = 0
        else:
            selective_kernel[int((path_radius_larg)) + 1 :] = 0
            the_other[: int((path_radius_larg / 2))] = 0
        eroded_fail_bef = mt.erosion(
            (cf > 0).astype(np.uint8), kernel_img=selective_kernel
        )
        extremes_dilated = mt.dilation(
            mt.hitmiss_ends_v2(cf > 1), kernel_size=path_radius_larg
        )
        reduced_contact = it.image_subtract(cf > 1, extremes_dilated)
        eroded_fail = it.sum_imgs(
            [eroded_fail_bef, mt.dilation(reduced_contact, the_other)]
        )
        eroded_closed_fail = mt.closing(eroded_fail, kernel_size=int(path_radius_larg))
        if np.sum(eroded_closed_fail) > 0:
            eroded_closed_fail = it.take_the_bigger_area(
                eroded_closed_fail.astype(bool)
            )
            eroded_connected_fails.append(eroded_closed_fail)
            all_eroded_connected_fails = np.logical_or(
                all_eroded_connected_fails, eroded_closed_fail
            )
    separated, _, _ = it.divide_by_connected(all_eroded_connected_fails)
    new_conections = []

    cleanned_separated = []
    for fail in separated:
        interface_line = np.logical_and(fail, old_zigzag)
        line_points = pt.img_to_points(mt.hitmiss_ends_v2(interface_line))
        if len(line_points) != 2:
            new_line = it.take_the_bigger_area(interface_line)
            other_lines = it.image_subtract(interface_line, new_line)
            a = it.image_subtract(
                fail, mt.dilation(other_lines, kernel_size=path_radius_larg)
            )
            b = mt.opening(a, kernel_size=path_radius_larg)
            if np.sum(b) == 0:
                new_line = interface_line
                new_separated = fail
            else:
                c = it.take_the_bigger_area(b)
                d = np.logical_or(c, new_line)
                e = mt.closing(d, kernel_size=int(path_radius_larg * 4))
                new_line = np.logical_and(e, old_zigzag)
                new_separated = np.logical_or(e, new_line)
            # line_points = pt.img_to_points(mt.hitmiss_ends_v2(new_line))
        else:
            new_line = interface_line
            new_separated = fail
        # new_separated = it.image_subtract(new_separated,new_line)
        new_conections.append(new_line)
        cleanned_separated.append(new_separated)
    # aaa = it.sum_imgs(separated + old_zigzag)
    # aaaa = it.sum_imgs(cleanned_separated + old_zigzag)
    return cleanned_separated, new_conections


def equidistant_in_seq(line, path_radius, internal_mask_dist, origin_pt=[]):
    line_img = copy.deepcopy(line)
    n_origens = 0
    adjust = 0
    endpoints_img = mt.hitmiss_ends_v2(line_img.astype(bool))
    endpoints = pt.img_to_points(endpoints_img)
    if len(endpoints) > 2:
        # i = 2
        # while len(endpoints) > 2:
        #     line_img, aa, aaa = sk.prune(line_img, size=path_radius * i)
        #     i += 1
        #     endpoints_img = mt.hitmiss_ends_v2(line_img.astype(bool))
        #     endpoints = pt.img_to_points(endpoints_img)
        a, aa = sk.segment_skeleton(line_img)
        lens = [len(x) for x in aa]
        idx1 = lens.index(max(lens))
        lst_copy = lens.copy()
        lst_copy[idx1] = float("-inf")  # Remove the biggest
        idx2 = lst_copy.index(max(lst_copy))
        idxs = set([idx1, idx2])
        toremove = tuple(x for i, x in enumerate(aa) if i not in idxs)
        line_toremove = np.zeros_like(line_img)
        for lin in toremove:
            b = pt.invert_x_y([list(x[0]) for x in lin])
            c = it.points_to_img(b, np.zeros_like(line_img))
            line_toremove = it.sum_imgs([line_toremove, c])
        line_img_cut = it.image_subtract(line_img, line_toremove)
        line_img, _, _ = sk.prune(line_img_cut, it_prune=2)
        endpoints_img = mt.hitmiss_ends_v2(line_img.astype(bool))
        endpoints = pt.img_to_points(endpoints_img)
        print("tentando cortar os galhos menores")
    if len(origin_pt) > 0:
        first, _ = pt.closest_point(origin_pt, endpoints)
        last = list(filter(lambda x: x != first, endpoints))[0]
    else:
        np.sort(endpoints, 1)[0]
        [first, last] = endpoints
    pontos_org = path_tools.line_img_to_freeman_chain(line_img, first)
    line_img = it.points_to_img(pontos_org, np.zeros_like(line_img))
    while n_origens % 2 != 1:
        origens_pontos = [pontos_org[0]]
        division_distance = (path_radius * 2) - adjust
        copied_origin = copy.deepcopy(line_img)
        while np.sum(copied_origin.astype(np.uint8)) > 0:
            canvas = np.zeros_like(line_img, np.uint8)
            centro = origens_pontos[-1]
            area_distance = it.draw_circle(canvas, centro, division_distance)
            candidates = np.logical_and(area_distance, copied_origin)
            candidates = pt.img_to_points(candidates)
            if candidates:
                new_point_candidates = pt.most_distant_from(
                    origens_pontos[-1], candidates, give_second=True
                )
                indexes = [pontos_org.index(x) for x in new_point_candidates]
                new_point = pontos_org[np.min(indexes)]
                dist = pt.distance_pts(new_point, origens_pontos[-1])
                if dist > (3 * division_distance) / 4:
                    origens_pontos.append(new_point)
            else:
                pass
            copied_origin = np.logical_and(copied_origin, np.logical_not(area_distance))
        n_origens = len(origens_pontos)
        adjust += 1
    origens_pontos[-1] = last
    origens_pontos = path_tools.rotate_if_last_is_closest(origens_pontos)
    aaa = it.sum_imgs_colored(
        [it.points_to_img([x], np.zeros_like(line_img)) for x in origens_pontos]
    )
    return origens_pontos


def equidistant_by_proximity(line_img, origin_lst, path_radius, img):
    # n_origens = 0
    endpoints_img = mt.hitmiss_ends_v2(line_img.astype(bool))
    endpoints = pt.img_to_points(endpoints_img)
    first, _ = pt.closest_point(origin_lst[0], endpoints)
    last = list(filter(lambda x: x != first, endpoints))[0]
    if len(endpoints) > 2:
        newlineimg, _, _ = sk.prune(line_img, it_prune=1)
        endpoints_img = mt.hitmiss_ends_v2(newlineimg.astype(bool))
        endpoints = pt.img_to_points(endpoints_img)
        first, _ = pt.closest_point(origin_lst[0], endpoints)
        last = list(filter(lambda x: x != first, endpoints))[0]
    elif len(endpoints) < 2:
        first, _ = pt.closest_point(origin_lst[0], endpoints)
        last = first
    origens_pontos = []
    for origin_pt in origin_lst:
        copied_origin = copy.deepcopy(line_img)
        canvas = np.zeros_like(line_img, np.uint8)
        centro = origin_pt
        area_distance = it.draw_circle(canvas, centro, path_radius)
        candidate, _ = pt.closest_point(centro, pt.img_to_points(copied_origin))
        origens_pontos.append(candidate)
        copied_origin = np.logical_and(copied_origin, np.logical_not(area_distance))
        # n_origens = len(origens_pontos)
    origens_pontos[0] = first
    origens_pontos[-1] = last
    origens_pontos = path_tools.rotate_if_last_is_closest(origens_pontos)
    aaa = it.points_to_img(origens_pontos, np.zeros_like(line_img))
    return origens_pontos


def internal_cut(new_contour, lines, extreme_internal_points, sentido):
    eip = copy.deepcopy(extreme_internal_points)
    if sentido:
        eip = [
            extreme_internal_points[1],
            extreme_internal_points[0],
            extreme_internal_points[3],
            extreme_internal_points[2],
        ]
    fila = new_contour.copy()
    fila = path_tools.set_first_pt_in_seq(fila, eip[0])
    ordem_na_fila_pontos = []
    for p in fila:
        if p in eip:
            ordem_na_fila_pontos.append(p)
    ordem_na_fila_pontos_idx = [ordem_na_fila_pontos.index(x) for x in eip]
    if ordem_na_fila_pontos_idx[1] == 3:
        fila.reverse()
        fila = path_tools.set_first_pt_in_seq(fila, eip[0])
    borda_cortada = np.zeros_like(lines)
    borda_normal = np.zeros_like(lines)
    lines_transversais = lines.copy()
    counter = 0
    counter_pixels = 0
    last_change = 0
    B = it.points_to_img(new_contour, np.zeros_like(lines_transversais))
    D = np.logical_and(lines_transversais, B)
    cruzamentos = pt.x_y_para_pontos(np.nonzero(D))
    for i in np.arange(0, len(fila)):
        borda_normal[fila[i][0]][fila[i][1]] = 1
        counter_pixels += 1
        y = fila[i][0]
        x = fila[i][1]
        ca = [y, x] == eip[0]
        cb = [y, x] == eip[1]
        cc = [y, x] == eip[2]
        cd = [y, x] == eip[3]
        ce = [y, x] in cruzamentos
        cn = False
        if i > 0:
            cn = last_change == fila[i - 1]
        if ce or cc or ca:  # cq
            if not cn:
                counter += 1
            last_change = fila[i]
        if counter % 2 == 0:
            borda_cortada[fila[i][0]][fila[i][1]] = 1
    return borda_cortada


def weaving_zigzag(
    new_contour,
    new_contour_img,
    lines_transversais,
    lines_limitrofes,
    extr_int_pts,
    sentido,
):
    cutted_border = internal_cut(new_contour, lines_transversais, extr_int_pts, sentido)
    new_zigzag = np.logical_or(lines_transversais, cutted_border)
    new_zigzag = np.logical_or(new_zigzag, lines_limitrofes)
    _, _, n = it.divide_by_connected(new_zigzag)
    reference_points_b = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(new_zigzag)))
    if n > 1:
        extr_int_pts = [
            extr_int_pts[0],
            extr_int_pts[2],
            extr_int_pts[1],
            extr_int_pts[3],
        ]
        print("   Corrected sequence: b and c inverted")
        new_zigzag = internal_cut(
            new_contour, lines_transversais, extr_int_pts, sentido
        )
        new_zigzag = np.logical_or(new_zigzag, lines_limitrofes)
    _, _, n = it.divide_by_connected(mt.hitmiss_ends_v2(new_zigzag))
    if n <= 1:
        cccc = mt.closing(new_zigzag, kernel_size=1)
        ddd, _, _ = sk.create_prune_skel(cccc, 1)
        _, eee, n = it.divide_by_connected(mt.hitmiss_ends_v2(ddd))
        if n > 1:
            new_zigzag = ddd
    if len(reference_points_b) < 2:
        print("   ERROR: no solution yet")
    return new_zigzag
