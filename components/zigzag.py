import concurrent.futures
import itertools
import math
import copy

from keyring import set_keyring
import numpy as np
from components import path_tools
from components import images_tools as it
from components import skeleton as sk
from components import points_tools as pt
from components import morphology_tools as mt
from typing import List
from skimage.measure import label
from skimage.morphology import disk
from cv2 import getStructuringElement, MORPH_RECT
from scipy.spatial import distance_matrix, distance
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import flood_fill
from timer import Timer


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
        self.all_linhas_corte = []
        self.linhas_corte = []
        self.labeled_monotonic_regions = []
        self.regions = []
        self.evento_limite = []

    def create_shadow_img(self, path_radius):
        """O conceito da imagem de sombras é uma operação de labeling para cada linha dos dois lados
        dessa forma conseguimeos ver as "sombras" se iluminarmos a mesma imagem nos dois sentidos
        """
        img = copy.deepcopy(self.img)
        shadows = []
        while np.sum(img) > 0:
            shadow_img_dir = np.zeros_like(img, int)
            shadow_img_esq = np.zeros_like(img, int)
            for index, linha in enumerate(img[0:]):
                labeled_line_dir = label(linha, connectivity=1)
                labeled_line_esq = label(np.flip(linha), connectivity=1)
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
        shadows_after_rejoined, labeled_shadows, num = it.divide_by_connected(new_shadow_img)
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

    def scan_monotonic(self, path_radius, base_frame, ideal_sum):
        shadow_img, areas = self.create_shadow_img(path_radius)
        areas = it.neighborhood_imgs(areas)
        if len(areas) == 1:
            if np.sum(mt.opening(self.img, kernel_size=path_radius*2)) > 0:
                self.labeled_monotonic_regions = self.img
                self.areas_somadas = self.img
                self.regions.append(ZigZag(0, self.labeled_monotonic_regions))
        else:
            monotonic_regions, self.labeled_monotonic_regions, self.areas_somadas = (
                self.unite_small_monotonic_areas(
                    areas, path_radius, base_frame, ideal_sum
                )
            )
            for i, mr in enumerate(monotonic_regions):
                if np.sum(mt.opening(mr.img, kernel_size=path_radius)) > 0:
                    self.regions.append(ZigZag(i, mr.img))
        return

    def trace_divisions(self, rest_of_picture_f2, base_frame, limites):
        mudancas_linha = []
        labeled_img = []
        for i in np.arange(0, limites[1]):
            linha = rest_of_picture_f2[i]
            _, labeled_line, changes = it.divide_by_connected(linha)
            mudancas_linha.append(changes)
            labeled_img.append(labeled_line)
        evento_limite = []
        linha_ant = 0
        for i in np.arange(0, len(mudancas_linha)):
            linha_now = mudancas_linha[i]
            if linha_now > linha_ant:
                evento_limite[-1] = 1
                evento_limite.append(0)
            elif linha_now < linha_ant:
                evento_limite.append(-1)
            elif linha_now == linha_ant and i > 0:
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
            linha_ant = linha_now
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
        linhas, _, n_linhas_corte = it.divide_by_connected(all_div_lines)
        procedencia = 0
        for i in np.arange(0, n_linhas_corte):
            linha_points = np.nonzero(linhas[i])
            y_line = np.unique(linha_points[0])
            if len(y_line) > 1:
                coords = pt.x_y_para_pontos(linha_points)
                n_points = []
                for y in y_line:
                    n_points.append(len(list(filter(lambda x: x[0] == y, coords))))
                y_line = y_line[np.argmax(n_points)]
                linha_points = list(filter(lambda x: x[0] == y_line, coords))
                linha_points = [
                    [x[0] for x in linha_points],
                    [x[1] for x in linha_points],
                ]
            xs_line = [
                np.min(np.unique(linha_points[1])),
                np.max(np.unique(linha_points[1])),
            ]
            for evento in evento_limite_reduct:
                if y_line == evento[0]:
                    procedencia = evento[1]
            self.linhas_corte.append(
                DivisionLine(i, linhas[i], procedencia, y_line, xs_line)
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

        monotonic_regions = copy.deepcopy(areas)
        for i in np.arange(0, len(monotonic_regions)):
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
    def __init__(self, name, img):
        self.name = name
        self.img = img
        self.route = []
        self.trail = []
        self.center = []
        self.remove = False
        self.region_path_radius = 0


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
        path_radius_internal,
        mask_full_int,
        zigzags,
        bridges,
        offsets,
        thin_walls,
        internal_weaving,
    ):
        if internal_weaving:
            mask_line = np.zeros(np.add(mask_full_int.shape, [4, 4]))
            mask_line[:, int(mask_full_int.shape[0] / 2)] = 1
            old_zigzag = all_internal_routes(macro_areas, base_frame)
            with Timer("Pegando as falhas internas"):
                separated_fail_imgs = find_internal_fails(
                    original_img,
                    base_frame,
                    bridges,
                    macro_areas,
                    mask_full_int,
                    path_radius_internal,
                    offsets,
                    thin_walls,
                )
            with Timer("Achando os contatos"):
                connected_fails, interface_lines = connect_fails_to_zigzags(old_zigzag, separated_fail_imgs, path_radius_internal)

            with Timer("criando os weavings"):
                # aaaa = it.sum_imgs(separated_fail_imgs+separated_connected_fails)
                fail_internal_zigzags = []
                for i, fail in enumerate(connected_fails):
                    try:
                        fail_internal_zigzags.append(internal_weaving_cut(interface_lines[i], path_radius_internal, fail))
                    except:
                        print("falhou um weaving aqui!")
                        pass
                if len(interface_lines) > 0:
                    all_new_zigzags = it.image_subtract(old_zigzag.astype(bool), it.sum_imgs(interface_lines))
                    all_new_zigzags = it.sum_imgs(all_new_zigzags + fail_internal_zigzags).astype(bool)
                else:
                    all_new_zigzags = old_zigzag
            new_macro_areas, _, _ = it.divide_by_connected(all_new_zigzags)
        else:
            all_new_zigzags = np.zeros(base_frame)
            for r in macro_areas:
                all_new_zigzags = np.logical_or(all_new_zigzags, r)
            new_macro_areas = macro_areas
        all_new_zigzags,_,_ = sk.create_prune_divide_skel(all_new_zigzags, path_radius_internal)
        return new_macro_areas, all_new_zigzags

    def connect_island_zigzags(self, path_radius_internal, mask_full_int, base_frame):
        interfaces, centers, interface_types = path_tools.find_points_of_contact(
            list(self.zigzags_mst.edges),
            path_radius_internal,
            mask_full_int,
            self.regions,
        )
        unified_zigzags = path_tools.draw_the_links(
            self,
            self.zigzags_mst,
            base_frame,
            interfaces,
            centers,
            path_radius_internal,
        )
        macro_area_list, _, _ = it.divide_by_connected(unified_zigzags)
        self.all_zigzags = unified_zigzags
        self.macro_areas = macro_area_list
        return

    def find_monotonic(self, rest_of_picture_f3, base_frame, path_radius, ideal_sum):
        sub_regions: Subregion = []
        separated_imgs, labeled, num = it.divide_by_connected(rest_of_picture_f3)
        for i in np.arange(0, num):
            sub_regions.append(Subregion(i, separated_imgs[i]))
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

    def make_routes_z(self, base_frame, path_radius, path_radius_int_ext):
        # def make_zigzag_route(region: ZigZag):
        for region in self.regions:
            region.center = pt.points_center(
                pt.contour_to_list(mt.detect_contours(region.img))
            )
            zig_options = []
            lines, n_lines, internal_border_img, contours, new_path_radius = (
                cut_in_lines(region.img, path_radius, path_radius_int_ext, var_path_width=0)
            )
            filled = it.fill_internal_area(
                internal_border_img.astype(np.uint8), np.ones_like(internal_border_img)
            )
            opened = mt.opening(filled, kernel_size=path_radius)
            with Timer("fazendo as tres opções:"):
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
                    region.img, lines, n_lines, new_path_radius, contours, base_frame
                )
                zig_options.append(new_zigzag_c)
            with Timer("escolhendo entre elas:"):
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
        pixel_linhas = lines[y][x]
        ca = [y, x] == points[0]
        cb = [y, x] == points[1]
        cc = [y, x] == points[2]
        cd = [y, x] == points[3]
        ce = pixel_linhas == 1
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
    no_failure_axis_img, _, _ = sk.create_prune_divide_skel(no_failure, new_path_radius)
    no_failure_axis_path, no_failure_axis_path_img = mt.detect_contours(
        no_failure_axis_img, return_img=True, only_external=True
    )
    path_candidates, _, _ = it.divide_by_connected(no_failure_axis_path_img)
    path = path_candidates[0]
    return path


def cut_in_lines(img, path_radius, path_radius_int_ext, var_path_width=0):
    img2 = mt.opening(img, kernel_size=(path_radius * 2))
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
        n_linhas = considered_height / (path_radius * 2)
        resto, divs = math.modf(n_linhas / 2)
        new_path_radius = (considered_height / divs) / 4
        # region_mask_full = disk(new_path_radius * 2)
    internal_border = mt.erosion(img2, kernel_size=path_radius_int_ext)
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

def internal_weaving_cut(interface_line, path_radius_internal, fail):
    def divide_in_pairs(interface_line, path_radius):
        line_points = pt.img_to_points(mt.hitmiss_ends_v2(interface_line))
        [origin_point, end_point] = line_points
        n_origens = 0
        adjust = 0
        pontos_org = path_tools.line_img_to_freeman_chain(interface_line, origin_point)
        if pt.distance_pts(pontos_org[0], pontos_org[1]) > 3:
            pontos_org.reverse()
            pontos_org = [pontos_org[-1]] + pontos_org[:-1]
        while n_origens % 2 == 1 or n_origens == 0:
            origens_pontos = [pontos_org[0]]
            division_distance = (path_radius * 2) - adjust
            copied_origin = interface_line.copy()
            while np.sum(copied_origin.astype(np.uint8)) > 0:
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
            n_origens = len(origens_pontos)
            adjust += 1
        return origens_pontos, line_points


    div_points, line_points = divide_in_pairs(interface_line, path_radius_internal)
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
                extreme_points[0] = list(filter(lambda x: x in line_points, crossings_pts))[0]
                extreme_points[3] = list(filter(lambda x: not (x in line_points), crossings_pts))[0]
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
                extreme_points[1] = list(filter(lambda x: x in line_points, crossings_pts))[0]
                extreme_points[2] = list(filter(lambda x: not (x in line_points), crossings_pts))[0]
            else:
                last_point = list(filter(lambda x: not (x in extreme_points), line_points))[0]
                fail_ctr, fail_contour_img = mt.detect_contours(fail_inside_divs,return_img=True,only_external=True)
                ctr_list = pt.contour_to_list(fail_ctr)
                if last_point not in ctr_list:
                    included_interface = pt.img_to_points(np.logical_and(fail_contour_img, interface_line))
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
            bordacortada = internal_oscilatory_cut(fail_ctr,crossings,extreme_points,zig_zag_zag_zig,fail)
            zigzag_candidate = np.logical_or(bordacortada, div_lines)
            _, _, n = it.divide_by_connected(zigzag_candidate)
            if n > 1:
                extreme_points = [
                    extreme_points[0],
                    extreme_points[2],
                    extreme_points[1],
                    extreme_points[3],
                ]
                print("sequncia corrigida, b e c trocado")
                bordacortada = internal_oscilatory_cut(
                    fail_ctr,crossings,extreme_points,zig_zag_zag_zig,fail
                )
                zigzag_candidate = np.logical_or(bordacortada, div_lines)
            ends = mt.hitmiss_ends_v2(zigzag_candidate)
            aaaaaa = it.sum_imgs([ends, interface_line])
            if np.sum(aaaaaa>=2)>0:
                new_zigzags.append(zigzag_candidate)
        sums = [np.sum(x) for x in new_zigzags]
        new_zigzag = np.add(new_zigzag, new_zigzags[np.argmax(sums)])
    return new_zigzag



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
    eiorja = it.points_to_img(points_external, np.zeros(base_frame))
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
    mask_full_int,
    path_radius_internal,
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
        trail = mt.dilation(r, kernel_img=mask_full_int)
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
            lambda x: it.comprimento_maior_que(x, (path_radius_internal * 4)),
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


def connect_fails_to_zigzags(
    old_zigzag, separated_fail_imgs, path_radius_internal
):
    def extend_until_it_touches(fail_img, sentido):
        # fail_img = separated_fail_imgs[j]
        num_parts = 99
        extension = 0
        if path_radius_internal%2 == 0:
            extension = extension+1
        interface_line = np.zeros_like(fail_img)
        while (num_parts > 1 or np.sum(interface_line) == 0) and extension < path_radius_internal*2.5:
            mask_line = np.zeros([path_radius_internal + extension, path_radius_internal + extension])
            mask_line[:, int((path_radius_internal + extension)/2)] = 1
            selective_kernel = mask_line.copy()
            if sentido == "baixo":
                selective_kernel[int((path_radius_internal+ extension)/2) + 1 :] = 0
            elif sentido == "cima":
                selective_kernel[: int((path_radius_internal+ extension)/2)] = 0
            interface_line_a = np.add(mt.dilation(fail_img.astype(np.uint8), kernel_img=selective_kernel),old_zigzag)
            interface_line = interface_line_a == 2
            _, labeled, num_parts = it.divide_by_connected(interface_line)
            extension = extension + 2
        # line_points = pt.img_to_points(mt.hitmiss_ends_v2(interface_line))
        # dilated_route = mt.dilation(fail_img, kernel_img=mask_line)
        # fail_contact = np.add(dilated_route, fail_img.astype(np.uint8)) == 2
        if num_parts>1 :
            if extension >= path_radius_internal*2:
                interface_line = it.take_the_bigger_area(labeled.astype(bool))
            else:
                interface_line = np.zeros_like(fail_img)
        return interface_line, extension

    contacts_imgs = []
    contacts_pts = []
    zigzag_contact_lines_imgs = []
    zigzag_contact_lines_pts = []
    all_connected_fails = np.zeros_like(old_zigzag)
    connected_fails = []
    for j, fail_img in enumerate(separated_fail_imgs):  # aqui comeca uma operacao nova
        contact_down, extensions_down = extend_until_it_touches(fail_img, "baixo")
        contact_up, extensions_up = extend_until_it_touches(fail_img, "cima")
        lens = [
            len(pt.img_to_points(contact_down)),
            len(pt.img_to_points(contact_up))
        ]
        zigzag_contact = [contact_down, contact_up][(np.argmax(lens))]
        if np.sum(zigzag_contact) > 0:
            _ , contact_xs = np.nonzero(zigzag_contact)
            contact_xs = [np.min(contact_xs), np.max(contact_xs)]
            pts_zigzag_contact = pt.img_to_points(zigzag_contact)
            pts_zigzag_contact = sorted(pts_zigzag_contact, key=lambda x: x[1])
            pts_zigzag_contact_extremes = [pts_zigzag_contact[0], pts_zigzag_contact[-1]]
            contacts_pts.append(pts_zigzag_contact_extremes)
            new_fail = copy.deepcopy(fail_img)
            new_fail[:, : contact_xs[0]] = 0  # zera tudo antes
            new_fail[:, (contact_xs[1] + 1) :] = 0  # zera tudo depois
            line_kernel = disk(path_radius_internal)
            line_image = np.zeros_like(line_kernel)
            center_row = path_radius_internal  # Linha do centro
            line_image[center_row, :] = 1  # Preenche a linha do centro
            new_fail = mt.opening(new_fail, kernel_img=line_image)
            if np.sum(new_fail)>0:
                pts_fail_contact = pt.img_to_points(new_fail)
                pts_fail_contact = sorted(pts_fail_contact, key=lambda x: x[1])
                pts_fail_contact_extremes = [pts_fail_contact[0], pts_fail_contact[-1]]
                canvas = np.zeros_like(old_zigzag)
                canvas = it.draw_line(zigzag_contact, *pts_fail_contact_extremes)
                canvas = it.draw_line(canvas, pts_zigzag_contact_extremes[0], pts_fail_contact_extremes[0])
                canvas = it.draw_line(canvas, pts_zigzag_contact_extremes[1], pts_fail_contact_extremes[1])
                _, connected_fail = mt.detect_contours(np.logical_or(new_fail, canvas),return_img=True, only_external=True)
                connected_fail = it.fill_internal_area(connected_fail, np.ones_like(canvas))
                connected_fails.append(it.sum_imgs([connected_fail, zigzag_contact]))
                all_connected_fails = np.logical_or(connected_fails, connected_fail)
        else:
            pass
    eroded_connected_fails = []
    all_eroded_connected_fails =  np.zeros_like(old_zigzag)
    for cf in connected_fails:
        area_center = pt.points_center(pt.img_to_points(cf>0))
        line_ys = [point[0] for point in pt.img_to_points(cf>1)]
        extension = 0
        if path_radius_internal%2 == 0:
            extension = extension+1
        mask_line = np.zeros([path_radius_internal+extension, path_radius_internal+extension])
        mask_line[:, int((path_radius_internal+extension)/2)] = 1
        mask_circ = disk(path_radius_internal)
        selective_kernel = mask_circ.copy()
        the_other = mask_line.copy()
        if area_center[0] > (max(line_ys) + min(line_ys))/2:
            selective_kernel[: int((path_radius_internal))] = 0
            the_other[int((path_radius_internal/2)) + 1 :] = 0
        else:
            selective_kernel[int((path_radius_internal)) + 1 :] = 0
            the_other[: int((path_radius_internal/2))] = 0
        eroded_fail_bef = mt.erosion((cf>0).astype(np.uint8), kernel_img=selective_kernel)
        extremes_dilated = mt.dilation(mt.hitmiss_ends_v2(cf>1),kernel_size=path_radius_internal)
        reduced_contact = it.image_subtract(cf>1,extremes_dilated)
        eroded_fail = it.sum_imgs([eroded_fail_bef, mt.dilation(reduced_contact,the_other)])
        eroded_connected_fails.append(eroded_fail)
        all_eroded_connected_fails = np.logical_or(all_eroded_connected_fails, eroded_fail)
    separated, _, _ = it.divide_by_connected(all_eroded_connected_fails)
    new_conections = []
    for fail in separated:
        new_conections.append(np.logical_and(fail, old_zigzag))
    return separated, new_conections
