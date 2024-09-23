import concurrent.futures
import math
from typing import List
import copy
from skimage.measure import label
from skimage.morphology import disk
from cv2 import getStructuringElement, MORPH_RECT
from scipy.spatial import distance_matrix, distance
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import flood_fill
import numpy as np
from components import path_tools
from components import images_tools as it
from components import skeleton as sk
from components import points_tools as pt
from components import morphology_tools as mt


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

        def largura_ok(img, path_radius):
            points = pt.img_to_points(img)
            xs = [x[1] for x in points]
            dif = max(xs) - min(xs)
            if dif > 6 * path_radius:
                return True
            return False

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
            all_ones = np.logical_or(
                shadow_img_dir_dd.astype(int), shadow_img_esq_dd.astype(int)
            )
            separated, _, num = it.divide_by_connected(all_ones)
            filtered_separated = []
            for s in separated:
                for index, linha in enumerate(s[0:]):
                    labeled_line_dir = label(linha, connectivity=1)
                    shadow_img_dir[index] = labeled_line_dir
                if len(np.unique(shadow_img_dir)) > 2:
                    filtered_separated.append(shadow_img_dir == 1)
                else:
                    filtered_separated.append(s)
            shadows = shadows + filtered_separated
            for s in filtered_separated:
                img = np.logical_and(img, np.logical_not(s))
        shadow_img = it.sum_imgs_colored(shadows)
        areas = []
        for d in shadows:
            labeled_divs, _, _ = it.divide_by_connected(d)
            areas = areas + labeled_divs
        final_areas = []
        for i, area in enumerate(areas):
            final_areas.append(ShadowArea(i, area))
            shadow_img = it.sum_imgs_colored([x.img for x in final_areas])
        return shadow_img, final_areas

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
            if np.sum(mt.opening(self.img, kernel_size=path_radius)) > 0:
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

    def find_monotonic(
        self, rest_of_picture_f3, base_frame, path_radius, ideal_sum
    ):
        sub_regions:Subregion = []
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
        self.zigzags_mst, zigzags_mst_sequence = path_tools.regions_mst(self.zigzags_graph)
        return

    def make_routes_z(self, base_frame, path_radius):
        def make_zigzag_route(region:ZigZag):
            region.center = pt.points_center(pt.contour_to_list(mt.detect_contours(region.img)))
            zig_options = []
            lines, n_lines, internal_border_img, contours, new_path_radius = (
                cut_in_lines(region.img, path_radius, var_path_width=0)
            )
            filled = it.fill_internal_area(
                internal_border_img.astype(np.uint8), np.ones_like(internal_border_img)
            )
            opened = mt.opening(filled, kernel_size=path_radius)
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
            return region
            
        processed_regions = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [
                executor.submit(make_zigzag_route, region) for region in self.regions
            ]
            for l in concurrent.futures.as_completed(results):
                processed_regions.append(l.result())
        processed_regions.sort(key=lambda x: x.name)
        self.regions = processed_regions
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


def cut_in_lines(img, path_radius, var_path_width=0):
    img2 = mt.opening(img, kernel_size=(path_radius * 2))
    considered = np.where(img2 != 0)
    if np.sum(considered[0]) == 0:
        print("pulei um!")
        return [], 0, [], [], []
    top = np.min(considered[0])
    bottom = np.max(considered[0])
    new_path_radius = path_radius
    region_mask_full = disk(new_path_radius * 2)
    if var_path_width:
        considered_height = bottom - top
        n_linhas = considered_height / (path_radius * 2)
        resto, divs = math.modf(n_linhas / 2)
        new_path_radius = (considered_height / divs) / 4
        region_mask_full = disk(new_path_radius * 2)
    internal_border = mt.erosion(img2, kernel_img=region_mask_full)
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
