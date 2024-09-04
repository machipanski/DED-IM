from components import images_tools as it
from typing import List
from components import points_tools as pt
from components import morphology_tools as mp
import numpy as np
import copy
from skimage.measure import label

# from components.zigzag import zigzag_tools
from components import path_tools
from cv2 import getStructuringElement, MORPH_RECT
from scipy.spatial import distance_matrix, distance
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import flood_fill


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
        counter_areas = 0
        for a in areas:
            square_mask = getStructuringElement(
                MORPH_RECT, (int(path_radius), int(path_radius))
            )
            faixa_apertada_a = np.logical_and(
                a, np.logical_not(mp.opening(a, kernel_img=square_mask))
            )
            parts, _, num = it.divide_by_connected(faixa_apertada_a)
            largos = list(map(lambda x: largura_ok(x, path_radius), parts))
            for j, b in enumerate(parts):
                if largos[j] == True:
                    alturas_da_faixa_apertada = pt.img_to_points(b)
                    alturas_da_faixa_apertada = [
                        x[0] for x in alturas_da_faixa_apertada
                    ]
                    alturas_da_faixa_apertada = [
                        min(alturas_da_faixa_apertada),
                        max(alturas_da_faixa_apertada),
                    ]
                    faixa_apertada_b = copy.deepcopy(a)
                    faixa_apertada_b[: (alturas_da_faixa_apertada[0] - 1)] = 0
                    faixa_apertada_b[(alturas_da_faixa_apertada[1] + 1) :] = 0
                    composto = np.add(
                        faixa_apertada_b.astype(np.uint8), a.astype(np.uint8)
                    )
                    faixa_a_ser_mantida = composto == 1
                    if np.sum(faixa_a_ser_mantida) > 4:
                        largura_a_ser_mantida = pt.img_to_points(faixa_a_ser_mantida)
                        largura_a_ser_mantida = [x[1] for x in largura_a_ser_mantida]
                        largura_a_ser_mantida = [
                            min(largura_a_ser_mantida),
                            max(largura_a_ser_mantida),
                        ]
                        faixa_a_ser_mantida_b = copy.deepcopy(a)
                        faixa_a_ser_mantida_b[:, : (largura_a_ser_mantida[0] - 1)] = 0
                        faixa_a_ser_mantida_b[:, (largura_a_ser_mantida[1] + 1) :] = 0
                        nova_div = np.logical_and(
                            a, np.logical_not(faixa_a_ser_mantida_b)
                        )
                        a = np.logical_and(a, np.logical_not(nova_div))
                        final_areas.append(ShadowArea(counter_areas, nova_div))
                        counter_areas += 1
            final_areas.append(ShadowArea(counter_areas, a))
            counter_areas += 1
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
            erodida = mp.erosion(a.img, kernel_size=path_radius)
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

    def scan_monotonic(self, rest_of_picture_f2, path_radius, void_max, base_frame):
        shadow_img, areas = self.create_shadow_img(path_radius)
        areas = it.neighborhood_imgs(areas)
        if len(areas) == 1:
            if np.sum(mp.opening(self.img, kernel_size=path_radius)) > 0:
                self.labeled_monotonic_regions = self.img
                self.areas_somadas = self.img
                self.regions.append(ZigZag(0, self.labeled_monotonic_regions))
        else:
            monotonic_regions, self.labeled_monotonic_regions, self.areas_somadas = (
                self.unite_small_monotonic_areas(
                    areas, path_radius, void_max, base_frame
                )
            )
            for i, mr in enumerate(monotonic_regions):
                if np.sum(mp.opening(mr.img, kernel_size=path_radius)) > 0:
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
        self, areas: List[ShadowArea], path_radius, void_max, base_frame
    ):
        monotonic_regions = copy.deepcopy(areas)
        for i in np.arange(0, len(monotonic_regions)):
            radius_pctg, _ = max_fit_inside(
                monotonic_regions[i].img, path_radius, void_max
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

    def find_monotonic(self, rest_of_picture_f3, base_frame, path_radius, void_max):
        sub_regions = []
        separated_imgs, labeled, num = it.divide_by_connected(rest_of_picture_f3)
        for i in np.arange(0, num):
            sub_regions.append(Subregion(i, separated_imgs[i]))
            sub_regions[-1].scan_monotonic(
                rest_of_picture_f3, path_radius, void_max, base_frame
            )
        regs_counter = 0
        for sub_region in sub_regions:
            for region in sub_region.regions:
                region.name = regs_counter
                self.regions.append(region)
                regs_counter += 1
        return

    def make_graph(self, zigzags_bridges, base_frame):
        zigzags_graph, pos_zigzag_nodes = path_tools.make_zigzag_graph(
            self.regions, zigzags_bridges, base_frame
        )
        zigzags_mst_graph, zigzags_mst_sequence = path_tools.regions_mst(zigzags_graph)
        return zigzags_graph, zigzags_mst_graph, pos_zigzag_nodes


def max_fit_inside(area, path_radius, void_max):
    from skimage.morphology import disk

    max_radius = np.max(distance_transform_edt(area))
    sum_area = np.sum(area)
    ideal_sum = np.sum(disk(path_radius))
    return max_radius / path_radius, sum_area / ideal_sum
