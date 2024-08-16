from __future__ import annotations
import itertools
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from files import Paths
    from typing import List
    from components.layer import Layer
from components import morphology_tools as mt
from components import images_tools as it
import numpy as np
from cv2 import drawContours
from skimage.morphology import disk
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


class Area:
    """Cada contorno fechado em um level gerado pelos Offsets"""

    def __init__(self, nome, img, origem, loops_inside):
        self.name = nome
        self.img = img
        self.offset_level = origem
        self.loops_inside = loops_inside
        self.internal_area = []


class Loop:
    """Cada contorno fechado em um level gerado pelos Offsets"""

    def __init__(self, nome, img_name, origem, trail_name):
        self.name = nome
        self.offset_level = origem
        self.internal_area = []
        self.trail = trail_name
        self.route = img_name
        self.region = ""
        self.lost_area = []
        self.lost_area_sum = 0
        self.acceptable = 0
        self.loops_filhos = []


class Level:
    """Cada elemento sequencial gerado pela operação de Offset em uma imagem"""

    def __init__(self, img: str, nome, pai, area):
        self.name = nome
        self.parent = pai
        self.img = img
        self.outer_loops: List[Loop] = []
        self.hole_loops: List[Loop] = []
        self.area = area
        # self.filled_area = area
        self.outer_areas = []
        self.hole_areas = []

    def create_level(
        self,
        folders: Paths,
        mask_full: np.ndarray,
        mask_double: np.ndarray,
        nome_filho,
        layer_name,
        island_name,
        first=False,
    ):
        """Erode + Openning = New Offset Level"""
        img = folders.load_img(self.img)
        if first:
            new_lvl_img = mt.erosion(img, kernel_img=mask_full)
            new_lvl_img = mt.opening(new_lvl_img, kernel_img=mask_full)
            new_lvl_area = mt.erosion(new_lvl_img, kernel_img=mask_full)
        else:
            new_lvl_img = mt.erosion(img, kernel_img=mask_double)
            new_lvl_img = mt.opening(new_lvl_img, kernel_img=mask_double)
            new_lvl_area = mt.erosion(new_lvl_img, kernel_img=mask_full)

        name_new_lvl = f"L{layer_name:03d}_I{island_name:03d}_LVL{nome_filho:03d}.png"
        folders.save_img(name_new_lvl, new_lvl_img)
        new_lvl_img = name_new_lvl

        name_new_lvl_area = (
            f"L{layer_name:03d}_I{island_name:03d}_LVL{nome_filho:03d}_Area.png"
        )
        folders.save_img(name_new_lvl_area, new_lvl_area)
        new_lvl_area = name_new_lvl_area

        return Level(new_lvl_img, nome_filho, self.name, new_lvl_area)

    def create_loops(
        self,
        name,
        img,
        mask,
        base_frame,
        orig_img,
        folders: Paths,
        layer_name,
        island_name,
        level_name,
    ):
        """Usa cv2.findContours() para separar cada loop dentro imagem do Offset Level"""
        contours, hierarchy = mt.detect_contours(img, return_hierarchy=True)
        in_counter = 0
        out_counter = 0
        for i in np.arange(0, len(contours)):
            loop = drawContours(np.zeros(base_frame), contours, i, 1)
            if hierarchy[0][i][3] == -1:
                name_loop = f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopOut{out_counter:03d}.npz"
                folders.save_npz(name_loop, loop)
                trail_name = f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopOut{out_counter:03d}_trail.npz"
                trail_img = mt.dilation(loop, kernel_img=mask)
                folders.save_npz(trail_name, trail_img)
                self.outer_loops.append(Loop(out_counter, name_loop, name, trail_name))
                self.outer_loops[-1].internal_area = it.fill_internal_area(
                    folders.load_npz(self.outer_loops[-1].trail), orig_img
                )
                name_loop_area = f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopOut{out_counter:03d}_Area.png"
                folders.save_img(name_loop_area, self.outer_loops[-1].internal_area)
                self.outer_loops[-1].internal_area = name_loop_area
                out_counter += 1
            else:
                name_loop = f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopInt{in_counter:03d}.npz"
                folders.save_npz(name_loop, loop)
                trail_name = f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopInt{in_counter:03d}_trail.npz"
                trail_img = mt.dilation(loop, kernel_img=mask)
                folders.save_npz(trail_name, trail_img)
                self.hole_loops.append(Loop(in_counter, name_loop, name, trail_name))
                self.hole_loops[-1].internal_area = it.fill_internal_area(
                    folders.load_npz(self.hole_loops[-1].trail), orig_img
                )
                name_loop_area = f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopInt{in_counter:03d}_Area.png"
                folders.save_img(name_loop_area, self.hole_loops[-1].internal_area)
                self.hole_loops[-1].internal_area = name_loop_area
                in_counter += 1
        return self

    def calculate_lost_area(self, result):
        """Faz o calculo da diferença entre o que foi coberto e a área ideal de cada Loop"""
        resultado_negado = np.logical_not(result)
        for area in self.outer_areas + self.hole_areas:
            for coords in area.loops_inside:
                diference = np.logical_and(area.img, resultado_negado)
                if coords[1] == 0:
                    self.outer_loops[coords[2]].lost_area = diference
                elif coords[1] == 1:
                    self.hole_loops[coords[2]].lost_area = diference
        for loop in self.outer_loops + self.hole_loops:
            loop.lost_area_sum = np.sum(loop.lost_area)
        return

    def divide_areas(self, base_frame, original_img, path_radius, folders:Paths):
        areas, A, B = it.divide_by_connected(self.area_lost)
        flags_areas_loops = [0 for x in areas]
        composed_areas = []
        for i, area in enumerate(areas):
            loops_inside = []
            for loop in self.outer_loops:
                if np.logical_and(folders.load_npz(loop.route), area).any():
                    loops_inside.append((self.name, 0, loop.name))
                    flags_areas_loops[i] = 1
            for loop in self.hole_loops:
                if np.logical_and(folders.load_npz(loop.route), area).any():
                    loops_inside.append((self.name, 1, loop.name))
                    flags_areas_loops[i] = 1
            final_area = Area(i, area, self.name, loops_inside)
            composed_areas.append(final_area)
        separated_loops = []
        for c_area in composed_areas:
            if len(c_area.loops_inside) >= 2:
                separated_loops = separated_loops + self.divide_composed_areas(
                    c_area, path_radius, folders
                )
            else:
                separated_loops.append(c_area)
        self.outer_areas = []
        self.hole_areas = []
        for s_area in separated_loops:
            AAAA = s_area.img
            if len(s_area.loops_inside) > 0:
                if s_area.loops_inside[0][1] == 0:
                    self.outer_areas.append(s_area)
                elif s_area.loops_inside[0][1] == 1:
                    self.hole_areas.append(s_area)
        not_used_areas = np.zeros(base_frame)
        for i, area in enumerate(areas):
            if not flags_areas_loops[i]:
                not_used_areas = np.logical_or(not_used_areas, area)
        return not_used_areas

    def divide_composed_areas(self, area, path_radius, folders:Paths):
        """Quando uma area tem maus de um loop, qual é a área ideal de CADA LOOP?"""
        A = np.zeros_like(area.img)
        for loop in area.loops_inside:
            if loop[1] == 0:
                A = np.add(A, folders.load_npz(self.outer_loops[loop[2]].route))
            else:
                A = np.add(A, folders.load_npz(self.hole_loops[loop[2]].route))
        distance_map = distance_transform_edt(area.img)
        A = mt.dilation(A, kernel_size=path_radius)
        _, markers, _ = it.divide_by_connected(A)
        labels = watershed(-distance_map, markers, mask=area.img)
        divisoes_incluidas = np.unique(labels)
        new_areas = []
        for divisao in divisoes_incluidas:
            if divisao != 0:
                new_areas.append(labels == divisao)
        children_areas = []
        for divided_area in new_areas:
            for loop in area.loops_inside:
                if loop[1] == 0:
                    loop_route = folders.load_npz(self.outer_loops[loop[2]].route)
                else:
                    loop_route = folders.load_npz(self.hole_loops[loop[2]].route)
                if np.logical_and(loop_route, divided_area).any():
                    children_areas.append(Area("test", divided_area, self.name, [loop]))
        return children_areas

class OffsetRegions:

    def __init__(self):
        self.all_valid_loops = []
        self.regions = []
        self.n_regions = []
        self.next_prohibited_areas = []
        self.levels: List[Level] = []
        self.n_levels = []
        self.influence_regions = []

    def calculate_voids(
        self,
        base_frame,
        start_of_process_img_name,
        levels: List[Level],
        n_levels,
        path_radius,
        folders: Paths,
    ):
        start_of_process_img = folders.load_img(start_of_process_img_name)
        offset_result = self.calculate_covered_area(base_frame, levels, folders)
        for l in np.arange(0, n_levels):
            if l == 0:
                levels[l].area_lost = np.logical_and(
                    start_of_process_img,
                    np.logical_not(folders.load_img(levels[l].area)),
                )
            else:
                levels[l].area_lost = np.logical_and(
                    folders.load_img(levels[l - 1].area),
                    np.logical_not(folders.load_img(levels[l].area)),
                )
            not_used_area = levels[l].divide_areas(
                base_frame, start_of_process_img, path_radius, folders
            )
            if np.sum(not_used_area) > 0:
                levels[l - 1].area_lost = np.logical_or(
                    levels[l - 1].area_lost, not_used_area
                )
                _ = levels[l - 1].divide_areas(
                    base_frame, start_of_process_img, path_radius, folders
                )
                levels[l - 1].calculate_lost_area(offset_result)
            levels[l].calculate_lost_area(offset_result)
        # AAAAA = levels[0].area_lost
        return

    def calculate_covered_area(self, base_frame, levels, folders: Paths):
        offset_simulated = np.zeros(base_frame)
        for level in levels:
            for loop in level.outer_loops + level.hole_loops:
                trail = folders.load_npz(loop.trail)
                offset_simulated = np.logical_or(offset_simulated, trail)
        return offset_simulated

    def create_levels(
        self,
        folders,
        rest_of_picture_f1,
        mask_full,
        mask_double,
        layer_name,
        island_name,
    ):
        """Vai de fora para dentro realizando erosões e organiza em níveis
        até não haver mais pixels a se processar"""
        levels = []
        n_levels = 0
        atual = Level(rest_of_picture_f1, 0, "", rest_of_picture_f1)
        atual = atual.create_level(
            folders,
            mask_full,
            mask_double,
            n_levels,
            layer_name,
            island_name,
            first=True,
        )
        levels.append(atual)
        n_levels += 1
        atual = atual.create_level(
            folders, mask_full, mask_double, n_levels, layer_name, island_name
        )
        while (
            np.sum(folders.load_img(atual.img)) > 0
        ):  # Enquanto a imagem não estiver vazia1
            levels.append(atual)
            n_levels += 1
            atual = atual.create_level(
                folders, mask_full, mask_double, n_levels, layer_name, island_name
            )
        print(f"Ilha: {island_name} Número de Níveis: {n_levels}")
        self.levels = levels
        self.n_levels = n_levels
        return

    def create_influence_regions(self, base_frame, folders):
        """Faz uma varredura de dentro para fora nos Levels da Layer, se o contorno externo está dentro do outro
        no level anterior são do mesmo grupo, se há dois ou nenhum, então é um novo grupo
        Depois disso, faz o mesmo de fora para dentro com os contornos internos dos buracos
        """
        influence_regions = []
        seq_in_out = np.arange(self.n_levels - 1, -1, -1)
        seq_out_in = np.arange(0, self.n_levels)
        region_counter = self.label_offset_regions(
            self.levels, seq_in_out, 0, 1, folders
        )
        region_counter_holes = self.label_offset_regions(
            self.levels, seq_out_in, region_counter, 0, folders
        )
        """Processo de eliminação de cada img. a região final é a img que contém completamente as outras"""
        influence_regions = self.sum_areas(
            self.levels,
            influence_regions,
            np.arange(1, region_counter + 1),
            0,
            base_frame,
            folders,
        )  # agrupa todas as áreas de loops externos
        influence_regions = self.sum_areas(
            self.levels,
            influence_regions,
            np.arange(region_counter + 1, region_counter_holes + 1),
            1,
            base_frame,
            folders,
        )  # agrupa todas as áreas de loops internos
        """No final retira as regiões mais internas das mais externas"""
        pare = 0
        while pare == 0:
            n_alteracoes = 0
            to_remove = []
            for i, j in itertools.permutations(influence_regions, 2):
                if (
                    i != j
                    and i.img.any()
                    and j.img.any()
                    and it.esta_contido(j.img, i.img)
                ):
                    i.img = np.logical_and(i.img, np.logical_not(j.img))
                    to_remove.append(j)
                    n_alteracoes += 1
            if n_alteracoes == 0:
                pare = 1
        # AAAA = influence_regions[0].img
        self.influence_regions = influence_regions
        return

    def label_offset_regions(
        self, levels: List[Level], sequence, counter_start, order, folders: Paths
    ):
        """Separa uma região para cada loop ao percoreer as camadas"""
        region_counter = counter_start
        loop_group = []
        loop_group_2 = []
        for level_number in sequence:
            if order == 1:
                loop_group = levels[level_number].outer_loops
            elif order == 0:
                loop_group = levels[level_number].hole_loops
            for loop_cam_atual in loop_group:
                contidos = 0
                tag_grupo = 0
                if level_number == sequence[0]:
                    region_counter += 1  # para cada região conta uma nova região
                    loop_cam_atual.region = (
                        region_counter  # e dá o nome dela. essa será uma seed
                    )
                else:
                    if order == 1:
                        loop_group_2 = levels[level_number + 1].outer_loops
                    elif order == 0:
                        loop_group_2 = levels[level_number - 1].hole_loops
                    for loop_cam_anterior in loop_group_2:
                        if it.esta_contido(
                            folders.load_img(loop_cam_anterior.internal_area),
                            folders.load_img(loop_cam_atual.internal_area),
                        ):
                            contidos += 1
                            tag_grupo = loop_cam_anterior.region
                            if order == 1:
                                loop_cam_atual.loops_filhos.append(loop_cam_anterior)
                            if order == 0:
                                loop_cam_anterior.loops_filhos.append(loop_cam_atual)
                    if contidos == 1:  # se só existe uma
                        loop_cam_atual.region = (
                            tag_grupo  # o grupo dessa é o mesmo da anterior
                        )
                    else:  # se for 0, 2 ou mais
                        region_counter += 1  # é uma nova região
                        loop_cam_atual.region = (
                            region_counter  # e assume o numero do contador
                        )
        return region_counter

    def make_regions(
        self,
        original_img,
        base_frame,
        path_radius_external,
        void_max,
        max_external_walls,
        max_internal_walls,
        folders:Paths
    ):
        acceptable = self.tag_loops_by_voids(
            base_frame,
            path_radius_external,
            void_max,
            max_external_walls,
            max_internal_walls,
            self.influence_regions,
            self.levels,
            folders
        )
        counter = 0
        level_accepted = 0
        all_loops_img = np.zeros_like(acceptable)
        influence_regions = sorted(
            self.influence_regions, key=lambda x: min(sublist[0] for sublist in x.loops)
        )
        all_loops_descrition = sum([x.loops for x in influence_regions], [])
        loops_accepted_desc = list(filter(lambda x: x[0] == 0, all_loops_descrition))
        loops_accepted = []
        ideal_sum = np.sum(disk(path_radius_external))
        for loop in loops_accepted_desc:
            if loop[1] == 0:
                loops_accepted.append(self.levels[loop[0]].outer_loops[loop[2]])
            elif loop[1] == 1:
                loops_accepted.append(self.levels[loop[0]].hole_loops[loop[2]])
        for region in influence_regions:
            fil_region_img = np.zeros(base_frame)
            fil_region_loops = []
            reference_region_img = np.zeros(base_frame)
            for loop in region.loops:
                candidate = {}
                if loop[1] == 0:
                    candidate = self.levels[loop[0]].outer_loops[loop[2]]
                elif loop[1] == 1:
                    candidate = self.levels[loop[0]].hole_loops[loop[2]]
                reference_region_img = np.logical_or(
                    reference_region_img, folders.load_npz(candidate.trail)
                )
                candidate.lost_area_sum = np.sum(candidate.lost_area)
                candidate_voides, A, B = it.divide_by_connected(candidate.lost_area)
                sums = [np.sum(x) for x in candidate_voides]
                if candidate.acceptable:
                    if len(sums) > 0:
                        maior = sums[np.argmax(sums)] / ideal_sum
                    else:
                        maior = 0
                    print(
                        str(loop)
                        + " Perdendo total:"
                        + str(candidate.lost_area_sum)
                        + " maior void:"
                        + str(maior)
                        + "Bw -> aceito"
                    )
                    if candidate in loops_accepted:
                        loops_accepted = loops_accepted + candidate.loops_filhos
                        fil_region_img = np.logical_or(fil_region_img, folders.load_npz(candidate.trail))
                        fil_region_loops.append(candidate)
                else:
                    print(
                        str(loop)
                        + " Perdendo total:"
                        + str(candidate.lost_area_sum)
                        + " maior void:"
                        + str(sums[np.argmax(sums)] / ideal_sum)
                        + "Bw -> bloqueado"
                    )
            if fil_region_img.any():  # and counter < 4:
                self.regions.append(Region(counter, fil_region_img, fil_region_loops))
                all_loops_img = np.logical_or(all_loops_img, fil_region_img)
                counter += 1
        self.n_regions = counter
        rest = np.logical_and(folders.load_img(original_img), np.logical_not(all_loops_img))
        _, labels, n_labels = it.divide_by_connected(rest)
        size_label_bfr = 0
        index_main_body = 0
        for i in np.arange(1, n_labels):
            size_label_now = np.sum(labels == i)
            if size_label_now > size_label_bfr:
                index_main_body = i
                size_label_bfr = size_label_now
        return labels == index_main_body

    def make_valid_loops(self, base_frame, folders:Paths):
        """Cria uma matriz com todas as rotas em cada Level somadas para a Layer"""
        all_valid_loops = np.zeros(base_frame)
        for region in self.regions:
            for loop in region.loops:
                all_valid_loops = np.logical_or(all_valid_loops, folders.load_npz(loop.route))
        return all_valid_loops

    def sum_areas(
        self,
        levels,
        influence_regions,
        interval,
        outer_inner,
        base_frame,
        folders: Paths,
    ):
        """Para cada numero de região, percorre todas as layers em busca de loops
        com esse numero. Quando encontra, faz um OR para somar as áreas internas ao total da região
        """
        loop_group = []
        for i in interval:
            loops_inside = []
            essa_reg = np.zeros(base_frame, dtype=bool)
            for level in levels:
                if outer_inner == 0:
                    loop_group = level.outer_loops
                elif outer_inner == 1:
                    loop_group = level.hole_loops
                for loop in loop_group:
                    if loop.region == i:
                        essa_reg = np.logical_or(
                            essa_reg, folders.load_img(loop.internal_area)
                        )
                        loops_inside.append([level.name, outer_inner, loop.name])
            influence_regions.append(Region(i, essa_reg, loops_inside))
        return influence_regions

    def tag_loops_by_voids(
        self,
        base_frame,
        path_radius_external,
        void_max,
        max_external_walls,
        max_internal_walls,
        influence_regions,
        levels,
        folders:Paths
    ):
        offset_simulated = np.zeros(base_frame)
        ideal_sum = np.sum(disk(path_radius_external))
        # exclude_region = []
        for region in influence_regions:
            internal_counter = 0
            external_counter = 0
            for loop in region.loops:
                if loop[1] == 1:
                    thisloop = levels[loop[0]].hole_loops[loop[2]]
                elif loop[1] == 0:
                    thisloop = levels[loop[0]].outer_loops[loop[2]]
                voids, _, _ = it.divide_by_connected(thisloop.lost_area)
                voids_sums = np.divide([np.sum(x) for x in voids], ideal_sum)
                # AAAA = np.add(thisloop.lost_area, thisloop.trail * 2)
                if any(voids_sums > void_max):
                    thisloop.acceptable = 0
                else:
                    thisloop.acceptable = 1
                    offset_simulated = np.logical_or(offset_simulated, folders.load_npz(thisloop.trail))
                    if loop[1] == 1:
                        internal_counter += 1
                    if loop[1] == 0:
                        external_counter += 1
        return offset_simulated


class Region:
    """Caminho fechado individual por Offset paralelo"""

    def __init__(self, name, img, loops):
        self.name = name
        self.img = img
        self.loops = loops
        self.limmit_coords = (
            []
        )  # coordenadas dos pontos onde se separam as regiões monotônicas
        self.center_coords = []  # coordenadas do centro geométrico de cada contorno
        self.area_contour = []  # contorno de cada área
        self.area_contour_img = []
        self.internal_area = []  # resultante de se pintar o interior de cada contorno
        self.hierarchy = (
            0  # hierarquia de contornos, quais são internos e quais são externos
        )
        self.paralel_points = []
        self.route = []
        self.trail = []
        self.next_prohibited_area = []
