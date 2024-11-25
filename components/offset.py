from __future__ import annotations
import itertools
from pathlib import Path
from components import morphology_tools as mt, path_tools
from components import images_tools as it
from components import points_tools as pt
import numpy as np
from cv2 import drawContours
from skimage.morphology import disk
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import concurrent.futures
from components import skeleton as sk
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from files import System_Paths
    from typing import List
    from components.layer import Layer



class Area:
    """Cada contorno fechado em um level gerado pelos Offsets"""

    def __init__(self, nome, img, origem, loops_inside):
        self.name = nome
        self.img = img
        self.offset_level = origem
        self.loops_inside = loops_inside
        self.internal_area = []


class ColecaoDeCoords:
    def __init__(self, origin, destiny):
        self.origin = origin
        self.destiny = destiny
        self.lista_a = []
        self.dist_a = []
        self.lista_b = []
        self.dist_b = []
        self.lista_c = []
        self.dist_c = []
        self.lista_d = []
        self.dist_d = []


class Loop:
    """Cada contorno fechado em um level gerado pelos Offsets"""

    def __init__(self, nome, img, origem, trail, **kwargs):
        self.name = nome
        self.offset_level = origem
        self.internal_area = []
        self.trail = trail
        self.route = img
        self.region = ""
        self.lost_area = []
        self.lost_area_sum = 0
        self.acceptable = 0
        self.loops_filhos = []
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)


class Level:
    """Cada elemento sequencial gerado pela operação de Offset em uma imagem"""

    def __init__(self, img: str, nome, pai, area, path):
        self.name = nome
        self.parent = pai
        self.img = img
        self.outer_loops: List[Loop] = []
        self.hole_loops: List[Loop] = []
        self.area = area
        self.path = path
        self.outer_areas = []
        self.hole_areas = []
        self.area_lost = []

    def create_level(
        self,
        mask_full: np.ndarray,
        mask_double: np.ndarray,
        nome_filho,
        path,
        first=False,
    ) -> Level:
        """Erode + Openning = New Offset Level"""
        # img = folders.load_img(self.img)
        if first:
            new_lvl_img = mt.erosion(self.img, kernel_img=mask_full)
            new_lvl_img = mt.opening(new_lvl_img, kernel_img=mask_full)
            new_lvl_area = mt.erosion(new_lvl_img, kernel_img=mask_full)
        else:
            new_lvl_img = mt.erosion(self.img, kernel_img=mask_double)
            new_lvl_img = mt.opening(new_lvl_img, kernel_img=mask_double)
            new_lvl_area = mt.erosion(new_lvl_img, kernel_img=mask_full)
        return Level(new_lvl_img, nome_filho, self.name, new_lvl_area, path)

    def create_loops(
        self,
        mask,
        base_frame,
        rest_f1,
        layer_name,
        island_name,
        level_name,
    ):
        """Usa cv2.findContours() para separar cada loop dentro imagem do Offset Level"""
        contours, hierarchy = mt.detect_contours(self.img, return_hierarchy=True)
        in_counter = 0
        out_counter = 0
        for i in np.arange(0, len(contours)):
            loop = drawContours(np.zeros(base_frame), contours, i, 1)
            if hierarchy[0][i][3] == -1:
                trail_img = mt.dilation(loop, kernel_img=mask)
                self.outer_loops.append(Loop(out_counter, loop, self.name, trail_img))
                self.outer_loops[-1].internal_area = it.fill_internal_area(
                    self.outer_loops[-1].trail, rest_f1
                )
                out_counter += 1
            else:
                trail_img = mt.dilation(loop, kernel_img=mask)
                self.hole_loops.append(Loop(in_counter, loop, self.name, trail_img))
                self.hole_loops[-1].internal_area = it.fill_internal_area(
                    self.hole_loops[-1].trail, rest_f1
                )
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

    def divide_areas(self, base_frame, path_radius):
        areas, A, B = it.divide_by_connected(self.area_lost)
        flags_areas_loops = [0 for x in areas]
        composed_areas = []
        for i, area in enumerate(areas):
            loops_inside = []
            for loop in self.outer_loops:
                if np.logical_and(loop.route, area).any():
                    loops_inside.append((self.name, 0, loop.name))
                    flags_areas_loops[i] = 1
            for loop in self.hole_loops:
                if np.logical_and(loop.route, area).any():
                    loops_inside.append((self.name, 1, loop.name))
                    flags_areas_loops[i] = 1
            final_area = Area(i, area, self.name, loops_inside)
            composed_areas.append(final_area)
        separated_loops = []
        for c_area in composed_areas:
            if len(c_area.loops_inside) >= 2:
                separated_loops = separated_loops + self.divide_composed_areas(
                    c_area, path_radius
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

    def divide_composed_areas(self, area, path_radius):
        """Quando uma area tem maus de um loop, qual é a área ideal de CADA LOOP?"""
        A = np.zeros_like(area.img)
        for loop in area.loops_inside:
            if loop[1] == 0:
                A = np.add(A, self.outer_loops[loop[2]].route)
            else:
                A = np.add(A, self.hole_loops[loop[2]].route)
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
                    loop_route = self.outer_loops[loop[2]].route
                else:
                    loop_route = self.hole_loops[loop[2]].route
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
        start_of_process_img,
        levels: List[Level],
        n_levels,
        path_radius,
    ):
        if np.sum(start_of_process_img) > 0:
            offset_result = self.calculate_covered_area(base_frame, levels)
            for l in np.arange(0, n_levels):
                if l == 0:
                    levels[l].area_lost = np.logical_and(
                        start_of_process_img,
                        np.logical_not(levels[l].area),
                    )
                else:
                    levels[l].area_lost = np.logical_and(
                        levels[l - 1].area,
                        np.logical_not(levels[l].area),
                    )
                not_used_area = levels[l].divide_areas(base_frame, path_radius)
                if (np.sum(not_used_area) > 0) and (l > 0):
                    levels[l - 1].area_lost = np.logical_or(
                        levels[l - 1].area_lost, not_used_area
                    )
                    _ = levels[l - 1].divide_areas(base_frame, path_radius)
                    levels[l - 1].calculate_lost_area(offset_result)
                levels[l].calculate_lost_area(offset_result)
            # AAAAA = levels[0].area_lost
        return

    def calculate_covered_area(self, base_frame, levels):
        offset_simulated = np.zeros(base_frame)
        for level in levels:
            for loop in level.outer_loops + level.hole_loops:
                trail = loop.trail
                offset_simulated = np.logical_or(offset_simulated, trail)
        return offset_simulated

    def create_levels(
        self,
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
        atual = Level(
            rest_of_picture_f1,
            f"Lvl_{n_levels:03d}",
            "",
            rest_of_picture_f1,
            f"/{layer_name}/{island_name}/offsets/levels",
        )
        atual = atual.create_level(
            mask_full,
            mask_double,
            f"Lvl_{n_levels:03d}",
            f"/{layer_name}/{island_name}/offsets/levels",
            first=True,
        )
        levels.append(atual)
        n_levels += 1
        atual = atual.create_level(
            mask_full,
            mask_double,
            f"Lvl_{n_levels:03d}",
            f"/{layer_name}/{island_name}/offsets/levels",
        )
        while np.sum(atual.img) > 0:
            levels.append(atual)
            n_levels += 1
            atual = atual.create_level(
                mask_full,
                mask_double,
                f"Lvl_{n_levels:03d}",
                f"/{layer_name}/{island_name}/offsets/levels",
            )
        print(f"Ilha: {island_name} Número de Níveis: {n_levels}")
        self.levels = levels
        self.n_levels = n_levels
        return

    def create_influence_regions(self, base_frame):
        """Faz uma varredura de dentro para fora nos Levels da Layer, se o contorno externo está dentro do outro
        no level anterior são do mesmo grupo, se há dois ou nenhum, então é um novo grupo
        Depois disso, faz o mesmo de fora para dentro com os contornos internos dos buracos
        """
        influence_regions = []
        seq_in_out = np.arange(self.n_levels - 1, -1, -1)
        seq_out_in = np.arange(0, self.n_levels)
        region_counter = self.label_offset_regions(seq_in_out, 0, 1)
        region_counter_holes = self.label_offset_regions(seq_out_in, region_counter, 0)
        """Processo de eliminação de cada img. a região final é a img que contém completamente as outras"""
        influence_regions = self.sum_areas(
            self.levels,
            influence_regions,
            np.arange(1, region_counter + 1),
            0,
            base_frame,
        )  # agrupa todas as áreas de loops externos
        influence_regions = self.sum_areas(
            self.levels,
            influence_regions,
            np.arange(region_counter + 1, region_counter_holes + 1),
            1,
            base_frame,
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

    def label_offset_regions(self, sequence, counter_start, order):
        """Separa uma região para cada loop ao percoreer as camadas"""
        region_counter = counter_start
        loop_group = []
        loop_group_2 = []
        for level_number in sequence:
            if order == 1:
                loop_group = self.levels[level_number].outer_loops
            elif order == 0:
                loop_group = self.levels[level_number].hole_loops
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
                        loop_group_2 = self.levels[level_number + 1].outer_loops
                    elif order == 0:
                        loop_group_2 = self.levels[level_number - 1].hole_loops
                    for loop_cam_anterior in loop_group_2:
                        if it.esta_contido(
                            loop_cam_anterior.internal_area,
                            loop_cam_atual.internal_area,
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

    def make_routes_o(
        self,
        base_frame,
        mask_full,
        path_radius,
        amendment_size,
    ):
        prohibited_areas = np.zeros(base_frame)

        def make_offset_route(region):
            route = np.zeros(base_frame)
            next_prohibited_area = np.zeros(base_frame)
            for loop in region.loops:
                route = np.logical_or(route, loop.route)
            n_loops = len(region.loops)
            if n_loops == 2:
                amendment = disk(path_radius * amendment_size)
                route, next_prohibited_area = link_spirals(
                    route, n_loops, amendment, region, base_frame, prohibited_areas
                )
            if n_loops > 2:
                amendment = disk(2 * path_radius * amendment_size)
                route, next_prohibited_area = link_spirals(
                    route, n_loops, amendment, region, base_frame, prohibited_areas
                )
            reparos = mt.find_failures(route, np.zeros_like(route))
            route = np.logical_or(reparos, route)
            route = mt.closing(route, kernel_size=1)
            region.route, _, _ = sk.create_prune_divide_skel(route, path_radius)
            region.trail = mt.dilation(route, kernel_img=mask_full)
            region.next_prohibited_area = next_prohibited_area
            return region

        processed_regions = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [
                executor.submit(make_offset_route, region) for region in self.regions
            ]
            for l in concurrent.futures.as_completed(results):
                processed_regions.append(l.result())
        processed_regions.sort(key=lambda x: x.name)
        self.regions = processed_regions
        self.next_prohibited_areas = np.zeros(base_frame)
        for r in self.regions:
            self.next_prohibited_areas = np.logical_or(
                self.next_prohibited_areas, r.next_prohibited_area
            )
        return

    def make_regions(
        self,
        rest_f1,
        base_frame,
        path_radius_external,
        void_max,
        max_external_walls,
        max_internal_walls,
        bead_size,
    ):
        acceptable = self.tag_loops_by_voids(
            base_frame,
            path_radius_external,
            void_max,
            max_external_walls,
            max_internal_walls,
            self.influence_regions,
            self.levels,
        )
        counter = 0
        level_accepted = 0
        all_loops_img = np.zeros_like(acceptable)
        influence_regions = sorted(
            self.influence_regions, key=lambda x: min(sublist[0] for sublist in x.loops)
        )
        all_loops_descrition = sum([x.loops for x in influence_regions], [])
        loops_accepted_desc = list(
            filter(lambda x: x[0] == "Lvl_000", all_loops_descrition)
        )
        loops_accepted = []
        ideal_sum = np.sum(disk(path_radius_external))
        for loop in loops_accepted_desc:
            if loop[1] == 0:
                loop_level = int(loop[0].replace("Lvl_", ""))
                loops_accepted.append(self.levels[loop_level].outer_loops[loop[2]])
            elif loop[1] == 1:
                loops_accepted.append(self.levels[loop_level].hole_loops[loop[2]])
        for region in influence_regions:
            fil_region_img = np.zeros(base_frame)
            fil_region_loops = []
            reference_region_img = np.zeros(base_frame)
            for loop in region.loops:
                candidate = {}
                loop_level = int(loop[0].replace("Lvl_", ""))
                if loop[1] == 0:
                    candidate = self.levels[loop_level].outer_loops[loop[2]]
                elif loop[1] == 1:
                    candidate = self.levels[loop_level].hole_loops[loop[2]]
                reference_region_img = np.logical_or(
                    reference_region_img, candidate.trail
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
                        fil_region_img = np.logical_or(fil_region_img, candidate.trail)
                        fil_region_loops.append(candidate)
                else:
                    if len(sums) > 0:
                        print(
                            str(loop)
                            + " Perdendo total:"
                            + str(candidate.lost_area_sum)
                            + " maior void:"
                            + str(sums[np.argmax(sums)] / ideal_sum)
                            + "Bw -> bloqueado"
                        )
                    else:
                        print(
                            str(loop)
                            + " Perdendo total:"
                            + str(candidate.lost_area_sum)
                            + "Bw -> bloqueado por limite maximo"
                        )
            if fil_region_img.any():  # and counter < 4:
                self.regions.append(Region(counter, fil_region_img, fil_region_loops))
                all_loops_img = np.logical_or(all_loops_img, fil_region_img)
                counter += 1
        self.n_regions = counter
        rest = np.logical_and(rest_f1, np.logical_not(all_loops_img))
        _, labels, n_labels = it.divide_by_connected(rest)
        size_label_bfr = 0
        index_main_body = 0
        rest_f2 = np.zeros(base_frame)
        for i in np.arange(1, n_labels):
            label_img = labels == i
            size_label_now = np.sum(label_img)
            if size_label_now > bead_size:
                rest_f2 = it.sum_imgs([rest_f2, label_img])
        return rest_f2

    def make_valid_loops(self, base_frame):
        """Cria uma matriz com todas as rotas em cada Level somadas para a Layer"""
        all_valid_loops = np.zeros(base_frame)
        for region in self.regions:
            for loop in region.loops:
                all_valid_loops = np.logical_or(all_valid_loops, loop.route)
        return all_valid_loops

    def sum_areas(self, levels, influence_regions, interval, outer_inner, base_frame):
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
                        essa_reg = np.logical_or(essa_reg, loop.internal_area)
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
    ):
        offset_simulated = np.zeros(base_frame)
        ideal_sum = np.sum(disk(path_radius_external))
        # exclude_region = []
        for region in influence_regions:
            internal_counter = 0
            external_counter = 0
            aaaaaaa = []
            for loop in region.loops:
                loop_level = int(loop[0].replace("Lvl_", ""))
                if loop[1] == 1:
                    thisloop = levels[loop_level].hole_loops[loop[2]]
                    this_counter = internal_counter
                    this_max = max_internal_walls - 1
                elif loop[1] == 0:
                    thisloop = levels[loop_level].outer_loops[loop[2]]
                    this_counter = external_counter
                    this_max = max_external_walls - 1
                aaaaaaa.append(thisloop)
                voids, _, _ = it.divide_by_connected(thisloop.lost_area)
                voids_sums = np.divide([np.sum(x) for x in voids], ideal_sum)
                # AAAA = np.add(thisloop.lost_area, thisloop.trail * 2)
                if (
                    loop_level != 0
                    and (any(voids_sums > void_max))
                    or this_counter > this_max
                ):
                    thisloop.acceptable = 0
                else:
                    thisloop.acceptable = 1
                    offset_simulated = np.logical_or(offset_simulated, thisloop.trail)
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
        self.limmit_coords = []
        # coordenadas dos pontos onde se separam as regiões monotônicas
        self.center_coords = []  # coordenadas do centro geométrico de cada contorno
        self.area_contour = []  # contorno de cada área
        self.area_contour_img = []
        self.internal_area = []  # resultante de se pintar o interior de cada contorno
        self.hierarchy = 0
        # hierarquia de contornos, quais são internos e quais são externos
        self.paralel_points = []
        self.route = []
        self.trail = []
        self.next_prohibited_area = []

    def make_contour(self, base_frame):
        self.area_contour, self.area_contour_img = mt.detect_contours(
            self.img, return_img=True, only_external=True
        )
        return

    def make_internal_area_and_center(self, original_img):
        self.internal_area = it.fill_internal_area(self.area_contour_img, original_img)
        self.center_coords = pt.points_center(pt.contour_to_list(self.area_contour))

    def make_limmit_coords(self, path_radius):
        limmit_coords = pt.extreme_points(self.area_contour_img, force_top=True)
        limmit_coords[0][0] = limmit_coords[0][0] + path_radius * 2
        limmit_coords[1][0] = limmit_coords[1][0] + path_radius * -2
        limmit_coords[2][0] = limmit_coords[2][0] + path_radius * -2
        limmit_coords[3][0] = limmit_coords[3][0] + path_radius * 2
        self.limmit_coords = limmit_coords
        return

    def out_area_inner_contour(self, base_frame):
        this_contours, this_hierarchy = mt.detect_contours(
            self.img, return_hierarchy=True
        )
        maior_soma = 0
        index_maior_contorno_interno = 0
        for i in np.arange(0, len(this_contours)):
            soma = len(this_contours[i])
            if soma > maior_soma and this_hierarchy[0][i][2] == -1:
                maior_soma = soma
                index_maior_contorno_interno = i
        area_internal_contour = this_contours[index_maior_contorno_interno]
        area_internal_contour_img = drawContours(
            np.zeros(base_frame), this_contours, index_maior_contorno_interno, 1
        )
        return area_internal_contour, area_internal_contour_img

    def make_paralel_points(
        self, regions, area_internal_contour_img, prohibited_areas, path_radius
    ):
        ys_do_buraco = [point[0] for point in self.limmit_coords]
        xs_do_buraco = [point[1] for point in self.limmit_coords]
        for region in regions:
            if region.name != self.name:
                self.paralel_points.append(ColecaoDeCoords(self.name, region.name))
                if region.hierarchy == 0:
                    destiny_points = np.nonzero(area_internal_contour_img)
                else:
                    destiny_points = np.nonzero(region.area_contour_img)
                destiny_points = pt.x_y_para_pontos(destiny_points)
                destiny_points = list(
                    filter(
                        lambda x: (min(ys_do_buraco) <= x[0] <= max(ys_do_buraco)),
                        destiny_points,
                    )
                )
                for i in np.arange(0, len(destiny_points)):
                    multiplier = 1
                    if (
                        destiny_points[i][0] == ys_do_buraco[0]
                        and destiny_points[i][1] <= xs_do_buraco[0]
                    ):
                        self.paralel_points[-1].lista_a.append(destiny_points[i])
                        linha = it.draw_line(
                            np.zeros_like(prohibited_areas),
                            self.limmit_coords[0],
                            destiny_points[i],
                        )
                        linha = mt.dilation(linha, kernel_size=path_radius * 2 + 2)
                        if np.sum(np.logical_and(linha, prohibited_areas)) > 0:
                            multiplier = 100
                        self.paralel_points[-1].dist_a.append(
                            pt.distance_pts(self.limmit_coords[0], destiny_points[i])
                            * multiplier
                        )
                    if (
                        destiny_points[i][0] == ys_do_buraco[1]
                        and destiny_points[i][1] <= xs_do_buraco[1]
                    ):
                        self.paralel_points[-1].lista_b.append(destiny_points[i])
                        linha = it.draw_line(
                            np.zeros_like(prohibited_areas),
                            self.limmit_coords[1],
                            destiny_points[i],
                        )
                        linha = mt.dilation(linha, kernel_size=path_radius * 2 + 2)
                        if np.sum(np.logical_and(linha, prohibited_areas)) > 0:
                            multiplier = 100
                        self.paralel_points[-1].dist_b.append(
                            pt.distance_pts(self.limmit_coords[1], destiny_points[i])
                            * multiplier
                        )
                    if (
                        destiny_points[i][0] == ys_do_buraco[2]
                        and destiny_points[i][1] >= xs_do_buraco[2]
                    ):
                        self.paralel_points[-1].lista_c.append(destiny_points[i])
                        linha = it.draw_line(
                            np.zeros_like(prohibited_areas),
                            self.limmit_coords[2],
                            destiny_points[i],
                        )
                        linha = mt.dilation(linha, kernel_size=path_radius * 2 + 2)
                        if np.sum(np.logical_and(linha, prohibited_areas)) > 0:
                            multiplier = 100
                        self.paralel_points[-1].dist_c.append(
                            pt.distance_pts(self.limmit_coords[2], destiny_points[i])
                            * multiplier
                        )
                    if (
                        destiny_points[i][0] == ys_do_buraco[3]
                        and destiny_points[i][1] >= xs_do_buraco[3]
                    ):
                        self.paralel_points[-1].lista_d.append(destiny_points[i])
                        linha = it.draw_line(
                            np.zeros_like(prohibited_areas),
                            self.limmit_coords[3],
                            destiny_points[i],
                        )
                        linha = mt.dilation(linha, kernel_size=path_radius * 2 + 2)
                        if np.sum(np.logical_and(linha, prohibited_areas)) > 0:
                            multiplier = 100
                        self.paralel_points[-1].dist_d.append(
                            pt.distance_pts(self.limmit_coords[3], destiny_points[i])
                            * multiplier
                        )
        return


def link_spirals(route, n_loops, mask, region, base_frame, prohibited_areas):
    guide_line, idx = path_tools.generate_guide_line(
        region, base_frame, prohibited_areas
    )
    work_area = mt.dilation(guide_line, kernel_img=mask)
    _, work_area_contour_img = mt.detect_contours(work_area, return_img=True)
    # work_area_contour = points_tools.contour_to_list(work_area_contour)
    points = path_tools.intersection_points_w_rectangle(
        work_area_contour_img, route, idx
    )
    work_area_contour = path_tools.line_img_to_freeman_chain(
        work_area_contour_img, points[0]
    )
    borda_cortada = path_tools.spiral_cut(
        work_area_contour, route, points, n_loops, base_frame, idx
    )
    route = np.logical_and(route, np.logical_not(work_area))
    route = np.logical_or(route, borda_cortada)
    if n_loops > 2:
        intersection_pol = it.draw_polyline(np.zeros(base_frame), points, True)
        intersection_pol = it.fill_internal_area(intersection_pol, np.ones(base_frame))
        guide_line = np.logical_and(guide_line, intersection_pol)
        rectangle_contour = mt.detect_contours(intersection_pol)
        rectangle_contour = pt.contour_to_list(rectangle_contour)
        cut_rectangle = path_tools.rectangle_cut(
            rectangle_contour, guide_line, points, n_loops, base_frame, 0, idx
        )
        route = np.logical_or(route, guide_line)
        route = np.logical_or(route, cut_rectangle)
    next_prohibited_areas = work_area
    return route, next_prohibited_areas
