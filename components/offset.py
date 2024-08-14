from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from files import Paths
    from typing import List
    from components.layer import Layer
from components import morphology_tools as mt
from components import images_tools as it
import numpy as np
from cv2 import drawContours
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

    def __init__(self, img:str, nome, pai, area):
        self.name = nome
        self.parent = pai
        self.img = img
        self.outer_loops:List[Loop] = []
        self.hole_loops:List[Loop] = []
        self.area = area
        #self.filled_area = area
        self.outer_areas = []
        self.hole_areas = []
    
    def create_level(self, folders: Paths, mask_full:np.ndarray, mask_double:np.ndarray, nome_filho, layer_name, island_name, first=False):
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
        
        name_new_lvl = (f"L{layer_name:03d}_I{island_name:03d}_LVL{nome_filho:03d}.png")
        folders.save_img(name_new_lvl,new_lvl_img)
        new_lvl_img = name_new_lvl

        name_new_lvl_area = (f"L{layer_name:03d}_I{island_name:03d}_LVL{nome_filho:03d}_Area.png")
        folders.save_img(name_new_lvl_area,new_lvl_area)
        new_lvl_area = name_new_lvl_area
        
        return Level(new_lvl_img, nome_filho, self.name, new_lvl_area)
    
        
    def create_loops(self, name, img, mask, base_frame, orig_img, folders:Paths, layer_name, island_name, level_name):
        """Usa cv2.findContours() para separar cada loop dentro imagem do Offset Level"""
        contours, hierarchy = mt.detect_contours(img, return_hierarchy=True)
        in_counter = 0
        out_counter = 0
        for i in np.arange(0, len(contours)):
            loop = drawContours(np.zeros(base_frame), contours, i, 1)
            if (hierarchy[0][i][3] == -1):  
                name_loop = (f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopOut{out_counter:03d}.npz")
                folders.save_npz(name_loop,loop)
                trail_name = (f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopOut{out_counter:03d}_trail.npz")
                trail_img = mt.dilation(loop, kernel_img=mask)
                folders.save_npz(trail_name, trail_img)
                self.outer_loops.append(Loop(out_counter, name_loop, name, trail_name))
                self.outer_loops[-1].internal_area = it.fill_internal_area(folders.load_npz(self.outer_loops[-1].trail), orig_img)
                name_loop_area = (f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopOut{out_counter:03d}_Area.png")
                folders.save_img(name_loop_area,self.outer_loops[-1].internal_area)
                self.outer_loops[-1].internal_area = name_loop_area
                out_counter += 1
            else:
                name_loop = (f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopInt{in_counter:03d}.npz")
                folders.save_npz(name_loop,loop)
                trail_name = (f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopInt{in_counter:03d}_trail.npz")
                trail_img = mt.dilation(loop, kernel_img=mask)
                folders.save_npz(trail_name, trail_img)
                self.hole_loops.append(Loop(in_counter, name_loop, name, trail_name))
                self.hole_loops[-1].internal_area = it.fill_internal_area(folders.load_npz(self.hole_loops[-1].trail), orig_img)
                name_loop_area = (f"L{layer_name:03d}_I{island_name:03d}_LVL{level_name:03d}_LoopInt{in_counter:03d}_Area.png")
                folders.save_img(name_loop_area,self.hole_loops[-1].internal_area)
                self.hole_loops[-1].internal_area = name_loop_area
                in_counter += 1
        return self
class OffsetRegions:

    def __init__(self):
        self.all_valid_loops = []
        self.regions = []
        self.n_regions = []
        self.next_prohibited_areas = []
        self.levels:List[Level] = []
        self.n_levels = []
    
    def create_levels(self, folders, rest_of_picture_f1, mask_full, mask_double, layer_name, island_name):
        """Vai de fora para dentro realizando erosões e organiza em níveis
        até não haver mais pixels a se processar"""
        levels = []
        n_levels = 0
        atual = Level(rest_of_picture_f1, 0, "", rest_of_picture_f1)
        atual = atual.create_level(folders, mask_full, mask_double, n_levels, layer_name, island_name, first=True)
        levels.append(atual)
        n_levels += 1
        atual = atual.create_level(folders, mask_full, mask_double, n_levels, layer_name, island_name)
        while np.sum(folders.load_img(atual.img)) > 0:  # Enquanto a imagem não estiver vazia1
            levels.append(atual)
            n_levels += 1
            atual = atual.create_level(folders, mask_full, mask_double, n_levels, layer_name, island_name)
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
        region_counter = self.label_offset_regions(self.levels, seq_in_out, 0, 1, folders)
        region_counter_holes = self.label_offset_regions(self.levels, seq_out_in, region_counter, 0, folders)
        """Processo de eliminação de cada img. a região final é a img que contém completamente as outras"""
        influence_regions = self.sum_areas(
            self.levels,
            influence_regions,
            np.arange(1, region_counter + 1),
            0,
            base_frame,
            folders
        )  # agrupa todas as áreas de loops externos
        influence_regions = self.sum_areas(
            self.levels,
            influence_regions,
            np.arange(region_counter + 1, region_counter_holes + 1),
            1,
            base_frame,
            folders
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
        return influence_regions
    
    def label_offset_regions(self, levels:List[Level], sequence, counter_start, order, folders:Paths):
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
                    loop_cam_atual.region = (region_counter)# e dá o nome dela. essa será uma seed
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
    
    def sum_areas(self, levels, influence_regions, interval, outer_inner, base_frame, folders: Paths):
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
                        essa_reg = np.logical_or(essa_reg, folders.load_img(loop.internal_area))
                        loops_inside.append([level.name, outer_inner, loop.name])
            influence_regions.append(Region(i, essa_reg, loops_inside))
        return influence_regions

    def calculate_voids_V2(
        self, base_frame, start_of_process_img, levels, n_levels, path_radius
    ):
        offset_result = self.calculate_covered_area(base_frame, levels)
        for l in np.arange(0, n_levels):
            if l == 0:
                levels[l].area_lost = np.logical_and(
                    start_of_process_img, np.logical_not(levels[l].area)
                )
            else:
                levels[l].area_lost = np.logical_and(
                    levels[l - 1].area, np.logical_not(levels[l].area)
                )
            not_used_area = levels[l].divide_areas_v2(
                base_frame, start_of_process_img, path_radius
            )
            if np.sum(not_used_area) > 0:
                levels[l - 1].area_lost = np.logical_or(
                    levels[l - 1].area_lost, not_used_area
                )
                _ = levels[l - 1].divide_areas_v2(
                    base_frame, start_of_process_img, path_radius
                )
                levels[l - 1].calculate_lost_area(offset_result)
            levels[l].calculate_lost_area(offset_result)
        # AAAAA = levels[0].area_lost
        return
    
class Region:
    """Caminho fechado individual por Offset paralelo"""

    def __init__(self, name, img, loops):
        self.name = name
        self.img = img
        self.loops = loops
        self.limmit_coords = ([])  # coordenadas dos pontos onde se separam as regiões monotônicas
        self.center_coords = []  # coordenadas do centro geométrico de cada contorno
        self.area_contour = []  # contorno de cada área
        self.area_contour_img = []
        self.internal_area = []  # resultante de se pintar o interior de cada contorno
        self.hierarchy = (0) # hierarquia de contornos, quais são internos e quais são externos
        self.paralel_points = []
        self.route = []
        self.trail = []
        self.next_prohibited_area = []

