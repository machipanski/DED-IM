from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from files import Paths
    from typing import List
from components import morphology_tools as mt
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

    def __init__(self, nome, img, origem, mask):
        self.name = nome
        self.offset_level = origem
        self.internal_area = []
        self.trail = mt.dilation(img, kernel_img=mask)
        self.route = img
        self.region = ""
        self.lost_area = np.zeros_like(img)
        self.lost_area_sum = 0
        self.acceptable = 0
        self.loops_filhos = []

class Level:
    """Cada elemento sequencial gerado pela operação de Offset em uma imagem"""

    def __init__(self, img:str, nome, pai, area):
        self.name = nome
        self.parent = pai
        self.img = img
        self.outer_loops = []
        self.hole_loops = []
        self.area = area
        self.filled_area = area
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
        return Level(new_lvl_img, nome_filho, self.name, new_lvl_area)
    
        
    def create_loops(self, name, img, mask, base_frame):
        """Usa cv2.findContours() para separar cada loop dentro imagem do Offset Level"""
        contours, hierarchy = mt.detect_contours(img, return_hierarchy=True)
        in_counter = 0
        out_counter = 0
        for i in np.arange(0, len(contours)):
            loop = drawContours(np.zeros(base_frame), contours, i, 1)
            if (hierarchy[0][i][3] == -1):  
                self.outer_loops.append(Loop(out_counter, loop, name, mask))
                out_counter += 1
            else:
                self.hole_loops.append(Loop(in_counter, loop, name, mask))
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

