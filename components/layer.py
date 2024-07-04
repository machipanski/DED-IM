from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from files import Paths
from components import images_tools as it
from components import morphology_tools as mt
from components.thin_walls import ThinWallRegions
from components.offset import OffsetRegions
from components import bottleneck
from components import zigzag
from timer import Timer
import numpy as np
from typing import List
import concurrent.futures
# import matplotlib.pyplot as plt

class Island:
    def __init__(self, name: int, img_name: str):
        self.name = name
        self.img = img_name
        self.thin_walls: ThinWallRegions
        self.offsets: OffsetRegions
        self.rest_of_picture_f1: np.ndarray

class Layer:
    """Cada uma das camadas geradas pelo processo de slicing no tratamento de modelos 3D"""

    def __init__(self):
        # TODO: retirar todos os parametros que nao uso
        # TODO: dar o Typing em todos
        self.name: int
        self.original_img: np.ndarray
        self.odd_layer: int = 0
        self.dpi: int = 0
        self.base_frame = []
        self.layer_height: float = 0
        self.nozzle_diam_external: float = 0
        self.nozzle_diam_internal: float = 0
        self.path_radius_external: float = 0
        self.path_radius_internal: float = 0
        # self.mask_full_ext: np.ndarray
        # self.mask_half_ext: np.ndarray
        # self.mask_3_4_ext: np.ndarray
        # self.mask_3_2_ext: np.ndarray
        # self.mask_double_ext: np.ndarray
        # self.mask_full_int: np.ndarray
        # self.mask_half_int: np.ndarray
        # self.mask_3_4_int: np.ndarray 
        # self.mask_3_2_int: np.ndarray
        # self.mask_double_int: np.ndarray
        self.pxl_per_mm: float = 0
        self.mm_per_pxl: float = 0
        self.islands: List[Island]

        # self.n_camadas = 0
        # self.void_max = 0
        # self.max_external_walls = 0
        # self.max_internal_walls = 0
        # 
        # self.offsets = []
        # self.zigzags = []
        # self.bridges = []
        # self.rest_of_picture_f1 = []
        # self.rest_of_picture_f2 = []
        # self.rest_of_picture_f3 = []
        # self.offsets_graph = []
        # self.offsets_mst = []
        # self.zigzags_graph = []
        # self.zigzags_mst = []
        # self.all_zigzags = []
        # self.macro_areas = []
        # self.pos_zigzag_nodes = []
        # self.external_tree_route = []
        # self.internal_tree_route = []
        # self.both_chains = []
        # self.final_chain = []
        # self.mudanca = []
        # self.saltos = []
        # self.final_route = []
        # self.n_max = []
        # self.prohibited_areas = []

    def make_input_img(
        self, name: int, img_path: str, dpi:int, odd_layer:bool, layer_height: float, n_camadas: int, arquivos: Paths
    ):
        """Usa o Path dos arquivos para importar as imagens e transforma-las em binarias, assim como ja cria um objeto Layer pra cada"""
        self.name = name
        img = it.read_img_add_border(img_path)
        if odd_layer:
            img = it.rotate_img_cw(img)
            self.odd_layer = 1
        # self.original_img = img
        self.dpi = dpi
        self.base_frame = img.shape
        self.n_camadas = n_camadas
        self.layer_height = layer_height
        return img

    def make_thin_walls(
        self, folders:Paths, nozzle_diam_external: float, nozzle_diam_internal: float
    ) -> None:

        def load_and_make_thinWalls(island:Island):
            island_img = folders.load_island_img(island)
            island.thin_walls = ThinWallRegions()
            island.thin_walls.make_thin_walls(self.name, island.name, island_img, self.base_frame, self.path_radius_external, mt.make_mask(self,"full_ext"), folders)
            return island
        
        self.pxl_per_mm = self.dpi / 25.4
        self.mm_per_pxl = 1 / self.pxl_per_mm
        nozzle_diam_external_pxl = nozzle_diam_external * self.pxl_per_mm
        self.path_radius_external = int(nozzle_diam_external_pxl * 0.5)
        nozzle_diam_internal_pxl = nozzle_diam_internal * self.pxl_per_mm
        self.path_radius_internal = int(nozzle_diam_internal_pxl * 0.5)
        self.nozzle_diam_external = nozzle_diam_external
        with Timer("Criando paredes finas"):

            processed_regions = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [executor.submit(load_and_make_thinWalls, island) for island in self.islands]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x.name)
            self.islands = processed_regions

            # for island in self.islands:
            #     island_img = folders.load_island_img(island)
            #     island.thin_walls = ThinWallRegions()
            #     island.thin_walls.make_thin_walls(self.name, island.name, island_img, self.base_frame, self.path_radius_external, mt.make_mask(self,"full_ext"), folders)
            #     folders.prepare_tw_json(self, island.thin_walls)
            with Timer("Retirando Paredes finas da camada"):
                for island in self.islands:
                    island_img = folders.load_island_img(island)
                    island_rest_of_picture_f1 = island.thin_walls.apply_thin_walls(
                        folders, island_img, self.base_frame
                    )
                    img_f1_name = (f"L{self.name:03d}_I{island.name:03d}_restF1.png")
                    folders.save_img(img_f1_name, island_rest_of_picture_f1)
                    island.rest_of_picture_f1 = img_f1_name
                folders.save_layer_json(self)
        return

def divide_islands(folders: Paths):
    layer_names = folders.list(layers = 1)
    for ln in layer_names:
        layer = folders.load_layer_json(ln)
        img = folders.load_layer_orig_img(layer)
        separated_imgs, labels, num = it.divide_by_connected(img)
        islands = []
        for i, si in enumerate(separated_imgs):
            island_name = (f"L{layer.name:03d}_I{i}.png")
            folders.save_img(island_name, si)
            islands.append(Island(i, island_name))
        layer.islands = islands
        folders.save_layer_json(layer)
    return