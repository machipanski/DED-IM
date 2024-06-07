from components import images_tools as it
from components import morphology_tools as mt
from components import thin_walls
from components import bottleneck
from components import offset
from components import zigzag
from timer import Timer
import numpy as np


class Layer:
    """Cada uma das camadas geradas pelo processo de slicing no tratamento de modelos 3D"""

    def __init__(self):
        # TODO: retirar todos os parametros que nao uso
        # TODO: dar o Typing em todos
        self.name: str = "n"
        self.original_img: np.ndarray = []
        self.odd_layer: int = 0
        self.dpi: int = 0
        self.base_frame = []
        self.nozzle_diam_external: float = 0
        self.nozzle_diam_internal: float = 0
        self.path_radius_external: float = 0
        self.path_radius_internal: float = 0
        self.mask_full_ext: np.ndarray = []
        self.mask_half_ext: np.ndarray = []
        self.mask_3_4_ext: np.ndarray = []
        self.mask_3_2_ext: np.ndarray = []
        self.mask_double_ext: np.ndarray = []
        self.mask_full_int: np.ndarray = []
        self.mask_half_int: np.ndarray = []
        self.mask_3_4_int: np.ndarray = []
        self.mask_3_2_int: np.ndarray = []
        self.mask_double_int: np.ndarray = []
        # self.n_camadas = 0
        self.layer_height: float = 0
        # self.void_max = 0
        # self.max_external_walls = 0
        # self.max_internal_walls = 0
        self.pxl_per_mm: float = 0
        self.mm_per_pxl: float = 0
        # self.thin_walls = []
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
        self, img_path, dpi, odd_layer, layer_height, n_camadas, arquivos
    ):
        """Usa o Path dos arquivos para importar as imagens e transforma-las em binarias, assim como ja cria um objeto Layer pra cada"""
        img = it.read_img_add_border(img_path)
        if odd_layer:
            img = it.rotate_img_cw(img)
            self.odd_layer = 1
        self.original_img = img
        self.dpi = dpi
        self.base_frame = img.shape
        self.n_camadas = n_camadas
        self.layer_height = layer_height
        return

    def make_thin_walls(
        self, nozzle_diam_external: float, nozzle_diam_internal: float
    ) -> None:

        self.thin_walls = thin_walls.ThinWallRegions()
        self.nozzle_diam_external = nozzle_diam_external
        self.pxl_per_mm = self.dpi / 25.4
        self.mm_per_pxl = 1 / self.pxl_per_mm
        nozzle_diam_external_pxl = nozzle_diam_external * self.pxl_per_mm
        self.path_radius_external = int(nozzle_diam_external_pxl * 0.5)
        nozzle_diam_internal_pxl = nozzle_diam_internal * self.pxl_per_mm
        self.path_radius_internal = int(nozzle_diam_internal_pxl * 0.5)
        mt.make_masks(self)
        with Timer("Criando paredes finas"):
            self.thin_walls.make_thin_wallsV2(
                self.original_img,
                self.base_frame,
                self.path_radius_external,
                self.mask_full_ext,
            )
        with Timer("Retirando Paredes finas da camada"):
            self.rest_of_picture_f1 = self.thin_walls.apply_thin_walls(
                self.original_img, self.base_frame
            )
        return
