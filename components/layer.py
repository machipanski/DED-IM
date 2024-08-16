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
from components import offset
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
        self.rest_of_picture_f1: np.ndarray = []
        self.rest_of_picture_f2: np.ndarray = []


class Layer:
    """Cada uma das camadas geradas pelo processo de slicing no tratamento de modelos 3D"""

    def __init__(self):
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
        self.pxl_per_mm: float = 0
        self.mm_per_pxl: float = 0
        self.islands: List[Island]

    def make_input_img(
        self,
        name: int,
        img_path: str,
        dpi: int,
        odd_layer: bool,
        layer_height: float,
        n_camadas: int,
        arquivos: Paths,
    ):
        """Usa o Path dos arquivos para importar as imagens e transforma-las em binarias, assim como ja cria um objeto Layer pra cada"""
        self.name = name
        img = it.read_img_add_border(img_path)
        if odd_layer:
            img = it.rotate_img_cw(img)
            self.odd_layer = 1
        self.dpi = dpi
        self.base_frame = img.shape
        self.n_camadas = n_camadas
        self.layer_height = layer_height
        return img

    def make_thin_walls(
        self, folders: Paths, nozzle_diam_external: float, nozzle_diam_internal: float
    ) -> None:

        def load_and_make_thinWalls(island: Island):
            island_img = folders.load_img(island.img)
            island.thin_walls = ThinWallRegions()
            island.thin_walls.make_thin_walls(
                self.name,
                island.name,
                island_img,
                self.base_frame,
                self.path_radius_external,
                mt.make_mask(self, "full_ext"),
                folders,
            )
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
                results = [
                    executor.submit(load_and_make_thinWalls, island)
                    for island in self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x.name)
            self.islands = processed_regions

            with Timer("Retirando Paredes finas da camada"):
                for island in self.islands:
                    island_img = folders.load_img(island.img)
                    island_rest_of_picture_f1 = island.thin_walls.apply_thin_walls(
                        folders, island_img, self.base_frame
                    )
                    img_f1_name = f"L{self.name:03d}_I{island.name:03d}_restF1.png"
                    folders.save_img(img_f1_name, island_rest_of_picture_f1)
                    island.rest_of_picture_f1 = img_f1_name
                folders.save_layer_json(self)
        return

    def make_offsets(
        self, folders: Paths, void_max: float, external_max: int, internal_max: int
    ) -> None:

        self.void_max = void_max
        self.max_external_walls = external_max
        self.max_internal_walls = internal_max
        # self.offsets = offset.OffsetRegions()

        def load_and_make_levels(island: Island) -> Island:
            island.offsets = OffsetRegions()
            island.offsets.create_levels(
                folders,
                island.rest_of_picture_f1,
                mt.make_mask(self, "full_ext"),
                mt.make_mask(self, "double_ext"),
                self.name,
                island.name,
            )
            return island

        with Timer("Criando Lvls"):
            processed_isl = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(load_and_make_levels, island)
                    for island in self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_isl.append(l.result())
            processed_isl.sort(key=lambda x: x.name)
            self.islands = processed_isl

        def create_loops_in_island(original_img, island: Island) -> Island:
            for level in island.offsets.levels:
                level.create_loops(
                    level.name,
                    folders.load_img(level.img),
                    mt.make_mask(self, "full_ext"),
                    self.base_frame,
                    original_img,
                    folders,
                    self.name,
                    island.name,
                    level.name,
                )
            return island

        with Timer("Criando os loops"):
            processed_regions = []
            original_img = folders.load_img(self.original_img)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(create_loops_in_island, original_img, island)
                    for island in self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x.name)
            self.islands = processed_regions
            # levels = self.offsets.create_loops(mask_full_ext, base_frame, levels)

        def create_influence_regions_in_island(base_frame, island: Island) -> Island:
            island.offsets.create_influence_regions(base_frame, folders)
            return island

        with Timer("Criando regiões de influência"):
            processed_regions = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(
                        create_influence_regions_in_island, self.base_frame, island
                    )
                    for island in self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x.name)
            self.islands = processed_regions
            # influence_regions = self.offsets.create_influence_regions(self, levels, n_levels)

        def calculate_island_voids(island: Island):
            island.offsets.calculate_voids(
                self.base_frame,
                island.rest_of_picture_f1,
                island.offsets.levels,
                island.offsets.n_levels,
                self.path_radius_external,
                folders,
            )
            return island

        with Timer("Criando as regiões de Offset"):
            processed_regions = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(calculate_island_voids, island)
                    for island in self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x.name)
            self.islands = processed_regions
            # self.offsets.calculate_voids_V2(self.base_frame, self.rest_of_picture_f1, levels, n_levels, self.path_radius_external)

        def make_regions_in_islands(island: Island) -> Island:
            island.rest_of_picture_f2 = island.offsets.make_regions(
                self.original_img,
                self.base_frame,
                self.path_radius_external,
                self.void_max,
                self.max_external_walls,
                self.max_internal_walls,
                folders
            )
            return island

        with Timer("Retirando regiões da camada"):
            processed_regions = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(make_regions_in_islands, island)
                    for island in self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x.name)
            self.islands = processed_regions
            # self.rest_of_picture_f2 = self.offsets.make_regions(
            #     self.original_img,
            #     self.base_frame,
            #     self.path_radius_external,
            #     self.void_max,
            #     self.max_external_walls,
            #     self.max_internal_walls,
            #     influence_regions,
            #     levels,
            # )
        def make_island_valid_loops(island:Island) -> Island:
            island.offsets.all_valid_loops = island.offsets.make_valid_loops(self.base_frame, folders)
            return island

        with Timer("Reunindo todos os loops em uma unica imagem"):
            processed_regions = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(make_island_valid_loops, island)
                    for island in self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x.name)
            self.islands = processed_regions
            # self.offsets.all_valid_loops = self.offsets.make_valid_loops(self)
        # BBBBB = images_tools.sum_imgs_colored([x.img for x in self.offsets.regions])
        return


def divide_islands(folders: Paths):
    layer_names = folders.list(layers=1)
    for ln in layer_names:
        layer = folders.load_layer_json(ln)
        img = folders.load_img(layer.original_img)
        separated_imgs, labels, num = it.divide_by_connected(img)
        islands = []
        for i, si in enumerate(separated_imgs):
            island_name = f"L{layer.name:03d}_I{i}.png"
            folders.save_img(island_name, si)
            islands.append(Island(i, island_name))
        layer.islands = islands
        folders.save_layer_json(layer)
    return
