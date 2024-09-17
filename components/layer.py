from __future__ import annotations
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from files import Paths
from components import images_tools as it
from components import morphology_tools as mt
from components.thin_walls import ThinWallRegions
from components.offset import OffsetRegions
from components.zigzag import ZigZagRegions
from components.bottleneck import BridgeRegions
from timer import Timer
import numpy as np
from typing import List
import concurrent.futures
from functools import wraps


class Island:
    def __init__(self, *args, **kwargs):
        self.name: str
        self.img = []
        self.thin_walls: ThinWallRegions
        self.offsets: OffsetRegions
        self.bridges: BridgeRegions
        self.zigzags: ZigZagRegions
        self.zigzags_graph = []
        self.zigzags_mst = []
        self.pos_zigzag_nodes = []
        self.rest_of_picture_f1 = np.ndarray([])
        self.rest_of_picture_f2 = np.ndarray([])
        self.rest_of_picture_f3 = np.ndarray([])
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        if args:
            self.name = args[0]
            self.img = args[1]


class Layer:
    """Cada uma das camadas geradas pelo processo de slicing no tratamento de modelos 3D"""

    def __init__(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
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
        """Usa o Path dos arquivos para importar as imagens e transforma-las em binarias,
        assim como ja cria um objeto Layer pra cada"""
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

        def load_and_make_thinWalls(island: Island) -> List[Island, dict]:
            island_img = folders.load_img_hdf5(f"/{self.name}/{island.name}", "img")
            island.thin_walls = ThinWallRegions()
            imgs_pack = island.thin_walls.make_thin_walls(
                self.name,
                island.name,
                island_img,
                self.base_frame,
                self.path_radius_external,
                mt.make_mask(self, "full_ext"),
            )
            return [island, imgs_pack]

        self.pxl_per_mm = self.dpi / 25.4
        self.mm_per_pxl = 1 / self.pxl_per_mm
        nozzle_diam_external_pxl = nozzle_diam_external * self.pxl_per_mm
        self.path_radius_external = int(nozzle_diam_external_pxl * 0.5)
        nozzle_diam_internal_pxl = nozzle_diam_internal * self.pxl_per_mm
        self.path_radius_internal = int(nozzle_diam_internal_pxl * 0.5)
        self.nozzle_diam_external = nozzle_diam_external
        for isl in self.islands:
            folders.create_new_hdf5_group(f"/{self.name}/{isl.name}/thin_walls")

        with Timer("Criando paredes finas"):
            processed_regions: List[Island] = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(load_and_make_thinWalls, island)
                    for island in self.islands  # self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x[0].name)
            self.islands = [x[0] for x in processed_regions]

            with Timer("salvando imagens das regiões"):
                img_pack = [x[1] for x in processed_regions]
                for pack in img_pack:
                    if not (pack["groups"] == []):
                        folders.save_img_hdf5(*pack["all_tw"])
                        folders.save_img_hdf5(*pack["all_tw_origins"])
                        for group in pack["groups"]:
                            folders.create_new_hdf5_group(group[0])
                        for img in pack["tw_img"]:
                            folders.save_img_hdf5(*img)
                        for img in pack["mat"]:
                            folders.save_img_hdf5(*img, type="f8")
                        for img in pack["l1"]:
                            folders.save_img_hdf5(*img)
                        for img in pack["l2"]:
                            folders.save_img_hdf5(*img)
                        for img in pack["lt"]:
                            folders.save_img_hdf5(*img)
                        for img in pack["lb"]:
                            folders.save_img_hdf5(*img)
                        for img in pack["origins"]:
                            folders.save_img_hdf5(*img)

            with Timer("Retirando Paredes finas da camada"):
                for island in self.islands:
                    island_img = folders.load_img_hdf5(
                        f"/{self.name}/{island.name}", "img"
                    )
                    if hasattr(island, 'bridges'):
                        island_rest_of_picture_f1 = island.thin_walls.apply_thin_walls(
                            folders, island_img, self.base_frame
                        )
                    else:
                        island_rest_of_picture_f1 = island.img
                    folders.save_img_hdf5(
                        f"{self.name}/{island.name}",
                        "rest_of_picture_f1",
                        island_rest_of_picture_f1,
                    )
        folders.save_props_hdf5(f"/{self.name}", self.__dict__)
        return

    def make_offsets(
        self, folders: Paths, void_max: float, external_max: int, internal_max: int
    ) -> None:

        def paralelizando(func):
            @wraps(func)
            def wrapper(*args):
                processed_isl = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    if [*args]:
                        results = [
                            executor.submit(func, island, *args)
                            for island in self.islands
                        ]
                    else:
                        results = [
                            executor.submit(func, island) for island in self.islands
                        ]
                    for l in concurrent.futures.as_completed(results):
                        processed_isl.append(l.result())
                    processed_isl.sort(key=lambda x: x.name)
                    self.islands = processed_isl

            return wrapper

        @paralelizando
        def load_and_make_levels(island: Island) -> Island:
            island.offsets = OffsetRegions()
            rest_of_picture_f1 = folders.load_img_hdf5(
                f"{self.name}/{island.name}", "rest_of_picture_f1"
            )
            if np.sum(rest_of_picture_f1) > 0:
                island.offsets.create_levels(
                    rest_of_picture_f1,
                    mt.make_mask(self, "full_ext"),
                    mt.make_mask(self, "double_ext"),
                    self.name,
                    island.name,
                )
            return island

        @paralelizando
        def create_loops_in_island(island: Island) -> Island:
            rest_of_picture_f1 = folders.load_img_hdf5(
                f"/{self.name}/{island.name}", "rest_of_picture_f1"
            )
            if np.sum(rest_of_picture_f1) > 0:
                for level in island.offsets.levels:
                    level.create_loops(
                        mt.make_mask(self, "full_ext"),
                        self.base_frame,
                        rest_of_picture_f1,
                        self.name,
                        island.name,
                        level.name,
                    )
            return island

        @paralelizando
        def create_influence_regions_in_island(island: Island) -> Island:
            rest_of_picture_f1 = folders.load_img_hdf5(
                f"/{self.name}/{island.name}", "rest_of_picture_f1"
            )
            if np.sum(rest_of_picture_f1) > 0:
                island.offsets.create_influence_regions(self.base_frame)
            return island

        @paralelizando
        def calculate_island_voids(island: Island):
            rest_of_picture_f1 = folders.load_img_hdf5(
                f"/{self.name}/{island.name}", "rest_of_picture_f1"
            )
            if np.sum(rest_of_picture_f1) > 0:
                island.offsets.calculate_voids(
                    self.base_frame,
                    rest_of_picture_f1,
                    island.offsets.levels,
                    island.offsets.n_levels,
                    self.path_radius_external,
                )
            return island

        @paralelizando
        def make_regions_in_islands(island: Island) -> Island:
            rest_of_picture_f1 = folders.load_img_hdf5(
                f"/{self.name}/{island.name}", "rest_of_picture_f1"
            )
            if np.sum(rest_of_picture_f1) > 0:
                island.rest_of_picture_f2 = island.offsets.make_regions(
                    rest_of_picture_f1,
                    self.base_frame,
                    self.path_radius_external,
                    self.void_max,
                    self.max_external_walls,
                    self.max_internal_walls,
                    np.sum(mt.make_mask(self, "full_int")),
                )
            return island

        @paralelizando
        def make_island_valid_loops(island: Island) -> Island:
            rest_of_picture_f1 = folders.load_img_hdf5(
                f"/{self.name}/{island.name}", "rest_of_picture_f1"
            )
            if np.sum(rest_of_picture_f1) > 0:
                island.offsets.all_valid_loops = island.offsets.make_valid_loops(
                    self.base_frame
                )
            return island

        self.void_max = void_max
        self.max_external_walls = external_max
        self.max_internal_walls = internal_max
        for isl in self.islands:
            folders.create_new_hdf5_group(f"/{self.name}/{isl.name}/offsets")

        with Timer("Criando Lvls"):
            load_and_make_levels()

        with Timer("Criando os loops"):
            create_loops_in_island()

        with Timer("Criando regiões de influência"):
            create_influence_regions_in_island()

        with Timer("Criando as regiões de Offset"):
            calculate_island_voids()

        with Timer("Retirando regiões da camada"):
            make_regions_in_islands()

        with Timer("Reunindo todos os loops em uma unica imagem"):
            make_island_valid_loops()

        with Timer("salvando imagens das regiões"):
            for isl in self.islands:
                if np.sum(isl.rest_of_picture_f2) > 0:
                    folders.save_img_hdf5(
                        f"/{self.name}/{isl.name}",
                        "rest_of_picture_f2",
                        isl.rest_of_picture_f2,
                    )
                    isl.rest_of_picture_f2 = (
                        f"/{self.name}/{isl.name}/rest_of_picture_f2"
                    )
                    folders.save_img_hdf5(
                        f"/{self.name}/{isl.name}/offsets",
                        "all_loops",
                        isl.offsets.all_valid_loops.astype(bool),
                    )
                    isl.offsets.all_valid_loops = (
                        f"/{self.name}/{isl.name}/offsets/all_loops"
                    )
                    for reg in isl.offsets.regions:
                        folders.create_new_hdf5_group(
                            f"/{self.name}/{isl.name}/offsets/Reg_{reg.name:03d}"
                        )
                        folders.save_props_hdf5(
                            f"/{self.name}/{isl.name}/offsets/Reg_{reg.name:03d}",
                            reg.__dict__,
                        )
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/offsets/Reg_{reg.name:03d}",
                            "img",
                            reg.img.astype(bool),
                        )
                        folders.create_new_hdf5_group(
                            f"/{self.name}/{isl.name}/offsets/Reg_{reg.name:03d}/loops"
                        )
                        for i, loop in enumerate(reg.loops):
                            folders.save_img_hdf5(
                                f"/{self.name}/{isl.name}/offsets/Reg_{reg.name:03d}/loops",
                                f"Lp_{i:03d}",
                                loop.route.astype(bool),
                            )
                            folders.save_props_hdf5(
                                f"/{self.name}/{isl.name}/offsets/Reg_{reg.name:03d}/loops/Lp_{i:03d}",
                                loop.__dict__,
                            )
        folders.save_props_hdf5(f"/{self.name}", self.__dict__)
        return

    def make_offset_routes(self, amendment_size, folders: Paths):
        folders.load_islands_hdf5(self)
        for isl in self.islands:
            folders.load_offsets_hdf5(self.name, isl)
            isl.offsets.make_routes_o(
                self.base_frame,
                mt.make_mask(self, "full_ext"),
                mt.make_mask(self, "double_ext"),
                [],
                self.path_radius_external,
                amendment_size,
                folders,
            )

    def make_bridges(self, n_max: float, nozzle_diam_internal: float, folders: Paths):

        def paralelizando(func):
            @wraps(func)
            def wrapper(*args):
                processed_isl = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    if [*args]:
                        results = [
                            executor.submit(func, island, *args)
                            for island in self.islands
                        ]
                    else:
                        results = [
                            executor.submit(func, island) for island in self.islands
                        ]
                    for l in concurrent.futures.as_completed(results):
                        processed_isl.append(l.result())
                    processed_isl.sort(key=lambda x: x.name)
                    self.islands = processed_isl

            return wrapper

        @paralelizando
        def make_islands_offset_bridges(island: Island) -> Island:
            folders.load_offsets_hdf5(self.name, isl)
            try:
                rest_of_picture_f2 = folders.load_img_hdf5(
                    f"/{self.name}/{island.name}", "rest_of_picture_f2"
                )
                island.offsets_graph, island.offsets_mst, island.prohibited_areas = (
                    island.bridges.make_offset_bridges(
                        rest_of_picture_f2,
                        isl.offsets,
                        self.base_frame,
                        self.path_radius_external,
                        rest_of_picture_f2,
                        # island.prohibited_areas,
                    )
                )
            except:
                pass
            return island

        @paralelizando
        def make_islands_zigzag_bridges(island: Island) -> Island:
            folders.load_offsets_hdf5(self.name, isl)
            rest_of_picture_f2 = folders.load_img_hdf5(
                f"/{self.name}/{island.name}", "rest_of_picture_f2"
            )
            if np.sum(rest_of_picture_f2) > 0:
                nozzle_diam_internal_pxl = nozzle_diam_internal * self.pxl_per_mm
                self.path_radius_internal = int(nozzle_diam_internal_pxl * 0.5)
                island.bridges.make_zigzag_bridges(
                    rest_of_picture_f2,
                    rest_of_picture_f2,
                    self.base_frame,
                    self.path_radius_internal,
                    n_max,
                    mt.make_mask(self, "full_int"),
                    isl.offsets.all_valid_loops,
                    isl.offsets.regions,
                )
            return island

        @paralelizando
        def make_islands_cross_over_bridges(island: Island) -> Island:
            try:
                rest_of_picture_f2 = folders.load_img_hdf5(
                    f"/{self.name}/{island.name}", "rest_of_picture_f2"
                )
                self.n_max = n_max
                island.bridges.all_bridges = island.bridges.make_cross_over_bridges(
                    island.prohibited_areas, island.offsets_mst
                )
            except:
                pass
            return island

        self.n_max = n_max
        self.nozzle_diam_internal = nozzle_diam_internal
        folders.save_props_hdf5(f"/{self.name}", self.__dict__)
        folders.load_islands_hdf5(self)
        for isl in self.islands:
            isl.bridges = BridgeRegions()
            folders.create_new_hdf5_group(f"/{self.name}/{isl.name}/bridges")

        with Timer("Criando pontes de Offset"):
            make_islands_offset_bridges()

        with Timer("Criando pontes de Zigzag"):
            make_islands_zigzag_bridges()

        with Timer("Criando pontes de Crossover"):
            make_islands_cross_over_bridges()

        with Timer("Retirando pontes da Camada"):
            for island in self.islands:
                rest_of_picture_f2 = folders.load_img_hdf5(
                    f"/{self.name}/{island.name}", "rest_of_picture_f2"
                )
                island.rest_of_picture_f3 = island.bridges.apply_bridges(
                    rest_of_picture_f2, self.base_frame
                )

        with Timer("salvando imagens das regiões"):
            for island in self.islands:
                if np.sum(isl.rest_of_picture_f3) > 0:
                    folders.save_img_hdf5(
                        f"/{self.name}/{isl.name}",
                        "rest_of_picture_f3",
                        isl.rest_of_picture_f3,
                    )
                    folders.save_img_hdf5(
                        f"/{self.name}/{isl.name}/bridges",
                        "all_bridges",
                        isl.bridges.all_bridges,
                    )
                    folders.create_new_hdf5_group(
                        f"/{self.name}/{isl.name}/bridges/offset_bridges"
                    )
                    folders.create_new_hdf5_group(
                        f"/{self.name}/{isl.name}/bridges/zigzag_bridges"
                    )
                    folders.create_new_hdf5_group(
                        f"/{self.name}/{isl.name}/bridges/cross_over_bridges"
                    )
                    isl.rest_of_picture_f3 = (
                        f"/{self.name}/{isl.name}/rest_of_picture_f3"
                    )
                    for reg in isl.bridges.offset_bridges:
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/bridges/offset_bridges",
                            f"OB_{reg.name:03d}",
                            reg.img,
                        )
                        folders.save_props_hdf5(
                            f"/{self.name}/{isl.name}/bridges/offset_bridges/OB_{reg.name:03d}",
                            reg.__dict__,
                        )
                    for reg in isl.bridges.zigzag_bridges:
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/bridges/zigzag_bridges",
                            f"ZB_{reg.name:03d}",
                            reg.img,
                        )
                        folders.save_props_hdf5(
                            f"/{self.name}/{isl.name}/bridges/zigzag_bridges/ZB_{reg.name:03d}",
                            reg.__dict__,
                        )
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/bridges/zigzag_bridges",
                            f"ZB_{reg.name:03d}_origin",
                            reg.origin,
                        )
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/bridges/zigzag_bridges",
                            f"ZB_{reg.name:03d}_contorno",
                            reg.contorno,
                        )
                    for reg in isl.bridges.cross_over_bridges:
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/bridges/cross_over_bridges",
                            f"CB_{reg.name:03d}",
                            reg.img,
                        )
                        folders.save_props_hdf5(
                            f"/{self.name}/{isl.name}/bridges/cross_over_bridges/CB_{reg.name:03d}",
                            reg.__dict__,
                        )
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/bridges/cross_over_bridges",
                            f"CB_{reg.name:03d}_origin",
                            reg.origin,
                        )
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/bridges/cross_over_bridges",
                            f"CB_{reg.name:03d}_contorno",
                            reg.contorno,
                        )

        return

    def make_bridges_routes(self, folders: Paths):
        folders.load_islands_hdf5(self)
        for isl in self.islands:
            folders.load_bridges_hdf5(self.name, isl)
            folders.load_offsets_hdf5(self.name, isl)
            isl.bridges.make_routes_b(
                isl.offsets.regions,
                self.path_radius_external,
                self.path_radius_internal,
                self.base_frame,
                isl.rest_of_picture_f2,
                self.odd_layer,
                isl.offsets.all_valid_loops,
            )

    def make_zigzags(self, folders: Paths):

        def paralelizando(func):
            @wraps(func)
            def wrapper(*args):
                processed_isl = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    if [*args]:
                        results = [
                            executor.submit(func, island, *args)
                            for island in self.islands
                        ]
                    else:
                        results = [
                            executor.submit(func, island) for island in self.islands
                        ]
                    for l in concurrent.futures.as_completed(results):
                        processed_isl.append(l.result())
                    processed_isl.sort(key=lambda x: x.name)
                    self.islands = processed_isl

            return wrapper

        @paralelizando
        def find_islands_monotonic(island: Island) -> Island:
            island.rest_of_picture_f3 = folders.load_img_hdf5(
                f"/{self.name}/{island.name}", "rest_of_picture_f3"
            )
            if np.sum(island.rest_of_picture_f3) > 0:
                island.zigzags.find_monotonic(
                    island.rest_of_picture_f3,
                    self.base_frame,
                    self.path_radius_internal,
                    self.void_max,
                )
            return island

        @paralelizando
        def make_island_graph(island: Island) -> Island:

            # island.rest_of_picture_f3 = folders.load_img_hdf5(
            #     f"/{self.name}/{island.name}", "rest_F3"
            # )
            if np.sum(island.rest_of_picture_f3) > 0:
                island.zigzags_graph, island.zigzags_mst, island.pos_zigzag_nodes = (
                    island.zigzags.make_graph(
                        island.bridges.zigzag_bridges, self.base_frame
                    )
                )

            return island

        @paralelizando
        def get_island_linked_zigzags(island: Island) -> Island:
            # rest_of_picture_f3 = folders.load_img_hdf5(
            #     f"/{self.name}/{island.name}", "rest_F3"
            # )
            if np.sum(island.rest_of_picture_f3) > 0:
                for zb in island.bridges.zigzag_bridges:
                    zb.get_linked_zigzags(island.zigzags.regions)
            return island

        folders.load_islands_hdf5(self)
        for island in self.islands:
            folders.load_bridges_hdf5(self.name, island)
        for isl in self.islands:
            isl.zigzags = ZigZagRegions()
            folders.create_new_hdf5_group(f"/{self.name}/{isl.name}/zigzags")
        with Timer("Encontrando areas monotonicas"):
            find_islands_monotonic()
        with Timer("Criando os grafos de regiões"):
            make_island_graph()
        with Timer("Conectando regiões de zigzag"):
            get_island_linked_zigzags()

        with Timer("salvando imagens das regiões"):
            for isl in self.islands:
                if np.sum(isl.rest_of_picture_f3) > 0:
                    folders.create_new_hdf5_group(f"/{self.name}/{isl.name}/zigzags")
                    folders.save_props_hdf5(
                        f"/{self.name}/{isl.name}/zigzags",
                        isl.__dict__,
                    )
                    for reg in isl.zigzags.regions:
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/zigzags",
                            f"zz_{reg.name:03d}",
                            reg.img,
                        )
        return


def divide_islands(folders: Paths):
    save_file = folders.load_hdf5_file(folders.save_file_name)
    layer_names = list(save_file.keys())
    for ln in layer_names:
        layer_group = save_file[ln]
        img = layer_group["original_img"]
        separated_imgs, labels, num = it.divide_by_connected(img)
        layer_group.attrs["islands_number"] = num
        islands = []
        for i, si in enumerate(separated_imgs):
            if not (f"I_{i:03d}" in layer_group.keys()):
                island_group = layer_group.create_group(f"I_{i:03d}")
                folders.save_img_hdf5(island_group.name, "img", si)
                island_group.attrs["name"] = f"I_{i:03d}"
                island_group.attrs["img_name"] = "img"
                islands.append(Island(f"I_{i:03d}", "img"))
        layer_group.attrs["islands"] = [x.name for x in islands]
    save_file.attrs["Fase 0"] = "OK"
    save_file.close()
    return
