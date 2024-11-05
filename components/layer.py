from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Dict, reveal_type
from components import images_tools as it, path_tools, points_tools
from components import morphology_tools as mt
from components.thin_walls import ThinWallRegions
from components.offset import OffsetRegions
from components.zigzag import ZigZagRegions
from components.bottleneck import BridgeRegions
from timer import Timer
import numpy as np
import concurrent.futures
from functools import wraps
from components import points_tools
from components.path_tools import Path

if TYPE_CHECKING:
    from files import System_Paths
    from components.zigzag import ZigZag
    from typing import List


class Island:
    def __init__(self, *args, **kwargs):
        self.name: str
        self.img = []
        self.thin_walls: ThinWallRegions
        self.offsets: OffsetRegions
        self.bridges: BridgeRegions
        self.zigzags: ZigZagRegions
        self.rest_of_picture_f1 = []
        self.rest_of_picture_f2 = []
        self.rest_of_picture_f3 = []
        self.comeco_ext = []
        self.comeco_int = []
        self.external_tree_route: List[Path] = []
        self.internal_tree_route: List[Path] = []
        self.thinwalls_tree_route: List[Path] = []
        self.island_route: List[Path] = []
        self.prohibited_areas = []
        if args:
            self.name = args[0]
            self.img = args[1]
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        return


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

    def close_final_path(self, folders: System_Paths):
        folders.load_islands_hdf5(self)
        for island in self.islands:
            folders.load_offsets_hdf5(self.name, island)
            folders.load_island_paths_hdf5(self.name, island)
            itr = island.internal_tree_route
            etr = island.external_tree_route
            twtr = island.thinwalls_tree_route
            with Timer("Encontrando ponto de união ext-int"):
                island.comeco_ext, island.comeco_int = (path_tools.connect_internal_external(island, self.path_radius_internal))
                etr.sequence = path_tools.set_first_pt_in_seq([list(x) for x in etr.sequence], island.comeco_ext)
                itr.sequence = path_tools.set_first_pt_in_seq([list(x) for x in itr.sequence], island.comeco_int)
            island_route_path_for_img = etr.sequence + itr.sequence + twtr.sequence
            island_island_route_img = it.points_to_img(island_route_path_for_img, np.zeros(self.base_frame))

            with Timer("Conectando ambas as partes"):
                internal_simpl = path_tools.simplifica_retas_master(itr.sequence, 0.0002, itr.saltos)
                external_simpl = path_tools.simplifica_retas_master(etr.sequence, 0.0002, etr.saltos)
                thinwalls_simpl = path_tools.simplifica_retas_master(twtr.sequence, 0.001, twtr.saltos)
            island_route_path = external_simpl + internal_simpl + thinwalls_simpl
            island_island_route_img = it.chain_to_lines(island_route_path, np.zeros(self.base_frame))
            if self.odd_layer == 1:
                print("layer rotacionada")
                etr.sequence = path_tools.rotate_path_odd_layer(external_simpl, self.base_frame)
                itr.sequence = path_tools.rotate_path_odd_layer(internal_simpl, self.base_frame)
                twtr.sequence = path_tools.rotate_path_odd_layer(thinwalls_simpl, self.base_frame)
                island_route_path = path_tools.rotate_path_odd_layer(island_route_path,self.base_frame)
                island_island_route_img = it.chain_to_lines(island_route_path, np.zeros([self.base_frame[1], self.base_frame[0]]))
            island_new_regions = [itr.regions + etr.regions + twtr.regions]
            island_saltos = [itr.saltos + etr.saltos + twtr.saltos]
            island.island_route = Path("island_route",island_route_path,island_new_regions,saltos=island_saltos,img=island_island_route_img,)

        with Timer("salvando imagens das rotas"):
            folders.save_final_routes_hdf5(self.name, self.islands)
        return

    def close_routes_external(self, folders: System_Paths):
        with Timer("Conectando rotas externas"):
            folders.load_islands_hdf5(self)
            for isl in self.islands:
                folders.load_offsets_hdf5(self.name, isl)
                folders.load_zigzags_hdf5(self.name, isl)
                folders.load_bridges_hdf5(self.name, isl)
                folders.load_thin_walls_hdf5(self.name, isl)
                with Timer("Conectando pontes de Offset"):
                    isl.external_tree_route = path_tools.connect_offset_bridges(
                        isl,
                        self.base_frame,
                        mt.make_mask(self, "3_4_ext"),
                        self.path_radius_external,
                    )
                with Timer("Conectando pontes de Crossover"):
                    isl.external_tree_route = path_tools.connect_cross_over_bridges(isl)
                isl.external_tree_route.get_img(self.base_frame)
        with Timer("salvando imagens das rotas"):
            folders.save_external_routes_hdf5(self.name, self.islands)
        return

    def close_routes_internal(self, folders: System_Paths):
        folders.load_islands_hdf5(self)
        for isl in self.islands:
            folders.load_bridges_hdf5(self.name, isl)
            folders.load_zigzags_hdf5(self.name, isl)
            folders.load_offsets_hdf5(self.name, isl)
            folders.load_thin_walls_hdf5(self.name, isl)
            # folders.load_island_paths_hdf5(self.name, isl)
            with Timer("Conectando zgzags vizinhos"):
                isl.internal_tree_route = path_tools.zigzag_imgs_to_path(
                    isl, mt.make_mask(self, "full_int"), self.path_radius_internal
                )
            with Timer("Conectando pontes de zigzag"):
                isl.internal_tree_route = path_tools.connect_zigzag_bridges(isl)
                isl.internal_tree_route.get_img(self.base_frame)

            # with Timer("casando começo int com final ext"):
            #     path_tools.set_first_pt_in_seq([list(x) for x in isl.internal_tree_route.sequence], list(isl.comeco_int))

        with Timer("salvando imagens das rotas"):
            folders.save_internal_routes_hdf5(self.name, self.islands)
        return

    def close_routes_thinwalls(self, folders: System_Paths):
        folders.load_islands_hdf5(self)
        for isl in self.islands:
            folders.load_thin_walls_hdf5(self.name, isl)
            with Timer("Convertendo paredes finas"):
                isl.thinwalls_tree_route = path_tools.connect_thin_walls(
                    isl, self.path_radius_external
                )
                isl.thinwalls_tree_route.get_img(self.base_frame)

        with Timer("salvando imagens das rotas"):
            folders.save_thinwall_final_routes_hdf5(self.name, self.islands)

    def create_layers(
        folders: System_Paths, path_input: str, file_name: str, dpi: int, layer_height
    ):

        def divide_islands(layer: Layer):
            img = layer.original_img
            separated_imgs, labels, num = it.divide_by_connected(img)
            layer.islands_number = num
            islands = []
            for i, si in enumerate(separated_imgs):
                islands.append(Island(f"I_{i:03d}", si))
            layer.islands = islands
            return

        list_layers = []
        with Timer("criando as camadas"):
            if file_name.endswith(".stl") or file_name.endswith(".STL"):
                hdf5_file_name = file_name.replace(".stl", "")
                hdf5_file_name = hdf5_file_name.replace(".STL", "")
                hdf5_file_name = hdf5_file_name.replace("stl_models", "")
                hdf5_file_name = hdf5_file_name.replace("/", "")
                list_layers = folders.call_slicer(
                    file_name, path_input, dpi, layer_height
                )
            else:
                hdf5_file_name = file_name.replace(".pgm", "")
                hdf5_file_name = hdf5_file_name.replace("/", "")
                layer = Layer()
                layer.make_input_img("L_000", path_input, dpi, 0, layer_height, 1)
                list_layers = [layer]
            for layer in list_layers:
                divide_islands(layer)
        with Timer("salvando as camadas"):
            folders.save_layers(hdf5_file_name, list_layers)
            folders.save_folders_structure(hdf5_file_name)
        return

    def make_input_img(
        self,
        name: str,
        img_path: str,
        dpi: int,
        odd_layer: bool,
        layer_height: float,
        n_camadas: int,
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
        self.original_img = img
        return

    def make_thin_walls(
        self,
        folders: System_Paths,
        nozzle_diam_external: float,
        nozzle_diam_internal: float,
    ) -> None:

        def make_islands_thinWalls(island: Island, mm_per_pxl) -> List[Island, dict]:
            island_img = folders.load_img_hdf5(f"/{self.name}/{island.name}", "img")
            island.thin_walls = ThinWallRegions()
            island.thin_walls.make_thin_walls(
                self.name,
                island.name,
                island_img,
                self.base_frame,
                self.path_radius_external,
                mt.make_mask(self, "full_ext"),
                mm_per_pxl,
            )
            return island

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
                    executor.submit(make_islands_thinWalls, island, self.mm_per_pxl)
                    for island in self.islands  # self.islands
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions.append(l.result())
            processed_regions.sort(key=lambda x: x.name)
            self.islands = processed_regions

        with Timer("Retirando Paredes finas da camada"):
            for island in self.islands:
                island.rest_of_picture_f1 = copy.deepcopy(self.original_img)
                for reg in island.thin_walls.regions:
                    island.rest_of_picture_f1 = it.image_subtract(
                        island.rest_of_picture_f1, reg.img
                    )
                # island_img = folders.load_img_hdf5(f"/{self.name}/{island.name}", "img")
                # if hasattr(island, "thin_walls"):
                #     self.rest_of_picture_f1 = island.thin_walls.apply_thin_walls(
                #         folders, island_img, self.base_frame
                #     )
                # else:
                #     self.rest_of_picture_f1 = island.img
                # folders.save_img_hdf5(
                #     f"{self.name}/{island.name}",
                #     "rest_of_picture_f1",
                #     self.rest_of_picture_f1,
                # )

        with Timer("salvando imagens das regiões"):
            folders.save_regs_thinwalls_hdf5(self.name, self.islands)
            folders.save_props_hdf5(f"/{self.name}", self.__dict__)
        return

    def make_thin_wall_routes(self, folders: System_Paths):
        with Timer("criando rotas TW"):
            folders.load_islands_hdf5(self)
            for isl in self.islands:
                folders.load_thin_walls_hdf5(self.name, isl)
                isl.thin_walls.make_routes_tw(self.path_radius_internal)
        with Timer("salvando imagens das rotas"):
            for isl in self.islands:
                for reg in isl.thin_walls.regions:
                    folders.save_img_hdf5(
                        f"/{self.name}/{isl.name}/thin_walls/{reg.name}",
                        "route",
                        reg.route.astype(bool),
                    )
                    folders.save_img_hdf5(
                        f"/{self.name}/{isl.name}/thin_walls/{reg.name}",
                        "trail",
                        reg.trail.astype(bool),
                    )
        return

    def make_offsets(
        self,
        folders: System_Paths,
        void_max: float,
        external_max: int,
        internal_max: int,
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
            folders.save_regs_offsets_hdf5(self.name, self.islands)
            folders.save_props_hdf5(f"/{self.name}", self.__dict__)
        return

    def make_offset_routes(self, amendment_size, folders: System_Paths):
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
        with Timer("salvando imagens das rotas"):
            for isl in self.islands:
                if np.sum(isl.rest_of_picture_f2) > 0:
                    for reg in isl.offsets.regions:
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/offsets/{reg.name}",
                            "route",
                            reg.route.astype(bool),
                        )
                        folders.save_img_hdf5(
                            f"/{self.name}/{isl.name}/offsets/{reg.name}",
                            "trail",
                            reg.trail.astype(bool),
                        )
        folders.save_props_hdf5(f"/{self.name}", self.__dict__)

    def make_bridges(
        self,
        n_max: float,
        nozzle_diam_internal: float,
        folders: System_Paths,
        n_camadas,
        sum_prohibited_areas,
    ):

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
                island.offsets_graph, island.offsets_mst = (
                    island.bridges.make_offset_bridges(
                        rest_of_picture_f2,
                        isl.offsets,
                        self.base_frame,
                        self.path_radius_external,
                        rest_of_picture_f2,
                        sum_prohibited_areas,
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
                    sum_prohibited_areas, island.offsets_mst
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
            # folders.create_new_hdf5_group(f"/{self.name}/{isl.name}/bridges")
        if self.name == "L_000":
            self.prohibited_areas = np.zeros_like(self.original_img)

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
        last_prohibited_areas = np.zeros_like(self.original_img)
        with Timer("Gerando áreas proibidas"):
            if not int(self.name[2:]) == n_camadas:
                for isl in self.islands:
                    for reg in isl.bridges.offset_bridges:
                        last_prohibited_areas = np.logical_or(
                            last_prohibited_areas, reg.img
                        )
                    for reg in isl.bridges.cross_over_bridges:
                        last_prohibited_areas = np.logical_or(
                            last_prohibited_areas, reg.img
                        )

        with Timer("salvando imagens das regiões"):
            folders.save_regs_bridges_hdf5(self.name, self.islands)
            folders.save_img_hdf5(
                f"/{self.name}", "prohibited_areas", self.prohibited_areas
            )
        return last_prohibited_areas

    def make_bridges_routes(self, folders: System_Paths):
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
        with Timer("salvando imagens das rotas"):
            folders.save_routes_bridges_hdf5(self.name, self.islands)
            folders.save_props_hdf5(f"/{self.name}", self.__dict__)

    def make_zigzags(self, folders: System_Paths):

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
            ideal_sum = np.sum(mt.make_mask(self, "full_int"))
            if np.sum(island.rest_of_picture_f3) > 0:
                island.zigzags.find_monotonic(
                    island.rest_of_picture_f3,
                    self.base_frame,
                    self.path_radius_internal,
                    ideal_sum,
                )
            return island

        folders.load_islands_hdf5(self)
        for island in self.islands:
            folders.load_bridges_hdf5(self.name, island)
        for isl in self.islands:
            isl.zigzags = ZigZagRegions()
        with Timer("Encontrando areas monotonicas"):
            find_islands_monotonic()
        with Timer("salvando imagens das regiões"):
            folders.save_regs_zigzags_hdf5(self.name, self.islands)
        return

    def make_zigzag_routes(self, folders: System_Paths):
        folders.load_islands_hdf5(self)
        for isl in self.islands:
            with Timer(f"criando as rotas de zigzag, camada:{self.name}"):
                folders.load_zigzags_hdf5(self.name, isl)
                isl.zigzags.make_routes_z(self.base_frame, self.path_radius_internal)

        with Timer("salvando imagens das rotas"):
            folders.save_regs_zigzags_hdf5(self.name, self.islands)
        return

    def connect_zigzags(self, folders: System_Paths):

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
        def make_island_graph(island: Island) -> Island:
            if np.sum(island.rest_of_picture_f3) > 0:
                island.zigzags.make_graph(
                    island.bridges.zigzag_bridges, self.base_frame
                )
            return island

        @paralelizando
        def get_island_linked_zigzags(island: Island) -> Island:
            if np.sum(island.rest_of_picture_f3) > 0:
                for zb in island.bridges.zigzag_bridges:
                    zb.get_linked_zigzags(island.zigzags.regions)
            return island

        for isl in self.islands:
            folders.load_bridges_hdf5(self.name, isl)
        with Timer("Criando os grafos de regiões"):
            make_island_graph()
        with Timer("Conectando regiões de zigzag"):
            get_island_linked_zigzags()

        for isl in self.islands:
            isl.zigzags.connect_island_zigzags(
                self.path_radius_internal,
                mt.make_mask(self, "full_int"),
                self.base_frame,
            )

        with Timer("salvando grafos"):
            zigzags_path = f"/{self.name}/{isl.name}/zigzags"
            for isl in self.islands:
                folders.save_graph_hdf5(
                    zigzags_path, f"zigzags_graph", isl.zigzags.zigzags_graph
                )
                folders.save_img_hdf5(
                    zigzags_path, f"all_zigzags", isl.zigzags.all_zigzags
                )
                folders.save_img_hdf5(
                    zigzags_path, f"macro_areas", isl.zigzags.macro_areas
                )
                for bridge in isl.bridges.zigzag_bridges:
                    folders.save_props_hdf5(
                        f"/{self.name}/{isl.name}/bridges/zigzag_bridges/{bridge.name}",
                        bridge.__dict__,
                    )
        return

    def internal_weaving(self, internal_weaving, folders: System_Paths):
        with Timer("gerando preenchimentos oscilatórios"):
            folders.load_islands_hdf5(self)
            for isl in self.islands:
                folders.load_offsets_hdf5(self.name, isl)
                folders.load_bridges_hdf5(self.name, isl)
                folders.load_zigzags_hdf5(self.name, isl)
                folders.load_thin_walls_hdf5(self.name, isl)
                isl.zigzags.macro_areas, isl.zigzags.all_zigzags = (
                    isl.zigzags.create_oscilatory_inner(
                        isl.zigzags.macro_areas,
                        self.original_img,
                        self.base_frame,
                        self.path_radius_internal,
                        mt.make_mask(self, "full_int"),
                        isl.zigzags.regions,
                        isl.bridges,
                        isl.offsets.regions,
                        isl.thin_walls.regions,
                        internal_weaving,
                    )
                )
        with Timer("salvando rotas"):
            for isl in self.islands:
                folders.save_img_hdf5(
                    f"/{self.name}/{isl.name}/zigzags",
                    f"all_zigzags",
                    isl.zigzags.all_zigzags,
                )
                folders.delete_item_hdf5(f"/{self.name}/{isl.name}/zigzags/macro_areas")
                folders.save_img_hdf5(
                    f"/{self.name}/{isl.name}/zigzags",
                    f"macro_areas",
                    isl.zigzags.macro_areas,
                )
        return
