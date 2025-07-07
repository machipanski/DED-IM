from __future__ import annotations
from bdb import effective
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from components.files import System_Paths
    from typing import List
import numpy as np
from components import images_tools as it
from components import morphology_tools as mt
from components import skeleton as sk
from components import bottleneck
from components import points_tools as pt
from components import path_tools as ptht
import concurrent.futures


class ThinWall:
    """Cada região que precisa ser feita com menos
    de duas trilhas antes de fazer os contours"""

    def __init__(self, *args, **kwargs):
        self.name: str
        self.img: np.ndarray
        self.origin: str
        self.destiny = 0
        self.n_paths = int
        self.origin_mark = []
        self.trunk = np.ndarray
        self.contour = np.ndarray
        self.extreme_points = []
        self.contour_elements = []
        self.extreme_points = []
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        if args:
            self.name = args[0]
            self.img = args[1]
            self.origin = args[2]
            self.trunk = args[3]
            self.n_paths = args[4]
            self.origin_mark = args[5]
            self.contour_elements = args[6]
            self.extreme_points = args[7]
            self.destiny = 0
        return

    def make_route(self, path_radius, sobrep):
        effective_origin = np.logical_and(self.origin, self.img)
        reduced_extremes_origin, _, _ = sk.prune(
            effective_origin, int(path_radius * 2 * (1 - sobrep / 100))
        )
        self.route = reduced_extremes_origin.astype(np.uint8)
        self.trail = mt.dilation(self.route, kernel_size=path_radius)
        extreme_points = np.nonzero(mt.hitmiss_ends_v2(self.route))
        extreme_points = pt.x_y_para_pontos(extreme_points)
        self.interruption_points = extreme_points
        return


class ThinWallRegions:
    """O grupo de regiões chamadas de Thin Walls, gruarda também a
    configuração geral para essas regiões"""

    def __init__(self):
        self.regions: List[ThinWall]
        self.medial_transform = []
        self.all_thin_walls = []
        self.all_origins = []
        return

    def make_thin_walls(
        self,
        layer_name,
        island_name,
        island_img: np.ndarray,
        base_frame,
        path_radius,
        mask,
        mm_per_pxl,
    ):

        def close_contour(reduced_continuous_origin, i):
            max_width = 2
            try:
                bridge_img, contour_elements, contour, extreme_points = (
                    bottleneck.close_bridge_contour(
                        reduced_continuous_origin[0],
                        base_frame,
                        max_width,
                        island_img,
                        path_radius,
                        pt.img_to_points(
                            mt.hitmiss_ends_v2(reduced_continuous_origin[0])
                        )[0],
                    )
                )
                if np.sum(bridge_img) > 0:
                    y_mark = np.where(reduced_continuous_origin[0])[1][
                        np.round(len(np.where(reduced_continuous_origin[0])))
                    ]
                    x_mark = np.where(reduced_continuous_origin[0])[0][
                        np.round(len(np.where(reduced_continuous_origin[0])))
                    ]
                    origin_mark = [y_mark, x_mark, str(n_trilhas_max)]
                    region = ThinWall(
                        f"TW_{i:03d}",
                        bridge_img,
                        reduced_continuous_origin[0],
                        reduced_continuous_origin[0],
                        n_trilhas_max,
                        origin_mark,
                        contour_elements,
                        extreme_points,
                    )
                    print("OK: closed contour")
                else:
                    region = []
            except Exception:
                print("\033[3#m" + "Error: didn´t closed one contour" + "\033[0m")
                # region = ThinWall([], [], [], [], 0, [], [], [])
                region = []
            return region

        new_medial_transforms = []
        sem_galhos, sem_galhos_dist, list_trunks = sk.create_prune_divide_skel(
            island_img.astype(np.uint8), 2 * path_radius
        )
        new_medial_transforms.append(
            [
                f"{layer_name}/{island_name}",
                "medial_transform",
                sem_galhos * sem_galhos_dist,
            ]
        )
        max_width = 2
        # trunks = [pt.contour_to_list([x]) for x in trunks]
        trunks_imgs = [
            it.points_to_img(x, np.zeros_like(sem_galhos)) for x in list_trunks
        ]
        norm_dist_map = sem_galhos_dist / path_radius
        norm_trunks = [trunk * norm_dist_map for trunk in trunks_imgs]
        n_trilhas_max = [(np.unique(trunk))[1] for trunk in norm_trunks]
        origin_candidates = [
            norm_trunks[i] for i, x in enumerate(n_trilhas_max) if x <= max_width
        ]
        reduced_origins = [
            bottleneck.reduce_origin(x, max_width, norm_dist_map)
            for x in origin_candidates
        ]
        processed_trunks = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [
                executor.submit(close_contour, origin_candidate, i)
                for i, origin_candidate in enumerate(reduced_origins)
            ]
            for l in concurrent.futures.as_completed(results):
                processed_trunks.append(l.result())
        processed_trunks = list(filter(lambda x: x != [], processed_trunks))
        processed_trunks.sort(key=lambda x: x.name)

        self.regions = [x for x in processed_trunks]
        self.all_thin_walls = np.zeros_like(island_img)
        self.all_origins = np.zeros_like(island_img)
        for tw in self.regions:
            self.all_thin_walls = np.logical_or(self.all_thin_walls, tw.img)
            self.all_origins = np.logical_or(self.all_origins, tw.origin)
        return

    def apply_thin_walls(self, folders: System_Paths, original, base_frame):
        rest_of_picture_f1 = np.zeros(base_frame)
        rest_of_picture_f1 = np.logical_or(original, rest_of_picture_f1)
        for region in self.regions:
            region_img = folders.load_img_hdf5(region.name, "img")
            rest_of_picture_f1 = np.logical_and(
                rest_of_picture_f1, np.logical_not(region_img)
            )
        return rest_of_picture_f1.astype(np.uint8)

    def make_routes_tw(self, path_radius, sobrep):
        """Chama a função make_route() para cada região"""
        for i in self.regions:
            i.make_route(path_radius, sobrep)
        return

    def check_thin_walls(
        self,
        layer_name,
        island_name,
        island_img: np.ndarray,
        base_frame,
        path_radius,
        mask,
        mm_per_pxl,
    ):
        filtered_tw = []
        for tw in self.regions:
            # img = np.logical_and(tw.img, island_img)
            # origin = np.logical_and(tw.origin, island_img)
            eroded_island_img = mt.erosion(island_img, kernel_size=2 * path_radius)
            _, eroded_island_border = mt.detect_contours(
                eroded_island_img,
                return_img=True,
                # only_external=True,
            )
            result_first_offset = it.fill_internal_area(
                mt.dilation(eroded_island_border, kernel_size=path_radius),
                island_img,
            )
            not_using_tw = np.logical_and(tw.img, result_first_offset)
            if 2 * np.sum(tw.img) / 3 > np.sum(not_using_tw):
                filtered_tw.append(tw)

        self.regions = filtered_tw
        self.all_thin_walls = np.zeros_like(island_img)
        self.all_origins = np.zeros_like(island_img)
        for tw in self.regions:
            self.all_thin_walls = np.logical_or(self.all_thin_walls, tw.img)
            self.all_origins = np.logical_or(self.all_origins, tw.origin)
        return
