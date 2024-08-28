from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from files import Paths
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
    """Cada região que precisa ser feita com menos de duas trilhas antes de fazer os contornos"""

    def __init__(
        self,
        name,
        img,
        origin,
        trunk,
        n_paths,
        origin_marks,
        elementos_contorno=None,
        pontos_extremos=None,
    ):
        if elementos_contorno is None:
            elementos_contorno = []
        if pontos_extremos is None:
            pontos_extremos = []
        self.name = name
        self.img = img
        self.origin = origin
        self.destiny = 0
        self.n_paths = n_paths
        self.origin_mark = origin_marks
        self.trunk = trunk
        self.contorno = elementos_contorno
        self.pontos_extremos = pontos_extremos
        # self.origin_coords = []
        # self.destiny_coords = []
        # self.route = []
        # self.trail = []
        # self.center = []
        # self.interruption_points = []
        # self.reference_points = []
        return


class ThinWallRegions:
    """O grupo de regiões chamadas de Thin Walls, gruarda também a configuração geral para essas regiões"""

    def __init__(self):
        self.regions: List[ThinWall]
        self.medial_transform: str
        self.all_thin_walls: str
        self.all_thin_origins: str
        # self.routes = []
        return

    def make_thin_walls(
        self,
        layer_name,
        island_name,
        island_img: np.ndarray,
        base_frame,
        path_radius,
        mask
    ):

        def close_contour(trunk, i):
            t_ends = pt.img_to_points(sk.find_tips(trunk.astype(bool)))
            if len(t_ends) < 2 or len(pt.img_to_points(trunk)) < path_radius * 2:
                return []
            else:
                trunk_chain = pt.invert_x_y(
                    ptht.make_a_chain_open_segment(trunk.astype(bool), t_ends)
                )
                n_trilhas = trunk / (2 * path_radius)
                n_trilhas_max = np.min(n_trilhas[np.nonzero(n_trilhas)])
                bridge_origin = np.logical_and(trunk != 0, trunk < max_width)
                if np.sum(bridge_origin) > 0:
                    region = ThinWall(0, "", [], [], 0, 0, [], [])
                    ends = pt.img_to_points(sk.find_tips(bridge_origin))
                    indices = [trunk_chain.index(x) for x in ends]
                    ends = [trunk_chain[np.min(indices)], trunk_chain[np.max(indices)]]
                    origin_chain = pt.invert_x_y(
                        ptht.make_a_chain_open_segment(trunk.astype(bool), ends)
                    )
                    new_origin = origin_chain.copy()
                    count_up = 0
                    count_down = -1
                    start_flag = 0
                    end_flag = 0
                    while not (start_flag and end_flag):
                        current_pt_1 = origin_chain[count_up]
                        current_pt_2 = origin_chain[count_down]
                        if current_pt_1 in ends:
                            start_flag = 1
                        else:
                            new_origin.remove(current_pt_1)
                            count_up += 1
                        if current_pt_2 in ends:
                            end_flag = 1
                        else:
                            new_origin.remove(current_pt_2)
                            count_down -= 1
                    bridge_origin = it.points_to_img(
                        new_origin, np.zeros_like(island_img)
                    )
                    try:
                        bridge_img, elementos_contorno, contorno, pontos_extremos = (
                            bottleneck.close_bridge_contour_v2(
                                bridge_origin,
                                base_frame,
                                max_width,
                                island_img,
                                mask,
                                path_radius,
                                sem_galhos,
                            )
                        )
                        if np.sum(bridge_img) > 0:
                            y_mark = np.where(bridge_origin)[1][
                                np.round(len(np.where(bridge_origin)))
                            ]
                            x_mark = np.where(bridge_origin)[0][
                                np.round(len(np.where(bridge_origin)))
                            ]
                            origin_mark = [y_mark, x_mark, str(n_trilhas_max)]
                            tw_group_path = f"{layer_name}/{island_name}/thin_walls/TW_{i:03d}"
                            new_groups.append([tw_group_path])
                            # tw_group_path = folders.create_new_hdf5_group(f"{layer_name}/{island_name}/thin_walls/TW_{i:03d}")
                            [linha1, linha2, linhatopo, linhabaixo] = elementos_contorno
                            # trunk = f"{layer_name}/{island_name}/thin_walls/TW_{i:03d}/trunk"
                            region = ThinWall(
                                f"{layer_name}/{island_name}/thin_walls/TW_{i:03d}",
                                f"{layer_name}/{island_name}/thin_walls/TW_{i:03d}/img",
                                trunk,
                                trunk,
                                n_trilhas_max,
                                origin_mark,
                                elementos_contorno,
                                pontos_extremos,
                            )
                            new_tw_imgs.append([tw_group_path, "img", bridge_img])
                            # folders.save_img_hdf5(tw_group_path, "img", bridge_img)
                            new_linha1s.append([tw_group_path, "linha1", linha1])
                            # folders.save_img_hdf5(tw_group_path, "linha1", linha1)
                            new_linha2s.append([tw_group_path, "linha2", linha2])
                            # folders.save_img_hdf5(tw_group_path, "linha2", linha2)
                            new_linhatopos.append([tw_group_path, "linhatopo", linhatopo])
                            # folders.save_img_hdf5(tw_group_path, "linhatopo", linhatopo)
                            new_linhabaixos.append([tw_group_path, "linhabaixo", linhabaixo])
                            # folders.save_img_hdf5(tw_group_path, "linhabaixo", linhabaixo)
                            new_origins.append([tw_group_path, "origin_img", trunk])
                            # folders.save_img_hdf5(tw_group_path, "origin_img", trunk)
                            print("OK: fechou contorno")
                    except Exception:
                        print("\033[3#m" + "Erro: nao fechou contorno" + "\033[0m")
                return region

        # Criando o MAT
        new_groups = []
        new_tw_imgs = []
        new_medial_transforms = []
        new_linha1s = []
        new_linha2s = []
        new_linhatopos = []
        new_linhabaixos = []
        new_origins = []
        sem_galhos, sem_galhos_dist, trunks = sk.create_prune_divide_skel(
            island_img.astype(np.uint8), path_radius
        )
        # TODO:Achar o melhor size para o serviço
        new_medial_transforms.append([f"{layer_name}/{island_name}","medial_transform",sem_galhos * sem_galhos_dist])
        # folders.save_img_hdf5(f"{layer_name}/{island_name}","medial_transform",sem_galhos * sem_galhos_dist,type="int8")

        # Dividindo partes do esquelet que podem ser paredes finas
        trunks = [pt.contour_to_list([x]) for x in trunks]
        trunks = [it.points_to_img(x, np.zeros(base_frame)) for x in trunks]
        all_thin_walls = np.zeros(base_frame)
        all_origins = np.zeros(base_frame)
        divided_by_large_areas = []
        for trunk in trunks:
            trunk_dist = trunk * sem_galhos_dist
            no_large_parts = np.logical_and(
                trunk_dist < 3 * path_radius, trunk_dist > 0
            )
            divided_segs, _, num = it.divide_by_connected(no_large_parts)
            if num > 1:
                for seg in divided_segs:
                    if len(pt.img_to_points(seg)) > path_radius:
                        divided_by_large_areas.append(seg)
            elif num == 0:
                pass
            else:
                divided_by_large_areas.append(no_large_parts)
        trunks = divided_by_large_areas

        # Transformando cada galho em uma região fechada
        max_width = 2 * path_radius
        counter = 0

        processed_trunks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [
                executor.submit(close_contour, trunk, i)
                for i, trunk in enumerate(trunks)
            ]
            for l in concurrent.futures.as_completed(results):
                processed_trunks.append(l.result())
        processed_trunks = list(filter(lambda x: x != [], processed_trunks))
        processed_trunks = list(filter(lambda x: x.img != [], processed_trunks))
        processed_trunks.sort(key=lambda x: x.name)
        self.regions = processed_trunks
        # Produzindo as imagens de resumoo
        all_thin_walls = np.zeros_like(island_img)
        all_origins = np.zeros_like(island_img)
        for tw in new_tw_imgs:
            # reg_img = folders.load_img_hdf5(reg.name, "img")
            reg_img = tw[2]
            all_thin_walls = np.logical_or(all_thin_walls, reg_img)
        for tw in new_origins:
            # reg_origin_img = folders.load_img_hdf5(reg.name, "origin_img")
            reg_origin_img = tw[2]
            all_origins = np.logical_or(all_origins, reg_origin_img)
        # group_tw_name = folders.create_new_hdf5_group(f"/{layer_name}/{island_name}/thin_walls")
        # folders.save_img_hdf5(group_tw_name, "all_tw", all_thin_walls)
        # folders.save_img_hdf5(group_tw_name, "all_tw_origins", all_origins)
        img_packs = {
            "groups" : new_groups,
            "tw_img" : new_tw_imgs,
            "mat" : new_medial_transforms,
            "l1" : new_linha1s,
            "l2" : new_linha2s,
            "lt" : new_linhatopos,
            "lb" : new_linhabaixos,
            "origins" : new_origins,
            "all_tw": [f"/{layer_name}/{island_name}/thin_walls", "all_thin_walls", all_thin_walls],
            "all_tw_origins": [f"/{layer_name}/{island_name}/thin_walls", "all_origins", all_origins]
        }
        return img_packs

    def apply_thin_walls(self, folders: Paths, original, base_frame):
        rest_of_picture_f1 = np.zeros(base_frame)
        rest_of_picture_f1 = np.logical_or(original, rest_of_picture_f1)
        for region in self.regions:
            region_img = folders.load_img_hdf5(region.name, "img")
            # region_img = folders.load_img(region.img)
            rest_of_picture_f1 = np.logical_and(
                rest_of_picture_f1, np.logical_not(region_img)
            )
        return rest_of_picture_f1.astype(np.uint8)
