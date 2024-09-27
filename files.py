from __future__ import annotations
from email.headerregistry import Group
from typing import TYPE_CHECKING
from networkx import bridges
from components.bottleneck import Bridge, BridgeRegions
from components.offset import Loop, OffsetRegions, Region
from components.thin_walls import ThinWallRegions, ThinWall
from components.zigzag import ZigZag, ZigZagRegions
from components.layer import Layer, Island
import os, shutil
import subprocess
from typing import List
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import h5py

if TYPE_CHECKING:
    from typing import List


class System_Paths:
    """Mantem a organizaçao dos caminhos dentro da pasta do programa para evitar carregar coisas que ele nao processa"""

    def __init__(self, home):
        self.home = home
        self.input = self.home + "/input"
        self.output = self.home + "/output"
        self.slicer = self.home + "/slicing-with-images"
        self.sliced = self.home + "/input/sliced"
        self.layers = []
        self.selected = ""
        self.save_file_name = ""

    def create_hdf5_file(self, name):
        os.chdir(self.output)
        save_file = h5py.File(f"{name}.hdf5", "a")
        save_file_name = save_file.name
        os.chdir(self.home)
        self.save_file_name = f"{name}.hdf5"
        return save_file_name

    def create_new_hdf5_group(self, path) -> str:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        if not (f.get(path)):
            group = f.create_group(path)
            group_name = group.name
        else:
            group_name = f.get(path).name
        f.close()
        os.chdir(self.home)
        return group_name

    def load_hdf5_file(self, save_file_name) -> h5py.File:
        os.chdir(self.output)
        save_file = h5py.File(save_file_name, "a")
        os.chdir(self.home)
        return save_file

    def create_layers_2d(self, path_input, dpi, layer_height, file_name: str):
        """No caso de um arquivo 2D cria um objeto Layer apenas (usado mais para testes mesmo)"""
        layer = Layer()
        img = layer.make_input_img(0, path_input, dpi, 0, layer_height, 1, self)
        save_name = file_name.replace(".pgm", "")
        save_name = save_name.replace("/", "")
        save_file = self.create_hdf5_file(save_name)
        layer_group_nome = self.create_new_hdf5_group(f"/L_000")
        props_layer = {
            "name": "L_000",
            "dpi": dpi,
            "base_frame": img.shape,
            "n_camadas": 1,
            "layer_height": layer_height,
            "odd_layer": 0,
            "img_name": "original_img",
        }
        self.save_props_hdf5(layer_group_nome, props_layer)
        self.save_img_hdf5(layer_group_nome, "original_img", img)
        return

    def create_layers_3d(self, path_input, dpi, layer_height, file_name):
        """No caso de um arquivo 3D o programa chama o algoritmo SliceWithImages do Prof Minetto e cria um objeto Layer por camada"""
        os.chdir(self.slicer)
        subprocess.run(["./run-single-model.sh", path_input, str(dpi)])
        camadas_imgs_names = self.list(origins=1)
        n_camadas = len(camadas_imgs_names)
        save_name = file_name.replace(".stl", "")
        save_name = save_name.replace(".STL", "")
        save_name = save_name.replace("stl_models", "")
        save_name = save_name.replace("/", "")
        save_file = self.create_hdf5_file(save_name)
        for i, name in enumerate(camadas_imgs_names):
            layer_group = self.create_new_hdf5_group(f"/L_{i:03d}")
            odd_layer = i % 2
            layer = Layer()
            os.chdir(self.sliced)
            img = layer.make_input_img(
                i, name, dpi, odd_layer, layer_height, n_camadas, self
            )
            os.chdir(self.home)
            layer_dataset = self.save_img_hdf5(layer_group, "original_img", img)
            props_layer = {
                "name": f"L_{i:03d}",
                "dpi": dpi,
                "base_frame": img.shape,
                "n_camadas": n_camadas,
                "layer_height": layer_height,
                "odd_layer": odd_layer,
                "img_name": "original_img",
            }
            self.save_props_hdf5(layer_group, props_layer)
        os.chdir(self.home)
        return

    def list(self, origins=0, layers=0, isles=0):
        list = []
        if origins == 1:
            os.chdir(self.sliced)
            list = sorted([x for x in os.listdir() if x.endswith(".pgm")])
        if layers == 1:
            os.chdir(self.output)
            list = sorted([x for x in os.listdir() if x.endswith(".json")])
        if isles == 1:
            os.chdir(self.output)
            list = sorted([x for x in os.listdir() if x.endswith("I", 5)])
        os.chdir(self.home)
        return list

    def load_bridges_hdf5(self, layer_name, island: Island) -> None:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        bridges_group = f.get(f"/{layer_name}/{island.name}/bridges")
        if bridges_group:
            island.bridges = BridgeRegions()
            island.bridges.offset_bridges = []
            island.bridges.zigzag_bridges = []
            island.bridges.cross_over_bridges = []
            for i_key, i_item in bridges_group.items():
                if isinstance(i_item, h5py.Group):
                    if i_key == "cross_over_bridges":
                        b_group = bridges_group.get(i_key)
                        cob = island.bridges.cross_over_bridges
                    if i_key == "zigzag_bridges":
                        b_group = bridges_group.get(i_key)
                        cob = island.bridges.zigzag_bridges
                    if i_key == "offset_bridges":
                        b_group = bridges_group.get(i_key)
                        cob = island.bridges.offset_bridges
                    for j_key, j_item in b_group.items():
                        b_region_group = b_group.get(j_key)
                        b_region_group_props = dict(b_region_group.attrs)
                        cob.append(
                            Bridge(
                                j_key,
                                [],
                                [],
                                [],
                                0,
                                [],
                                pontos_extremos=b_region_group_props["pontos_extremos"],
                                linked_offset_regions=b_region_group_props[
                                    "linked_offset_regions"
                                ],
                                linked_zigzag_regions=b_region_group_props[
                                    "linked_zigzag_regions"
                                ],
                            )
                        )
                        for k_key, k_item in b_region_group.items():
                            setattr(cob[-1], k_key, np.array(k_item))
                        setattr(
                            cob[-1],
                            "reference_points",
                            b_region_group_props["reference_points"],
                        )
                        setattr(
                            cob[-1],
                            "reference_points_b",
                            b_region_group_props["reference_points_b"],
                        )
        f.close()
        os.chdir(self.home)
        return

    def load_graph_hdf5(self, path, name) -> np.array:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        try:
            local = f.get(path)
            adj_matrix = local[name][:]
        except:
            adj_matrix = []
        finally:
            f.close()
            os.chdir(self.home)
            G_loaded = nx.from_numpy_array(adj_matrix)
        return G_loaded

    def load_img_hdf5(self, path, name) -> np.array:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        try:
            local = f.get(path)
            img = np.array(local[name])
        except:
            img = []
        finally:
            f.close()
            os.chdir(self.home)
        return img

    def load_islands_hdf5(self, layer: Layer) -> None:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        layer_group = f.get(layer.name)
        layer.islands = []
        for l_key, l_item in layer_group.items():
            if isinstance(l_item, h5py.Group):
                island_group = layer_group.get(l_key)
                layer.islands.append(Island(**island_group.attrs))
                for i_key, i_item in island_group.items():
                    if isinstance(i_item, h5py.Dataset):
                        setattr(layer.islands[-1], i_key, np.array(i_item))
        f.close()
        os.chdir(self.home)
        return

    def load_layers_hdf5(self) -> List[Layer]:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        layers = []
        for key, item in f.items():
            # isinstance(item, h5py.Group)
            layers.append(Layer(**dict(f[key].attrs)))
            layers[-1].original_img = np.array(f.get(f"/{key}/original_img"))
        f.close()
        os.chdir(self.home)
        return layers

    def load_offsets_hdf5(self, layer_name, island: Island) -> None:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        group = f.get(f"/{layer_name}/{island.name}/offsets")
        island.offsets = OffsetRegions()
        try:
            island.offsets.all_valid_loops = np.array(group["all_loops"])
            for region_name in list(group.keys()):
                if region_name.startswith("Reg"):
                    island.offsets.regions.append(
                        Region(region_name, np.array(group[region_name + "/img"]), [])
                    )
                    region_group = group.get(region_name)
                    for i_key, i_item in region_group.items():
                        if isinstance(i_item, h5py.Group):
                            loops_group = region_group.get(i_key)
                            for loop_name in list(loops_group.keys()):
                                island.offsets.regions[-1].loops.append(
                                    Loop(
                                        loops_group[loop_name].attrs["name"],
                                        np.array(loops_group[loop_name]),
                                        loops_group[loop_name].attrs["offset_level"],
                                        [],
                                        **loops_group[loop_name].attrs,
                                    )
                                )
                        elif isinstance(i_item, h5py.Dataset):
                            setattr(island.offsets.regions[-1], i_key, np.array(i_item))
        except:
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return

    def load_thin_walls_hdf5(self, layer_name: str, island: Island) -> None:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        island_group = f.get(f"/{layer_name}/{island.name}")
        twr_group = island_group.get("thin_walls")
        if twr_group:
            island.thin_walls = ThinWallRegions()
            island.thin_walls.regions = []
            for i_key, i_item in twr_group.items():
                if isinstance(i_item, h5py.Group):
                    tw_group = twr_group.get(i_key)
                    island.thin_walls.regions.append(ThinWall(**tw_group.attrs))
                    setattr(island.thin_walls.regions[-1], "name", i_key)
                    for i_key, i_item in tw_group.items():
                        setattr(island.thin_walls.regions[-1], i_key, np.array(i_item))
                if isinstance(i_item, h5py.Dataset):
                    setattr(island.thin_walls, i_key, np.array(i_item))
        f.close()
        os.chdir(self.home)
        return

    def load_zigzags_hdf5(self, layer_name: str, island: Island) -> None:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        island_group = f.get(f"/{layer_name}/{island.name}")
        zzr_group = island_group.get("zigzags")
        if zzr_group:
            island.zigzags = ZigZagRegions()
            cob = island.zigzags.regions
            for i_key, i_item in zzr_group.items():
                region_group = zzr_group.get(i_key)
                if isinstance(i_item, h5py.Group):
                    cob.append(ZigZag(i_key, []))
                    for k_key, k_item in region_group.items():
                        setattr(cob[-1], k_key, np.array(k_item))
                else:
                    setattr(island.zigzags, i_key, np.array(i_item))
        f.close()
        os.chdir(self.home)
        return

    def save_bridges_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            if np.sum(isl.rest_of_picture_f3) > 0:
                self.save_img_hdf5(
                    f"/{layer_name}/{isl.name}",
                    "rest_of_picture_f3",
                    isl.rest_of_picture_f3,
                )
                self.save_img_hdf5(
                    f"/{layer_name}/{isl.name}/bridges",
                    "all_bridges",
                    isl.bridges.all_bridges,
                )
                self.create_new_hdf5_group(
                    f"/{layer_name}/{isl.name}/bridges/offset_bridges"
                )
                self.create_new_hdf5_group(
                    f"/{layer_name}/{isl.name}/bridges/zigzag_bridges"
                )
                self.create_new_hdf5_group(
                    f"/{layer_name}/{isl.name}/bridges/cross_over_bridges"
                )
                for reg in isl.bridges.offset_bridges:
                    path_region = f"/{layer_name}/{isl.name}/bridges/offset_bridges/OB_{reg.name:03d}"
                    self.create_new_hdf5_group(path_region)
                    self.save_img_hdf5(path_region, f"img", reg.img)
                    self.save_img_hdf5(path_region, f"origin", reg.origin)
                    self.save_props_hdf5(path_region, reg.__dict__)
                for reg in isl.bridges.zigzag_bridges:
                    path_region = f"/{layer_name}/{isl.name}/bridges/zigzag_bridges/ZB_{reg.name:03d}"
                    self.create_new_hdf5_group(path_region)
                    self.save_img_hdf5(path_region, f"img", reg.img)
                    self.save_img_hdf5(path_region, f"origin", reg.origin)
                    self.save_img_hdf5(path_region, f"contorno", reg.contorno)
                    self.save_props_hdf5(path_region, reg.__dict__)
                for reg in isl.bridges.cross_over_bridges:
                    path_region = f"/{layer_name}/{isl.name}/bridges/cross_over_bridges/CB_{reg.name:03d}"
                    self.create_new_hdf5_group(path_region)
                    self.save_img_hdf5(path_region, f"img", reg.img)
                    self.save_img_hdf5(path_region, f"origin", reg.origin)
                    self.save_img_hdf5(path_region, f"contorno", reg.contorno)
                    self.save_props_hdf5(path_region, reg.__dict__)
        return

    def save_bridges_routes_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            if np.sum(isl.rest_of_picture_f3) > 0:
                for reg in isl.bridges.offset_bridges:
                    region_path = (
                        f"/{layer_name}/{isl.name}/bridges/offset_bridges/{reg.name}"
                    )
                    self.save_img_hdf5(region_path, "route", reg.route.astype(bool))
                    self.save_img_hdf5(region_path, f"trail", reg.trail.astype(bool))
                    self.save_props_hdf5(region_path, reg.__dict__)
                for reg in isl.bridges.zigzag_bridges:
                    region_path = (
                        f"/{layer_name}/{isl.name}/bridges/zigzag_bridges/{reg.name}"
                    )
                    self.save_img_hdf5(region_path, f"route", reg.route.astype(bool))
                    self.save_img_hdf5(region_path, f"trail", reg.trail.astype(bool))
                    self.save_props_hdf5(region_path, reg.__dict__)
                for reg in isl.bridges.cross_over_bridges:
                    region_path = f"/{layer_name}/{isl.name}/bridges/cross_over_bridges/{reg.name}"
                    self.save_img_hdf5(region_path, f"route", reg.route.astype(bool))
                    self.save_img_hdf5(
                        region_path, f"route_b", reg.route_b.astype(bool)
                    )
                    self.save_img_hdf5(region_path, f"trail", reg.trail.astype(bool))
                    self.save_img_hdf5(
                        region_path, f"trail_b", reg.trail_b.astype(bool)
                    )
                    self.save_props_hdf5(region_path, reg.__dict__)
        return

    def save_external_routes_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            element_path = f"/{layer_name}/{isl.name}/external_tree_route"
            self.create_new_hdf5_group(element_path)
            self.save_props_hdf5(element_path, isl.external_tree_route.__dict__)
            self.save_seq_hdf5(
                element_path, "sequence", isl.external_tree_route.sequence
            )
            self.save_props_hdf5(f"/{layer_name}/{isl.name}", isl.__dict__)

    def save_img_hdf5(self, path, name, img, type="bool"):
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        try:
            local = f.get(path)
            if local.get(name):
                local[name][...] = img.astype(bool)
            else:
                local.create_dataset(name, compression="gzip", data=img, dtype=type)
        except:
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return
    
    def delete_img_hdf5(self, path):
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        try:
            del f[path]
        except:
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return

    def save_graph_hdf5(self, path, name, G):
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        adj_matrix = nx.to_numpy_array(G)
        try:
            local = f.get(path)
            if local.get(name):
                local[name][...] = adj_matrix
            else:
                local.create_dataset(name, data=adj_matrix)
        except:
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return

    def save_props_hdf5(self, path, dict) -> None:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        local = f.get(path)
        for key, value in dict.items():
            try:
                local.attrs[key] = value
            except:
                pass
        f.close()
        os.chdir(self.home)
        return

    def save_seq_hdf5(self, path, name, seq):
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        try:
            local = f.get(path)
            if local.get(name):
                local[name] = seq
            else:
                local.create_dataset(name, data=seq)
        except:
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return

    def save_zigzags_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            base_path = f"/{layer_name}/{isl.name}/zigzags"
            if np.sum(isl.rest_of_picture_f3) > 0:
                self.create_new_hdf5_group(base_path)
                self.save_props_hdf5(base_path, isl.__dict__)
                for reg in isl.zigzags.regions:
                    if isinstance(reg.name, int):
                        reg.name = f"ZZ_{reg.name:03d}"
                    self.create_new_hdf5_group(f"{base_path}/{reg.name}")
                    self.save_img_hdf5(
                        f"{base_path}/{reg.name}", "img", reg.img
                    )
                    if len(reg.route) > 0:
                        self.save_img_hdf5(
                            f"{base_path}/{reg.name}", "route", reg.route
                        )
                    else:
                        self.delete_img_hdf5(f"{base_path}/{reg.name}/route")
                    if len(reg.trail) > 0:
                        self.save_img_hdf5(
                            f"{base_path}/{reg.name}", "trail", reg.trail
                        )
                    else:
                        self.delete_img_hdf5(f"{base_path}/{reg.name}/trail")
        return
