from __future__ import annotations
from email.headerregistry import Group
from typing import TYPE_CHECKING
from components.bottleneck import Bridge, BridgeRegions
from components.offset import Loop, OffsetRegions, Region
from components.thin_walls import ThinWallRegions, ThinWall
from components.zigzag import ZigZag, ZigZagRegions
from components.layer import Layer, Island
from components.path_tools import Path
from cv2 import imread
from dataclasses import dataclass
from typing import List
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import h5py
import yaml

if TYPE_CHECKING:
    from typing import List


class System_Paths:
    """Mantem a organizaçao dos caminhos dentro da pasta do programa para evitar carregar coisas que ele nao processa"""

    def __init__(self, home: str):
        """Initializes the System_Paths object with the home directory."""
        self.home = home
        self.input = self.home + "/input"
        self.output = self.home + "/output"
        self.slicer = self.home + "/slicing-with-images"
        self.sliced = self.home + "/input/sliced"
        self.layers: List[Layer] = []
        self.selected = ""
        self.save_file_name = ""

    def create_hdf5_file(self, name: str) -> str:
        """Creates a new HDF5 file with the given name in the output directory."""
        os.chdir(self.output)
        save_file = h5py.File(f"{name}.hdf5", "a")
        save_file_name = save_file.name
        os.chdir(self.home)
        self.save_file_name = f"{name}.hdf5"
        return save_file_name

    def create_new_hdf5_group(self, path: str) -> str:
        """Creates a new HDF5 group with the given path in the current HDF5 file."""
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

    def load_hdf5_file(self, save_file_name: str) -> h5py.File:
        """Loads the HDF5 file with the given name."""
        os.chdir(self.output)
        save_file = h5py.File(save_file_name, "a")
        os.chdir(self.home)
        return save_file

    def list(self, origins=0, layers=0, isles=0):
        """Lists files in the specified directories."""
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

    def load_bridges_hdf5(self, layer_name: str, island: Island) -> None:
        """Loads bridge data from the HDF5 file for the given layer and island."""
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
                            Bridge(j_key, [], [], [], 0, [], **b_region_group_props)
                        )
                        for k_key, k_item in b_region_group.items():
                            setattr(cob[-1], k_key, np.array(k_item))
        f.close()
        os.chdir(self.home)
        return

    def load_graph_hdf5(self, path: str, name: str) -> np.array:
        """Loads a graph from the HDF5 file."""
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

    def load_img_hdf5(self, path: str, name: str) -> np.array:
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
        if hasattr(layer, "name"):
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

    def load_island_paths_hdf5(self, layer_name: str, island: Island) -> None:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        island_group = f.get(f"/{layer_name}/{island.name}")
        etr_group = island_group.get("external_tree_route")
        itr_group = island_group.get("internal_tree_route")
        twtr_group = island_group.get("thinwalls_tree_route")
        isltr_group = island_group.get("island_route")
        if etr_group:
            island.external_tree_route = Path(
                "external_tree_route",
                list(etr_group.get("sequence")),
                saltos=list(etr_group.get("saltos")),
                img= self.load_img_hdf5(f"/{layer_name}/{island.name}/external_tree_route","img")
            )
            # island.external_tree_route.img = list(etr_group.get("img"))
        if itr_group:
            island.internal_tree_route = Path(
                "internal_tree_route",
                list(itr_group.get("sequence")),
                saltos=list(itr_group.get("saltos")),
                img= self.load_img_hdf5(f"/{layer_name}/{island.name}/internal_tree_route","img")
            )
            # island.internal_tree_route.img = list(itr_group.get("img"))
        if twtr_group:
            island.thinwalls_tree_route = Path(
                "thinwalls_tree_route",
                list(twtr_group.get("sequence")),
                saltos=list(twtr_group.get("saltos")),
                img= self.load_img_hdf5(f"/{layer_name}/{island.name}/thinwalls_tree_route","img")
            )
            # island.thinwalls_tree_route.img = list(twtr_group.get("img"))
        if isltr_group:
            island.island_route = Path(
                "island_route",
                list(isltr_group.get("sequence")),
                saltos=list(isltr_group.get("saltos")),
                img= self.load_img_hdf5(f"/{layer_name}/{island.name}/island_route","img")
            )
            # island.island_route.img = list(isltr_group.get("img"))
        f.close()
        os.chdir(self.home)
        return

    def load_layers_hdf5(self) -> List[Layer]:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        layers = []
        for key, item in f.items():
            if key == "folders_structure":
                for nome_atributo, valor_atributo in f[key].attrs.items(): 
                    setattr(self,nome_atributo,valor_atributo)
            else:
                layers.append(Layer(**dict(f[key].attrs)))
                layers[-1].original_img = np.array(f.get(f"/{key}/original_img"))
                layers[-1].prohibited_areas = np.array(
                    f.get(f"/{key}/prohibited_areas")
                )
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

    def save_layers(self, save_name, layers_list: List[Layer]):
        os.chdir(self.output)
        try:
            f = h5py.File(save_name, "r")
        except:
            self.create_hdf5_file(save_name)
            # f = h5py.File(save_name, "a")
        for layer in layers_list:
            self.create_new_hdf5_group(layer.name)
            self.save_props_hdf5(layer.name, layer.__dict__)
            self.save_img_hdf5(layer.name, "original_img", layer.original_img)
            for island in layer.islands:
                self.create_new_hdf5_group(f"{layer.name}/{island.name}")
                self.save_props_hdf5(f"{layer.name}/{island.name}", island.__dict__)
                self.save_img_hdf5(f"{layer.name}/{island.name}", "img", island.img)
        os.chdir(self.home)
        return

    def save_folders_structure(self, hdf5_file_name):
        os.chdir(self.output)
        f = h5py.File(hdf5_file_name + ".hdf5", "a")
        self.create_new_hdf5_group("folders_structure")
        self.selected = hdf5_file_name
        self.save_props_hdf5("/folders_structure", self.__dict__)
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
                self.delete_item_hdf5(path + "/" + name)
                # local[name][...] = adj_matrix
            # else:
            local.create_dataset(name, data=adj_matrix)
        except:
            print("ERRO: não salvou o grafo!")
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return

    def save_props_hdf5(self, path, dict, **kwargs) -> None:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        local = f.get(path)
        if kwargs:
            for key, value in kwargs.items():
                try:
                    local.attrs[key] = value
                except:
                    pass
        else:
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
                self.delete_item_hdf5(path + "/" + name)
                # local[name] = seq
            # else:
            local.create_dataset(name, data=seq)
        except:
            print("ERRO: não salvou sequencia!")
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return

    def save_regs_thinwalls_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            island_group_name = f"/{layer_name}/{isl.name}"
            self.create_new_hdf5_group(f"{island_group_name}/thin_walls")
            self.delete_item_hdf5(island_group_name + "rest_of_picture_f1")
            self.save_img_hdf5(
                island_group_name, "rest_of_picture_f1", isl.rest_of_picture_f1
            )
            if len(isl.thin_walls.regions) > 0:
                island_tw_group_name = f"/{layer_name}/{isl.name}/thin_walls"
                self.save_img_hdf5(
                    island_tw_group_name,
                    "all_thin_walls",
                    isl.thin_walls.all_thin_walls,
                )
                self.save_img_hdf5(
                    island_tw_group_name, "all_origins", isl.thin_walls.all_origins
                )
                for reg in isl.thin_walls.regions:
                    tw_group_path = f"{island_tw_group_name}/{reg.name}"
                    self.delete_item_hdf5(tw_group_path)
                    self.create_new_hdf5_group(tw_group_path)
                    self.save_props_hdf5(tw_group_path, reg.__dict__)
                    self.save_img_hdf5(tw_group_path, f"img", reg.img)
                    self.save_img_hdf5(tw_group_path, f"origin", reg.origin)
                    self.save_img_hdf5(
                        tw_group_path, f"linha1", reg.elementos_contorno[0]
                    )
                    self.save_img_hdf5(
                        tw_group_path, f"linha2", reg.elementos_contorno[1]
                    )
                    self.save_img_hdf5(
                        tw_group_path, f"linhabaixo", reg.elementos_contorno[3]
                    )
                    self.save_img_hdf5(
                        tw_group_path, f"linhatopo", reg.elementos_contorno[2]
                    )
        return

    def save_regs_bridges_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            if np.sum(isl.rest_of_picture_f3) > 0:
                path_island = f"/{layer_name}/{isl.name}"
                self.save_img_hdf5(
                    path_island, "rest_of_picture_f3", isl.rest_of_picture_f3
                )
                path_island_bridges = f"{path_island}/bridges"
                self.delete_item_hdf5(f"{path_island_bridges}")
                self.create_new_hdf5_group(f"{path_island_bridges}")
                self.save_img_hdf5(
                    path_island_bridges, "all_bridges", isl.bridges.all_bridges
                )
                self.create_new_hdf5_group(f"{path_island_bridges}/offset_bridges")
                self.create_new_hdf5_group(f"{path_island_bridges}/zigzag_bridges")
                self.create_new_hdf5_group(f"{path_island_bridges}/cross_over_bridges")
                for reg in isl.bridges.offset_bridges:
                    path_region = f"{path_island_bridges}/offset_bridges/{reg.name}"
                    self.create_new_hdf5_group(path_region)
                    self.save_img_hdf5(path_region, f"img", reg.img)
                    self.save_img_hdf5(path_region, f"origin", reg.origin)
                    self.save_props_hdf5(path_region, reg.__dict__)
                for reg in isl.bridges.zigzag_bridges:
                    path_region = f"{path_island_bridges}/zigzag_bridges/{reg.name}"
                    self.create_new_hdf5_group(path_region)
                    self.save_img_hdf5(path_region, f"img", reg.img)
                    self.save_img_hdf5(path_region, f"origin", reg.origin)
                    self.delete_item_hdf5(path_region + f"/contorno")
                    self.save_img_hdf5(path_region, f"contorno", reg.contorno)
                    self.save_props_hdf5(path_region, reg.__dict__)
                for reg in isl.bridges.cross_over_bridges:
                    path_region = f"{path_island_bridges}/cross_over_bridges/{reg.name}"
                    self.create_new_hdf5_group(path_region)
                    self.save_img_hdf5(path_region, f"img", reg.img)
                    self.save_img_hdf5(path_region, f"origin", reg.origin)
                    self.delete_item_hdf5(path_region + f"/contorno")
                    self.save_img_hdf5(path_region, f"contorno", reg.contorno)
                    self.save_props_hdf5(path_region, reg.__dict__)
        return

    def save_regs_offsets_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            path_island = f"/{layer_name}/{isl.name}"
            if np.sum(isl.rest_of_picture_f2) > 0:
                self.save_img_hdf5(
                    path_island, "rest_of_picture_f2", isl.rest_of_picture_f2
                )
                path_island_offsets = f"{path_island}/offsets"
                self.delete_item_hdf5(path_island_offsets)
                self.create_new_hdf5_group(path_island_offsets)
                self.save_img_hdf5(
                    path_island_offsets,
                    "all_loops",
                    isl.offsets.all_valid_loops.astype(bool),
                )
                for reg in isl.offsets.regions:
                    reg_path = f"{path_island_offsets}/Reg_{reg.name:03d}"
                    self.create_new_hdf5_group(reg_path)
                    self.save_props_hdf5(reg_path, reg.__dict__)
                    self.save_img_hdf5(
                        reg_path,
                        "img",
                        reg.img.astype(bool),
                    )
                    self.create_new_hdf5_group(f"{reg_path}/loops")
                    for i, loop in enumerate(reg.loops):
                        self.save_img_hdf5(
                            f"{reg_path}/loops",
                            f"Lp_{i:03d}",
                            loop.route.astype(bool),
                        )
                        self.save_props_hdf5(
                            f"{reg_path}/loops/Lp_{i:03d}",
                            loop.__dict__,
                        )
        return

    def save_regs_zigzags_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            if hasattr(isl,"zigzags"):
                base_path = f"/{layer_name}/{isl.name}/zigzags"
                if np.sum(isl.rest_of_picture_f3) > 0:
                    self.delete_item_hdf5(base_path)
                    self.create_new_hdf5_group(base_path)
                    self.save_props_hdf5(base_path, isl.__dict__)
                    for reg in isl.zigzags.regions:
                        if isinstance(reg.name, int):
                            reg.name = f"ZZ_{reg.name:03d}"
                        self.create_new_hdf5_group(f"{base_path}/{reg.name}")
                        self.save_img_hdf5(f"{base_path}/{reg.name}", "img", reg.img)
                        if len(reg.route) > 0:
                            self.save_img_hdf5(
                                f"{base_path}/{reg.name}", "route", reg.route
                            )
                        else:
                            self.delete_item_hdf5(f"{base_path}/{reg.name}/route")
                        if len(reg.trail) > 0:
                            self.save_img_hdf5(
                                f"{base_path}/{reg.name}", "trail", reg.trail
                            )
                        else:
                            self.delete_item_hdf5(f"{base_path}/{reg.name}/trail")
        return

    def save_routes_bridges_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            if np.sum(isl.rest_of_picture_f3) > 0:
                path_bridges = f"/{layer_name}/{isl.name}/bridges"
                for reg in isl.bridges.offset_bridges:
                    region_path = f"{path_bridges}/offset_bridges/{reg.name}"
                    self.save_img_hdf5(region_path, "route", reg.route.astype(bool))
                    self.save_img_hdf5(region_path, f"trail", reg.trail.astype(bool))
                    self.save_props_hdf5(region_path, reg.__dict__)
                for reg in isl.bridges.zigzag_bridges:
                    region_path = f"{path_bridges}/zigzag_bridges/{reg.name}"
                    self.save_img_hdf5(region_path, f"route", reg.route.astype(bool))
                    self.save_img_hdf5(region_path, f"trail", reg.trail.astype(bool))
                    self.save_props_hdf5(region_path, reg.__dict__)
                for reg in isl.bridges.cross_over_bridges:
                    region_path = f"{path_bridges}/cross_over_bridges/{reg.name}"
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
            self.delete_item_hdf5(element_path)
            self.create_new_hdf5_group(element_path)
            self.save_props_hdf5(element_path, isl.external_tree_route.__dict__)
            self.save_seq_hdf5(
                element_path, "sequence", isl.external_tree_route.sequence
            )
            self.save_seq_hdf5(element_path, "saltos", isl.external_tree_route.saltos)
            self.delete_item_hdf5(element_path + "/img")
            self.save_props_hdf5(f"/{layer_name}/{isl.name}", isl.__dict__)
            self.save_img_hdf5(element_path, "img", isl.external_tree_route.img)
        return

    def save_final_routes_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            element_path = f"/{layer_name}/{isl.name}/island_route"
            self.delete_item_hdf5(element_path)
            self.create_new_hdf5_group(element_path)
            self.save_props_hdf5(element_path, isl.island_route.__dict__)
            # self.delete_item_hdf5(element_path + "/sequence")
            self.save_seq_hdf5(element_path, "sequence", isl.island_route.sequence)
            self.save_seq_hdf5(element_path, "saltos", isl.island_route.saltos)
            self.delete_item_hdf5(element_path + "/img")
            self.save_props_hdf5(f"/{layer_name}/{isl.name}", isl.__dict__)
            self.save_img_hdf5(element_path, "img", isl.island_route.img)
            print(isl.island_route.sequence)
        return

    # save_final_routes_hdf5

    def save_internal_routes_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            element_path = f"/{layer_name}/{isl.name}/internal_tree_route"
            self.delete_item_hdf5(element_path)
            self.create_new_hdf5_group(element_path)
            self.save_props_hdf5(element_path, isl.internal_tree_route.__dict__)
            # self.delete_item_hdf5(f"{element_path}/sequence")
            self.save_seq_hdf5(
                element_path, "sequence", isl.internal_tree_route.sequence
            )
            self.save_seq_hdf5(element_path, "saltos", isl.internal_tree_route.saltos)
            self.save_props_hdf5(f"/{layer_name}/{isl.name}", isl.__dict__)
            self.save_img_hdf5(element_path, "img", isl.internal_tree_route.img)
        return

    def save_thinwall_final_routes_hdf5(self, layer_name, islands: List[Island]):
        for isl in islands:
            element_path = f"/{layer_name}/{isl.name}/thinwalls_tree_route"
            self.create_new_hdf5_group(element_path)
            self.save_props_hdf5(element_path, isl.thinwalls_tree_route.__dict__)
            self.save_seq_hdf5(
                element_path, "sequence", isl.thinwalls_tree_route.sequence
            )
            self.save_seq_hdf5(element_path, "saltos", isl.thinwalls_tree_route.saltos)
            self.save_props_hdf5(f"/{layer_name}/{isl.name}", isl.__dict__)
            self.save_img_hdf5(element_path, "img", isl.thinwalls_tree_route.img)
        return

    def save_img_hdf5(self, path, name, img, type="bool"):
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        try:
            local = f.get(path)
            if local.get(name):
                # self.delete_img_hdf5(path+"/"+name)
                local[name][...] = img.astype(local.get(name).dtype)
            else:
                local.create_dataset(name, compression="gzip", data=img)
        except:
            print("ERRO: não salvou imagem!")
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return

    def delete_item_hdf5(self, path):
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        try:
            del f[path]
            print(f"deletado: {path}")
        except:
            # print("ERRO: não deletou a coisa")
            pass
        finally:
            f.close()
            os.chdir(self.home)
        return

    def call_slicer(self, file_name: str, path_input, dpi, layer_height):
        list_layers = []
        os.chdir(self.slicer)
        subprocess.run(["./run-single-model.sh", path_input, str(dpi)])
        camadas_imgs_names = self.list(origins=1)
        n_camadas = len(camadas_imgs_names)
        os.chdir(self.sliced)
        for i, file_path in enumerate(camadas_imgs_names):
            img = imread(file_path, 0)
            if np.sum(img) > 0:
                layer = Layer()
                layer.make_input_img(
                    f"L_{i:03d}", img, dpi, i % 2, layer_height, n_camadas
                )
            list_layers.append(layer)
        os.chdir(self.home)
        return list_layers

@dataclass
class ProgramaDeSolda:
    nome: str
    reg_associada: str
    estrategia: str
    diam_cord: float
    sobrep_cord: float
    vel_desloc: float
    wire_speed: float
    tensao: float
    p_religamento: float
    p_desligamento: float

class Config:
    def __init__(self, target_file):
        self.file = target_file
        self.lista_programas = []
        self.loadYamlConfigs()
        self.updateConfigs()

    def loadYamlConfigs(self):
        with open(self.file, 'r') as file:
            try:
                lista = yaml.safe_load(file)["lista_programas"]
                self.lista_programas = lista
            except:
                pass


    def updateConfigs(self):
        lista_de_programas = []
        for prog in self.lista_programas:
            lista_de_programas.append({
                    "nome": prog['nome'],
                    "reg_associada": prog['reg_associada'],
                    "estrategia": prog['estrategia'],
                    "diam_cord": prog['diam_cord'],
                    "sobrep_cord": prog['sobrep_cord'],
                    "vel_desloc": prog['vel_desloc'],
                    "wire_speed": prog['wire_speed'],
                    "tensao": prog['tensao'],
                    "p_religamento": prog['p_religamento'],
                    "p_desligamento": prog['p_desligamento'],
                })
        configs = {
            "file": self.file,
            "lista_programas": lista_de_programas,
        }
        with open(self.file, "w") as file:
            yaml.dump(configs, file)

    
    def salvar_programaDeSolda(self, 
                               nome, 
                               reg_associada, 
                               estrategia, 
                               diam_cord , 
                               sobrep_cord, 
                               vel_desloc, 
                               wire_speed, 
                               tensao,
                               p_religamento,
                               p_desligamento,):
        self.lista_programas.append((ProgramaDeSolda(nome=nome, 
                                                    reg_associada=reg_associada, 
                                                    estrategia=estrategia, 
                                                    diam_cord=float(diam_cord) , 
                                                    sobrep_cord=float(sobrep_cord), 
                                                    vel_desloc=float(vel_desloc), 
                                                    wire_speed=float(wire_speed), 
                                                    tensao=float(tensao),
                                                    p_religamento=float(p_religamento),
                                                    p_desligamento=float(p_desligamento),
                                                    )).__dict__)
        self.updateConfigs()
