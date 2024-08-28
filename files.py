from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from components.thin_walls import ThinWallRegions, ThinWall
    from typing import List
from components.layer import Layer, Island
import os, shutil
import subprocess
from typing import List
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import h5py


class Paths:
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

    def create_hdf5_file(self, name):
        os.chdir(self.output)
        save_file = h5py.File(f"{name}.hdf5", "a")
        save_file_name = save_file.name
        os.chdir(self.home)
        self.save_file_name = f"{name}.hdf5"
        return save_file_name

    def load_hdf5_file(self, save_file_name) -> h5py.File:
        os.chdir(self.output)
        save_file = h5py.File(save_file_name, "a")
        os.chdir(self.home)
        return save_file

    def create_layers_3d(
        self, path_input, dpi, layer_height, file_name, folders: Paths
    ):
        """No caso de um arquivo 3D o programa chama o algoritmo SliceWithImages do Prof Minetto e cria um objeto Layer por camada"""
        os.chdir(self.slicer)
        subprocess.run(["./run-single-model.sh", path_input, str(dpi)])
        camadas_imgs_names = self.list(origins=1)
        n_camadas = len(camadas_imgs_names)
        for root, dirs, files in os.walk(self.output):
            for f in files:
                if f.endswith(".json") or f.endswith(".png") or f.endswith(".npz"):
                    os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        save_name = file_name.replace(".stl", "")
        save_name = save_name.replace(".STL", "")
        save_name = save_name.replace("stl_models", "")
        save_name = save_name.replace("/", "")
        save_file = folders.create_hdf5_file(save_name)
        for i, name in enumerate(camadas_imgs_names):
            layer_group = folders.create_new_hdf5_group(f"/L_{i:03d}")
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
            folders.save_props_hdf5(layer_group, props_layer)
        os.chdir(self.home)
        return

    def create_layers_2d(
        self, path_input, dpi, layer_height, file_name: str, folders: Paths
    ):
        """No caso de um arquivo 2D cria um objeto Layer apenas (usado mais para testes mesmo)"""
        for root, dirs, files in os.walk(self.output):
            for f in files:
                if f.endswith(".json") or f.endswith(".png") or f.endswith(".npz"):
                    os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        layer = Layer()
        img = layer.make_input_img(0, path_input, dpi, 0, layer_height, 1, self)
        save_name = file_name.replace(".pgm", "")
        save_name = save_name.replace("/", "")
        save_file = folders.create_hdf5_file(save_name)
        layer_group_nome = folders.create_new_hdf5_group(f"/L_000")
        props_layer = {
            "name": "L_000",
            "dpi": dpi,
            "base_frame": img.shape,
            "n_camadas": 1,
            "layer_height": layer_height,
            "odd_layer": 0,
            "img_name": "original_img",
        }
        folders.save_props_hdf5(layer_group_nome, props_layer)
        folders.save_img_hdf5(layer_group_nome, "original_img", img)
        # layer_group.attrs["name"] = 0
        # layer_group.attrs["dpi"] = dpi
        # layer_group.attrs["base_frame"] = img.shape
        # layer_group.attrs["n_camadas"] = 1
        # layer_group.attrs["layer_height"] = layer_height
        # save_file.close()
        # os.chdir(self.home)
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

    def load_layers_hdf5(self) -> List[Layer]:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        layers = []
        for l in f:
            layers.append(Layer(**dict(f[l].attrs)))
        f.close()
        os.chdir(self.home)
        return layers

    def load_islands_hdf5(self, layer) -> List[Island]:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        islands = []
        group_entries = list(f[layer.name])
        list_islands = list(filter(lambda x: x.startswith("I_"), group_entries))
        for isl in list_islands:
            attributes = dict(f[layer.name][isl].attrs)
            islands.append(Island(**attributes))
        f.close()
        os.chdir(self.home)
        return islands

    def load_img_hdf5(self, path, name) -> np.array:
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "r")
        local = f.get(path)
        img = np.array(local[name])
        f.close()
        os.chdir(self.home)
        return img

    def save_img_hdf5(self, path, name, img, type="bool"):
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        local = f.get(path)
        if local.get(name):
            local[name][...] = img.astype(bool)
        else:
            local.create_dataset(name, compression="gzip", data=img, dtype=type)
        f.close()
        os.chdir(self.home)
        return

    def save_props_hdf5(self, path, dict):
        os.chdir(self.output)
        f = h5py.File(self.save_file_name, "a")
        local = f.get(path)
        for key, value in dict.items():
            local.attrs[key] = value
        f.close()
        os.chdir(self.home)
        return

    # def save_npz(self, name, array):
    #     os.chdir(self.output)
    #     img_zigzag_bridge_sparse = scipy.sparse.csr_matrix(array)
    #     scipy.sparse.save_npz(name, img_zigzag_bridge_sparse)
    #     os.chdir(self.home)
    #     return

    # def load_layer_json(self, layer_name) -> Layer:
    #     os.chdir(self.output)
    #     f = open(layer_name)
    #     layer = jsonpickle.decode(json.load(f))
    #     f.close()
    #     os.chdir(self.home)
    #     return layer

    # def load_img(self, name: str) -> np.ndarray:
    #     os.chdir(self.output)
    #     img = cv2.imread(name, 0)
    #     _, img_bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    #     img_bin[img_bin > 0] = 1
    #     return img_bin.astype(np.uint8)

    # def load_npz(self, name: str) -> np.ndarray:
    #     os.chdir(self.output)
    #     medial_sparse = scipy.sparse.load_npz(name)
    #     array = medial_sparse.toarray()
    #     return array

    # def save_img(self, name, img):
    #     os.chdir(self.output)
    #     plt.imsave(name, img, cmap="gray")
    #     os.chdir(self.home)
    #     return

    # def save_layer_json(self, layer: Layer) -> None:
    #     os.chdir(self.output)
    #     copied_layer = copy.deepcopy(layer)
    #     layer_encoded = jsonpickle.encode(copied_layer)
    #     with open((f"{layer.name}.json"), "w") as f:
    #         json.dump(layer_encoded, f, indent=1)
    #     os.chdir(self.home)
    #     return
