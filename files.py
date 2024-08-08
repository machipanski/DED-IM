from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from components.thin_walls import ThinWallRegions, ThinWall
    # from components.offset import OffsetRegions
    # from components.bridge import BridgeRegions
    # from components.zigzag import ZigZagRegions
    from typing import List
# from components.layer import Layer
from components.layer import Layer, Island
import os, shutil
import subprocess
from typing import List
import json
import jsonpickle
import scipy.sparse
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np

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


    def create_layers_3d(self, path_input, dpi, layer_height):
        """No caso de um arquivo 3D o programa chama o algoritmo SliceWithImages do Prof Minetto e cria um objeto Layer por camada"""
        os.chdir(self.slicer)
        subprocess.run(["./run-single-model.sh", path_input, str(dpi)])
        camadas_imgs_names = self.list(origins = 1)
        n_camadas = len(camadas_imgs_names)
        for root, dirs, files in os.walk(self.output):
                for f in files:
                    if f.endswith(".json") or f.endswith(".png") or f.endswith(".npz"):
                        os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        for i, name in enumerate(camadas_imgs_names):
            odd_layer = i % 2
            layer = Layer()
            os.chdir(self.sliced)
            img = layer.make_input_img(i, name, dpi, odd_layer, layer_height, n_camadas, self)
            os.chdir(self.home)
            img_name = (f"L{layer.name:03d}_original_img.png")
            self.save_img(img_name, img)
            layer.original_img = img_name
            self.save_layer_json(layer)
        os.chdir(self.home)
        return


    def create_layers_2d(self, path_input, dpi, layer_height):
        """No caso de um arquivo 2D cria um objeto Layer apenas (usado mais para testes mesmo)"""
        for root, dirs, files in os.walk(self.output):
                for f in files:
                    if f.endswith(".json") or f.endswith(".png") or f.endswith(".npz"):
                        os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        layer = Layer()
        img = layer.make_input_img(0, path_input, dpi, 0, layer_height, 1, self)
        img_name = (f"L{layer.name:03d}_original_img.png")
        self.save_img(img_name, img)
        layer.original_img = img_name
        self.save_layer_json(layer)
        os.chdir(self.home)
        return
    
    def list(self,origins=0, layers=0, isles=0):
        list = []
        if origins == 1:
            os.chdir(self.sliced)
            list = sorted([x for x in os.listdir() if x.endswith(".pgm")])
        if layers == 1:
            os.chdir(self.output)
            list = sorted([x for x in os.listdir() if x.endswith(".json")])
        if isles == 1:
            os.chdir(self.output)
            list = sorted([x for x in os.listdir() if x.endswith("I",5)])
        os.chdir(self.home)
        return list
    
    def load_layer_json(self, layer_name) -> Layer:
        os.chdir(self.output)
        f = open(layer_name)
        layer = jsonpickle.decode(json.load(f))           
        f.close()
        os.chdir(self.home)
        return layer
    
    def load_img(self, name:str) -> np.ndarray:
        os.chdir(self.output)
        img = cv2.imread(name, 0)
        _ , img_bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img_bin[img_bin > 0] = 1
        return img_bin.astype(np.uint8)
    
    # def load_layer_orig_img(self, layer: Layer) -> np.ndarray:
    #     os.chdir(self.output)
    #     img = cv2.imread(layer.original_img, 0)
    #     _ , img_bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    #     img_bin[img_bin > 0] = 1
    #     return img_bin.astype(np.uint8)
    
    # def load_island_img(self, island: Island)-> np.ndarray:
    #     os.chdir(self.output)
    #     img = cv2.imread(island.img, 0)
    #     _ , img_bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    #     img_bin[img_bin > 0] = 1
    #     return img_bin.astype(np.uint8)
    
    # def load_thin_wall_img(self, tw: ThinWallRegions)-> np.ndarray:
    #     os.chdir(self.output)
    #     img = cv2.imread(tw.img, 0)
    #     _ , img_bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    #     img_bin[img_bin > 0] = 1
    #     return img_bin.astype(np.uint8)
    
    def load_npz(self, name:str)-> np.ndarray:
        os.chdir(self.output)
        medial_sparse = scipy.sparse.load_npz(name)
        array = medial_sparse.toarray()
        return array

    def save_img(self, name, img):
        os.chdir(self.output)
        plt.imsave(name, img, cmap='gray')
        os.chdir(self.home)
        return
    
    def save_npz(self, name, array):
        os.chdir(self.output)
        img_zigzag_bridge_sparse = scipy.sparse.csr_matrix(array)
        scipy.sparse.save_npz(name, img_zigzag_bridge_sparse)
        os.chdir(self.home)
        return
        
    def save_layer_json(self, layer: Layer) -> None:
        os.chdir(self.output)
        copied_layer = copy.deepcopy(layer)
        layer_encoded = jsonpickle.encode(copied_layer)
        with open((f"L{layer.name:03d}.json"), 'w') as f:
            json.dump(layer_encoded, f, indent=1)
        os.chdir(self.home)
        return
    