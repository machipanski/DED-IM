from components.layer import Layer
import os
import subprocess
from typing import List


class Paths:
    """Mantem a organizaçao dos caminhos dentro da pasta do programa para evitar carregar coisas que ele nao processa"""

    def __init__(self):
        self.home = os.getcwd()
        self.input = self.home + "/input"
        self.output = self.home + "/output"
        self.slicer = self.home + "/slicing-with-images"
        self.layers = []
        self.selected = ""


def load_layers_3d(arquivos: Paths, path_input, dpi, layer_height):
    """No caso de um arquivo 3D o programa chama o algoritmo SliceWithImages do Prof Minetto e cria um objeto Layer por camada"""
    os.chdir(arquivos.slicer)
    subprocess.run(["./run-single-model.sh", path_input])
    camadas: List[Layer] = []
    os.chdir(arquivos.home + "/input/sliced")
    camadas_imgs_names = sorted([x for x in os.listdir() if x.endswith(".pgm")])
    n_camadas = len(camadas_imgs_names)
    for i, file in enumerate(camadas_imgs_names):
        print(file)
        odd_layer = i % 2
        layer = Layer()
        layer.make_input_img(file, dpi, odd_layer, layer_height, n_camadas, arquivos)
        camadas.append(layer)
    os.chdir(arquivos.home)
    return camadas


def load_layers_2D(arquivos: Paths, path_input, dpi, layer_height):
    """No caso de um arquivo 2D cria um objeto Layer apenas (usado mais para testes mesmo)"""
    camadas: List[Layer] = []
    layer: Layer = Layer()
    layer.make_input_img(path_input, dpi, 0, layer_height, 1, arquivos)
    camadas.append(layer)
    os.chdir(arquivos.home)
    return camadas
