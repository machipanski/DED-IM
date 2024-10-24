from easygui import multenterbox, fileopenbox, buttonbox
from files import System_Paths
import os
import re
from typing import List
from components.layer import Layer


def ask_load_or_begin():
    msg = "Bem vindo! Começamos por onde?"
    choices = ["Carregar Salvo", "Novo Projeto", "Cancelar"]
    reply = buttonbox(msg, choices=choices)
    return reply


def load_model(folders: System_Paths) -> List[str]:
    """Abre uma caixa para explorar os arquivos e captura o caminho"""
    os.chdir(folders.input)
    path_input = fileopenbox()
    file_name = re.sub(folders.input, "", path_input)
    os.chdir(folders.home)
    return path_input, file_name


def find_saved_file(folders: System_Paths) -> List[str]:
    """Abre uma caixa para explorar os arquivos e captura o caminho"""
    os.chdir(folders.output)
    path_input = fileopenbox()
    os.chdir(folders.home)
    file_name = re.sub(folders.input, "", path_input)
    folders.save_file_name = file_name
    return file_name


def ask_parameters_constructor(msg, title, fieldNames, fieldDefs):
    """Modelo basico para criar a caixa para pedir os parametros"""
    fieldValues = multenterbox(msg, title, fieldNames, fieldDefs)
    while 1:
        if fieldValues == None:
            break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "":
            break  # no problems found
        fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)
        print("Reply was:", fieldValues)
    fieldValues = [float(x) for x in fieldValues]
    return fieldValues

def ask_parameters_Gcode():
    """Pede os parametros para a produção, permitindo alterar o que ja esta padronizado"""
    msg = "Gcode parameters"
    title = "Gcode parameters"
    fieldNames = ["vel_int", "vel_ext", "vel_thin_wall", "pausa_religamento(ms)", "pausa_desligamento(ms)", "vel_movimento_vazio", "pausa_entre_camadas"]
    fieldDefs = [360, 360, 360, 1200, 700, 4000, 40000]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_input():
    """Pede todos os parametros de fatiamento, permitindo alterar o que ja esta padronizado"""
    msg = "Slicing parameters"
    title = "Slicing parameters"
    fieldNames = [
        "DPI",
        "Layer Height",
    ]
    fieldDefs = [300, 2]
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_thin_walls():
    """Pede os parametros de thinwalls, permitindo alterar o que ja esta padronizado"""
    msg = "Thin walls parameters"
    title = "Thin walls parameters"
    fieldNames = ["Internal Bw", "External Bw"]
    fieldDefs = [3, 2.8]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_weaving():
    """Pergunta se será aplicado o weaving de preenchimento"""
    msg = "Internal Weaving parameters"
    title = "Internal Weaving parameters"
    fieldNames = ["Internal Weaving"]
    fieldDefs = [1]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_offsets():
    """Pede os parametros de offsets, permitindo alterar o que ja esta padronizado"""
    msg = "Contour parameters"
    title = "Contour parameters"
    fieldNames = ["Vmax", "External max", "Internal max"]
    fieldDefs = [1, 99, 99]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_bridges():
    """Pede os parametros de gargalos, permitindo alterar o que ja esta padronizado"""
    msg = "Bridges parameters"
    title = "Bridges parameters"
    fieldNames = ["n_max", "nozzle_diam_internal"]
    fieldDefs = [4, 2.8]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)
