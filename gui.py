from easygui import multenterbox, fileopenbox
from files import Paths
import os
import re
from typing import List
from components.layer import Layer


def load_model(folders: Paths) -> List[str]:
    """Abre uma caixa para explorar os arquivos e captura o caminho"""
    os.chdir(folders.input)
    path_input = fileopenbox()
    os.chdir(folders.home)
    file_name = re.sub(folders.input, "", path_input)
    return path_input, file_name

def ask_parameters_constructor(msg, title, fieldNames, fieldDefs):
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

def ask_parameters_offsets():
    """Pede os parametros de offsets, permitindo alterar o que ja esta padronizado"""
    msg = "Contour parameters"
    title = "Contour parameters"
    fieldNames = ["Vmax", "External max", "Internal max"]
    fieldDefs = [1, 99, 99]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)