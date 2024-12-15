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


def ask_parameters_input():
    """Pede todos os parametros de fatiamento, permitindo alterar o que ja esta padronizado"""
    msg = "Slicing parameters"
    title = "Slicing parameters"
    fieldNames = [
        "DPI",
        "Layer Height(mm)",
    ]
    fieldDefs = [300, 1.5]
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_thin_walls():
    """Pede os parametros de thinwalls, permitindo alterar o que ja esta padronizado"""
    msg = "Thin walls parameters"
    title = "Thin walls parameters"
    fieldNames = ["Dimetro externo real(mm)", "Sobreposiao contornos(%)"]
    fieldDefs = [6, 50]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_offsets():
    """Pede os parametros de offsets, permitindo alterar o que ja esta padronizado"""
    msg = "Contour parameters"
    title = "Contour parameters"
    fieldNames = ["Void maximo(%)", "Contonrnos externos max(nº)", "Contonrnos internos max(nº)"]
    fieldDefs = [1, 99, 99]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_bridges():
    """Pede os parametros de gargalos, permitindo alterar o que ja esta padronizado"""
    msg = "Bridges parameters"
    title = "Bridges parameters"
    fieldNames = ["Passagens internas minimas", "Dimetro interno real", "Sobreposiao internos(%)"]
    fieldDefs = [4, 7, 30]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)

def ask_parameters_internal_routes():
    """Pede os parametros de rotas internas"""
    msg = "internal_routes parameters"
    title = "internal_routes parameters"
    fieldNames = ["Sobreposiao interno-externo(%)"]
    fieldDefs = [40]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)

def ask_parameters_weaving():
    """Pergunta se será aplicado o weaving de preenchimento"""
    msg = "Internal Weaving parameters"
    title = "Internal Weaving parameters"
    fieldNames = ["Internal Weaving? (1sim/0nao)"]
    fieldDefs = [1]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_Gcode():
    """Pede os parametros para a produção, permitindo alterar o que ja esta padronizado"""
    msg = "Gcode parameters"
    title = "Gcode parameters"
    fieldNames = [
        "Velocidade internos(mm/m)",
        "Velocidade externos(mm/m)",
        "Velocidade thin_wall(mm/m)",
        "Velocidade movimento_vazio(mm/m)",
        "Pausa religamento(ms)",
        "Pausa desligamento(ms)",
        "Pausa entre interno e externo(ms)",
        "Pausa entre_camadas(ms)",
        "Coordenadas substrato y(mm)",
        "Coordenadassubstrato x(mm)",
        "Coordenadascorte y(mm)",
        "Coordenadascorte x(mm)",
    ]
    fieldDefs = [
        300,
        360,
        400,
        500,
        700,
        1200,
        60000,
        400000,
        70,
        50,
        50,
        200,
    ]  # we start with blanks for the values
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)
