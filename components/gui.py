from easygui import multenterbox, fileopenbox, buttonbox, choicebox, msgbox
from components.files import System_Paths
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


def ask_parameters_thin_walls(configuracoes):
    """Pede os parametros de thinwalls, permitindo alterar o que ja esta padronizado"""
    # msg = "Thin walls parameters"
    # title = "Thin walls parameters"
    # fieldNames = ["Dimetro externo real(mm)", "Sobreposiao contornos(%)"]
    # fieldDefs = [6, 50]  # we start with blanks for the values
    # return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)
    d_tw, sob_tw_per, name_prog = select_or_input(configuracoes, thinwalls=True)
    return d_tw, sob_tw_per, name_prog


def ask_parameters_offsets(configuracoes):
    """Pede os parametros de offsets, permitindo alterar o que ja esta padronizado"""
    [d_cont,sob_cont_per, name_prog] = select_or_input(configuracoes, contonos=True)
    msg = "Contour parameters"
    title = "Contour parameters"
    fieldNames = ["Void maximo(%)", "Contonrnos externos max(nº)", "Contonrnos internos max(nº)"]
    fieldDefs = [1, 99, 99]  # we start with blanks for the values
    [void_max,external_max,internal_max] = ask_parameters_constructor(msg, title, fieldNames, fieldDefs)
    return void_max,external_max,internal_max,d_cont,sob_cont_per, name_prog


def ask_parameters_bridges(configuracoes):
    """Pede os parametros de gargalos, permitindo alterar o que ja esta padronizado"""
    d_bridg, sob_bridg_per, name_prog = select_or_input(configuracoes, estrang=True)
    msg = "Bridges parameters"
    title = "Bridges parameters"
    # fieldNames = ["Passagens internas minimas", "Dimetro interno real", "Sobreposiao internos(%)"]
    fieldNames = ["Passagens internas minimas"]
    fieldDefs = [4]  # we start with blanks for the values
    n_max = ask_parameters_constructor(msg, title, fieldNames, fieldDefs)[0]
    return n_max, d_bridg, sob_bridg_per, name_prog

def ask_parameters_zigzags(configuracoes):
    """Pede os parametros de thinwalls, permitindo alterar o que ja esta padronizado"""
    # msg = "Thin walls parameters"
    # title = "Thin walls parameters"
    # fieldNames = ["Dimetro externo real(mm)", "Sobreposiao contornos(%)"]
    # fieldDefs = [6, 50]  # we start with blanks for the values
    # return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)
    d_larg, sob_tw_larg, name_prog = select_or_input(configuracoes, largas=True)
    return d_larg, sob_tw_larg, name_prog

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
        "Velocidade movimento_vazio(mm/m)",
        "Pausa entre interno e externo(ms)",
        "Pausa entre camadas(ms)",
        "Coordenadas substrato y(mm)",
        "Coordenadas substrato x(mm)",
        "Coordenadas corte y(mm)",
        "Coordenadas corte x(mm)",
    ]
    fieldDefs = [500,60000,400000,70,50,50,200]
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)

def select_or_input(configuracoes, thinwalls=False, contonos=False, estrang=False, largas=False):
    if thinwalls: regiao_chamada = "Thin Walls"
    elif contonos: regiao_chamada = "Contornos"
    elif estrang: regiao_chamada = "Estrangulamentos"
    elif largas: regiao_chamada = "Areas Largas"
    item_list = [x["nome"] for x in configuracoes.lista_programas if x["reg_associada"] == regiao_chamada]
    selected_item = choicebox(f"Selecione o programa de solda para {regiao_chamada}:", 
                                  "Select Item", 
                                  item_list + ["Add New Item"])
    if selected_item == "Add New Item":
        field_names = ["nome","estrategia","diam_cord","sobrep_cord","vel_desloc","wire_speed","tensao","p_religamento","p_desligamento"]
        field_values = multenterbox("Enter values for the fields:", "Input New Values", field_names)
        if field_values is not None:
            msgbox(f"Nova config salva", "Entered Values")
            configuracoes.salvar_programaDeSolda(nome=field_values[0],
                                     reg_associada=regiao_chamada,
                                     estrategia=field_values[1],
                                     diam_cord=field_values[2],
                                     sobrep_cord=field_values[3],
                                     vel_desloc=field_values[4],
                                     wire_speed=field_values[5],
                                     tensao=field_values[6],
                                     p_religamento=field_values[7],
                                     p_desligamento=field_values[8],)
            d_ext = field_values[2]
            sob_ext_per = field_values[3]
            name_prog = field_values[0]
            return float(d_ext), float(sob_ext_per), name_prog
        else:
            msgbox("Input was cancelled.", "Cancelled")
    elif selected_item is not None:
        # msgbox(f"You selected: {selected_item}", "Selected Item")
        # if thinwalls:
        perf_selecionado = list(filter(lambda x: x["nome"] == selected_item, configuracoes.lista_programas))[0]
        d_ext = perf_selecionado["diam_cord"]
        sob_ext_per = perf_selecionado["sobrep_cord"]
        name_prog = perf_selecionado["nome"]
        return float(d_ext), float(sob_ext_per), name_prog
    else:
        msgbox("No selection made.", "Cancelled")