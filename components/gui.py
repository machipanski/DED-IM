from easygui import multenterbox, fileopenbox, buttonbox, choicebox, msgbox
from components.files import System_Paths
import os
import re
from typing import List


def ask_load_or_begin():
    """Uses easygui to ask about a new document or load a saved file"""
    msg = "Welcome! Where do we begin?"
    choices = ["Load", "New project", "Cancel"]
    reply = buttonbox(msg, choices=choices)
    return reply


def load_model(folders: System_Paths) -> List[str]:
    """Opens a box to explore files and captures the path"""
    os.chdir(folders.input)
    path_input = fileopenbox()
    file_name = re.sub(folders.input, "", path_input)
    os.chdir(folders.home)
    return path_input, file_name


def find_saved_file(folders: System_Paths) -> List[str]:
    """Opens a box to explore files and captures the path"""
    os.chdir(folders.output)
    path_input = fileopenbox(msg="Select an HDF5 file", filetypes=["*.hdf5"])
    os.chdir(folders.home)
    file_name = re.sub(folders.input, "", path_input)
    folders.save_file_name = file_name
    return file_name


def ask_parameters_constructor(msg, title, fieldNames, fieldDefs):
    """Basic model to create the box to ask for parameters"""
    fieldValues = multenterbox(msg, title, fieldNames, fieldDefs)
    while 1:
        if fieldValues is None:
            break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "":
            break
        fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)
        print("   Reply was:", fieldValues)
    fieldValues = [float(x) for x in fieldValues]
    return fieldValues


def ask_parameters_input():
    """Asks for all slicing parameters, allowing changes to what is already standardized"""
    msg = "Slicing parameters"
    title = "Slicing parameters"
    fieldNames = [
        "DPI",
        "Layer Height(mm)",
        "First Layer Height(mm)",
    ]
    fieldDefs = [300, 1.5, 2.0]
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_thin_walls(configurations):
    """Asks for thin walls parameters, allowing changes to what is already standardized"""
    d_tw, fst_d_tw, sob_tw_per, name_prog = select_or_input(
        configurations, thinwalls=True
    )
    return d_tw, fst_d_tw, sob_tw_per, name_prog


def ask_parameters_offsets(configurations):
    """Asks for offsets mapping parameters, allowing changes to what is already standardized"""
    [d_cont, frst_d_cont, sob_cont_per, name_prog] = select_or_input(
        configurations, contours=True
    )
    msg = "Contour parameters"
    title = "Contour parameters"
    fieldNames = [
        "Maximum void(%)",
        "Maximum external contours(nº)",
        "Maximum internal contours(nº)",
    ]
    fieldDefs = [1, 2, 2]
    [void_max, external_max, internal_max] = ask_parameters_constructor(
        msg, title, fieldNames, fieldDefs
    )
    return (
        void_max,
        external_max,
        internal_max,
        d_cont,
        frst_d_cont,
        sob_cont_per,
        name_prog,
    )


def ask_parameters_bridges(configurations):
    """Asks for bridge mapping parameters, allowing changes to what is already standardized"""
    d_bridg, frst_d_bridg, sob_bridg_per, name_prog = select_or_input(
        configurations, bottlenecks=True
    )
    msg = "Bridges parameters"
    title = "Bridges parameters"
    fieldNames = ["Minimum internal passages", "Connect Offset with Bridges"]
    fieldDefs = [2, 1]
    n_max, connect_offsets = ask_parameters_constructor(
        msg, title, fieldNames, fieldDefs
    )
    return n_max, d_bridg, frst_d_bridg, sob_bridg_per, name_prog, connect_offsets


def ask_parameters_zigzags(configurations):
    """Asks for zigzag mapping parameters, allowing changes to what is already standardized"""
    d_larg, frst_d_larg, sob_tw_larg, name_prog = select_or_input(
        configurations, wide=True
    )
    msg = "Wide parameters"
    title = "Wide parameters"
    fieldNames = ["Max number of tracks to consider for weaving"]
    fieldDefs = [4]
    large_weaving_limmit = ask_parameters_constructor(
        msg, title, fieldNames, fieldDefs
    )[0]
    return d_larg, frst_d_larg, sob_tw_larg, name_prog, large_weaving_limmit


def ask_parameters_zigzags_routes(configurations):
    """Asks for zigzag mapping parameters, allowing changes to what is already standardized"""
    msg = "Wide parameters"
    title = "Wide parameters"
    fieldNames = [
        "Style (0=Fermat zigzag, 1=Simple zigzag)",
        "sobreposição(%) interna",
        "sobreposição(%) externa",
        "sobreposição(%) zigzag",
    ]
    fieldDefs = [1, 20, 20, 20]
    style, sob_int, sob_out, sob_zig = ask_parameters_constructor(
        msg, title, fieldNames, fieldDefs
    )
    return style, sob_int, sob_out, sob_zig


def ask_parameters_offset_routes(configurations):
    """Asks for offset route parameters, allowing changes to what is already standardized"""
    msg = "Contour routes parameters"
    title = "Contour routes parameters"
    fieldNames = [
        "Spiralize outer Contour",
        "Ammendment percentage",
    ]
    fieldDefs = [1, 0.7]
    [outer_spiral, amendment_size] = ask_parameters_constructor(
        msg, title, fieldNames, fieldDefs
    )
    return outer_spiral, amendment_size


def ask_parameters_internal_routes():
    """Asks for internal routes parameters"""
    msg = "Internal routes parameters"
    title = "Internal routes parameters"
    fieldNames = ["Internal-external overlap(%)"]
    fieldDefs = [20]
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_weaving():
    """Asks if internal weaving will be applied"""
    msg = "Internal Weaving parameters"
    title = "Internal Weaving parameters"
    fieldNames = ["Internal Weaving? (1yes/0no)"]
    fieldDefs = [1]
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def ask_parameters_Gcode():
    """Asks for the parameters for production, allowing changes to what is already standardized"""
    msg = "Gcode parameters"
    title = "Gcode parameters"
    fieldNames = [
        "Movement speed_empty(mm/m)",
        "Pause between internal and external(ms)",
        "Pause between layers(ms)",
        "Substrate coordinates y(mm)",
        "Substrate coordinates x(mm)",
        "Cut coordinates y(mm)",
        "Cut coordinates x(mm)",
    ]
    fieldDefs = [500, 60000, 400000, 70, 50, 50, 200]
    return ask_parameters_constructor(msg, title, fieldNames, fieldDefs)


def select_or_input(
    configuracoes, thinwalls=False, contours=False, bottlenecks=False, wide=False
):
    if thinwalls:
        called_region = "Thin Walls"
    elif contours:
        called_region = "contours"
    elif bottlenecks:
        called_region = "bottleneck"
    elif wide:
        called_region = "wide_areas"
    item_list = [
        x["name"]
        for x in configuracoes.lista_programas
        if x["used_region"] == called_region
    ]
    selected_item = choicebox(
        f"Select the soldering program for {called_region}:",
        "Select Item",
        item_list + ["Add New Item"],
    )
    if selected_item == "Add New Item":
        field_names = [
            "name",
            "filling_strategy",
            "bead_diameter",
            "bead_diameter_in_first_layer",
            "bead_superposition",
            "travel_speed",
            "wire_speed",
            "voltage",
            "on_pause",
            "off_pause",
        ]
        field_values = multenterbox(
            "Enter values for the fields:", "Input New Values", field_names
        )
        if field_values is not None:
            msgbox(f"New configuration", "Entered Values")
            configuracoes.salvar_programaDeSolda(
                name=field_values[0],
                used_region=called_region,
                filling_strategy=field_values[1],
                diambead_diameter_cord=field_values[2],
                diam_bead_first_layer=field_values[3],
                bead_superposition=field_values[4],
                travel_speed=field_values[5],
                wire_speed=field_values[6],
                voltage=field_values[7],
                on_pause=field_values[8],
                off_pause=field_values[9],
            )
            d_ext = field_values[2]
            sob_ext_per = field_values[3]
            name_prog = field_values[0]
            return float(d_ext), float(sob_ext_per), name_prog
        else:
            msgbox("Input was cancelled.", "Cancelled")
    elif selected_item is not None:
        # msgbox(f"You selected: {selected_item}", "Selected Item")
        # if thinwalls:
        perf_selecionado = list(
            filter(lambda x: x["name"] == selected_item, configuracoes.lista_programas)
        )[0]
        d_ext = perf_selecionado["bead_diameter"]
        frst_d_ext = perf_selecionado["bead_diameter_first_layer"]
        sob_ext_per = perf_selecionado["bead_superposition"]
        name_prog = perf_selecionado["name"]
        return float(d_ext), float(frst_d_ext), float(sob_ext_per), name_prog
    else:
        msgbox("No selection made.", "Cancelled")
