from __future__ import annotations
import datetime
import os
import numpy as np
from typing import TYPE_CHECKING
from scipy.spatial import distance
from components import path_tools

if TYPE_CHECKING:
    from typing import List
    from components.layer import Layer
    from components.files import System_Paths


def get_program_params(program, lista_programas):
    A = list(filter(lambda x: x["name"] == program, lista_programas))[0]
    diam = A["bead_diameter"]
    sobrep = A["bead_superposition"]
    vel = A["travel_speed"]
    on_pause = A["on_pause"]
    off_pause = A["off_pause"]
    wfs = A["wire_speed"]
    return diam, sobrep, vel, wfs, on_pause, off_pause


def turn_on(output, flag_on, on_pause):
    if flag_on == 0:
        output += ";-------Turn ON Welding------\n"
        output += "M42 P4 S0\n"
        # output += f"G4 P{p_trigger_longa}\n"
        output += f"G4 P{on_pause}\n"
        # output += "M42 P4 S255\n"
        # output += f"G4 P{on_pause-p_trigger_longa}\n"
        output += ";------------------------\n"
    return output, 1


def turn_on_UFSC(output, flag_on, on_pause):
    if flag_on == 0:
        output += ";-------Turn ON Welding------\n"
        output += (
            f"M200 P{on_pause/1000}; Liga a tocha e aguarda {on_pause/1000} segundos\n"
        )
        output += ";------------------------\n"
    return output, 1


def turn_off(output, flag_on, off_pause=3000):
    if flag_on == 1:
        output += ";-------Turn OFF Welding------\n"
        # output += "M42 P4 S0\n"
        # output += f"G4 P{p_trigger_longa}\n"
        output += "M42 P4 S255\n"
        output += f"G4 P{off_pause}\n"
        # output += f"G4 P{off_pause-p_trigger_longa}\n"
        output += ";-------------------------\n"
    return output, 0


def turn_off_UFSC(output, flag_on, off_pause=3000):
    if flag_on == 1:
        output += ";-------Turn OFF Welding------\n"
        output += f"M200 P{off_pause/1000}; Desliga a tocha e aguarda {off_pause/1000} segundos\n"
        output += ";-------------------------\n"
    return output, 0


def program_change(
    output,
    now,
    next_program,
    flag_on_before,
    vel_cont,
    vel_bridg,
    vel_larg,
    vel_tw,
    vel_vazio,
    p_entre_int_ext,
    p_trigger_curta,
    p_trigger_longa,
    off_pause_cont,
    off_pause_bridg,
    off_pause_larg,
    off_pause_tw,
    on_pause_cont,
    on_pause_bridg,
    on_pause_larg,
    on_pause_tw,
    flag_path_type,
    off_pause=3000,
):
    if flag_on_before == 1:
        output, _ = turn_off(output, flag_on_before, off_pause)
    if next_program == 1:
        vel = vel_cont
        texto_mudanca = ";----Contour----\n;TYPE:WALL-OUTER\n"
        const_perf = 5
        off_pause = off_pause_cont
        on_pause = on_pause_cont
    elif next_program == 2:
        vel = vel_bridg
        texto_mudanca = ";----Bottleneck----\n;TYPE:SKIN\n"
        const_perf = 8
        off_pause = off_pause_bridg
        on_pause = on_pause_bridg
    elif next_program == 3:
        vel = vel_larg
        texto_mudanca = ";----Wide area----\n;TYPE:WALL-INNER\n"
        const_perf = 0.5
        off_pause = off_pause_larg
        on_pause = on_pause_larg
    elif next_program == 4:
        vel = vel_tw
        texto_mudanca = ";----ThinWalls----\n;TYPE:SUPPORT\n"
        const_perf = 0.5
        off_pause = off_pause_tw
        on_pause = on_pause_tw
    else:
        vel = vel_vazio
        texto_mudanca = ";----Lost----\n"
        off_pause = 0
        on_pause = on_pause_cont
        const_perf = 0
    output += f";-------Changing program {now}->{next_program}------\n"
    print(f"Switched to {flag_path_type}")
    output += texto_mudanca
    diferenca = next_program - now
    if diferenca < 0:
        diferenca = 4 + diferenca
    if now == 1:
        output += "M400\n"
        output += f"G4 P{p_entre_int_ext}\n"
    for toque in range(diferenca):
        output += "M400\n"
        output += "M42 P4 S0\n"
        output += f"G4 P{p_trigger_curta}\n"
        output += "M42 P4 S255\n"
        output += f"G4 P{p_trigger_curta}\n"
        output += f"G4 P{p_trigger_curta}\n"
    output += ";-------------------------\n"
    output += f"G1 F{vel}; speed g1\n"
    if flag_on_before == 1:
        output, _ = turn_on(output, 0, on_pause)
    return output, off_pause, on_pause


def program_change_UFSC(
    output,
    now,
    next_program,
    flag_on_before,
    vel_cont,
    wfs_cont,
    vel_bridg,
    wfs_bridg,
    vel_larg,
    wfs_larg,
    vel_tw,
    wfs_tw,
    vel_vazio,
    p_entre_int_ext,
    p_trigger_curta,
    p_trigger_longa,
    off_pause_cont,
    off_pause_bridg,
    off_pause_larg,
    off_pause_tw,
    on_pause_cont,
    on_pause_bridg,
    on_pause_larg,
    on_pause_tw,
    flag_path_type,
    off_pause=3000,
):
    if flag_on_before == 1:
        output, _ = turn_off(output, flag_on_before, off_pause)
    if next_program == 1:
        vel = vel_cont
        wfs = wfs_cont
        texto_mudanca = ";----Contour----\n;TYPE:WALL-OUTER\n"
        const_perf = 5
        off_pause = off_pause_cont
        on_pause = on_pause_cont
    elif next_program == 2:
        vel = vel_bridg
        wfs = wfs_bridg
        texto_mudanca = ";----Bottleneck----\n;TYPE:SKIN\n"
        const_perf = 8
        off_pause = off_pause_bridg
        on_pause = on_pause_bridg
    elif next_program == 3:
        vel = vel_larg
        wfs = wfs_larg
        texto_mudanca = ";----Wide area----\n;TYPE:WALL-INNER\n"
        const_perf = 0.5
        off_pause = off_pause_larg
        on_pause = on_pause_larg
    elif next_program == 4:
        vel = vel_tw
        wfs = wfs_tw
        texto_mudanca = ";----ThinWalls----\n;TYPE:SUPPORT\n"
        const_perf = 0.5
        off_pause = off_pause_tw
        on_pause = on_pause_tw
    else:
        vel = vel_vazio
        wfs = 0
        texto_mudanca = ";----Lost----\n"
        off_pause = 0
        on_pause = on_pause_cont
        const_perf = 0
    output += f";-------Changing program {now}->{next_program}------\n"
    print(f"Switched to {flag_path_type}")
    output += texto_mudanca
    # output += "M400\n"
    output += f"M202 P{next_program}\n"
    output += f"G1 F{vel}; speed g1\n"
    output += f";wire feed speed {wfs}\n"
    output += ";-------------------------\n"
    if flag_on_before == 1:
        output, _ = turn_on(output, 0, on_pause)
    return output, off_pause, on_pause


def cleanning_position(output, coords, vel_vazio, p_entre_layers):
    output += ";-------CLEANNING POSITION------\n"
    # output += f";POS de Corte\n"
    output += f"G90\n"
    output += f"G0 Y{coords[0]} F{vel_vazio}\n"
    # output += f"M400\n"
    output += f"G0 x{coords[1]} F{vel_vazio}\n"
    # output += f"M400\n"
    output += f"G4 P{p_entre_layers}\n"
    output += f"G91\n"
    output += ";------------------------\n"
    return output


def cleanning_position_UFSC(output, coords, vel_vazio, p_entre_layers):
    output += ";-------CLEANNING POSITION------\n"
    # output += f";POS de Corte\n"
    output += f"G90\n"
    output += f"G0 Y{coords[0]} F{vel_vazio}\n"
    # output += f"M400\n"
    output += f"G0 x{coords[1]} F{vel_vazio}\n"
    # output += f"M400\n"
    output += f"G4 P{p_entre_layers}\n"
    output += f"G91\n"
    output += ";------------------------\n"
    return output


def initial_position(output, coords, height, vel_vazio, n_layer):
    # output += f";_______LAYER{n_layer + 1}_____\n"
    output += f";LAYER:{n_layer}\n"
    output += ";-------INITIAL POSITION------\n"
    output += f"G90\n"
    # output += f";LAYER:{i}\n"
    output += f"G1 Z{height} ; Layer + 10mm\n"
    output += f"G1 X{coords[1]} Y{coords[0]} F{vel_vazio}\n"
    output += f"M400\n"
    output += f"G91\n"
    output += ";------------------------\n"
    return output


def initial_position_UFSC(output, coords, height, vel_vazio, n_layer):
    # output += f";_______LAYER{n_layer + 1}_____\n"
    output += f";LAYER:{n_layer}\n"
    output += ";-------INITIAL POSITION------\n"
    output += f"G90\n"
    # output += f";LAYER:{i}\n"
    output += f"G1 Z{height} ; Layer + 10mm\n"
    output += f"G1 X{coords[1]} Y{coords[0]} F{vel_vazio}\n"
    # output += f"M400\n"
    output += f"G91\n"
    output += ";------------------------\n"
    return output


def layers_to_Gcode(
    layers: List[Layer],
    folders: System_Paths,
    configuracoes,
    vel_vazio,
    p_entre_int_ext,
    p_entre_layers,
    # layer_heights,
    base_coords,
    coords_corte,
    flag_drawing=False,
):
    """Originaly used in an Okerlion machine using 2T mode to operate the soldering (FCT-NOVA in Portugal)"""

    def code_start(output, flag_ligado):
        output, flag_ligado = turn_off(output, flag_ligado)
        output += ";-------MAPPING------\n"
        output += f";DPI: {layers[0].dpi} ppp\n"
        output += f";Void_max: {layers[0].void_max} % of path_radius\n"
        output += f";Max_internal_walls: {layers[0].max_internal_walls}\n"
        output += f";Max_external_walls: {layers[0].max_external_walls}\n"
        output += f";Bottleneck_n_max: {layers[0].n_max} trilhas para bottleneck\n"
        output += ";-------Welding program 1 contours------\n"
        output += f";Program name: {layers[0].program_cont}\n"
        output += f";Bead diameter: {diam_cont} mm\n"
        output += f";Bead superposition: {sobrep_cont} % raio real \n"
        output += f";Travel_speed: {vel_cont} mm/min \n"
        output += f";Path_radius: {layers[0].path_radius_cont} pixels\n"
        # output += f";On_pause: {on_pause_cont} ms \n"
        # output += f";Off_pause: {off_pause_cont} ms \n"
        output += ";-------Welding program 2 bottlenecks------\n"
        output += f";Program name: {layers[0].program_bridg}\n"
        output += f";Bead diameter: {diam_bridg} mm\n"
        output += f";Bead superposition: {sobrep_bridg} % raio real \n"
        output += f";Travel_speed: {vel_bridg} mm/min \n"
        output += f";Path_radius: {layers[0].path_radius_bridg} pixels\n"
        output += f";On_pause: {on_pause_bridg} ms \n"
        output += f";Off_pause: {off_pause_bridg} ms \n"
        output += ";-------Welding program 3 wide areas------\n"
        output += f";Program name: {layers[0].program_larg}\n"
        output += f";Bead diameter: {diam_larg} mm\n"
        output += f";Bead superposition: {sobrep_larg} % raio real \n"
        output += f";Travel_speed: {vel_larg} mm/min \n"
        output += f";Path_radius: {layers[0].path_radius_larg} pixels\n"
        output += f";On_pause: {on_pause_larg} ms \n"
        output += f";Off_pause: {off_pause_larg} ms \n"
        output += ";-------Welding program 4 thin walls------\n"
        output += f";Program name: {layers[0].program_tw}\n"
        output += f";Bead diameter: {diam_tw} mm\n"
        output += f";Bead superposition: {sobrep_tw} % raio real \n"
        output += f";Travel_speed: {vel_tw} mm/min \n"
        output += f";Path_radius: {layers[0].path_radius_tw} pixels\n"
        output += f";On_pause: {on_pause_tw} ms \n"
        output += f";Off_pause: {off_pause_tw} ms \n"
        output += ";------------OTHER------------\n"
        output += (
            f";Ext int superposition: {layers[0].sob_int_ext_per} % raio interno \n"
        )
        output += f";N# of layers: {layers[0].n_layers}\n"
        output += f";Pause_between_interanal_and_external;: {p_entre_int_ext} ms \n"
        output += f";Pause_between_layers: {p_entre_layers} ms \n"
        output += f";First Layer_height: {layers[0].layer_height} mm \n"
        if len(layers) > 1:
            output += f";Normal Layer_height: {layers[1].layer_height} mm \n"
        output += f";Coodinates_cleaning: {coords_corte} mm \n"
        output += f";Coordinates_base: {base_coords} mm \n"
        output += f";Empty_movement_speed: {vel_vazio} mm/min \n"
        output += ";------------INPUTS END------------\n"
        output += f"G91\n"
        # output += f"M42 P4 S255; turn off welder\n"
        output += f"G28 X0 Y0 Z0\n"
        # output += f"G1 F360; speed g1\n"
        return output

    diam_cont, sobrep_cont, vel_cont, on_pause_cont, off_pause_cont = (
        get_program_params(layers[0].program_cont, configuracoes.lista_programas)
    )
    diam_bridg, sobrep_bridg, vel_bridg, on_pause_bridg, off_pause_bridg = (
        get_program_params(layers[0].program_bridg, configuracoes.lista_programas)
    )
    diam_larg, sobrep_larg, vel_larg, on_pause_larg, off_pause_larg = (
        get_program_params(layers[0].program_larg, configuracoes.lista_programas)
    )
    diam_tw, sobrep_tw, vel_tw, on_pause_tw, off_pause_tw = get_program_params(
        layers[0].program_tw, configuracoes.lista_programas
    )
    p_trigger_longa = 800
    p_trigger_curta = 300
    coords = [0, 0]
    bfr = [0, 0]
    base_frame = layers[0].base_frame
    ts = datetime.datetime.now()
    outFile = f"{folders.selected} {ts.date()} {ts.hour}_{ts.minute}.gcode"
    flag_on = 1
    flag_path_type = 0
    output = ""
    output = code_start(output, flag_on)
    mm_per_pixel = layers[0].mm_per_pxl
    for n_layer, layer in enumerate(layers):
        diam_cont, sobrep_cont, vel_cont, on_pause_cont, off_pause_cont = (
            get_program_params(layer.program_cont, configuracoes.lista_programas)
        )
        diam_bridg, sobrep_bridg, vel_bridg, on_pause_bridg, off_pause_bridg = (
            get_program_params(layer.program_bridg, configuracoes.lista_programas)
        )
        diam_larg, sobrep_larg, vel_larg, on_pause_larg, off_pause_larg = (
            get_program_params(layer.program_larg, configuracoes.lista_programas)
        )
        diam_tw, sobrep_tw, vel_tw, on_pause_tw, off_pause_tw = get_program_params(
            layer.program_tw, configuracoes.lista_programas
        )
        layer_tot_lenght = 0
        bfr = base_coords
        layer_height = layer.layer_height
        output = initial_position(output, base_coords, layer_height, vel_vazio, n_layer)
        if layer.islands == []:
            folders.load_islands_hdf5(layer)
        for n_island, island in enumerate(layer.islands):
            counter = 0
            last_flag = 0
            flag_on = 0
            if flag_drawing == True:
                pts_cont = island.contours.pts_cont
                pts_bridg, pts_tw, pts_larg = [], [], []
                chain = island.island_route
            else:
                pts_bridg, pts_tw, pts_cont, pts_larg = path_tools.region_points(
                    layer, island, folders
                )
                folders.load_island_paths_hdf5(layer.name, island)
                chain = [list(x) for x in island.island_route.sequence]

            for i, p in enumerate(chain):
                if i <= 2:
                    flag_salto = 1
                if p == [0, 0]:
                    output, flag_on = turn_off(output, flag_on)
                    const_perf = 0
                    flag_salto = 1
                else:
                    coords = p
                    coords = [
                        base_frame[0] - coords[0] + base_coords[0],
                        coords[1] + base_coords[1],
                    ]
                    if p in pts_cont:
                        flag_path_type = 1
                    elif p in pts_bridg:
                        flag_path_type = 2
                    elif p in pts_larg:
                        flag_path_type = 3
                    elif p in pts_tw:
                        flag_path_type = 4
                    else:
                        flag_path_type = 0
                    if flag_path_type != last_flag:
                        output, off_pause, on_pause = program_change(
                            output,
                            last_flag,
                            flag_path_type,
                            flag_on,
                            vel_cont,
                            vel_bridg,
                            vel_larg,
                            vel_tw,
                            vel_vazio,
                            p_entre_int_ext,
                            p_trigger_curta,
                            p_trigger_longa,
                            off_pause_cont,
                            off_pause_bridg,
                            off_pause_larg,
                            off_pause_tw,
                            on_pause_cont,
                            on_pause_bridg,
                            on_pause_larg,
                            on_pause_tw,
                            flag_path_type,
                        )
                        output += f"G117 {{Trocou o perfil para {flag_path_type}}}\n"
                        last_flag = flag_path_type
                    desloc = np.subtract(coords, bfr)
                    dist = distance.euclidean(coords, bfr)
                    layer_tot_lenght += dist
                    output += (
                        f"G1 X{desloc[1] * mm_per_pixel} Y{desloc[0] * mm_per_pixel}\n"
                    )
                    output += "M400\n"
                    bfr = coords
                    counter += 1
                    if flag_salto == 1:
                        output, flag_on = turn_on(output, flag_on, on_pause)
                        flag_salto = 0
        output, flag_on = turn_off(output, flag_on, off_pause)
        output = cleanning_position(output, coords_corte, vel_vazio, p_entre_layers)
        output += ";____________________________________\n"
        output += f"G28 X0 Y0\n"
        print(f"Total travel distance {n_layer} = {layer_tot_lenght*mm_per_pixel}mm")
        print(
            f"Estimated time with speed {vel_cont}mm/min = {layer_tot_lenght*mm_per_pixel/vel_cont}min\n"
        )
    output += f"G1 Z20\n"
    output += f"G28 X0\n"
    output += f"G28 Y0\n"
    output += f"M104 S0; End of Gcode\n"
    os.chdir(folders.output)
    f = open(outFile, "w")
    f.write(output)
    f.close()
    os.chdir(folders.home)
    return


def layers_to_Gcode_UFSC(
    layers: List[Layer],
    folders: System_Paths,
    configuracoes,
    vel_vazio,
    p_entre_int_ext,
    p_entre_layers,
    # layer_heights,
    base_coords,
    coords_corte,
    flag_drawing=False,
):

    def code_start(output, flag_ligado):
        output, flag_ligado = turn_off_UFSC(output, flag_ligado)
        output += f"G91 ; Modo incremental\n"
        output += f"G28 X0 Y0 Z0 ; Vai à origem da máquina\n"
        output += f"G21 ; Unidades em milímetros\n"
        output += f"G64P0.25 ; Modo contínuo com tolerância 0.25mm\n"
        output += f"G40 ; Cancela compensação raio\n"
        output += f"G49 ; Cancela compensação comprimento\n"
        output += f"G80 ; Cancela ciclos fixos\n"
        output += f"G92.1 ; Reseta deslocamento coordenadas\n"
        output += f"G94 ; Avanço em mm/min\n"
        output += f"G97 ; Velocidade fuso em RPM\n"
        output += f"M200 P5.0 ; Comando personalizado - Configura solda\n"
        return output

    diam_cont, sobrep_cont, vel_cont, on_pause_cont, off_pause_cont, wfs_cont = (
        get_program_params(layers[0].program_cont, configuracoes.lista_programas)
    )
    diam_bridg, sobrep_bridg, vel_bridg, on_pause_bridg, off_pause_bridg, wfs_bridg = (
        get_program_params(layers[0].program_bridg, configuracoes.lista_programas)
    )
    diam_larg, sobrep_larg, vel_larg, on_pause_larg, off_pause_larg, wfs_larg = (
        get_program_params(layers[0].program_larg, configuracoes.lista_programas)
    )
    diam_tw, sobrep_tw, vel_tw, on_pause_tw, off_pause_tw, wfs_tw = get_program_params(
        layers[0].program_tw, configuracoes.lista_programas
    )
    p_trigger_longa = 800
    p_trigger_curta = 300
    coords = [0, 0]
    bfr = [0, 0]
    base_frame = layers[0].base_frame
    ts = datetime.datetime.now()
    outFile = f"{folders.selected} {ts.date()} {ts.hour}_{ts.minute}.gcode"
    flag_on = 1
    flag_path_type = 0
    output = ""
    output = code_start(output, flag_on)
    mm_per_pixel = layers[0].mm_per_pxl
    for n_layer, layer in enumerate(layers):
        diam_cont, sobrep_cont, vel_cont, on_pause_cont, off_pause_cont, wfs_cont = (
            get_program_params(layer.program_cont, configuracoes.lista_programas)
        )
        (
            diam_bridg,
            sobrep_bridg,
            vel_bridg,
            on_pause_bridg,
            off_pause_bridg,
            wfs_bridg,
        ) = get_program_params(layer.program_bridg, configuracoes.lista_programas)
        diam_larg, sobrep_larg, vel_larg, on_pause_larg, off_pause_larg, wfs_larg = (
            get_program_params(layer.program_larg, configuracoes.lista_programas)
        )
        diam_tw, sobrep_tw, vel_tw, on_pause_tw, off_pause_tw, wfs_tw = (
            get_program_params(layer.program_tw, configuracoes.lista_programas)
        )
        layer_tot_lenght = 0
        bfr = base_coords
        layer_height = layer.layer_height
        output = initial_position_UFSC(
            output, base_coords, layer_height, vel_vazio, n_layer
        )
        if layer.islands == []:
            folders.load_islands_hdf5(layer)
        for n_island, island in enumerate(layer.islands):
            counter = 0
            last_flag = 0
            flag_on = 0
            if flag_drawing == True:
                pts_cont = island.contours.pts_cont
                pts_bridg, pts_tw, pts_larg = [], [], []
                chain = island.island_route
            else:
                pts_bridg, pts_tw, pts_cont, pts_larg = path_tools.region_points(
                    layer, island, folders
                )
                folders.load_island_paths_hdf5(layer.name, island)
                chain = [list(x) for x in island.island_route.sequence]

            for i, p in enumerate(chain):
                if i <= 2:
                    flag_salto = 1
                if p == [0, 0]:
                    output, flag_on = turn_off_UFSC(output, flag_on)
                    const_perf = 0
                    flag_salto = 1
                else:
                    coords = p
                    coords = [
                        base_frame[0] - coords[0] + base_coords[0],
                        coords[1] + base_coords[1],
                    ]
                    if p in pts_cont:
                        flag_path_type = 1
                    elif p in pts_bridg:
                        flag_path_type = 2
                    elif p in pts_larg:
                        flag_path_type = 3
                    elif p in pts_tw:
                        flag_path_type = 4
                    else:
                        flag_path_type = 0
                    if flag_path_type != last_flag:
                        output, off_pause, on_pause = program_change_UFSC(
                            output,
                            last_flag,
                            flag_path_type,
                            flag_on,
                            vel_cont,
                            wfs_cont,
                            vel_bridg,
                            wfs_bridg,
                            vel_larg,
                            wfs_larg,
                            vel_tw,
                            wfs_tw,
                            vel_vazio,
                            p_entre_int_ext,
                            p_trigger_curta,
                            p_trigger_longa,
                            off_pause_cont,
                            off_pause_bridg,
                            off_pause_larg,
                            off_pause_tw,
                            on_pause_cont,
                            on_pause_bridg,
                            on_pause_larg,
                            on_pause_tw,
                            flag_path_type,
                        )
                        # output += f"G117 {{Trocou o perfil para {flag_path_type}}}\n"
                        last_flag = flag_path_type
                    desloc = np.subtract(coords, bfr)
                    dist = distance.euclidean(coords, bfr)
                    layer_tot_lenght += dist
                    output += (
                        f"G1 X{desloc[1] * mm_per_pixel} Y{desloc[0] * mm_per_pixel}\n"
                    )
                    # output += "M400\n"
                    bfr = coords
                    counter += 1
                    if flag_salto == 1:
                        output, flag_on = turn_on_UFSC(output, flag_on, on_pause)
                        flag_salto = 0
        output, flag_on = turn_off_UFSC(output, flag_on, off_pause)
        output = cleanning_position_UFSC(
            output, coords_corte, vel_vazio, p_entre_layers
        )
        output += ";____________________________________\n"
        output += f"G28 X0 Y0\n"
        print(f"Total travel distance {n_layer} = {layer_tot_lenght*mm_per_pixel}mm")
        print(
            f"Estimated time with speed {vel_cont}mm/min = {layer_tot_lenght*mm_per_pixel/vel_cont}min\n"
        )
    output += f"G1 Z20\n"
    output += f"G28 X0\n"
    output += f"G28 Y0\n"
    output += f"M104 S0; End of Gcode\n"
    os.chdir(folders.output)
    f = open(outFile, "w")
    f.write(output)
    f.close()
    os.chdir(folders.home)
    return
