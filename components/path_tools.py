from __future__ import annotations
import copy
from ctypes.wintypes import HSTR
from tkinter import X
from typing import TYPE_CHECKING, Dict, reveal_type
import itertools
import math
import random

from networkx import bridges
import numpy as np
import networkx as nx
import skimage
from components import points_tools as pt
from components import morphology_tools as mt
from components import images_tools as it
from components import skeleton as sk
from cv2 import boundingRect, arcLength, approxPolyDP
from scipy.spatial import distance_matrix, distance

if TYPE_CHECKING:
    from components.zigzag import ZigZag
    from typing import List
    from components.layer import Layer, Island
    from files import System_Paths
    from components.bottleneck import Bridge

"""Parte do código direcionado para grafos e sequencias determinadas de pontos"""


class Path:

    def __init__(self, name, seq, regions=None, img=None, saltos=None):
        if regions is None:
            regions = []
        if img is None:
            img = []
        if saltos is None:
            saltos = []
        self.name = name
        self.sequence = seq
        self.regions = regions
        self.img = img
        self.saltos = saltos
        return

    def get_img(self, base_frame):
        self.img = it.points_to_img(self.sequence, np.zeros(base_frame))
        return self.img

    def get_regions(self, island: Island):
        self.regions = {
            "offsets": [],
            "zigzags": [],
            "cross_over_bridges": [],
            "offset_bridges": [],
            "zigzag_bridges": [],
            "thin walls": [],
        }
        if hasattr(island, "offsets"):
            if len(island.offsets.regions):
                for o in island.offsets.regions:
                    if np.logical_and(o.img, self.img).any():
                        self.regions["offsets"].append(o.name)
        if hasattr(island, "zigzags"):
            if len(island.zigzags.regions):
                for z in island.zigzags.regions:
                    if np.logical_and(z.img, self.img).any():
                        self.regions["zigzags"].append(z.name)
        if hasattr(island, "bridges"):
            if len(island.bridges.cross_over_bridges):
                for cb in island.bridges.cross_over_bridges:
                    if np.logical_and(cb.img, self.img).any():
                        self.regions["cross_over_bridges"].append(cb.name)
        if hasattr(island, "bridges"):
            if len(island.bridges.offset_bridges):
                for ob in island.bridges.offset_bridges:
                    if np.logical_and(ob.img, self.img).any():
                        self.regions["offset_bridges"].append(ob.name)
        if hasattr(island, "bridges"):
            if len(island.bridges.zigzag_bridges):
                for zb in island.bridges.zigzag_bridges:
                    if np.logical_and(zb.img, self.img).any():
                        self.regions["zigzag_bridges"].append(zb.name)
        if hasattr(island, "thin"):
            if len(island.thin_walls.regions):
                for tw in island.thin_walls.regions:
                    if np.logical_and(tw.img, self.img).any():
                        self.regions["thin walls"].append(tw.name)
        return


def add_routes_by_sequence(
    nova_rota,
    island: Island,
    interruption_points,
    order_crossover_regions,
    cross_overs_included,
    offssets_included,
    saltos,
):
    indexes = []
    for i, int_pt in enumerate(interruption_points):
        idx_on_route = nova_rota.index(int_pt)
        name_of_cob = order_crossover_regions[i]
        cross_overs_included.add(name_of_cob)
        i_b_cob = island.bridges.cross_over_bridges
        cob_names = [x.name for x in i_b_cob]
        references_a = i_b_cob[cob_names.index(name_of_cob)].reference_points
        references_b = i_b_cob[cob_names.index(name_of_cob)].reference_points_b
        distances_a = [
            pt.distance_pts(references_a[0], int_pt),
            pt.distance_pts(references_a[1], int_pt),
        ]
        distances_b = [
            pt.distance_pts(references_b[0], int_pt),
            pt.distance_pts(references_b[1], int_pt),
        ]
        min_dists = [np.min(distances_a), np.min(distances_b)]
        rota_ponte = np.zeros_like(island.img)
        refs = []
        dists = []
        if np.argmin(min_dists) == 0:
            rota_ponte = i_b_cob[cob_names.index(name_of_cob)].route
            refs = references_a
            dists = distances_a
        if np.argmin(min_dists) == 1:
            rota_ponte = i_b_cob[cob_names.index(name_of_cob)].route_b
            refs = references_b
            dists = distances_b
        linked_offset = list(
            filter(
                lambda x: x != "Reg_000",
                i_b_cob[cob_names.index(name_of_cob)].linked_offset_regions,
            )
        )[0]
        A = [
            (linked_offset in y)
            for y in [x.regions["offsets"] for x in island.external_tree_route]
        ]
        linked_offset_seq = island.external_tree_route[A.index(True)].sequence
        pt_close_to_start, _ = pt.closest_point(
            refs[np.argmax(dists)], linked_offset_seq
        )
        linked_offset_seq = set_first_pt_in_seq(linked_offset_seq, pt_close_to_start)
        offssets_included = offssets_included.union(
            island.external_tree_route[A.index(True)].regions["offsets"]
        )
        indexes.append(idx_on_route)
        linked_bridge_seq = img_to_chain(rota_ponte.astype(np.uint8))[0]
        if len(linked_bridge_seq) < 2:
            linked_bridge_seq = img_to_chain(
                it.take_the_bigger_area(rota_ponte.astype(np.uint8))
            )[0]
            print("sdasda")
        linked_bridge_seq = set_first_pt_in_seq(
            linked_bridge_seq, list(refs[np.argmax(dists)])
        )
        linked_bridge_seq = cut_repetition(linked_bridge_seq)
        nova_rota = (
            nova_rota[:idx_on_route]
            + linked_offset_seq
            + linked_bridge_seq
            + nova_rota[idx_on_route:]
        )
        saltos.append(nova_rota[idx_on_route])
        print("salto: ", nova_rota[idx_on_route])
    return nova_rota, cross_overs_included, offssets_included, saltos


def add_routes_by_sequence_internal(
    nova_rota,
    island: Island,
    interruption_points,
    order_zigzag_bridges_regions,
    zigzag_bridges_included,
    zigzags_included,
    saltos,
):
    indexes = []
    for i, int_pt in enumerate(interruption_points):
        idx_on_route = nova_rota.index(int_pt)
        name_of_zigzag_bridge = order_zigzag_bridges_regions[i]
        i_b_zb = island.bridges.zigzag_bridges
        i_b_zb_names = [x.name for x in i_b_zb]
        references_a = i_b_zb[
            i_b_zb_names.index(name_of_zigzag_bridge)
        ].reference_points
        distances_a = [
            pt.distance_pts(references_a[0], int_pt),
            pt.distance_pts(references_a[1], int_pt),
        ]
        rota_ponte = i_b_zb[i_b_zb_names.index(name_of_zigzag_bridge)].route
        refs = references_a
        dists = distances_a
        linked_zigzag = list(
            filter(
                lambda x: not (x in zigzags_included),
                i_b_zb[i_b_zb_names.index(name_of_zigzag_bridge)].linked_zigzag_regions,
            )
        )[0]
        A = [
            (linked_zigzag in y)
            for y in [x.regions["zigzags"] for x in island.internal_tree_route]
        ]
        linked_zigzag_seq = island.internal_tree_route[A.index(True)].sequence
        pt_close_to_start, _ = pt.closest_point(
            refs[np.argmax(dists)], linked_zigzag_seq
        )
        linked_zigzag_seq = set_first_pt_in_seq(linked_zigzag_seq, pt_close_to_start)
        zigzags_included = zigzags_included.union(
            island.internal_tree_route[A.index(True)].regions["zigzags"]
        )
        indexes.append(idx_on_route)
        linked_bridge_seq = img_to_chain(rota_ponte.astype(np.uint8))[0]
        linked_bridge_seq = set_first_pt_in_seq(
            linked_bridge_seq, list(refs[np.argmax(dists)])
        )
        linked_bridge_seq = cut_repetition(linked_bridge_seq)
        nova_rota = (
            nova_rota[:idx_on_route]
            + linked_zigzag_seq
            + linked_bridge_seq
            + nova_rota[idx_on_route:]
        )
        saltos.append(nova_rota[idx_on_route])
        print("salto: ", nova_rota[idx_on_route])
        zigzag_bridges_included.add(name_of_zigzag_bridge)
    return nova_rota, zigzag_bridges_included, zigzags_included, saltos


def connect_thin_walls(island: Island, path_radius_external):
    new_route = Path("thin wall tree", [], [], saltos=[])
    if hasattr(island.thin_walls, "all_origins"):
        if np.sum(island.thin_walls.all_origins):
            thinwall_list, _, _ = it.divide_by_connected(island.thin_walls.all_origins)
            thinwall_path_list = []
            for i, tw in enumerate(thinwall_list):
                tw, _, _ = sk.create_prune_divide_skel(tw, path_radius_external)
                tw_path = img_to_chain(tw.astype(np.uint8))[0]
                one_of_the_tips = pt.x_y_para_pontos(
                    np.nonzero(mt.hitmiss_ends_v2(tw))
                )[0]
                tw_path = set_first_pt_in_seq(tw_path, one_of_the_tips)
                tw_path = cut_repetition(tw_path)
                thinwall_path_list.append(Path(i, tw_path, img=tw))
                thinwall_path_list[-1].get_regions(island)
            nova_rota = []
            saltos = []
            thinwalls_included = []
            for tw_p in thinwall_path_list:
                saltos.append(tw_p.sequence[-1])
                nova_rota = nova_rota + tw_p.sequence
                thinwalls_included = thinwalls_included + tw_p.regions["thin walls"]
            new_regions = {
                "offsets": [],
                "zigzags": [],
                "cross_over_bridges": [],
                "offset_bridges": [],
                "zigzag_bridges": [],
                "thin walls": list(thinwalls_included),
            }
            new_route = Path("thin wall tree", nova_rota, new_regions, saltos=saltos)
    return new_route

def middle_of_the_line(line_img):
    seq = make_a_chain(line_img, pt.img_to_points(mt.hitmiss_ends_v2(line_img))[0])
    # seq = set_first_pt_in_seq(seq, pt.img_to_points(mt.hitmiss_ends_v2(line_img))[0])
    return pt.invert_x_y([seq[int(len(seq)/2)]])[0]


def connect_cross_over_bridges(island: Island) -> Path:
    def find_interruption_points(
        island,
        nova_rota,
        cross_overs_included,
        offssets_included,
        order_crossover_regions,
        pts_valido_comeco
    ):
        closest_points = {}
        closest_centers = []
        flag_first = len(pts_valido_comeco)
        for bridge in island.bridges.cross_over_bridges:
            if not (bridge.name in list(cross_overs_included)):
                if set(bridge.linked_offset_regions).intersection(offssets_included):
                    A = bridge.pontos_extremos
                    closest_a = pt.closest_point(A[0], nova_rota)
                    closest_b = pt.closest_point(A[1], nova_rota)
                    closest_c = pt.closest_point(A[2], nova_rota)
                    closest_d = pt.closest_point(A[3], nova_rota)
                    cp = [closest_a, closest_b, closest_c, closest_d]
                    cp.sort(key=lambda x: x[1])
                    cp = cp[:2]
                    cp = [x[0] for x in cp]
                    if cp[0] == cp[1]:
                        nova_rota.remove(cp[0])
                        A = bridge.pontos_extremos
                        closest_a = pt.closest_point(A[0], nova_rota)
                        closest_b = pt.closest_point(A[1], nova_rota)
                        closest_c = pt.closest_point(A[2], nova_rota)
                        closest_d = pt.closest_point(A[3], nova_rota)
                        cp = [closest_a, closest_b, closest_c, closest_d]
                        cp.sort(key=lambda x: x[1])
                        cp = cp[:2]
                        cp = [x[0] for x in cp]
                    closest_points[str(bridge.name)] = cp
                    origin_mid = middle_of_the_line(bridge.origin)
                    closest_mid, _ = pt.closest_point(origin_mid, nova_rota)
                    closest_centers.append(closest_mid)
        special = []
        for k in closest_points.values():
            special = special + k
        interruption_points = []
        flags = np.zeros(len(closest_points.keys()))
        flag_valido_comeco = 0
        for point in nova_rota:
            if point in special:
                flag_valido_comeco += 1
                for i, cp in enumerate(list(closest_points.values())):
                    if point in cp:
                        flags[i] += 1
                        if flags[i] == 2:
                            B = [(point in x) for x in list(closest_points.values())]
                            idx = B.index(True)
                            order_crossover_regions.append(
                                list(closest_points.keys())[idx]
                            )
                            interruption_points.append(point)
            if (flags == 1).any() and not flag_first:
                pts_valido_comeco.append(point)
            # if flag_valido_comeco%2 == 0 and not flag_first:
            #     pts_valido_comeco.append(point)
        if not flag_first:
            aaaa = it.points_to_img(pts_valido_comeco, np.zeros_like(island.img))
            bbbb = it.sum_imgs([aaaa, it.points_to_img(closest_centers, np.zeros_like(island.img))])
            if pt.list_inside_list(closest_centers,pts_valido_comeco):
                pts_valido_comeco = list(filter(lambda x: x not in pts_valido_comeco, nova_rota))
                cccc = it.points_to_img(pts_valido_comeco, np.zeros_like(island.img))
                dddd = it.sum_imgs([cccc, it.points_to_img(closest_centers, np.zeros_like(island.img))])
        return interruption_points, order_crossover_regions, pts_valido_comeco

    start_path = list(
        filter(lambda x: "Reg_000" in x.regions["offsets"], island.external_tree_route)
    )[0]
    offssets_included = set(start_path.regions["offsets"])
    cross_overs_included = set(start_path.regions["cross_over_bridges"])
    offset_bridges_included = set(start_path.regions["offset_bridges"])
    all_its = []
    pts_valido_comeco = []
    if hasattr(island, "bridges"):
        rota_antiga = start_path.sequence.copy()
        nova_rota = []
        stop = 0
        saltos = []
        while not stop:
            order_crossover_regions = []
            interruption_points, order_crossover_regions, pts_valido_comeco = find_interruption_points(
                island,
                rota_antiga,
                cross_overs_included,
                offssets_included,
                order_crossover_regions,
                pts_valido_comeco
            )
            all_its = all_its + interruption_points
            if len(interruption_points) > 0:
                nova_rota, cross_overs_included, offssets_included, saltos = (
                    add_routes_by_sequence(
                        rota_antiga,
                        island,
                        interruption_points,
                        order_crossover_regions,
                        cross_overs_included,
                        offssets_included,
                        saltos,
                    )
                )
                rota_antiga = nova_rota
            else:
                stop = 1
    else:
        nova_rota = start_path.sequence
        saltos = []
    new_regions = {
        "offsets": list(offssets_included),
        "zigzags": [],
        "cross_over_bridges": list(cross_overs_included),
        "offset_bridges": list(offset_bridges_included),
        "zigzag_bridges": [],
    }
    if len(list(cross_overs_included)) > 0:
        cccc = it.points_to_img(pts_valido_comeco, np.zeros_like(island.img))
        new_start = random.choice(pts_valido_comeco)
        nova_rota = set_first_pt_in_seq(nova_rota, new_start)
    new_route = Path("exterior tree", nova_rota, new_regions, saltos=saltos)
    # aaa = new_route.get_img()
    return new_route


def connect_internal_external(island: Island, path_radius_internal):
    filling = island.zigzags.all_zigzags
    if np.sum(filling) > 0:
        most_external = island.offsets.regions[0].route.astype(np.uint8)
        dilation_kernel = int(path_radius_internal * 2)
        touching = np.zeros_like(island.img)
        while np.sum(touching) == 0:
            aaa = it.sum_imgs(
                [filling, mt.dilation(most_external, kernel_size=dilation_kernel)]
            )
            touching = aaa == 2
            dilation_kernel = dilation_kernel + 2
        candidates_internal = pt.img_to_points(touching)
        chosen_internal = random.choice(candidates_internal)
        external_pts = pt.img_to_points(most_external)
        chosen_external, _ = pt.closest_point(chosen_internal, external_pts)
    elif hasattr(island, "bridges"):
        filling = [x.route for x in island.bridges.zigzag_bridges]
        if len(filling) > 0:
            filling = it.sum_imgs(filling)
            print("FAZER A PARTE QUE TEM SÓ PONTES DE ZZ")
        else:
            most_external = island.offsets.regions[0].route.astype(np.uint8)
            external_pts = pt.img_to_points(most_external)
            chosen_external = random.choice(external_pts)
            chosen_internal = []
    return chosen_external, chosen_internal


def connect_offset_bridges(
    island: Island, base_frame, mask_3_4, path_radius_external
) -> Path:
    def integrate_bridge(todas_espirais, path_radius, pontos_extremos, base_frame):

        def filtrar_pontos(y_val, x_val):
            validos_no_y = [p for p in todas_espirais_points if p[0] == y_val]
            validos_no_yx = [
                p for p in validos_no_y if p[1] <= max(x_val) and p[1] >= min(x_val)
            ]
            return validos_no_yx

        def find_contacts(pontos, condicao):
            contacts = [p for p in pontos if condicao(p[1])]
            if len(contacts) == 0:
                contacts = pontos
            return contacts

        def closest_to_center(pontos):
            return pontos[np.argmin([pt.distance_pts(midle_point, p) for p in pontos])]

        y_da_ponte = pontos_extremos[0][0]
        y_de_cima = y_da_ponte - path_radius
        y_de_baixo = y_da_ponte + path_radius
        x_pontos_extremos = [x[1] for x in pontos_extremos]
        midle_point = [
            y_da_ponte,
            int((x_pontos_extremos[0] + x_pontos_extremos[1]) / 2),
        ]
        limites_x = [
            min(x_pontos_extremos) - 2 * path_radius,
            max(x_pontos_extremos) + 2 * path_radius,
        ]
        todas_espirais_points = pt.x_y_para_pontos(np.nonzero(todas_espirais))
        same_y_up_inside = filtrar_pontos(y_de_cima, limites_x)
        if len(same_y_up_inside) < 2:
            same_y_up_inside = filtrar_pontos(y_de_cima + 1, limites_x)
            if len(same_y_up_inside) < 2:
                same_y_up_inside = filtrar_pontos(y_de_cima + 2, limites_x)
        same_y_down_inside = filtrar_pontos(y_de_baixo, limites_x)
        if len(same_y_down_inside) < 2:
            same_y_down_inside = filtrar_pontos(y_de_baixo - 1, limites_x)
            if len(same_y_down_inside) < 2:
                same_y_down_inside = filtrar_pontos(y_de_baixo - 2, limites_x)
        contact_ec = find_contacts(same_y_up_inside, lambda x: x < midle_point[1])
        contact_dc = find_contacts(same_y_up_inside, lambda x: x > midle_point[1])
        contact_eb = find_contacts(same_y_down_inside, lambda x: x < midle_point[1])
        contact_db = find_contacts(same_y_down_inside, lambda x: x > midle_point[1])
        ponto_esq_cima = closest_to_center(contact_ec)
        ponto_dir_cima = closest_to_center(contact_dc)
        ponto_esq_baixo = closest_to_center(contact_eb)
        ponto_dir_baixo = closest_to_center(contact_db)
        linha_cima = it.draw_line(np.zeros(base_frame), ponto_esq_cima, ponto_dir_cima)
        linha_baixo = it.draw_line(
            np.zeros(base_frame), ponto_esq_baixo, ponto_dir_baixo
        )
        retangulo = it.draw_polyline(
            np.zeros(base_frame),
            [ponto_esq_cima, ponto_dir_cima, ponto_dir_baixo, ponto_esq_baixo],
            1,
        )
        retangulo = it.fill_internal_area(retangulo, np.ones_like(retangulo))
        new_todas_espirais = np.logical_and(todas_espirais, np.logical_not(retangulo))
        new_todas_espirais = it.sum_imgs([new_todas_espirais, linha_baixo, linha_cima])
        cleaned_new_todas_espirais, _, _ = sk.create_prune_divide_skel(
            new_todas_espirais, path_radius
        )
        return cleaned_new_todas_espirais

    def integrate_contact(todas_espirais, path_radius, bridge: Bridge, base_frame):
        route = bridge.route
        eraser = mt.dilation(bridge.origin, kernel_size=path_radius - 2)
        aaa = it.sum_imgs([route, todas_espirais])
        new_todas_espirais = it.image_subtract(aaa, eraser)
        A, _, _ = sk.create_prune_divide_skel(new_todas_espirais, path_radius)
        return A

    lista_de_rotas = []
    todas_espirais_img = np.zeros(base_frame)
    for region in island.offsets.regions:
        todas_espirais_img = np.logical_or(todas_espirais_img, region.route)
    if hasattr(island, "bridges"):
        for bridge in island.bridges.offset_bridges:
            pontos_extremos = pt.x_y_para_pontos(
                np.nonzero(mt.hitmiss_ends_v2(bridge.origin))
            )
            if bridge.type == "common_offset_bridge":
                todas_espirais_img = integrate_bridge(
                    todas_espirais_img,
                    path_radius_external,
                    pontos_extremos,
                    base_frame,
                )
            elif bridge.type == "contact_offset_bridge":
                todas_espirais_img = integrate_contact(
                    todas_espirais_img,
                    path_radius_external,
                    bridge,
                    base_frame,
                )
    rotas_isoladas = img_to_chain(todas_espirais_img.astype(np.uint8))
    lens = [len(x) for x in rotas_isoladas]
    circunf = 2 * 3.14 * path_radius_external
    for i, rota in enumerate(rotas_isoladas):
        if lens[i] > 2 * circunf:
            lista_de_rotas.append(Path(i, rota))
            lista_de_rotas[-1].sequence = set_first_pt_in_seq(
                lista_de_rotas[-1].sequence,
                list(island.comeco_ext),
            )
            lista_de_rotas[-1].get_img(base_frame)
            lista_de_rotas[-1].get_regions(island)
    return lista_de_rotas


def connect_zigzag_bridges(island: Island):
    start_path = island.internal_tree_route[0]
    zigzags_included = set(start_path.regions["zigzags"])
    zigzag_bridges_included = set(start_path.regions["zigzag_bridges"])
    zigzag_bridges_number = len(island.bridges.zigzag_bridges)
    if zigzag_bridges_number == 0:
        nova_rota = start_path.sequence
        saltos = []
    else:
        rota_antiga = start_path.sequence.copy()
        nova_rota = []
        stop = 0
        saltos = []
        while not stop:
            order_zigzag_bridges_regions = []
            interruption_points, order_zigzag_bridges_regions = (
                find_interruption_points_v2(
                    island,
                    rota_antiga,
                    zigzag_bridges_included,
                    zigzags_included,
                    order_zigzag_bridges_regions,
                )
            )
            if len(interruption_points) > 0:
                nova_rota, zigzag_bridges_included, zigzags_included, saltos = (
                    add_routes_by_sequence_internal(
                        rota_antiga,
                        island,
                        interruption_points,
                        order_zigzag_bridges_regions,
                        zigzag_bridges_included,
                        zigzags_included,
                        saltos,
                    )
                )
                rota_antiga = nova_rota
                # asfdfadsf = images_tools.points_to_img(nova_rota, np.zeros(island.base_frame))
            else:
                stop = 1
    new_regions = {
        "offsets": [],
        "zigzags": list(zigzags_included),
        "cross_over_bridges": [],
        "offset_bridges": [],
        "zigzag_bridges": list(zigzag_bridges_included),
    }
    new_route = Path("interior tree", nova_rota, new_regions, saltos=saltos)
    # aaa = new_route.get_img(island.img.shape)
    return new_route


def cut_repetition(seq):
    new_seq = []
    last = seq[0]
    for point in seq:
        dist = pt.distance_pts(point, last)
        if (point in new_seq) or (dist > 1.5):
            pass
        else:
            new_seq.append(point)
            last = point
    return new_seq


def draw_interface(composed_img, base_frame, jump):
    interface_img = np.zeros(base_frame)
    for y, line in enumerate(composed_img):
        for x, pixel in enumerate(line):
            if (
                pixel == 1
                and x != 0
                and x != base_frame[1] - 1
                and y != 0
                and y != base_frame[0] - 1
            ):
                a2 = composed_img[y - jump][x]
                b1 = composed_img[y][x - jump]
                b3 = composed_img[y][x + jump]
                c2 = composed_img[y + jump][x]
                all_pixels = [a2, b1, b3, c2]
                if 2 in all_pixels:
                    interface_img[y][x] = 1
    return interface_img


def draw_the_links(
    zigzags, zigzags_mst, base_frame, interfaces, centers, path_radius_internal
):

    def perpendicular_on_point(line_img, center, base_frame, path_radius):
        n_contatos = 0
        img_points = mt.hitmiss_ends_v2(line_img)
        if np.sum(img_points) < 2:
            line_img, _, _ = sk.create_prune_divide_skel(line_img, path_radius)
            img_points = mt.hitmiss_ends_v2(line_img)
        [p1, p2] = pt.x_y_para_pontos(np.nonzero(img_points))
        p3 = [0, 0]
        p4 = [0, 0]
        overshoot = path_radius
        while n_contatos < 2:
            if p1[0] == p2[0]:
                p3 = [center[0] + overshoot, center[1]]
                p4 = [center[0] - overshoot, center[1]]
            elif p1[1] == p2[1]:
                p3 = [center[0], center[1] + overshoot]
                p4 = [center[0], center[1] - overshoot]
            else:
                slope = (p2[0] - p1[0]) / (p2[1] - p1[1])
                dy = math.sqrt(overshoot**2 / (slope**2 + 1))
                dx = -slope * dy
                p3[0] = int(center[0] + dy)
                p3[1] = int(center[1] + dx)
                p4[0] = int(center[0] - dy)
                p4[1] = int(center[1] - dx)
            link = it.draw_line(np.zeros(base_frame), p3, p4)
            contatos = np.logical_and(link, all_zigzags)
            n_contatos = np.sum(contatos)
            overshoot += int(path_radius / 2)
        return link

    all_zigzags = np.zeros(base_frame)
    for i in zigzags.regions:
        all_zigzags = np.add(all_zigzags, i.route)
    for i, line in enumerate(interfaces):
        edge_list = list(zigzags_mst.edges)
        occurence_edges = []
        for j in edge_list:
            occurence_edges = occurence_edges + list(j)
        bridges_present = list(
            filter(lambda x: x[0] == "b", np.unique(occurence_edges))
        )
        bridges_on_end = []
        for bridge in bridges_present:
            if occurence_edges.count(bridge) == 1:
                bridges_on_end.append(bridge)
        link = perpendicular_on_point(
            line, centers[i], base_frame, path_radius_internal
        )
        mask_line = np.zeros((path_radius_internal * 2, path_radius_internal * 2))
        mask_line[int(path_radius_internal) - 1] = 1
        mask_line[int(path_radius_internal)] = 1
        mask_line[int(path_radius_internal) + 1] = 1
        work_area = mt.dilation(link, mask_line)
        _, work_area_contour_img = mt.detect_contours(work_area, return_img=True)
        a, b, n_points = it.divide_by_connected(
            np.logical_and(work_area_contour_img, all_zigzags)
        )
        while n_points > 4:
            y = np.min(np.nonzero(work_area)[0])
            work_area[y] = 0
            _, work_area_contour_img = mt.detect_contours(work_area, return_img=True)
            _, _, n_points = it.divide_by_connected(
                np.logical_and(work_area_contour_img, all_zigzags)
            )
        interface_points = intersection_points_w_rectangle(
            work_area_contour_img, all_zigzags
        )
        if np.size(interface_points) < 8:
            idx_list = list(zigzags_mst.edges)[i]
            zigzag_1 = zigzags.regions[int(idx_list[0][1])].route
            zigzag_2 = zigzags.regions[int(idx_list[1][1])].route
            a = pt.img_to_points(zigzag_1)
            origin_pt = pt.closest_point(centers[i], a)
            b = pt.img_to_points(zigzag_2)
            destiny_pt = pt.closest_point(centers[i], b)
            link = it.draw_line(np.zeros(base_frame), origin_pt[0], destiny_pt[0])
            work_area = mt.dilation(link, kernel_img=mask_line)
            _, work_area_contour_img = mt.detect_contours(work_area, return_img=True)
            interface_points = intersection_points_w_rectangle(
                work_area_contour_img, all_zigzags
            )
        intersection_pol = it.draw_polyline(
            np.zeros(base_frame), interface_points, True
        )
        intersection_pol = it.fill_internal_area(intersection_pol, np.ones(base_frame))
        rectangle_contour = mt.detect_contours(intersection_pol)
        rectangle_contour = pt.contour_to_list(rectangle_contour)
        cut_rectangle = rectangle_cut(
            rectangle_contour,
            np.zeros(base_frame),
            interface_points,
            2,
            base_frame,
            mode=1,
        )
        all_zigzags = np.logical_and(all_zigzags, np.logical_not(work_area))
        all_zigzags = np.add(all_zigzags, cut_rectangle)
    return all_zigzags


def find_points_of_contact(
    edges, path_radius_internal, mask_full_int, zigzags: List[ZigZag]
):

    def dilate_and_search(a1, a2, grow):
        mask_line = np.zeros(np.add(mask_full_int.shape, [grow, grow]))
        mask_line[:, int(mask_line.shape[0] / 2)] = 1
        # a1_reg = zigzags[int(edge[0][1])]
        a1_vertical_trail = mt.dilation(a1.route, kernel_img=mask_line)
        # a2_reg = zigzags[int(edge[1][1])]
        a2_vertical_trail = mt.dilation(a2.route, kernel_img=mask_line)
        interface = np.add(a1_vertical_trail, a2_vertical_trail) == 2
        aaa = np.add(a1_vertical_trail, a2_vertical_trail)
        return interface

    def vert_connection(a1, a2):
        # a1 = zigzags[int(edge[0][1])]
        a1_trail = mt.dilation(a1.trail, kernel_size=1)
        # a2 = zigzags[int(edge[1][1])]
        a2_trail = mt.dilation(a2.trail, kernel_size=1)
        interface = np.add(a1_trail, a2_trail) == 2
        return interface

    interfaces = []
    centers = []
    interface_types = []
    translated_edges = [(f[0] + f[4:], e[0] + e[4:]) for f, e in edges]
    for edge in translated_edges:
        has_bridge = False
        type_a1 = edge[0][0]
        type_a2 = edge[1][0]
        if type_a1 == "z" and type_a2 == "z":
            a1 = zigzags[int(edge[0][1:])]
            a2 = zigzags[int(edge[1][1:])]
            grow = 1
            interface = dilate_and_search(a1, a2, 1)
            while np.sum(interface) == 0:
                grow = grow + 1
                interface = dilate_and_search(a1, a2, grow)
            separated, _, num = it.divide_by_connected(interface)
            if num > 1:
                sums = [np.sum(x) for x in separated]
                interface = separated[np.argmax(sums)]
            interface_pts = pt.x_y_para_pontos(np.nonzero(interface))
            center = pt.points_center(interface_pts)
            interfaces.append(interface)
            centers.append(center)
            interface_types.append(has_bridge)
    return interfaces, centers, interface_types


def find_interruption_points_v2(
    isl: Island,
    nova_rota,
    zigzag_bridges_included,
    zigzags_included,
    order_zigzag_bridges_regions,
):
    from components.points_tools import closest_point

    closest_points = {}
    for bridge in isl.bridges.zigzag_bridges:
        if not (bridge.name in zigzag_bridges_included):
            if set(bridge.linked_zigzag_regions).intersection(zigzags_included):
                A = bridge.pontos_extremos
                if len(bridge.route) > 0 and np.sum(A) != 0:
                    closest_a = closest_point(A[0], nova_rota)
                    closest_b = closest_point(A[1], nova_rota)
                    closest_c = closest_point(A[2], nova_rota)
                    closest_d = closest_point(A[3], nova_rota)
                    cp = [closest_a, closest_b, closest_c, closest_d]
                    cp.sort(key=lambda x: x[1])
                    cp = cp[:2]
                    cp = [x[0] for x in cp]
                    closest_points[bridge.name] = cp
    special = []
    for k in closest_points.values():
        special = special + k
    interruption_points = []
    flags = np.zeros(len(closest_points.keys()))
    for pt in nova_rota:
        if pt in special:
            for i, cp in enumerate(list(closest_points.values())):
                if pt in cp:
                    flags[i] += 1
                    if flags[i] == 2:
                        B = [(pt in x) for x in list(closest_points.values())]
                        idx = B.index(True)
                        order_zigzag_bridges_regions.append(
                            list(closest_points.keys())[idx]
                        )
                        interruption_points.append(pt)
    return interruption_points, order_zigzag_bridges_regions


def generate_guide_line(region, base_frame, prohibited_areas):
    """
    lembrando dos indices:
        _______1______
        |            |
        0            2
        |______3_____|
    """
    region.make_contour(base_frame)
    bound_box = boundingRect(region.area_contour[0])
    region.center_coords = pt.points_center(pt.contour_to_list(region.area_contour))
    end_of_lines = [
        [bound_box[0], region.center_coords[0]],
        [region.center_coords[1], bound_box[1]],
        [bound_box[0] + bound_box[2], region.center_coords[0]],
        [region.center_coords[1], bound_box[1] + bound_box[3]],
    ]
    end_of_lines = pt.invert_x_y(end_of_lines)
    candidates = end_of_lines.copy()
    master_line = []
    while np.sum(master_line) == 0:
        if candidates:
            closest_point_indx = np.argmin(
                distance_matrix([list(region.center_coords)], candidates, 2)
            )
            closest_point = random.choice(candidates)
            line = it.draw_line(
                np.zeros(base_frame), region.center_coords, closest_point
            )
            dilated_line = mt.dilation(line, kernel_size=16)
            master_line = line
        else:
            candidates = end_of_lines.copy()
            closest_point_indx = np.argmin(
                distance_matrix([list(region.center_coords)], candidates, 2)
            )
            closest_point = candidates[closest_point_indx]
            master_line = it.draw_line(
                np.zeros(base_frame), region.center_coords, closest_point
            )
    return master_line, end_of_lines.index(closest_point)


def img_to_chain(img: np.ndarray, init_area=None, minimal_seq: int = 0):
    if init_area is None:
        init_area = []
    contours = mt.detect_contours(img.astype(np.uint8))
    multiple_lines = pt.multiple_contours_to_list(contours, minimal_seq)
    if len(multiple_lines) > 1:
        multiple_lines = remove_repeated_contours(multiple_lines, img.shape)
    if len(init_area) > 0:
        for idx, contour in enumerate(contours):
            init_area_pts = pt.img_to_points(init_area)
            point = [contour[0].tolist()[0][1], contour[0].tolist()[0][0]]
            if point in init_area_pts:
                multiple_lines = multiple_lines[idx:] + multiple_lines[:idx]
    for l, line in enumerate(multiple_lines):
        max_y = np.max([x[0] for x in line])
        bottom_pts = list(filter(lambda x: x[0] == max_y, line))
        max_x = np.max([x[1] for x in bottom_pts])
        start_pt_idx = line.index([max_y, max_x])
        linha_com_comeco_certo = line[start_pt_idx:] + line[:start_pt_idx]
        if len(linha_com_comeco_certo) > 1:
            if linha_com_comeco_certo[1][0] == max_y:
                # print("reverti um deles!")
                linha_com_comeco_certo.reverse()
        multiple_lines[l] = linha_com_comeco_certo
    return multiple_lines


def img_to_graph(im):
    hy, hx = np.where(im[1:] & im[:-1])  # horizontal edge start positions
    h_units = np.array([hx, hy]).T
    h_starts = [tuple(n) for n in h_units]
    h_ends = [
        tuple(n) for n in h_units + (0, 1)
    ]  # end positions = start positions shifted by vector (1,0)
    horizontal_edges = zip(h_starts, h_ends)
    # CONSTRUCTION OF VERTICAL EDGES
    vy, vx = np.where(im[:, 1:] & im[:, :-1])  # vertical edge start positions
    v_units = np.array([vx, vy]).T
    v_starts = [tuple(n) for n in v_units]
    v_ends = [
        tuple(n) for n in v_units + (1, 0)
    ]  # end positions = start positions shifted by vector (0,1)
    vertical_edges = zip(v_starts, v_ends)
    # CONSTRUCTION OF POSITIVE DIAGONAL EDGES
    pdy, pdx = np.where(
        im[1:][:, 1:] & im[:-1][:, :-1]
    )  # vertical edge start positions
    pd_units = np.array([pdx, pdy]).T
    pd_starts = [tuple(n) for n in pd_units]
    pd_ends = [
        tuple(n) for n in pd_units + (1, 1)
    ]  # end positions = start positions shifted by vector (1,1)
    positive_diagonal_edges = zip(pd_starts, pd_ends)
    # CONSTRUCTION OF NEGATIVE DIAGONAL EDGES
    ndy, ndx = np.where(
        im[:-1][:, 1:] & im[1:][:, :-1]
    )  # vertical edge start positions
    ndx = ndx + 1
    nd_units = np.array([ndx, ndy]).T
    nd_starts = [tuple(n) for n in nd_units]
    nd_ends = [
        tuple(n) for n in nd_units + (-1, 1)
    ]  # end positions = start positions shifted by vector (-1,1)
    negative_diagonal_edges = zip(nd_starts, nd_ends)
    G = nx.Graph()
    G.add_edges_from(horizontal_edges, weight=1)
    G.add_edges_from(vertical_edges, weight=1)
    G.add_edges_from(positive_diagonal_edges, weight=1)
    G.add_edges_from(negative_diagonal_edges, weight=1)
    return G


def img_to_graph_com_distancias(im):
    image_array = np.array(im)
    # Create an empty graph
    G = nx.Graph()
    # Get the dimensions of the image
    rows, cols = image_array.shape
    # Define the offsets for the 8-neighborhood
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # Add nodes for each non-zero pixel
    for i in range(rows):
        for j in range(cols):
            if image_array[i, j] != 0:
                G.add_node((i, j), value=image_array[i, j])
                # Connect to neighbors
                for dy, dx in offsets:
                    ni, nj = i + dy, j + dx
                    # Check if the neighbor is within bounds and is non-zero
                    if 0 <= ni < rows and 0 <= nj < cols and image_array[ni, nj] != 0:
                        weight = abs(image_array[i, j] - image_array[ni, nj])
                        G.add_edge((i, j), (ni, nj), weight=weight)
    return G


def intersection_points_w_rectangle(border, spiral, idx=0):
    # intersection = np.add(border, spiral)
    intersection = np.logical_and(border, spiral)
    considered = pt.img_to_points(intersection)
    pts = [[], [], [], []]
    sums = [x[0] + x[1] for x in considered]
    pts[0] = considered[np.argmin(sums)]
    pts[2] = considered[np.argmax(sums)]
    rest = list(filter(lambda x: not (x in pts), considered))
    difs_x_a = [abs(pts[0][1] - x[1]) for x in rest]
    pts[1] = rest[np.argmin(difs_x_a)]
    pts[3] = rest[np.argmax(difs_x_a)]
    return pts


def line_img_to_freeman_chain(img, origin_point):
    pontos_ctr = mt.detect_contours(img, only_external=True)
    pontos = pt.contour_to_list(pontos_ctr)
    pontos_org = set_first_pt_in_seq(pontos, origin_point)
    pontos_org = cut_repetition(pontos_org)
    return pontos_org


def make_offset_graph(filtered_regions, regs_touching):
    graph = nx.MultiGraph()
    for i in np.arange(0, len(filtered_regions)):
        graph.add_node(filtered_regions[i].name)
    for origem in graph.nodes:
        origem_num = int(origem.replace("Reg_", ""))
        for elem_paralelo in filtered_regions[origem_num].paralel_points:
            area_origem = origem_num = elem_paralelo.origin
            area_origem_num = int(elem_paralelo.origin.replace("Reg_", ""))
            area_destino = origem_num = elem_paralelo.destiny
            area_destino_num = int(elem_paralelo.destiny.replace("Reg_", ""))
            for i in np.arange(0, len(elem_paralelo.dist_a)):
                coord_origem = filtered_regions[area_origem_num].limmit_coords[0]
                graph.add_edge(
                    area_origem,
                    area_destino,
                    weight=elem_paralelo.dist_a[i],
                    coord_origem=coord_origem,
                    coord_destino=elem_paralelo.lista_a[i],
                    extremo_origem="a",
                )
            for i in np.arange(0, len(elem_paralelo.dist_b)):
                coord_origem = filtered_regions[area_origem_num].limmit_coords[1]
                graph.add_edge(
                    area_origem,
                    area_destino,
                    weight=elem_paralelo.dist_b[i],
                    coord_origem=coord_origem,
                    coord_destino=elem_paralelo.lista_b[i],
                    extremo_origem="b",
                )
            for i in np.arange(0, len(elem_paralelo.dist_c)):
                coord_origem = filtered_regions[area_origem_num].limmit_coords[2]
                graph.add_edge(
                    area_origem,
                    area_destino,
                    weight=elem_paralelo.dist_c[i],
                    coord_origem=coord_origem,
                    coord_destino=elem_paralelo.lista_c[i],
                    extremo_origem="c",
                )
            for i in np.arange(0, len(elem_paralelo.dist_d)):
                coord_origem = filtered_regions[area_origem_num].limmit_coords[3]
                graph.add_edge(
                    area_origem,
                    area_destino,
                    weight=elem_paralelo.dist_d[i],
                    coord_origem=coord_origem,
                    coord_destino=elem_paralelo.lista_d[i],
                    extremo_origem="d",
                )
    for origem_a, origem_b in regs_touching:
        region_a = [x for x in filtered_regions if x.name == origem_a][0]
        region_b = [x for x in filtered_regions if x.name == origem_b][0]
        region_a_all_loops = it.sum_imgs([x.route for x in region_a.loops])
        region_b_all_loops = it.sum_imgs([x.route for x in region_b.loops])
        coord_origem_a, coord_origem_b = it.closest_points_btwn_imgs(
            region_a_all_loops, region_b_all_loops
        )
        coord_origem_a = pt.invert_x_y([coord_origem_a])
        coord_origem_b = pt.invert_x_y([coord_origem_b])
        graph.add_edge(
            origem_a,
            origem_b,
            weight=0,
            coord_origem=coord_origem_a[0],
            coord_destino=coord_origem_b[0],
            extremo_origem="e",
        )
    return graph


def make_zigzag_graph(zigzag_regions, zigzags_bridges, base_frame):
    graph = nx.Graph()
    pos_zigzag_nodes = {}
    for i in zigzag_regions:
        new_center = i.center
        graph.add_node("z" + str(i.name))
        pos_zigzag_nodes.update({"z" + str(i.name): new_center})
    if not zigzags_bridges:
        reg_neig = it.neighborhood(zigzag_regions)
    else:
        reg_neig, _, comb_neig = it.neighborhood(zigzag_regions, zigzags_bridges)
        for j in zigzags_bridges:
            new_center = j.center
            graph.add_node("b" + str(j.name))
            pos_zigzag_nodes.update({"b" + str(j.name): new_center})
        for ligacao in comb_neig:
            graph.add_edge("z" + str(ligacao[0]), "b" + str(ligacao[1]), weight=2)
    for ligacao in reg_neig:
        graph.add_edge("z" + str(ligacao[0]), "z" + str(ligacao[1]), weight=1)
    return graph, pos_zigzag_nodes


def make_a_chain(image, start_point) -> list:
    im = copy.deepcopy(image)
    disconection = start_point
    im[disconection[0], disconection[1]] = 0
    [start, end] = pt.img_to_points(sk.find_tips(im.astype(bool)))
    G = img_to_graph(im)
    path = nx.shortest_path(G, source=tuple(np.flip(start)), target=tuple(np.flip(end)))
    path = path[1:-1]
    return path


def make_a_chain_open_segment(im, ext_point) -> list:
    [start, end] = ext_point
    G = img_to_graph(im)
    path = nx.shortest_path(G, source=tuple(np.flip(start)), target=tuple(np.flip(end)))
    return path


def organize_points_cw(pts, origin=[]):
    refvec = [0, 1]
    if not origin:
        origin = pt.points_center(pts)

    def clockwiseangle_and_distance(point):
        vector = [point[0] - origin[0], point[1] - origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = (
            refvec[1] * normalized[0] - refvec[0] * normalized[1]
        )  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        return angle, lenvector

    pts = pt.invert_x_y(pts)
    organized = sorted(pts, key=clockwiseangle_and_distance)
    organized = pt.invert_x_y(organized)
    return organized


def rotate_path_odd_layer(coords, base_frame):
    new_coords = []
    for p in coords:
        if p == [0, 0]:
            new_coords.append(p)
        else:
            (y, x) = p
            angler = (270) * math.pi / 180
            newx = int(x * math.cos(angler) - y * math.sin(angler))
            newy = int(x * math.sin(angler) + y * math.cos(angler)) + base_frame[1]
            new_coords.append([newy, newx])
    return new_coords


def one_pixel_wide(img):
    return skimage.morphology.thin(img, max_num_iter=None)


def rectangle_cut(contours, linha, points, n_loops, base_frame, mode=0, idx=0):
    fila = contours
    rotations = fila.index(points[0])
    fila = fila[rotations:] + fila[:rotations]  # garante que a fila começa pelo ponto A
    borda_cortada = np.zeros(base_frame)
    borda_normal = np.zeros(base_frame)
    counter = 0
    counter_pixels = 0
    for i in np.arange(0, len(fila)):
        borda_normal[fila[i][0]][fila[i][1]] = 1
        counter_pixels += 1
        y = fila[i][0]
        x = fila[i][1]
        pixel_linhas = linha[y][x]
        ca = [y, x] == points[0]
        cb = [y, x] == points[1]
        cc = [y, x] == points[2]
        cd = [y, x] == points[3]
        ce = pixel_linhas == 1
        # cf = n_loops == 2
        cg = n_loops % 2
        ch = idx == 3 or idx == 1
        if mode:  # versão zigzag
            if ca or cb or cc or cd:
                counter += 1
                borda_cortada[fila[i][0]][fila[i][1]] = 1
        else:  # versão espiral
            if ch:
                if (not cg) and (ce or cc or cd):  # par
                    counter += 1
                    borda_cortada[fila[i][0]][fila[i][1]] = 1
                elif cg and (ce or cc):  # impar
                    counter += 1
                    borda_cortada[fila[i][0]][fila[i][1]] = 1
            else:
                if (not cg) and (ce or ca or cd):  # par
                    counter += 1
                    borda_cortada[fila[i][0]][fila[i][1]] = 1
                elif cg and (ce or ca or cc):  # impar
                    counter += 1
                    borda_cortada[fila[i][0]][fila[i][1]] = 1
        if counter % 2 != 0:
            borda_cortada[fila[i][0]][fila[i][1]] = 1
    return borda_cortada


def regions_mst(regions_graph):
    inner_way = nx.algorithms.tree.minimum_spanning_tree(
        regions_graph, algorithm="prim"
    )
    return inner_way, inner_way.edges


def remove_repeated_contours(multiple_lines_lists, canvas_size):
    cleaned_multiple_lines = multiple_lines_lists.copy()
    cleaned_multiple_lines.sort(key=len)
    remove_idx = []
    for a, b in list(itertools.combinations(cleaned_multiple_lines, 2)):
        a_img = it.points_to_img(a, np.zeros(canvas_size))
        b_img = it.points_to_img(b, np.zeros(canvas_size))
        repetitions_percent = np.sum(np.logical_and(a_img, b_img)) / len(b)
        if repetitions_percent > 0.8:
            remove_idx.append(cleaned_multiple_lines.index(a))
    for i in sorted(remove_idx, reverse=True):
        del cleaned_multiple_lines[i]
    return cleaned_multiple_lines


def set_first_pt_in_seq(seq, first_point):
    fila = seq.copy()
    if not (first_point in seq):
        first_point, _ = pt.closest_point(first_point, seq)
        # first_point = [first_point[1],first_point[0]]
    rotations = fila.index(first_point)
    fila = fila[rotations:] + fila[:rotations]
    if pt.distance_pts(fila[0], fila[1]) >= 3:
        fila.reverse()
        fila = [fila[-1]] + fila[:-1]
    return fila


def simplifica_retas_master(seq_pts, factor_epilson, saltos):
    # if len(seq_pts) == 0:
    #     return []
    sequence = [list(x) for x in seq_pts]
    approx_seq = []
    if len(sequence) > 0:
        saltos = [list(x) for x in saltos]
        segmentos = np.array_split(
            sequence, np.where([(x in saltos) for x in sequence])[0]
        )
        for s in segmentos:
            candidates = []
            candidates.append(factor_epilson * arcLength(s, False))
            candidates.append(factor_epilson * arcLength(s, True))
            epsilon = candidates[np.argmax(candidates)]
            approx_seg = approxPolyDP(np.ascontiguousarray(s), epsilon, False)
            approx_seg = [list(x[0]) for x in approx_seg]
            if approx_seg is None:
                pass
            else:
                approx_seq += approx_seg
                # approx_seq += [["a", "a"]]
                approx_seq += [[0, 0]]
    return approx_seq


def spiral_cut(contours, spiral, points, n_loops, base_frame, idx):

    fila = contours.copy()
    fila = set_first_pt_in_seq(fila, points[0])
    ordem_na_fila_pontos = []
    for p in fila:
        if p in points:
            ordem_na_fila_pontos.append(p)
    ordem_na_fila_pontos_idx = [ordem_na_fila_pontos.index(x) for x in points]
    if ordem_na_fila_pontos_idx[1] == 3:
        fila.reverse()
        fila = set_first_pt_in_seq(fila, points[0])
    fila = contours.copy()
    rotations = fila.index(points[0])
    fila = fila[rotations:] + fila[:rotations]  # garante que a fila começa pelo ponto A
    borda_cortada = np.zeros(base_frame)
    borda_normal = np.zeros(base_frame)
    counter = 0
    counter_pixels = 0
    for i in np.arange(0, len(fila)):
        # borda_normal[fila[0][i][0][1]][fila[0][i][0][0]] = 1
        borda_normal[fila[i][0]][fila[i][1]] = 1
        counter_pixels += 1
        y = fila[i][0]
        x = fila[i][1]
        pixel_linhas = spiral[y][x]
        ca = [y, x] == points[0]
        cb = [y, x] == points[1]
        cc = [y, x] == points[2]
        cd = [y, x] == points[3]
        ce = pixel_linhas == 1
        cf = n_loops == 2
        cg = n_loops % 2
        if cf and ce:
            counter += 1
            borda_cortada[fila[i][0]][fila[i][1]] = 1
        elif idx % 2:  # idx indica que o corte é no topo ou em baixo
            if (not cg) and (ce and not (cd or cc)):  # par
                counter += 1
                borda_cortada[fila[i][0]][fila[i][1]] = 1
            elif cg and (ce and not (cc or ca)):  # impar
                counter += 1
                borda_cortada[fila[i][0]][fila[i][1]] = 1
        else:  # idx indica que o corte é nas laterais da figura
            if (not cg) and (ce and not (cd or ca)):  # par
                counter += 1
                borda_cortada[fila[i][0]][fila[i][1]] = 1
            elif cg and (ce and not (cc or ca)):  # impar
                counter += 1
                borda_cortada[fila[i][0]][fila[i][1]] = 1
        if counter % 2 != 0:
            borda_cortada[fila[i][0]][fila[i][1]] = 1
    if (idx % 2 == 0) and n_loops == 2:
        borda_cortada = np.logical_and(borda_normal, np.logical_not(borda_cortada))
    return borda_cortada


def start_internal_route(isl: Island, mask_full_int, path_radius_internal):
    path_list = []
    if hasattr(isl, "zigzags"):
        for i, ma in enumerate(isl.zigzags.macro_areas):
            zigzag_path = img_to_chain(ma.astype(np.uint8), isl.zigzags.regions[0].img)
            if len(zigzag_path) > 0:
                path_list.append(Path(i, zigzag_path[0], img=ma))
                path_list[-1].sequence = set_first_pt_in_seq(
                    path_list[-1].sequence, list(isl.comeco_int)
                )
                path_list[-1].get_regions(isl)
    else:
        if hasattr(isl, "bridges"):
            for i, zb in enumerate(isl.bridges.zigzag_bridges):
                zigzag_b_path = img_to_chain(
                    zb.astype(np.uint8), isl.bridges.zigzag_bridges[0].img
                )
                if len(zigzag_b_path) > 0:
                    path_list.append(Path(i, zigzag_b_path[0], img=zb))
                    path_list[-1].sequence = set_first_pt_in_seq(
                        path_list[-1].sequence, list(isl.comeco_int)
                    )
                    path_list[-1].get_regions(isl)
    return path_list


def layers_to_Gcode(
    layers: List[Layer],
    folders: System_Paths,
    vel_int,
    vel_ext,
    vel_vazio,
    vel_thin_wall,
    p_religamento,
    p_desligamento,
    p_entre_int_ext,
    p_entre_camadas,
    layer_heights,
    coords_substrato,
    coords_corte,
):
    import os

    def religamento(output):
        output += ";-------RELIGAMENTO------\n"
        output += "M42 P4 S0\n"
        output += f"G4 P{p_religamento}\n"
        output += ";------------------------\n"
        return output

    def desligamento(output):
        output += ";-------DESLIGAMENTO------\n"
        output += "M42 P4 S255\n"
        output += f"G4 P{p_desligamento}\n"
        output += ";-------------------------\n"
        return output

    def posicao_de_corte(output, coords):
        output += ";-------POS de CORTE------\n"
        output += f";POS de Corte\n"
        output += f"G90\n"
        output += f"G0 Y{coords[0]} F{vel_vazio}\n"
        output += f"M400\n"
        output += f"G0 x{coords[1]} F{vel_vazio}\n"
        output += f"M400\n"
        output += f"G4 P{p_entre_camadas}\n"
        output += f"G91\n"
        output += ";------------------------\n"
        return output

    def posicao_inicial(output, coords):
        output += f";_______LAYER{n_layer + 1}_____\n"
        # output += f"G1 F{vel_ext}; speed g1\n"
        output += f"G90\n"
        output += f"G1 Z{layer_heights[n_layer]} ; Camada + 10mm\n"
        output += f"G1 X{coords[1]} Y{coords[0]} F{vel_vazio}; POS INICIAL\n"
        output += f"G91\n"
        return output

    mm_per_pixel = layers[0].mm_per_pxl
    # layer_height = layers[0].layer_height
    outFile = f"{folders.selected}.gcode"
    output = ""
    # output += f";Layer height: {layer_height}\n"
    output += f";DPI: {layers[0].dpi}\n"
    output += f";Camadas: {layers[0].n_camadas}\n"
    output += f";void_max: {layers[0].void_max}\n"
    output += f";max_internal_walls: {layers[0].max_internal_walls}\n"
    output += f";max_external_walls: {layers[0].max_external_walls}\n"
    output += f";max_external_walls: {layers[0].max_external_walls}\n"
    output += f";path_radius: {layers[0].path_radius_internal}\n"
    output += f";n_max: {layers[0].n_max}\n"
    output += f"G91\n"
    output += f"M42 P4 S255; turn off welder\n"
    output += f"G28 X0 Y0 Z0\n"
    output += f"G1 F360; speed g1\n"
    bfr = [0, 0]
    base_frame = layers[0].base_frame

    for n_layer, layer in enumerate(layers):
        soma_do_deslocamento = 0
        output = posicao_inicial(output, coords_substrato)
        bfr = coords_substrato
        folders.load_islands_hdf5(layer)
        # print(f"nome: {layer.name}")
        for n_island, island in enumerate(layer.islands):
            folders.load_island_paths_hdf5(layer.name, island)
            folders.load_island_paths_hdf5(layer.name, island)
            itr = [list(x) for x in island.internal_tree_route.sequence]
            etr = [list(x) for x in island.external_tree_route.sequence]
            twtr = [list(x) for x in island.thinwalls_tree_route.sequence]
            itr_img = it.points_to_img(itr, np.zeros(layer.base_frame))
            etr_img = it.points_to_img(etr, np.zeros(layer.base_frame))
            twtr_img = it.points_to_img(twtr, np.zeros(layer.base_frame))
            folders.load_bridges_hdf5(layer.name, island)
            print(f"nome: {layer.name}/{island.name}")
            aaa = island.island_route.img
            pts_crossover = []
            if hasattr(island, "bridge"):
                A1 = [
                    pt.img_to_points(x.route) for x in island.bridges.cross_over_bridges
                ]
                A2 = [
                    pt.img_to_points(x.route_b)
                    for x in island.bridges.cross_over_bridges
                ]
                A = A1 + A2
                for x in A:
                    pts_crossover = pts_crossover + x
            if n_layer % 2:
                etr = rotate_path_odd_layer(etr, layer.base_frame)
                itr = rotate_path_odd_layer(itr, layer.base_frame)
                twtr = rotate_path_odd_layer(twtr, layer.base_frame)
                itr_img = it.points_to_img(
                    itr, np.zeros(pt.invert_x_y([layer.base_frame])[0])
                )
                etr_img = it.points_to_img(
                    etr, np.zeros(pt.invert_x_y([layer.base_frame])[0])
                )
                twtr_img = it.points_to_img(
                    twtr, np.zeros(pt.invert_x_y([layer.base_frame])[0])
                )
                aaaa = it.points_to_img(pts_crossover, np.zeros(layer.base_frame))
                pts_crossover = rotate_path_odd_layer(pts_crossover, layer.base_frame)
                aaaa = it.points_to_img(
                    pts_crossover, np.zeros(pt.invert_x_y([layer.base_frame])[0])
                )
            pontos_int = [list(x) for x in itr] + pts_crossover
            pontos_ext = [list(x) for x in etr]
            pontos_ext = [coord for coord in pontos_ext if coord not in pts_crossover]
            pontos_tw = [list(x) for x in island.thinwalls_tree_route.sequence]
            chain = [list(x) for x in island.island_route.sequence]
            counter = 0
            flag_salto = 0
            flag_path_type = 99
            last_flag = 0
            print(chain)
            for i, p in enumerate(chain):
                if p == [0, 0]:
                    output = desligamento(output)
                    flag_salto = 1
                else:
                    coords = p
                    coords = [
                        base_frame[0] - coords[0] + coords_substrato[0],
                        coords[1] + coords_substrato[1],
                    ]
                    if p in pontos_int:
                        flag_path_type = 0
                        vel = vel_int
                        texto_mudanca = ";----Interno----\n"
                    elif p in pontos_ext:
                        flag_path_type = 1
                        vel = vel_ext
                        texto_mudanca = ";----Externo----\n"
                    elif p in pontos_tw:
                        flag_path_type = 2
                        vel = vel_thin_wall
                        texto_mudanca = ";----ThinWalls----\n"
                    else:
                        flag_path_type = 3
                        vel = vel_ext
                        texto_mudanca = ";----perdido----\n"
                    if i == 1:
                        output = religamento(output)
                    if flag_path_type != last_flag:
                        output += f"G1 F{vel}; speed g1\n"
                        last_flag = flag_path_type
                        print(f"trocou para {flag_path_type}")
                        output += texto_mudanca
                    desloc = np.subtract(coords, bfr)
                    dist = distance.euclidean(coords, bfr)
                    soma_do_deslocamento += dist
                    output += (
                        f"G1 X{desloc[1] * mm_per_pixel} Y{desloc[0] * mm_per_pixel}\n"
                    )
                    output += "M400\n"
                    bfr = coords
                    counter += 1
                    if flag_salto == 1:
                        output = religamento(output)
                        flag_salto = 0
            output = posicao_de_corte(output, coords_corte)
            output += ";____________________________________\n"
            output += f"G28 X0 Y0\n"
            print(
                f"Deslocamento total da camada {n_layer} = {soma_do_deslocamento*mm_per_pixel}mm"
            )
            print(
                f"Tempo estimado com Vel={vel_ext}mm/min = {soma_do_deslocamento*mm_per_pixel/vel_ext}min\n"
            )
            # os.chdir(folders.output)
            # f = open(f"{layer.name}_{outFile}", "w")
            # f.write(output)
            # f.close()
            # os.chdir(folders.home)
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


def layers_to_Gcode_FFF(camadas: List[Layer], arquivos: System_Paths, file_name):
    # ISSO É PARA CURA
    import os
    import matplotlib.pyplot as plt

    os.chdir(arquivos.home)
    # img_name = arquivos.all_figs[7]
    recovered_img = np.zeros(np.multiply(camadas[0].original_img.shape, 3))
    inFile = file_name
    i = open(inFile, "r")
    coordinates = []
    extrusion = 0
    pixel_per_mm = camadas[0].dpi / 25.4
    this_point = []
    last_point = [0, 0]
    desloc = [200, 300]
    coordinates.append(last_point)
    last_e = 0
    start = False
    for linenumber, line in enumerate(i):
        if line == ";LAYER:2\n":
            start = True
        if start:
            if line == ";LAYER:3\n":
                break
            line = line.strip()
            if line.startswith("G1") or line.startswith("G0"):
                # print(line.split())
                data = line.split()
                if data[1][0] == "F":
                    adder = 1
                else:
                    adder = 0
                if data[1 + adder][0] == "X":
                    x = data[1 + adder][1:]
                    try:
                        x = [int(round(float(x) * pixel_per_mm))]
                    except:
                        x = []
                    y = data[2 + adder][1:]
                    try:
                        y = [int(round(float(y) * pixel_per_mm))]
                    except:
                        y = []
                    try:
                        e = [float(data[3 + adder][1:])]
                    except:
                        e = []
                    if len(x) > 0 and len(y) > 0:
                        this_point = np.subtract([*y, *x], desloc)
                        coordinates.append(this_point)
                        if len(coordinates) > 1:
                            if len(e) > 0:
                                this_e = e[0]
                                if data[3 + adder][0] == "E":
                                    if this_e > last_e:
                                        last_e = e[0]
                                        recovered_img = it.draw_line(
                                            recovered_img, last_point, this_point
                                        )
                                    else:
                                        last_e = 0
                                else:
                                    print(data)
                    last_point = np.subtract([*y, *x], desloc)
    i.close()
    plt.figure()
    plt.imshow(recovered_img)
