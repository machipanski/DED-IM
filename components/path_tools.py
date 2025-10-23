from __future__ import annotations
import itertools
import math
import random
import datetime
from components.zigzag import ZigZagRegions
import numpy as np
import math
import copy
import networkx as nx
import skimage
from re import L
from tkinter import X
from typing import TYPE_CHECKING
from cv2 import boundingRect, arcLength, approxPolyDP
from scipy.spatial import distance_matrix, distance
from components import points_tools as pt
from components import morphology_tools as mt
from components import images_tools as it
from components import skeleton as sk
import os

if TYPE_CHECKING:
    from components.zigzag import ZigZag
    from typing import List
    from components.layer import Layer, Island
    from components.files import System_Paths
    from components.bottleneck import Bridge

"""Part of the code dedicated to graphs and specific sequences of points"""


class Path:

    def __init__(self, name, seq, regions=None, img=None, jumps=None):
        if regions is None:
            regions = []
        if img is None:
            img = []
        if jumps is None:
            jumps = []
        self.name = name
        self.sequence = seq
        self.regions = regions
        self.img = img
        self.jumps = jumps
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
                    if np.logical_and(o.route, self.img).any():
                        self.regions["offsets"].append(o.name)
        if hasattr(island, "zigzags"):
            if len(island.zigzags.regions):
                for z in island.zigzags.regions:
                    if np.logical_and(z.route, self.img).any():
                        self.regions["zigzags"].append(z.name)
        if hasattr(island, "bridges"):
            if len(island.bridges.cross_over_bridges):
                for cb in island.bridges.cross_over_bridges:
                    if np.logical_and(cb.route, self.img).any():
                        self.regions["cross_over_bridges"].append(cb.name)
        if hasattr(island, "bridges"):
            if len(island.bridges.offset_bridges):
                for ob in island.bridges.offset_bridges:
                    if np.logical_and(ob.route, self.img).any():
                        self.regions["offset_bridges"].append(ob.name)
        if hasattr(island, "bridges"):
            if len(island.bridges.zigzag_bridges):
                for zb in island.bridges.zigzag_bridges:
                    if np.logical_and(zb.route, self.img).any():
                        self.regions["zigzag_bridges"].append(zb.name)
        if hasattr(island, "thin"):
            if len(island.thin_walls.regions):
                for tw in island.thin_walls.regions:
                    if np.logical_and(tw.route, self.img).any():
                        self.regions["thin walls"].append(tw.name)
        return


def calculate_angle(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    angulo_rad = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    angulo_deg = np.degrees(angulo_rad)
    return abs(angulo_deg)


def find_curvature_pts(seq, ang=60):
    curvature_pts = []
    pontos = seq + seq
    for i in range(1, len(pontos) - 1):
        p1 = pontos[i - 1]
        p2 = pontos[i]
        p3 = pontos[i + 1]
        angulo = calculate_angle(p1, p2, p3)
        if angulo > ang:
            if len(curvature_pts) == 0:
                first_angled = p2
            else:
                if p2 == first_angled:
                    break
            curvature_pts.append(p2)
    return curvature_pts


def add_routes_by_sequence(
    nova_rota,
    island: Island,
    interruption_points,
    order_crossover_regions,
    cross_overs_included,
    offssets_included,
    jumps,
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
            # print("sdasda")
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
        jumps.append(nova_rota[idx_on_route])
        print("Jump: ", nova_rota[idx_on_route])
    return nova_rota, cross_overs_included, offssets_included, jumps


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
        print("Jump: ", nova_rota[idx_on_route])
        zigzag_bridges_included.add(name_of_zigzag_bridge)
    return nova_rota, zigzag_bridges_included, zigzags_included, saltos


def connect_thin_walls(island: Island, path_radius_tw):
    new_route = Path("thin wall tree", [], [], jumps=[])
    if hasattr(island.thin_walls, "all_origins"):
        if np.sum(island.thin_walls.all_origins):
            thinwall_list, _, _ = it.divide_by_connected(island.thin_walls.all_origins)
            thinwall_path_list = []
            for i, tw in enumerate(thinwall_list):
                # tw, _, _ = sk.create_prune_skel(tw, path_radius_tw)
                tw, _, _ = sk.create_prune_skel(tw, 0)
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
            new_route = Path("thin wall tree", nova_rota, new_regions, jumps=saltos)
    return new_route


def middle_of_the_line(line_img):
    seq = make_a_chain(line_img, pt.img_to_points(mt.hitmiss_ends_v2(line_img))[0])
    # seq = set_first_pt_in_seq(seq, pt.img_to_points(mt.hitmiss_ends_v2(line_img))[0])
    return pt.invert_x_y([seq[int(len(seq) / 2)]])[0]


def connect_cross_over_bridges(island: Island) -> Path:
    def find_interruption_points(
        island,
        nova_rota,
        cross_overs_included,
        offssets_included,
        order_crossover_regions,
        pts_valido_comeco,
    ):
        closest_points = {}
        closest_centers = []
        flag_first = len(pts_valido_comeco)
        for bridge in island.bridges.cross_over_bridges:
            if not (bridge.name in list(cross_overs_included)):
                if set(bridge.linked_offset_regions).intersection(offssets_included):
                    A = bridge.extreme_points
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
                        A = bridge.extreme_points
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
            bbbb = it.sum_imgs(
                [aaaa, it.points_to_img(closest_centers, np.zeros_like(island.img))]
            )
            if pt.list_inside_list(closest_centers, pts_valido_comeco):
                pts_valido_comeco = list(
                    filter(lambda x: x not in pts_valido_comeco, nova_rota)
                )
                cccc = it.points_to_img(pts_valido_comeco, np.zeros_like(island.img))
                dddd = it.sum_imgs(
                    [cccc, it.points_to_img(closest_centers, np.zeros_like(island.img))]
                )
        return interruption_points, order_crossover_regions, pts_valido_comeco

    start_path = list(
        filter(lambda x: "Reg_000" in x.regions["offsets"], island.external_tree_route)
    )
    if len(start_path) > 0:
        start_path = start_path[0]
    else:
        start_path = list(
            filter(
                lambda x: "Reg_000" in x.regions["thin_walls"],
                island.external_tree_route,
            )
        )
        if len(start_path) > 0:
            start_path = start_path[0]
        else:
            raise ValueError("Error: no start path found")
    offssets_included = set(start_path.regions["offsets"])
    cross_overs_included = set(start_path.regions["cross_over_bridges"])
    offset_bridges_included = set(start_path.regions["offset_bridges"])
    all_its = []
    pts_valido_comeco = []
    rota_antiga = start_path.sequence.copy()
    nova_rota = start_path.sequence.copy()
    stop = 0
    saltos = []
    counter = 0
    if hasattr(island, "bridges"):
        while not stop:
            order_crossover_regions = []
            interruption_points, order_crossover_regions, pts_valido_comeco = (
                find_interruption_points(
                    island,
                    rota_antiga,
                    cross_overs_included,
                    offssets_included,
                    order_crossover_regions,
                    pts_valido_comeco,
                )
            )
            if len(list(pts_valido_comeco)) > 0 and counter == 0:
                pts_valido_comeco = list(
                    filter(lambda x: x not in interruption_points, pts_valido_comeco)
                )
                new_start = random.choice(pts_valido_comeco)
                if len(new_start) == 0:
                    print("Error: no solution yet")
                rota_antiga = set_first_pt_in_seq(
                    rota_antiga, new_start, evitar_saltos=interruption_points
                )
                aaaa = it.sum_imgs(
                    [
                        it.points_to_img(pts_valido_comeco, np.zeros_like(island.img)),
                        it.points_to_img([new_start], np.zeros_like(island.img)),
                        it.points_to_img(nova_rota, np.zeros_like(island.img)),
                        it.points_to_img(
                            interruption_points, np.zeros_like(island.img)
                        ),
                    ]
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
            counter += 1
    else:
        # nova_rota = start_path.sequence
        saltos = []
    new_regions = {
        "offsets": list(offssets_included),
        "zigzags": [],
        "cross_over_bridges": list(cross_overs_included),
        "offset_bridges": list(offset_bridges_included),
        "zigzag_bridges": [],
    }
    # if len(list(cross_overs_included)) > 0:
    #     cccc = it.points_to_img(pts_valido_comeco, np.zeros_like(island.img))
    #     new_start = random.choice(pts_valido_comeco)
    #     nova_rota = set_first_pt_in_seq(nova_rota, new_start)
    new_route = Path("exterior tree", nova_rota, new_regions, jumps=saltos)
    # aaa = new_route.get_img()
    return new_route


def connect_internal_external(island: Island, path_radius_int_ext):
    filling = island.zigzags.all_zigzags
    if np.sum(filling) > 0:
        most_external = island.offsets.regions[0].route.astype(np.uint8)
        dilation_kernel = int(path_radius_int_ext * 2)
        touching = np.zeros_like(island.img)
        while np.sum(touching) == 0:
            aaa = it.sum_imgs(
                [filling, mt.dilation(most_external, kernel_size=dilation_kernel)]
            )
            touching = aaa == 2
            dilation_kernel = dilation_kernel + 2
        dilation_kernel = dilation_kernel + 2
        aaa = it.sum_imgs(
            [filling, mt.dilation(most_external, kernel_size=dilation_kernel)]
        )
        touching = aaa == 2
        candidates_internal = pt.img_to_points(touching)
        chosen_internal = random.choice(candidates_internal)
        external_pts = pt.img_to_points(most_external)
        chosen_external, _ = pt.closest_point(chosen_internal, external_pts)
    elif hasattr(island, "bridges"):
        filling = it.sum_imgs([x.route for x in island.bridges.zigzag_bridges])
        if len(filling) > 0:
            # filling = it.sum_imgs(filling)
            print("Only zigzag bridges")
            most_external = island.offsets.regions[0].route.astype(np.uint8)
            dilation_kernel = int(path_radius_int_ext * 2)
            touching = np.zeros_like(island.img)
            while np.sum(touching) == 0:
                aaa = it.sum_imgs(
                    [filling, mt.dilation(most_external, kernel_size=dilation_kernel)]
                )
                touching = aaa == 2
                dilation_kernel = dilation_kernel + 2
            candidates_internal = pt.img_to_points(touching)
            chosen_internal = random.choice(
                pt.img_to_points(mt.hitmiss_ends_v2(filling))
            )
            external_pts = pt.img_to_points(most_external)
            chosen_external, _ = pt.closest_point(chosen_internal, external_pts)
        else:
            most_external = island.offsets.regions[0].route.astype(np.uint8)
            external_pts = pt.img_to_points(most_external)
            chosen_external = random.choice(external_pts)
            chosen_internal = []
        if chosen_external == []:
            print("Error: no solution yet")
            chosen_external = random.choice(pt.img_to_points(most_external))
            chosen_internal = []
    return chosen_external, chosen_internal


def connect_offset_bridges(
    island: Island, base_frame, mask_3_4, path_radius_cont, sob_cont_per
) -> Path:

    def integrate_bridge(
        todas_espirais, path_radius, extreme_points, base_frame, sob_cont_per, origin
    ):

        # def filtrar_pontos(y_val, x_val):
        #     validos_no_y = [p for p in todas_espirais_points if p[0] == y_val]
        #     validos_no_yx = [
        #         p for p in validos_no_y if p[1] <= max(x_val) and p[1] >= min(x_val)
        #     ]
        #     return validos_no_yx

        # def find_contacts(pontos, condicao):
        #     contacts = [p for p in pontos if condicao(p[1])]
        #     if len(contacts) == 0:
        #         contacts = pontos
        #     return contacts

        # def closest_to_center(pontos):
        #     return pontos[np.argmin([pt.distance_pts(midle_point, p) for p in pontos])]

        # bridge_middle_y = extreme_points[0][0]
        # bridge_higher_y = (
        #     bridge_middle_y - path_radius + np.round(path_radius * sob_cont_per / 200)
        # )
        # bridge_lower_y = (
        #     bridge_middle_y + path_radius - np.round(path_radius * sob_cont_per / 200)
        # )
        # x_extreme_points = [x[1] for x in extreme_points]
        # midle_point = [
        #     bridge_middle_y,
        #     int((x_extreme_points[0] + x_extreme_points[1]) / 2),
        # ]
        # limites_x = [
        #     min(x_extreme_points) - 2 * path_radius,
        #     max(x_extreme_points) + 2 * path_radius,
        # ]
        # todas_espirais_points = pt.x_y_para_pontos(np.nonzero(todas_espirais))
        # same_y_up_inside = filtrar_pontos(bridge_higher_y, limites_x)
        # if len(same_y_up_inside) < 2:
        #     same_y_up_inside = filtrar_pontos(bridge_higher_y + 1, limites_x)
        #     if len(same_y_up_inside) < 2:
        #         same_y_up_inside = filtrar_pontos(bridge_higher_y + 2, limites_x)
        # same_y_down_inside = filtrar_pontos(bridge_lower_y, limites_x)
        # if len(same_y_down_inside) < 2:
        #     same_y_down_inside = filtrar_pontos(bridge_lower_y - 1, limites_x)
        #     if len(same_y_down_inside) < 2:
        #         same_y_down_inside = filtrar_pontos(bridge_lower_y - 2, limites_x)
        # contact_ec = find_contacts(same_y_up_inside, lambda x: x < midle_point[1])
        # contact_dc = find_contacts(same_y_up_inside, lambda x: x > midle_point[1])
        # contact_eb = find_contacts(same_y_down_inside, lambda x: x < midle_point[1])
        # contact_db = find_contacts(same_y_down_inside, lambda x: x > midle_point[1])
        # ponto_esq_cima = closest_to_center(contact_ec)
        # ponto_dir_cima = closest_to_center(contact_dc)
        # ponto_esq_baixo = closest_to_center(contact_eb)
        # ponto_dir_baixo = closest_to_center(contact_db)
        # line_cima = it.draw_line(np.zeros(base_frame), ponto_esq_cima, ponto_dir_cima)
        # line_baixo = it.draw_line(
        #     np.zeros(base_frame), ponto_esq_baixo, ponto_dir_baixo
        # )
        # retangulo = it.draw_polyline(
        #     np.zeros(base_frame),
        #     [ponto_esq_cima, ponto_dir_cima, ponto_dir_baixo, ponto_esq_baixo],
        #     1,
        # )
        # retangulo = it.fill_internal_area(retangulo, np.ones_like(retangulo))
        # new_todas_espirais = np.logical_and(todas_espirais, np.logical_not(retangulo))
        # new_todas_espirais = it.sum_imgs([new_todas_espirais, line_baixo, line_cima])
        # cleaned_new_todas_espirais, _, _ = sk.create_prune_skel(
        #     new_todas_espirais, path_radius
        # )
        distance_between_centers = int(
            np.round(2 * path_radius * (100 - sob_cont_per) / 100)
        )
        mask_line = np.zeros(
            (distance_between_centers, distance_between_centers)
        )  # TODO: adicionar a sobreposição
        mask_line[:, int(distance_between_centers / 2)] = 1
        origin_center = pt.points_center(pt.img_to_points(origin.astype(np.uint8)))
        transversal_origin = mt.dilation(
            it.points_to_img([origin_center], np.zeros(base_frame)),
            kernel_img=mask_line,
        )
        transversal_origin = np.logical_and(
            transversal_origin,
            np.logical_not(todas_espirais),
        )
        distanced_points_img = mt.hitmiss_ends_v2(transversal_origin)
        distanced_points = pt.img_to_points(distanced_points_img)
        top_bottom_lines = np.zeros(base_frame)
        for point in distanced_points:
            for dir in [2, 0]:
                this_line, _, _ = it.extend_line_random_to_touch(
                    todas_espirais * 10,
                    point,
                    minimum=11,
                    pre_dettermined=dir,
                )
                top_bottom_lines = np.logical_or(top_bottom_lines, this_line)
        bbbbbbb = it.sum_imgs(
            [origin, transversal_origin, top_bottom_lines, todas_espirais]
        )
        _, labeled, num_features = it.divide_by_connected(top_bottom_lines)
        if num_features != 2:
            raise ValueError("Image does not contain exactly two lines.")
        lines = []
        for i in range(1, num_features + 1):
            coords = np.where(labeled == i)
            y_mean = np.mean(coords[0])
            lines.append((y_mean, i))
        lines.sort()
        line_cima = (labeled == lines[0][1]).astype(np.uint8)
        line_baixo = (labeled == lines[1][1]).astype(np.uint8)
        up_points = pt.img_to_points(mt.hitmiss_ends_v2(line_cima))
        down_points = pt.img_to_points(mt.hitmiss_ends_v2(line_baixo))
        ponto_dir_cima = up_points[np.argmax([x[1] for x in up_points])]
        ponto_esq_cima = up_points[np.argmin([x[1] for x in up_points])]
        ponto_dir_baixo = down_points[np.argmax([x[1] for x in up_points])]
        ponto_esq_baixo = down_points[np.argmin([x[1] for x in up_points])]
        retangulo = it.draw_polyline(
            np.zeros(base_frame),
            [ponto_esq_cima, ponto_dir_cima, ponto_dir_baixo, ponto_esq_baixo],
            1,
        )
        retangulo = it.fill_internal_area(retangulo, np.ones_like(retangulo))
        new_todas_espirais = np.logical_and(todas_espirais, np.logical_not(retangulo))
        new_todas_espirais = it.sum_imgs([new_todas_espirais, line_baixo, line_cima])
        cleaned_new_todas_espirais, _, _ = sk.create_prune_skel(
            new_todas_espirais, path_radius
        )

        return cleaned_new_todas_espirais

    def integrate_contact(todas_espirais, path_radius, bridge: Bridge, base_frame):
        route = bridge.route
        eraser = mt.dilation(bridge.origin, kernel_size=path_radius - 2)
        aaa = it.sum_imgs([route, todas_espirais])
        new_todas_espirais = it.image_subtract(aaa, eraser)
        A, _, _ = sk.create_prune_skel(new_todas_espirais, path_radius)
        return A

    lista_de_rotas = []
    todas_espirais_img = np.zeros(base_frame)
    for region in island.offsets.regions:
        todas_espirais_img = np.logical_or(todas_espirais_img, region.route)
    if hasattr(island, "bridges"):
        for bridge in island.bridges.offset_bridges:
            extreme_points = pt.x_y_para_pontos(
                np.nonzero(mt.hitmiss_ends_v2(bridge.origin))
            )
            if bridge.type == "common_offset_bridge":
                todas_espirais_img = integrate_bridge(
                    todas_espirais_img,
                    path_radius_cont,
                    extreme_points,
                    base_frame,
                    sob_cont_per,
                    bridge.origin,
                )
            elif bridge.type == "contact_offset_bridge":
                todas_espirais_img = integrate_contact(
                    todas_espirais_img,
                    path_radius_cont,
                    bridge,
                    base_frame,
                )
    rotas_isoladas = img_to_chain(todas_espirais_img.astype(np.uint8))
    lens = [len(x) for x in rotas_isoladas]
    circunf = 2 * 3.14 * path_radius_cont
    for i, rota in enumerate(rotas_isoladas):
        # if lens[i] > 2 * circunf:
        lista_de_rotas.append(Path(i, rota))
        if len(island.ext_start) == 0:
            island.ext_start = lista_de_rotas[-1].sequence[0]
        lista_de_rotas[-1].sequence = set_first_pt_in_seq(
            lista_de_rotas[-1].sequence,
            list(island.ext_start),
        )
        lista_de_rotas[-1].get_img(base_frame)
        lista_de_rotas[-1].get_regions(island)
    if len(lista_de_rotas) == 0:
        print("No offset bridges")
    return lista_de_rotas


def connect_zigzag_bridges(island: Island):
    start_path = island.internal_tree_route[0]
    zigzags_included = set(start_path.regions["zigzags"])
    # zigzag_bridges_included = set(start_path.regions["zigzag_bridges"])
    zigzag_bridges_included = set()
    zigzag_bridges_number = len(island.bridges.zigzag_bridges)
    if zigzag_bridges_number == 0:
        nova_rota = start_path.sequence
        saltos = []
    elif len(zigzags_included) == 0:
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
    new_route = Path("interior tree", nova_rota, new_regions, jumps=saltos)
    # aaa = new_route.get_img(island.img.shape)
    return new_route


def colorbyevent(seq, eventlist, img):
    """Follows the sequence and assigns a new label each time it encounters an event point"""
    result = copy.deepcopy(img).astype(np.uint8)
    label = 0
    for p in seq:
        occurences = list(filter(lambda x: x == p, eventlist))
        if len(occurences) > 0:
            label = label + 1
        result[p[0]][p[1]] = label
    return result


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


def draw_tangent_from_seq(points, length, img):
    # Create a binary image
    binary_image = copy.deepcopy(img)
    # Convert points to integer coordinates
    points = [(int(y), int(x)) for y, x in points]
    # Draw the original points on the binary image
    for y, x in points:
        binary_image[y, x] = 1  # Mark the point
    # Calculate the tangent at the last point
    if len(points) < 2:
        print("Not enough points to calculate tangent.")
        return tng_img
    last_point = points[-1]
    second_last_point = points[-2]
    slope = pt.calculate_tangent(second_last_point, last_point)
    # if slope == None:
    #     slope = -9999
    # Extend the tangent line
    tangent_line = it.extend_tangent(last_point, second_last_point, slope, length)
    # Draw the tangent line on the binary image
    tng_img = it.draw_line(img, np.uint64(tangent_line[0]), np.uint64(tangent_line[1]))
    return tng_img


def draw_the_links(
    zigzags, zigzags_mst, base_frame, interfaces, centers, path_radius_larg
):

    def perpendicular_on_point(line_img, center, base_frame, path_radius):
        n_contatos = 0
        img_points = mt.hitmiss_ends_v2(line_img)
        if np.sum(img_points) < 2:
            line_img, _, _ = sk.create_prune_skel(line_img, path_radius)
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
        link = perpendicular_on_point(line, centers[i], base_frame, path_radius_larg)
        mask_line = np.zeros((path_radius_larg * 2, path_radius_larg * 2))
        mask_line[int(path_radius_larg) - 1] = 1
        mask_line[int(path_radius_larg)] = 1
        mask_line[int(path_radius_larg) + 1] = 1
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
    edges, path_radius_larg, mask_full_int, zigzags: List[ZigZag]
):

    def dilate_and_search(a1, a2, grow):
        mask_line = np.zeros(np.add(mask_full_int.shape, [grow, grow]))
        mask_line[:, int(mask_line.shape[0] / 2)] = 1
        # a1_reg = zigzags[int(edge[0][1])]
        a1_vertical_trail = mt.dilation(a1.route, kernel_img=mask_line)
        # a2_reg = zigzags[int(edge[1][1])]
        a2_vertical_trail = mt.dilation(a2.route, kernel_img=mask_line)
        interface = np.add(a1_vertical_trail, a2_vertical_trail) == 2
        tips = pt.img_to_points(mt.hitmiss_ends_v2(interface))
        if len(tips) != 2:
            interface, _, _ = sk.create_prune_skel(interface, 1)
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
            while np.sum(interface) == 0 and grow < path_radius_larg * 2:
                grow = grow + 1
                interface = dilate_and_search(a1, a2, grow)
            separated, _, num = it.divide_by_connected(interface)
            if num > 1:
                sums = [np.sum(x) for x in separated]
                interface = separated[np.argmax(sums)]
            if np.sum(interface) > 0:
                interface_pts = pt.x_y_para_pontos(np.nonzero(interface))
                center = pt.points_center(interface_pts)
                interfaces.append(interface)
                centers.append(center)
                interface_types.append(has_bridge)
            else:
                print("Error: no interface found")
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
    if zigzags_included:
        for bridge in isl.bridges.zigzag_bridges:
            if not (bridge.name in zigzag_bridges_included):
                print(bridge.linked_zigzag_regions)
                if set(bridge.linked_zigzag_regions).intersection(zigzags_included):
                    A = bridge.extreme_points
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
    else:
        # ARRUMAR AQUI PARA MAIS DE UMA PONTEEEEEEE
        for i, bridge in enumerate(isl.bridges.zigzag_bridges):
            if not (bridge.name in zigzag_bridges_included) or i == 0:
                A = bridge.extreme_points
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
    remembering the indices:
        _______1______
        |            |
        0            2
        |______3_____|
    """
    region.make_contour(base_frame)
    region.center_coords = pt.calculate_centroid(region.img)
    all_loops = np.zeros(base_frame)
    loops_counter = 0
    for loop in region.loops:
        all_loops = np.logical_or(all_loops, loop.route)
        loops_counter += 1
    cutter_line, _, direction_index = it.extend_line_random_to_touch(
        all_loops.astype(np.uint8),
        region.center_coords,
        minimum=2,
        touches=loops_counter,
        print_from_first=True,
    )
    return (cutter_line == True).astype(np.uint8), direction_index


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
        line_com_comeco_certo = line[start_pt_idx:] + line[:start_pt_idx]
        if len(line_com_comeco_certo) > 1:
            if line_com_comeco_certo[1][0] == max_y:
                # print("reverti um deles!")
                line_com_comeco_certo.reverse()
        multiple_lines[l] = line_com_comeco_certo
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
        for elem_parallelo in filtered_regions[origem_num].parallel_points:
            area_origem = origem_num = elem_parallelo.origin
            area_origem_num = int(elem_parallelo.origin.replace("Reg_", ""))
            area_destino = origem_num = elem_parallelo.destiny
            area_destino_num = int(elem_parallelo.destiny.replace("Reg_", ""))
            for i in np.arange(0, len(elem_parallelo.dist_a)):
                origin_coords = filtered_regions[area_origem_num].limmit_coords[0]
                graph.add_edge(
                    area_origem,
                    area_destino,
                    weight=elem_parallelo.dist_a[i],
                    origin_coords=origin_coords,
                    destiny_coords=elem_parallelo.lista_a[i],
                    extreme_origin="a",
                )
            for i in np.arange(0, len(elem_parallelo.dist_b)):
                origin_coords = filtered_regions[area_origem_num].limmit_coords[1]
                graph.add_edge(
                    area_origem,
                    area_destino,
                    weight=elem_parallelo.dist_b[i],
                    origin_coords=origin_coords,
                    destiny_coords=elem_parallelo.lista_b[i],
                    extreme_origin="b",
                )
            for i in np.arange(0, len(elem_parallelo.dist_c)):
                origin_coords = filtered_regions[area_origem_num].limmit_coords[2]
                graph.add_edge(
                    area_origem,
                    area_destino,
                    weight=elem_parallelo.dist_c[i],
                    origin_coords=origin_coords,
                    destiny_coords=elem_parallelo.lista_c[i],
                    extreme_origin="c",
                )
            for i in np.arange(0, len(elem_parallelo.dist_d)):
                origin_coords = filtered_regions[area_origem_num].limmit_coords[3]
                graph.add_edge(
                    area_origem,
                    area_destino,
                    weight=elem_parallelo.dist_d[i],
                    origin_coords=origin_coords,
                    destiny_coords=elem_parallelo.lista_d[i],
                    extreme_origin="d",
                )
    for origem_a, origem_b in regs_touching:
        region_a = [x for x in filtered_regions if x.name == origem_a][0]
        region_b = [x for x in filtered_regions if x.name == origem_b][0]
        region_a_all_loops = it.sum_imgs([x.route for x in region_a.loops])
        region_b_all_loops = it.sum_imgs([x.route for x in region_b.loops])
        origin_coords_a, origin_coords_b = it.closest_points_btwn_imgs(
            region_a_all_loops, region_b_all_loops
        )
        origin_coords_a = pt.invert_x_y([origin_coords_a])
        origin_coords_b = pt.invert_x_y([origin_coords_b])
        graph.add_edge(
            origem_a,
            origem_b,
            weight=0,
            origin_coords=origin_coords_a[0],
            destiny_coords=origin_coords_b[0],
            extreme_origin="e",
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
        if list(p) == [0, 0]:
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


def rectangle_cut(contours, line, points, n_loops, base_frame, mode=0, idx=0):
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
        pixel_lines = line[y][x]
        ca = [y, x] == points[0]
        cb = [y, x] == points[1]
        cc = [y, x] == points[2]
        cd = [y, x] == points[3]
        ce = pixel_lines == 1
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


def set_first_pt_in_seq(seq, first_point, evitar_saltos=[]):
    def invert_if_close_to_jump(fila, evitar_saltos):
        fila2 = fila[::-1]
        fila2 = [fila[0]] + fila2[:-1]
        indexeses = [fila.index(x) for x in evitar_saltos]
        indexeses2 = [fila2.index(x) for x in evitar_saltos]
        if min(indexeses) < min(indexeses2):
            return fila2
        elif min(indexeses2) < min(indexeses):
            return fila
        else:
            print("Error: special case")
        # primeiro_ponto_lista2 = None

        # for ponto in evitar_saltos:
        #     if ponto in fila:
        #         primeiro_ponto_lista2 = ponto
        #         break
        # if primeiro_ponto_lista2 is not None:
        #     indice_ponto_lista1 = fila.index(primeiro_ponto_lista2)
        #     elementos_antes_lista1 = fila[:indice_ponto_lista1]
        # else:
        #     elementos_antes_lista1 = fila  # Se não houver pontos em comum, consideramos toda a lista1
        # primeiro_ponto_lista1 = None
        # for ponto in fila:
        #     if ponto in evitar_saltos:
        #         primeiro_ponto_lista1 = ponto
        #         break
        # if primeiro_ponto_lista1 is not None:
        #     indice_ponto_lista2 = evitar_saltos.index(primeiro_ponto_lista1)
        #     elementos_antes_lista2 = evitar_saltos[:indice_ponto_lista2]
        # else:
        #     elementos_antes_lista2 = evitar_saltos  # Se não houver pontos em comum, consideramos toda a lista2
        # if len(elementos_antes_lista1) > len(elementos_antes_lista2):
        #     return fila
        # else:
        #     print("invertido foi")
        #     return fila[::-1]  # Retorna a lista1 invertida

    fila = seq.copy()
    if not (first_point in seq):
        first_point, _ = pt.closest_point(first_point, seq)
        # first_point = [first_point[1],first_point[0]]
    rotations = fila.index(first_point)
    fila = fila[rotations:] + fila[:rotations]
    if len(evitar_saltos) > 0:
        fila = invert_if_close_to_jump(fila, evitar_saltos)
    else:
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
        if len(saltos) > 0:
            segmentos = np.array_split(
                sequence, np.where([(x in saltos) for x in sequence])[0]
            )
        else:
            segmentos = [sequence]
        segmentos = list(filter(lambda x: len(x) > 0, segmentos))
        for s in segmentos:
            candidates = []
            candidates.append(factor_epilson * arcLength(np.float32(s), False))
            candidates.append(factor_epilson * arcLength(np.float32(s), True))
            epsilon = candidates[np.argmax(candidates)]
            approx_seg = approxPolyDP(
                np.ascontiguousarray(np.float32(s)), epsilon, False
            )
            approx_seg = [list(x[0]) for x in approx_seg]
            if approx_seg is None:
                pass
            else:
                approx_seq += approx_seg
                # approx_seq += [["a", "a"]]
                approx_seq += [[0, 0]]
    return approx_seq


def simplifica_retas_masterV2(seq_pts, factor_epilson, saltos):
    # if len(seq_pts) == 0:
    #     return []
    def perpendicular_distance(point, start, end):
        """Calculate the perpendicular distance from a point to a line segment."""
        start = np.array(start)
        end = np.array(end)
        point = np.array(point)
        if np.array_equal(start, end):
            return np.linalg.norm(point - start)
        # Vector from start to end
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec = point - start
        t = np.dot(point_vec, line_unitvec)
        if t < 0:
            nearest = start
        elif t > line_len:
            nearest = end
        else:
            nearest = start + t * line_unitvec
        return np.linalg.norm(point - nearest)

    def douglas_peucker(points, epsilon):
        """Simplify a sequence of points using the Ramer-Douglas-Peucker algorithm."""
        points = np.array(points)  # Ensure points is a NumPy array
        if len(points) < 2:
            return points
        # Find the point with the maximum distance from the line between the endpoints
        start, end = points[0], points[-1]
        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d
        # If the maximum distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            left = douglas_peucker(points[: index + 1], epsilon)
            right = douglas_peucker(points[index:], epsilon)
            # Combine the results
            return np.vstack(
                (left[:-1], right)
            )  # Exclude the last point of left to avoid duplication
        else:
            return np.array([start, end])

    sequence = [list(x) for x in seq_pts]
    approx_seq = []
    if len(sequence) > 0:
        saltos = [list(x) for x in saltos]
        if len(saltos) > 0:
            segmentos = np.array_split(
                sequence, np.where([(x in saltos) for x in sequence])[0]
            )
        else:
            segmentos = [sequence]
        segmentos = list(filter(lambda x: len(x) > 0, segmentos))
        for s in segmentos:
            simplified_coordinates = douglas_peucker(s, factor_epilson)
            if approx_seq == []:
                approx_seq = simplified_coordinates.tolist()
            else:
                approx_seq = approx_seq + simplified_coordinates.tolist()
            approx_seq = approx_seq + [[0, 0]]
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
        pixel_lines = spiral[y][x]
        ca = [y, x] == points[0]
        cb = [y, x] == points[1]
        cc = [y, x] == points[2]
        cd = [y, x] == points[3]
        ce = pixel_lines == 1
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


def start_internal_route(isl: Island, mask_full_int, path_radius_larg):
    path_list = []
    if hasattr(isl, "zigzags"):
        if hasattr(isl.zigzags, "macro_areas_weaved"):
            list_of_reagions = isl.zigzags.macro_areas_weaved
        else:
            list_of_reagions = isl.zigzags.macro_areas
        for i, ma in enumerate(list_of_reagions):
            zigzag_path = img_to_chain(ma.astype(np.uint8), isl.zigzags.regions[0].img)
            if len(zigzag_path) > 0:
                path_list.append(Path(i, zigzag_path[0], img=ma))
                path_list[-1].sequence = set_first_pt_in_seq(
                    path_list[-1].sequence, list(isl.int_start)
                )
                path_list[-1].sequence = cut_repetition(path_list[-1].sequence)
                path_list[-1].get_regions(isl)
                # path_list[-1].regions = {
                #                             "offsets": [],
                #                             "zigzags": [f"{isl.zigzags.regions[0].name}"],
                #                             "cross_over_bridges": [],
                #                             "offset_bridges": [],
                #                             "zigzag_bridges": [],
                #                             "thin walls": [],
                #                         }
    if path_list == []:
        if hasattr(isl, "bridges"):
            for i, zb in enumerate(isl.bridges.zigzag_bridges):
                zigzag_b_path = img_to_chain(
                    zb.route, isl.bridges.zigzag_bridges[0].route
                )
                if len(zigzag_b_path) > 0:
                    path_list.append(Path(i, zigzag_b_path[0], img=zb.route))
                    path_list[-1].sequence = set_first_pt_in_seq(
                        path_list[-1].sequence, list(isl.int_start)
                    )
                    path_list[-1].sequence = cut_repetition(path_list[-1].sequence)
                    path_list[-1].get_regions(isl)
    return path_list


# def layers_to_Gcode_4t(
#     layers: List[Layer],
#     folders: System_Paths,
#     configuracoes,
#     vel_vazio,
#     p_entre_int_ext,
#     p_entre_layers,
#     layer_heights,
#     coords_substrato,
#     coords_corte,
# ):
#     """modo 4T na maquina Okerlion na FCT-NOVA"""
#     import os

#     def religamento(output, flag_ligado, on_pause):
#         if flag_ligado == 0:
#             output += ";-------RELIGAMENTO------\n"
#             output += "M42 P4 S0\n"
#             output += f"G4 P{p_trigger_longa}\n"
#             # output += f"G4 P{on_pause}\n"
#             output += "M42 P4 S255\n"
#             output += f"G4 P{on_pause-p_trigger_longa}\n"
#             output += ";------------------------\n"
#         return output, 1

#     def desligamento(output, flag_ligado, off_pause):
#         if flag_ligado == 1:
#             output += ";-------DESLIGAMENTO------\n"
#             output += "M42 P4 S0\n"
#             output += f"G4 P{p_trigger_longa}\n"
#             # output += f"G4 P{off_pause}\n"
#             output += "M42 P4 S255\n"
#             output += f"G4 P{off_pause-p_trigger_longa}\n"
#             output += ";-------------------------\n"
#         return output, 0

#     def posicao_de_corte(output, coords):
#         output += ";-------POS de CORTE------\n"
#         output += f";POS de Corte\n"
#         output += f"G90\n"
#         output += f"G0 Y{coords[0]} F{vel_vazio}\n"
#         output += f"M400\n"
#         output += f"G0 x{coords[1]} F{vel_vazio}\n"
#         output += f"M400\n"
#         output += f"G4 P{p_entre_layers}\n"
#         output += f"G91\n"
#         output += ";------------------------\n"
#         return output

#     def posicao_inicial(output, coords, i):
#         output += f";_______LAYER{n_layer + 1}_____\n"
#         output += f"G90\n"
#         output += f";LAYER:{i}\n"
#         output += f"G1 Z{layer_heights[n_layer]} ; Layer + 10mm\n"
#         output += f"G1 X{coords[1]} Y{coords[0]} F{vel_vazio}; POS INICIAL\n"
#         output += f"G91\n"
#         return output

#     mm_per_pixel = layers[0].mm_per_pxl
#     ts = datetime.datetime.now()
#     outFile = f"{folders.selected} {ts.date()} {ts.hour}_{ts.minute}.gcode"
#     output = ""
#     output += ";-------MAPPING------\n"
#     output += f";DPI: {layers[0].dpi} ppp\n"
#     output += f";void_max: {layers[0].void_max} % of path_radius\n"
#     output += f";max_internal_walls: {layers[0].max_internal_walls}\n"
#     output += f";max_external_walls: {layers[0].max_external_walls}\n"
#     output += f";n_max: {layers[0].n_max} trilhas para bottleneck\n"
#     output += ";-------Welding program 1 contours------\n"
#     output += f";Program name: {layers[0].program_cont}\n"
#     diam_cont = configuracoes.lista_programas["name"==layers[0].program_cont]["bead_diameter"]
#     output += f";Bead diameter: {diam_cont} mm\n"
#     sobrep_cont = configuracoes.lista_programas["name"==layers[0].program_cont]["bead_superposition"]
#     output += f";Bead superposition: {sobrep_cont} % raio real \n"
#     vel_cont = configuracoes.lista_programas["name"==layers[0].program_cont]["travel_speed"]
#     output += f";travel_speed: {vel_cont} mm/min \n"
#     output += f";path_radius: {layers[0].path_radius_cont} pixels\n"
#     on_pause_cont = configuracoes.lista_programas["name"==layers[0].program_cont]["on_pause"]
#     output += f";on_pause: {on_pause_cont} ms \n"
#     off_pause_cont = configuracoes.lista_programas["name"==layers[0].program_cont]["off_pause"]
#     output += f";off_pause: {off_pause_cont} ms \n"
#     output += ";-------Welding program 2 bottleneck------\n"
#     output += f";Program name: {layers[0].program_bridg}\n"
#     diam_bridg = configuracoes.lista_programas["name"==layers[0].program_bridg]["bead_diameter"]
#     output += f";Bead diameter: {diam_bridg} mm\n"
#     sobrep_bridg = configuracoes.lista_programas["name"==layers[0].program_bridg]["bead_superposition"]
#     output += f";Bead superposition: {sobrep_bridg} % raio real \n"
#     vel_bridg = configuracoes.lista_programas["name"==layers[0].program_bridg]["travel_speed"]
#     output += f";travel_speed: {vel_bridg} mm/min \n"
#     output += f";path_radius: {layers[0].path_radius_bridg} pixels\n"
#     on_pause_bridg = configuracoes.lista_programas["name"==layers[0].program_bridg]["on_pause"]
#     output += f";on_pause: {on_pause_bridg} ms \n"
#     off_pause_bridg = configuracoes.lista_programas["name"==layers[0].program_bridg]["off_pause"]
#     output += f";off_pause: {off_pause_bridg} ms \n"
#     output += ";-------Welding program 3 wide_areas------\n"
#     output += f";Program name: {layers[0].program_larg}\n"
#     diam_larg = configuracoes.lista_programas["name"==layers[0].program_larg]["bead_diameter"]
#     output += f";Bead diameter: {diam_larg} mm\n"
#     sobrep_larg = configuracoes.lista_programas["name"==layers[0].program_larg]["bead_superposition"]
#     output += f";Bead superposition: {sobrep_larg} % raio real \n"
#     vel_larg = configuracoes.lista_programas["name"==layers[0].program_larg]["travel_speed"]
#     output += f";travel_speed: {vel_larg} mm/min \n"
#     output += f";path_radius: {layers[0].path_radius_larg} pixels\n"
#     on_pause_larg = configuracoes.lista_programas["name"==layers[0].program_larg]["on_pause"]
#     output += f";on_pause: {on_pause_larg} ms \n"
#     off_pause_larg = configuracoes.lista_programas["name"==layers[0].program_larg]["off_pause"]
#     output += f";off_pause: {off_pause_larg} ms \n"
#     output += ";-------Welding program 4 Paredes finas------\n"
#     output += f";Program name: {layers[0].program_tw}\n"
#     diam_tw = configuracoes.lista_programas["name"==layers[0].program_tw]["bead_diameter"]
#     output += f";Bead diameter: {diam_tw} mm\n"
#     sobrep_tw = configuracoes.lista_programas["name"==layers[0].program_tw]["bead_superposition"]
#     output += f";Bead superposition: {sobrep_tw} % raio real \n"
#     vel_tw = configuracoes.lista_programas["name"==layers[0].program_tw]["travel_speed"]
#     output += f";travel_speed: {vel_tw} mm/min \n"
#     output += f";path_radius: {layers[0].path_radius_tw} pixels\n"
#     on_pause_tw = configuracoes.lista_programas["name"==layers[0].program_tw]["on_pause"]
#     output += f";on_pause: {on_pause_tw} ms \n"
#     off_pause_tw = configuracoes.lista_programas["name"==layers[0].program_tw]["off_pause"]
#     output += f";off_pause: {off_pause_tw} ms \n"
#     output += ";------------OUTROS------------\n"
#     output += f";Ext int superposition: {layers[0].sob_int_ext_per} % raio interno \n"
#     output += f";N# layers: {layers[0].n_layers}\n"
#     output += f";p_entre_int_ext;: {p_entre_int_ext} ms \n"
#     output += f";p_entre_layers: {p_entre_layers} ms \n"
#     output += f";layer_heights: {layer_heights} mm \n"
#     output += f";coords_corte: {coords_corte} mm \n"
#     output += f";coords_substrato: {coords_substrato} mm \n"
#     output += f";vel_vazio: {vel_vazio} mm/min \n"
#     output += ";------------FIM INPUTS------------\n"
#     output += f"G91\n"
#     output += f"M42 P4 S255; turn off welder\n"
#     output += f"G28 X0 Y0 Z0\n"
#     output += f"G1 F360; speed g1\n"
#     bfr = [0, 0]
#     base_frame = layers[0].base_frame
#     p_trigger_longa = 2000
#     p_trigger_curta = 400

#     for n_layer, layer in enumerate(layers):
#         layer_tot_lenght = 0
#         output = posicao_inicial(output, coords_substrato, n_layer)
#         bfr = coords_substrato
#         folders.load_islands_hdf5(layer)
#         for n_island, island in enumerate(layer.islands):
#             folders.load_island_paths_hdf5(layer.name, island)
#             folders.load_island_paths_hdf5(layer.name, island)
#             # itr = [list(x) for x in island.internal_tree_route.sequence]
#             # etr = [list(x) for x in island.external_tree_route.sequence]
#             # twtr = [list(x) for x in island.thinwalls_tree_route.sequence]
#             folders.load_bridges_hdf5(layer.name, island)
#             print(f"name: {layer.name}/{island.name}")
#             pts_bridg = points_from_region(layer.name,folders,island,bridges=True)
#             pts_tw = points_from_region(layer.name,folders,island,tw=True)
#             pts_cont = points_from_region(layer.name,folders,island,offsets=True)
#             pts_larg = points_from_region(layer.name,folders,island,zigzags=True)
#             # if hasattr(island, "bridges"):
#                 # A1 = [
#                 #     pt.img_to_points(x.route)
#                 #     for x in island.bridges.cross_over_bridges + island.bridges.zigzag_bridges
#                 # ]
#                 # A2 = [
#                 #     pt.img_to_points(x.route_b)
#                 #     for x in island.bridges.cross_over_bridges + island.bridges.zigzag_bridges
#                 # ]
#                 # A = A1 + A2
#                 # for x in A:
#                 #     pts_bridg = pts_bridg + x
#             if n_layer % 2:
#                 # etr = rotate_path_odd_layer(etr, layer.base_frame)
#                 # itr = rotate_path_odd_layer(itr, layer.base_frame)
#                 # twtr = rotate_path_odd_layer(twtr, layer.base_frame)
#                 # pts_bridg = rotate_path_odd_layer(pts_bridg, layer.base_frame)
#                 pts_bridg = rotate_path_odd_layer(pts_bridg, layer.base_frame)
#                 pts_tw = rotate_path_odd_layer(pts_tw, layer.base_frame)
#                 pts_cont = rotate_path_odd_layer(pts_cont, layer.base_frame)
#                 pts_larg = rotate_path_odd_layer(pts_larg, layer.base_frame)
#             # pontos_int = [list(x) for x in itr] + pts_bridg
#             # pontos_ext = [list(x) for x in etr]
#             # pontos_ext = [coord for coord in pontos_ext if coord not in pts_bridg]
#             # pontos_tw = [list(x) for x in island.thinwalls_tree_route.sequence]
#             # pontos_larg = [list(x) for x in itr]
#             # pontos_cont = [list(x) for x in etr]
#             # pontos_brdg = pts_bridg
#             # pontos_tw = [list(x) for x in island.thinwalls_tree_route.sequence]
#             chain = [list(x) for x in island.island_route.sequence]
#             counter = 0
#             flag_salto = 0
#             flag_path_type = 99
#             last_flag = 0
#             flag_ligado = 0
#             print(chain)
#             for i, p in enumerate(chain):
#                 if p == [0, 0]:
#                     output, flag_ligado = desligamento(output, flag_ligado, off_pause)
#                     const_perf = 0
#                     flag_salto = 1
#                 else:
#                     coords = p
#                     coords = [
#                         base_frame[0] - coords[0] + coords_substrato[0],
#                         coords[1] + coords_substrato[1],
#                     ]
#                     # if p in pontos_ext:
#                     #     flag_path_type = 0
#                     #     vel = travel_speed
#                     #     texto_mudanca = ";----Externo----\n;TYPE:WALL-OUTER\n"
#                     #     const_perf = 5
#                     # elif p in pontos_int:
#                     #     flag_path_type = 1
#                     #     vel = vel_int
#                     #     texto_mudanca = ";----Interno----\n;TYPE:SKIN\n"
#                     #     const_perf = 8
#                     # elif p in pontos_tw:
#                     #     flag_path_type = 2
#                     #     vel = vel_thin_wall
#                     #     texto_mudanca = ";----ThinWalls----\n;TYPE:WALL-INNER\n"
#                     #     const_perf = 0.5
#                     if p in pts_cont:
#                         flag_path_type = 1
#                         vel = vel_cont
#                         texto_mudanca = ";----contour----\n;TYPE:WALL-OUTER\n"
#                         const_perf = 5
#                         off_pause = off_pause_cont
#                         on_pause = on_pause_cont
#                     elif p in pts_bridg:
#                         flag_path_type = 2
#                         vel = vel_bridg
#                         texto_mudanca = ";----Estrangulamento----\n;TYPE:SKIN\n"
#                         const_perf = 8
#                         off_pause = off_pause_bridg
#                         on_pause = on_pause_bridg
#                     elif p in pts_larg:
#                         flag_path_type = 3
#                         vel = vel_larg
#                         texto_mudanca = ";----Area Larga----\n;TYPE:WALL-INNER\n"
#                         const_perf = 0.5
#                         off_pause = off_pause_larg
#                         on_pause = on_pause_larg
#                     elif p in pts_tw:
#                         flag_path_type = 4
#                         vel = vel_tw
#                         texto_mudanca = ";----ThinWalls----\n;TYPE:SUPPORT\n"
#                         const_perf = 0.5
#                         off_pause = off_pause_tw
#                         on_pause = on_pause_tw
#                     else:
#                         flag_path_type = 0
#                         vel = vel_vazio
#                         texto_mudanca = ";----perdido----\n"
#                         const_perf = 0
#                     if i == 1:
#                         output, flag_ligado = religamento(output, flag_ligado, on_pause)
#                     if flag_path_type != last_flag:
#                         if flag_path_type == 1 and last_flag == 0:
#                             output, flag_ligado = desligamento(output, flag_ligado, off_pause)
#                             output += f"G4 P{p_entre_int_ext}\n"
#                             output, flag_ligado = religamento(output, flag_ligado, on_pause)
#                         output += f"G1 F{vel}; speed g1\n"
#                         output += "M42 P4 S0\n"
#                         output += f"G4 P{p_trigger_curta}\n"
#                         output += "M42 P4 S255\n"
#                         output += f"G117 {{Trocou o perfil para {flag_path_type}}}\n"
#                         last_flag = flag_path_type
#                         print(f"Changed program to: {flag_path_type}")
#                         output += texto_mudanca
#                     desloc = np.subtract(coords, bfr)
#                     dist = distance.euclidean(coords, bfr)
#                     if flag_salto == 1 or flag_ligado == 0:
#                         const_perf = 0
#                     extrus = dist*const_perf
#                     layer_tot_lenght += dist
#                     # output += (
#                     #     f"G1 X{desloc[1] * mm_per_pixel} Y{desloc[0] * mm_per_pixel} E{extrus}\n"
#                     # )
#                     output += (
#                         f"G1 X{desloc[1] * mm_per_pixel} Y{desloc[0] * mm_per_pixel}\n"
#                     )
#                     output += "M400\n"
#                     bfr = coords
#                     counter += 1
#                     if flag_salto == 1:
#                         output, flag_ligado = religamento(output, flag_ligado, on_pause)
#                         flag_salto = 0
#             output = posicao_de_corte(output, coords_corte)
#             output += ";____________________________________\n"
#             output += f"G28 X0 Y0\n"
#             print(
#                 f"Total layer linear lenght {n_layer} = {layer_tot_lenght*mm_per_pixel}mm"
#             )
#             print(
#                 f"Time estimation with {vel_cont}mm/min = {layer_tot_lenght*mm_per_pixel/vel_cont}min\n"
#             )
#     output += f"G1 Z20\n"
#     output += f"G28 X0\n"
#     output += f"G28 Y0\n"
#     output += f"M104 S0; End of Gcode\n"
#     os.chdir(folders.output)
#     f = open(outFile, "w")
#     f.write(output)
#     f.close()
#     os.chdir(folders.home)
#     return


def get_program_params(program, lista_programas):
    A = list(filter(lambda x: x["name"] == program, lista_programas))[0]
    diam = A["bead_diameter"]
    sobrep = A["bead_superposition"]
    vel = A["travel_speed"]
    on_pause = A["on_pause"]
    off_pause = A["off_pause"]
    return diam, sobrep, vel, on_pause, off_pause


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


def cleanning_position(output, coords, vel_vazio, p_entre_layers):
    output += ";-------CLEANNING POSITION------\n"
    # output += f";POS de Corte\n"
    output += f"G90\n"
    output += f"G0 Y{coords[0]} F{vel_vazio}\n"
    output += f"M400\n"
    output += f"G0 x{coords[1]} F{vel_vazio}\n"
    output += f"M400\n"
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


def region_points(layer: Layer, island: Island, folders: System_Paths):
    # folders.load_bridges_hdf5(layer.name, island)
    # print(f"name: {layer.name}/{island.name}")
    pts_bridg = points_from_region(layer.name, folders, island, bridges=True)
    pts_tw = points_from_region(layer.name, folders, island, tw=True)
    pts_cont = points_from_region(layer.name, folders, island, offsets=True)
    pts_larg = points_from_region(layer.name, folders, island, zigzags=True)
    if layer.odd_layer == 1:
        pts_bridg = rotate_path_odd_layer(pts_bridg, layer.base_frame)
        pts_tw = rotate_path_odd_layer(pts_tw, layer.base_frame)
        pts_cont = rotate_path_odd_layer(pts_cont, layer.base_frame)
        pts_larg = rotate_path_odd_layer(pts_larg, layer.base_frame)
    return pts_bridg, pts_tw, pts_cont, pts_larg


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
        folders.load_islands_hdf5(layer)
        for n_island, island in enumerate(layer.islands):
            counter = 0
            last_flag = 0
            flag_on = 0
            pts_bridg, pts_tw, pts_cont, pts_larg = region_points(
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


def points_from_region(
    layer_name, folders, island, zigzags=False, offsets=False, tw=False, bridges=False
):
    from components import points_tools as pt

    points = []
    region_list = []
    if bridges:
        folders.load_bridges_hdf5(layer_name, island)
        if hasattr(island, "bridges"):
            region_list = (
                island.bridges.cross_over_bridges + island.bridges.zigzag_bridges
            )
    if zigzags:
        from components.zigzag import ZigZag

        folders.load_zigzags_hdf5(layer_name, island)
        if hasattr(island, "zigzags"):
            region_list = island.zigzags.regions
            if hasattr(island.zigzags, "macro_areas_weaved"):
                for maw in island.zigzags.macro_areas_weaved:
                    region_list.append(ZigZag(0, maw, route=maw))
                # region_list += island.zigzags.macro_areas_weaved
    if offsets:
        folders.load_offsets_hdf5(layer_name, island)
        if hasattr(island, "offsets"):
            region_list = island.offsets.regions
    if tw:
        folders.load_thin_walls_hdf5(layer_name, island)
        if hasattr(island, "thin_walls"):
            region_list = island.thin_walls.regions
    for reg in region_list:
        image_routes = reg.route
        if hasattr(reg, "route_b"):
            if np.sum(reg.route_b) > 0:
                image_routes = np.logical_or(image_routes, reg.route_b)
        image_routes = mt.dilation(image_routes, kernel_size=6)
        points_poutes = pt.img_to_points(image_routes)
        # A2 = pt.img_to_points(mt.dilation(reg.route_b, kernel_size=6))
        # A1 = np.logical_or(A1, A2)
        for pnt in points_poutes:
            points.append(pnt)
    # aaaa = it.points_to_img(points, np.zeros_like(reg.img))
    return points


def skel_to_graph(sem_galhos, separation_degree):
    """Separates the graph into groups of connected nodes based on nodes with degree > degree.
    Parameters:(networkx.Graph): The input graph.
    Returns: list: A list of groups of connected nodes."""

    def condense_nodes(J, nodes, label):
        for i, a in enumerate(nodes):
            S = J.subgraph(a)
            coords = [0, 0]
            for j, c in enumerate(coords):
                coords[j] = int(sum([x[j] for x in a]) / len(a))
            # J.add_node(f"{label}{i}", data=pt.invert_x_y(a), weight=len(a), coords=pt.invert_x_y([coords])[0])
            J.add_node(
                f"{label}{i}", data=pt.invert_x_y(a), weight=len(a), coords=coords
            )
            for no in S.nodes:
                nbrs = set(J.neighbors(no))
                for nbr in nbrs - set([S]):
                    if f"{label}{i}" != nbr:
                        J.add_edge(f"{label}{i}", nbr)
        for i, a in enumerate(nodes):
            for n in a:
                if n in J.nodes:
                    J.remove_node(n)
        return J

    G = img_to_graph(one_pixel_wide(sem_galhos))
    G_copy = G.copy()
    H = G.copy()
    F = G.copy()
    trunks_pxls = []
    joints_pxls = []
    separators = [node for node in G.nodes() if G.degree(node) > separation_degree]
    for separator in separators:
        G_copy.remove_node(separator)
    components = list(nx.connected_components(G_copy))
    for lista in components:
        for point in lista:
            H.remove_node(point)
    not_components = list(nx.connected_components(H))
    trunks_pxls.extend(components)
    joints_pxls.extend(not_components)
    trunks_nodes = trunks_pxls
    junction_nodes = joints_pxls
    F = condense_nodes(F, trunks_nodes, "T")
    F = condense_nodes(F, junction_nodes, "J")
    B = [pt.invert_x_y(l) for l in trunks_nodes]
    # aaaa = it.sum_imgs([it.points_to_img(g, np.zeros_like(sem_galhos)) for g in B] + path_tools.one_pixel_wide(sem_galhos))
    aaaa = it.sum_imgs_colored(
        [it.points_to_img(g, np.zeros_like(sem_galhos)) for g in B]
    )
    # from matplotlib import pyplot
    # pyplot.gca().invert_yaxis()
    # pyplot.gca().invert_xaxis()
    # nx.draw(F, nx.get_node_attributes(F, 'coords'), with_labels=True)
    # F.nodes._nodes["J1"]
    return F, aaaa, trunks_pxls


def comprimento_da_trajetoria():
    with open("traj interna.txt") as f:
        lido = f.readlines()
        f.close()
    lido = [x.strip("\n") for x in lido]
    lido = [x.split(", ") for x in lido]
    lido = lido[:-1]
    lido = [[float(x[0]), float(x[1])] for x in lido]
    modulos = [math.sqrt((x[0] ** 2) + (x[1] ** 2)) for x in lido]
    comprimento = np.sum(modulos)
    print(f"comprimento da trajetoria={comprimento}")

    area_preench = 11  # mm² do imageJ
    raio_toroide = 37.5  # mm medido
    comp_traj = comprimento  # mm do codigo G calculado acima
    diam_fio = 1.2  # mm medido
    area_fio = math.pi * ((diam_fio / 2) ** 2)  # mm²
    vol_preench = 2 * math.pi * raio_toroide * area_preench
    Ws_Vd = vol_preench / (area_fio * comp_traj)
    print(f"Relação de velocidades:{Ws_Vd}")


def rotate_if_last_is_closest(points):
    """
    Recebe uma sequência de pontos [[y,x], ...].
    Mede qual ponto é mais próximo do último.
    Se for o primeiro, rotaciona a sequência para que o último seja o novo primeiro
    e o antigo primeiro se torne o segundo.
    """

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    last = points[-1]
    distances = [distance(last, p) for p in points[:-1]]
    closest_idx = np.argmin(distances)
    if closest_idx == 0:
        # Rotaciona: último vira primeiro, resto segue
        rotated = [last] + points[:-1]
        return rotated
    return points
