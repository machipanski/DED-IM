from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from components.zigzag import ZigZag
    from typing import List
import itertools
import math
import random
import numpy as np
import networkx as nx
from components import points_tools as pt
from components import morphology_tools as mt
from components import images_tools as it
from components import skeleton as sk
from cv2 import boundingRect, arcLength, approxPolyDP
from scipy.spatial import distance_matrix


"""Parte do código direcionado para grafos e sequencias determinadas de pontos"""


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
        mask_line[:, int(mask_full_int.shape[0] / 2)] = 1
        a1 = zigzags[int(edge[0][1])]
        a1_vertical_trail = mt.dilation(a1.route, kernel_img=mask_line)
        a2 = zigzags[int(edge[1][1])]
        a2_vertical_trail = mt.dilation(a2.route, kernel_img=mask_line)
        interface = np.add(a1_vertical_trail, a2_vertical_trail) == 2
        return interface

    def vert_connection(a1, a2):
        a1 = zigzags[int(edge[0][1])]
        a1_trail = mt.dilation(a1.trail, kernel_size=1)
        a2 = zigzags[int(edge[1][1])]
        a2_trail = mt.dilation(a2.trail, kernel_size=1)
        interface = np.add(a1_trail, a2_trail) == 2
        return interface

    interfaces = []
    centers = []
    interface_types = []
    for edge in edges:
        has_bridge = False
        type_a1 = edge[0][0]
        type_a2 = edge[1][0]
        if type_a1 == "z" and type_a2 == "z":
            a1 = zigzags[int(edge[0][1])]
            a2 = zigzags[int(edge[1][1])]
            grow = 1
            interface = dilate_and_search(a1, a2, 1)
            while np.sum(interface) == 0:
                grow += 1
                interface = dilate_and_search(a1, a2, grow)
                if grow > path_radius_internal:
                    interface = vert_connection(a1, a2)
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


def make_offset_graph(filtered_regions):
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


def make_a_chain(
    im,
    div_point,
    path_radius,
    odd_layer,
    original_size,
    factor_epilson,
    cross_over_bridges=[],
    zigzag_bridges=[],
):
    saltos = []
    canvas = np.zeros_like(im)
    canvas[div_point[0], div_point[1]] = 1
    corte = mt.dilation(canvas, kernel_size=path_radius)
    im = np.logical_and(im, np.logical_not(corte))
    acrescimo = 1
    flagOk = 0
    while flagOk == 0:
        proximos = mt.dilation(canvas, kernel_size=(path_radius + acrescimo))
        pontos = np.add(im, proximos)
        pontos = pontos == 2
        candidates = pt.x_y_para_pontos(np.nonzero(pontos))
        if len(candidates) > 2:
            points = pt.most_distant(candidates)
            flagOk = 1
        elif len(candidates) == 2:
            points = candidates
            flagOk = 1
        else:
            acrescimo += 1
            print("nope")
    [start, end] = points
    # CONSTRUCTION OF HORIZONTAL EDGES
    G = img_to_graph(im)
    if cross_over_bridges:
        for b in cross_over_bridges:
            if odd_layer:
                print("antes: ", b.interruption_points)
                canvas_interruptions = np.zeros(original_size)
                for p in b.interruption_points:
                    canvas_interruptions[p[0], p[1]] = 1
                canvas_interruptions = it.rotate_img_ccw(canvas_interruptions)
                b.interruption_points = pt.x_y_para_pontos(
                    np.nonzero(canvas_interruptions)
                )
            print("depois: ", b.interruption_points)
            G.add_edge(
                tuple([b.interruption_points[0][1], b.interruption_points[0][0]]),
                tuple([b.interruption_points[1][1], b.interruption_points[1][0]]),
                weight=0.1,
            )
            saltos.append(b.interruption_points)
    if zigzag_bridges:
        for b in zigzag_bridges:
            if odd_layer:
                print("antes: ", b.interruption_points)
                canvas_interruptions = np.zeros(original_size)
                for p in b.interruption_points:
                    canvas_interruptions[p[0], p[1]] = 1
                canvas_interruptions = it.rotate_img_ccw(canvas_interruptions)
                b.interruption_points = pt.x_y_para_pontos(
                    np.nonzero(canvas_interruptions)
                )
            print("depois: ", b.interruption_points)
            G.add_edge(
                tuple([b.interruption_points[0][1], b.interruption_points[0][0]]),
                tuple([b.interruption_points[1][1], b.interruption_points[1][0]]),
                weight=0.1,
            )
            saltos.append(b.interruption_points)
    # pos = dict(zip(G.nodes(), G.nodes()))  # map node names to coordinates
    path = nx.shortest_path(G, source=tuple(np.flip(start)), target=tuple(np.flip(end)))
    chain = simplifica_retas_master(path, np.zeros_like(im), factor_epilson)
    novos_saltos = []
    if saltos:
        saltos_unpack = saltos
        chain_unpack = list(map(lambda x: x[0], chain))
        for segment in saltos_unpack:
            for p in segment:
                p_invert = [p[1], p[0]]
                if not (p_invert in chain_unpack):
                    novos_saltos.append(pt.closest_point(p_invert, chain_unpack)[0])
                else:
                    novos_saltos.append(p_invert)
    return chain, G, novos_saltos


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
    if len(seq_pts) == 0:
        return []
    sequence = seq_pts.copy()
    segmentos = np.array_split(sequence, np.where([(x in saltos) for x in sequence])[0])
    approx_seq = []
    for s in segmentos:
        candidates = []
        candidates.append(
            factor_epilson * arcLength(s, False)
        )  # diminuir valor para aumentar quantidade de segmentos
        candidates.append(
            factor_epilson * arcLength(s, True)
        )  # diminuir valor para aumentar quantidade de segmentos
        epsilon = candidates[np.argmax(candidates)]
        approx_seg = approxPolyDP(np.ascontiguousarray(s), epsilon, False)
        if approx_seg is None:
            pass
        else:
            approx_seq += approx_seg.tolist()
            approx_seq += [[["a", "a"]]]
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
