import itertools
from components import points_tools as pt
from components import morphology_tools as mt
from components import images_tools as it
import numpy as np
import networkx as nx

"""Parte do código direcionado para grafos e sequencias determinadas de pontos"""

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

def make_a_chain(im, div_point, path_radius, odd_layer, original_size, factor_epilson, cross_over_bridges=[], zigzag_bridges=[]):
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
                b.interruption_points = pt.x_y_para_pontos(np.nonzero(canvas_interruptions))
            print("depois: ", b.interruption_points)
            G.add_edge(tuple([b.interruption_points[0][1], b.interruption_points[0][0]]),
                       tuple([b.interruption_points[1][1], b.interruption_points[1][0]]), weight= 0.1)
            saltos.append(b.interruption_points)
    if zigzag_bridges:
        for b in zigzag_bridges:
            if odd_layer:
                print("antes: ", b.interruption_points)
                canvas_interruptions = np.zeros(original_size)
                for p in b.interruption_points:
                    canvas_interruptions[p[0], p[1]] = 1
                canvas_interruptions = it.rotate_img_ccw(canvas_interruptions)
                b.interruption_points = pt.x_y_para_pontos(np.nonzero(canvas_interruptions))
            print("depois: ", b.interruption_points)
            G.add_edge(tuple([b.interruption_points[0][1], b.interruption_points[0][0]]),
                       tuple([b.interruption_points[1][1], b.interruption_points[1][0]]), weight= 0.1)
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

def img_to_chain(img: np.ndarray, init_area=None, minimal_seq:int=0):
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
    h_ends = [tuple(n) for n in h_units + (0, 1)]  # end positions = start positions shifted by vector (1,0)
    horizontal_edges = zip(h_starts, h_ends)
    # CONSTRUCTION OF VERTICAL EDGES
    vy, vx = np.where(im[:, 1:] & im[:, :-1])  # vertical edge start positions
    v_units = np.array([vx, vy]).T
    v_starts = [tuple(n) for n in v_units]
    v_ends = [tuple(n) for n in v_units + (1, 0)]  # end positions = start positions shifted by vector (0,1)
    vertical_edges = zip(v_starts, v_ends)
    # CONSTRUCTION OF POSITIVE DIAGONAL EDGES
    pdy, pdx = np.where(im[1:][:, 1:] & im[:-1][:, :-1])  # vertical edge start positions
    pd_units = np.array([pdx, pdy]).T
    pd_starts = [tuple(n) for n in pd_units]
    pd_ends = [tuple(n) for n in pd_units + (1, 1)]  # end positions = start positions shifted by vector (1,1)
    positive_diagonal_edges = zip(pd_starts, pd_ends)
    # CONSTRUCTION OF NEGATIVE DIAGONAL EDGES
    ndy, ndx = np.where(im[:-1][:, 1:] & im[1:][:, :-1])  # vertical edge start positions
    ndx = ndx + 1
    nd_units = np.array([ndx, ndy]).T
    nd_starts = [tuple(n) for n in nd_units]
    nd_ends = [tuple(n) for n in nd_units + (-1, 1)]  # end positions = start positions shifted by vector (-1,1)
    negative_diagonal_edges = zip(nd_starts, nd_ends)
    G = nx.Graph()
    G.add_edges_from(horizontal_edges, weight=1)
    G.add_edges_from(vertical_edges, weight=1)
    G.add_edges_from(positive_diagonal_edges, weight=1)
    G.add_edges_from(negative_diagonal_edges, weight=1)
    return G

def remove_repeated_contours(multiple_lines_lists, canvas_size):
    cleaned_multiple_lines = multiple_lines_lists.copy()
    cleaned_multiple_lines.sort(key=len)
    remove_idx = []
    for a, b in list(itertools.combinations(cleaned_multiple_lines, 2)):
        a_img = it.points_to_img(a, np.zeros(canvas_size))
        b_img = it.points_to_img(b, np.zeros(canvas_size))
        repetitions_percent = np.sum(np.logical_and(a_img,b_img))/ len(b)
        if repetitions_percent > 0.8:
            remove_idx.append(cleaned_multiple_lines.index(a))
    for i in sorted(remove_idx, reverse=True):
        del cleaned_multiple_lines[i]
    return cleaned_multiple_lines

def make_a_chain_open_segment(im, ext_point) -> list:
    [start, end] = ext_point
    G = img_to_graph(im)
    path = nx.shortest_path(G, source=tuple(np.flip(start)), target=tuple(np.flip(end)))
    return path