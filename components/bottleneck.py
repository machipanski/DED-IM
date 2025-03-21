from email.mime import base
import random
import re
from unittest import skip
import numpy as np
from components import morphology_tools as mt
from components import images_tools as it
from components import points_tools as pt
from components import skeleton as sk
from components import path_tools
import itertools
from components.offset import OffsetRegions, Region
from networkx import bridges, get_edge_attributes
from cv2 import getStructuringElement, MORPH_RECT
from typing import List
import copy
from components.skeleton import prune
from skimage.morphology import disk
import networkx as nx
import concurrent.futures
from scipy.ndimage import distance_transform_edt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from components.timer import Timer
from skimage.feature import corner_harris, corner_subpix, corner_peaks


class Bottleneck:

    def __init__(
        self,
        name,
        img,
        origin,
        trunk,
        n_paths,
        origin_marks,
        elementos_contorno=None,
        pontos_extremos=None,
        linked_offset_regions=None,
        linked_zigzag_regions=None,
    ):
        if elementos_contorno is None:
            elementos_contorno = []
        if pontos_extremos is None:
            pontos_extremos = []
        if linked_offset_regions is None:
            linked_offset_regions = []
        if linked_zigzag_regions is None:
            linked_zigzag_regions = []
        self.name = name
        self.img = img
        self.origin = origin
        self.destiny = 0
        self.n_paths = n_paths
        self.origin_mark = origin_marks
        self.trunk = trunk
        self.contorno = elementos_contorno
        self.pontos_extremos = pontos_extremos
        self.origin_coords = []
        self.destiny_coords = []
        self.route = []
        self.trail = []
        self.center = []
        self.interruption_points = []
        self.reference_points = []
        self.linked_offset_regions = linked_offset_regions
        self.linked_zigzag_regions = linked_zigzag_regions
        return

    def make_offset_bridge_route(
        self, offsets_regions, path_radius, base_frame, rest_of_picture
    ):
        all_offsets = it.sum_imgs([x.img for x in offsets_regions])
        _, outer = mt.detect_contours(self.img, return_img=True)
        objective_lines = np.logical_and(
            outer.astype(np.uint8),
            np.logical_not(mt.dilation(all_offsets, kernel_size=1).astype(np.uint8)),
        )
        square_mask = getStructuringElement(
            MORPH_RECT, (int(path_radius * 2), int(path_radius * 2))
        )
        objective_lines_dilated = mt.dilation(objective_lines, kernel_img=square_mask)
        outer_offseted = np.logical_and(
            self.img, np.logical_not(objective_lines_dilated)
        )
        outer_offseted = it.take_the_bigger_area(outer_offseted)
        _, outer_new = mt.detect_contours(outer_offseted, return_img=True)
        objective_lines_new = np.logical_and(
            outer_new.astype(np.uint8),
            np.logical_not(mt.dilation(all_offsets, kernel_size=1).astype(np.uint8)),
        )
        self.route = objective_lines_new
        self.trail = mt.dilation(self.route, kernel_size=path_radius)
        return


class Bridge:

    def __init__(self, *args, **kwargs):
        self.contorno = []
        self.pontos_extremos = []
        self.linked_offset_regions = []
        self.linked_zigzag_regions = []
        self.origin_coords = []
        self.destiny_coords = []
        self.route = []
        self.trail = []
        self.route_b = []
        self.trail_b = []
        self.center = []
        self.interruption_points = []
        self.reference_points = []
        self.reference_points_b = []
        self.destiny = 0
        self.type = []
        if args:
            self.name = args[0]
            self.img = args[1]
            self.origin = args[2]
            self.trunk = args[3]
            self.n_paths = args[4]
            self.origin_mark = args[5]
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        return

    def make_internal_border(
        self, internal, filled_external_borders, eroded, origin_axis, path_radius_bridg
    ):
        def make_closest_path_to(
            internal_borders, original_line, start_orig, end_orig, internal_extreme
        ):
            distance_map = np.multiply(
                internal_borders, distance_transform_edt(np.logical_not(original_line))
            )
            internal_borders_pts = pt.img_to_points(internal_borders)
            start_ci, _ = pt.closest_point(start_orig, internal_extreme)
            internal_extreme.remove(start_ci)
            end_ci, _ = pt.closest_point(end_orig, internal_extreme)
            internal_extreme.remove(end_ci)
            G = path_tools.img_to_graph_com_distancias(distance_map)
            _, path1 = nx.bidirectional_dijkstra(G, tuple(start_ci), tuple(end_ci))
            linha_ci = it.points_to_img(path1, np.zeros_like(internal_borders))
            return linha_ci

        internal = np.logical_or(internal, origin_axis)
        _, internal_borders = mt.detect_contours(internal, return_img=True)
        internal_borders = np.logical_and(internal_borders, filled_external_borders)
        _, labeled, labeled_n = it.divide_by_connected(internal_borders)
        linha_ci1 = np.zeros_like(self.img)
        linha_ci2 = np.zeros_like(self.img)
        if labeled_n == 1:
            internal_extreme = pt.img_to_points(mt.hitmiss_ends_v2(internal_borders))
            if len(internal_extreme)==0:
                possible_c1_c2, counter_accepted, internal_extreme = decompose_pol_cont_by_corners(internal_borders, origin_axis, path_radius_bridg)
                # bbb = it.sum_imgs([tng_end,tng_start,all_borders_img])
                labeled = possible_c1_c2
                labeled_n = counter_accepted
            linha_ci1 = make_closest_path_to(
                    internal_borders,
                    self.contorno[0],
                    self.pontos_extremos[0],
                    self.pontos_extremos[1],
                    internal_extreme,
                )
            if len(internal_extreme) >= 2:
                linha_ci2 = make_closest_path_to(
                    internal_borders,
                    self.contorno[1],
                    self.pontos_extremos[2],
                    self.pontos_extremos[3],
                    internal_extreme,
                )
            else:
                linha_ci2 = linha_ci1
        elif labeled_n > 2:
            sums = []
            for l in np.arange(0, labeled_n):
                sums.append(np.sum(labeled == l + 1))
            idx = [sums.index(i) for i in sorted(sums, reverse=True)][:2]
            linha_ci1 = labeled == idx[0] + 1
            linha_ci2 = labeled == idx[1] + 1
        elif labeled_n == 2:
            linha_ci1 = labeled == 1
            linha_ci2 = labeled == 2
        internal_borders_closed = np.add(
            internal_borders, np.logical_or(self.contorno[2], self.contorno[3])
        )
        internal_borders_closed = np.logical_and(internal, internal_borders_closed)
        _, labeled, labeled_n = it.divide_by_connected(internal_borders_closed)
        if labeled_n > 1:
            tri_1 = labeled == 1
            tri_1 = it.fill_internal_area(tri_1.astype(np.uint8), np.ones_like(tri_1))
            tri_2 = labeled == 2
            tri_2 = it.fill_internal_area(tri_2.astype(np.uint8), np.ones_like(tri_2))
            new_fig = np.logical_or(tri_1, tri_2)
            new_fig = np.logical_or(new_fig, self.origin.astype(bool))
            new_fig = np.logical_and(new_fig, filled_external_borders)
            _, internal_borders_closed = mt.detect_contours(new_fig, return_img=True)
        linha_ci1 = np.logical_and(eroded, linha_ci1)
        linha_ci1, _, _ = sk.create_prune_skel(linha_ci1, 1)
        linha_ci2 = np.logical_and(eroded, linha_ci2)
        linha_ci2, _, _ = sk.create_prune_skel(linha_ci2, 1)
        if np.sum(linha_ci1) == 0:
            print("jsnsdjfs")
        if np.sum(linha_ci2) == 0:
            print("jsnsdjfs")
        return linha_ci1, linha_ci2

    def find_center(self, base_frame):
        contour = mt.detect_contours(self.img)
        contour = pt.contour_to_list(contour)
        pt.points_center(contour)

    def get_linked_offsets(self, offset_regions):
        linked_offsets = []
        for offset_region in offset_regions:
            combined_imgs = np.logical_or(self.img, offset_region.img)
            _, _, num = it.divide_by_connected(combined_imgs)
            if num == 1:
                linked_offsets.append(offset_region.name)
        self.linked_offset_regions = linked_offsets

    def get_linked_zigzags(self, zigzag_regions):
        linked_zigzags = []
        for zr in zigzag_regions:
            combined_imgs = np.logical_or(self.img, zr.img)
            _, _, num = it.divide_by_connected(combined_imgs)
            if num == 1:
                linked_zigzags.append(zr.name)
        self.linked_zigzag_regions = linked_zigzags


class BridgeRegions:
    """Lista de listas das diferentes regiões de ponte na camada"""

    def __init__(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            self.medial_transform = []
            self.offset_bridges: List[Bridge] = []
            self.zigzag_bridges: List[Bridge] = []
            self.cross_over_bridges: List[Bridge] = []
            self.all_bridges = []
            self.all_origins = []
            self.routes = []
        return

    def make_offset_bridges(
        self,
        rest_of_picture,
        offsets_regions: List[Region],
        base_frame,
        path_radius,
        original_img,
        prohibited_areas,
    ):
        """determina pontos de conexão entre os diferentes contornos,
        traçando uma ponte no sentido do offset da camada"""
        offreg: List[Region] = offsets_regions.regions
        regs_touching = []
        for region in offreg:
            # desenha o centro, contorno, área interna e pontos extremos de cada região
            region.make_contour(base_frame)
            region.make_internal_area_and_center(original_img)
            region.make_limmit_coords(path_radius)
        for region, other_region in list(itertools.permutations(offreg, 2)):
            if it.esta_contido(region.internal_area, other_region.internal_area):
                region.hierarchy += 1
            _, aaaa, num = it.divide_by_connected(
                np.logical_or(
                    mt.dilation(region.img, kernel_size=int(path_radius / 2)),
                    mt.dilation(other_region.img, kernel_size=int(path_radius / 2)),
                )
            )
            if num == 1:
                regs_touching.append(set([region.name, other_region.name]))
        regs_touching = set(tuple(sorted(p)) for p in regs_touching)
        external_areas = list(filter(lambda x: (x.hierarchy == 0), offreg))
        internal_areas = list(filter(lambda x: (x.hierarchy >= 1), offreg))
        for region in external_areas:
            area_internal_contour, area_internal_contour_img = (
                region.out_area_inner_contour(base_frame)
            )
        for region in internal_areas:
            region.make_paralel_points(
                offreg, area_internal_contour_img, prohibited_areas, path_radius
            )
            # encontra pontes para cada ponto extremo do contorno
        areas_graph = path_tools.make_offset_graph(offreg, regs_touching)
        offsets_paralel_mst, paralel_sequence = path_tools.regions_mst(areas_graph)

        bridge_imgs = self.draw_offset_paralel_links(
            offsets_paralel_mst,
            paralel_sequence,
            rest_of_picture,
            base_frame,
            path_radius,
            offreg,
        )
        if len(bridge_imgs)> 0 :
            self.all_bridges = it.sum_imgs(bridge_imgs)
        return areas_graph, offsets_paralel_mst

    def draw_offset_paralel_links(
        self,
        offsets_paralel_mst,
        paralel_sequence,
        rest_of_picture,
        base_frame,
        path_radius,
        offsets_regs,
    ):
        counter = 0
        lista_origem = get_edge_attributes(offsets_paralel_mst, "coord_origem")
        lista_destino = get_edge_attributes(offsets_paralel_mst, "coord_destino")
        lista_tipo = get_edge_attributes(offsets_paralel_mst, "extremo_origem")
        separated_imgs = []
        for i, line in enumerate(list(paralel_sequence)):
            img = np.zeros_like(rest_of_picture)
            mask_line = np.zeros((4 * path_radius, 4 * path_radius))
            mask_line[:, int(path_radius * 2)] = 1
            mask_square = np.ones((2 * path_radius, 2 * path_radius))
            if lista_tipo[line] != "e":
                pontos_origem = sorted(
                    [lista_origem[line], lista_destino[line]],
                    key=lambda x: [x[1], x[0]],
                )
                ponto_origem = [
                    pontos_origem[0][0],
                    pontos_origem[0][1] - (2 * path_radius),
                ]
                ponto_destino = [
                    pontos_origem[1][0],
                    pontos_origem[1][1] + (0 * path_radius),
                ]
                origin = it.draw_line(np.zeros(base_frame), ponto_origem, ponto_destino)
                img = mt.dilation(origin, kernel_img=mask_line)
                img = np.logical_and(img, rest_of_picture)
                self.offset_bridges.append(
                    Bridge(f"OB_{counter:03d}", img, origin, [], 2, [])
                )
                self.offset_bridges[-1].origin_coords = lista_origem[line]
                self.offset_bridges[-1].destiny_coords = lista_destino[line]
                self.offset_bridges[-1].linked_offset_regions = [line[0], line[1]]
                self.offset_bridges[-1].img = img
                self.offset_bridges[-1].type = "common_offset_bridge"
                counter += 1
                separated_imgs.append(img)
            else:  # quando é do tipo "colado"
                reg_a = [x for x in offsets_regs if x.name == line[0]][0]
                reg_b = [x for x in offsets_regs if x.name == line[1]][0]
                sum = 0
                divisor = 4
                while sum == 0 and divisor > 0:
                    union = mt.closing(
                        it.sum_imgs(
                            [reg_a.img.astype(np.uint8), reg_b.img.astype(np.uint8)]
                        ),
                        kernel_size=path_radius / divisor,
                    )
                    img = mt.erosion(union, kernel_size=path_radius / 2)
                    img = mt.opening(img, kernel_size=path_radius)
                    reg_a_routes = it.sum_imgs([x.route for x in reg_a.loops])
                    reg_b_routes = it.sum_imgs([x.route for x in reg_b.loops])
                    bbb = np.add(img, it.sum_imgs([reg_a_routes, reg_b_routes]))
                    eee = bbb == 2
                    linhas_separadas, _, _ = it.divide_by_connected(eee)
                    sum = np.sum(eee)
                    divisor = divisor - 1
                pontos_centrais = [
                    pt.points_center(pt.img_to_points(x)) for x in linhas_separadas
                ]
                origin = it.draw_line(
                    np.zeros(base_frame), pontos_centrais[0], pontos_centrais[1]
                )
                new_img = mt.dilation(origin, kernel_img=mask_square)
                self.offset_bridges.append(
                    Bridge(f"OB_{counter:03d}", img, origin, [], 2, [])
                )
                self.offset_bridges[-1].origin_coords = pontos_centrais[0]
                self.offset_bridges[-1].destiny_coords = pontos_centrais[1]
                self.offset_bridges[-1].linked_offset_regions = [line[0], line[1]]
                self.offset_bridges[-1].img = np.logical_and(new_img, rest_of_picture)
                self.offset_bridges[-1].type = "contact_offset_bridge"
                separated_imgs.append(img)
                counter += 1
        return separated_imgs

    def make_zigzag_bridges(
        self,
        rest_of_picture,
        base_frame,
        path_radius_bridg,
        necks_max_paths,
        offset_regions,
    ):

        def separate_trunks():
            """Quebra a imagem em suas componentes do MAT - já devolve nomralizado para o numero de rotas que cabem no sentido paralelo"""
            sem_galhos, sem_galhos_dist, trunks = sk.create_prune_divide_skel(
                rest_of_picture.astype(np.uint8), path_radius_bridg
            )
            self.medial_transform = sem_galhos * sem_galhos_dist
            # trunks = [pt.contour_to_list([x]) for x in trunks]
            trunks = [it.points_to_img(x, np.zeros(base_frame)) for x in trunks]
            trunks = it.eliminate_duplicates(trunks)
            normalized_dist_map = sem_galhos_dist / path_radius_bridg
            normalized_trunks = [trunk * normalized_dist_map for trunk in trunks]
            return normalized_trunks, normalized_dist_map

        def break_too_big_parts(normalized_trunks, normalized_dist_map):
            """Divide ainda mais os trunks para evitar regiões grandes demais"""
            minus_bigger_than_2wd = []
            for trunk in normalized_trunks:
                less_than_2wd = np.logical_and(trunk > 0, trunk <= 2 * necks_max_paths)
                sep_trunks, _, num = it.divide_by_connected(less_than_2wd)
                for new_trunk in sep_trunks:
                    if len(pt.img_to_points(new_trunk)) > path_radius_bridg * 2:
                        minus_bigger_than_2wd.append(
                            new_trunk.astype(float) * normalized_dist_map
                        )
            return minus_bigger_than_2wd

        def separate_truks_with_botlenecks(minus_bigger_than_2wd):
            """Retorna apenas os trunks que tem trechos de estrangulamento"""
            n_trilhas_minima_by_trunk = [
                (np.unique(trunk))[1] for trunk in minus_bigger_than_2wd
            ]
            return [
                minus_bigger_than_2wd[i]
                for i, x in enumerate(n_trilhas_minima_by_trunk)
                if x < necks_max_paths
            ]

        def close_bridges(norm_reduced_origins, initial_points, skel):
            self.all_bridges = np.zeros(base_frame)
            self.all_origins = np.zeros(base_frame)
            for i, candidate in enumerate(norm_reduced_origins):
                inicial_pnt = initial_points[i]
                try:
                    bridge_img, elementos_contorno, contorno, pontos_extremos = (
                        close_bridge_contour(
                            candidate,
                            base_frame,
                            1.5 * necks_max_paths,  # TODO ver esse valor aqui
                            rest_of_picture,
                            path_radius_bridg,
                            inicial_pnt,
                        )
                    )
                    print("Fechou uma ponte OK")
                except Exception:
                    print("Erro: nao fechou ponte")
                    bridge_img = []
                    pass
                counter = 0
                if np.sum(bridge_img) > 0:
                    if len(self.all_bridges) > 0:
                        self.all_bridges = np.logical_or(self.all_bridges, bridge_img)
                    else:
                        self.all_bridges = np.zeros(base_frame)
                    if len(self.all_origins) > 0:
                        self.all_origins = np.logical_or(self.all_origins, candidate)
                    else:
                        self.all_origins = np.zeros(base_frame)
                    y_mark = np.where(candidate)[1][np.round(len(np.where(candidate)))]
                    x_mark = np.where(candidate)[0][np.round(len(np.where(candidate)))]
                    origin_mark = [y_mark, x_mark, str("não usado")]
                    self.zigzag_bridges.append(
                        Bridge(
                            f"ZB_{counter:03d}",
                            bridge_img,
                            # np.logical_and(bridge_img, candidate).astype(np.uint8),
                            np.logical_and(bridge_img, skel).astype(np.uint8),
                            candidate,
                            [],
                            origin_mark,
                            contorno=elementos_contorno,
                            pontos_extremos=pontos_extremos,
                        )
                    )
                    self.zigzag_bridges[-1].get_linked_offsets(offset_regions)
                    counter += 1
            return

        norm_trunks, norm_dist_map = separate_trunks()
        minus_bigger_than_2wd = break_too_big_parts(norm_trunks, norm_dist_map)
        origin_candidates = separate_truks_with_botlenecks(minus_bigger_than_2wd)
        reduced = [
            reduce_origin(x, necks_max_paths, norm_dist_map) for x in origin_candidates
        ]
        norm_reduced_origins = [y[0] for y in reduced]
        initial_points = [x[1] for x in reduced]
        close_bridges(norm_reduced_origins, initial_points, self.medial_transform)
        return

    def make_cross_over_bridges(self, prohibited_areas, offsets_mst):
        """Une regiões dos pescoços às pontes de offset para possibilitar uma rota que cubra toda a área"""
        counter = 0
        combinations = list(itertools.product(self.zigzag_bridges, self.offset_bridges))
        substitutions = []
        for n, [zigzag_bridge, offset_bridge] in enumerate(combinations):
            if np.equal(zigzag_bridge.contorno[0], zigzag_bridge.contorno[1]).all():
                print("Aqui eu não deixei a parede unica virar crossover")
            elif set(zigzag_bridge.linked_offset_regions) == set(
                offset_bridge.linked_offset_regions
            ):
                # se as pontes conectam os mesmos contornos, organiza a prioridade deles e paga a mais alta pra cada grupo
                priority = 0
                sobreposition = pt.img_to_points(
                    np.logical_and(offset_bridge.img, zigzag_bridge.img)
                )
                if len(sobreposition) > 0:
                    priority = 10 * len(sobreposition)
                else:
                    xs_offset_b = np.unique(
                        [x[1] for x in pt.img_to_points(offset_bridge.img)]
                    )
                    xs_zigzag_b = np.unique(
                        [x[1] for x in pt.img_to_points(zigzag_bridge.img)]
                    )
                    coincidencia = np.intersect1d(xs_zigzag_b, xs_offset_b)
                    if len(coincidencia) > 0:
                        priority = len(coincidencia)
                substitutions.append(
                    [offset_bridge.linked_offset_regions, combinations[n], priority]
                )

        substitutions_filtradas = []
        listas_mesmos_elementos = {}
        for i, sublista in enumerate([x[0] for x in substitutions]):
            tupla_sublista = tuple(sublista)
            if tupla_sublista in listas_mesmos_elementos:
                listas_mesmos_elementos[tupla_sublista].append(i)
            else:
                listas_mesmos_elementos[tupla_sublista] = [i]
        for elementos, posicoes in listas_mesmos_elementos.items():
            maior_prioridade = posicoes[
                np.argmax([substitutions[x][2] for x in posicoes])
            ]
            print("Elemento:", elementos, "Maior prioridade:", maior_prioridade)
            substitutions_filtradas.append(substitutions[maior_prioridade][1])
        for zigzag_bridge, offset_bridge in substitutions_filtradas:
            origin_marks = zigzag_bridge.origin_mark
            self.cross_over_bridges.append(
                Bridge(
                    f"CB_{counter:03d}",
                    zigzag_bridge.img,
                    zigzag_bridge.origin,
                    zigzag_bridge.trunk,
                    zigzag_bridge.n_paths,
                    origin_marks,
                    contorno=zigzag_bridge.contorno,
                    pontos_extremos=zigzag_bridge.pontos_extremos,
                    linked_offset_regions=zigzag_bridge.linked_offset_regions,
                )
            )
            if zigzag_bridge in self.zigzag_bridges:
                self.zigzag_bridges.remove(zigzag_bridge)
            if offset_bridge in self.offset_bridges:
                self.offset_bridges.remove(offset_bridge)
            for i, zigzag_bridge in enumerate(self.zigzag_bridges):
                zigzag_bridge.name = f"ZB_{i:03d}"
            for j, offset_bridge in enumerate(self.offset_bridges):
                offset_bridge.name = f"OB_{j:03d}"
            counter += 1
        all_bridges = np.zeros_like(self.all_origins)
        for region in (
            self.zigzag_bridges + self.offset_bridges + self.cross_over_bridges
        ):
            all_bridges = it.sum_imgs([all_bridges, region.img])
        return all_bridges

    def make_routes_b(
        self,
        offsets_regions,
        path_radius_cont,
        path_radius_bridg,
        mask_distancer,
        internal_mask_dist,
        base_frame,
        rest_of_picture,
    ):
        """Chama a função make_route() para cada região"""
        with Timer("Making Offset bridges routes"):
            processed_regions_ob = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(
                        make_offset_bridge_route,
                        region,
                        offsets_regions,
                        path_radius_cont,
                        base_frame,
                    )
                    for region in self.offset_bridges
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions_ob.append(l.result())
            processed_regions_ob.sort(key=lambda x: x.name)
            self.offset_bridges = processed_regions_ob
        with Timer("Making Zigzag bridges routes"):
            processed_regions_zb = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(
                        make_zz_or_co_bridge_route,
                        region,
                        path_radius_bridg,
                        mask_distancer,
                        internal_mask_dist,
                        rest_of_picture,
                        base_frame,
                    )
                    for region in self.zigzag_bridges
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions_zb.append(l.result())
            processed_regions_zb.sort(key=lambda x: x.name)
            self.zigzag_bridges = processed_regions_zb
        with Timer("Making Crossover bridges routes"):
            processed_regions_cob = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(
                        make_zz_or_co_bridge_route,
                        region,
                        path_radius_bridg,
                        mask_distancer,
                        rest_of_picture,
                        base_frame,
                    )
                    for region in self.cross_over_bridges
                ]
                for l in concurrent.futures.as_completed(results):
                    processed_regions_cob.append(l.result())
            processed_regions_cob.sort(key=lambda x: x.name)
            self.cross_over_bridges = processed_regions_cob

        self.routes = np.zeros(base_frame)
        for x in self.zigzag_bridges + self.offset_bridges + self.cross_over_bridges:
            self.routes = it.sum_imgs([self.routes, x.route])
        return

    def apply_bridges(self, rest_of_picture, base_frame):
        rest_of_picture_f3 = np.zeros(base_frame)
        rest_of_picture_f3 = np.logical_or(rest_of_picture, rest_of_picture_f3)
        all_bridges_regions = [
            x.img
            for x in self.offset_bridges + self.zigzag_bridges + self.cross_over_bridges
        ]
        for bridge_region in all_bridges_regions:
            rest_of_picture_f3 = np.logical_and(
                rest_of_picture_f3, np.logical_not(bridge_region)
            )
        return rest_of_picture_f3


def connect_origin_parts(origin, eroded):
    extreme = pt.img_to_points(mt.hitmiss_ends_v2(origin))
    fila = path_tools.img_to_chain(origin)[0]
    fila = path_tools.set_first_pt_in_seq(fila, extreme[0])
    fila = path_tools.cut_repetition(fila)
    # borda_cortada = np.zeros_like(origin)
    borda_normal = np.zeros_like(origin)
    # counter = 0
    counter_pixels = 0
    first_cross_point = []
    last_cross_point = []
    for i in np.arange(0, len(fila)):
        borda_normal[fila[i][0]][fila[i][1]] = 1
        counter_pixels += 1
        y = fila[i][0]
        x = fila[i][1]
        ca = eroded[y][x].astype(bool)
        if ca:
            if len(first_cross_point) == 0:
                first_cross_point = [y, x]
            else:
                last_cross_point = [y, x]
    start_idx = fila.index(first_cross_point)
    end_idx = fila.index(last_cross_point)
    new_fila = fila[start_idx:end_idx]
    origin_axis_reconected = it.points_to_img(new_fila, np.zeros_like(origin))
    new_eroded = np.logical_or(origin_axis_reconected, eroded)
    new_eroded = mt.closing(new_eroded, kernel_size=2)
    return origin_axis_reconected, new_eroded

def reduce_lines_overshoot(candidate, origin_points):
    """Determina inicio e fim para cada trunk, então vai reduzindo as margens até englobar todos os estrangulamentos dele"""
    t_ends = pt.img_to_points(sk.find_tips(candidate.astype(bool)))
    # if len(t_ends) == 0:  # se for um ciclo fechado
    #     start_pnt = random.choice(pt.x_y_para_pontos(np.nonzero(candidate)))
    #     origin_chain = pt.invert_x_y(
    #         path_tools.make_a_chain(candidate.astype(bool), start_pnt)
    #     )
    #     origin_chain_img = it.points_to_img(origin_chain, np.zeros_like(candidate))
    #     new_ends = pt.img_to_points(sk.find_tips(origin_chain_img.astype(bool)))
    #     origin_chain = path_tools.set_first_pt_in_seq(origin_chain, new_ends[0])
    #     candidate = np.multiply(candidate, origin_chain_img)
    # else:  # se for um trunk aberto
    # start_pnt = []
    origin_chain = pt.invert_x_y(
        path_tools.make_a_chain_open_segment(candidate.astype(bool), t_ends)
    )
    # reduced_origin = np.logical_and(candidate != 0, candidate < necks_max_paths)
    new_ends = [pt.closest_point(x, origin_chain)[0] for x in origin_points] 
    new_origin = origin_chain.copy()
    count_up = 0
    count_down = -1
    start_flag = 0
    end_flag = 0
    while not (start_flag and end_flag):
        current_pt_1 = origin_chain[count_up]
        current_pt_2 = origin_chain[count_down]
        if current_pt_1 in new_ends:
            start_flag = 1
        else:
            new_origin.remove(current_pt_1)
            count_up += 1
        if current_pt_2 in new_ends:
            end_flag = 1
        else:
            new_origin.remove(current_pt_2)
            count_down -= 1
    reduced_origin = it.points_to_img(new_origin, np.zeros_like(candidate))
    return reduced_origin

def close_bridge_contour(
    trunk,
    base_frame,
    max_accepted,
    rest_of_picture,
    path_radius_bridg,
    inicial_pnt,
):
    def find_contours_around_origin(
        rest_of_picture, base_frame, max_accepted, path_radius_bridg, trunk
    ):
        all_borders, all_borders_img = mt.detect_contours(
            rest_of_picture, return_img=True
        )
        # area_pescocal = mt.dilation(trunk.astype(bool), kernel_size=(dist + 1.5 * path_radius))
        area_pescocal = mt.dilation(
            trunk.astype(bool), kernel_size=(max_accepted * path_radius_bridg)
        )
        overlap = np.add(area_pescocal, all_borders_img)
        linhas_do_limite = overlap == 2
        _, labeled, labeled_n = it.divide_by_connected(linhas_do_limite)
        if labeled_n == 1:
            print("teste: caso de uma unica linha no entorno da origem")
            possible_c1_c2, counter_accepted, pontos_curvatura = decompose_pol_cont_by_corners(linhas_do_limite, trunk, path_radius_bridg)
            # bbb = it.sum_imgs([tng_end,tng_start,all_borders_img])
            labeled = possible_c1_c2
            labeled_n = counter_accepted
            # print("pausadsddass")
        if labeled_n > 2:
            dists = []
            trunk_pts = pt.x_y_para_pontos(np.nonzero(trunk))
            trunk_center = pt.points_center(trunk_pts)
            trunk_center_pt, _ = pt.closest_point(trunk_center, trunk_pts)
            for l in np.arange(1, labeled_n + 1):
                line_pts = pt.x_y_para_pontos(np.nonzero(labeled == l))
                _, dist = pt.closest_point(trunk_center_pt, line_pts)
                dists.append(dist)
            lista_dist = dists.copy()
            idx1 = np.argmin(lista_dist)
            lista_dist[idx1] = 999999
            idx2 = np.argmin(lista_dist)
            linha1 = labeled == idx1 + 1
            #ATENçÂO PARA NOVOS CASOS AQUI!!!!!!!
            linha2 = labeled == idx2 + 1
        elif labeled_n == 2:
            linha1 = labeled == 1
            linha2 = labeled == 2
            # linha1 = labeled == 1
            return linha1, linha2
        else:
            print("ERRO: não haviam paredes no entorno da origem!")
            return np.zeros_like(trunk), np.zeros_like(trunk)
        points_trunk = pt.img_to_points(mt.hitmiss_ends_v2(trunk.astype(bool)))
        points_line1 = pt.img_to_points(mt.hitmiss_ends_v2(linha1.astype(bool)))
        if len(points_line1)>0:
            linha1 = reduce_lines_overshoot(linha1, points_trunk)
        points_line2 = pt.img_to_points(mt.hitmiss_ends_v2(linha2.astype(bool)))
        if len(points_line2)>0:
            linha2 = reduce_lines_overshoot(linha2, points_trunk)
        return linha1, linha2
    
    def close_area_from_lines(
        linha1: np.ndarray, 
        linha2: np.ndarray, 
        base_frame, 
        inicial_pnt,
    ):
        inicios_e_fins1 = pt.x_y_para_pontos(
            np.where(sk.find_tips(linha1.astype(np.uint8)))
        )
        inicios_e_fins2 = pt.x_y_para_pontos(
            np.where(sk.find_tips(linha2.astype(np.uint8)))
        )
        # Se alguma delas é um circulo fechado, interrompe perto da origem indicada poe Inicial_pt
        if len(inicios_e_fins1) == 0:
            line_pts = pt.img_to_points(linha1.astype(np.uint8))
            break_point, _ = pt.closest_point(inicial_pnt, line_pts)
            # linha1[break_point[0]][break_point[1]] = 0
            origin_chain = pt.invert_x_y(path_tools.make_a_chain(linha1, break_point))
            linha1 = it.points_to_img(origin_chain, np.zeros_like(linha1))
            inicios_e_fins1 = pt.x_y_para_pontos(
                np.where(sk.find_tips(linha1.astype(np.uint8)))
            )
        if len(inicios_e_fins2) == 0:
            line_pts = pt.img_to_points(linha2.astype(np.uint8))
            break_point, _ = pt.closest_point(inicial_pnt, line_pts)
            # linha2[break_point[0]][break_point[1]] = 0
            origin_chain = pt.invert_x_y(path_tools.make_a_chain(linha2, break_point))
            linha2 = it.points_to_img(origin_chain, np.zeros_like(linha2))
            inicios_e_fins2 = pt.x_y_para_pontos(
                np.where(sk.find_tips(linha2.astype(np.uint8)))
            )
        # Se houverem pontos demais por causa dos contornos, faz uma poda
        if len(inicios_e_fins1) > 2:
            linha1, _, _ = prune(skel_img=linha1, size=2)
            inicios_e_fins1 = pt.x_y_para_pontos(
                np.where(sk.find_tips(linha1.astype(np.uint8)))
            )
        if len(inicios_e_fins2) > 2:
            linha2, _, _ = prune(skel_img=linha2, size=2)
            inicios_e_fins2 = pt.x_y_para_pontos(
                np.where(sk.find_tips(linha2.astype(np.uint8)))
            )
        # Se as duas linhas coincidem o final e o começo
        if inicios_e_fins1 == inicios_e_fins2:
            linha1 = linha2
            pontos_fins = mt.hitmiss_ends_v2(linha1)
            pontos_fins = pt.img_to_points(pontos_fins)
            if len(pontos_fins) == 2:
                linhabaixo = linhatopo = it.draw_line(
                    np.zeros(base_frame), inicios_e_fins1[0], inicios_e_fins1[1]
                )
                bridge_border = (
                    it.sum_imgs([linha1, linhatopo, linha2, linhabaixo]) >= 1
                )
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(pontos_fins) > 2:
                bridge_border = it.draw_polyline(
                    np.zeros(base_frame), pontos_fins, closed=True
                )
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
                linhatopo = linhabaixo = np.zeros_like(linha1)
            else:
                print("ai fodeu")
        else:
            unique_points = []
            for p in inicios_e_fins1 + inicios_e_fins2:
                if p not in unique_points:
                    unique_points.append(p)
            if len(unique_points) == 4:
                extreme_points = [
                    inicios_e_fins1[0],
                    inicios_e_fins1[1],
                    inicios_e_fins2[1],
                    inicios_e_fins2[0],
                ]
                if pt.intersects(
                    [extreme_points[0], extreme_points[2]],
                    [extreme_points[1], extreme_points[3]],
                ):
                    extreme_points = [
                        inicios_e_fins1[0],
                        inicios_e_fins1[1],
                        inicios_e_fins2[0],
                        inicios_e_fins2[1],
                    ]
                linhatopo = it.draw_line(
                    np.zeros(base_frame), extreme_points[0], extreme_points[2]
                )
                linhabaixo = it.draw_line(
                    np.zeros(base_frame), extreme_points[1], extreme_points[3]
                )
                bridge_border = it.sum_imgs([linha1, linhatopo, linha2, linhabaixo])
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(unique_points) == 2:
                fechamento1_pts = unique_points
                linhatopo = it.draw_line(
                    np.zeros(base_frame), fechamento1_pts[0], fechamento1_pts[1]
                )
                linhabaixo = np.zeros_like(linhatopo)
                bridge_border = it.sum_imgs([linha1, linhatopo])
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(unique_points) == 0:
                if np.sum(linha1) > 0:
                    fechamento1_pts = unique_points
                    linhatopo = np.zeros_like(linha1)
                    linhabaixo = np.zeros_like(linha1)
                    bridge_border = linha1.copy()
                    bridge_img = it.fill_internal_area(
                        bridge_border, np.ones(base_frame)
                    )
                    bridge_img = np.logical_and(bridge_img, rest_of_picture)
        return bridge_img, linhatopo, linhabaixo, bridge_border, linha1, linha2

    linha1, linha2 = find_contours_around_origin(
        rest_of_picture, base_frame, max_accepted, path_radius_bridg, trunk
    )
    bridge_img, linhatopo, linhabaixo, bridge_border, linha1, linha2 = (
        close_area_from_lines(linha1, linha2, base_frame, inicial_pnt)
    )
    bridge_border_seq = path_tools.img_to_chain(bridge_border)
    # if np.sum(linha1) == np.sum(linha2):
    #     extreme_external_points = [[], [], [], []]
    # else:
    while np.sum(bridge_border == 2) > 4 and len(bridge_border_seq) > 1:
        opened = mt.opening(bridge_img, kernel_size=1)
        linha1b = np.logical_and(linha1, opened)
        linha2b = np.logical_and(linha2, opened)
        if linha1b.any() and linha2b.any():
            linha1c = it.restore_continuous(linha1b)
            linha2c = it.restore_continuous(linha2b)
            bridge_img, linhatopo, linhabaixo, bridge_border, linha1, linha2 = (
                close_area_from_lines(linha1c, linha2c, base_frame, inicial_pnt)
            )
            bridge_border_seq = path_tools.img_to_chain(bridge_border)
        else:
            extreme_external_points = [[], [], [], []]
            break
    lens = [len(x) for x in bridge_border_seq]
    bridge_border_seq = bridge_border_seq[np.argmax(lens)]
    ends_topo = pt.img_to_points(sk.find_tips(linhatopo))
    ends_baixo = pt.img_to_points(sk.find_tips(linhabaixo))
    bridge_border_seq = path_tools.set_first_pt_in_seq(
        bridge_border_seq, ends_topo[0]
    )
    counter = 0
    flag = 0
    extreme_external_points = [ends_topo[0], [], [], []]
    for p in bridge_border_seq:
        if p in ends_topo + ends_baixo:
            if (counter == 1) and (p in ends_topo):
                flag = 1
            if flag and counter > 0:
                extreme_external_points[4 - counter] = p
            else:
                extreme_external_points[counter] = p
            counter += 1
    return (
        bridge_img,
        [linha1, linha2, linhatopo, linhabaixo],
        bridge_border,
        extreme_external_points,
    )


def remove_zigzag_bridges_conflict(bridge_a, bridge_b, rest_of_picture):
    commom_sides = np.logical_and(bridge_a.contorno[0], bridge_b.contorno[0])
    commom_sides_pts = pt.x_y_para_pontos(np.nonzero(commom_sides))
    if len(commom_sides_pts) == 0:
        extreme_pointe = mt.hitmiss_ends_v2(
            bridge_a.contorno[0], np.zeros_like(bridge_a.contorno[0])
        )
        new_sides_a = np.logical_and(
            bridge_a.contorno[0], np.logical_not(extreme_pointe)
        )
        new_bridge_a, new_elementos_contorno_a = adjust_bridge_end_lines(new_sides_a)
        bridge_a.img = new_bridge_a
        bridge_a.contorno = new_elementos_contorno_a

        extreme_pointe = mt.hitmiss_ends_v2(
            bridge_b.contorno[0], np.zeros_like(bridge_b.contorno[0])
        )
        new_sides_b = np.logical_and(
            bridge_b.contorno[0], np.logical_not(extreme_pointe)
        )
        new_bridge_b, new_elementos_contorno_b = adjust_bridge_end_lines(new_sides_b)
        bridge_b.img = new_bridge_b
        bridge_b.contorno = new_elementos_contorno_b
    elif 0 < len(commom_sides_pts) <= 3:
        new_sides_a = np.logical_and(bridge_a.contorno[0], np.logical_not(commom_sides))
        new_bridge_a, new_elementos_contorno_a = adjust_bridge_end_lines(new_sides_a)
        bridge_a.img = new_bridge_a
        bridge_a.contorno = new_elementos_contorno_a

        new_sides_b = np.logical_and(bridge_b.contorno[0], np.logical_not(commom_sides))
        new_bridge_b, new_elementos_contorno_b = adjust_bridge_end_lines(new_sides_b)
        bridge_b.img = new_bridge_b
        bridge_b.contorno = new_elementos_contorno_b
    else:
        print("aqui fudeu")
    return


def adjust_bridge_end_lines(bridge_sides):
    new_base = np.zeros_like(bridge_sides)
    linhas_do_limite = bridge_sides
    _, labeled, labeled_n = mt.detect_contours(linhas_do_limite)
    if labeled_n > 2:
        sums = []
        for l in np.arange(0, labeled_n):
            sums.append(np.sum(labeled == l + 1))
        idx = [sums.index(i) for i in sorted(sums, reverse=True)][:2]
        linha1 = labeled == idx[0] + 1
        linha2 = labeled == idx[1] + 1
        linhas_do_limite = np.logical_or(linha1, linha2)
        inicios_e_fins1 = pt.x_y_para_pontos(
            np.where(mt.hitmiss_ends_v2(linha1.astype(np.uint8), new_base))
        )
        inicios_e_fins2 = pt.x_y_para_pontos(
            np.where(mt.hitmiss_ends_v2(linha2.astype(np.uint8), new_base))
        )
        dist_1a_2 = list(
            map(
                lambda x: pt.distance_pts(inicios_e_fins1[0], x),
                inicios_e_fins2,
            )
        )
        dist_1b_2 = list(
            map(
                lambda x: pt.distance_pts(inicios_e_fins1[1], x),
                inicios_e_fins2,
            )
        )
        ponto_destino_1 = inicios_e_fins2[np.argmin(dist_1a_2)]
        ponto_destino_2 = inicios_e_fins2[np.argmin(dist_1b_2)]
        fechamento1_pts = [inicios_e_fins1[0], ponto_destino_1]
        fechamento2_pts = [inicios_e_fins1[1], ponto_destino_2]
        linhatopo = it.draw_line(
            np.zeros_like(bridge_sides), fechamento1_pts[0], fechamento1_pts[1]
        )
        linhabaixo = it.draw_line(
            np.zeros_like(bridge_sides), fechamento2_pts[0], fechamento2_pts[1]
        )
        bridge = np.logical_or(linhas_do_limite, linhatopo)
        bridge = np.logical_or(bridge, linhabaixo)
        bridge = it.fill_internal_area(bridge, np.ones_like(bridge_sides, np.uint8))
        bridge = mt.dilation(bridge, kernel_size=1)
    elif labeled_n == 2:
        linha1 = labeled == 1
        linha2 = labeled == 2
        inicios_e_fins1 = pt.x_y_para_pontos(
            np.where(mt.hitmiss_ends_v2(linha1.astype(np.uint8), new_base))
        )
        inicios_e_fins2 = pt.x_y_para_pontos(
            np.where(mt.hitmiss_ends_v2(linha2.astype(np.uint8), new_base))
        )
        dist_1a_2 = list(
            map(
                lambda x: pt.distance_pts(inicios_e_fins1[0], x),
                inicios_e_fins2,
            )
        )
        dist_1b_2 = list(
            map(
                lambda x: pt.distance_pts(inicios_e_fins1[1], x),
                inicios_e_fins2,
            )
        )
        ponto_destino_1 = inicios_e_fins2[np.argmin(dist_1a_2)]
        ponto_destino_2 = inicios_e_fins2[np.argmin(dist_1b_2)]
        fechamento1_pts = [inicios_e_fins1[0], ponto_destino_1]
        fechamento2_pts = [inicios_e_fins1[1], ponto_destino_2]
        linhatopo = it.draw_line(
            np.zeros_like(bridge_sides), fechamento1_pts[0], fechamento1_pts[1]
        )
        linhabaixo = it.draw_line(
            np.zeros_like(bridge_sides), fechamento2_pts[0], fechamento2_pts[1]
        )
        bridge = np.logical_or(linhas_do_limite, linhatopo)
        bridge = np.logical_or(bridge, linhabaixo)
        bridge = it.fill_internal_area(bridge, np.ones_like(bridge_sides, np.uint8))
        bridge = mt.dilation(bridge, kernel_size=1)
    else:
        return np.zeros_like(bridge_sides), [
            np.zeros_like(bridge_sides),
            np.zeros_like(bridge_sides),
            np.zeros_like(bridge_sides),
        ]
    return bridge, [linhas_do_limite, linhatopo, linhabaixo]


def external_cut_zigzag(
    external_borders,
    linhas_internas,
    linhas_transversais,
    extreme_external_points,
    extreme_internal_points,
):
    borda_cortada = np.zeros_like(linhas_internas)
    borda_normal = np.zeros_like(linhas_internas)
    external_borders_list = [x[0] for x in external_borders[0].tolist()]
    external_borders_img = it.points_to_img(external_borders_list, borda_cortada)
    contours = mt.detect_contours(external_borders_img)
    comeco = pt.contour_to_list(contours)[0]
    while comeco != extreme_external_points[0]:
        contours = contours[1:] + contours[:1]
        comeco = np.flip(contours[0]).tolist()[0]
    reference_points = extreme_external_points
    contours = external_borders[0]
    fila = contours.copy()
    fila = fila.tolist()
    rotations = fila.index([[reference_points[0][1], reference_points[0][0]]])
    fila = fila[rotations:] + fila[:rotations]
    counter = 0
    counter_pixels = 0
    first_flag = False
    last_change = 0
    for i in np.arange(0, len(fila)):
        borda_normal[fila[i][0][1]][fila[i][0][0]] = 1
        counter_pixels += 1
        y = fila[i][0][1]
        x = fila[i][0][0]
        ca = [y, x] == extreme_external_points[0]
        cb = [y, x] == extreme_external_points[1]
        cc = [y, x] == extreme_external_points[2]
        cd = [y, x] == extreme_external_points[3]
        cg = first_flag
        ch = [y, x] == extreme_internal_points[0]
        ci = [y, x] == extreme_internal_points[1]
        cj = [y, x] == extreme_internal_points[2]
        ck = [y, x] == extreme_internal_points[3]
        pixel_linhas = linhas_transversais[y][x]
        ce = pixel_linhas == 1
        cl = linhas_transversais[y + 1][x] and linhas_transversais[y][x + 1]
        cm = linhas_transversais[y + 1][x] and linhas_transversais[y][x - 1]
        co = linhas_transversais[y - 1][x] and linhas_transversais[y][x + 1]
        cp = linhas_transversais[y - 1][x] and linhas_transversais[y][x - 1]
        cq = cl or cm or co or cp

        cr = linhas_internas[y + 1][x]
        cs = linhas_internas[y + 1][x]
        ct = linhas_internas[y - 1][x]
        cu = linhas_internas[y - 1][x]
        cw = linhas_internas[y][x + 1]
        cx = linhas_internas[y][x + 1]
        cy = linhas_internas[y][x - 1]
        cz = linhas_internas[y][x - 1]
        cv = cr or cs or cu or ct or cw or cx or cy or cz
        cn = False
        if i > 0:
            cn = last_change == [fila[i - 1][0][1], fila[i - 1][0][0]]

        if cv or ch or ci or cj or ck:
            last_change = [y, x]
            if not cn:
                counter += 1
                borda_cortada[fila[i][0][1]][fila[i][0][0]] = 1
        if counter % 2 != 0:
            borda_cortada[fila[i][0][1]][fila[i][0][0]] = 1
    return borda_cortada, reference_points


def internal_cut(new_contour, linhas, extreme_internal_points, sentido):
    eip = copy.deepcopy(extreme_internal_points)
    if sentido:
        eip = [
            extreme_internal_points[1],
            extreme_internal_points[0],
            extreme_internal_points[3],
            extreme_internal_points[2],
        ]
    fila = new_contour.copy()
    fila = path_tools.set_first_pt_in_seq(fila, eip[0])
    ordem_na_fila_pontos = []
    for p in fila:
        if p in eip:
            ordem_na_fila_pontos.append(p)
    ordem_na_fila_pontos_idx = [ordem_na_fila_pontos.index(x) for x in eip]
    if ordem_na_fila_pontos_idx[1] == 3:
        fila.reverse()
        fila = path_tools.set_first_pt_in_seq(fila, eip[0])
    borda_cortada = np.zeros_like(linhas)
    borda_normal = np.zeros_like(linhas)
    linhas_transversais = linhas.copy()
    counter = 0
    counter_pixels = 0
    last_change = 0
    B = it.points_to_img(new_contour, np.zeros_like(linhas_transversais))
    D = np.logical_and(linhas_transversais, B)
    cruzamentos = pt.x_y_para_pontos(np.nonzero(D))
    for i in np.arange(0, len(fila)):
        borda_normal[fila[i][0]][fila[i][1]] = 1
        counter_pixels += 1
        y = fila[i][0]
        x = fila[i][1]
        ca = [y, x] == eip[0]
        cb = [y, x] == eip[1]
        cc = [y, x] == eip[2]
        cd = [y, x] == eip[3]
        ce = [y, x] in cruzamentos
        cn = False
        if i > 0:
            cn = last_change == fila[i - 1]
        if ce or cc or ca:  # cq
            if not cn:
                counter += 1
            last_change = fila[i]
        if counter % 2 == 0:
            borda_cortada[fila[i][0]][fila[i][1]] = 1
    return borda_cortada


def external_cut(
    external_borders,
    internal_borders,
    linhas_transversais,
    extreme_external_points,
    extreme_internal_points,
    odd_layer,
):
    borda_cortada = np.zeros_like(external_borders)
    borda_normal = np.zeros_like(external_borders)
    contours = mt.detect_contours(external_borders)
    contours = pt.contour_to_list(contours)
    comeco = contours[0]
    interruption_points = []
    while comeco != extreme_external_points[0]:
        contours = contours[1:] + contours[:1]
        comeco = np.flip(contours[0]).tolist()[0]
    counter_pixels = 0
    first_flag = False
    last_change = 0
    mark_next = False
    for i in np.arange(0, len(contours)):
        borda_normal[contours[i][0][1]][contours[i][0][0]] = 1
        counter_pixels += 1
        y = contours[i][0][1]
        x = contours[i][0][0]
        ca = [y, x] == extreme_external_points[0]
        cb = [y, x] == extreme_external_points[1]
        cc = [y, x] == extreme_external_points[2]
        cd = [y, x] == extreme_external_points[3]
        cg = first_flag
        ch = [y, x] == extreme_internal_points[0]
        ci = [y, x] == extreme_internal_points[1]
        cj = [y, x] == extreme_internal_points[2]
        ck = [y, x] == extreme_internal_points[3]
        pixel_linhas = linhas_transversais[y][x]
        ce = pixel_linhas == 1
        cl = linhas_transversais[y + 1][x] and linhas_transversais[y][x + 1]
        cm = linhas_transversais[y + 1][x] and linhas_transversais[y][x - 1]
        co = linhas_transversais[y - 1][x] and linhas_transversais[y][x + 1]
        cp = linhas_transversais[y - 1][x] and linhas_transversais[y][x - 1]
        cq = cl or cm or co or cp
        cn = False
        if i > 0:
            cn = last_change == [contours[i - 1][0][1], contours[i - 1][0][0]]
        cr = internal_borders[y + 1][x]
        cs = internal_borders[y + 1][x]
        ct = internal_borders[y - 1][x]
        cu = internal_borders[y - 1][x]
        cw = internal_borders[y][x + 1]
        cx = internal_borders[y][x + 1]
        cy = internal_borders[y][x - 1]
        cz = internal_borders[y][x - 1]
        cv = cr or cs or cu or ct or cw or cx or cy or cz
        if ca or cc:
            mark_next = True
        if ca or cb or cc or cd:  # impar
            last_change = [y, x]
            if not cn:
                first_flag = False
                borda_cortada[contours[i][0][1]][contours[i][0][0]] = 1
        if (not cg) and ce or ci or ck or cq or cv:
            first_flag = True
        if first_flag:
            borda_cortada[contours[i][0][1]][contours[i][0][0]] = 1
            if mark_next:
                mark_next = False
                interruption_points.append([y, x])
        canvas = np.zeros_like(borda_cortada)
        for p in extreme_external_points:
            canvas[p[0], p[1]] = 1
    return borda_cortada, interruption_points


# formula de transversais aqui!!!!!
def cut_in_transversals(origens_circulos, linha_c1, linha_c2, close_ends=False):
    transversais = []
    canvas = np.zeros_like(linha_c1)
    contorno_1_lista = pt.x_y_para_pontos(np.nonzero(linha_c1))
    extremos_1 = pt.img_to_points(mt.hitmiss_ends_v2(linha_c1))
    contorno_2_lista = pt.x_y_para_pontos(np.nonzero(linha_c2))
    extremos_2 = pt.img_to_points(mt.hitmiss_ends_v2(linha_c2))
    for i, o in enumerate(origens_circulos):
        if close_ends and (i == 0 or i == len(origens_circulos) - 1):
            dists = [pt.distance_pts(o, x) for x in extremos_1]
            transv1 = extremos_1[np.argmin(dists)]
        else:
            o_vec = np.array(o) - np.array(origens_circulos[i - 1])
            o_hat = o_vec / np.linalg.norm(o_vec)
            angles_w_o1 = []
            for v1 in contorno_1_lista:
                v1_vec = np.array(v1) - np.array(o)
                v1_hat = v1_vec / np.linalg.norm(v1_vec)
                cos_teta1 = np.dot(o_hat, v1_hat)
                angles_w_o1.append(abs(0 - cos_teta1))
            transv1 = contorno_1_lista[np.argmin(angles_w_o1)]
        angles_w_o2 = []
        if close_ends and (i == 0 or i == len(origens_circulos) - 1):
            dists = [pt.distance_pts(o, x) for x in extremos_2]
            transv2 = extremos_2[np.argmin(dists)]
        else:
            for v2 in contorno_2_lista:
                v2_vec = np.array(v2) - np.array(o)
                v2_hat = v2_vec / np.linalg.norm(v2_vec)
                cos_teta2 = np.dot(o_hat, v2_hat)
                angles_w_o2.append(abs(0 - cos_teta2))
            transv2 = contorno_2_lista[np.argmin(angles_w_o2)]
        transversais.append([transv1, transv2])
        canvas = it.points_to_img([transv1], canvas)
        canvas = it.points_to_img([transv2], canvas)
        canvas = it.points_to_img([o], canvas)
    return transversais, canvas


def find_contours_around_origin(rest_of_picture, base_frame, dist, path_radius, trunk):
    all_borders, all_borders_img = mt.detect_contours(rest_of_picture, return_img=True)
    area_pescocal = mt.dilation(trunk, kernel_size=(dist + 1.5 * path_radius))
    overlap = np.add(area_pescocal, all_borders_img)
    linhas_do_limite = overlap == 2
    _, labeled, labeled_n = it.divide_by_connected(linhas_do_limite)
    if labeled_n > 2:
        dists = []
        trunk_pts = pt.x_y_para_pontos(np.nonzero(trunk))
        trunk_center = pt.points_center(trunk_pts)
        trunk_center_pt, _ = pt.closest_point(trunk_center, trunk_pts)
        for l in np.arange(1, labeled_n + 1):
            line_pts = pt.x_y_para_pontos(np.nonzero(labeled == l))
            _, dist = pt.closest_point(trunk_center_pt, line_pts)
            dists.append(dist)
        lista_dist = dists.copy()
        idx1 = np.argmin(lista_dist)
        lista_dist[idx1] = 999999
        idx2 = np.argmin(lista_dist)
        linha1 = labeled == idx1 + 1
        linha2 = labeled == idx2 + 1
    elif labeled_n == 2:
        linha1 = labeled == 1
        linha2 = labeled == 2
    else:
        return [], []
    return linha1, linha2


def close_area_from_lines(linha1, linha2, base_frame, new_base):
    # ORGANIZAR LINHAS E PONTOS EXTREMOS
    inicios_e_fins1 = pt.x_y_para_pontos(
        np.where(mt.hitmiss_ends_v2(linha1.astype(np.uint8)))
    )
    inicios_e_fins2 = pt.x_y_para_pontos(
        np.where(mt.hitmiss_ends_v2(linha2.astype(np.uint8)))
    )
    dist_1a_2 = list(
        map(lambda x: pt.distance_pts(inicios_e_fins1[0], x), inicios_e_fins2)
    )
    dist_1b_2 = list(
        map(lambda x: pt.distance_pts(inicios_e_fins1[1], x), inicios_e_fins2)
    )
    ponto_destino_1 = inicios_e_fins2[np.argmin(dist_1a_2)]
    ponto_destino_2 = inicios_e_fins2[np.argmin(dist_1b_2)]
    fechamento1_pts = [inicios_e_fins1[0], ponto_destino_1]
    fechamento2_pts = [inicios_e_fins1[1], ponto_destino_2]
    linhatopo = it.draw_line(
        np.zeros(base_frame), fechamento1_pts[0], fechamento1_pts[1]
    )
    linhabaixo = it.draw_line(
        np.zeros(base_frame), fechamento2_pts[0], fechamento2_pts[1]
    )
    bridge_border = it.sum_imgs([linha1, linhatopo, linha2, linhabaixo])
    bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
    return bridge_img, linhatopo, linhabaixo, bridge_border


def organize_extreme_zb_points(linha, bridge_img, path_radius):
    ends = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(linha)))
    centro = pt.points_center(pt.x_y_para_pontos(np.nonzero(linha)))
    centro_img = it.points_to_img([centro], np.zeros_like(bridge_img))
    area = mt.dilation(centro_img, kernel_size=(2 * path_radius))
    candidates_img = np.logical_and(area, bridge_img)
    candidates_pts = pt.x_y_para_pontos(np.nonzero(candidates_img))
    triangle_pt = pt.most_distant_from(centro, candidates_pts)
    candidates = [triangle_pt, *ends]
    candidates = path_tools.organize_points_cw(candidates, origin=centro)
    rotations = candidates.index(triangle_pt)
    candidates = candidates[rotations:] + candidates[:rotations]
    return candidates[1], candidates[2]

#TODO: continuar daqui a arrumar as distancias de maneira mais padrão, hoje não tem ajuste
def equidistant_in_seq(line_img, path_radius,internal_mask_dist, origin_pt=[]):
    n_origens = 0
    adjust = 0
    endpoints_img = mt.hitmiss_ends_v2(line_img.astype(bool))
    endpoints = pt.img_to_points(endpoints_img)
    if len(origin_pt) > 0:
        first, _ = pt.closest_point(origin_pt, endpoints)
        last = list(filter(lambda x: x != first, endpoints))[0]
    else:
        np.sort(endpoints, 1)[0]
        [first, last] = endpoints
    pontos_org = path_tools.line_img_to_freeman_chain(line_img, first)
    while n_origens % 2 != 1:
        origens_pontos = [pontos_org[0]]
        division_distance = (path_radius * 2) - adjust
        copied_origin = copy.deepcopy(line_img)
        while np.sum(copied_origin.astype(np.uint8)) > 0:
            canvas = np.zeros_like(line_img, np.uint8)
            centro = origens_pontos[-1]
            area_distance = it.draw_circle(canvas, centro, division_distance)
            candidates = np.logical_and(area_distance, copied_origin)
            candidates = pt.img_to_points(candidates)
            if candidates:
                new_point_candidates = pt.most_distant_from(origens_pontos[-1], candidates, give_second=True)
                indexes = [pontos_org.index(x) for x in new_point_candidates]
                new_point = pontos_org[np.min(indexes)]
                dist = pt.distance_pts(new_point, origens_pontos[-1])
                if dist > (3 * division_distance) / 4:
                    origens_pontos.append(new_point)
            else:
                pass
            copied_origin = np.logical_and(copied_origin, np.logical_not(area_distance))
        n_origens = len(origens_pontos)
        adjust += 1
    origens_pontos[-1] = last
    return origens_pontos


def equidistant_by_proximity(linha_img, origin_lst, path_radius, img):
    n_origens = 0
    endpoints_img = mt.hitmiss_ends_v2(linha_img.astype(bool))
    endpoints = pt.img_to_points(endpoints_img)
    linha_lst = pt.img_to_points(linha_img)
    first, _ = pt.closest_point(origin_lst[0], endpoints)
    last = list(filter(lambda x: x != first, endpoints))[0]
    origens_pontos = []
    for origin_pt in origin_lst:
        copied_origin = copy.deepcopy(linha_img)
        canvas = np.zeros_like(linha_img, np.uint8)
        centro = origin_pt
        area_distance = it.draw_circle(canvas, centro, path_radius)
        candidate, _ = pt.closest_point(centro, pt.img_to_points(copied_origin))
        origens_pontos.append(candidate)
        copied_origin = np.logical_and(copied_origin, np.logical_not(area_distance))
        n_origens = len(origens_pontos)
    origens_pontos[0] = first
    origens_pontos[-1] = last
    return origens_pontos

def internal_adapted_polygon(
    linha_ci1, linha_ci2, linhas_transversais, pontos_extremos
):
    new_contour = np.logical_or(linha_ci1, linha_ci2)
    new_contour = np.logical_or(new_contour, linhas_transversais).astype(np.uint8)
    new_contour, _, _ = sk.create_prune_skel(new_contour, 20)
    new_contour_cnt = mt.detect_contours(new_contour, only_external=True)
    new_contour_pts = pt.contour_to_list(new_contour_cnt)
    new_contour_img = it.points_to_img(new_contour_pts, np.zeros_like(linha_ci1))
    ref_internos = new_contour_pts
    ia, _ = pt.closest_point(pontos_extremos[0], ref_internos)
    ib, _ = pt.closest_point(pontos_extremos[1], ref_internos)
    ic, _ = pt.closest_point(pontos_extremos[2], ref_internos)
    id, _ = pt.closest_point(pontos_extremos[3], ref_internos)
    extreme_internal_points = [ia, ib, ic, id]
    return new_contour_pts, extreme_internal_points


def oscilatory_start_and_end(new_zigzag, pontos_extremos):
    fins_da_rota = pt.x_y_para_pontos(
        np.nonzero(mt.hitmiss_ends_v2(new_zigzag.astype(bool)))
    )
    return fins_da_rota


def reduce_origin(candidate, necks_max_paths, norm_dist_map):
    """Determina inicio e fim para cada trunk, então vai reduzindo as margens até englobar todos os estrangulamentos dele"""
    t_ends = pt.img_to_points(sk.find_tips(candidate.astype(bool)))
    if len(t_ends) == 0:  # se for um ciclo fechado
        start_pnt = random.choice(pt.x_y_para_pontos(np.nonzero(candidate)))
        origin_chain = pt.invert_x_y(
            path_tools.make_a_chain(candidate.astype(bool), start_pnt)
        )
        origin_chain_img = it.points_to_img(origin_chain, np.zeros_like(candidate))
        new_ends = pt.img_to_points(sk.find_tips(origin_chain_img.astype(bool)))
        origin_chain = path_tools.set_first_pt_in_seq(origin_chain, new_ends[0])
        candidate = np.multiply(candidate, origin_chain_img)
    else:  # se for um trunk aberto
        start_pnt = []
        origin_chain = pt.invert_x_y(
            path_tools.make_a_chain_open_segment(candidate.astype(bool), t_ends)
        )
    reduced_origin = np.logical_and(candidate != 0, candidate < necks_max_paths)
    ends = pt.img_to_points(sk.find_tips(reduced_origin.astype(bool)))
    new_origin = origin_chain.copy()
    count_up = 0
    count_down = -1
    start_flag = 0
    end_flag = 0
    while not (start_flag and end_flag):
        current_pt_1 = origin_chain[count_up]
        current_pt_2 = origin_chain[count_down]
        if current_pt_1 in ends:
            start_flag = 1
        else:
            new_origin.remove(current_pt_1)
            count_up += 1
        if current_pt_2 in ends:
            end_flag = 1
        else:
            new_origin.remove(current_pt_2)
            count_down -= 1
    reduced_origin = it.points_to_img(new_origin, np.zeros_like(candidate))
    separated, _, n = it.divide_by_connected(reduced_origin)
    if n > 1:
        reduced_origin = it.take_the_bigger_area(reduced_origin)
        print("POBREMAAAA")
    return reduced_origin * norm_dist_map, origin_chain[0]


def make_offset_bridge_route(
    bridge_region: Bridge, offsets_regions, path_radius_cont, base_frame
) -> Bridge:
    all_offsets = it.sum_imgs([x.img for x in offsets_regions])
    square_mask = getStructuringElement(
        MORPH_RECT, (int(path_radius_cont * 2), int(path_radius_cont * 2))
    )
    if bridge_region.type == "common_offset_bridge":
        _, outer = mt.detect_contours(bridge_region.img, return_img=True)
        objective_lines = np.logical_and(
            outer.astype(np.uint8),
            np.logical_not(mt.dilation(all_offsets, kernel_size=1).astype(np.uint8)),
        )
        objective_lines_dilated = mt.dilation(objective_lines, kernel_img=square_mask)
        outer_offseted = np.logical_and(
            bridge_region.img, np.logical_not(objective_lines_dilated)
        )
        outer_offseted = it.take_the_bigger_area(outer_offseted)
        _, outer_new = mt.detect_contours(outer_offseted, return_img=True)
        objective_lines_new = np.logical_and(
            outer_new.astype(np.uint8),
            np.logical_not(mt.dilation(all_offsets, kernel_size=1).astype(np.uint8)),
        )
    elif bridge_region.type == "contact_offset_bridge":
        bridge_area = mt.dilation(bridge_region.origin, kernel_img=square_mask)
        offsets_routes = it.sum_imgs(
            [
                x.route
                for x in offsets_regions
                if x.name in bridge_region.linked_offset_regions
            ]
        )
        out_contour = it.sum_imgs([bridge_area, offsets_routes])
        _, objective_lines = mt.detect_contours(
            out_contour, return_img=True, only_external=True
        )
        objective_lines_new = np.logical_and(
            objective_lines, np.logical_not(offsets_routes)
        )
    bridge_region.route = objective_lines_new
    bridge_region.trail = mt.dilation(bridge_region.route, kernel_size=path_radius_cont)
    bridge_region.find_center(base_frame)
    return bridge_region


def make_zz_or_co_bridge_route(region: Bridge, path_radius, mask_distancer, internal_mask_dist, rest_of_picture, base_frame):

    total_sobreposition = mt.dilation(region.img, kernel_size=path_radius)
    eroded = mt.erosion(total_sobreposition, kernel_img=mask_distancer)
    rest_pict_total_sobreposition = mt.dilation(rest_of_picture, kernel_size=path_radius)
    rest_pict_eroded = mt.erosion(rest_pict_total_sobreposition, kernel_img=mask_distancer)
    _, eroded_border = mt.detect_contours(eroded, return_img=True, only_external=True)
    if len(np.unique(it.sum_imgs([region.origin, eroded_border]))) < 3:
        origin_seq = path_tools.img_to_chain(region.origin)[0]
        origin_seq = path_tools.set_first_pt_in_seq(origin_seq, pt.img_to_points(mt.hitmiss_ends_v2(region.origin))[0])
        origin_seq = path_tools.cut_repetition(origin_seq)
        tng_end = path_tools.draw_tangent_from_seq(list(reversed(origin_seq)), path_radius*4, np.zeros_like(eroded))
        tng_start = path_tools.draw_tangent_from_seq(origin_seq, path_radius*4, np.zeros_like(eroded))
        origin = np.logical_or(tng_start,np.logical_or(region.origin,tng_end))
    else:
        origin = region.origin
    origin_axis = np.logical_and(origin, eroded)
    _,_,n = it.divide_by_connected(origin_axis)
    if np.sum(origin_axis) > 0 and n == 1:
        _, _, n_divisions = it.divide_by_connected(origin_axis)
        if n_divisions > 1:
            origin_axis, eroded = connect_origin_parts(region.origin, eroded)
        linha_ci1, linha_ci2 = region.make_internal_border(
            rest_pict_eroded, total_sobreposition, eroded, origin_axis, path_radius
        )
        if np.equal(linha_ci1,linha_ci2).all():
            new_zigzag = region.origin
            new_zigzag_b = region.route
        else:
            pts_trns_origin = equidistant_in_seq(origin_axis, path_radius, internal_mask_dist)
            pts_trns_ci1 = equidistant_by_proximity(
                linha_ci1, pts_trns_origin, path_radius, total_sobreposition
            )
            pts_trns_ci2 = equidistant_by_proximity(
                linha_ci2, pts_trns_origin, path_radius, total_sobreposition
            )
            linhas_transversais = np.zeros_like(region.img)
            linhas_limitrofes = np.zeros_like(region.img)
            for i, point in enumerate(pts_trns_origin):
                if i == 0 or i == len(pts_trns_origin) - 1:
                    thisline = it.draw_polyline(
                        linhas_limitrofes,
                        [pts_trns_ci1[i], pts_trns_origin[i], pts_trns_ci2[i]],
                        False,
                    )
                    linhas_limitrofes = np.logical_or(linhas_limitrofes, thisline)
                else:
                    thisline = it.draw_polyline(
                        linhas_transversais,
                        [pts_trns_ci1[i], pts_trns_origin[i], pts_trns_ci2[i]],
                        False,
                    )
                    linhas_transversais = np.logical_or(linhas_transversais, thisline)
            new_contour, new_contour_img = mt.detect_contours(
                it.sum_imgs([linhas_limitrofes, linha_ci1, linha_ci2]),
                return_img=True,
                only_external=True,
            )
            new_contour = pt.contour_to_list(new_contour)
            ends_linha_ci1 = pt.img_to_points(mt.hitmiss_ends_v2(linha_ci1))
            ends_linha_ci2 = pt.img_to_points(mt.hitmiss_ends_v2(linha_ci2))
            ia, _ = pt.closest_point(region.pontos_extremos[0], ends_linha_ci1)
            ib = [x for x in ends_linha_ci1 if x != ia][0]
            ic, _ = pt.closest_point(region.pontos_extremos[2], ends_linha_ci2)
            id = [x for x in ends_linha_ci2 if x != ic][0]
            extr_int_pts = [ia, ib, ic, id]
            new_contour = path_tools.set_first_pt_in_seq(new_contour, extr_int_pts[0])
            new_zigzag = weaving_zigzag(
                new_contour,
                new_contour_img,
                linhas_transversais,
                linhas_limitrofes,
                extr_int_pts,
                0,
            )
            new_zigzag = np.logical_or(new_zigzag,linhas_transversais)
            new_zigzag, _, _ = sk.create_prune_skel(new_zigzag, 2)

            new_zigzag_b = weaving_zigzag(
                new_contour,
                new_contour_img,
                linhas_transversais,
                linhas_limitrofes,
                extr_int_pts,
                1,
            )
            new_zigzag_b = np.logical_or(new_zigzag_b,linhas_transversais)
            new_zigzag_b, _, _ = sk.create_prune_skel(new_zigzag_b, 2)

        region.reference_points = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(new_zigzag)))
        region.reference_points_b = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(new_zigzag_b)))
        if len(region.reference_points) < 2:
            new_zigzag = path_tools.one_pixel_wide(mt.dilation(new_zigzag, kernel_size=2))
            region.reference_points = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(new_zigzag)))
            print("ajdnsabhfsfb")
        if len(region.reference_points_b) < 2:
            new_zigzag_b = path_tools.one_pixel_wide(mt.dilation(new_zigzag_b, kernel_size=2))
            region.reference_points_b = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(new_zigzag_b)))
            print("ajdnsabhfsfb")
        region.find_center(base_frame)
    else:
        reducted_origin = it.take_the_bigger_area(region.origin)
        reduction_points = mt.dilation(mt.hitmiss_ends_v2(reducted_origin),kernel_size=path_radius)
        reducted_origin = it.image_subtract(reducted_origin, reduction_points)
        new_zigzag = reducted_origin
        new_zigzag_b = new_zigzag
        region.reference_points = pt.x_y_para_pontos(
            np.nonzero(mt.hitmiss_ends_v2(region.origin.astype(bool)))
        )
        region.reference_points_b = region.reference_points
        region.find_center(base_frame)
    region.route = new_zigzag
    region.route_b = new_zigzag_b
    aaaa = it.sum_imgs([region.route, linha_ci1,linha_ci2,eroded, it.points_to_img(pts_trns_ci1, np.zeros_like(eroded)), region.img, linhas_limitrofes, origin_axis, region.origin])
    region.trail = mt.dilation(region.route, kernel_size=path_radius)
    region.trail_b = mt.dilation(region.route_b, kernel_size=path_radius)
    return region

def weaving_zigzag(
    new_contour,
    new_contour_img,
    linhas_transversais,
    linhas_limitrofes,
    extr_int_pts,
    sentido,
):
    cutted_border = internal_cut(new_contour, linhas_transversais, extr_int_pts, sentido)
    new_zigzag = np.logical_or(linhas_transversais, cutted_border)
    new_zigzag = np.logical_or(new_zigzag, linhas_limitrofes)
    _, _, n = it.divide_by_connected(new_zigzag)
    reference_points_b = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(new_zigzag)))
    if n > 1:
        extr_int_pts = [
            extr_int_pts[0],
            extr_int_pts[2],
            extr_int_pts[1],
            extr_int_pts[3],
        ]
        print("sequncia corrigida, b e c trocado")
        new_zigzag = internal_cut(
            new_contour, linhas_transversais, extr_int_pts, sentido
        )
        new_zigzag = np.logical_or(new_zigzag, linhas_limitrofes)
    _, _, n = it.divide_by_connected(mt.hitmiss_ends_v2(new_zigzag))
    if n <= 1:
        cccc = mt.closing(new_zigzag, kernel_size=1)
        ddd, _, _ = sk.create_prune_skel(cccc, 1)
        _, eee, n = it.divide_by_connected(mt.hitmiss_ends_v2(ddd))
        if n > 1:
            new_zigzag = ddd
    if len(reference_points_b) < 2:
        print("ajdnsabhfsfb")
    return new_zigzag

def decompose_pol_cont_by_corners(linhas_do_limite, trunk, path_radius_bridg):
    pontos = path_tools.img_to_chain(linhas_do_limite)[0]
    # pontos_curvatura = path_tools.encontrar_pontos_curvatura(pontos+[pontos[0]]+[pontos[1]])
    pontos_curvatura = path_tools.encontrar_pontos_curvatura(pontos)
    if len(pontos_curvatura) <4:
        pontos_harris = corner_peaks(corner_harris(linhas_do_limite), min_distance=5, threshold_rel=0.02)
        pontos_harris = [list(x) for x in pontos_harris]
        pontos_corrigidos = []
        for point in pontos_harris:
            if not (point in pontos):
                pnt, _ = pt.closest_point(point, pontos)
                pontos_corrigidos.append(pnt)
            else:
                pontos_corrigidos.append(point)
        pontos_curvatura = pontos_corrigidos
    pontos_curvatura_img = it.points_to_img(pontos_curvatura, np.zeros_like(linhas_do_limite))
    pontos = path_tools.set_first_pt_in_seq(pontos, pontos_curvatura[0])
    segments = path_tools.colorbyevent(pontos,pontos_curvatura,np.zeros_like(linhas_do_limite))
    segments_n = max(np.unique(segments))
    trunque = trunk>0
    trunk_seq = path_tools.img_to_chain(trunque)[0]
    trunk_seq = path_tools.set_first_pt_in_seq(trunk_seq, pt.img_to_points(mt.hitmiss_ends_v2(trunque))[0])
    trunk_seq = path_tools.cut_repetition(trunk_seq)
    # TODO: ver se aqui tem como arrumar esses tamanhos
    tng_end = path_tools.draw_tangent_from_seq(list(reversed(trunk_seq)), path_radius_bridg*4, np.zeros_like(linhas_do_limite))
    tng_start = path_tools.draw_tangent_from_seq(trunk_seq, path_radius_bridg*4, np.zeros_like(linhas_do_limite))
    possible_c1_c2 = np.zeros_like(linhas_do_limite)
    counter_accepted = 0
    for labl in np.add(list(range(segments_n)),1):
        contact = it.sum_imgs([tng_end,tng_start,np.int32(segments==labl)])
        if not((contact == 2).any()):
            counter_accepted += 1
            possible_c1_c2 = it.sum_imgs([possible_c1_c2, np.multiply(segments==labl,counter_accepted)])
    return possible_c1_c2, counter_accepted, pontos_curvatura