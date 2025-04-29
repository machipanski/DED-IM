import random
import numpy as np
import itertools
import copy
import networkx as nx
import concurrent.futures
from typing import List
from components.offset import Region
from networkx import get_edge_attributes
from cv2 import getStructuringElement, MORPH_RECT
from scipy.ndimage import distance_transform_edt
from skimage.feature import corner_harris, corner_peaks
from components.skeleton import prune
from components import morphology_tools as mt
from components import images_tools as it
from components import points_tools as pt
from components import skeleton as sk
from components import path_tools
from components.timer import Timer


class Bottleneck:

    def __init__(
        self,
        name,
        img,
        origin,
        trunk,
        n_paths,
        origin_marks,
        contour_elements=None,
        extreme_points=None,
        linked_offset_regions=None,
        linked_zigzag_regions=None,
    ):
        if contour_elements is None:
            contour_elements = []
        if extreme_points is None:
            extreme_points = []
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
        self.contour = contour_elements
        self.extreme_points = extreme_points
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

    # def make_offset_bridge_route(
    #     self, offsets_regions, path_radius, base_frame, rest_of_picture
    # ):
    #     all_offsets = it.sum_imgs([x.img for x in offsets_regions])
    #     _, outer = mt.detect_contours(self.img, return_img=True)
    #     objective_lines = np.logical_and(
    #         outer.astype(np.uint8),
    #         np.logical_not(mt.dilation(all_offsets, kernel_size=1).astype(np.uint8)),
    #     )
    #     square_mask = getStructuringElement(
    #         MORPH_RECT, (int(path_radius * 2), int(path_radius * 2))
    #     )
    #     objective_lines_dilated = mt.dilation(objective_lines, kernel_img=square_mask)
    #     outer_offseted = np.logical_and(
    #         self.img, np.logical_not(objective_lines_dilated)
    #     )
    #     outer_offseted = it.take_the_bigger_area(outer_offseted)
    #     _, outer_new = mt.detect_contours(outer_offseted, return_img=True)
    #     objective_lines_new = np.logical_and(
    #         outer_new.astype(np.uint8),
    #         np.logical_not(mt.dilation(all_offsets, kernel_size=1).astype(np.uint8)),
    #     )
    #     self.route = objective_lines_new
    #     self.trail = mt.dilation(self.route, kernel_size=path_radius)
    #     return


class Bridge:

    def __init__(self, *args, **kwargs):
        self.contour = []
        self.extreme_points = []
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
            # internal_borders_pts = pt.img_to_points(internal_borders)
            start_ci, _ = pt.closest_point(start_orig, internal_extreme)
            internal_extreme.remove(start_ci)
            end_ci, _ = pt.closest_point(end_orig, internal_extreme)
            internal_extreme.remove(end_ci)
            G = path_tools.img_to_graph_com_distancias(distance_map)
            _, path1 = nx.bidirectional_dijkstra(G, tuple(start_ci), tuple(end_ci))
            line_ci = it.points_to_img(path1, np.zeros_like(internal_borders))
            return line_ci

        internal = np.logical_or(internal, origin_axis)
        _, internal_borders = mt.detect_contours(internal, return_img=True)
        internal_borders = np.logical_and(internal_borders, filled_external_borders)
        _, labeled, labeled_n = it.divide_by_connected(internal_borders)
        line_ci1 = np.zeros_like(self.img)
        line_ci2 = np.zeros_like(self.img)
        if labeled_n == 1:
            internal_extreme = pt.img_to_points(mt.hitmiss_ends_v2(internal_borders))
            if len(internal_extreme) == 0:
                possible_c1_c2, counter_accepted, internal_extreme = (
                    decompose_pol_cont_by_corners(
                        internal_borders, origin_axis, path_radius_bridg
                    )
                )
                # bbb = it.sum_imgs([tng_end,tng_start,all_borders_img])
                labeled = possible_c1_c2
                labeled_n = counter_accepted
            line_ci1 = make_closest_path_to(
                internal_borders,
                self.contour[0],
                self.extreme_points[0],
                self.extreme_points[1],
                internal_extreme,
            )
            if len(internal_extreme) >= 2:
                line_ci2 = make_closest_path_to(
                    internal_borders,
                    self.contour[1],
                    self.extreme_points[2],
                    self.extreme_points[3],
                    internal_extreme,
                )
            else:
                line_ci2 = line_ci1
        elif labeled_n > 2:
            sums = []
            for l in np.arange(0, labeled_n):
                sums.append(np.sum(labeled == l + 1))
            idx = [sums.index(i) for i in sorted(sums, reverse=True)][:2]
            line_ci1 = labeled == idx[0] + 1
            line_ci2 = labeled == idx[1] + 1
        elif labeled_n == 2:
            line_ci1 = labeled == 1
            line_ci2 = labeled == 2
        internal_borders_closed = np.add(
            internal_borders, np.logical_or(self.contour[2], self.contour[3])
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
        line_ci1 = np.logical_and(eroded, line_ci1)
        line_ci1, _, _ = sk.create_prune_skel(line_ci1, 1)
        line_ci2 = np.logical_and(eroded, line_ci2)
        line_ci2, _, _ = sk.create_prune_skel(line_ci2, 1)
        if np.sum(line_ci1) == 0:
            print("Error: no line 1")
        if np.sum(line_ci2) == 0:
            print("Error: no line 2")
        return line_ci1, line_ci2

    def find_center(self):
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
        return

    def get_linked_zigzags(self, zigzag_regions):
        linked_zigzags = []
        for zr in zigzag_regions:
            combined_imgs = np.logical_or(self.img, zr.img)
            _, _, num = it.divide_by_connected(combined_imgs)
            if num == 1:
                linked_zigzags.append(zr.name)
        self.linked_zigzag_regions = linked_zigzags
        return


class BridgeRegions:
    """List of lists of the different bridge regions in the Layer"""

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
        """determines connection points between the different contours,
        drawing a bridge in the direction of the Layer's offset"""
        offreg: List[Region] = offsets_regions.regions
        regs_touching = []
        for region in offreg:
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
            region.make_parallel_points(
                offreg, area_internal_contour_img, prohibited_areas, path_radius
            )
        areas_graph = path_tools.make_offset_graph(offreg, regs_touching)
        offsets_parallel_mst, parallel_sequence = path_tools.regions_mst(areas_graph)

        bridge_imgs = self.draw_offset_parallel_links(
            offsets_parallel_mst,
            parallel_sequence,
            rest_of_picture,
            base_frame,
            path_radius,
            offreg,
        )

        if len(bridge_imgs) > 0:
            self.all_bridges = it.sum_imgs(bridge_imgs)
        return areas_graph, offsets_parallel_mst

    def draw_offset_parallel_links(
        self,
        offsets_parallel_mst,
        parallel_sequence,
        rest_of_picture,
        base_frame,
        path_radius,
        offsets_regs,
    ):
        counter = 0
        lista_origem = get_edge_attributes(offsets_parallel_mst, "origin_coords")
        lista_destino = get_edge_attributes(offsets_parallel_mst, "destiny_coords")
        lista_tipo = get_edge_attributes(offsets_parallel_mst, "extreme_origin")
        separated_imgs = []
        for i, line in enumerate(list(parallel_sequence)):
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
                destiny_point = [
                    pontos_origem[1][0],
                    pontos_origem[1][1] + (0 * path_radius),
                ]
                origin = it.draw_line(np.zeros(base_frame), ponto_origem, destiny_point)
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
            else:  # contact_offset_bridge
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
                    lines_separadas, _, _ = it.divide_by_connected(eee)
                    sum = np.sum(eee)
                    divisor = divisor - 1
                pontos_centrais = [
                    pt.points_center(pt.img_to_points(x)) for x in lines_separadas
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
            """Breaks the image into its MAT components - already normalized to the number of routes that fit in the parallel direction"""
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
            """Further divides the trunks to avoid overly large regions"""
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
            """Returns only the trunks that have bottleneck sections"""
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
                    bridge_img, contour_elements, contour, extreme_points = (
                        close_bridge_contour(
                            candidate,
                            base_frame,
                            1.5 * necks_max_paths,  # TODO check this value
                            rest_of_picture,
                            path_radius_bridg,
                            inicial_pnt,
                        )
                    )
                    print("Closed a bridge")
                except Exception:
                    print("Bridge failed")
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
                    origin_mark = [y_mark, x_mark, str("not used")]
                    self.zigzag_bridges.append(
                        Bridge(
                            f"ZB_{counter:03d}",
                            bridge_img,
                            # np.logical_and(bridge_img, candidate).astype(np.uint8),
                            np.logical_and(bridge_img, skel).astype(np.uint8),
                            candidate,
                            [],
                            origin_mark,
                            contour=contour_elements,
                            extreme_points=extreme_points,
                        )
                    )
                    self.zigzag_bridges[-1].get_linked_offsets(offset_regions)
                    counter += 1
            return

        def filter_trunks_if_tip_minimum(norm_reduced_origins):
            filtered_trunks = []
            for trunk in norm_reduced_origins:
                tips = pt.img_to_points(sk.find_tips(trunk.astype(bool)))
                if not tips:
                    continue
                tip_values = [trunk[tip[0], tip[1]] for tip in tips]
                non_zero_min = np.min(trunk[trunk > 0])
                if not non_zero_min in tip_values:
                    filtered_trunks.append(trunk)
            return filtered_trunks

        norm_trunks, norm_dist_map = separate_trunks()
        minus_bigger_than_2wd = break_too_big_parts(norm_trunks, norm_dist_map)
        origin_candidates = separate_truks_with_botlenecks(minus_bigger_than_2wd)
        reduced = [
            reduce_origin(x, necks_max_paths, norm_dist_map) for x in origin_candidates
        ]
        norm_reduced_origins = [y[0] for y in reduced]
        norm_reduced_filtered_origins = filter_trunks_if_tip_minimum(
            norm_reduced_origins
        )
        initial_points = [x[1] for x in reduced]
        close_bridges(
            norm_reduced_filtered_origins, initial_points, self.medial_transform
        )
        return

    def make_cross_over_bridges(self, prohibited_areas, offsets_mst):
        """Joins neck regions to offset bridges to enable a route that covers the entire area"""
        counter = 0
        combinations = list(itertools.product(self.zigzag_bridges, self.offset_bridges))
        substitutions = []
        for n, [zigzag_bridge, offset_bridge] in enumerate(combinations):
            if np.equal(zigzag_bridge.contour[0], zigzag_bridge.contour[1]).all():
                # print("Aqui eu não deixei a parede unica virar crossover")
                print("ERROR: special case")
            elif set(zigzag_bridge.linked_offset_regions) == set(
                offset_bridge.linked_offset_regions
            ):
                # se as pontes conectam os mesmos contours, organiza a prioridade deles e paga a mais alta pra cada grupo
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
            print("Element:", elementos, "Higher priority:", maior_prioridade)
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
                    contour=zigzag_bridge.contour,
                    extreme_points=zigzag_bridge.extreme_points,
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
        """Calls the make_route() function for each region"""
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
                        internal_mask_dist,
                        rest_of_picture,
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
    """Determines the start and end for each trunk, then reduces the margins until all its bottlenecks are encompassed"""
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
            trunk.astype(bool), kernel_size=(max_accepted * path_radius_bridg + 8)
        )
        overlap = np.add(area_pescocal, all_borders_img)
        lines_do_limite = overlap == 2
        _, labeled, labeled_n = it.divide_by_connected(lines_do_limite)
        if labeled_n == 1:
            print("Special case: Only one line around origin")
            possible_c1_c2, counter_accepted, curvature_points = (
                decompose_pol_cont_by_corners(lines_do_limite, trunk, path_radius_bridg)
            )
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
            line1 = labeled == idx1 + 1
            # ATENçÂO PARA NOVOS CASOS AQUI!!!!!!!
            line2 = labeled == idx2 + 1
        elif labeled_n == 2:
            line1 = labeled == 1
            line2 = labeled == 2
            # line1 = labeled == 1
            return line1, line2
        else:
            print("ERROR: No lines around origin")
            return np.zeros_like(trunk), np.zeros_like(trunk)
        points_trunk = pt.img_to_points(mt.hitmiss_ends_v2(trunk.astype(bool)))
        points_line1 = pt.img_to_points(mt.hitmiss_ends_v2(line1.astype(bool)))
        if len(points_line1) > 0:
            line1 = reduce_lines_overshoot(line1, points_trunk)
        points_line2 = pt.img_to_points(mt.hitmiss_ends_v2(line2.astype(bool)))
        if len(points_line2) > 0:
            line2 = reduce_lines_overshoot(line2, points_trunk)
        return line1, line2

    def close_area_from_lines(
        line1: np.ndarray,
        line2: np.ndarray,
        base_frame,
        inicial_pnt,
    ):
        starts_and_ends1 = pt.x_y_para_pontos(
            np.where(sk.find_tips(line1.astype(np.uint8)))
        )
        starts_and_ends2 = pt.x_y_para_pontos(
            np.where(sk.find_tips(line2.astype(np.uint8)))
        )
        # Se alguma delas é um circulo fechado, interrompe perto da origem indicada poe Inicial_pt
        if len(starts_and_ends1) == 0:
            line_pts = pt.img_to_points(line1.astype(np.uint8))
            break_point, _ = pt.closest_point(inicial_pnt, line_pts)
            # line1[break_point[0]][break_point[1]] = 0
            origin_chain = pt.invert_x_y(path_tools.make_a_chain(line1, break_point))
            line1 = it.points_to_img(origin_chain, np.zeros_like(line1))
            starts_and_ends1 = pt.x_y_para_pontos(
                np.where(sk.find_tips(line1.astype(np.uint8)))
            )
        if len(starts_and_ends2) == 0:
            line_pts = pt.img_to_points(line2.astype(np.uint8))
            break_point, _ = pt.closest_point(inicial_pnt, line_pts)
            # line2[break_point[0]][break_point[1]] = 0
            origin_chain = pt.invert_x_y(path_tools.make_a_chain(line2, break_point))
            line2 = it.points_to_img(origin_chain, np.zeros_like(line2))
            starts_and_ends2 = pt.x_y_para_pontos(
                np.where(sk.find_tips(line2.astype(np.uint8)))
            )
        # Se houverem pontos demais por causa dos contours, faz uma poda
        if len(starts_and_ends1) > 2:
            line1, _, _ = prune(skel_img=line1, size=2)
            starts_and_ends1 = pt.x_y_para_pontos(
                np.where(sk.find_tips(line1.astype(np.uint8)))
            )
        if len(starts_and_ends2) > 2:
            line2, _, _ = prune(skel_img=line2, size=2)
            starts_and_ends2 = pt.x_y_para_pontos(
                np.where(sk.find_tips(line2.astype(np.uint8)))
            )
        # Se as duas lines coincidem o final e o começo
        if starts_and_ends1 == starts_and_ends2:
            line1 = line2
            pontos_fins = mt.hitmiss_ends_v2(line1)
            pontos_fins = pt.img_to_points(pontos_fins)
            if len(pontos_fins) == 2:
                linebaixo = linetopo = it.draw_line(
                    np.zeros(base_frame), starts_and_ends1[0], starts_and_ends1[1]
                )
                bridge_border = it.sum_imgs([line1, linetopo, line2, linebaixo]) >= 1
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(pontos_fins) > 2:
                bridge_border = it.draw_polyline(
                    np.zeros(base_frame), pontos_fins, closed=True
                )
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
                linetopo = linebaixo = np.zeros_like(line1)
            else:
                print("Special case: no solution yet")
                # TODO: still need to find a workaround here
        else:
            unique_points = []
            for p in starts_and_ends1 + starts_and_ends2:
                if p not in unique_points:
                    unique_points.append(p)
            if len(unique_points) == 4:
                extreme_points = [
                    starts_and_ends1[0],
                    starts_and_ends1[1],
                    starts_and_ends2[1],
                    starts_and_ends2[0],
                ]
                if pt.intersects(
                    [extreme_points[0], extreme_points[2]],
                    [extreme_points[1], extreme_points[3]],
                ):
                    extreme_points = [
                        starts_and_ends1[0],
                        starts_and_ends1[1],
                        starts_and_ends2[0],
                        starts_and_ends2[1],
                    ]
                linetopo = it.draw_line(
                    np.zeros(base_frame), extreme_points[0], extreme_points[2]
                )
                linebaixo = it.draw_line(
                    np.zeros(base_frame), extreme_points[1], extreme_points[3]
                )
                bridge_border = it.sum_imgs([line1, linetopo, line2, linebaixo])
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(unique_points) == 2:
                fechamento1_pts = unique_points
                linetopo = it.draw_line(
                    np.zeros(base_frame), fechamento1_pts[0], fechamento1_pts[1]
                )
                linebaixo = np.zeros_like(linetopo)
                bridge_border = it.sum_imgs([line1, linetopo])
                bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
                bridge_img = np.logical_and(bridge_img, rest_of_picture)
            elif len(unique_points) == 0:
                if np.sum(line1) > 0:
                    fechamento1_pts = unique_points
                    linetopo = np.zeros_like(line1)
                    linebaixo = np.zeros_like(line1)
                    bridge_border = line1.copy()
                    bridge_img = it.fill_internal_area(
                        bridge_border, np.ones(base_frame)
                    )
                    bridge_img = np.logical_and(bridge_img, rest_of_picture)
        return bridge_img, linetopo, linebaixo, bridge_border, line1, line2

    line1, line2 = find_contours_around_origin(
        rest_of_picture, base_frame, max_accepted, path_radius_bridg, trunk
    )
    try:
        bridge_img, linetopo, linebaixo, bridge_border, line1, line2 = (
            close_area_from_lines(line1, line2, base_frame, inicial_pnt)
        )
        bridge_border_seq = path_tools.img_to_chain(bridge_border)
    except:
        return [[], [], [], []]

    # if np.sum(line1) == np.sum(line2):
    #     extreme_external_points = [[], [], [], []]
    # else:
    while np.sum(bridge_border == 2) > 4 and len(bridge_border_seq) > 1:
        opened = mt.opening(bridge_img, kernel_size=1)
        line1b = np.logical_and(line1, opened)
        line2b = np.logical_and(line2, opened)
        if line1b.any() and line2b.any():
            line1c = it.restore_continuous(line1b)
            line2c = it.restore_continuous(line2b)
            bridge_img, linetopo, linebaixo, bridge_border, line1, line2 = (
                close_area_from_lines(line1c, line2c, base_frame, inicial_pnt)
            )
            bridge_border_seq = path_tools.img_to_chain(bridge_border)
        else:
            extreme_external_points = [[], [], [], []]
            break
    lens = [len(x) for x in bridge_border_seq]
    bridge_border_seq = bridge_border_seq[np.argmax(lens)]
    ends_topo = pt.img_to_points(sk.find_tips(linetopo))
    ends_baixo = pt.img_to_points(sk.find_tips(linebaixo))
    bridge_border_seq = path_tools.set_first_pt_in_seq(bridge_border_seq, ends_topo[0])
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
        [line1, line2, linetopo, linebaixo],
        bridge_border,
        extreme_external_points,
    )


def remove_zigzag_bridges_conflict(bridge_a, bridge_b, rest_of_picture):
    commom_sides = np.logical_and(bridge_a.contour[0], bridge_b.contour[0])
    commom_sides_pts = pt.x_y_para_pontos(np.nonzero(commom_sides))
    if len(commom_sides_pts) == 0:
        extreme_pointe = mt.hitmiss_ends_v2(
            bridge_a.contour[0], np.zeros_like(bridge_a.contour[0])
        )
        new_sides_a = np.logical_and(
            bridge_a.contour[0], np.logical_not(extreme_pointe)
        )
        new_bridge_a, new_contour_elements_a = adjust_bridge_end_lines(new_sides_a)
        bridge_a.img = new_bridge_a
        bridge_a.contour = new_contour_elements_a

        extreme_pointe = mt.hitmiss_ends_v2(
            bridge_b.contour[0], np.zeros_like(bridge_b.contour[0])
        )
        new_sides_b = np.logical_and(
            bridge_b.contour[0], np.logical_not(extreme_pointe)
        )
        new_bridge_b, new_contour_elements_b = adjust_bridge_end_lines(new_sides_b)
        bridge_b.img = new_bridge_b
        bridge_b.contour = new_contour_elements_b
    elif 0 < len(commom_sides_pts) <= 3:
        new_sides_a = np.logical_and(bridge_a.contour[0], np.logical_not(commom_sides))
        new_bridge_a, new_contour_elements_a = adjust_bridge_end_lines(new_sides_a)
        bridge_a.img = new_bridge_a
        bridge_a.contour = new_contour_elements_a

        new_sides_b = np.logical_and(bridge_b.contour[0], np.logical_not(commom_sides))
        new_bridge_b, new_contour_elements_b = adjust_bridge_end_lines(new_sides_b)
        bridge_b.img = new_bridge_b
        bridge_b.contour = new_contour_elements_b
    else:
        print("Special case: no solution yet")
        # TODO: still need to find a workaround here
    return


def adjust_bridge_end_lines(bridge_sides):
    new_base = np.zeros_like(bridge_sides)
    lines_do_limite = bridge_sides
    _, labeled, labeled_n = mt.detect_contours(lines_do_limite)
    if labeled_n > 2:
        sums = []
        for l in np.arange(0, labeled_n):
            sums.append(np.sum(labeled == l + 1))
        idx = [sums.index(i) for i in sorted(sums, reverse=True)][:2]
        line1 = labeled == idx[0] + 1
        line2 = labeled == idx[1] + 1
        lines_do_limite = np.logical_or(line1, line2)
        starts_and_ends1 = pt.x_y_para_pontos(
            np.where(mt.hitmiss_ends_v2(line1.astype(np.uint8), new_base))
        )
        starts_and_ends2 = pt.x_y_para_pontos(
            np.where(mt.hitmiss_ends_v2(line2.astype(np.uint8), new_base))
        )
        dist_1a_2 = list(
            map(
                lambda x: pt.distance_pts(starts_and_ends1[0], x),
                starts_and_ends2,
            )
        )
        dist_1b_2 = list(
            map(
                lambda x: pt.distance_pts(starts_and_ends1[1], x),
                starts_and_ends2,
            )
        )
        destiny_point_1 = starts_and_ends2[np.argmin(dist_1a_2)]
        destiny_point_2 = starts_and_ends2[np.argmin(dist_1b_2)]
        fechamento1_pts = [starts_and_ends1[0], destiny_point_1]
        fechamento2_pts = [starts_and_ends1[1], destiny_point_2]
        linetopo = it.draw_line(
            np.zeros_like(bridge_sides), fechamento1_pts[0], fechamento1_pts[1]
        )
        linebaixo = it.draw_line(
            np.zeros_like(bridge_sides), fechamento2_pts[0], fechamento2_pts[1]
        )
        bridge = np.logical_or(lines_do_limite, linetopo)
        bridge = np.logical_or(bridge, linebaixo)
        bridge = it.fill_internal_area(bridge, np.ones_like(bridge_sides, np.uint8))
        bridge = mt.dilation(bridge, kernel_size=1)
    elif labeled_n == 2:
        line1 = labeled == 1
        line2 = labeled == 2
        starts_and_ends1 = pt.x_y_para_pontos(
            np.where(mt.hitmiss_ends_v2(line1.astype(np.uint8), new_base))
        )
        starts_and_ends2 = pt.x_y_para_pontos(
            np.where(mt.hitmiss_ends_v2(line2.astype(np.uint8), new_base))
        )
        dist_1a_2 = list(
            map(
                lambda x: pt.distance_pts(starts_and_ends1[0], x),
                starts_and_ends2,
            )
        )
        dist_1b_2 = list(
            map(
                lambda x: pt.distance_pts(starts_and_ends1[1], x),
                starts_and_ends2,
            )
        )
        destiny_point_1 = starts_and_ends2[np.argmin(dist_1a_2)]
        destiny_point_2 = starts_and_ends2[np.argmin(dist_1b_2)]
        fechamento1_pts = [starts_and_ends1[0], destiny_point_1]
        fechamento2_pts = [starts_and_ends1[1], destiny_point_2]
        linetopo = it.draw_line(
            np.zeros_like(bridge_sides), fechamento1_pts[0], fechamento1_pts[1]
        )
        linebaixo = it.draw_line(
            np.zeros_like(bridge_sides), fechamento2_pts[0], fechamento2_pts[1]
        )
        bridge = np.logical_or(lines_do_limite, linetopo)
        bridge = np.logical_or(bridge, linebaixo)
        bridge = it.fill_internal_area(bridge, np.ones_like(bridge_sides, np.uint8))
        bridge = mt.dilation(bridge, kernel_size=1)
    else:
        return np.zeros_like(bridge_sides), [
            np.zeros_like(bridge_sides),
            np.zeros_like(bridge_sides),
            np.zeros_like(bridge_sides),
        ]
    return bridge, [lines_do_limite, linetopo, linebaixo]


def external_cut_zigzag(
    external_borders,
    lines_internas,
    lines_transversais,
    extreme_external_points,
    extreme_internal_points,
):
    borda_cortada = np.zeros_like(lines_internas)
    borda_normal = np.zeros_like(lines_internas)
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
        pixel_lines = lines_transversais[y][x]
        ce = pixel_lines == 1
        cl = lines_transversais[y + 1][x] and lines_transversais[y][x + 1]
        cm = lines_transversais[y + 1][x] and lines_transversais[y][x - 1]
        co = lines_transversais[y - 1][x] and lines_transversais[y][x + 1]
        cp = lines_transversais[y - 1][x] and lines_transversais[y][x - 1]
        cq = cl or cm or co or cp

        cr = lines_internas[y + 1][x]
        cs = lines_internas[y + 1][x]
        ct = lines_internas[y - 1][x]
        cu = lines_internas[y - 1][x]
        cw = lines_internas[y][x + 1]
        cx = lines_internas[y][x + 1]
        cy = lines_internas[y][x - 1]
        cz = lines_internas[y][x - 1]
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


def internal_cut(new_contour, lines, extreme_internal_points, sentido):
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
    borda_cortada = np.zeros_like(lines)
    borda_normal = np.zeros_like(lines)
    lines_transversais = lines.copy()
    counter = 0
    counter_pixels = 0
    last_change = 0
    B = it.points_to_img(new_contour, np.zeros_like(lines_transversais))
    D = np.logical_and(lines_transversais, B)
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
    lines_transversais,
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
        pixel_lines = lines_transversais[y][x]
        ce = pixel_lines == 1
        cl = lines_transversais[y + 1][x] and lines_transversais[y][x + 1]
        cm = lines_transversais[y + 1][x] and lines_transversais[y][x - 1]
        co = lines_transversais[y - 1][x] and lines_transversais[y][x + 1]
        cp = lines_transversais[y - 1][x] and lines_transversais[y][x - 1]
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
def cut_in_transversals(origens_circulos, line_c1, line_c2, close_ends=False):
    transversais = []
    canvas = np.zeros_like(line_c1)
    contour_1_lista = pt.x_y_para_pontos(np.nonzero(line_c1))
    extremos_1 = pt.img_to_points(mt.hitmiss_ends_v2(line_c1))
    contour_2_lista = pt.x_y_para_pontos(np.nonzero(line_c2))
    extremos_2 = pt.img_to_points(mt.hitmiss_ends_v2(line_c2))
    for i, o in enumerate(origens_circulos):
        if close_ends and (i == 0 or i == len(origens_circulos) - 1):
            dists = [pt.distance_pts(o, x) for x in extremos_1]
            transv1 = extremos_1[np.argmin(dists)]
        else:
            o_vec = np.array(o) - np.array(origens_circulos[i - 1])
            o_hat = o_vec / np.linalg.norm(o_vec)
            angles_w_o1 = []
            for v1 in contour_1_lista:
                v1_vec = np.array(v1) - np.array(o)
                v1_hat = v1_vec / np.linalg.norm(v1_vec)
                cos_teta1 = np.dot(o_hat, v1_hat)
                angles_w_o1.append(abs(0 - cos_teta1))
            transv1 = contour_1_lista[np.argmin(angles_w_o1)]
        angles_w_o2 = []
        if close_ends and (i == 0 or i == len(origens_circulos) - 1):
            dists = [pt.distance_pts(o, x) for x in extremos_2]
            transv2 = extremos_2[np.argmin(dists)]
        else:
            for v2 in contour_2_lista:
                v2_vec = np.array(v2) - np.array(o)
                v2_hat = v2_vec / np.linalg.norm(v2_vec)
                cos_teta2 = np.dot(o_hat, v2_hat)
                angles_w_o2.append(abs(0 - cos_teta2))
            transv2 = contour_2_lista[np.argmin(angles_w_o2)]
        transversais.append([transv1, transv2])
        canvas = it.points_to_img([transv1], canvas)
        canvas = it.points_to_img([transv2], canvas)
        canvas = it.points_to_img([o], canvas)
    return transversais, canvas


def find_contours_around_origin(rest_of_picture, base_frame, dist, path_radius, trunk):
    all_borders, all_borders_img = mt.detect_contours(rest_of_picture, return_img=True)
    area_pescocal = mt.dilation(trunk, kernel_size=(dist + 1.5 * path_radius))
    overlap = np.add(area_pescocal, all_borders_img)
    lines_do_limite = overlap == 2
    _, labeled, labeled_n = it.divide_by_connected(lines_do_limite)
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
        line1 = labeled == idx1 + 1
        line2 = labeled == idx2 + 1
    elif labeled_n == 2:
        line1 = labeled == 1
        line2 = labeled == 2
    else:
        return [], []
    return line1, line2


def close_area_from_lines(line1, line2, base_frame, new_base):
    # ORGANIZAR lineS E PONTOS EXTREMOS
    starts_and_ends1 = pt.x_y_para_pontos(
        np.where(mt.hitmiss_ends_v2(line1.astype(np.uint8)))
    )
    starts_and_ends2 = pt.x_y_para_pontos(
        np.where(mt.hitmiss_ends_v2(line2.astype(np.uint8)))
    )
    dist_1a_2 = list(
        map(lambda x: pt.distance_pts(starts_and_ends1[0], x), starts_and_ends2)
    )
    dist_1b_2 = list(
        map(lambda x: pt.distance_pts(starts_and_ends1[1], x), starts_and_ends2)
    )
    destiny_point_1 = starts_and_ends2[np.argmin(dist_1a_2)]
    destiny_point_2 = starts_and_ends2[np.argmin(dist_1b_2)]
    fechamento1_pts = [starts_and_ends1[0], destiny_point_1]
    fechamento2_pts = [starts_and_ends1[1], destiny_point_2]
    linetopo = it.draw_line(
        np.zeros(base_frame), fechamento1_pts[0], fechamento1_pts[1]
    )
    linebaixo = it.draw_line(
        np.zeros(base_frame), fechamento2_pts[0], fechamento2_pts[1]
    )
    bridge_border = it.sum_imgs([line1, linetopo, line2, linebaixo])
    bridge_img = it.fill_internal_area(bridge_border, np.ones(base_frame))
    return bridge_img, linetopo, linebaixo, bridge_border


def organize_extreme_zb_points(line, bridge_img, path_radius):
    ends = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(line)))
    centro = pt.points_center(pt.x_y_para_pontos(np.nonzero(line)))
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


# TODO: continuar daqui a arrumar as distancias de maneira mais padrão, hoje não tem ajuste
def equidistant_in_seq(line_img, path_radius, internal_mask_dist, origin_pt=[]):
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
                new_point_candidates = pt.most_distant_from(
                    origens_pontos[-1], candidates, give_second=True
                )
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


def equidistant_by_proximity(line_img, origin_lst, path_radius, img):
    # n_origens = 0
    endpoints_img = mt.hitmiss_ends_v2(line_img.astype(bool))
    endpoints = pt.img_to_points(endpoints_img)
    # line_lst = pt.img_to_points(line_img)
    first, _ = pt.closest_point(origin_lst[0], endpoints)
    last = list(filter(lambda x: x != first, endpoints))[0]
    origens_pontos = []
    for origin_pt in origin_lst:
        copied_origin = copy.deepcopy(line_img)
        canvas = np.zeros_like(line_img, np.uint8)
        centro = origin_pt
        area_distance = it.draw_circle(canvas, centro, path_radius)
        candidate, _ = pt.closest_point(centro, pt.img_to_points(copied_origin))
        origens_pontos.append(candidate)
        copied_origin = np.logical_and(copied_origin, np.logical_not(area_distance))
        # n_origens = len(origens_pontos)
    origens_pontos[0] = first
    origens_pontos[-1] = last
    return origens_pontos


def internal_adapted_polygon(line_ci1, line_ci2, lines_transversais, extreme_points):
    new_contour = np.logical_or(line_ci1, line_ci2)
    new_contour = np.logical_or(new_contour, lines_transversais).astype(np.uint8)
    new_contour, _, _ = sk.create_prune_skel(new_contour, 20)
    new_contour_cnt = mt.detect_contours(new_contour, only_external=True)
    new_contour_pts = pt.contour_to_list(new_contour_cnt)
    # new_contour_img = it.points_to_img(new_contour_pts, np.zeros_like(line_ci1))
    ref_internos = new_contour_pts
    ia, _ = pt.closest_point(extreme_points[0], ref_internos)
    ib, _ = pt.closest_point(extreme_points[1], ref_internos)
    ic, _ = pt.closest_point(extreme_points[2], ref_internos)
    id, _ = pt.closest_point(extreme_points[3], ref_internos)
    extreme_internal_points = [ia, ib, ic, id]
    return new_contour_pts, extreme_internal_points


def oscilatory_start_and_end(new_zigzag, extreme_points):
    fins_da_rota = pt.x_y_para_pontos(
        np.nonzero(mt.hitmiss_ends_v2(new_zigzag.astype(bool)))
    )
    return fins_da_rota


def reduce_origin(candidate, necks_max_paths, norm_dist_map):
    """Determines the start and end for each trunk, then reduces the margins until all its bottlenecks are encompassed"""
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
    if np.sum(reduced_origin) == 0:
        return np.zeros_like(candidate), []
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
        print("ERROR")
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
    bridge_region.find_center()
    return bridge_region


def make_zz_or_co_bridge_route(
    region: Bridge, path_radius, mask_distancer, internal_mask_dist, rest_of_picture
):

    total_sobreposition = mt.dilation(region.img, kernel_size=path_radius)
    eroded = mt.erosion(total_sobreposition, kernel_img=mask_distancer)
    rest_pict_total_sobreposition = mt.dilation(
        rest_of_picture, kernel_size=path_radius
    )
    rest_pict_eroded = mt.erosion(
        rest_pict_total_sobreposition, kernel_img=mask_distancer
    )
    _, eroded_border = mt.detect_contours(eroded, return_img=True, only_external=True)
    if len(np.unique(it.sum_imgs([region.origin, eroded_border]))) < 3:
        origin_seq = path_tools.img_to_chain(region.origin)[0]
        origin_seq = path_tools.set_first_pt_in_seq(
            origin_seq, pt.img_to_points(mt.hitmiss_ends_v2(region.origin))[0]
        )
        origin_seq = path_tools.cut_repetition(origin_seq)
        tng_end = path_tools.draw_tangent_from_seq(
            list(reversed(origin_seq)), path_radius * 4, np.zeros_like(eroded)
        )
        tng_start = path_tools.draw_tangent_from_seq(
            origin_seq, path_radius * 4, np.zeros_like(eroded)
        )
        origin = np.logical_or(tng_start, np.logical_or(region.origin, tng_end))
    else:
        origin = region.origin
    origin_axis = np.logical_and(origin, eroded)
    _, _, n = it.divide_by_connected(origin_axis)
    if np.sum(origin_axis) > 0 and n == 1:
        _, _, n_divisions = it.divide_by_connected(origin_axis)
        if n_divisions > 1:
            origin_axis, eroded = connect_origin_parts(region.origin, eroded)
        line_ci1, line_ci2 = region.make_internal_border(
            rest_pict_eroded, total_sobreposition, eroded, origin_axis, path_radius
        )
        if np.equal(line_ci1, line_ci2).all():
            new_zigzag = region.origin
            new_zigzag_b = region.route
        else:
            pts_trns_origin = equidistant_in_seq(
                origin_axis, path_radius, internal_mask_dist
            )
            pts_trns_ci1 = equidistant_by_proximity(
                line_ci1, pts_trns_origin, path_radius, total_sobreposition
            )
            pts_trns_ci2 = equidistant_by_proximity(
                line_ci2, pts_trns_origin, path_radius, total_sobreposition
            )
            lines_transversais = np.zeros_like(region.img)
            lines_limitrofes = np.zeros_like(region.img)
            for i, point in enumerate(pts_trns_origin):
                if i == 0 or i == len(pts_trns_origin) - 1:
                    thisline = it.draw_polyline(
                        lines_limitrofes,
                        [pts_trns_ci1[i], pts_trns_origin[i], pts_trns_ci2[i]],
                        False,
                    )
                    lines_limitrofes = np.logical_or(lines_limitrofes, thisline)
                else:
                    thisline = it.draw_polyline(
                        lines_transversais,
                        [pts_trns_ci1[i], pts_trns_origin[i], pts_trns_ci2[i]],
                        False,
                    )
                    lines_transversais = np.logical_or(lines_transversais, thisline)
            new_contour, new_contour_img = mt.detect_contours(
                it.sum_imgs([lines_limitrofes, line_ci1, line_ci2]),
                return_img=True,
                only_external=True,
            )
            new_contour = pt.contour_to_list(new_contour)
            ends_line_ci1 = pt.img_to_points(mt.hitmiss_ends_v2(line_ci1))
            ends_line_ci2 = pt.img_to_points(mt.hitmiss_ends_v2(line_ci2))
            ia, _ = pt.closest_point(region.extreme_points[0], ends_line_ci1)
            ib = [x for x in ends_line_ci1 if x != ia][0]
            ic, _ = pt.closest_point(region.extreme_points[2], ends_line_ci2)
            id = [x for x in ends_line_ci2 if x != ic][0]
            extr_int_pts = [ia, ib, ic, id]
            new_contour = path_tools.set_first_pt_in_seq(new_contour, extr_int_pts[0])
            new_zigzag = weaving_zigzag(
                new_contour,
                new_contour_img,
                lines_transversais,
                lines_limitrofes,
                extr_int_pts,
                0,
            )
            new_zigzag = np.logical_or(new_zigzag, lines_transversais)
            new_zigzag, _, _ = sk.create_prune_skel(new_zigzag, 2)

            new_zigzag_b = weaving_zigzag(
                new_contour,
                new_contour_img,
                lines_transversais,
                lines_limitrofes,
                extr_int_pts,
                1,
            )
            new_zigzag_b = np.logical_or(new_zigzag_b, lines_transversais)
            new_zigzag_b, _, _ = sk.create_prune_skel(new_zigzag_b, 2)

        region.reference_points = pt.x_y_para_pontos(
            np.nonzero(mt.hitmiss_ends_v2(new_zigzag))
        )
        region.reference_points_b = pt.x_y_para_pontos(
            np.nonzero(mt.hitmiss_ends_v2(new_zigzag_b))
        )
        if len(region.reference_points) < 2:
            new_zigzag = path_tools.one_pixel_wide(
                mt.dilation(new_zigzag, kernel_size=2)
            )
            region.reference_points = pt.x_y_para_pontos(
                np.nonzero(mt.hitmiss_ends_v2(new_zigzag))
            )
            # print("ajdnsabhfsfb")
        if len(region.reference_points_b) < 2:
            new_zigzag_b = path_tools.one_pixel_wide(
                mt.dilation(new_zigzag_b, kernel_size=2)
            )
            region.reference_points_b = pt.x_y_para_pontos(
                np.nonzero(mt.hitmiss_ends_v2(new_zigzag_b))
            )
            # print("ajdnsabhfsfb")
        region.find_center()
    else:
        reducted_origin = it.take_the_bigger_area(region.origin)
        reduction_points = mt.dilation(
            mt.hitmiss_ends_v2(reducted_origin), kernel_size=path_radius
        )
        reducted_origin = it.image_subtract(reducted_origin, reduction_points)
        new_zigzag = reducted_origin
        new_zigzag_b = new_zigzag
        region.reference_points = pt.x_y_para_pontos(
            np.nonzero(mt.hitmiss_ends_v2(region.origin.astype(bool)))
        )
        region.reference_points_b = region.reference_points
        region.find_center()
    region.route = new_zigzag
    region.route_b = new_zigzag_b
    # aaaa = it.sum_imgs([region.route, line_ci1,line_ci2,eroded, it.points_to_img(pts_trns_ci1, np.zeros_like(eroded)), region.img, lines_limitrofes, origin_axis, region.origin])
    region.trail = mt.dilation(region.route, kernel_size=path_radius)
    region.trail_b = mt.dilation(region.route_b, kernel_size=path_radius)
    return region


def weaving_zigzag(
    new_contour,
    new_contour_img,
    lines_transversais,
    lines_limitrofes,
    extr_int_pts,
    sentido,
):
    cutted_border = internal_cut(new_contour, lines_transversais, extr_int_pts, sentido)
    new_zigzag = np.logical_or(lines_transversais, cutted_border)
    new_zigzag = np.logical_or(new_zigzag, lines_limitrofes)
    _, _, n = it.divide_by_connected(new_zigzag)
    reference_points_b = pt.x_y_para_pontos(np.nonzero(mt.hitmiss_ends_v2(new_zigzag)))
    if n > 1:
        extr_int_pts = [
            extr_int_pts[0],
            extr_int_pts[2],
            extr_int_pts[1],
            extr_int_pts[3],
        ]
        print("Corrected sequence: b and c inverted")
        new_zigzag = internal_cut(
            new_contour, lines_transversais, extr_int_pts, sentido
        )
        new_zigzag = np.logical_or(new_zigzag, lines_limitrofes)
    _, _, n = it.divide_by_connected(mt.hitmiss_ends_v2(new_zigzag))
    if n <= 1:
        cccc = mt.closing(new_zigzag, kernel_size=1)
        ddd, _, _ = sk.create_prune_skel(cccc, 1)
        _, eee, n = it.divide_by_connected(mt.hitmiss_ends_v2(ddd))
        if n > 1:
            new_zigzag = ddd
    if len(reference_points_b) < 2:
        print("ERROR: no solution yet")
    return new_zigzag


def decompose_pol_cont_by_corners(lines_do_limite, trunk, path_radius_bridg):
    pontos = path_tools.img_to_chain(lines_do_limite)[0]
    # pontos_curvatura = path_tools.encontrar_pontos_curvatura(pontos+[pontos[0]]+[pontos[1]])
    pontos_curvatura = path_tools.find_curvature_pts(pontos)
    if len(pontos_curvatura) < 4:
        harris_pts = corner_peaks(
            corner_harris(lines_do_limite), min_distance=5, threshold_rel=0.02
        )
        harris_pts = [list(x) for x in harris_pts]
        pontos_corrigidos = []
        for point in harris_pts:
            if not (point in pontos):
                pnt, _ = pt.closest_point(point, pontos)
                pontos_corrigidos.append(pnt)
            else:
                pontos_corrigidos.append(point)
        pontos_curvatura = pontos_corrigidos
    # pontos_curvatura_img = it.points_to_img(pontos_curvatura, np.zeros_like(lines_do_limite))
    pontos = path_tools.set_first_pt_in_seq(pontos, pontos_curvatura[0])
    segments = path_tools.colorbyevent(
        pontos, pontos_curvatura, np.zeros_like(lines_do_limite)
    )
    segments_n = max(np.unique(segments))
    trunque = trunk > 0
    trunk_seq = path_tools.img_to_chain(trunque)[0]
    trunk_seq = path_tools.set_first_pt_in_seq(
        trunk_seq, pt.img_to_points(mt.hitmiss_ends_v2(trunque))[0]
    )
    trunk_seq = path_tools.cut_repetition(trunk_seq)
    # TODO: ver se aqui tem como arrumar esses tamanhos
    tng_end = path_tools.draw_tangent_from_seq(
        list(reversed(trunk_seq)), path_radius_bridg * 4, np.zeros_like(lines_do_limite)
    )
    tng_start = path_tools.draw_tangent_from_seq(
        trunk_seq, path_radius_bridg * 4, np.zeros_like(lines_do_limite)
    )
    possible_c1_c2 = np.zeros_like(lines_do_limite)
    counter_accepted = 0
    for labl in np.add(list(range(segments_n)), 1):
        contact = it.sum_imgs([tng_end, tng_start, np.int32(segments == labl)])
        if not ((contact == 2).any()):
            counter_accepted += 1
            possible_c1_c2 = it.sum_imgs(
                [possible_c1_c2, np.multiply(segments == labl, counter_accepted)]
            )
    return possible_c1_c2, counter_accepted, pontos_curvatura
