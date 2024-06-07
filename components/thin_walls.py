import numpy as np
from components import images_tools as it
from components import morphology_tools as mt
from components import skeleton as sk
from components import bottleneck
from components import points_tools as pt

class ThinWall:
    """Cada região que precisa ser feita com menos de duas trilhas antes de fazer os contornos"""

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
    ):
        if elementos_contorno is None:
            elementos_contorno = []
        if pontos_extremos is None:
            pontos_extremos = []
        self.name = name
        self.img = img
        self.origin = origin
        self.destiny = 0
        self.n_paths = n_paths
        self.origin_mark = origin_marks
        self.trunk = trunk
        self.contorno = elementos_contorno
        self.pontos_extremos = pontos_extremos
        # self.origin_coords = []
        # self.destiny_coords = []
        # self.route = []
        # self.trail = []
        # self.center = []
        # self.interruption_points = []
        # self.reference_points = []
        return


class ThinWallRegions:
    """O grupo de regiões chamadas de Thin Walls, gruarda também a configuração geral para essas regiões"""

    def __init__(self):
        self.medial_transform = []
        self.regions = []
        self.all_thin_walls = [] 
        self.all_thin_origins = []
        self.routes = []
        return

    def make_thin_wallsV2(self, original_img, base_frame, path_radius, mask):
        sem_galhos, sem_galhos_dist, trunks = sk.create_prune_divide_skel(
            original_img.astype(np.uint8), 16*path_radius# TODO:Achar o melhor size para o serviço
        )
        self.medial_transform = sem_galhos * sem_galhos_dist
        trunks = [pt.contour_to_list([x]) for x in trunks]
        trunks = [it.points_to_img(x, np.zeros(base_frame)) for x in trunks]
        all_thin_walls = np.zeros(base_frame)
        all_origins = np.zeros(base_frame)
        max_width = 2 * path_radius
        counter = 0
        for i, trunk in enumerate(trunks):
            trunk = (trunk * sem_galhos_dist) 
            n_trilhas = trunk / (2*path_radius)
            n_trilhas_max = np.min(n_trilhas[np.nonzero(n_trilhas)])
            bridge_origin = np.logical_and(trunk != 0, trunk < max_width)
            if np.sum(bridge_origin) > 0:
                bridge_origin = it.take_the_bigger_area(bridge_origin) # TODO:ARRUMAR A CRIAÇÃO DAS ORIGENS
                try:
                    bridge_img, elementos_contorno, contorno, pontos_extremos = (
                        bottleneck.close_bridge_contour_v2(
                            bridge_origin,
                            base_frame,
                            max_width,
                            original_img,
                            mask,
                            path_radius,
                        )
                    )
                    print("OK: fechou contorno")
                    # if np.sum(bridge_img) > 0:
                    all_thin_walls = np.logical_or(all_thin_walls, bridge_img)
                    all_origins = np.logical_or(all_origins, bridge_origin)
                    y_mark = np.where(bridge_origin)[1][
                        np.round(len(np.where(bridge_origin)))
                    ]
                    x_mark = np.where(bridge_origin)[0][
                        np.round(len(np.where(bridge_origin)))
                    ]
                    origin_mark = [y_mark, x_mark, str(n_trilhas_max)]
                    self.regions.append(
                        ThinWall(
                            i,
                            bridge_img,
                            trunk,
                            trunk,
                            n_trilhas_max,
                            origin_mark,
                            elementos_contorno,
                            pontos_extremos,
                        )
                    )
                    counter += 1
                except Exception:
                    print("\033[3#m" + "Erro: nao fechou contorno" + "\033[0m")
                    break
        self.all_thin_walls = all_thin_walls
        self.all_origins = all_origins
        return

    def apply_thin_walls(self, original, base_frame):
        rest_of_picture_f1 = np.zeros(base_frame)
        rest_of_picture_f1 = np.logical_or(original, rest_of_picture_f1)
        for region in self.regions:
            rest_of_picture_f1 = np.logical_and(
                rest_of_picture_f1, np.logical_not(region.img)
            )
        return rest_of_picture_f1.astype(np.uint8)