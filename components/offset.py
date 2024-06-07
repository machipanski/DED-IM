class Area:
    """Cada contorno fechado em um level gerado pelos Offsets"""

    def __init__(self, nome, img, origem, loops_inside):
        self.name = nome
        self.img = img
        self.offset_level = origem
        self.loops_inside = loops_inside
        self.internal_area = []


class Region:
    """Caminho fechado individual por Offset paralelo"""

    def __init__(self, name, img, loops):
        self.name = name
        self.img = img
        self.loops = loops
        self.limmit_coords = (
            []
        )  # coordenadas dos pontos onde se separam as regiões monotônicas
        self.center_coords = []  # coordenadas do centro geométrico de cada contorno
        self.area_contour = []  # contorno de cada área
        self.area_contour_img = []
        self.internal_area = []  # resultante de se pintar o interior de cada contorno
        self.hierarchy = (
            0  # hierarquia de contornos, quais são internos e quais são externos
        )
        self.paralel_points = []
        self.route = []
        self.trail = []
        self.next_prohibited_area = []
