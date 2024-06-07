import itertools
import numpy as np
from components import path_tools
from scipy.spatial import distance
from skimage.morphology import convex_hull_image

def closest_point(pt, pts_list):
    """returns the closest point in the list from the point indicated as (y,x) and the minimal distance"""
    dist_2 = [distance_pts(x, pt) for x in pts_list]
    return pts_list[np.argmin(dist_2)], np.min(dist_2)

def contour_to_list(ctr):
    """takes the contour notation of CV2 returns and gives back a list of points (y,x)"""
    return [[x[0][1], x[0][0]] for x in ctr[0].tolist()]

def distance_pts(a, b):
    return distance.euclidean(a, b)

def img_to_points(img: np.ndarray):
    """Translates a img array as a list of points (y,x)"""
    pts = x_y_para_pontos(np.nonzero(img.astype(bool)))
    return pts

def invert_x_y(pts_list):
    """change (x,y) to (y,x) in a list or vice-versa"""
    return [[a[1], a[0]] for a in pts_list]

def points_center(pts_list):
    """returns the average of all x and y coordinates of an list as a point (y,x)"""
    n_points = len(pts_list)
    x = 0
    y = 0
    for kp in pts_list:
        x = x + kp[1]
        y = y + kp[0]
    center_coords = [int(np.ceil(y / n_points)), int(np.ceil(x / n_points))]
    return center_coords

def x_y_para_pontos(Xlist_Ylist):
    """[listX, listY] to list(y,x)"""
    pts_list = []
    for i in np.arange(0, len(Xlist_Ylist[0])):
        pts_list.append([Xlist_Ylist[1][i], Xlist_Ylist[0][i]])
    pts_list = invert_x_y(pts_list)
    return pts_list

def most_distant(pts_list):
    """Returns the pair of the most distant points in a list as ([y,x],[y,x])"""
    pairs = []
    distances = []
    for p1, p2 in itertools.combinations(pts_list, 2):
        pairs.append([p1, p2])
        distances.append(distance_pts(p1, p2))
    return pairs[np.argmax(distances)]

def multiple_contours_to_list(ctrs_list, minimal_seq=0):
    list_of_contours_pts = []
    for i, ctr in enumerate(ctrs_list):
        seq = [[x[0][1], x[0][0]] for x in ctrs_list[i].tolist()]
        if len(seq) >= minimal_seq:
            list_of_contours_pts.append(seq)
    return list_of_contours_pts