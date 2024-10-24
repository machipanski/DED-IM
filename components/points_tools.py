from ast import List
import itertools
import numpy as np
from components import path_tools
from scipy.spatial import distance
from skimage.morphology import convex_hull_image
from components import images_tools as it


def closest_point(pt, pts_list):
    """returns the closest point in the list from the point indicated as (y,x)
    and the minimal distance"""
    dist_2 = [distance_pts(x, pt) for x in pts_list]
    return pts_list[np.argmin(dist_2)], np.min(dist_2)


def closest_line(pt, lines_pts_list):
    """returns the closest list of points from the point indicated as a list of
    points(y,x) and the minimal distance"""
    list_distances = []
    for line in lines_pts_list:
        _, this_line_dist = closest_point(pt, line)
        list_distances.append(this_line_dist)
    return lines_pts_list[np.argmin(list_distances)], np.min(list_distances)


def contour_to_list(ctr):
    """takes the contour notation of CV2 returns and gives back a list of points (y,x)"""
    return [[x[0][1], x[0][0]] for x in ctr[0].tolist()]


def distance_pts(a, b):
    return distance.euclidean(a, b)


def extreme_points(img, force_top=False):
    """gives the 4 most extreme points (max_and min for Y and X)
    of an image in form of [a,b,c,d] each as (y,x)"""
    _, _, n = it.divide_by_connected(img)
    if n == 1 and force_top == False:
        chull = convex_hull_image(img)
        sequence = path_tools.img_to_chain(chull)
        sequence_simpl = path_tools.simplifica_retas_master(sequence, 0.001, [])
        sequence_simpl.remove(["a", "a"])
        regyar = it.chain_to_lines(
            [[x[1], x[0]] for x in sequence_simpl], np.zeros_like(img)
        )
        if len(sequence_simpl) < 4:
            considered = np.where(img != 0)
        else:
            ffdfsfdf = [x[0] for x in sequence_simpl]
            rtertrtet = [x[1] for x in sequence_simpl]
            considered = [ffdfsfdf, rtertrtet]
            reduced = x_y_para_pontos(considered)
            first_pair = most_distant(reduced)
            reduced.remove(first_pair[0])
            reduced.remove(first_pair[1])
            second_pair = most_distant(reduced)
            the_farthest = first_pair + second_pair
            ffdfsfdf = [x[0] for x in the_farthest]
            rtertrtet = [x[1] for x in the_farthest]
            considered = [ffdfsfdf, rtertrtet]
    else:
        considered = np.where(img != 0)
    top = np.min(considered[0])
    bottom = np.max(considered[0])
    point_a = [top, np.min(np.where(img[top]))]
    point_d = [top, np.max(np.where(img[top]))]
    point_b = [bottom, np.min(np.where(img[bottom]))]
    point_c = [bottom, np.max(np.where(img[bottom]))]
    extreme = [point_a, point_b, point_c, point_d]
    fsff = np.add(img, it.points_to_img(extreme, np.zeros_like(img)))
    return extreme


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


def max_e_min_coords_img(img, axis):
    """coordinates of the points in a (y,x) array"""
    A = img_to_points(img)
    max = np.max([x[axis] for x in A])
    min = np.min([x[axis] for x in A])
    return max, min


def most_distant(pts_list):
    """Returns the pair of the most distant points in a list as ([y,x],[y,x])"""
    pairs = []
    distances = []
    for p1, p2 in itertools.combinations(pts_list, 2):
        pairs.append([p1, p2])
        distances.append(distance_pts(p1, p2))
    return pairs[np.argmax(distances)]


def most_distant_from(pt, pts_list):
    """Returns the most distant point in a list as ([y,x],[y,x])"""
    pairs = []
    distances = []
    for p2 in pts_list:
        pairs.append([pt, p2])
        distances.append(distance_pts(pt, p2))
    return pairs[np.argmax(distances)][1]


def multiple_contours_to_list(ctrs_list, minimal_seq=0):
    list_of_contours_pts = []
    for i, ctr in enumerate(ctrs_list):
        seq = [[x[0][1], x[0][0]] for x in ctrs_list[i].tolist()]
        if len(seq) >= minimal_seq:
            list_of_contours_pts.append(seq)
    return list_of_contours_pts


def organize_points_to_a_polygon(pts_list):
    from math import atan2
    """Organize points to avoid crossing lines when traced."""
    def calculate_centroid(pts_list):
        """Calculate the centroid of a list of points."""
        x_coords = [point[0] for point in pts_list]
        y_coords = [point[1] for point in pts_list]
        centroid_x = sum(x_coords) / len(pts_list)
        centroid_y = sum(y_coords) / len(pts_list)
        return (centroid_x, centroid_y)

    def angle_from_centroid(point, centroid):
        """Calculate the angle of the point relative to the centroid."""
        return atan2(point[1] - centroid[1], point[0] - centroid[0])

    if len(pts_list) < 3:
        return pts_list  # Not enough points to form a polyline
    
    centroid = calculate_centroid(pts_list)
    # Sort points based on the angle from the centroid
    sorted_points = sorted(pts_list, key=lambda point: angle_from_centroid(point, centroid))
    return sorted_points
