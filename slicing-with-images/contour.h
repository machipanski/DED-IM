#ifndef _contour_h_
#define _contour_h_

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <set>
#include <unordered_map>
#include <limits>
#include <array>
#include <map>
#include <sys/stat.h>
#include <unistd.h>

#include "point_r3_t.h"
#include "mesh_t.h"
#include "line_t.h"
#include "triangle_t.h"

typedef struct _contour {
   bool external;
   bool clockwise;
   vector<point_r3_t> P;
} contour;

typedef struct _bounding_box {
  double xMin;
  double xMax;
  double yMin;
  double yMax;
  bool firstPoint;
} bounding_box;

void ray_casting (vector<contour> &polygons);

bool ray_intersect (line_t ray, line_t side);

bool is_inside (line_t line, point_r3_t point);

line_t create_ray (point_r3_t point, bounding_box bb, int index);

bounding_box create_bounding_box ();

void update_bounding_box (point_r3_t point, bounding_box *bb);

bool insided_bounding_box (point_r3_t point, bounding_box bb);

bool contains (point_r3_t point, bounding_box bb, vector<line_t> sides, int index);

void add_point (point_r3_t p1, point_r3_t p2, vector<line_t> &t, bounding_box *bb, bool first, int index);

#endif
