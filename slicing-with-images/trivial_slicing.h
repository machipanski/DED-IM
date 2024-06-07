#ifndef _trivial_slicing_h_
#define _trivial_slicing_h_

#include <string>
#include <vector>
#include <cmath>
#include <ctime>
#include <set>
#include <unordered_map>
#include <limits>
#include <array>
#include <map>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>

#include "point_r3_t.h"
#include "mesh_t.h"
#include "line_t.h"
#include "triangle_t.h"
#include "write_svg.h"
#include "contour.h"
#include "incr_slicing.h"

using namespace std;

point_r3_t TrivialClosure_find_B (vector<line_t>& segs, point_r3_t u, int plane);

point_r3_t TrivialClosure_find_A (vector<line_t>& segs, point_r3_t u, int plane);

void TrivialLoopClosure (vector<line_t> segs, vector<contour> polygons[], int plane);

void TrivialSlicing (const mesh_t *mesh, vector<float> P, vector<contour> polygons[], bool chaining, bool orienting, long *intersections, double *slicing_time, double *loopclosure_time); 

#endif
