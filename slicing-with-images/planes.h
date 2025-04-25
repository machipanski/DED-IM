#ifndef _planes_h_
#define _planes_h_

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

#include "io.h"
#include "point_r3_t.h"
#include "mesh_t.h"
#include "line_t.h"
#include "triangle_t.h"
#include "write_svg.h"
#include "contour.h"
#include "incr_slicing.h"

using namespace std;

vector<float> compute_planes (const mesh_t *mesh, float max_thickness, char *adaptive, double eps, float *delta);

#endif
