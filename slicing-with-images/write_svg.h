#ifndef _write_svg_h_
#define _write_svg_h_

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

#include "point_r3_t.h"
#include "mesh_t.h"
#include "line_t.h"
#include "triangle_t.h"
#include "write_svg.h"
#include "contour.h"

#define DEG_TO_RAD(x) (x*0.0174532925199f)

typedef struct _rgb {
    int r, g, b;
} rgb;

using namespace std;

void export_svg (char filename[], const vector<vector<point_r3_t> > &Polygons, int nslices, int nsegments, bool video, vector<bool> orientation);

void export_svg_2d (string fileName, vector<contour> polygons[], int nplanes, const point_r3_t &aabbSize);

void export_svg_no_chaining (string fileName, vector<line_t> Segments[], int k, const point_r3_t &aabbSize);

void export_video (vector<contour> polygons[], int nplanes);

void export_svg_3d (vector<contour> polygons[], int nplanes);

void fill_colors (rgb colors[], int nrgb);

void add_svg_information (FILE *file);

#endif
