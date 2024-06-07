#ifndef _io_h_
#define _io_h_

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
#include <fstream>
#include <unistd.h>
#include <algorithm>

#include "point_r3_t.h"
#include "mesh_t.h"
#include "line_t.h"
#include "triangle_t.h"
#include "write_svg.h"
#include "contour.h"

using namespace std;

#define FILE_STL_BIN 0
#define FILE_STL_ASCII 1
#define FILE_AMF_ASCII 2

float xround (float x, double eps, int mod, int rem);

point_r3_t point_r3_t_round (float x, float y, float z, double eps);

bool degenerate (triangle_t t);

triangle_t make_triangle (float n0, float n1, float n2, float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, const char *rotate, double eps); 

int stlToMeshInMemory (const char *stlFile, mesh_t *mesh, bool isBinaryFormat, const char *rotate, double eps);

int checkASCIIFile (const char *fileName);

void record_polygons (vector<contour> polygons, FILE *file);

#endif
