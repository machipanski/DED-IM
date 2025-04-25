#ifndef _incr_slicing_h_
#define _incr_slicing_h_

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

using namespace std;

/*-----------------------------------------------------------------------*/
/*Structures*/
typedef struct node {
  triangle_t t;
  struct node *next;
  struct node *prev;
} Mesh_triangle_t_Node_t;

typedef struct _list {
   Mesh_triangle_t_Node_t *head;
   Mesh_triangle_t_Node_t *tail;
} Mesh_triangle_t_List_t;

/*-----------------------------------------------------------------------*/
/*Hash table*/ 
template<typename T> inline void hash_combine(size_t &seed, const T &v) {
  hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct HashV3 {
  size_t operator() (const point_r3_t &v) const {
    size_t h = hash<float>()(v.x);
    hash_combine (h, v.y);
    hash_combine (h, v.z);
    return h;
  }
};

typedef unordered_map<point_r3_t, vector<point_r3_t>, HashV3> PointMesh;

/*-----------------------------------------------------------------------*/
int IncrementalSlicing_binary_search (float zMin, vector<float> P);

/*-----------------------------------------------------------------------*/
Mesh_triangle_t_List_t* Mesh_triangle_t_List_create (void);

/*-----------------------------------------------------------------------*/
void Mesh_triangle_t_List_insert (triangle_t t, Mesh_triangle_t_List_t *L);

/*-----------------------------------------------------------------------*/
void Mesh_triangle_t_List_union (Mesh_triangle_t_List_t *L1, Mesh_triangle_t_List_t *L2);

/*-----------------------------------------------------------------------*/
Mesh_triangle_t_Node_t* Mesh_triangle_t_List_remove (Mesh_triangle_t_List_t *L, Mesh_triangle_t_Node_t *node);

/*-----------------------------------------------------------------------*/
point_r3_t R3_Mesh_Side_slice (point_r3_t vi, point_r3_t vj, float Z);

/*-----------------------------------------------------------------------*/
line_t R3_Mesh_triangle_t_slice (Mesh_triangle_t_Node_t *t, float Z);

/*-----------------------------------------------------------------------*/
line_t R3_Mesh_triangle_t_slice (triangle_t t, float Z);

/*----------------------------------------------------------------------*/
Mesh_triangle_t_List_t** IncrementalSlicing_buildLists (bool srt, double delta, const mesh_t *mesh, vector<float> P);

/*-----------------------------------------------------------------------*/
vector<point_r3_t> IncrementalStartLoop(vector<PointMesh> &H);

/*-----------------------------------------------------------------------*/
void IncrementalExtendLoop(vector<point_r3_t> &P, vector<PointMesh> &H);

/*-----------------------------------------------------------------------*/
void IncrementalReverseLoop(vector<point_r3_t> &P);

/*-----------------------------------------------------------------------*/
void ContourConstruction (vector<line_t> segs, vector<contour> polygons[], int plane);

/*-----------------------------------------------------------------------*/
void IncrementalSlicing ( const mesh_t *mesh, vector<float> P, float delta, bool srt, vector<contour> polygons[], bool chaining, bool orienting, long *intersections, double *slicing_time, double *loopclosure_time);

#endif
