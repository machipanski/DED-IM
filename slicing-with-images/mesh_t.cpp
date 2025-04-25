#include "point_r3_t.h"
#include "triangle_t.h"
#include "mesh_t.h"

using namespace std;

  //TriangleMesh() : bottomLeftVertex(999999,999999,999999), upperRightVertex(-999999,-999999,-999999) { meshSize = 0;}

  size_t mesh_t::size() const {
    return meshSize;
  }

  void mesh_t::push_back (triangle_t &t) {
     meshSize++;
     vTriangle.push_back(t);
     for (size_t i = 0; i < 3; ++i) {
        if (t.v[i].x < bottomLeftVertex.x) { bottomLeftVertex.x = t.v[i].x; }
        if (t.v[i].y < bottomLeftVertex.y) { bottomLeftVertex.y = t.v[i].y; }
        if (t.v[i].z < bottomLeftVertex.z) { bottomLeftVertex.z = t.v[i].z; }
        if (t.v[i].x > upperRightVertex.x) { upperRightVertex.x = t.v[i].x; }
        if (t.v[i].y > upperRightVertex.y) { upperRightVertex.y = t.v[i].y; }
        if (t.v[i].z > upperRightVertex.z) { upperRightVertex.z = t.v[i].z; }
     }
  }

  point_r3_t mesh_t::meshAABBSize() const {
     return point_r3_t ( upperRightVertex.x - bottomLeftVertex.x,
                       upperRightVertex.y - bottomLeftVertex.y,
                       upperRightVertex.z - bottomLeftVertex.z );
  }

  const vector<triangle_t>& mesh_t::getvTriangle() const { return vTriangle; }

  point_r3_t mesh_t::getBottomLeftVertex() const { return bottomLeftVertex; }

  point_r3_t mesh_t::getUpperRightVertex() const { return upperRightVertex; }
