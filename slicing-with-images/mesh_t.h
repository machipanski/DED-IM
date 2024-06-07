#ifndef _mesh_h_
#define _mesh_h_

#include <vector>
#include "point_r3_t.h"
#include "triangle_t.h"

using namespace std;

class mesh_t {

  public:

    mesh_t() : bottomLeftVertex(999999,999999,999999), upperRightVertex(-999999,-999999,-999999) { meshSize = 0;}

    size_t size() const;

    void push_back (triangle_t &t);

    point_r3_t meshAABBSize() const;

    const vector<triangle_t>& getvTriangle() const;

    point_r3_t getBottomLeftVertex() const;

    point_r3_t getUpperRightVertex() const;

  public:

    int meshSize;
    point_r3_t bottomLeftVertex;
    point_r3_t upperRightVertex;
    vector<triangle_t> vTriangle;
};

#endif
