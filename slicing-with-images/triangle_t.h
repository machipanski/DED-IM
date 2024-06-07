#ifndef _triangle_h_
#define _triangle_h_

#include <sstream>
#include <iostream>
#include "point_r3_t.h"

using namespace std;

class triangle_t {

  public:

    triangle_t (point_r3_t n, point_r3_t v0, point_r3_t v1, point_r3_t v2, double tangle);

    void setZMin (float z);

    void setZMax (float z);

    triangle_t& operator-=(const point_r3_t &pt);

    bool operator<(const triangle_t &t);

    friend std::ostream& operator<<(std::ostream &os, const triangle_t &t);

  public:

    point_r3_t v[3];
    point_r3_t normal;
    float zMin;
    float zMax;
    double tangle;
};

#endif
