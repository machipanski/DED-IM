#include "point_r3_t.h"
#include "triangle_t.h"

  /*----------------------*/
  triangle_t::triangle_t (point_r3_t n, point_r3_t v0, point_r3_t v1, point_r3_t v2, double tang) : normal(n) {
     v[0] = v0;
     v[1] = v1;
     v[2] = v2;
     zMin = +99999999.9;
     zMax = -99999999.9;
     setZMin(v0.z); setZMin(v1.z); setZMin(v2.z);
     setZMax(v0.z); setZMax(v1.z); setZMax(v2.z);
     tangle = tang;
  }

  /*----------------------*/
  void triangle_t::setZMin (float z) {
     if (z < zMin) {
        zMin = z;
     }
  }

  /*----------------------*/
  void triangle_t::setZMax (float z) {
     if (z > zMax) {
        zMax = z;
     }
  }

  /*----------------------*/
  triangle_t& triangle_t::operator-=(const point_r3_t &pt) {
     v[0] -= pt;
     v[1] -= pt;
     v[2] -= pt;
     return *this;
  }

  /*----------------------*/
  bool triangle_t::operator<(const triangle_t &t) {
     return zMin < t.zMin;
  }

  /*----------------------*/
  std::ostream& operator<<(std::ostream& os, const triangle_t &t) {
     os << "V0: (" << t.v[0] << "); V1: (" << t.v[1] << "); V2: (" << t.v[2] << ")";
     return os;
  }

