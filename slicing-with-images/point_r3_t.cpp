#include "point_r3_t.h"

  //point_r3_t::point_r3_t (float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}

  /*----------------------*/
  float point_r3_t::distTo (const point_r3_t &pt) const {
     return sqrt ( pow(fabs(x-pt.x), 2.0) + pow(fabs(y-pt.y), 2.0) + pow(fabs(z-pt.z), 2.0) );
  }

  /*----------------------*/
  array<float,3> point_r3_t::getCoords() {
     array<float,3> c = {{x, y, z}};
     return c;
  }

  /*----------------------*/
  float point_r3_t::dotproduct (const point_r3_t &v) const {
     return (x*v.x + y*v.y + z*v.z);
  }

  /*----------------------*/
  void point_r3_t::transform (const glm::mat4 &mat) {
     glm::vec4 v = glm::vec4(x, y, z, 1.0);
     glm::vec4 vt = mat*v;
     x = (vt.x); y = (vt.y); z = (vt.z);
  }

  /*----------------------*/
  point_r3_t& point_r3_t::operator-=(const point_r3_t &pt) {
     x = (x-pt.x);
     y = (y-pt.y);
     z = (z-pt.z);
     return *this;
  }

  /*----------------------*/
  point_r3_t point_r3_t::operator-(const point_r3_t &pt) {
     return point_r3_t ((x-pt.x), (y-pt.y), (z-pt.z));
  }

  /*----------------------*/
  point_r3_t point_r3_t::operator+(const point_r3_t &pt) {
     return point_r3_t ((x+pt.x), (y+pt.y), (z+pt.z));
  }

  /*----------------------*/
  point_r3_t point_r3_t::operator/(float a) {
     return point_r3_t ((x/a), (y/a), (z/a));
  }

  /*----------------------*/
  point_r3_t point_r3_t::operator*(float a) {
     return point_r3_t ((x*a), (y*a), (z*a));
  }


  /*----------------------*/
  /*bool point_r3_t::operator<(const point_r3_t &pt) const {
     return z < pt.z;
  }*/

  /*----------------------*/
  /*bool point_r3_t::operator>(const point_r3_t &pt) const {
     return z > pt.z;
  }*/

  /*----------------------*/
  bool point_r3_t::operator==(const point_r3_t &pt) const {
     return distTo(pt) < 0.005;
  }

  /*bool operator==(const point_r3_t& pa, const point_r3_t& pb) {
     return (pa.x==pb.x && pa.y==pb.y);
  }*/

  /*----------------------*/
  /*bool point_r3_t::operator==(const point_r3_t &pa, const point_r3_t &pb) const {
     return (pa.x == pb.x && pa.y == pb.y);
  }*/

  /*----------------------*/
  bool point_r3_t::operator!=(const point_r3_t &pt) const {
     return distTo(pt) > 0.005;
  }

  /*----------------------*/
  float point_r3_t::normalize() const {
     return sqrt(x*x+y*y+z*z);
  }

  /*----------------------*/
  string point_r3_t::getLabel() const {
     stringstream ss;
     ss << x << "|" << y << "|" << z;
     return ss.str();
  }

  /*----------------------*/
  bool operator>(const point_r3_t& pa, const point_r3_t& pb) {
     return( (pa.y > pb.y) || ( (pa.y==pb.y) && (pa.x < pb.x)) );
  }

  /*----------------------*/
  bool operator<(const point_r3_t& pa, const point_r3_t& pb) {
     return( (pa.y < pb.y) || ( (pa.y==pb.y) && (pa.x > pb.x)) );
  }

  /*----------------------*/
  std::ostream& operator<<(std::ostream& os, const point_r3_t& pt) {
     os << "x: " << pt.x << "; y: " << pt.y << "; z: " << pt.z;
     return os;
  }

