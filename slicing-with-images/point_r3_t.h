#ifndef _point_r3_h_
#define _point_r3_h_

#include <array>
#include <math.h>
#include <sstream>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/vector_angle.hpp"



#define GLM_FORCE_RADIANS

enum Type {UNKNOWN, INPUT, INSERT, START, END, MERGE, SPLIT, REGULAR_UP, REGULAR_DOWN};

using namespace std;

class point_r3_t {

  public:

    //point_r3_t (float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
    
    point_r3_t (float _x = 0, float _y = 0, float _z = 0, double _tangle = 0) : x(_x), y(_y), z(_z), tangle(_tangle) {}
    
    //point_r3_t (float _x = 0, float _y = 0, float _z = 0, double _tangle = 0.0) : x(_x), y(_y), z(_z), tangle(_tangle) {}

    point_r3_t (int idd, float xx, float yy, Type ttype, double _tangle = 0) : id(idd), x(xx), y(yy), type(ttype), tangle(_tangle) {}
    
    float distTo (const point_r3_t &pt) const;

    array<float,3> getCoords();

    float dotproduct (const point_r3_t &v) const;

    void transform (const glm::mat4 &mat);

    point_r3_t& operator-=(const point_r3_t &pt);

    point_r3_t operator-(const point_r3_t &pt);

    point_r3_t operator+(const point_r3_t &pt);

    point_r3_t operator/(float a);

    point_r3_t operator*(float a);

    bool operator==(const point_r3_t &pt) const;

    bool operator!=(const point_r3_t &pt) const;

    float normalize() const;

    string getLabel() const;
    
    friend bool operator<(const point_r3_t &pa, const point_r3_t &pb);
    
    friend bool operator>(const point_r3_t &pa, const point_r3_t &pb);

    friend std::ostream& operator<<(std::ostream& os, const point_r3_t &pt);

  public:
    float x, y, z;   /*Point coordinates.*/
    double tangle;
    unsigned int id; /*Point id.*/
    Type type;       /*Point type. */
    bool left;       /*Left chain or not.*/
    int  label;
};

#endif
