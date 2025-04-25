#ifndef _line_t_h_
#define _line_t_h_

#include <array>
#include <math.h>
#include <iostream>

#include "point_r3_t.h"
#include "triangle_t.h"
#include "mesh_t.h"
#include "line_t.h"

using namespace std;

class line_t {

  public:

    line_t (void);

    line_t (point_r3_t p0, point_r3_t p1, int i);

    line_t (point_r3_t *ep1, point_r3_t *ep2, Type type, long int l_id);

    bool operator==(const line_t &ls) const;

    friend ostream& operator<<(ostream &os, const line_t &ls);

      //constructors and destructor       
      //Linebase();
      //Linebase(Pointbase* ep1, Pointbase* ep2, Type type);
      //Linebase(point_r3_t* ep1, point_r3_t* ep2, Type type, long int l_id);
      //Linebase (const Linebase& line);
      //~Linebase() {};

      unsigned int id() const { return _id; }
      point_r3_t* endPoint (int i) const { return _endp[i]; }
      Type type() const { return _type; }
      double keyValue() const { return _key; }
      void setKeyValue (double y);
      void setKey (double y);
      void increaseKeyValue (const double diff) { _key += diff; }
      void reverse();
      void setHelper (unsigned int i) { _helper = i; }
      unsigned int helper() { return _helper; }

      //operator overloading             
      //friend ostream &operator<<(ostream &os, const Linebase& line);

  public:

    point_r3_t v[2];
    double a;
    double b;
    bool vertical;
    int index;
    double tangle;

    unsigned int _id;     /*Id of a line segment.*/
    point_r3_t* _endp[2]; /*Two end points.*/
    Type _type;           /*Type of a line segement, input/insert.*/
    double _key;          /*Key of a line segment for splay tree searching.*/
    unsigned int _helper; /*Helper of a line segment.*/
};

#endif
