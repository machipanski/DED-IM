#include "line_t.h"

/*----------------------*/
line_t::line_t (void) {
   v[0] = point_r3_t();
   v[1] = point_r3_t();
   index = 0;
   vertical = false; 
}

/*----------------------*/
line_t::line_t (point_r3_t p0, point_r3_t p1, int i = 0) {
   v[0] = p0;
   v[1] = p1;
   index = i;
   vertical = false;
   if ((v[1].x - v[0].x) != 0) {
      a = (v[1].y - v[0].y)/(v[1].x - v[0].x);
      b = (v[0].y - (a * v[0].x));
   }
   else {
      vertical = true;
   }
}

/*----------------------*/
line_t::line_t (point_r3_t* sp, point_r3_t* ep, Type type, long int l_id):_type(type) {
   _endp[0] = sp;
   _endp[1] = ep;
   _id = l_id;
}

/*----------------------*/
bool line_t::operator==(const line_t &ls) const {
  return ((v[0] == ls.v[0]) && (v[1] == ls.v[1]));
}

/*----------------------*/
ostream& operator<<(ostream& os, const line_t &ls) {
  os << "V0: (" << ls.v[0] << "); V1: (" << ls.v[1] << ")";
  return os;
}

/*----------------------*/
void line_t::reverse () {
  assert(_type==INSERT);
  point_r3_t* tmp = _endp[0];
  _endp[0] = _endp[1];
  _endp[1] = tmp;
}

/*----------------------*/
void line_t::setKeyValue (double y) {
  if (_endp[1]->y == _endp[0]->y)
    _key = _endp[0]->x < _endp[1]->x ? _endp[0]->x : _endp[1]->x;
  else    
    _key = (y - _endp[0]->y) * (_endp[1]->x - _endp[0]->x)/(_endp[1]->y - _endp[0]->y) + _endp[0]->x;
}

/*----------------------*/
void line_t::setKey (double y) {
  _key = y;
}

