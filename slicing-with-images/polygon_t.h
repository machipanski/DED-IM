#ifndef _polygon_t_h_
#define _polygon_t_h_

#include <array>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <list>
#include <queue>
#include <stack>
#include <set>
#include "point_r3_t.h"
#include "line_t.h"
#include "contour.h"
#include "avl_t.h"

#define sqr(t) (t)*(t)

using namespace std;

extern double orient2d (double* pa, double* pb, double* pc);

template <class T, class KeyType>      class    SplayTree;
typedef map<unsigned int, point_r3_t*>          PointbaseMap;
typedef map<unsigned int, line_t*>              LineMap;
typedef priority_queue<point_r3_t>              PQueue;
typedef SplayTree<line_t*, double>              EdgeBST;
typedef list<unsigned int>                      Monopoly;
typedef list<Monopoly>                          Monopolys;
typedef vector<unsigned int>                    Triangle;
typedef list<Triangle>                          Triangles;
typedef map<unsigned int, set<unsigned int> >   AdjEdgeMap;

class Polygon
{
   public:
      Polygon (vector<contour> P);
      void triangulation();
      void searchMonotones();
      bool is_exist (double, double);
      bool read_bdm (vector<contour> P);
      Triangles triangles() { return _triangles; }
      PointbaseMap& points() { return _points; }
      LineMap& edges() { return _edges; }
      bool getstop () { return stop; }

   private:
      PQueue _qpoints;                          //priority queue for event points
      Monopolys _mpolys;                        //all monotone polygon piece list;
      unsigned int _ncontours;                  //number of contours
      LineMap _edges;                           //all edges]
      PointbaseMap _points;                     //all vertices
      vector<unsigned int> _nVertices;          //
      bool stop;
      bool _debug;                              //debug option;
      fstream _logfile;                         //log file for debug purpose;
      Triangles _triangles;                     //all triangle list;
      double _xmin, _xmax, _ymin, _ymax;        //boundary box for polygon


      //Triangles   _triangles;      /*All triangle list*/

      AdjEdgeMap  _startAdjEdgeMap;                    //all edges starting from given points (map)   
      LineMap     _diagonals;                          //added diagonals to partition polygon to 
                                                         //monotont pieces, not all diagonals of 
                                                         //given polygon


      void handleStartVertex (unsigned int i, avl_t **t);
      void handleEndVertex (unsigned int, avl_t **t);
      void handleSplitVertex (unsigned int, avl_t **t);
      void handleMergeVertex (unsigned int, avl_t **t);
      void handleRegularVertexUp (unsigned int, avl_t **t);
      void handleRegularVertexDown (unsigned int, avl_t **t);
      void addDiagonal (unsigned int i, unsigned int j);
      void partition2Monotone ();
      unsigned int prev (unsigned int i);
      unsigned int next (unsigned int i);
      void triangulateMonotone (Monopoly& mpoly);
      double angleCosb (double *A, double *B, double *C);
      unsigned int selectNextEdge (line_t* edge);
      void initializate();
};

#endif
