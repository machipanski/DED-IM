#include "contour.h"

/*-----------------------------------------------------------------------*/
/*Function to (re)orient the contour clockwise and counter-clockwise.*/
void ray_casting (vector<contour> &polygons) {

  vector<line_t> segments;

  bounding_box bb = create_bounding_box ();

  /*Creating the line segments of each contour: */
  for (int i = 0; i < polygons.size(); i++) {
    double area = 0.0;
    vector<point_r3_t> Pi = polygons.at(i).P;
    for (int j = 1; j < Pi.size(); j++) {
      point_r3_t p0 = Pi.at(j-1);
      point_r3_t p1 = Pi.at(j+0);
      area += (p0.x * p1.y - p0.y * p1.x);
      add_point (p0, p1, segments, &bb, (j == 1 ? true : false), i);
      if (j == Pi.size()-1) {
         add_point (p1, Pi.at(0), segments, &bb, (j == 1 ? true : false), i);
         area += (p1.x * Pi.at(0).y - p1.y * Pi.at(0).x);
      }
    }
    area /= 2.0;
    if (area < 0.0) {
      polygons.at(i).clockwise = true;
    }
    else {
      polygons.at(i).clockwise = false;
    }
  }

  /*Using the point in polygon algorithm to test the first segment of each contour: */
  for (int i = 0; i < polygons.size(); i++) {
    int in = 0, out = 0;
    vector<point_r3_t> Pi = polygons.at(i).P;
    for (int k = 0; k < Pi.size(); k++) {
       bool f = contains (Pi.at(k), bb, segments, i);
       if (f) { in++; } 
       else { out++; } 
    
    }
    //printf("In: %d, out: %d\n", in, out);
    //if (contains (Pi.at(0), bb, segments, i)) {
    if (in > out) {
      /*Internal contour: */
      polygons.at(i).external = false;
    }
    else {
      /*External contour: */
      polygons.at(i).external = true;
    }

    /*Reversing contours: */
    if (polygons.at(i).external && polygons.at(i).clockwise) {
       std::reverse(polygons.at(i).P.begin(), polygons.at(i).P.end());
       polygons.at(i).clockwise = false;
    }
    else if (!polygons.at(i).external && !polygons.at(i).clockwise) {
       std::reverse(polygons.at(i).P.begin(), polygons.at(i).P.end());
       polygons.at(i).clockwise = true;
    }
  }
  segments.clear();
}

/*-----------------------------------------------------------------------*/
bool is_inside (line_t line, point_r3_t point) {
  double maxX = (line.v[0].x > line.v[1].x) ? line.v[0].x : line.v[1].x;
  double minX = (line.v[0].x < line.v[1].x) ? line.v[0].x : line.v[1].x;
  double maxY = (line.v[0].y > line.v[1].y) ? line.v[0].y : line.v[1].y;
  double minY = (line.v[0].y < line.v[1].y) ? line.v[0].y : line.v[1].y;
  if ((point.x >= minX && point.x <= maxX) && (point.y >= minY && point.y <= maxY)) {
    return true;
  }
  return false;
}

/*-----------------------------------------------------------------------*/
bool ray_intersect (line_t ray, line_t side) {
  point_r3_t intersectPoint;
  /* If both vectors aren't from the kind of x=1 lines then go into: */
  if (!ray.vertical && !side.vertical) {
    /* Check if both vectors are parallel. If they are parallel then no intersection point will exist: */
    if (ray.a - side.a == 0) {
      return false;
    }
    intersectPoint.x = ((side.b - ray.b) / (ray.a - side.a)); 
    intersectPoint.y = side.a * intersectPoint.x + side.b; 
  }
  else if (ray.vertical && !side.vertical) {
    intersectPoint.x = ray.v[0].x;
    intersectPoint.y = side.a * intersectPoint.x + side.b;
  }
  else if (!ray.vertical && side.vertical) {
    intersectPoint.x = side.v[0].x;
    intersectPoint.y = ray.a * intersectPoint.x + ray.b;
  }
  else {
    return false;
  }
  if (is_inside(side, intersectPoint) && is_inside(ray, intersectPoint)) {
    return true;
  }
  return false;
}

/*-----------------------------------------------------------------------*/
line_t create_ray (point_r3_t point, bounding_box bb, int index) {
  /* Create outside point: */
  double epsilon = (bb.xMax - bb.xMin) / 100.0;
  point_r3_t outside (bb.xMin - epsilon, bb.yMin);
  line_t v (outside, point, index);
  return v;
}

/*-----------------------------------------------------------------------*/
bounding_box create_bounding_box () {
  bounding_box bb;
  bb.xMax = std::numeric_limits<double>::min();
  bb.xMin = std::numeric_limits<double>::max();
  bb.yMax = std::numeric_limits<double>::min();
  bb.yMin = std::numeric_limits<double>::max();
  return bb;
}

/*-----------------------------------------------------------------------*/
void update_bounding_box (point_r3_t point, bounding_box *bb) {
  /* Setting the bounding box: */
  if (point.x > bb->xMax) {
    bb->xMax = point.x;
  }
  else if (point.x < bb->xMin) {
    bb->xMin = point.x;
  }
  if (point.y > bb->yMax) {
    bb->yMax = point.y;
  }
  else if (point.y < bb->yMin) {
    bb->yMin = point.y;
  }
}

/*-----------------------------------------------------------------------*/
bool insided_bounding_box (point_r3_t point, bounding_box bb) {
  if ( (point.x < bb.xMin) || (point.x > bb.xMax) || (point.y < bb.yMin) || (point.y > bb.yMax) ) {
     return false;
  }
  return true;
}

/*-----------------------------------------------------------------------*/
bool contains (point_r3_t point, bounding_box bb, vector<line_t> sides, int index) {
  if (insided_bounding_box(point, bb)) {
    line_t ray = create_ray (point, bb, index);
    int intersection = 0;
    for (int i = 0; i < sides.size(); i++) {
      if ((sides.at(i).index != index) && ray_intersect(ray, sides.at(i))) {
         intersection++;
      }
    }
    /* If the number of intersections is odd, then the point is inside the polygon: */
    if ((intersection % 2) == 1) {
      return true;
    }
  }
  return false;
}

/*-----------------------------------------------------------------------*/
void add_point (point_r3_t p1, point_r3_t p2, vector<line_t> &t, bounding_box *bb, bool first, int index) {
   if (first) {
     update_bounding_box (p1, bb);
   }
   update_bounding_box (p2, bb);
   line_t line (p1, p2, index);
   t.push_back(line);
}

