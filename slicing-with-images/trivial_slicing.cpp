#include "trivial_slicing.h"

/*-----------------------------------------------------------------------*/
point_r3_t TrivialClosure_find_B (vector<line_t>& segs, point_r3_t u, int plane) {
  double dist = +INFINITY;
  point_r3_t sp;
  int piter = 1, pitem = -1;
  for (vector<line_t>::iterator j = segs.begin(); j != segs.end(); j++) {
    point_r3_t p0 = (*j).v[0];
    point_r3_t p1 = (*j).v[1];
    if ((p0.x == u.x) && (p0.y == u.y)) {
      segs.erase(j);
      return p1;
    } else if ((p1.x == u.x) && (p1.y == u.y)) {
      segs.erase(j);
      return p0;
    } else {
      double d0 = sqrt ((p0.x - u.x) * (p0.x - u.x) + (p0.y - u.y) * (p0.y - u.y));
      double d1 = sqrt ((p1.x - u.x) * (p1.x - u.x) + (p1.y - u.y) * (p1.y - u.y));
      if (d0 < dist) { dist = d0; sp = p0; pitem = piter; }
      if (d1 < dist) { dist = d1; sp = p1; pitem = piter; }
    }
    piter++;
  }
  if ((pitem != -1) && (dist <= 1)) {
    segs.erase(segs.begin() + pitem);
    return sp;
  }
  else {
    cout << "Distance: " << dist << ", plane : "<< plane << endl; 
    return {+INFINITY,+INFINITY,+INFINITY};
  }
}

/*-----------------------------------------------------------------------*/
point_r3_t TrivialClosure_find_A (vector<line_t>& segs, point_r3_t u, int plane) {
  for (vector<line_t>::iterator j = segs.begin(); j != segs.end(); ++j) {
    point_r3_t p0 = (*j).v[0];
    point_r3_t p1 = (*j).v[1];
    if ((p0.x == u.x) && (p0.y == u.y)) {
      segs.erase(j);
      return p1;
    } else if ((p1.x == u.x) && (p1.y == u.y)) {
      segs.erase(j);
      return p0;
    }
  }
  return {+INFINITY,+INFINITY,+INFINITY};
}

/*-----------------------------------------------------------------------*/
void TrivialLoopClosure (vector<line_t> segs, vector<contour> polygons[], int plane) {

  while (segs.size() > 0) {

    /* Get another contour: */
    vector<point_r3_t> P;

    /* Get the first segment: */
    point_r3_t last;
    point_r3_t current;
    point_r3_t prev;
    {
      std::vector<line_t>::iterator i = segs.begin();
      last = (*i).v[0];
      current = (*i).v[1];
      {
         P.push_back (last);
      }
      prev = last;
      segs.erase(i);
    }

    /* Get additional segments until loop is closed: */
    bool open = false;
    do {
      /* Find and delete another segment with endpoint {current}, advance {prev,current} */
      point_r3_t next = TrivialClosure_find_A (segs, current, plane);
      //point_r3_t next = TrivialClosure_find_B (segs, current, plane);
      if ((next.x == +INFINITY) || (next.y == +INFINITY)) {
        /* Open contour?! */
        open = true;
        break;
      }
      {
         P.push_back (current);
      }
      prev = current;
      current = next;
    } while ((current.x != last.x) || (current.y != last.y));
    if (!open) {
      P.push_back(last);
    } 
    polygons[plane].push_back({false, false, P});
  }
}

/*-----------------------------------------------------------------------*/
void TrivialSlicing (const mesh_t *mesh, vector<float> P, vector<contour> polygons[], bool chaining, bool orienting, long *intersections, double *slicing_time, double *loopclosure_time) {

  /*Slicing*/
  clock_t slice_begin = clock();

  int k = P.size(); /* Number of planes. */

  const vector<triangle_t> &T = mesh->getvTriangle();

  vector<line_t> segs[k];

  /* Enumerate the slicing planes: */
  for (int p = 0; p < k; p++) {
    /* Enumerate all triangles of the mesh:*/
    int intersections_per_plane = 0;
    for (auto it = T.begin(), itEnd = T.end(); it != itEnd; ++it) {
      triangle_t t = *it; /*Current triangle.*/
      /*Does the plane intersect the triangle?:*/
      if ((t.zMin < P[p]) && (t.zMax > P[p])) {
        /* Compute and save the intersection: */
        line_t seg = R3_Mesh_triangle_t_slice (t, P[p]);
        seg.v[0].x = round(seg.v[0].x * 100.0) / 100.0;
        seg.v[0].y = round(seg.v[0].y * 100.0) / 100.0;
        seg.v[1].x = round(seg.v[1].x * 100.0) / 100.0;
        seg.v[1].y = round(seg.v[1].y * 100.0) / 100.0;
        if (seg.v[0].distTo(seg.v[1]) > 0.0001) {
           segs[p].push_back(seg);
        }
        (*intersections)++;
        intersections_per_plane++;
      }
    }
  }
  clock_t slice_end = clock();
  *slicing_time = double(slice_end - slice_begin)/CLOCKS_PER_SEC;
  /*End-Slicing*/

  if (chaining) {
    /*Loop-Closure:*/
    clock_t contour_begin = clock();
    for (size_t p = 0; p < k; p++) {
      if (!segs[p].empty()) {
        TrivialLoopClosure (segs[p], polygons, p);
        if (orienting) {
           ray_casting (polygons[p]);
        }
        #ifdef DEBUG
           char fname[256];
           sprintf(fname, "slice_%03d.txt", (int)p);
           FILE *fslice = fopen(fname, "w");
           record_polygons (polygons[p], fslice);
           fclose(fslice);
         #endif
        segs[p].clear();
      }
    }
    clock_t contour_end = clock();
    *loopclosure_time = double(contour_end - contour_begin)/CLOCKS_PER_SEC;
    /*End-Loop-Closure*/
  }
  else {
    export_svg_no_chaining ("segments.svg", segs, k, mesh->meshAABBSize());
  }
}


