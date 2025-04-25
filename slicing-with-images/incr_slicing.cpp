#include "incr_slicing.h"

/* Assumes that {P} is a list of {k} strictly increasing {Z} coordinates. 
   Returns an integer {p} such that {P[p-1] < zMin < P[p]}. As special cases, 
   if {zMin < P[0]} returns 0; if {zMin > P[k-1]} returns {k}. */
int IncrementalSlicing_binary_search (float zMin, vector<float> P) {
  int k = P.size();
  assert(k >= 1);
  if (zMin >= P[k-1]) { return k; }
  /* Binary search: */
  int l = -1; /* Inferior Z index. */
  int r = k;  /* Superior Z index. */
  while (r - l > 1) {
    /* At this point, {zMin} is between {P[l]} and {P[r]}. */
    int m = (l + r)/2;
    assert((0 <= m) && (m < k));
    if (zMin >= P[m]) {
      l = m;
    } else {
      r = m;
    }
  }
  return r;
}

/*-----------------------------------------------------------------------*/
Mesh_triangle_t_List_t* Mesh_triangle_t_List_create (void) {
  Mesh_triangle_t_List_t *L = (Mesh_triangle_t_List_t *)malloc(sizeof(Mesh_triangle_t_List_t));
  L->head = NULL;
  L->tail = NULL;
  return L;
}

/*-----------------------------------------------------------------------*/
void Mesh_triangle_t_List_insert (triangle_t t, Mesh_triangle_t_List_t *L) {
  Mesh_triangle_t_Node_t *node = (Mesh_triangle_t_Node_t *)malloc(sizeof(Mesh_triangle_t_Node_t));
  node->t = t;
  node->next = L->head;
  node->prev = NULL;
  if (L->head == NULL) {
    /*New head*/
    L->head = L->tail = node;
  }
  else {
    L->head->prev = node;
    L->head = node;
  }
}

/*-----------------------------------------------------------------------*/
void Mesh_triangle_t_List_union (Mesh_triangle_t_List_t *L1, Mesh_triangle_t_List_t *L2) {
  if ( (L1->head != NULL) && (L2->head != NULL) ) {
    L1->tail->next = L2->head;
    L2->head->prev = L1->tail;
    L1->tail = L2->tail;;
  }
  else if (L2->head != NULL) {
    L1->head = L2->head;
    L1->tail = L2->tail;
  }
}

/*-----------------------------------------------------------------------*/
Mesh_triangle_t_Node_t* Mesh_triangle_t_List_remove (Mesh_triangle_t_List_t *L, Mesh_triangle_t_Node_t *node) {
  if ((node->prev == NULL) && (node->next == NULL)) {
    free (node);
    L->head = NULL;
    L->tail = NULL;
    return NULL;
  }
  else if (node->prev == NULL) {
    node->next->prev = NULL;
    L->head = node->next;
    free (node);
    return L->head;
  }
  else if (node->next == NULL) {
    node->prev->next = NULL;
    L->tail = node->prev;
    free (node);
    return NULL;
  }
  else {
    Mesh_triangle_t_Node_t *next = node->next;
    node->next->prev = node->prev;
    node->prev->next = next;
    free (node);
    return next;
  }
}

/*-----------------------------------------------------------------------*/
point_r3_t R3_Mesh_Side_slice (point_r3_t vi, point_r3_t vj, float Z) {
   double dx = vj.x - vi.x;
   double dy = vj.y - vi.y;
   double dz = vj.z - vi.z;
   assert(dz != 0);
   double frac = (Z - vi.z)/dz;
   float xint = (float)(frac*dx + (double)vi.x);
   float yint = (float)(frac*dy + (double)vi.y);
   point_r3_t out;
   out.x = xint;
   out.y = yint;
   out.z = Z;
   return out;
   //return (point_r3_t){ .x = xint, .y = yint, .z = Z };
}

/*-----------------------------------------------------------------------*/
line_t R3_Mesh_triangle_t_slice (Mesh_triangle_t_Node_t *t, float Z) {
   assert((t->t.zMin < Z) && (t->t.zMax > Z));
   int np = 0; /* Number of segment endpoints found */
   line_t seg;
   for (int i = 0; i < 3; i++) {
      /* Get side {i} of triangle: */
      int j = (i == 2 ? 0 : i+1);
      point_r3_t vi = (t->t.v[i]);
      point_r3_t vj = (t->t.v[j]);
      /* Check for intersection of plane with {vi--vj}. */
      /* Must consider segment closed at bottom and open at top in case {Z} goes through a vertex. */
      float vzMin = (vi.z < vj.z ? vi.z : vj.z);
      float vzMax = (vi.z > vj.z ? vi.z : vj.z);
      if ((vzMin <= Z) && (vzMax > Z)) {
         point_r3_t p = R3_Mesh_Side_slice (vi, vj, Z);
         assert(np < 2);
         seg.v[np] = p;
         np++;
      }
   }
   seg.tangle = t->t.tangle;
   assert(np == 2);
   return seg; 
}

/*-----------------------------------------------------------------------*/
line_t R3_Mesh_triangle_t_slice (triangle_t t, float Z) {
   assert((t.zMin < Z) && (t.zMax > Z));
   int np = 0; /* Number of segment endpoints found */
   line_t seg;
   for (int i = 0; i < 3; i++) {
      /* Get side {i} of triangle: */
      int j = (i == 2 ? 0 : i+1);
      point_r3_t vi = (t.v[i]);
      point_r3_t vj = (t.v[j]);
      /* Check for intersection of plane with {vi--vj}. */
      /* Must consider segment closed at bottom and open at top in case {Z} goes through a vertex. */
      float vzMin = (vi.z < vj.z ? vi.z : vj.z);
      float vzMax = (vi.z > vj.z ? vi.z : vj.z);
      if ((vzMin <= Z) && (vzMax > Z)) {
         point_r3_t p = R3_Mesh_Side_slice (vi, vj, Z);
         assert(np < 2);
         seg.v[np] = p;
         np++;
      }
   }
   assert(np == 2);
   return seg; 
}

/*----------------------------------------------------------------------*/
/* Assumes that {P[0..k-1]} is a list of {k} strictly increasing {Z} 
  coordinates. Returns a vector of {k+1} lists {L[0..k]} such that {L[p]} 
  contains all triangles of the {mesh} that have {zMin} between {P[p-1]} 
  and {P[p]}, assuming that {P[-1] = -oo} and {P[k] = +oo}. If {delta > 0}, 
  assumes that {P[p]-P[p-1] = delta} for {p} in {1..k-1}. If {srt} is true, 
  assumes that the triangles are already sorted by increasing {zMin}. */
Mesh_triangle_t_List_t** IncrementalSlicing_buildLists (bool srt, double delta, const mesh_t *mesh, vector<float> P) {

  int k = P.size(); /* Number of planes. */

  Mesh_triangle_t_List_t **L = (Mesh_triangle_t_List_t **)malloc((k+1) * sizeof(Mesh_triangle_t_List_t *));

  for (size_t p = 0; p <= k; p++) { L[p] = Mesh_triangle_t_List_create(); }

  const vector<triangle_t> &T = mesh->getvTriangle();

  int n = T.size(); /* Number of triangles. */

  if (delta > 0.0) {
    /* Uniform slicing - compute list index: */
    for (auto it = T.begin(), itEnd = T.end(); it != itEnd; ++it) {
      triangle_t t = *it;
      int p;
      if (t.zMin < P[0]) {
        p = 0;
      }
      else if (t.zMin > P[k-1]) {
        p = k;
      }
      else {
        p = floor((t.zMin - P[0])/delta) + 1;
      }
      Mesh_triangle_t_List_insert (t, L[p]);
    }
  } else if (srt) {
    /* Slicing of a pre-sorted mesh - merge {zMin}s and {P}: */
    auto it = T.begin();
    auto itEnd = T.end();
    double zprev = -INFINITY;
    for (int p = 0; p <= k; p++) {
      float Zp = (p < k ? P[k] : +INFINITY);
      const triangle_t t = *it;
      assert(t.zMin >= zprev);
      while ((it != itEnd) && (t.zMin < Zp)) {
        Mesh_triangle_t_List_insert (t, L[p]);
        zprev = t.zMin;
        it++;
      }
    }
  } else {
    /* General case: */
    for (auto it = T.begin(), itEnd = T.end(); it != itEnd; ++it) {
      const triangle_t t = *it;
      int p = IncrementalSlicing_binary_search(t.zMin, P);
      assert((p >= 0) && (p <= k));
      Mesh_triangle_t_List_insert (t, L[p]);
    }
  }
  return L;
}

/*-----------------------------------------------------------------------*/
/*Gets an arbitrary segment from {H}, removes it from {H} and returns it as a trivial chain. */
vector<point_r3_t> IncrementalStartLoop(vector<PointMesh> &H) {
   vector<point_r3_t> P;
   auto it = (H[0]).begin();
   point_r3_t u = (*it).first;
   vector<point_r3_t> vw = (*it).second;
   point_r3_t v = vw.at(0);
   P.push_back(u);
   P.push_back(v);
   (H[0][u]).erase(std::remove((H[0][u]).begin(), (H[0][u]).end(), v), (H[0][u]).end());
   if (H[0][u].size() == 0) { (H[0]).erase(u); }
   (H[0][v]).erase(std::remove((H[0][v]).begin(), (H[0][v]).end(), u), (H[0][v]).end());
   if (H[0][v].size() == 0) { (H[0]).erase(v); }
   return P;
}

/*-----------------------------------------------------------------------*/
/*Extends the chain {P} wih segments from {H}, removing them, while possible. */
void IncrementalExtendLoop(vector<point_r3_t> &P, vector<PointMesh> &H) { 
  int index = 0;
  int n = P.size();
  point_r3_t first = P.front();  
  point_r3_t current = P.back(); 
  point_r3_t last;
         
  /* Collect other vertices: */
  while (true) {
    auto it = (H[0]).find(current);
    if (it == (H[0]).end()) { /*Vertex {current} is a dead end:*/ break; }
    point_r3_t key1 = (*it).first; assert(key1 == current);  /*Paranoia check.*/
            
    /*Get {next}, the first unused neighbor of {current}:*/
    vector<point_r3_t> vw = (*it).second; /*Unused neighbors of {current}.*/
    assert (vw.size() != 0); 
    point_r3_t next = vw.at(0); /*First unused neighbor of {current}.*/

    /*Append the segment {(current,next)} to {P} and delete from {H}:*/
    P.push_back(next);

    /*Remove the segment {(current,next)} from {H}:*/
    (H[0][current]).erase(std::remove((H[0][current]).begin(), (H[0][current]).end(), next), (H[0][current]).end());
    if (H[0][current].size() == 0) { (H[0]).erase(current); } 
    (H[0][next]).erase(std::remove((H[0][next]).begin(), (H[0][next]).end(), current), (H[0][next]).end());
    if (H[0][next].size() == 0) { (H[0]).erase(next); } 

    if (next == first) { /*We closed a loop:*/ break; }

    /*Move on:*/
    current = next;
  }
}

/*Reverses the chain {P}.*/
void IncrementalReverseLoop(vector<point_r3_t> &P) { 
  std::reverse(P.begin(),P.end());
}

/*-----------------------------------------------------------------------
  Groups the segments {segs} into zero or more open or closed polygonal chains.
A chain with {n} segments has {n+1} vertices (of type {point_r3_t}). There may be
repeated vertices.  The chain is closed iff the last vertex is equal to the
first. Vertices have their {x} and {y} coordinates rounded to a multiple of a
certain {eps}. The output will have one open chain for every pair of vertices
of odd degree. The rest of the segments will be grouped into closed chains.
All chains are returned in the {Polygons} list. */
void ContourConstruction (vector<line_t> segs, vector<contour> polygons[], int plane, double *loopclosure_time) {

   bool verbose = false;

   clock_t contour_begin = clock();

   /*Creating the hash table.*/
   vector<PointMesh> H(1);

   /*Rounding vertices and filling the hash table.*/
   double eps = 1/128.0;
   for (std::vector<line_t>::iterator i = segs.begin(); i != segs.end(); i++) {
      line_t q = *i;
      q.v[0].x = round(q.v[0].x / eps) * eps;
      q.v[0].y = round(q.v[0].y / eps) * eps;
      q.v[0].z = plane;
      q.v[0].tangle = q.tangle;
      q.v[1].x = round(q.v[1].x / eps) * eps;
      q.v[1].y = round(q.v[1].y / eps) * eps;
      q.v[1].z = plane;
      q.v[1].tangle = q.tangle;
      //printf ("Triangle inclination: %f %f %f\n", q.tangle, q.v[0].tangle, q.v[1].tangle);
      if (q.v[0].distTo(q.v[1]) > 0.0001) {
         (H[0][q.v[0]]).push_back(q.v[1]);
         (H[0][q.v[1]]).push_back(q.v[0]);
      }
   }

   /* Count vertices by degree: */
   if (verbose) {
     int degmax = 10;
     int ctdeg[degmax+1];
     for (int deg = 0; deg <= degmax; deg++) { ctdeg[deg] = 0; }
     for (auto i = (H[0]).begin(); i != (H[0]).end(); i++) {
        vector<point_r3_t> L = (*i).second;
        int deg = L.size();
        if (deg > degmax) { deg = degmax; }
        ctdeg[deg]++;
     }
     assert(ctdeg[0] == 0);
     bool closedSlice = true;
     for (int deg = 1; deg <= degmax; deg++) { 
       if (((deg % 2) != 0) && (ctdeg[deg] > 0)) { closedSlice = false; }
       if ((verbose || (deg != 2)) && (ctdeg[deg] != 0))
         { cout << "there are " << ctdeg[deg] << " vertices of degree " << deg << " on plane " << plane << endl; }
     }
     if (!closedSlice) { cout << "** contours of plane " << plane << " are not closed" << endl; }
   }

   /*Contour construction.*/
   bool maximal = true;
   while (!(H[0]).empty()) {
     if (maximal) {
       vector<point_r3_t> P = IncrementalStartLoop(H);
       IncrementalExtendLoop(P,H);
       if (P.front() != P.back()) { //Chain {P} is open
         IncrementalReverseLoop(P);
         IncrementalExtendLoop(P,H);
       }
       polygons[plane].push_back({false, false, P});
     }
     else {
       vector<point_r3_t> P = IncrementalStartLoop(H);
       IncrementalExtendLoop(P,H);
       polygons[plane].push_back({false, false, P});
     }
   }
   clock_t contour_end = clock();
   *loopclosure_time += double(contour_end - contour_begin)/CLOCKS_PER_SEC;
}

/*-----------------------------------------------------------------------*/
void IncrementalSlicing ( const mesh_t *mesh, vector<float> P, float delta, bool srt, vector<contour> polygons[], 
                          bool chaining, bool orienting, long *intersections, double *slicing_time, double *loopclosure_time) {

  /*Slicing*/
  clock_t slice_begin = clock();

  int k = P.size();

  vector<line_t> segs[k];

  /* Classify triangles by the plane gaps that contain their {zMin}: */
  Mesh_triangle_t_List_t **L = IncrementalSlicing_buildLists (srt, delta, mesh, P);
  /* Now perform a plane sweep from bottom to top: */

  Mesh_triangle_t_List_t *A = Mesh_triangle_t_List_create(); /* Active triangle list. */
  for (int p = 0; p < k; p++) {
    /* Add triangles that start between {P[p-1]} and {P[p]}: */
    Mesh_triangle_t_List_union (A, L[p]);
    /* Scan the active triangles: */
    Mesh_triangle_t_Node_t *aux = A->head;
    while (aux != NULL) {
      Mesh_triangle_t_Node_t *next = aux->next;
      if (aux->t.zMax < P[p]) {
        /* triangle_t is exhausted: */
        Mesh_triangle_t_List_remove (A, aux);
      } else {
        /* Compute intersection: */
        if ((aux->t.zMin < P[p]) && (aux->t.zMax > P[p])) {
          line_t seg = R3_Mesh_triangle_t_slice (aux, P[p]);
          segs[p].push_back(seg);
          (*intersections)++;
        }
      }
      aux = next;
    }
  }
  free(L);
  clock_t slice_end = clock();
  *slicing_time = double(slice_end - slice_begin)/CLOCKS_PER_SEC;
  /*End-Slicing*/

  if (chaining) {
    /*Contour construction:*/
    for (size_t p = 0; p < k; p++) {
      if (!segs[p].empty()) {
         ContourConstruction (segs[p], polygons, p, loopclosure_time);
         #ifdef DEBUG
           char fname[256];
           sprintf(fname, "slice_%03d.txt", (int)p);
           FILE *fslice = fopen(fname, "w");
           fprintf(fslice, "----------- Segmentos ---------------\n");
           for (int ii = 0; ii < segs[p].size(); ii++) {
              line_t ss = segs[p].at(ii); 
              fprintf(fslice, "%f %f   %f %f\n", ss.v[0].x, ss.v[0].y, ss.v[1].x, ss.v[1].y);           
           }
           fprintf(fslice, "---------- End Segmentos --------------\n");
           record_polygons (polygons[p], fslice);
           fclose(fslice);
         #endif
         if (orienting) {   
           ray_casting (polygons[p]);
         }
         segs[p].clear(); 
      }
    }
    /*End construction.*/
  }
  else {
    export_svg_no_chaining ("segments.svg", segs, k, mesh->meshAABBSize());
  }
}

