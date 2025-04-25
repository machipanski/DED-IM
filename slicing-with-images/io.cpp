#include "io.h"
#include <math.h>

/*-----------------------------------------------------------------------*/
/*Rounds x to an integer multiple of {eps}. If {mod} > 1,
  the factor will be congruent to {rem} modulo {mod}. */
float xround (float x, double eps, int mod, int rem) {
  double y = round((double)x/(mod * eps));
  double z = (y * mod + rem) * eps;
  return (float)z;
}

/*-----------------------------------------------------------------------*/
/*Rounds {x,y,z} to an even multiple of {eps}. */
point_r3_t point_r3_t_round (float x, float y, float z, double eps) {
   point_r3_t p;
   p.x = xround(x, eps, 2, 0);
   p.y = xround(y, eps, 2, 0);
   p.z = xround(z, eps, 2, 0);
   return p;
}

/*-----------------------------------------------------------------------*/
/*Returns true {t} has two or more coicident vertices.*/
bool degenerate (triangle_t t) {
   if (t.v[0].distTo(t.v[1]) < 0.000001) { return true; }
   if (t.v[1].distTo(t.v[2]) < 0.000001) { return true; }
   if (t.v[2].distTo(t.v[0]) < 0.000001) { return true; }
   return false;
}

/*-----------------------------------------------------------------------*/
triangle_t make_triangle ( 
   float n0, float n1, float n2,
   float f0, float f1, float f2,
   float f3, float f4, float f5,
   float f6, float f7, float f8,
   const char *rotate, double eps, double tangle
 ) {

    if (strcmp(rotate,"true") == 0) {
      return triangle_t (point_r3_t(n0, n1, n2), point_r3_t_round(f0, f2, f1, eps), point_r3_t_round(f3, f5, f4, eps), point_r3_t_round(f6, f8, f7, eps), tangle);
    }
    else {
      return triangle_t (point_r3_t(n0, n1, n2), point_r3_t_round(f0, f1, f2, eps), point_r3_t_round(f3, f4, f5, eps), point_r3_t_round(f6, f7, f8, eps), tangle);
    }
}

/*-----------------------------------------------------------------------*/
/* Read the given STL file name (ascii or binary is set using ‘isBinaryFormat’)
   and generate a triangle_t Mesh object in output parameter ‘mesh’. */
int stlToMeshInMemory (const char *stlFile, mesh_t *mesh, bool isBinaryFormat, const char *rotate, double eps) {

  int ndegenerated = 0;

  if (!isBinaryFormat) {

    ifstream in(stlFile);

    if (!in.good()) {
      return 1;
    }

    std::string s0, s1;

    float n0, n1, n2;
    float f0, f1, f2;
    float f3, f4, f5;
    float f6, f7, f8;

    while (!in.eof()) {
      in >> s0; /*s0 can be facet or solid!*/
      if (s0 == "facet") {
        in >> s1 >> n0 >> n1 >> n2; /* normal x y z. */
        in >> s0 >> s1;             /* loop. */
        in >> s0 >> f0 >> f1 >> f2; /* vertex x y z. */
        in >> s0 >> f3 >> f4 >> f5; /* vertex x y z. */
        in >> s0 >> f6 >> f7 >> f8; /* vertex x y z. */
        in >> s0;                   /* endloop. */
        in >> s0;                   /* endfacet.*/


        double magnitude_n = sqrt((n0 * n0) + (n1 * n1) + (n2 * n2));

	double n0_n = n0/magnitude_n;	
	double n1_n = n1/magnitude_n;	
	double n2_n = n2/magnitude_n;	

        double vector_r[3] = {0.0, 1.0 ,0.0};
        double dot_product = n0_n * vector_r[0] + n1_n * vector_r[1] + n2_n * vector_r[2];
        double tangle = acos(dot_product);

        triangle_t t = make_triangle (n0, n1, n2, f0, f2, f1, f3, f5, f4, f6, f8, f7, rotate, eps, tangle);
        if (!degenerate(t)) {
           mesh->push_back (t);
        }
        else {
          ndegenerated++;
        }
      }
      else if (s0 == "endsolid") {
         break;
      }
    }
    in.close();
  }
  else {
    FILE *f = fopen (stlFile, "rb");
    if (!f) {
      return 1;
    }
    char title[80];
    int nFaces;
    int err;
    err = fread (title, 80, 1, f);
    err = fread ((void*)&nFaces, 4, 1, f);
    float v[12]; /* normal = 3, vertices = 9 (12) */
    unsigned short uint16;
    /* Every Face is 50 Bytes: Normal(3*float), Vertices(9*float), 2 Bytes Spacer */
    for (size_t i=0; i<nFaces; ++i) {
      for (size_t j=0; j<12; ++j) {
        err = fread((void*)&v[j], sizeof(float), 1, f);
      }
      err = fread((void*)&uint16, sizeof(unsigned short), 1, f); // spacer between successive faces

      double magnitude_n = sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]));

      double n0_n = v[0]/magnitude_n;	
      double n1_n = v[1]/magnitude_n;	
      double n2_n = v[2]/magnitude_n;	

      double vector_r[3] = {0.0, 1.0 ,0.0};
      double dot_product = n0_n * vector_r[0] + n1_n * vector_r[1] + n2_n * vector_r[2];
      double tangle = acos(dot_product);

      triangle_t t = make_triangle (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], rotate, eps, tangle);
      if (!degenerate(t)) {
         mesh->push_back (t);
      }
      else {
        ndegenerated++;
      }
    }
    fclose(f);
  }
  cout << "number of degenerated triangles = " << ndegenerated << endl;
  return 0;
}

/*-----------------------------------------------------------------------*/
int checkASCIIFile (const char *fileName) {
  string line1, line2;
  ifstream input(fileName);
  if (!input.good()) {
    return -1;
  }
  getline(input, line1);
  getline(input, line2);
  if (line1.find("solid")!=string::npos && line2.find("facet")!=string::npos) {
    return FILE_STL_ASCII;
  }
  if (line1.find("xml")!=string::npos && line2.find("amf")!=string::npos) {
    return FILE_AMF_ASCII;
  }
  return FILE_STL_BIN;
}

/*-----------------------------------------------------------------------*/
void record_polygons (vector<contour> polygons, FILE *file) {
  for (int i = 0; i < polygons.size(); i++) {
    vector<point_r3_t> Pi = polygons.at(i).P;
    int size = (int)(Pi.size()) - 1;
    fprintf(file, "%d\n", size);
    for (int p = 0; p < size; p++) {
       point_r3_t v = Pi.at(p);
       fprintf(file, "%f %f\n", v.x, v.y);
    }
  }
}

