#include "planes.h"

/*-----------------------------------------------------------------------*/
/*Compute uniform and adaptive z-plane coordinates!*/
vector<float> compute_planes (const mesh_t *mesh, float max_thickness, char *adaptive, double eps, float *delta) {

  bool rounding = true; /*To avoid that the Z-coordinates of all planes are distinct from the Z-coordinates of all vertices.*/

  /* Vector to keep the plane coordinates: */
  vector<float> Planes;

  /* Assuming the model as a 3D axis-aligned bounding-box: */
  double model_zmax = std::max(mesh->getUpperRightVertex().z, mesh->meshAABBSize().z);

  double model_zmin = mesh->getBottomLeftVertex().z;

  if (strcmp(adaptive, "false") == 0)  { /*Uniform slicing: */

    double spacing = (rounding ? xround (max_thickness, eps, 2, 0) : max_thickness); /*Plane spacing even multiple of {eps}*/

    double P0 = xround (model_zmin - spacing, eps, 2, 1); /*First plane odd multiple of {eps}.*/

    int no_planes = 1 + (int)((model_zmax + spacing - P0)/spacing); /* Number of planes: */

    cout << "eps = " << eps << endl;
    cout << "max thickness = " << max_thickness << endl;
    cout << "rounded plane spacing spacing = " << spacing << endl;
    cout << "model zmin = " << model_zmin << ", model zmax = " << model_zmax << ", first plane Z = " << P0 << ", number of planes = " << no_planes << endl;

    for (size_t i = 0; i < no_planes; i++) {
      /* Building the vector with the slice z coordinates: */
      float Pi = (float)(P0 + i * spacing);
      if ((Pi > model_zmin) && (Pi < model_zmax)) {
          Planes.push_back ((float)(P0 + i * spacing));
      }
    }
    *delta = (float)(spacing);
  }
  else { /*Adaptive slicing z-planes: */

    float zplane = 0.0;
    float min_thickness = 0.016;
    Planes.push_back (model_zmin + zplane);

    while ((model_zmin + zplane) <= model_zmax) {
      double vrandom = min_thickness + (max_thickness - min_thickness) * (rand() / (double)RAND_MAX);
      double coordinate = xround (model_zmin + zplane + vrandom, eps, 2, 1);
      if (coordinate >= model_zmax) { break; }
      Planes.push_back (coordinate);
      zplane += vrandom;
    }
  }
  return Planes;
}

