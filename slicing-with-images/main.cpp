#include "io.h"
#include "planes.h"
#include "point_r3_t.h"
#include "mesh_t.h"
#include "line_t.h"
#include "triangle_t.h"
#include "write_svg.h"
#include "contour.h"
#include "incr_slicing.h"
#include "trivial_slicing.h"
#include "image.h"
#include "polygon_t.h"

#define GLM_ENABLE_EXPERIMENTAL

using namespace std;

int main(int argc, char **argv)
{

  long intersections = 0;

  double slicing_time = 0.0;

  double loopclosure_time = 0.0;

  bool chaining = true;

  double eps = 0.004;

  bool orienting = false;

  bool write_images = false;

  bool fill_triangles = false;

  int dpi = 300;

  /*Total time:*/
  clock_t begin = clock();

  char *model;

  if (strcmp(argv[2], "-model") == 0)
  {
    model = argv[3];
  }

  float max_thickness, delta;

  if (strcmp(argv[4], "-thickness") == 0)
  {
    max_thickness = atof(argv[5]);
  }
  else
  {
    printf("Error: specify the slicing spacing in mm (thickness)!!!\n");
  }

  char *adaptive;

  if (strcmp(argv[6], "-adaptive") == 0)
  {
    adaptive = argv[7];
  }

  char *write_option = argv[8];

  char *rotate;

  if (strcmp(argv[9], "-rotate") == 0)
  {
    rotate = argv[10];
  }

  if (strcmp(argv[11], "-orienting_contours") == 0)
  {
    if (strcmp(argv[12], "true") == 0)
    {
      orienting = true;
    }
  }

  if (strcmp(argv[13], "-write_images") == 0)
  {
    if (strcmp(argv[14], "true") == 0)
    {
      write_images = true;
    }
  }

  if (strcmp(argv[15], "-dpi") == 0)
  {
    dpi = atoi(argv[16]);
  }

  if (strcmp(argv[17], "-fill_triangles") == 0)
  {
    if (strcmp(argv[18], "true") == 0)
    {
      fill_triangles = true;
    }
  }

  mesh_t mesh;

  switch (checkASCIIFile(model))
  {
  case FILE_STL_ASCII:
    if (stlToMeshInMemory(model, &mesh, false, rotate, eps) != 0)
      return 1;
    break;
  case FILE_STL_BIN:
    if (stlToMeshInMemory(model, &mesh, true, rotate, eps) != 0)
      return 1;
    break;
  default:
    cerr << "Unexpected error" << endl;
    return 1;
  }

  std::string path = model;

  std::string lastFileName;

  size_t pos = path.find_last_of("/");

  if (pos != std::string::npos)
    lastFileName.assign(path.begin() + pos + 1, path.end());
  else
    lastFileName = path;

  vector<float> P = compute_planes(&mesh, max_thickness, adaptive, eps, &delta);

  int nplanes = P.size();

  vector<contour> polygons[nplanes];

  bool srt = false;

  if (strcmp(write_option, "-No") == 0)
  {
    chaining = false;
  }

  if (strcmp(argv[1], "-Trivial") == 0)
  {
    TrivialSlicing(&mesh, P, polygons, chaining, orienting, &intersections, &slicing_time, &loopclosure_time);
  }
  else if (strcmp(argv[1], "-Incremental") == 0)
  {
    if (strcmp(adaptive, "true") == 0)
    {
      delta = 0.0;
    }
    IncrementalSlicing(&mesh, P, delta, srt, polygons, chaining, orienting, &intersections, &slicing_time, &loopclosure_time);
  }

  clock_t end = clock();

  double total_time = double(end - begin) / CLOCKS_PER_SEC;

  cout << argv[1] << ", " << lastFileName << ", thickness = " << delta;
  cout << ", #T = " << mesh.size() << ", #P = " << nplanes;
  cout << ", #S = " << intersections;
  cout << ", " << slicing_time << "s (slicing), " << loopclosure_time << "s (polygon assembling), " << total_time << "s (total)";
  cout << ", *K: " << ((double)intersections / (double)mesh.size()) << endl;

  if (write_images)
  {
    convert_polygons_to_images(polygons, nplanes, &mesh, dpi, fill_triangles);
  }

  if (chaining)
  {
    if (strcmp(write_option, "-2D") == 0)
    {
      export_svg_2d("out_2d.svg", polygons, nplanes, mesh.meshAABBSize());
    }
    else if (strcmp(write_option, "-3D") == 0)
    {
      export_svg_3d(polygons, nplanes);
    }
    else if (strcmp(write_option, "-video") == 0)
    {
      export_video(polygons, nplanes);
    }
  }

  return 0;
}
