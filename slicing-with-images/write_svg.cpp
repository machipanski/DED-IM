#include "write_svg.h"

/*-----------------------------------------------------------------------*/
/*Output a SVG file with the whole 3D model*/
void export_svg (char filename[], const vector<vector<point_r3_t> > &Polygons, int nslices, int nsegments, bool video, vector<bool> orientation) {
  glm::vec3 fromEuler (0.0f, 0.0f, 60.0f);
  glm::quat quaternion (DEG_TO_RAD(fromEuler)); 
  glm::vec3 toEuler = glm::eulerAngles(quaternion);
  float angle = glm::angle(quaternion);
  glm::vec3 axis = glm::axis(quaternion);
  glm::mat4 View = glm::rotate(glm::mat4(1.0), angle, axis);

  //float zoom = 0.6f; /*01.liver*/
  //float zoom = 0.2f; /*02.femur*/
  float zoom = 0.25f;  /*04.demon*/
  glm::mat4 Projection = glm::perspective (zoom, 4.0f / 3.0f, 0.1f, 100.f);
  int shift_x = +400;
  int shift_y = +400;
  
  glm::mat4 Model = glm::lookAt (
        glm::vec3(1, 1, 1),    /* Eye point (where am I?) */
        glm::vec3(0, 0, 0),    /* Look at Center point (What am I looking at?) */
        glm::vec3(0, 1, 0));   /* UP vector of the camera (Where is my UP vector?) */
  glm::mat4 MVP = Projection * View * Model;
  FILE *file = fopen (filename, "w");
  add_svg_information (file);

  /*Writing previous slices:*/
  int i; 
  for (i = 0; i < nslices; i++) {
    for (size_t index = 0; index < Polygons[i].size(); index++) {
      point_r3_t p0 = Polygons[i].at(index);
      if (index < (Polygons[i].size() - 1)) {
        point_r3_t p1 = Polygons[i].at(index + 1);
        p0.transform(MVP);
        p1.transform(MVP);
        if (orientation.at(i)) {
          fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(255,0,0)\"/>\n",
                   p0.x + shift_x, p0.y + shift_y, p1.x + shift_x,p1.y + shift_y);
        }
        else {
          fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(0,0,0)\"/>\n",
                   p0.x + shift_x, p0.y + shift_y, p1.x + shift_x,p1.y + shift_y);
        }
      }
    }
  }
  if (video) {
    /*Writing current slice:*/
    for (int j = i; j <= nslices; j++) {
      for (int index = 0; index < nsegments; index++) {
        point_r3_t p0 = Polygons[j].at(index);
        if (index < (Polygons[j].size() - 1)) {
          point_r3_t p1 = Polygons[j].at(index + 1);
          p0.transform(MVP);
          p1.transform(MVP);
          if (orientation.at(i)) {
             fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(255,0,0)\"/>\n",
                      p0.x + shift_x, p0.y + shift_y, p1.x + shift_x,p1.y + shift_y);
          }
          else {
             fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(0,0,0)\"/>\n",
                      p0.x + shift_x, p0.y + shift_y, p1.x + shift_x,p1.y + shift_y);
          }
        }
      }
    }
  }
  fprintf (file,"</svg>\n");
  fclose(file);
}


/*-----------------------------------------------------------------------*/
/*Output a SVG file with each model layer (list of oriented points in {Polygons})*/
void export_svg_2d (string fileName, vector<contour> polygons[], int nplanes, const point_r3_t &aabbSize) {

  int nrgb = 8;
  rgb colors[nrgb];
  fill_colors (colors, nrgb);

  FILE *file = NULL;

  float dx = 0.0, dy = 0.0;

  file = fopen (fileName.c_str(), "w");
  printf("\n\nwriting output file: %s\n", fileName.c_str());

  if (!file) { exit(1); }

  add_svg_information (file);

  for (int p = 0; p < nplanes; p++) {
    vector<contour> P = polygons[p];
    const size_t k = P.size();
    const size_t slicePerRow = (size_t)sqrt((float)nplanes)*2;
    for (size_t i = 0; i < k; ++i) {
      for (size_t index = 0; index < P[i].P.size(); index++) {
        const point_r3_t &p0 = P[i].P.at(index);
        dx = (float)(p % slicePerRow)*(aabbSize.x * 1.05f);
        dy = (float)(p / slicePerRow)*(aabbSize.y * 1.05f);
        if (index < (P[i].P.size() - 1)) {
          const point_r3_t &p1 = P[i].P.at(index + 1);
          //if (P[i].external) {
          //if (true) {
          if (P[i].clockwise) {
            //fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(%d,%d,%d)\" marker-end=\"url(#arrow)\"/> \n",
            fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(%d,%d,%d)\"/>\n",
            dx + p0.x, dy + p0.y, dx + p1.x, dy + p1.y, colors[0].r, colors[0].g, colors[0].b);
            //dx + p0.x, dy + p0.y, dx + p1.x, dy + p1.y, colors[i % nrgb].r, colors[i % nrgb].g, colors[i % nrgb].b);
          }
          else {
            //fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(%d,%d,%d)\" marker-end=\"url(#arrow)\"/>\n",
            fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(%d,%d,%d)\"/>\n",
            dx + p0.x, dy + p0.y, dx + p1.x, dy + p1.y, colors[1].r, colors[1].g, colors[1].b);
            //dx + p0.x, dy + p0.y, dx + p1.x, dy + p1.y, colors[i % nrgb].r, colors[i % nrgb].g, colors[i % nrgb].b);
          }
        }
      }
    }
  }
  fprintf (file,"</svg>\n");
  fclose (file);
}

/*-----------------------------------------------------------------------*/
/*Output a SVG file without chaining (list of unoriented line segments).*/
void export_svg_no_chaining (string fileName, vector<line_t> Segments[], int k, const point_r3_t &aabbSize) {

  int nrgb = 8;
  rgb colors[nrgb];
  fill_colors (colors, nrgb);

  FILE *file = NULL;

  float dx = 0.0, dy = 0.0;

  file = fopen (fileName.c_str(), "w");
  printf("\n\nwriting output file: %s\n", fileName.c_str());

  if (!file) { exit(1); }

  add_svg_information (file);

  const size_t slicePerRow = (size_t)sqrt((float)k)*2;

  for (size_t i = 0; i < k; ++i) {
    dx = (float)(i%slicePerRow)*(aabbSize.x * 1.05f);
    dy = (float)(i/slicePerRow)*(aabbSize.y * 1.05f);
    for (const line_t &ls : Segments[i]) {
    fprintf (file, "   <line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"1\" stroke=\"rgb(%d,%d,%d)\"/>\n",
             dx + ls.v[0].x, dy + ls.v[0].y, dx + ls.v[1].x, dy + ls.v[1].y,
             colors[i % nrgb].r, colors[i % nrgb].g, colors[i % nrgb].b);
    }
    Segments[i].clear();
  }
  fprintf (file,"</svg>\n");
  fclose (file);
}


/*-----------------------------------------------------------------------*/
void export_video (vector<contour> polygons[], int nplanes) {

  vector<vector<point_r3_t>> P;
  vector<bool> orientation;
  for (int k = 0; k < nplanes; k++) {
    const size_t ncontours = polygons[k].size();
    for (int c = 0; c < ncontours; c++) {
      vector<point_r3_t> Pc = polygons[k].at(c).P;
      orientation.push_back(polygons[k].at(c).clockwise);
      P.push_back(Pc);
    }
  }

  for (int k = 0; k < P.size(); k++) {
    const size_t nsegments = P.at(k).size();
    for (int j = 0; j < nsegments; j++) {
      char filename[256];
      sprintf(filename, "./video/%05d_%05d.svg", k, j);
      printf("writing output file: %s\n", filename);
      export_svg (filename, P, k, j, true, orientation);
      printf("... done\n");
    }
  }
  printf("... done\n");
}

/*-----------------------------------------------------------------------*/
void export_svg_3d (vector<contour> polygons[], int nplanes) {

  vector<vector<point_r3_t>> P;
  vector<bool> orientation;
  for (int k = 0; k < nplanes; k++) {
    const size_t ncontours = polygons[k].size();
    for (int c = 0; c < ncontours; c++) {
      vector<point_r3_t> Pc = polygons[k].at(c).P;
      orientation.push_back(polygons[k].at(c).clockwise);
      P.push_back(Pc);
    }
  }

  char filename[256];
  sprintf(filename, "out_3d.svg");
  printf("\n\nwriting output file: %s\n", filename);
  export_svg (filename, P, P.size(), 0, false, orientation);
  printf("... done\n\n\n");
}


/*-----------------------------------------------------------------------*/
void fill_colors (rgb colors[], int nrgb) {
  assert (nrgb == 8);
  colors[0] = {  0,   0,   0}; /*Black*/
  colors[1] = {255,   0,   0}; /*Red*/
  colors[2] = {128,   0, 128}; /*Purple*/
  colors[3] = {  0, 128,   0}; /*Green*/
  colors[4] = {  0,   0, 255}; /*Blue*/
  colors[5] = {  0, 255, 255}; /*Cyan*/
  colors[6] = {128, 128,   0}; /*Olive*/
  colors[7] = {128, 128, 128}; /*Gray*/
}

/*-----------------------------------------------------------------------*/
/*SVG header information:*/
void add_svg_information (FILE *file) {
  fprintf (file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
  fprintf (file, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");
  fprintf (file, "<svg preserveAspectRatio=\"xMidYMid meet\" width=\"1024\" height=\"768\" viewBox=\"0 0 1024 768\"");
  fprintf (file, " xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\"");
  fprintf (file, " xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\"");
  fprintf (file, " xmlns:cc=\"http://web.resource.org/cc/\">\n");
  fprintf (file, " <defs>\n");
  fprintf (file, "  <marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refx=\"0\" refy=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\" viewBox=\"0 0 20 20\">\n");
  fprintf (file, "    <path d=\"M0,0 L0,6 L9,3 z\" fill=\"#f00\" />\n");
  fprintf (file, "  </marker>\n");
  fprintf (file, "</defs>\n");
}
