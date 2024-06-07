#ifndef _image_h_
#define _image_h_

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include "contour.h"
#include "triangle_t.h"
#include "polygon_t.h"

using namespace std;

typedef struct _image {
  int nrows;
  int ncols;
  unsigned char *pixels;
} Image;

typedef struct edgebucket
{
	int ymax; //max y-coordinate of edge
	float xofymin; //x-coordinate of lowest edge point updated only in aet
	float slopeinverse;
	float angleA;
       	float angleB;
}EdgeBucket;

typedef struct edgetabletup
{
	// the array will give the scanline number
	// The edge table (ET) with edges entries sorted
	// in increasing y and x of the lower end
	
	int countEdgeBucket; //no. of edgebuckets
	vector<EdgeBucket> buckets;
}EdgeTableTuple;

typedef struct edgebucketVertical
{
	int xmax;
	float yofxmin;
	float slope;
	float angleA;
       	float angleB;
}EdgeBucketVertical;

typedef struct edgetabletupVertical
{
	// the array will give the scanline number
	// The edge table (ET) with edges entries sorted
	// in increasing x and y of the lower end
	
	int countEdgeBucket; //no. of edgebuckets
	vector<EdgeBucketVertical> buckets;
}EdgeTableTupleVertical;

Image* create_image (int nrows, int ncols);

void free_image (Image *image);

void fill_pixel (Image *image, int x, int y, int level);

void fill_image (Image *image);

void fill_pixel2 (Image *image, int x, int y);

void write_image (Image *image, char *filename);

void drawline (Image *image,  int x1,  int y1,  int x2,  int y2, int level);

void convert_polygons_to_images (vector<contour> polygons[], int nplanes, const mesh_t *mesh, int dpi, bool fill_triangles);

void PolygonToRaster(vector<contour> polygons, Image *image, int dpi, double xmin, double ymin, int level);

void PolygonToiVertical(vector<contour> polygons, Image *image, int dpi, double xmin, double ymin, int level);

void draw_contour (vector<contour> polygons, Image *image, int dpi, double xmin, double ymin, int level);

bool compareBuckets(EdgeBucket a, EdgeBucket b);
bool compareBucketsVErtical(EdgeBucket a, EdgeBucket b);

void storeEdgeInTuple(EdgeTableTuple *receiver,int ym,int xm,float slopInv, float angleA, float angleB);
void storeEdgeInTupleVertical(EdgeTableTupleVertical *receiver,int ym,int xm,float slope, float angleA, float angleB);

void storeEdgeInTable(EdgeTableTuple *EdgeTable, int x1,int y1, int x2, int y2, float angleA, float angleB);
void storeEdgeInTableVertical(EdgeTableTupleVertical *EdgeTable, int x1,int y1, int x2, int y2, float angleA, float angleB);

void removeEdgeByYmax(EdgeTableTuple *Tup,int yy);
void removeEdgeByXmax(EdgeTableTupleVertical *Tup,int xx);

void updatexbyslopeinv(EdgeTableTuple *Tup);
void updatexbyslope(EdgeTableTupleVertical *Tup);

void ScanlineFill(EdgeTableTuple *EdgeTable, Image *image, int level);

void ScanlineFillVertical(EdgeTableTupleVertical *EdgeTable, Image *image, int level);

#endif
