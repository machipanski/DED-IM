#include "image.h"

/* */
Image *create_image(int nrows, int ncols)
{
	Image *image = (Image *)malloc(sizeof(Image));
	image->nrows = nrows;
	image->ncols = ncols;
	image->pixels = (unsigned char *)malloc(nrows * ncols * sizeof(unsigned char));
	return image;
}

void printTuple(EdgeTableTuple *tup)
{
	int j;

	if (tup->buckets.size())
		printf("\nCount %d-----\n", (int)tup->buckets.size());

	for (j = 0; j < tup->buckets.size(); j++)
	{
		printf(" %d+%.2f+%.2f",
			   tup->buckets[j].ymax, tup->buckets[j].xofymin, tup->buckets[j].slopeinverse);
	}
}

/* */
void free_image(Image *image)
{
	free(image->pixels);
	free(image);
}

/* */
void fill_pixel(Image *image, int x, int y, int level)
{
	image->pixels[y * image->ncols + x] = level;
}

/* */
void fill_pixel2(Image *image, int x, int y)
{
	image->pixels[y * image->ncols + x] = 125;
}

/* */
void fill_image(Image *image)
{
	memset(image->pixels, 0, image->nrows * image->ncols * sizeof(unsigned char));
}

/* */
void write_image(Image *image, char *filename)
{
	FILE *file = fopen(filename, "w");
	fprintf(file, "P2\n");
	fprintf(file, "%d %d\n", image->ncols, image->nrows);
	fprintf(file, "255\n");
	int size = image->nrows * image->ncols;
	int i;
	for (i = 0; i < size; i++)
	{
		fprintf(file, "%d ", image->pixels[i]);
		if (i % 12 == 0)
		{
			fprintf(file, "\n");
		}
	}
	fclose(file);
}

/*Bresenham’s Line Drawing Algorithm*/
void drawline(Image *image, int x1, int y1, int x2, int y2, int level)
{

	int x, y;
	int dx, dy;
	int sx, sy;
	int err, e2;

	dx = x2 >= x1 ? x2 - x1 : x1 - x2;
	dy = y2 >= y1 ? y1 - y2 : y2 - y1;
	sx = x1 < x2 ? 1 : -1;
	sy = y1 < y2 ? 1 : -1;
	err = dx + dy;
	x = x1;
	y = y1;

	while (1)
	{
		fill_pixel(image, x, y, level);
		if ((x == x2) && (y == y2))
			break;
		e2 = 2 * err;
		if (e2 >= dy)
		{ // step x
			err += dy;
			x += sx;
		}
		if (e2 <= dx)
		{ // step y
			err += dx;
			y += sy;
		}
	}
}

void PolygonToRaster(vector<contour> polygons, Image *image, int dpi, double xmin, double ymin, int level)
{

	EdgeTableTuple EdgeTable[image->nrows];
	for (int i = 0; i < polygons.size(); i++)
	{
		vector<point_r3_t> Pi = polygons.at(i).P;
		for (int j = 1; j < Pi.size(); j++)
		{

			point_r3_t p0 = Pi.at(j - 1);
			point_r3_t p1 = Pi.at(j + 0);

			double angleA = p0.tangle * 180.0 / M_PI;
			double angleB = p1.tangle * 180.0 / M_PI;

			int dp0x = (int)round(dpi / 25.4 * (p0.x - xmin));
			int dp1x = (int)round(dpi / 25.4 * (p1.x - xmin));
			int dp0y = (int)round(dpi / 25.4 * (p0.y - ymin));
			int dp1y = (int)round(dpi / 25.4 * (p1.y - ymin));

			// printf("(%d,%d) -- (%d,%d)\n", dp0x, dp0y, dp1x, dp1y);

			dp0x = (dp0x < image->ncols) ? dp0x : image->ncols - 1;
			dp0x = (dp0x > 0) ? dp0x : 0;
			dp0y = (dp0y < image->nrows) ? dp0y : image->nrows - 1;
			dp0y = (dp0y > 0) ? dp0y : 0;
			dp1x = (dp1x < image->ncols) ? dp1x : image->ncols - 1;
			dp1x = (dp1x > 0) ? dp1x : 0;
			dp1y = (dp1y < image->nrows) ? dp1y : image->nrows - 1;
			dp1y = (dp1y > 0) ? dp1y : 0;

			storeEdgeInTable(EdgeTable, dp0x, dp0y, dp1x, dp1y, angleA, angleB);
		}
	}
	ScanlineFill(EdgeTable, image, level);
}

void PolygonToRasterVertical(vector<contour> polygons, Image *image, int dpi, double xmin, double ymin, int level)
{

	EdgeTableTupleVertical EdgeTable[image->ncols];
	for (int i = 0; i < polygons.size(); i++)
	{
		vector<point_r3_t> Pi = polygons.at(i).P;
		for (int j = 1; j < Pi.size(); j++)
		{
			//	    printf("%d %d\n", i, j);

			point_r3_t p0 = Pi.at(j - 1);
			point_r3_t p1 = Pi.at(j + 0);

			double angleA = p0.tangle * 180.0 / M_PI;
			double angleB = p1.tangle * 180.0 / M_PI;

			int dp0x = (int)round(dpi / 25.4 * (p0.x - xmin));
			int dp1x = (int)round(dpi / 25.4 * (p1.x - xmin));
			int dp0y = (int)round(dpi / 25.4 * (p0.y - ymin));
			int dp1y = (int)round(dpi / 25.4 * (p1.y - ymin));

			dp0x = (dp0x < image->ncols) ? dp0x : image->ncols - 1;
			dp0x = (dp0x > 0) ? dp0x : 0;
			dp0y = (dp0y < image->nrows) ? dp0y : image->nrows - 1;
			dp0y = (dp0y > 0) ? dp0y : 0;
			dp1x = (dp1x < image->ncols) ? dp1x : image->ncols - 1;
			dp1x = (dp1x > 0) ? dp1x : 0;
			dp1y = (dp1y < image->nrows) ? dp1y : image->nrows - 1;
			dp1y = (dp1y > 0) ? dp1y : 0;

			storeEdgeInTableVertical(EdgeTable, dp0x, dp0y, dp1x, dp1y, angleA, angleB);
		}
	}
	ScanlineFillVertical(EdgeTable, image, level);
}

void draw_contour(vector<contour> polygons, Image *image, int dpi, double xmin, double ymin, int level)
{

	for (int i = 0; i < polygons.size(); i++)
	{
		vector<point_r3_t> Pi = polygons.at(i).P;
		for (int j = 1; j < Pi.size(); j++)
		{

			point_r3_t p0 = Pi.at(j - 1);
			point_r3_t p1 = Pi.at(j + 0);

			double angleA = p0.tangle * 180.0 / M_PI;
			double angleB = p1.tangle * 180.0 / M_PI;

			// printf("Diomio (coordenadas) (%f %f) -- (%f,%f) (Angulo de cada segmento): %f - %f\n", p0.x, p0.y, p1.x, p1.y, angleA, angleB);

			int dp0x = (int)round(dpi / 25.4 * (p0.x - xmin));
			int dp1x = (int)round(dpi / 25.4 * (p1.x - xmin));
			int dp0y = (int)round(dpi / 25.4 * (p0.y - ymin));
			int dp1y = (int)round(dpi / 25.4 * (p1.y - ymin));

			dp0x = (dp0x < image->ncols) ? dp0x : image->ncols - 1;
			dp0x = (dp0x > 0) ? dp0x : 0;
			dp0y = (dp0y < image->nrows) ? dp0y : image->nrows - 1;
			dp0y = (dp0y > 0) ? dp0y : 0;
			dp1x = (dp1x < image->ncols) ? dp1x : image->ncols - 1;
			dp1x = (dp1x > 0) ? dp1x : 0;
			dp1y = (dp1y < image->nrows) ? dp1y : image->nrows - 1;
			dp1y = (dp1y > 0) ? dp1y : 0;

			drawline(image, dp0x, dp0y, dp1x, dp1y, level);
			/*area += (p0.x * p1.y - p0.y * p1.x);
			add_point (p0, p1, segments, &bb, (j == 1 ? true : false), i);
			if (j == Pi.size()-1) {
			   add_point (p1, Pi.at(0), segments, &bb, (j == 1 ? true : false), i);
			   area += (p1.x * Pi.at(0).y - p1.y * Pi.at(0).x);
			}*/
		}
	}
}

void convert_polygons_to_images(vector<contour> polygons[], int nplanes, const mesh_t *mesh, int dpi, bool fill_triangles)
{
	for (int k = 0; k < nplanes; k++)
	{

		// cout << "Processando plano: " << k << endl;
		/*http://www.pixelcalculator.com*/
		double w = mesh->getUpperRightVertex().x - mesh->getBottomLeftVertex().x;
		double h = mesh->getUpperRightVertex().y - mesh->getBottomLeftVertex().y;
		int nrows = (int)round((dpi * h) / 25.4);
		int ncols = (int)round((dpi * w) / 25.4);
		Image *image = create_image(nrows, ncols);
		fill_image(image);

		// Normal scan.
		PolygonToRaster(polygons[k], image, dpi, mesh->getBottomLeftVertex().x, mesh->getBottomLeftVertex().y, 125);
		// draw_contour(polygons[k], image, dpi, mesh->getBottomLeftVertex().x, mesh->getBottomLeftVertex().y, 125);

		// Vertical scan (invert the roles of x and y)
		PolygonToRasterVertical(polygons[k], image, dpi, mesh->getBottomLeftVertex().x, mesh->getBottomLeftVertex().y, 125);

		char filename[256];
		sprintf(filename, "../input/sliced/layer_%03d.pgm", k);
		printf("\n\nwriting image: %s\n", filename);
		write_image(image, filename);
		free_image(image);
		// printf("\n\nafter free image: %s\n", filename);
	}
}

bool compareBuckets(EdgeBucket a, EdgeBucket b)
{
	return a.xofymin < b.xofymin;
}

bool compareBucketsVertical(EdgeBucketVertical a, EdgeBucketVertical b)
{
	return a.yofxmin < b.yofxmin;
}

void storeEdgeInTuple(EdgeTableTuple *receiver, int ym, int xm, float slopInv, float angleA, float angleB)
{
	receiver->buckets.push_back(EdgeBucket{ym, (float)xm, slopInv, angleA, angleB});

	// sort the buckets
	sort(receiver->buckets.begin(), receiver->buckets.end(), compareBuckets);
}

void storeEdgeInTable(EdgeTableTuple *EdgeTable, int x1, int y1, int x2, int y2, float angleA, float angleB)
{
	float m, minv;
	int ymaxTS, xwithyminTS, scanline; // ts stands for to store

	if (x2 == x1)
	{
		minv = 0.000000;
	}
	else
	{
		m = ((float)(y2 - y1)) / ((float)(x2 - x1));

		// horizontal lines are not stored in edge table
		if (y2 == y1)
			return;

		minv = (float)1.0 / m;
	}

	if (y1 > y2)
	{
		scanline = y2;
		ymaxTS = y1;
		xwithyminTS = x2;
	}
	else
	{
		scanline = y1;
		ymaxTS = y2;
		xwithyminTS = x1;
	}
	// the assignment part is done..now storage..
	storeEdgeInTuple(&EdgeTable[scanline], ymaxTS, xwithyminTS, minv, angleA, angleB);
}

void storeEdgeInTupleVertical(EdgeTableTupleVertical *receiver, int ym, int xm, float slope, float angleA, float angleB)
{
	receiver->buckets.push_back(EdgeBucketVertical{xm, (float)ym, slope, angleA, angleB});

	// sort the buckets
	sort(receiver->buckets.begin(), receiver->buckets.end(), compareBucketsVertical);
}

void storeEdgeInTableVertical(EdgeTableTupleVertical *EdgeTable, int x1, int y1, int x2, int y2, float angleA, float angleB)
{
	float m;
	int xmaxTS, ywithxminTS, scanline; // ts stands for to store

	if (x2 == x1)
	{
		m = 0.000000;
	}
	else
	{
		m = ((float)(y2 - y1)) / ((float)(x2 - x1));

		// vertical lines are not stored in edge table
		if (x2 == x1)
			return;
	}

	if (x1 > x2)
	{
		scanline = x2;
		xmaxTS = x1;
		ywithxminTS = y2;
	}
	else
	{
		scanline = x1;
		xmaxTS = x2;
		ywithxminTS = y1;
	}
	// the assignment part is done..now storage..
	storeEdgeInTupleVertical(&EdgeTable[scanline], xmaxTS, ywithxminTS, m, angleA, angleB);
}

void removeEdgeByXmax(EdgeTableTupleVertical *Tup, int xx)
{
	int i, j;
	i = 0;
	while (i < Tup->buckets.size())
	{
		if (Tup->buckets[i].xmax == xx)
		{
			Tup->buckets.erase(Tup->buckets.begin() + i);
		}
		else
			i++;
	}
}

void removeEdgeByYmax(EdgeTableTuple *Tup, int yy)
{
	int i, j;
	i = 0;
	while (i < Tup->buckets.size())
	{
		if (Tup->buckets[i].ymax == yy)
		{
			Tup->buckets.erase(Tup->buckets.begin() + i);
		}
		else
			i++;
	}
}

void updatexbyslopeinv(EdgeTableTuple *Tup)
{
	int i;

	for (i = 0; i < Tup->buckets.size(); i++)
	{
		Tup->buckets[i].xofymin = Tup->buckets[i].xofymin + Tup->buckets[i].slopeinverse;
	}
}

void updatexbyslope(EdgeTableTupleVertical *Tup)
{
	int i;

	for (i = 0; i < Tup->buckets.size(); i++)
	{
		Tup->buckets[i].yofxmin = Tup->buckets[i].yofxmin + Tup->buckets[i].slope;
	}
}

void ScanlineFill(EdgeTableTuple *EdgeTable, Image *image, int level)
{
	/* Follow the following rules:
	1. Horizontal edges: Do not include in edge table
	2. Horizontal edges: Drawn either on the bottom or on the top.
	3. Vertices: If local max or min, then count twice, else count
		once.
	4. Either vertices at local minima or at local maxima are drawn.*/

	int i, j, x1, ymax1, x2, ymax2, FillFlag = 0, coordCount;
	EdgeTableTuple ActiveEdgeTuple;

	// we will start from scanline 0;
	// Repeat until last scanline:
	for (i = 0; i < image->nrows; i++) // 4. Increment y by 1 (next scan line)
	{

		// 1. Move from ET bucket y to the
		// AET those edges whose ymin = y (entering edges)
		for (j = 0; j < EdgeTable[i].buckets.size(); j++)
		{
			storeEdgeInTuple(&ActiveEdgeTuple, EdgeTable[i].buckets[j].ymax, EdgeTable[i].buckets[j].xofymin,
							 EdgeTable[i].buckets[j].slopeinverse,
							 EdgeTable[i].buckets[j].angleA,
							 EdgeTable[i].buckets[j].angleB);
		}
		// printTuple(&ActiveEdgeTuple);

		// 2. Remove from AET those edges for
		// which y=ymax (not involved in next scan line)
		removeEdgeByYmax(&ActiveEdgeTuple, i);

		// sort AET (remember: ET is presorted)
		sort(ActiveEdgeTuple.buckets.begin(), ActiveEdgeTuple.buckets.end(), compareBuckets);

		// printTuple(&ActiveEdgeTuple);

		// 3. Fill lines on scan line y by using pairs of x-coords from AET
		j = 0;
		FillFlag = 0;
		coordCount = 0;
		x1 = -1;
		x2 = -1;
		ymax1 = -1;
		ymax2 = -1;
		while (j < ActiveEdgeTuple.buckets.size())
		{
			if (coordCount % 2 == 0)
			{
				x1 = (int)(ActiveEdgeTuple.buckets[j].xofymin);
				ymax1 = ActiveEdgeTuple.buckets[j].ymax;
				if (x1 == x2)
				{
					/* three cases can arrive-
						1. lines are towards top of the intersection
						2. lines are towards bottom
						3. one line is towards top and other is towards bottom
					*/
					if (((x1 == ymax1) && (x2 != ymax2)) || ((x1 != ymax1) && (x2 == ymax2)))
					{
						x2 = x1;
						ymax2 = ymax1;
					}

					else
					{
						coordCount++;
					}
				}

				else
				{
					coordCount++;
				}
			}
			else
			{
				x2 = (int)ActiveEdgeTuple.buckets[j].xofymin;
				ymax2 = ActiveEdgeTuple.buckets[j].ymax;

				FillFlag = 0;

				// checking for intersection...
				if (x1 == x2)
				{
					/*three cases can arrive-
						1. lines are towards top of the intersection
						2. lines are towards bottom
						3. one line is towards top and other is towards bottom
					*/
					if (((x1 == ymax1) && (x2 != ymax2)) || ((x1 != ymax1) && (x2 == ymax2)))
					{
						x1 = x2;
						ymax1 = ymax2;
					}
					else
					{
						coordCount++;
						FillFlag = 1;
					}
				}
				else
				{
					coordCount++;
					FillFlag = 1;
				}

				if (FillFlag)
				{
					for (int x = x1 + 1; x < x2; x++)
						fill_pixel(image, x, i, level);

					// Angulos aqui
					fill_pixel(image, x1, i, 255);
					fill_pixel(image, x2, i, 255);
				}
			}
			j++;
		}

		// 5. For each nonvertical edge remaining in AET, update x for new y
		updatexbyslopeinv(&ActiveEdgeTuple);
	}
}

void ScanlineFillVertical(EdgeTableTupleVertical *EdgeTable, Image *image, int level)
{
	/* Follow the following rules:
	1. Horizontal edges: Do not include in edge table
	2. Horizontal edges: Drawn either on the bottom or on the top.
	3. Vertices: If local max or min, then count twice, else count
		once.
	4. Either vertices at local minima or at local maxima are drawn.*/

	int i, j, y1, xmax1, y2, xmax2, FillFlag = 0, coordCount;
	EdgeTableTupleVertical ActiveEdgeTuple;

	// we will start from scanline 0;
	// Repeat until last scanline:
	for (i = 0; i < image->ncols; i++) // 4. Increment y by 1 (next scan line)
	{

		// 1. Move from ET bucket y to the
		// AET those edges whose ymin = y (entering edges)
		for (j = 0; j < EdgeTable[i].buckets.size(); j++)
		{
			storeEdgeInTupleVertical(&ActiveEdgeTuple, EdgeTable[i].buckets[j].xmax, EdgeTable[i].buckets[j].yofxmin,
									 EdgeTable[i].buckets[j].slope,
									 EdgeTable[i].buckets[j].angleA,
									 EdgeTable[i].buckets[j].angleB);
		}
		// printTuple(&ActiveEdgeTuple);

		// 2. Remove from AET those edges for
		// which y=ymax (not involved in next scan line)
		removeEdgeByXmax(&ActiveEdgeTuple, i);

		// sort AET (remember: ET is presorted)
		sort(ActiveEdgeTuple.buckets.begin(), ActiveEdgeTuple.buckets.end(), compareBucketsVertical);

		// printTuple(&ActiveEdgeTuple);

		// 3. Fill lines on scan line y by using pairs of x-coords from AET
		j = 0;
		FillFlag = 0;
		coordCount = 0;
		y1 = -1;
		y2 = -1;
		xmax1 = -1;
		xmax2 = -1;
		while (j < ActiveEdgeTuple.buckets.size())
		{
			if (coordCount % 2 == 0)
			{
				y1 = (int)(ActiveEdgeTuple.buckets[j].yofxmin);
				xmax1 = ActiveEdgeTuple.buckets[j].xmax;
				if (y1 == y2)
				{
					/* three cases can arrive-
						1. lines are towards top of the intersection
						2. lines are towards bottom
						3. one line is towards top and other is towards bottom
					*/
					if (((y1 == xmax1) && (y2 != xmax2)) || ((y1 != xmax1) && (y2 == xmax2)))
					{
						y2 = y1;
						xmax2 = xmax1;
					}

					else
					{
						coordCount++;
					}
				}

				else
				{
					coordCount++;
				}
			}
			else
			{
				y2 = (int)ActiveEdgeTuple.buckets[j].yofxmin;
				xmax2 = ActiveEdgeTuple.buckets[j].xmax;

				FillFlag = 0;

				// checking for intersection...
				if (y1 == y2)
				{
					/*three cases can arrive-
						1. lines are towards top of the intersection
						2. lines are towards bottom
						3. one line is towards top and other is towards bottom
					*/
					if (((y1 == xmax1) && (y2 != xmax2)) || ((y1 != xmax1) && (y2 == xmax2)))
					{
						y1 = y2;
						xmax1 = xmax2;
					}
					else
					{
						coordCount++;
						FillFlag = 1;
					}
				}
				else
				{
					coordCount++;
					FillFlag = 1;
				}

				if (FillFlag)
				{
					// Angulos aqui
					fill_pixel(image, i, y1, 255);
					fill_pixel(image, i, y2, 255);
				}
			}
			j++;
		}

		// 5. For each nonvertical edge remaining in AET, update x for new y
		updatexbyslope(&ActiveEdgeTuple);
	}
}
