#ifndef _avl_t_h_
#define _avl_t_h_

#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include "line_t.h"

using namespace std;

typedef struct _avl_node_t {
   double key;                 /*key of a line segment for tree searching.*/
   int height;                 /*Tree height*/
   struct _avl_node_t* left;
   struct _avl_node_t* right;
   line_t *line;               /*Data*/
} avl_node_t, avl_t;


/*AVL functions*/
int avl_max (int left, int right);
int avl_height (avl_t *t);
int avl_update_height (avl_t *t);
int avl_balance (avl_t *t);
avl_t* avl_simple_left_rotation (avl_t *r);
avl_t* avl_simple_right_rotation (avl_t *r);
avl_t* avl_double_left_rotation (avl_t *t);
avl_t* avl_double_right_rotation (avl_t *t);
avl_t* avl_update_bf_right (avl_t *t);
avl_t* avl_update_bf_left (avl_t *t);
//void avl_insert (avl_t **a, double key);
void avl_update_key (avl_t **a, double key);
void avl_insert_line_base (avl_t **a, line_t *line);
avl_node_t* avl_find_element_smaller_than (avl_t* a, double key);
void avl_remove (avl_t **a, double key);
void avl_print_in_order (avl_t *a, int level);

#endif
