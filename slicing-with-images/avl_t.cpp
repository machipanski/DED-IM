#include "avl_t.h"

/*----------------------*/
int avl_max (int left, int right) {
   return (left > right ? left : right);
}

/*----------------------*/
int avl_height (avl_t *t) {
   if (t == NULL) {
      return -1;
   }
   else {
      return t->height;
   }
}

/*----------------------*/
int avl_update_height (avl_t *t) {
   return (avl_max(avl_height(t->left), avl_height(t->right)) + 1);
}

/*----------------------*/
int avl_balance (avl_t *t) {
   return (avl_height(t->right) - avl_height(t->left));
}

/*----------------------*/
avl_t* avl_simple_left_rotation (avl_t *r) {
   avl_node_t *t = r->right;
   r->right = t->left;
   t->left = r;
   r->height = avl_update_height(r);
   t->height = avl_update_height(t);
   return t;
}

/*----------------------*/
avl_t* avl_simple_right_rotation (avl_t *r) {
   avl_node_t *t = r->left;
   r->left = t->right;
   t->right = r;
   r->height = avl_update_height(r);
   t->height = avl_update_height(t);
   return t;
}

/*----------------------*/
avl_t* avl_double_left_rotation (avl_t *t) {
   t->right = avl_simple_right_rotation (t->right);
   return avl_simple_left_rotation (t);
}

/*----------------------*/
avl_t* avl_double_right_rotation (avl_t *t) {
   t->left = avl_simple_left_rotation (t->left);
   return avl_simple_right_rotation (t);
}

/*----------------------*/
avl_t* avl_update_bf_right (avl_t *t) {
   t->height = avl_update_height (t);
   if (avl_balance(t) == +2) {
      if (avl_balance(t->right) >= 0) {
         t = avl_simple_left_rotation (t);
      }
      else {
         t = avl_double_left_rotation (t);
      }
   }
   return t;
}

/*----------------------*/
avl_t* avl_update_bf_left (avl_t *t) {
   t->height = avl_update_height (t);
   if (avl_balance(t) == -2) {
      if (avl_balance(t->left) <= 0) {
         t = avl_simple_right_rotation (t);
      }
      else {
         t = avl_double_right_rotation (t);
      }
   }
   return t;
}

/*----------------------*/
/*void avl_insert (avl_t **a, double key) {
   if ((*a) == NULL) {
      (*a) = (avl_node_t*)malloc(sizeof(avl_node_t));
      (*a)->key = key;
      (*a)->height = 0;
      (*a)->left = (*a)->right = NULL;
   }
   else if (key < (*a)->key) {
      avl_insert (&((*a)->left), key);
      (*a) = avl_update_bf_left (*a);
   }
   else if (key > (*a)->key) {
      avl_insert (&((*a)->right), key);
      (*a) = avl_update_bf_right (*a);
   }
}*/

/*----------------------*/
void avl_update_key (avl_t **a, double key) {
   if (*a != NULL) {
     avl_update_key (&((*a)->left),  key);
     if ( (*a)->line->_endp[1]->y == (*a)->line->_endp[0]->y ) {
        (*a)->key = ( (*a)->line->_endp[0]->x < (*a)->line->_endp[1]->x ) ? (*a)->line->_endp[0]->x : (*a)->line->_endp[1]->x;
        (*a)->line->setKey((*a)->key);
     }
     else {
        (*a)->key = (key - (*a)->line->_endp[0]->y)*( (*a)->line->_endp[1]->x - (*a)->line->_endp[0]->x )/((*a)->line->_endp[1]->y - (*a)->line->_endp[0]->y ) + (*a)->line->_endp[0]->x;
        (*a)->line->setKey((*a)->key);
     }
     avl_update_key (&((*a)->right),  key);
   }
}

/*----------------------*/
void avl_insert_line_base (avl_t **a, line_t *line) {
   if ((*a) == NULL) {
      (*a) = (avl_node_t*)malloc(sizeof(avl_node_t));
      (*a)->key = line->keyValue();
      (*a)->height = 0;
      (*a)->left = (*a)->right = NULL;
      (*a)->line = line;
   }
   else if (line->keyValue() < (*a)->key) {
      avl_insert_line_base (&((*a)->left), line);
      (*a) = avl_update_bf_left (*a);
   }
   else if (line->keyValue() > (*a)->key) {
      avl_insert_line_base (&((*a)->right), line);
      (*a) = avl_update_bf_right (*a);
   }
   else {
      /*No repetitions allowed*/
      assert(false);
   }
}

/*----------------------*/
avl_node_t* avl_find_element_smaller_than (avl_t* a, double key) {

   avl_node_t* closest = NULL;
   double min = std::numeric_limits<double>::max();
   avl_node_t* pavl_node_t = a;
   while(pavl_node_t != NULL){
      //printf("avl_find_element_smaller_than, testando com: %f\n", pavl_node_t->key);
      double distance = fabs(pavl_node_t->key - key);
      if((distance < min) && (pavl_node_t->key < key)){
         min = distance;
         closest = pavl_node_t;
      }
      if (distance == 0) {
         //printf("avl_find_element_smaller_than, distancia = 0, cai no break:\n");
         break;
      }

      if(pavl_node_t->key > key) {
         //printf("avl_find_element_smaller_than, pavl_node_t->key > key, pegando a esq\n");
         pavl_node_t = pavl_node_t->left;
      }
      else if(pavl_node_t->key < key) {
         //printf("avl_find_element_smaller_than, pavl_node_t->key < key, pegando a dir\n");
         pavl_node_t = pavl_node_t->right;
      }
   }
   if (closest->key <= key) {
     return closest;
   }
   else {
     //printf("Cai no NULL com chave: %f queria um menor que %f\n", closest->key, key);
     return NULL;
   }
}

/*----------------------*/
void avl_remove (avl_t **a, double key) {
   if ((*a) != NULL) {
      if (key < (*a)->key) {
         avl_remove (&((*a)->left), key);
         (*a) = avl_update_bf_right (*a);
      }
      else if (key > (*a)->key) {
         avl_remove (&((*a)->right), key);
         (*a) = avl_update_bf_left (*a);
      }
      else {
         if (((*a)->left == NULL) && ((*a)->right == NULL)) {
            free (*a);
            *a = NULL;
         }
         else if ((*a)->left == NULL) {
            avl_node_t *tmp = *a;
            *a = (*a)->right;
            free (tmp);
         }
         else if ((*a)->right == NULL) {
            avl_node_t *tmp = *a;
            *a = (*a)->left;
            free (tmp);
         }
         else {
            avl_node_t *tmp = (*a)->left;
            while (tmp->right != NULL) {
               tmp = tmp->right;
            }
            (*a)->key = tmp->key;
            (*a)->line = tmp->line;
            tmp->key = key;
            avl_remove (&((*a)->left), key);
            (*a) = avl_update_bf_right (*a);
         }
      }
   }
}

/*----------------------*/
void avl_print_in_order (avl_t* a, int level) {
   if (a != NULL) {
      int i;
      for (i = 0; i < level; i++) {
         printf("      ");
      }
      printf("(bf: %+d) at level: %d  (%f:%f  ..  %f:%f), helper: %d, id: %d, (edge %d : %d)\n", avl_balance(a), level,
           a->line->_endp[0]->x, a->line->_endp[0]->y, a->line->_endp[1]->x, a->line->_endp[1]->y, a->line->_helper, a->line->_id, a->line->_endp[0]->label, a->line->_endp[1]->label);
      avl_print_in_order (a->left, level + 1);
      avl_print_in_order (a->right, level + 1);
   }
   else {
      int i;
      for (i = 0; i < level; i++) {
         printf("      ");
      }
      printf("----\n");
   }
}
