#include "polygon_t.h"

long int l_id = 0, p_id = 0;

/*Return the previous point (or edge) id for a given ith point (or edge): */
unsigned int Polygon::prev(unsigned int i)
{
   unsigned int j(0),prevLoop(0),currentLoop(0);

   while ( i > _nVertices[currentLoop] )
   {
     prevLoop=currentLoop;
     currentLoop++;
   }

   if( i==1 || (i==_nVertices[prevLoop]+1) ) j=_nVertices[currentLoop];
   else if( i <= _nVertices[currentLoop] ) j=i-1;

   return j;
}

//----------------------------------------------------------------------------
//return the next point (or edge) id for a given ith point (or edge);
//----------------------------------------------------------------------------
unsigned int Polygon::next(unsigned int i)
{
   unsigned int j(0),prevLoop(0),currentLoop(0);

   while ( i > _nVertices[currentLoop] )
   {
     prevLoop=currentLoop;
     currentLoop++;
   }

   if( i < _nVertices[currentLoop] ) j=i+1;
   else if ( i==_nVertices[currentLoop] )
   {
           if( currentLoop==0) j=1;
           else j=_nVertices[prevLoop]+1;
   }

   return j;
}

/*-----------------------------------------------------------------------*/
void Polygon::addDiagonal (unsigned int i, unsigned int j) {
  Type type = INSERT;
  line_t *diag = new line_t (_points[i], _points[j], type, ++l_id);
  _edges[diag->id()] = diag;
  _startAdjEdgeMap[i].insert(diag->id());
  _startAdjEdgeMap[j].insert(diag->id());
  _diagonals[diag->id()] = diag;
}

/*-----------------------------------------------------------------------*/
void Polygon::handleStartVertex (unsigned int i, avl_t **t) {
  double y = _points[i]->y;
  avl_update_key (t, y);
  //avl_print_in_order (*t, 0);
  _edges[i]->setHelper(i);
  _edges[i]->setKeyValue(y);
  avl_insert_line_base (t, _edges[i]);
}

/*-----------------------------------------------------------------------*/
void Polygon::handleEndVertex (unsigned int i, avl_t **t) {
  double y = _points[i]->y;
  avl_update_key (t, y);
  unsigned int previ = prev(i);
  line_t *edge = _edges[previ];
  unsigned int helper = _edges[previ]->helper();
  if(_points[helper]->type == MERGE) {
    addDiagonal(i, helper);
  }
  avl_remove (t, edge->keyValue());
}

/*-----------------------------------------------------------------------*/
void Polygon::handleSplitVertex (unsigned int i, avl_t **t) {
  double x = _points[i]->x;
  double y = _points[i]->y;
  avl_update_key (t, y);
  avl_node_t *tmp = avl_find_element_smaller_than (*t, x);
  assert (tmp != NULL);
  unsigned int helper = tmp->line->_helper;
  addDiagonal (i, helper);
  tmp->line->_helper = i;
  _edges[i]->setHelper(i);
  _edges[i]->setKeyValue(y);
  avl_insert_line_base (t, _edges[i]);
}

/*-----------------------------------------------------------------------*/
void Polygon::handleMergeVertex (unsigned int i, avl_t **t) {
  double x = _points[i]->x;
  double y = _points[i]->y;
  avl_update_key (t, y);
  unsigned int previ = prev(i);
  unsigned int helper = _edges[previ]->helper();
  if (_points[helper]->type == MERGE) {
    addDiagonal (i, helper);
  }
  avl_remove (t, _edges[previ]->keyValue());
  avl_node_t *tmp = avl_find_element_smaller_than (*t, x);
  assert (tmp != NULL);
  helper = tmp->line->_helper;
  if(_points[helper]->type == MERGE) {
    addDiagonal (i, helper);
  }
  tmp->line->_helper = i;
}

/*-----------------------------------------------------------------------*/
void Polygon::handleRegularVertexDown (unsigned int i, avl_t **t) {
  double y = _points[i]->y;
  avl_update_key (t, y);
  unsigned int previ = prev(i);
  unsigned int helper = _edges[previ]->helper();
  if(_points[helper]->type == MERGE) {
    addDiagonal(i, helper);
  }
  avl_remove (t, _edges[previ]->keyValue());
  _edges[i]->setHelper(i);
  _edges[i]->setKeyValue(y);
  avl_insert_line_base (t, _edges[i]);
}

/*-----------------------------------------------------------------------*/
void Polygon::handleRegularVertexUp (unsigned int i, avl_t **t) {
  avl_update_key (t, _points[i]->y);
  avl_node_t *tmp = avl_find_element_smaller_than (*t, _points[i]->x);
  assert (tmp != NULL);
  unsigned int helper = tmp->line->_helper;
  if(_points[helper]->type == MERGE) {
    addDiagonal (i, helper);
  }
  tmp->line->_helper = i;
}


/* Partition polygon to monotone polygon pieces: */
void Polygon::partition2Monotone()
{

     avl_t *t = NULL;

     //_edgebst.MakeEmpty();
     if(!_qpoints.empty() && _qpoints.top().type!=START) 
     {
                cout<<"Please check your input polygon:\n1)orientations?\n2)duplicated points?\n";
                cout<<"poly2tri stopped.\n";
                //exit(1);
                stop = true;
                return;
     }
     while(!_qpoints.empty() && (!stop) )
     {
          point_r3_t vertex=_qpoints.top();
          _qpoints.pop();
          unsigned int id=vertex.id;


          int idp = prev(id);
          int idn = next(id);
          point_r3_t p = *_points[id], pnext=*_points[idn], pprev=*_points[idp];
          
          //printf("Caracas = %f\n", p.tangle);
          /*printf("pprev = (%f, %f),  p = (%f, %f), pnext = (%f, %f)\n", pprev.x, pprev.y, p.x, p.y, pnext.x, pnext.y);

          printf("vertext x,y = %f %f - vertext id: %d, vertex type: %d\n", vertex.x, vertex.y, vertex.id, vertex.type);

          cout << "id: " << id << ", size: " << _edges.size() << ", type: " << vertex.type  << endl;
          cout << "start: " << START << ", end: " << END << ", merge: " << MERGE << endl;
          cout << "split: " << SPLIT << ", regup: " << REGULAR_UP << ", refdown: " << REGULAR_DOWN << endl;


          printf("avl_t -------------------------\n");
          avl_print_in_order (t, 0);
          printf("END -------------------------\n"); */

          switch(vertex.type)
          {
                case START:        handleStartVertex(id, &t);       break;
                case END:          handleEndVertex(id, &t);         break;
                case MERGE:        handleMergeVertex(id, &t);       break;
                case SPLIT:        handleSplitVertex(id, &t);       break;
                case REGULAR_UP:   handleRegularVertexUp(id, &t);   break;
                case REGULAR_DOWN: handleRegularVertexDown(id, &t); break;
                default:
                                   cout<<"No duplicated points please! poly2tri stopped\n";
                                   stop = true;
                                   //exit(1); break;                         
          }
          //cout << "after switch" << endl << endl;
     }
}

//----------------------------------------------------------------------------
//searching all monotone pieces;
//----------------------------------------------------------------------------
void Polygon::searchMonotones()
{
    int loop=0;
    
    LineMap edges=_edges;
    
    while( edges.size() > _diagonals.size() )
    {
	    loop++;
	    Monopoly poly;
	    LineMap::iterator it=edges.begin();
	    point_r3_t* startp=startp=it->second->endPoint(0);
	    point_r3_t* endp=0;
	    line_t *next = it->second;
	    
	    poly.push_back(startp->id);

	    if(_debug)
	    {
	    	_logfile<<"Searching for loops:"<<loop<<'\n';
	    	_logfile<<"vertex index:"<<startp->id<<" ";
	    }
	    
	    for(;;) 
	    {
		    endp=next->endPoint(1);
		    if(next->type()!=INSERT) 
		    	{ 
				edges.erase(next->id());
				_startAdjEdgeMap[next->endPoint(0)->id].erase(next->id());
			}
		    if(endp==startp) break; 
		    poly.push_back(endp->id);
		    
		    if(_debug) _logfile<<endp->id<<" ";
		    
		    unsigned int nexte=selectNextEdge(next);
		    
		    if(nexte==0) 
		    {
			    cout<<"Please check your input polygon:\n";
		            cout<<"1)orientations?\n2)with duplicated points?\n3)is a simple one?\n";
			    cout<<"poly2tri stopped.\n";
			    //exit(1); 
                            stop = true;
		    }
		    //assert( nexte > 0);
		    next=edges[nexte];
		    if(next->endPoint(0) !=endp ) next->reverse(); 
	    }
	   
	    if(_debug) _logfile<<"\nloop closed!\n\n";

	    _mpolys.push_back(poly);
    }
}


//----------------------------------------------------------------------------
//triangulate a monotone polygon;
//----------------------------------------------------------------------------
void  Polygon::triangulateMonotone(Monopoly& mpoly)
{
       	
      PQueue qvertex; 	
      Monopoly::iterator it=mpoly.begin(), itnext;
      for(; itnext=it, it!=mpoly.end(); it++) 
      {
	      itnext++;  
	      if(itnext==mpoly.end()) itnext=mpoly.begin();
	      point_r3_t point=*_points[*it], pointnext=*_points[*itnext];
	      point.left=(point > pointnext)? true:false;
	      qvertex.push(point);
      }
     
     stack<point_r3_t> spoint;
     for(int i=0; i<2; i++) { spoint.push(qvertex.top()); qvertex.pop(); }
     
     while ( qvertex.size() > 1 )
     {
	point_r3_t topQueuePoint=qvertex.top();
	point_r3_t topStackPoint=spoint.top();
		
        if(topQueuePoint.left!=topStackPoint.left)
	{
            while ( spoint.size()  > 1 )
	    {
		  point_r3_t p1=spoint.top();
		  spoint.pop(); 
		  point_r3_t p2=spoint.top();
		  Triangle v(3);
		  v[0]=topQueuePoint.id;
		  v[1]=p1.id;
		  v[2]=p2.id;
		  _triangles.push_back(v);
		  
		  if(_debug) _logfile<<"Add triangle:"<<v[0]<<" "<<v[1]<<" "<<v[2]<<'\n';
		  
	    }
	    spoint.pop();
            spoint.push(topStackPoint);
            spoint.push(topQueuePoint); 	    
	}
	else
	{
	    while( spoint.size() > 1 )
	    {	    
            	point_r3_t stack1Point=spoint.top();
	    	spoint.pop(); 
                point_r3_t stack2Point=spoint.top();
	    	spoint.push(stack1Point);
	    	double pa[2], pb[2], pc[2];
	    	pa[0]=topQueuePoint.x; pa[1]=topQueuePoint.y;
	    	pb[0]=stack2Point.x;   pb[1]=stack2Point.y;
	    	pc[0]=stack1Point.x;   pc[1]=stack1Point.y;
		
		if(_debug)
		{
	    		_logfile<<"current top queue vertex index="<<topQueuePoint.id<<'\n';
	    		_logfile<<"Current top stack vertex index="<<stack1Point.id<<'\n';
	    		_logfile<<"Second stack vertex index="<<stack2Point.id<<'\n';
		}

            	double area=orient2d(pa,pb,pc);
	    	bool   left=stack1Point.left;
            	if( (area > 0 && left) || (area < 0 && !left ) ) 
	    	{
                        Triangle v(3);
                        v[0]=topQueuePoint.id;
                        v[1]=stack2Point.id;
                        v[2]=stack1Point.id;
                        _triangles.push_back(v);
	                if(_debug) _logfile<<"Add triangle:"<<v[0]<<" "<<v[1]<<" "<<v[2]<<'\n';
	        	spoint.pop();
	    	} else break;
	    }
	    
	    spoint.push(topQueuePoint); 
	    
	}
	
	qvertex.pop();
	
     }
     
     point_r3_t lastQueuePoint=qvertex.top();
     while( spoint.size() !=1 )
     {
	     point_r3_t topPoint=spoint.top();
	     spoint.pop();
	     point_r3_t top2Point=spoint.top();
	      
             Triangle v(3);
             v[0]=lastQueuePoint.id;
             v[1]=topPoint.id;
             v[2]=top2Point.id;
             _triangles.push_back(v);
	     
	     if(_debug) _logfile<<"Add triangle:"<<v[0]<<" "<<v[1]<<" "<<v[2]<<'\n';
     }
}

//----------------------------------------------------------------------------
//two Auxiliary functions to find monotone polygon pieces
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//calculate angle B for A, B, C three given points
//----------------------------------------------------------------------------
double Polygon::angleCosb(double *pa, double *pb, double *pc)
{
  double dxab = pa[0] - pb[0];
  double dyab = pa[1] - pb[1];

  double dxcb = pc[0] - pb[0];
  double dycb = pc[1] - pb[1];

  double dxab2 = dxab * dxab;
  double dyab2 = dyab * dyab;
  double dxcb2 = dxcb * dxcb;
  double dycb2 = dycb * dycb;
  double ab = dxab2 + dyab2;
  double cb = dxcb2 + dycb2;

  double cosb = dxab * dxcb + dyab * dycb;
  double denom = sqrt( ab * cb);

  cosb/=denom;
  
  return cosb;
}

//----------------------------------------------------------------------------
//for any given edge, find the next edge we should choose when searching for
//monotone polygon pieces; 
//----------------------------------------------------------------------------
unsigned int Polygon::selectNextEdge (line_t *edge)
{

    unsigned int eid= edge->endPoint(1)->id;
    set<unsigned int> edges=_startAdjEdgeMap[eid];
    assert(!edges.empty());

    unsigned int nexte=0;
    if( edges.size() == 1 )  nexte=*(edges.begin());
    else if( edges.size() > 1 )
    {
        unsigned int nexte_ccw(0), nexte_cw(0);
        double max=-2.0,min=2.0;


        set<unsigned int>::iterator it=edges.begin();
        for(; it!=edges.end(); it++)
        {
                if(*it==edge->id()) continue;
                double A[2], B[2], C[2];
                A[0]=edge->endPoint(0)->x;        A[1]=edge->endPoint(0)->y;
                B[0]=edge->endPoint(1)->x;        B[1]=edge->endPoint(1)->y;

                if(edge->endPoint(1)!=_edges[*it]->endPoint(0)) _edges[*it]->reverse();
                C[0]=_edges[*it]->endPoint(1)->x; C[1]=_edges[*it]->endPoint(1)->y;

                double area=orient2d(A, B, C);
                double cosb=angleCosb(A, B, C);

                if( area > 0 && max < cosb ) { nexte_ccw=*it; max=cosb; }
                else if( min > cosb ) { nexte_cw=*it; min=cosb; }
        }

        nexte = (nexte_ccw!=0) ? nexte_ccw : nexte_cw;
    }

   return nexte;
}

void Polygon::triangulation()
{

     //while(!_qpoints.empty()) _qpoints.pop();
/*#if defined(WIN32)
     struct _timeb tv0, tv1;    
     _ftime_s( &tv0);
#else
     struct timeval tv0, tv1;
     struct timezone tz;
     gettimeofday(&tv0, &tz);
#endif*/
                                  
     partition2Monotone();
     if (!stop) {
     searchMonotones();
     if (!stop) {
       //printf("apos search monotones\n");
       Monopolys::iterator it=_mpolys.begin();
       for(; it!=_mpolys.end(); it++)
          triangulateMonotone(*it);
/*#if defined(WIN32)
     _ftime_s( &tv1);
#else
     gettimeofday(&tv1, &tz);
#endif
     CheckTime(tv0, tv1); */    
     cout<<"Total number of triangles:"<<_triangles.size()<<'\n';
    //~Polygon();
   }
   }
   /*Cleaning*/
   /*PointbaseMap::iterator itp=_points.begin();
   for(; itp!=_points.end(); itp++)
   {
           delete itp->second;
   }

   LineMap::iterator itl=_edges.begin();
   for(; itl!=_edges.end(); itl++)
   {
          delete itl->second;
   }*/
}

bool Polygon::read_bdm (vector<contour> P) {

  //printf("lendo o bdm!\n");
  unsigned int i = 1, first, last;
  double x,y;
  Type type;

   _nVertices.clear();
   _points.clear();
   _edges.clear();
   /*cout << "_nVertices size: " << _nVertices.size() << endl;
   cout << "_points size: " << _points.size() << endl;
   cout << "_edges size: " << _edges.size() << endl;*/

  _ncontours = 0; 
  int _NC = P.size(); 
  //cout << "_NC size: " << _NC << endl;

  l_id = 0;
  p_id = 0;

  for (int c = 0; c < _NC; c++) {
     vector<point_r3_t> C = P.at(c).P;
     if (C.front() == C.back()) { //Chain {P} is closed
     //cout << "Vou adicionar mais um contorno com numero de pontos: " << C.size() << endl;
     int npoints = C.size()-1; /*excluding the last points that close the polygon.*/ 
     for (int j = 0; j < npoints; j++) {
        double x = C.at(j).x;
        double y = C.at(j).y;
	double tangle = C.at(j).tangle;
        if( is_exist(x, y) ) {
           cout<<"Error! I found the duplicate point!!, the point is:";
           cout<<i<<"("<<x<<","<<y<<").\n";
           //exit(1);
           return true;
        }
        else {
          point_r3_t* point=new point_r3_t(i,x,y,type,tangle);
          if(x > _xmax) _xmax=x; 
          if(x < _xmin) _xmin=x;
          if(y > _ymax) _ymax=y;
          if(y < _ymin) _ymin=y;
          _points[i]=point;
          i++;
        }
     }
     _nVertices.push_back(npoints);
     _ncontours++;
     //printf("Adicionando mais %d pontos\n", npoints);
    }
     //printf("Quanto vale o C.size %d\n", (int)C.size());
  }

  int sid,eid;
  int num=0;
  int lixo = 0;
  int j = 0;
  //for(unsigned int j=0; j<_ncontours; j++)
  for (int c = 0; c < _NC; c++) {
     vector<point_r3_t> C = P.at(c).P;
     if (C.front() == C.back()) { //Chain {P} is closed
     int npoints = C.size()-1; /*excluding the last points that close the polygon.*/ 
     for(i=1; i<=_nVertices[j]; i++) {
          sid=num+i;
          eid=(i==_nVertices[j])?num+1:num+i+1;
          type = INPUT;
          line_t *line = new line_t (_points[sid], _points[eid], type, ++l_id);
          _edges[l_id]=line;
          lixo++;
        }
        //_nVertices.push_back(C.size());
       //_nVertices[j] = C.size();
       num+=_nVertices[j];
       j++;
    }
  }

   int sum=0;
   for(unsigned int i=0; i<_ncontours; i++)
   {
       //printf("i = %d, _nVertices = %d, sum = %d, _ncontours: %d\n", i, _nVertices[i], sum, _ncontours);
           sum+= _nVertices[i];
           _nVertices[i]=sum;
   }
           
   p_id=num;

   //cout << "_edges size: " << _edges.size() << endl;
   //cout << "_nVertices size: " << _nVertices.size() << endl;

   //printf("passei aqui 0: l_id %d, p_id %d, _ncontours: %d\n", l_id, p_id, _ncontours);

  if (_ncontours != _NC) { return true; }

  return false;
}

//---------------------------------------------------------------------------
//check there exists duplicated points or not;
//---------------------------------------------------------------------------
bool Polygon::is_exist(double x, double y)
{
   bool result=false;

   PointbaseMap::iterator it=_points.begin();
   for(; it!=_points.end(); it++)
    {
      if( it->second->x==x && it->second->y==y )  { result=true; break; }
    }

   return result;
}

//----------------------------------------------------------------------------
//polygon initialization;
//to find types of all polygon vertices;
//create a priority queue for all vertices;
//construct an edge set for each vertex (the set holds all edges starting from 
//the vertex, only for loop searching purpose). 
//----------------------------------------------------------------------------
void Polygon::initializate()
{
     //printf("initializate 1\n");
     while(!_qpoints.empty()) _qpoints.pop();
     _startAdjEdgeMap.clear(); 
     //printf("initializate 2\n");

     PointbaseMap::iterator it=_points.begin();
     for(; it!=_points.end(); it++)
	{
            //printf("-----------------------------------\n");
	    int id=it->first;
            //if (id == 185) { id = 184; }
	    int idp=prev(id);
	    int idn=next(id);
	    point_r3_t p=*_points[id], pnext=*_points[idn], pprev=*_points[idp];
	    //printf("Caracas: %f\n", p.tangle);
            //printf("initializate 16\n");
	   
	    if( p > pnext && pprev > p ) {
	        _points[id]->type=REGULAR_DOWN;
            }
	    else if (p > pprev && pnext > p) {
		_points[id]->type=REGULAR_UP;
            }
	    else
	    {
               double pa[2], pb[2], pc[2];
	       
               pa[0]=_points[idp]->x;
	       pa[1]=_points[idp]->y;

	       pb[0]=_points[id]->x;
	       pb[1]=_points[id]->y;

	       pc[0]=_points[idn]->x;
	       pc[1]=_points[idn]->y;
 
	       double area=orient2d(pa,pb,pc);

               /*Meu angulo*/
                   //pb[0] - pa[0]                    
               /*End*/


               if (area == 0.0) {
                  printf("############################\n");
                  printf("############################\n");
                  printf("####### AREA  0.0 ##########\n");
                  printf("############################\n");
                  printf("############################\n");
                  stop = true;
               }
	       //double area2=orient2dslow(pa,pb,pc);

	       if( pprev > p && pnext > p ) { 
                  _points[id]->type=(area > 0) ? END: MERGE;
                   //printf("TIPO %d\n", _points[id]->type);	   
               }
	       if( pprev < p && pnext < p ) {
                  _points[id]->type=(area > 0) ? START : SPLIT;
               }

               //printf("area: %f, area2: %f \n", area, area2);	   
               //printf("area: %f\n", area);	   
	     }

            //printf("initializate 3\n");
	    _qpoints.push(*(it->second));
            //printf("initializate 4\n");

	    _startAdjEdgeMap[id].insert(id);
            //printf("initializate 5\n");
	}
     //printf("initializate 6\n");
}



Polygon::Polygon (vector<contour> P)
{
    //double infinity=numeric_limits<double>::max();
  const double infinity=1.0e38;
  _xmin=_ymin=infinity;
  _xmax=_ymax=-infinity;
  stop = false;
  //if(parse) read_bdm(filename);
  //else read_bdm(filename, false);
  if (!(stop = read_bdm(P))) {
    /*_prefix=filename;
    int pos=_prefix.find_last_of('.');
    if(pos!=-1) _prefix.erase(pos);*/

    //printf("antes de inicializar\n"); 
    initializate();
    //printf("depois de inicializar\n"); 
     //_debug=false;
  }
}

