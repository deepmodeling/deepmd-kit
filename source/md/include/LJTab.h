#pragma once

#include <vector> 
#include "Tabulated.h"

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class LJTab 
{
public:
  LJTab (const VALUETYPE & c6,
	 const VALUETYPE & c12,
	 const VALUETYPE & rc);
public:
  void compute (VALUETYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const SimulationRegion<VALUETYPE> &	region, 
		const vector<vector<int > > &	nlist)
      {lj_tab.compute (ener, force, virial, coord, atype, region, nlist);};
private:
  Tabulated lj_tab;
}
    ;

