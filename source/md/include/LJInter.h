#pragma once

#include "SimulationRegion.h"
#include <vector> 

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class LJInter
{
public:
  LJInter (const VALUETYPE & c6,
	   const VALUETYPE & c12,
	   const VALUETYPE & rc);
public:
  void compute (VALUETYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const SimulationRegion<VALUETYPE> &	region, 
		const vector<vector<int > > &	nlist);
private:
  VALUETYPE c6, c12, rc, rc2, one_over_6, one_over_12, one_over_rc6, one_over_rc12;
  void 
  lj_inner (VALUETYPE & ae,
	    VALUETYPE & af,
	    const VALUETYPE & r2);
}
    ;


