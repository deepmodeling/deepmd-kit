#pragma once

#include "SimulationRegion.h"
#include <vector>

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class HarmonicAngle 
{
public:
  HarmonicAngle (const VALUETYPE & kk,
		const VALUETYPE & tt);
public:
  void compute (VALUETYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const SimulationRegion<VALUETYPE> &	region, 
		const vector<int > &		alist);
private:
  VALUETYPE ka, tt;
  void 
  hb_inner (VALUETYPE & ae,
	    VALUETYPE & af,
	    const VALUETYPE & r2);
}
    ;

