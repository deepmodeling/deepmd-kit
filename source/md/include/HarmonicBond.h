#pragma once

#include "SimulationRegion.h"
#include <vector>

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class HarmonicBond 
{
public:
  HarmonicBond (const VALUETYPE & kk,
		const VALUETYPE & bb);
public:
  void compute (VALUETYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const SimulationRegion<VALUETYPE> &	region, 
		const vector<int > &		blist);
private:
  VALUETYPE kk, bb;
  void 
  hb_inner (VALUETYPE & ae,
	    VALUETYPE & af,
	    const VALUETYPE & r2);
}
    ;

