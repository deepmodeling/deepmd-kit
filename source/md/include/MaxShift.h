#pragma once

#include "SimulationRegion.h"
#include <vector> 

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class MaxShift 
{
public:
  MaxShift (const vector<VALUETYPE> & dcoord, 
	    const VALUETYPE & shell);
  
  bool rebuild (const vector<VALUETYPE> & coord, 
		const SimulationRegion<VALUETYPE> & region) ;
private:
  VALUETYPE
  max_shift2 (const vector<VALUETYPE> & coord, 
	      const SimulationRegion<VALUETYPE> & region) ;
  vector<VALUETYPE> record;
  VALUETYPE shell;
  VALUETYPE max_allow2;
};

