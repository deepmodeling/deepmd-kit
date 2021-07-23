#pragma once

#include "SimulationRegion.h"
#include <vector> 

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class Tabulated
{
public:
  Tabulated () {};
  Tabulated (const VALUETYPE rc,
	     const VALUETYPE hh,
	     const vector<VALUETYPE> & tab);
  void reinit (const VALUETYPE rc,
	       const VALUETYPE hh,
	       const vector<VALUETYPE> & tab);
public:
  void compute (VALUETYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const SimulationRegion<VALUETYPE> &	region, 
		const vector<vector<int > > &	nlist);
  void compute (VALUETYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<VALUETYPE> &	charge,
		const vector<int> &		atype,
		const SimulationRegion<VALUETYPE> &	region, 
		const vector<vector<int > > &	nlist);
  void tb_inner (VALUETYPE & ae,
		 VALUETYPE & af,
		 const VALUETYPE & r2);
private:
  VALUETYPE rc2, hi;
  vector<VALUETYPE> data;
  void compute_posi (int & idx, 
		     VALUETYPE & eps,
		     const VALUETYPE & rr);
}
    ;
