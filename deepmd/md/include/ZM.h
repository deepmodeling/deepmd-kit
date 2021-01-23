#pragma once

#include <vector> 
#include "Tabulated.h"
#include "ZMFunctions.h"
#include "SimulationRegion.h"

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class ZM 
{
public:
  ZM (const int & order,
      const VALUETYPE & alpha,
      const VALUETYPE & rc);
public:
  void compute (VALUETYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<VALUETYPE> &	charge,
		const vector<int> &		atype,
		const SimulationRegion<VALUETYPE> &	region, 
		const vector<vector<int > > &	nlist)
      {zm_tab.compute (ener, force, virial, coord, charge, atype, region, nlist);};
  void exclude  (VALUETYPE &			ener,
		 vector<VALUETYPE> &		force,
		 vector<VALUETYPE> &		virial,
		 const vector<VALUETYPE> &	coord,
		 const vector<VALUETYPE> &	charge,
		 const vector<int> &		atype,
		 const SimulationRegion<VALUETYPE> &	region, 
		 const vector<int > &		elist);
  VALUETYPE e_corr (const vector<VALUETYPE> & charge) const;
private:
  Tabulated zm_tab;
  void ex_inner (VALUETYPE & ae,
		 VALUETYPE & af,
		 const VALUETYPE & r2);
  ZeroMultipole::Potential potzm;
}
    ;

