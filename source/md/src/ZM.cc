#include "ZM.h"
#include "UnitManager.h"
#include "common.h"
#include <cmath>
#include <iostream>

ZM::
ZM (const int & order,
    const VALUETYPE & alpha,
    const VALUETYPE & rc)
    : potzm (order, alpha, rc)
{
  VALUETYPE rcp = rc + 2;
  VALUETYPE hh = 2e-3;
  int nn = rcp / hh;
  vector<VALUETYPE> tab;

  for (int ii = 0; ii < nn; ++ii){
    VALUETYPE xx = ii * hh;
    VALUETYPE value, deriv;
    if (xx <= rc) {
      value = potzm.pot (xx);
      deriv = potzm.mpotp (xx);      
    }
    else {
      value = deriv = 0;
    }
    tab.push_back (value);
    tab.push_back (deriv);
  }
  zm_tab.reinit (rcp, hh, tab);
}

VALUETYPE
ZM::
e_corr (const vector<VALUETYPE> & charge) const
{
  double sum = 0;
  sum += potzm.energyCorr (charge);
  return sum;
}

inline void 
ZM::
ex_inner (VALUETYPE & ae,
	  VALUETYPE & af,
	  const VALUETYPE & r2)
{
  VALUETYPE r1 = sqrt (r2);
  ae = 1./r1;
  af = 1./(r2 * r1);
}

void 
ZM::
exclude  (VALUETYPE &			ener,
	  vector<VALUETYPE> &		force,
	  vector<VALUETYPE> &		virial,
	  const vector<VALUETYPE> &	coord,
	  const vector<VALUETYPE> &	charge,
	  const vector<int> &		atype,
	  const SimulationRegion<VALUETYPE> &	region, 
	  const vector<int > &		elist)
{
  for (unsigned _ = 0; _ < elist.size(); _ += 2){
    int ii = elist[_];
    int jj = elist[_+1];
    VALUETYPE diff[3];
    region.diffNearestNeighbor (&coord[ii*3], &coord[jj*3], diff);      
    VALUETYPE r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];    
    VALUETYPE ae, af;
    ex_inner (ae, af, r2);
    // VALUETYPE ae1, af1;
    // zm_tab.tb_inner (ae1, af1, r2);
    // cout << ae << " " << ae1 << endl;
    {
      VALUETYPE qiqj = charge[ii] * charge[jj] * UnitManager::ElectrostaticConvertion;
      ae *= qiqj;
      af *= qiqj;
    }
    for (int dd = 0; dd < 3; ++dd) force[ii*3+dd] -= af * diff[dd];
    for (int dd = 0; dd < 3; ++dd) force[jj*3+dd] += af * diff[dd];    
    ener -= ae;
    for (int dd0 = 0; dd0 < 3; ++dd0){
      for (int dd1 = 0; dd1 < 3; ++dd1){
	virial[dd0*3+dd1] += 0.5 * diff[dd0] * af * diff[dd1];
      }
    }
  }
}



