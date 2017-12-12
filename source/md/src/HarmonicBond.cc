#include "HarmonicBond.h"
#include "common.h"
#include <cmath>
#include <iostream> 

HarmonicBond::
HarmonicBond (const VALUETYPE & kk_,
	      const VALUETYPE & bb_)
    : kk(kk_), bb(bb_)
{
}

void 
HarmonicBond::
hb_inner (VALUETYPE & ae,
	  VALUETYPE & af,
	  const VALUETYPE & r1)
{
  VALUETYPE diff = r1 - bb;
  // cout << bb << " " << r1 << endl;
  VALUETYPE pdiff = kk * diff;
  af = - pdiff / r1;
  ae = 0.5 * pdiff * diff;
}

void
HarmonicBond::
compute (VALUETYPE &			ener,
	 vector<VALUETYPE> &		force,
	 vector<VALUETYPE> &		virial,
	 const vector<VALUETYPE> &	coord,
	 const vector<int> &		atype,
	 const SimulationRegion<VALUETYPE> &	region, 
	 const vector<int > &		blist)
{
  // all set zeros
  for (unsigned _ = 0; _ < blist.size(); _ += 2){
    int ii = blist[_];
    int jj = blist[_+1];
    VALUETYPE diff[3];
    region.diffNearestNeighbor (&coord[ii*3], &coord[jj*3], diff);      
    VALUETYPE r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    VALUETYPE r1 = sqrt(r2);
    VALUETYPE ae, af;
    hb_inner (ae, af, r1);
    for (int dd = 0; dd < 3; ++dd) force[ii*3+dd] += af * diff[dd];
    for (int dd = 0; dd < 3; ++dd) force[jj*3+dd] -= af * diff[dd];    
    ener += ae;
    for (int dd0 = 0; dd0 < 3; ++dd0) {
      for (int dd1 = 0; dd1 < 3; ++dd1) {
	virial[dd0*3+dd1] -= 0.5 * diff[dd0] * af * diff[dd1];
      }
    }
  }
}

