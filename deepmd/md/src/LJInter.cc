#include "common.h"
#include "LJInter.h"
#include <cmath>

LJInter::
LJInter (const VALUETYPE & c6_,
	 const VALUETYPE & c12_,
	 const VALUETYPE & rc_)
    : c6(6. * c6_), c12(12. * c12_), rc(rc_), rc2 (rc * rc)
{
  one_over_6 = 1./6.;
  one_over_12 = 1./12.;
  VALUETYPE rc6 = rc2 * rc2 * rc2;
  one_over_rc6 = 1./rc6;
  one_over_rc12 = 1./rc6/rc6;
}

void 
LJInter::
lj_inner (VALUETYPE & ae,
	  VALUETYPE & af,
	  const VALUETYPE & r2)
{
  VALUETYPE rinv = 1./sqrt(r2);
  VALUETYPE rinv2 = rinv * rinv;
  VALUETYPE rinv6   = rinv2 * rinv2 * rinv2;
  VALUETYPE vvdw6   = c6 * rinv6;
  VALUETYPE vvdw12  = c12 * rinv6 * rinv6;
  ae = (vvdw12 - c12 * one_over_rc12) * one_over_12 - (vvdw6  - c6  * one_over_rc6 ) * one_over_6;
  af = (vvdw12 - vvdw6) * rinv2;  
}

void
LJInter::
compute (VALUETYPE &			ener,
	 vector<VALUETYPE> &		force,
	 vector<VALUETYPE> &		virial,
	 const vector<VALUETYPE> &	coord,
	 const vector<int> &		atype,
	 const SimulationRegion<VALUETYPE> &	region, 
	 const vector<vector<int > > &	nlist)
{
  for (unsigned ii = 0; ii < nlist.size(); ++ii){
    for (unsigned _ = 0; _ < nlist[ii].size(); ++_){
      int jj = nlist[ii][_];
      if (jj < ii) continue;
      VALUETYPE diff[3];
      region.diffNearestNeighbor (&coord[ii*3], &coord[jj*3], diff);      
      VALUETYPE r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
      if (r2 < rc2) {
	VALUETYPE ae, af;
	lj_inner (ae, af, r2);
	for (int dd = 0; dd < 3; ++dd) force[ii*3+dd] += af * diff[dd];
	for (int dd = 0; dd < 3; ++dd) force[jj*3+dd] -= af * diff[dd];
	ener += ae;
	for (int dd0 = 0; dd0 < 3; ++dd0){
	  for (int dd1 = 0; dd1 < 3; ++dd1){
	    virial[dd0*3+dd1] -= 0.5 * diff[dd0] * af * diff[dd1];
	  }
	}
      }      
    }
  }

  // for (int ii = 0; ii < natoms; ++ii){
  //   for (int jj = ii+1; jj < natoms; ++jj){
  //     VALUETYPE diff[3];
  //     for (int dd = 0; dd < 3; ++dd) diff[dd] = coord[ii*3+dd] - coord[jj*3+dd];
  //     diff_pbc (diff, box);
  //     VALUETYPE r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
  //     if (r2 < rc2) {
  // 	VALUETYPE ae, af;
  // 	lj_inner (ae, af, r2);
  // 	for (int dd = 0; dd < 3; ++dd) force[ii*3+dd] += af * diff[dd];
  // 	for (int dd = 0; dd < 3; ++dd) force[jj*3+dd] -= af * diff[dd];
  // 	ener += ae;
  //     }
  //   }
  // }
}

