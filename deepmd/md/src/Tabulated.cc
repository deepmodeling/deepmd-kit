#include "UnitManager.h"
#include "Tabulated.h"
#include "common.h"
#include <cmath>
#include <iostream>

Tabulated::
Tabulated (const VALUETYPE rc,
	   const VALUETYPE hh,
	   const vector<VALUETYPE> & tab) 
{
  reinit (rc, hh, tab) ;
}

void 
Tabulated::
reinit (const VALUETYPE rc,
	const VALUETYPE hh,
	const vector<VALUETYPE> & tab)
{
  int numbFunc = 1;
  int stride = numbFunc * 4;
  int mystride = numbFunc * 2;
  unsigned tableLength = tab.size() / mystride;

  hi = 1./hh;
  rc2 = rc * rc;

  data.resize (tableLength * stride);
  
  int ii;
  for (ii = 0; ii < tableLength-1; ++ii){
      const double & v0 (tab[ii*mystride + 0]);
      const double & f0 (tab[ii*mystride + 1]);
      const double & v1 (tab[(ii+1)*mystride + 0]);
      const double & f1 (tab[(ii+1)*mystride + 1]);
      VALUETYPE &dv (data[ii*stride + 0]);
      VALUETYPE &df (data[ii*stride + 1]);
      VALUETYPE &dg (data[ii*stride + 2]);
      VALUETYPE &dh (data[ii*stride + 3]);
      dv = v0;
      df = -f0 * hh;
      dg =  3*(v1 - v0) + (f1 + 2*f0)*hh;
      dh = -2*(v1 - v0) - (f1 +   f0)*hh;
  }
  {
    const double & v0 (tab[ii*mystride + 0]);
    const double & f0 (tab[ii*mystride + 1]);
    VALUETYPE &dv (data[ii*stride + 0]);
    VALUETYPE &df (data[ii*stride + 1]);
    VALUETYPE &dg (data[ii*stride + 2]);
    VALUETYPE &dh (data[ii*stride + 3]);
    dv = v0;
    df = -f0 * hh;
    dg = 0; 
    dh = 0;
  }
}


inline void
Tabulated::
compute_posi (int & idx, 
	      VALUETYPE & eps,
	      const VALUETYPE & rr)
{
  VALUETYPE rt = rr * hi;
  idx = int(rt);
  eps = rt - idx;
}

inline void
Tabulated::
tb_inner (VALUETYPE & ae,
	  VALUETYPE & af,
	  const VALUETYPE & r2)
{
  if (r2 > rc2) {
    ae = af = 0;
    return;
  }

  VALUETYPE rr = sqrt(r2);
  int idx;
  VALUETYPE eps;
  compute_posi (idx, eps, rr);
  idx *= 4;  

  VALUETYPE table_param[4];
  for (int ii = 0; ii < 4; ++ii){
    table_param[ii] = data[ii+idx];
  }
  const VALUETYPE &Y(table_param[0]);
  const VALUETYPE &F(table_param[1]);
  const VALUETYPE &G(table_param[2]);
  const VALUETYPE &H(table_param[3]);

  VALUETYPE Heps = eps * H;
  VALUETYPE Fp = (F + eps * (G + Heps));
  VALUETYPE FF = (Fp + (eps * (G + (Heps + Heps))));

  af = FF * hi;
  af = - af / rr;  
  ae = (Y + (eps * Fp));
}

void
Tabulated::
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
	tb_inner (ae, af, r2);
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
}

void
Tabulated::
compute (VALUETYPE &			ener,
	 vector<VALUETYPE> &		force,
	 vector<VALUETYPE> &		virial,
	 const vector<VALUETYPE> &	coord,
	 const vector<VALUETYPE> &	charge,
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
	tb_inner (ae, af, r2);
	{
	  VALUETYPE qiqj = charge[ii] * charge[jj] * UnitManager::ElectrostaticConvertion;
	  ae *= qiqj;
	  af *= qiqj;
	}
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
}


