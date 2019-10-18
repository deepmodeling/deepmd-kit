#pragma once

#include<algorithm>
#include<cassert>

#include "SimulationRegion.h"

const double ElectrostaticConvertion = 14.39964535475696995031;

template <typename VALUETYPE>
struct EwaldParameters 
{
  VALUETYPE rcut = 6.0;
  VALUETYPE beta = 2;
  VALUETYPE spacing = 4;
};

template<typename VALUETYPE> 
VALUETYPE
dir_err_esti(const VALUETYPE & test_q,
	     const VALUETYPE & c2,
	     const VALUETYPE & nn,
	     const EwaldParameters<VALUETYPE> & param) 
{
  const VALUETYPE & rcut = param.rcut;
  const VALUETYPE & beta = param.beta;
  const VALUETYPE rho_q2 = c2/nn;  
  VALUETYPE sum = 2 * test_q 
      * sqrt (rho_q2 / rcut)
      * exp (- beta*beta*rcut*rcut) * ElectrostaticConvertion;
  return sum;
}

template<typename VALUETYPE> 
VALUETYPE
rec_err_esti(const VALUETYPE & test_q,
	     const VALUETYPE & c2,
	     const VALUETYPE & nn,
	     const EwaldParameters<VALUETYPE>&	param,
	     const SimulationRegion<double>&	region) 
{
  const VALUETYPE & beta = param.beta;
  vector<int> KK;
  cmpt_k(KK, region, param);
  const double * rec_box = region.getRecBoxTensor();
  double sum = 0;
  int BD[3];
  for (int dd = 0; dd < 3; ++dd){
    BD[dd] = KK[dd]/2 + 10;
  }
  int mm[3];
  for (mm[0] = -BD[0]; mm[0] <= BD[0]; ++mm[0]){
    for (mm[1] = -BD[1]; mm[1] <= BD[1]; ++mm[1]){
      for (mm[2] = -BD[2]; mm[2] <= BD[2]; ++mm[2]){
        if (mm[0] >= - int(KK[0])/2 && mm[0] <= int(KK[0])/2 &&
            mm[1] >= - int(KK[1])/2 && mm[1] <= int(KK[1])/2 &&
            mm[2] >= - int(KK[2])/2 && mm[2] <= int(KK[2])/2) continue;
	VALUETYPE rm[3] = {0,0,0};	  
	for (int dd = 0; dd < 3; ++dd){
	  rm[0] += mm[dd] * rec_box[dd*3+0];
	  rm[1] += mm[dd] * rec_box[dd*3+1];
	  rm[2] += mm[dd] * rec_box[dd*3+2];
	}
	VALUETYPE mm2 = rm[0] * rm[0] + rm[1] * rm[1] + rm[2] * rm[2];
        sum += exp (-2 * M_PI * M_PI / beta / beta * mm2) / mm2;
      }
    }
  }
  VALUETYPE vol = region.getVolume();
  // cout << "sum: " << sqrt(sum) 
  //      << " KK: " << KK[0] 
  //      << " rbox: " << rec_box[0] 
  //      << " c2: " << c2 
  //      << " vol: " << vol << endl;
  sum = test_q * 2 * sqrt(sum) * sqrt(c2) / vol * ElectrostaticConvertion;
  return sum;
}

template <typename VALUETYPE>
void
cmpt_k(vector<int> & KK,
       const SimulationRegion<VALUETYPE>&	region, 
       const EwaldParameters<VALUETYPE>&	param)
{
  const double * boxt_ = region.getBoxTensor();
  VALUETYPE boxt[9];
  for (int dd = 0; dd < 9; ++dd){
    boxt[dd] = static_cast<VALUETYPE>(boxt_[dd]);
  }  
  KK.resize(3);
  for (int dd = 0; dd < 3; ++dd){
    VALUETYPE ll = sqrt(MathUtilities::dot<VALUETYPE>(boxt+dd*3, boxt+dd*3));
    KK[dd] = ll / param.spacing;
    // KK[dd] should be large enough 
    if (KK[dd] * param.spacing < ll) KK[dd] += 1;
    assert(KK[dd] * param.spacing >= ll);
    // KK[dd] should be even
    if ((KK[dd] / 2) * 2 != KK[dd]) KK[dd] += 1;
    assert((KK[dd] / 2) * 2 == KK[dd]);
  }
}

// compute the reciprocal part of the Ewald sum.
// outputs: energy force virial
// inputs: coordinates charges region
template <typename VALUETYPE>
void 
EwaldReciprocal(VALUETYPE &			ener, 
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE>&	coord,
		const vector<VALUETYPE>&	charge,
		const SimulationRegion<double>& region, 
		const EwaldParameters<VALUETYPE>&	param)
{
  // natoms
  int natoms = charge.size();
  // init returns
  force.resize(natoms * 3);  
  virial.resize(9);
  ener = 0;
  fill(force.begin(), force.end(), static_cast<VALUETYPE>(0));
  fill(virial.begin(), virial.end(), static_cast<VALUETYPE>(0));
  
  vector<int> KK(3);
  int totK = 1;
  cmpt_k<VALUETYPE>(KK, region, param);
  for (int dd = 0; dd < 3; ++dd){
    totK *= (KK[dd]+1);
  }  
  
  // compute the sq
  VALUETYPE * sqr = new VALUETYPE[totK];
  VALUETYPE * sqi = new VALUETYPE[totK];
  for (int ii = 0; ii < totK; ++ii){
    sqr[ii] = static_cast<VALUETYPE>(0);
    sqi[ii] = static_cast<VALUETYPE>(0);
  }
  // firstly loop over particles then loop over m
  int mm[3];
  for (int ii = 0; ii < natoms; ++ii){
    double ir[3];
    region.phys2Inter(ir, &coord[ii*3]);
    double mr[3];
    int mc = 0;
    for (mm[0] = -KK[0]/2; mm[0] <= KK[0]/2; ++mm[0]){
      mr[0] = ir[0] * mm[0];
      for (mm[1] = -KK[1]/2; mm[1] <= KK[1]/2; ++mm[1]){
	mr[1] = ir[1] * mm[1];
	for (mm[2] = -KK[2]/2; mm[2] <= KK[2]/2; ++mm[2]){
	  if (mm[0] == 0 && mm[1] == 0 && mm[2] == 0) continue;
	  mr[2] = ir[2] * mm[2];
	  double mdotr = 2. * M_PI * (mr[0]+mr[1]+mr[2]);
	  sqr[mc] += charge[ii] * cos(mdotr);
	  sqi[mc] += charge[ii] * sin(mdotr);
	  ++mc;
	}
      }
    }
  }
  VALUETYPE rec_box[9];
  const double * rec_box_ = region.getRecBoxTensor();
  for (int ii = 0; ii < 9; ++ii){
    rec_box[ii] = static_cast<VALUETYPE>(rec_box_[ii]);
  }
  // calculate ener, force and virial
  // firstly loop over particles then loop over m
  int mc = 0;
  for (mm[0] = -KK[0]/2; mm[0] <= KK[0]/2; ++mm[0]){
    for (mm[1] = -KK[1]/2; mm[1] <= KK[1]/2; ++mm[1]){
      for (mm[2] = -KK[2]/2; mm[2] <= KK[2]/2; ++mm[2]){
	if (mm[0] == 0 && mm[1] == 0 && mm[2] == 0) continue;
	// \bm m and \vert m \vert^2
	VALUETYPE rm[3] = {0,0,0};	  
	for (int dd = 0; dd < 3; ++dd){
	  rm[0] += mm[dd] * rec_box[0*3+dd];
	  rm[1] += mm[dd] * rec_box[1*3+dd];
	  rm[2] += mm[dd] * rec_box[2*3+dd];
	  // rm[0] += mm[dd] * rec_box[dd*3+0];
	  // rm[1] += mm[dd] * rec_box[dd*3+1];
	  // rm[2] += mm[dd] * rec_box[dd*3+2];
	}
	VALUETYPE mm2 = rm[0] * rm[0] + rm[1] * rm[1] + rm[2] * rm[2];
	// energy
	VALUETYPE expmm2 = exp(- M_PI * M_PI * mm2 / (param.beta * param.beta)) / mm2;
	VALUETYPE eincr = expmm2 * (sqr[mc] * sqr[mc] + sqi[mc] * sqi[mc]);
	ener += eincr;
	// virial
	VALUETYPE vpref = -2. * (1. + M_PI * M_PI * mm2 / (param.beta * param.beta)) / mm2;
	for (int dd0 = 0; dd0 < 3; ++dd0){
	  for (int dd1 = 0; dd1 < 3; ++dd1){	    
	    VALUETYPE tmp = vpref * rm[dd0] * rm[dd1];
	    if (dd0 == dd1) tmp += 1;
	    virial[dd0*3+dd1] += eincr * tmp;
	  }
	}
	// force
	for (int ii = 0; ii < natoms; ++ii){
	  VALUETYPE mdotr = - 2. * M_PI * (coord[ii*3+0]*rm[0] + coord[ii*3+1]*rm[1] + coord[ii*3+2]*rm[2]);
	  VALUETYPE tmpr = charge[ii] * cos(mdotr);
	  VALUETYPE tmpi = charge[ii] * sin(mdotr);
	  VALUETYPE cc = 4. * M_PI * (tmpr * sqi[mc] + tmpi * sqr[mc]) * expmm2;
	  force[ii*3+0] -= rm[0] * cc;
	  force[ii*3+1] -= rm[1] * cc;
	  force[ii*3+2] -= rm[2] * cc;
	}	  
	++mc;
      }
    }
  }
  VALUETYPE vol = static_cast<VALUETYPE>(region.getVolume());
  ener /= 2 * M_PI * vol;
  ener *= ElectrostaticConvertion;
  for (int ii = 0; ii < 3*natoms; ++ii){
    force[ii] /= 2 * M_PI * vol;
    force[ii] *= ElectrostaticConvertion;
  }  
  for (int ii = 0; ii < 3*3; ++ii){
    virial[ii] /= 2 * M_PI * vol;
    virial[ii] *= ElectrostaticConvertion;
  }  
  delete[]sqr;
  delete[]sqi;
}

