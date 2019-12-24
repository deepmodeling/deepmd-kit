#pragma once

#include<algorithm>
#include<cassert>
#include<omp.h>

#include "SimulationRegion.h"

// 8.988e9 / pc.electron_volt / pc.angstrom * (1.602e-19)**2
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
       const SimulationRegion<double>&		region, 
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

  // number of threads
  int nthreads = 1;
#pragma omp parallel 
  {
    if (0 == omp_get_thread_num()) {
      nthreads = omp_get_num_threads();
    }
  }

  // K grid
  vector<int> KK(3);
  int totK = 1;
  cmpt_k<VALUETYPE>(KK, region, param);
  for (int dd = 0; dd < 3; ++dd){
    totK *= (KK[dd]+1);
  }  
  int stride[3];
  for (int dd = 0; dd < 3; ++dd) stride[dd] = KK[dd]+1;
  
  // compute the sq
  vector<vector<VALUETYPE> > thread_sqr(nthreads), thread_sqi(nthreads);
  for (int ii = 0; ii < nthreads; ++ii){
    thread_sqr[ii].resize(totK, static_cast<VALUETYPE>(0));
    thread_sqi[ii].resize(totK, static_cast<VALUETYPE>(0));
  }  
  // firstly loop over particles then loop over m
#pragma omp parallel for num_threads(nthreads)
  for (int ii = 0; ii < natoms; ++ii){
    int thread_id = omp_get_thread_num();
    double ir[3];
    double tmpcoord[3] = {coord[ii*3], coord[ii*3+1], coord[ii*3+2]};
    region.phys2Inter(ir, tmpcoord);
    for (int mm0 = -KK[0]/2; mm0 <= KK[0]/2; ++mm0){
      double mr[3];
      mr[0] = ir[0] * mm0;      
      int shift0 = (mm0 + KK[0]/2) * stride[1] * stride[2];
      for (int mm1 = -KK[1]/2; mm1 <= KK[1]/2; ++mm1){
	mr[1] = ir[1] * mm1;
	int shift1 = (mm1 + KK[1]/2) * stride[2];
	for (int mm2 = -KK[2]/2; mm2 <= KK[2]/2; ++mm2){
	  if (mm0 == 0 && mm1 == 0 && mm2 == 0) continue;
	  int mc = shift0 + shift1 + mm2 + KK[2]/2;
	  mr[2] = ir[2] * mm2;
	  double mdotr = 2. * M_PI * (mr[0]+mr[1]+mr[2]);
	  thread_sqr[thread_id][mc] += charge[ii] * cos(mdotr);
	  thread_sqi[thread_id][mc] += charge[ii] * sin(mdotr);
	}
      }
    }
  }
  VALUETYPE * sqr = new VALUETYPE[totK];
  VALUETYPE * sqi = new VALUETYPE[totK];
  for (int ii = 0; ii < totK; ++ii){
    sqr[ii] = static_cast<VALUETYPE>(0);
    sqi[ii] = static_cast<VALUETYPE>(0);
    for (int jj = 0; jj < nthreads; ++jj){
      sqr[ii] += thread_sqr[jj][ii];
      sqi[ii] += thread_sqi[jj][ii];
    }
  }  

  // get rbox
  VALUETYPE rec_box[9];
  const double * rec_box_ = region.getRecBoxTensor();
  for (int ii = 0; ii < 9; ++ii){
    rec_box[ii] = static_cast<VALUETYPE>(rec_box_[ii]);
  }
  
  vector<VALUETYPE> thread_ener(nthreads, 0.);
  vector<vector<VALUETYPE> > thread_force(nthreads);
  vector<vector<VALUETYPE> > thread_virial(nthreads);
  for (int ii = 0; ii < nthreads; ++ii){
    thread_force[ii].resize(natoms * 3, 0.);
    thread_virial[ii].resize(9, 0.);
  }
  // calculate ener, force and virial
  // firstly loop over particles then loop over m  
#pragma omp parallel for num_threads(nthreads)
  for (int mc = 0; mc < totK; ++mc){
    int thread_id = omp_get_thread_num();
    int mm0 = mc / (stride[1] * stride[2]);
    int left = mc - mm0 * stride[1] * stride[2];
    int mm1 = left / stride[2];
    int mm2 = left - mm1 * stride[2];
    mm0 -= KK[0]/2;
    mm1 -= KK[1]/2;
    mm2 -= KK[2]/2;
  // for (int mm0 = -KK[0]/2; mm0 <= KK[0]/2; ++mm0){
  //   int shift0 = (mm0 + KK[0]/2) * stride[1] * stride[2];
  //   for (int mm1 = -KK[1]/2; mm1 <= KK[1]/2; ++mm1){
  //     int shift1 = (mm1 + KK[1]/2) * stride[2];
  //     for (int mm2 = -KK[2]/2; mm2 <= KK[2]/2; ++mm2){
  // 	int mc = shift0 + shift1 + mm2 + KK[2]/2;
	if (mm0 == 0 && mm1 == 0 && mm2 == 0) continue;
	// \bm m and \vert m \vert^2
	VALUETYPE rm[3] = {0,0,0};	  
	rm[0] += mm0 * rec_box[0*3+0];
	rm[1] += mm0 * rec_box[1*3+0];
	rm[2] += mm0 * rec_box[2*3+0];
	rm[0] += mm1 * rec_box[0*3+1];
	rm[1] += mm1 * rec_box[1*3+1];
	rm[2] += mm1 * rec_box[2*3+1];
	rm[0] += mm2 * rec_box[0*3+2];
	rm[1] += mm2 * rec_box[1*3+2];
	rm[2] += mm2 * rec_box[2*3+2];
	VALUETYPE nmm2 = rm[0] * rm[0] + rm[1] * rm[1] + rm[2] * rm[2];
	// energy
	VALUETYPE expnmm2 = exp(- M_PI * M_PI * nmm2 / (param.beta * param.beta)) / nmm2;
	VALUETYPE eincr = expnmm2 * (sqr[mc] * sqr[mc] + sqi[mc] * sqi[mc]);
	thread_ener[thread_id] += eincr;
	// virial
	VALUETYPE vpref = -2. * (1. + M_PI * M_PI * nmm2 / (param.beta * param.beta)) / nmm2;
	for (int dd0 = 0; dd0 < 3; ++dd0){
	  for (int dd1 = 0; dd1 < 3; ++dd1){	    
	    VALUETYPE tmp = vpref * rm[dd0] * rm[dd1];
	    if (dd0 == dd1) tmp += 1;
	    thread_virial[thread_id][dd0*3+dd1] += eincr * tmp;
	  }
	}
	// force
	for (int ii = 0; ii < natoms; ++ii){
	  VALUETYPE mdotr = - 2. * M_PI * (coord[ii*3+0]*rm[0] + coord[ii*3+1]*rm[1] + coord[ii*3+2]*rm[2]);
	  VALUETYPE tmpr = charge[ii] * cos(mdotr);
	  VALUETYPE tmpi = charge[ii] * sin(mdotr);
	  VALUETYPE cc = 4. * M_PI * (tmpr * sqi[mc] + tmpi * sqr[mc]) * expnmm2;
	  thread_force[thread_id][ii*3+0] -= rm[0] * cc;
	  thread_force[thread_id][ii*3+1] -= rm[1] * cc;
	  thread_force[thread_id][ii*3+2] -= rm[2] * cc;
	}
    //   }
    // }
  }
  // reduce thread results
  for (int ii = 0; ii < nthreads; ++ii){
    ener += thread_ener[ii];
  }
  for (int jj = 0; jj < 9; ++jj){
    for (int ii = 0; ii < nthreads; ++ii){
      virial[jj] += thread_virial[ii][jj];
    }
  }
  for (int jj = 0; jj < natoms * 3; ++jj){
    for (int ii = 0; ii < nthreads; ++ii){
      force[jj] += thread_force[ii][jj];
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

