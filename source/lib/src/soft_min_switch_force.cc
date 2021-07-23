#include "soft_min_switch_force.h"
#include <iostream>

template<typename FPTYPE>
void deepmd::soft_min_switch_force_cpu(
    FPTYPE * force, 
    const FPTYPE * du, 
    const FPTYPE * sw_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
//
//	force :		nall * 3
//	du :		nloc
//	sw_deriv :	nloc * nnei * 3
//
{
  // set zeros
  for (int ii = 0; ii < nall; ++ii){
    int i_idx = ii;
    force[i_idx * 3 + 0] = 0;
    force[i_idx * 3 + 1] = 0;
    force[i_idx * 3 + 2] = 0;
  }
  // compute force of a frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;	
    for (int jj = 0; jj < nnei; ++jj){	  
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int rij_idx_shift = (ii * nnei + jj) * 3;
      force[i_idx * 3 + 0] += du[i_idx] * sw_deriv[rij_idx_shift + 0];
      force[i_idx * 3 + 1] += du[i_idx] * sw_deriv[rij_idx_shift + 1];
      force[i_idx * 3 + 2] += du[i_idx] * sw_deriv[rij_idx_shift + 2];
      force[j_idx * 3 + 0] -= du[i_idx] * sw_deriv[rij_idx_shift + 0];
      force[j_idx * 3 + 1] -= du[i_idx] * sw_deriv[rij_idx_shift + 1];
      force[j_idx * 3 + 2] -= du[i_idx] * sw_deriv[rij_idx_shift + 2];
    }
  }  
}

template
void deepmd::soft_min_switch_force_cpu<double>(
    double * force, 
    const double * du, 
    const double * sw_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

template
void deepmd::soft_min_switch_force_cpu<float>(
    float * force, 
    const float * du, 
    const float * sw_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);
