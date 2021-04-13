#include "soft_min_switch_force_grad.h"
#include <iostream>

template<typename FPTYPE>
void deepmd::soft_min_switch_force_grad_cpu(
    FPTYPE * grad_net, 
    const FPTYPE * grad,
    const FPTYPE * sw_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei)
//
//	grad_net :	nloc
//	grad :		nloc * 3
//	sw_deriv :	nloc * nnei * 3
//	nlist:		nloc * nnei
//
{
  // reset the frame to 0
  for (int ii = 0; ii < nloc; ++ii){
    grad_net[ii] = 0;
  }      

  // compute grad of one frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
    // deriv wrt center atom	
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist [i_idx * nnei + jj];
      if (j_idx >= nloc) j_idx = j_idx % nloc;
      if (j_idx < 0) continue;
      int rij_idx_shift = (ii * nnei + jj) * 3;
      grad_net[i_idx] += grad[i_idx * 3 + 0] * sw_deriv[rij_idx_shift + 0];
      grad_net[i_idx] += grad[i_idx * 3 + 1] * sw_deriv[rij_idx_shift + 1];
      grad_net[i_idx] += grad[i_idx * 3 + 2] * sw_deriv[rij_idx_shift + 2];
      grad_net[i_idx] -= grad[j_idx * 3 + 0] * sw_deriv[rij_idx_shift + 0];
      grad_net[i_idx] -= grad[j_idx * 3 + 1] * sw_deriv[rij_idx_shift + 1];
      grad_net[i_idx] -= grad[j_idx * 3 + 2] * sw_deriv[rij_idx_shift + 2];
    }
  }
}

template
void deepmd::soft_min_switch_force_grad_cpu<double>(
    double * grad_net, 
    const double * grad,
    const double * sw_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei);

template
void deepmd::soft_min_switch_force_grad_cpu<float>(
    float * grad_net, 
    const float * grad,
    const float * sw_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei);
