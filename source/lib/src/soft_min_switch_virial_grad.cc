#include "soft_min_switch_virial_grad.h"

template<typename FPTYPE>
void deepmd::soft_min_switch_virial_grad_cpu(
    FPTYPE * grad_net, 
    const FPTYPE * grad,
    const FPTYPE * sw_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nnei)
//
//	grad_net:	nloc
//	grad:		9
//	sw_deriv:	nloc * nnei * 3
//	rij:		nloc * nnei * 3
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
    // loop over neighbors
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int rij_idx_shift = (ii * nnei + jj) * 3;
      for (int dd0 = 0; dd0 < 3; ++dd0){
	for (int dd1 = 0; dd1 < 3; ++dd1){
	  grad_net[i_idx] -= 
	      grad[dd0 * 3 + dd1] * sw_deriv[rij_idx_shift + dd0] * rij[rij_idx_shift + dd1];
	}
      }
    }
  }
}

template
void deepmd::soft_min_switch_virial_grad_cpu<double>(
    double * grad_net, 
    const double * grad,
    const double * sw_deriv, 
    const double * rij, 
    const int * nlist, 
    const int nloc, 
    const int nnei);

template
void deepmd::soft_min_switch_virial_grad_cpu<float>(
    float * grad_net, 
    const float * grad,
    const float * sw_deriv, 
    const float * rij, 
    const int * nlist, 
    const int nloc, 
    const int nnei);



