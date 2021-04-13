#include <iostream>
#include <stdexcept>
#include <cstring>
#include "prod_force_grad.h"

inline void
make_index_range (
    int & idx_start,
    int & idx_end,
    const int & nei_idx, 
    const int & nnei) 
{
  if (nei_idx < nnei) {
    idx_start = nei_idx * 4;
    idx_end   = nei_idx * 4 + 4;
  }
  else {
    throw std::runtime_error("should no reach here");
  }
}


template<typename FPTYPE>
void 
deepmd::
prod_force_grad_a_cpu(
    FPTYPE * grad_net, 
    const FPTYPE * grad, 
    const FPTYPE * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei) 
{
  const int ndescrpt = nnei * 4;
  
  // reset the frame to 0
  for (int ii = 0; ii < nloc; ++ii){
    for (int aa = 0; aa < ndescrpt; ++aa){
      grad_net[ii * ndescrpt + aa] = 0;
    }
  }      

  // compute grad of one frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
	
    // deriv wrt center atom
    for (int aa = 0; aa < ndescrpt; ++aa){
      for (int dd = 0; dd < 3; ++dd){
	grad_net[i_idx * ndescrpt + aa] -= grad[i_idx * 3 + dd] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd];
      }
    }

    // loop over neighbors
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx >= nloc) j_idx = j_idx % nloc;
      if (j_idx < 0) continue;
      int aa_start, aa_end;
      make_index_range(aa_start, aa_end, jj, nnei);
      for (int aa = aa_start; aa < aa_end; ++aa){
	for (int dd = 0; dd < 3; ++dd){
	  grad_net[i_idx * ndescrpt + aa] += grad[j_idx * 3 + dd] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd];
	}
      }
    }
  }
}


template
void 
deepmd::
prod_force_grad_a_cpu<double>(
    double * grad_net, 
    const double * grad, 
    const double * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei) ;

template
void 
deepmd::
prod_force_grad_a_cpu<float>(
    float * grad_net, 
    const float * grad, 
    const float * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei) ;



template<typename FPTYPE>
void 
deepmd::
prod_force_grad_r_cpu(
    FPTYPE * grad_net, 
    const FPTYPE * grad, 
    const FPTYPE * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei) 
//
//	grad_net:	nloc x ndescrpt
//	grad:		nloc x 3
//	env_deriv:	nloc x ndescrpt x 3
//	nlist:		nloc x nnei
//
{
  const int ndescrpt = nnei * 1;
  
  // reset the frame to 0
  for (int ii = 0; ii < nloc; ++ii){
    for (int aa = 0; aa < ndescrpt; ++aa){
      grad_net[ii * ndescrpt + aa] = 0;
    }
  }      

  // compute grad of one frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
	
    // deriv wrt center atom
    for (int aa = 0; aa < ndescrpt; ++aa){
      for (int dd = 0; dd < 3; ++dd){
	grad_net[i_idx * ndescrpt + aa] -= grad[i_idx * 3 + dd] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd];
      }
    }

    // loop over neighbors
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx >= nloc) j_idx = j_idx % nloc;
      if (j_idx < 0) continue;
      for (int dd = 0; dd < 3; ++dd){
	grad_net[i_idx * ndescrpt + jj] += grad[j_idx * 3 + dd] * env_deriv[i_idx * ndescrpt * 3 + jj * 3 + dd];
      }
    }
  }
}

template
void 
deepmd::
prod_force_grad_r_cpu<double>(
    double * grad_net, 
    const double * grad, 
    const double * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei) ;

template
void
deepmd::
prod_force_grad_r_cpu<float>(
    float * grad_net, 
    const float * grad, 
    const float * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nnei) ;



