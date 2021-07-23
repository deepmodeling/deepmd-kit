#include <stdexcept>
#include <cstring>
#include "prod_virial_grad.h"

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
prod_virial_grad_a_cpu(
    FPTYPE * grad_net,
    const FPTYPE * grad,
    const FPTYPE * env_deriv,
    const FPTYPE * rij,
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
	
    // loop over neighbors
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int aa_start, aa_end;
      make_index_range (aa_start, aa_end, jj, nnei);
      for (int aa = aa_start; aa < aa_end; ++aa){
	for (int dd0 = 0; dd0 < 3; ++dd0){
	  for (int dd1 = 0; dd1 < 3; ++dd1){
	    grad_net[i_idx * ndescrpt + aa] -= 
		-1.0 * grad[dd0 * 3 + dd1] * rij[i_idx * nnei * 3 + jj * 3 + dd1] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd0];
	  }
	}
      }
    }
  }
}


template
void 
deepmd::
prod_virial_grad_a_cpu<double>(
    double * grad_net,
    const double * grad,
    const double * env_deriv,
    const double * rij,
    const int * nlist,
    const int nloc,
    const int nnei);

template
void 
deepmd::
prod_virial_grad_a_cpu<float>(
    float * grad_net,
    const float * grad,
    const float * env_deriv,
    const float * rij,
    const int * nlist,
    const int nloc,
    const int nnei);


template<typename FPTYPE>
void 
deepmd::
prod_virial_grad_r_cpu(
    FPTYPE * grad_net,
    const FPTYPE * grad,
    const FPTYPE * env_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int nloc,
    const int nnei)
//
//	grad_net:	nloc x ndescrpt
//	grad:		9
//	env_deriv:	nloc x ndescrpt x 3
//	rij:		nloc x nnei x 3
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
	
    // loop over neighbors
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];	  
      if (j_idx < 0) continue;
      for (int dd0 = 0; dd0 < 3; ++dd0){
	for (int dd1 = 0; dd1 < 3; ++dd1){
	  grad_net[i_idx * ndescrpt + jj] -= 
	      -1.0 * grad[dd0 * 3 + dd1] * rij[i_idx * nnei * 3 + jj * 3 + dd1] * env_deriv[i_idx * ndescrpt * 3 + jj * 3 + dd0];
	}
      }
    }
  }
}


template
void 
deepmd::
prod_virial_grad_r_cpu<double>(
    double * grad_net,
    const double * grad,
    const double * env_deriv,
    const double * rij,
    const int * nlist,
    const int nloc,
    const int nnei);

template
void 
deepmd::
prod_virial_grad_r_cpu<float>(
    float * grad_net,
    const float * grad,
    const float * env_deriv,
    const float * rij,
    const int * nlist,
    const int nloc,
    const int nnei);

