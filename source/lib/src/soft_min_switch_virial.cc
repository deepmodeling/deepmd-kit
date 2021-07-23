#include "soft_min_switch_virial.h"
#include <iostream>

template<typename FPTYPE>
void deepmd::soft_min_switch_virial_cpu(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * du, 
    const FPTYPE * sw_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
//
//	virial :	9
//	atom_virial :	nall * 9
//	du :		nloc
//	sw_deriv :	nloc * nnei * 3
//
{
  for (int ii = 0; ii < 9; ++ ii){
    virial[ii] = 0.;
  }
  for (int ii = 0; ii < 9 * nall; ++ ii){
    atom_virial[ii] = 0.;
  }

  // compute virial of a frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
    // loop over neighbors
    for (int jj = 0; jj < nnei; ++jj){	  
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int rij_idx_shift = (ii * nnei + jj) * 3;
      for (int dd0 = 0; dd0 < 3; ++dd0){
	for (int dd1 = 0; dd1 < 3; ++dd1){
	  FPTYPE tmp_v = du[i_idx] * sw_deriv[rij_idx_shift + dd0] * rij[rij_idx_shift + dd1];
	  virial[dd0 * 3 + dd1] -= tmp_v;		  
	  atom_virial[j_idx * 9 + dd0 * 3 + dd1] -= tmp_v;
	}
      }
    }
  }  
}


template
void deepmd::soft_min_switch_virial_cpu<double>(
    double * virial, 
    double * atom_virial, 
    const double * du, 
    const double * sw_deriv, 
    const double * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

template
void deepmd::soft_min_switch_virial_cpu<float>(
    float * virial, 
    float * atom_virial, 
    const float * du, 
    const float * sw_deriv, 
    const float * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);



