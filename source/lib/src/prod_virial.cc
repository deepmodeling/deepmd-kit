#include <iostream>
#include <stdexcept>
#include <cstring>
#include "prod_virial.h"

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
prod_virial_a_cpu(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
{
  const int ndescrpt = 4 * nnei;

  for (int ii = 0; ii < 9; ++ ii){
    virial[ii] = 0.;
  }
  for (int ii = 0; ii < 9 * nall; ++ ii){
    atom_virial[ii] = 0.;
  }

  // compute virial of a frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;

    // deriv wrt neighbors
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int aa_start, aa_end;
      make_index_range (aa_start, aa_end, jj, nnei);
      for (int aa = aa_start; aa < aa_end; ++aa) {
	FPTYPE pref = -1.0 * net_deriv[i_idx * ndescrpt + aa];
	for (int dd0 = 0; dd0 < 3; ++dd0){
	  for (int dd1 = 0; dd1 < 3; ++dd1){
	    FPTYPE tmp_v = pref * rij[i_idx * nnei * 3 + jj * 3 + dd1] *  env_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd0];
	    virial[dd0 * 3 + dd1] -= tmp_v;
	    atom_virial[j_idx * 9 + dd0 * 3 + dd1] -= tmp_v;
	  }
	}
      }
    }
  }  
}

template
void 
deepmd::
prod_virial_a_cpu<double>(
    double * virial, 
    double * atom_virial, 
    const double * net_deriv, 
    const double * env_deriv, 
    const double * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei) ;

template
void 
deepmd::
prod_virial_a_cpu<float>(
    float * virial, 
    float * atom_virial, 
    const float * net_deriv, 
    const float * env_deriv, 
    const float * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei) ;


template<typename FPTYPE>
void 
deepmd::
prod_virial_r_cpu(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
{
  const int ndescrpt = nnei;

  for (int ii = 0; ii < 9; ++ ii){
    virial[ii] = 0.;
  }
  for (int ii = 0; ii < 9 * nall; ++ ii){
    atom_virial[ii] = 0.;
  }

  // compute virial of a frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;

    // deriv wrt neighbors
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      FPTYPE pref = -1.0 * net_deriv[i_idx * ndescrpt + jj];
      for (int dd0 = 0; dd0 < 3; ++dd0){
	for (int dd1 = 0; dd1 < 3; ++dd1){
	  FPTYPE tmp_v = pref * rij[i_idx * nnei * 3 + jj * 3 + dd1] *  env_deriv[i_idx * ndescrpt * 3 + jj * 3 + dd0];
	  virial[dd0 * 3 + dd1] -= tmp_v;
	  atom_virial[j_idx * 9 + dd0 * 3 + dd1] -= tmp_v;
	}
      }
    }
  }
}

template
void 
deepmd::
prod_virial_r_cpu<double>(
    double * virial, 
    double * atom_virial, 
    const double * net_deriv, 
    const double * env_deriv, 
    const double * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

template
void 
deepmd::
prod_virial_r_cpu<float>(
    float * virial, 
    float * atom_virial, 
    const float * net_deriv, 
    const float * env_deriv, 
    const float * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);
