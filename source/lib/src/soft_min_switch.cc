#include <iostream>
#include <cmath>
#include "soft_min_switch.h"
#include "switcher.h"

template <typename FPTYPE>
void deepmd::soft_min_switch_cpu(
    FPTYPE * sw_value,
    FPTYPE * sw_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int & nloc,
    const int & nnei, 
    const FPTYPE & alpha,
    const FPTYPE & rmin,
    const FPTYPE & rmax)
{
  // fill results with 0
  for (int ii = 0; ii < nloc; ++ii){
    sw_value[ii] = 0;
  }
  for (int ii = 0; ii < nloc * nnei; ++ii){
    sw_deriv[ii * 3 + 0] = 0;
    sw_deriv[ii * 3 + 1] = 0;
    sw_deriv[ii * 3 + 2] = 0;
  }
  // compute force of a frame      
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
    FPTYPE aa = 0;
    FPTYPE bb = 0;
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist [i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int rij_idx_shift = (i_idx * nnei + jj) * 3;
      FPTYPE dr[3] = {
	rij[rij_idx_shift + 0],
	rij[rij_idx_shift + 1],
	rij[rij_idx_shift + 2]
      };
      FPTYPE rr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
      FPTYPE rr = sqrt(rr2);
      FPTYPE ee = exp(-rr / alpha);
      aa += ee;
      bb += rr * ee;
    }
    FPTYPE smin = bb / aa;
    FPTYPE vv, dd;
    spline5_switch(vv, dd, smin, rmin, rmax);
    // value of switch
    sw_value[i_idx] = vv;
    // deriv of switch distributed as force
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist [i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int rij_idx_shift = (ii * nnei + jj) * 3;
      FPTYPE dr[3] = {
	rij[rij_idx_shift + 0],
	rij[rij_idx_shift + 1],
	rij[rij_idx_shift + 2]
      };
      FPTYPE rr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
      FPTYPE rr = sqrt(rr2);
      FPTYPE ee = exp(-rr / alpha);
      FPTYPE pref_c = (1./rr - 1./alpha) * ee ;
      FPTYPE pref_d = 1./(rr * alpha) * ee;
      FPTYPE ts;
      ts = dd / (aa * aa) * (aa * pref_c + bb * pref_d);
      sw_deriv[rij_idx_shift + 0] += ts * dr[0];
      sw_deriv[rij_idx_shift + 1] += ts * dr[1];
      sw_deriv[rij_idx_shift + 2] += ts * dr[2];
      // std::cout << ii << " "  << jj << " " << j_idx << "   "
      //      << vv << " " 
      //      << sw_deriv[rij_idx_shift+0) << " " 
      //      << sw_deriv[rij_idx_shift+1) << " " 
      //      << sw_deriv[rij_idx_shift+2) << " " 
      //      << std::endl;
    }
  }
}

template
void deepmd::soft_min_switch_cpu<double>(
    double * sw_value,
    double * sw_deriv,
    const double * rij,
    const int * nlist,
    const int & nloc,
    const int & nnei, 
    const double & alpha,
    const double & rmin,
    const double & rmax);

template
void deepmd::soft_min_switch_cpu<float>(
    float * sw_value,
    float * sw_deriv,
    const float * rij,
    const int * nlist,
    const int & nloc,
    const int & nnei, 
    const float & alpha,
    const float & rmin,
    const float & rmax);

