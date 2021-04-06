#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include "pair_tab.h"

inline 
void _pair_tabulated_inter (
    double & ener, 
    double & fscale, 
    const double * table_info,
    const double * table_data,
    const double * dr)
{
  // info size: 3
  const double & rmin = table_info[0];
  const double & hh = table_info[1];
  const double hi = 1./hh;
  const unsigned nspline = unsigned(table_info[2] + 0.1);
  const unsigned ndata = nspline * 4;

  double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
  double rr = sqrt(r2);
  double uu = (rr - rmin) * hi;
  // std::cout << rr << " " << rmin << " " << hh << " " << uu << std::endl;
  if (uu < 0) {
    std::cerr << "coord go beyond table lower boundary" << std::endl;
    exit(1);
  }
  int idx = uu;
  if (idx >= nspline) {
    fscale = ener = 0;
    return;
  }
  uu -= idx;
  assert(idx >= 0);
  assert(uu >= 0 && uu < 1);

  const double & a3 = table_data[4 * idx + 0];
  const double & a2 = table_data[4 * idx + 1];
  const double & a1 = table_data[4 * idx + 2];
  const double & a0 = table_data[4 * idx + 3];
  
  double etmp = (a3 * uu + a2) * uu + a1;
  ener = etmp * uu + a0;
  fscale = (2. * a3 * uu + a2) * uu + etmp;
  fscale *= -hi;
}

template<typename FPTYPE>
void _pair_tab_jloop(
    FPTYPE * energy,
    FPTYPE * force,
    FPTYPE * virial,
    int & jiter,
    const int & i_idx,
    const int & nnei,
    const int & i_type_shift,
    const double * p_table_info,
    const double * p_table_data,
    const int & tab_stride,
    const FPTYPE * rij,
    const FPTYPE * scale,
    const int * type,
    const int * nlist,
    const int * natoms,
    const std::vector<int> & sel
    )
{
  const FPTYPE i_scale = scale[i_idx];
  for (int ss = 0; ss < sel.size(); ++ss){
    int j_type = ss;
    const double * cur_table_data = 
	p_table_data + (i_type_shift + j_type) * tab_stride;
    for (int jj = 0; jj < sel[ss]; ++jj){
      int j_idx = nlist[i_idx * nnei + jiter];
      if (j_idx < 0){
	jiter++;
	continue;
      }
      assert(j_type == type[j_idx]);
      double dr[3];
      for (int dd = 0; dd < 3; ++dd){
	dr[dd] = rij[(i_idx * nnei + jiter) * 3 + dd];
      }
      double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
      double ri = 1./sqrt(r2);
      double ener, fscale;
      _pair_tabulated_inter(
	  ener,
	  fscale, 
	  p_table_info, 
	  cur_table_data, 
	  dr);
      energy[i_idx] += 0.5 * ener;
      for (int dd = 0; dd < 3; ++dd) {
	force[i_idx * 3 + dd] -= fscale * dr[dd] * ri * 0.5 * i_scale;
	force[j_idx * 3 + dd] += fscale * dr[dd] * ri * 0.5 * i_scale;
      }
      for (int dd0 = 0; dd0 < 3; ++dd0) {
	for (int dd1 = 0; dd1 < 3; ++dd1) {
	  virial[i_idx * 9 + dd0 * 3 + dd1]
	      += 0.5 * fscale * dr[dd0] * dr[dd1] * ri * 0.5 * i_scale;
	  virial[j_idx * 9 + dd0 * 3 + dd1]
	      += 0.5 * fscale * dr[dd0] * dr[dd1] * ri * 0.5 * i_scale;
	}
      }
      jiter++;
    }
  }
}

inline void
_cum_sum (
    std::vector<int> & sec,
    const std::vector<int> & n_sel) {
  sec.resize (n_sel.size() + 1);
  sec[0] = 0;
  for (int ii = 1; ii < sec.size(); ++ii){
    sec[ii] = sec[ii-1] + n_sel[ii-1];
  }
}

template<typename FPTYPE>
void 
deepmd::pair_tab_cpu(
    FPTYPE * energy,
    FPTYPE * force,
    FPTYPE * virial,
    const double * p_table_info,
    const double * p_table_data,
    const FPTYPE * rij,
    const FPTYPE * scale,
    const int * type,
    const int * nlist,
    const int * natoms,
    const std::vector<int> & sel_a,
    const std::vector<int> & sel_r
    )
{
  std::vector<int> sec_a;
  std::vector<int> sec_r;
  _cum_sum(sec_a, sel_a);
  _cum_sum(sec_r, sel_r);
  const int nloc = natoms[0];
  const int nall = natoms[1];
  const int nnei = sec_a.back() + sec_r.back();
  const int ntypes = int(p_table_info[3]+0.1);
  const int nspline = p_table_info[2]+0.1;
  const int tab_stride = 4 * nspline;
  
  // fill results with 0
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
    energy[i_idx] = 0;
  }
  for (int ii = 0; ii < nall; ++ii){
    int i_idx = ii;
    force[i_idx * 3 + 0] = 0;
    force[i_idx * 3 + 1] = 0;
    force[i_idx * 3 + 2] = 0;
    for (int dd = 0; dd < 9; ++dd) {
      virial[i_idx * 9 + dd] = 0;
    }
  }
  // compute force of a frame
  int i_idx = 0;
  for (int tt = 0; tt < ntypes; ++tt) {
    for (int ii = 0; ii < natoms[2+tt]; ++ii){
      int i_type = type[i_idx];
      assert(i_type == tt) ;
      const int i_type_shift = i_type * ntypes;
      int jiter = 0;
      // a neighbor
      _pair_tab_jloop(energy,
		      force,
		      virial,
		      jiter,
		      i_idx, 
		      nnei,
		      i_type_shift,
		      p_table_info,
		      p_table_data,
		      tab_stride,
		      rij,
		      scale,
		      type, 
		      nlist,
		      natoms,
		      sel_a);
      // r neighbor
      _pair_tab_jloop(energy,
		      force,
		      virial,
		      jiter,
		      i_idx, 
		      nnei,
		      i_type_shift,
		      p_table_info,
		      p_table_data,
		      tab_stride,
		      rij,
		      scale,
		      type, 
		      nlist,
		      natoms,
		      sel_r);
      i_idx ++;
    }
  }
}


template
void deepmd::pair_tab_cpu<float>(
    float * energy,
    float * force,
    float * virial,
    const double * table_info,
    const double * table_data,
    const float * rij,
    const float * scale,
    const int * type,
    const int * nlist,
    const int * natoms,
    const std::vector<int> & sel_a,
    const std::vector<int> & sel_r
    );

template
void deepmd::pair_tab_cpu<double>(
    double * energy,
    double * force,
    double * virial,
    const double * table_info,
    const double * table_data,
    const double * rij,
    const double * scale,
    const int * type,
    const int * nlist,
    const int * natoms,
    const std::vector<int> & sel_a,
    const std::vector<int> & sel_r
    );

