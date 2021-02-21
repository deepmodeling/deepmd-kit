#include <cassert>
#include <iostream>
#include "prod_env_mat.h"
#include "fmt_nlist.h"
#include "env_mat.h"

template<typename FPTYPE>
void prod_env_mat_a_cpu(
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const int ntypes, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec) 
{
  const int nnei = sec.back();
  const int nem = nnei * 4;

  // set & normalize coord
  std::vector<FPTYPE> d_coord3(nall * 3);
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      d_coord3[ii * 3 + dd] = coord[ii * 3 + dd];
    }
  }

  // set type
  std::vector<int> d_type (nall);
  for (int ii = 0; ii < nall; ++ii) {
    d_type[ii] = type[ii];
  }
    
  // build nlist
  std::vector<std::vector<int > > d_nlist_a(nloc);

  for (unsigned ii = 0; ii < nloc; ++ii) {
    d_nlist_a.reserve (jrange[nloc] / nloc + 10);
  }
  for (unsigned ii = 0; ii < nloc; ++ii) {
    int i_idx = ilist[ii];
    for (unsigned jj = jrange[ii]; jj < jrange[ii+1]; ++jj) {
      int j_idx = jlist[jj];
      d_nlist_a[i_idx].push_back (j_idx);
    }
  }
    
#pragma omp parallel for 
  for (int ii = 0; ii < nloc; ++ii) {
    std::vector<int> fmt_nlist_a;
    int ret = format_nlist_cpu(fmt_nlist_a, d_coord3, ntypes, d_type, ii, d_nlist_a[ii], rcut, sec);
    std::vector<FPTYPE> d_em_a;
    std::vector<FPTYPE> d_em_a_deriv;
    std::vector<FPTYPE> d_em_r;
    std::vector<FPTYPE> d_em_r_deriv;
    std::vector<FPTYPE> d_rij_a;
    env_mat_a_cpu (d_em_a, d_em_a_deriv, d_rij_a, d_coord3, ntypes, d_type, ii, fmt_nlist_a, sec, rcut_smth, rcut);

    // check sizes
    assert (d_em_a.size() == nem);
    assert (d_em_a_deriv.size() == nem * 3);
    assert (d_rij_a.size() == nnei * 3);
    assert (fmt_nlist_a.size() == nnei);
    // record outputs
    for (int jj = 0; jj < nem; ++jj) {
      em[ii * nem + jj] = (d_em_a[jj] - avg[d_type[ii] * nem + jj]) / std[d_type[ii] * nem + jj];
    }
    for (int jj = 0; jj < nem * 3; ++jj) {
      em_deriv[ii * nem * 3 + jj] = d_em_a_deriv[jj] / std[d_type[ii] * nem + jj / 3];
    }
    for (int jj = 0; jj < nnei * 3; ++jj) {
      rij[ii * nnei * 3 + jj] = d_rij_a[jj];
    }
    for (int jj = 0; jj < nnei; ++jj) {
      nlist[ii * nnei + jj] = fmt_nlist_a[jj];
    }
  }
}


template
void prod_env_mat_a_cpu<double>(
    double * em, 
    double * em_deriv, 
    double * rij, 
    int * nlist, 
    const double * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    const double * avg, 
    const double * std, 
    const int nloc, 
    const int nall, 
    const int ntypes, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

template
void prod_env_mat_a_cpu<float>(
    float * em, 
    float * em_deriv, 
    float * rij, 
    int * nlist, 
    const float * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    const float * avg, 
    const float * std, 
    const int nloc, 
    const int nall, 
    const int ntypes, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

#if GOOGLE_CUDA
template<typename FPTYPE> 
void prod_env_mat_a_gpu_nv(    
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    int * array_int, 
    unsigned long long * array_longlong,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const int ntypes, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec) {
  DescrptSeAFunctor<FPTYPE>()(
      em, em_deriv, rij, nlist,
      coord, type, ilist, jrange, jlist, array_int, array_longlong, avg, std, nloc, nall, rcut, rcut_smth, sec, max_nbor_size);
}

template
void prod_env_mat_a_gpu_nv<double>(    
    double * em, 
    double * em_deriv, 
    double * rij, 
    int * nlist, 
    const double * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    int * array_int, 
    unsigned long long * array_longlong,
    const int max_nbor_size,
    const double * avg, 
    const double * std, 
    const int nloc, 
    const int nall, 
    const int ntypes, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

template
void prod_env_mat_a_gpu_nv<float>(   
    float * em, 
    float * em_deriv, 
    float * rij, 
    int * nlist, 
    const float * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    int * array_int, 
    unsigned long long * array_longlong,
    const int max_nbor_size,
    const float * avg, 
    const float * std, 
    const int nloc, 
    const int nall, 
    const int ntypes, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);
#endif
