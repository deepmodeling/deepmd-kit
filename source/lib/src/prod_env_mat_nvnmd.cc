// SPDX-License-Identifier: LGPL-3.0-or-later
/*
//==================================================
 _   _  __     __  _   _   __  __   ____
| \ | | \ \   / / | \ | | |  \/  | |  _ \
|  \| |  \ \ / /  |  \| | | |\/| | | | | |
| |\  |   \ V /   | |\  | | |  | | | |_| |
|_| \_|    \_/    |_| \_| |_|  |_| |____/

//==================================================

code: nvnmd
reference: deepmd
author: mph (pinghui_mo@outlook.com)
date: 2021-12-6

*/

#include "prod_env_mat_nvnmd.h"

#include <string.h>

#include <cassert>
#include <iostream>

#include "env_mat_nvnmd.h"
#include "fmt_nlist.h"

using namespace deepmd;

/*
//==================================================
  prod_env_mat_a_nvnmd_cpu
//==================================================
*/

// have been remove for the same function

/*
//==================================================
  prod_env_mat_a_nvnmd_quantize_cpu
//==================================================
*/

template <typename FPTYPE>
void deepmd::prod_env_mat_a_nvnmd_quantize_cpu(FPTYPE *em,
                                               FPTYPE *em_deriv,
                                               FPTYPE *rij,
                                               int *nlist,
                                               const FPTYPE *coord,
                                               const int *type,
                                               const InputNlist &inlist,
                                               const int max_nbor_size,
                                               const FPTYPE *avg,
                                               const FPTYPE *std,
                                               const int nloc,
                                               const int nall,
                                               const float rcut,
                                               const float rcut_smth,
                                               const std::vector<int> sec,
                                               const int *f_type) {
  if (f_type == NULL) {
    f_type = type;
  }
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
  std::vector<int> d_f_type(nall);
  for (int ii = 0; ii < nall; ++ii) {
    d_f_type[ii] = f_type[ii];
  }

  // build nlist
  std::vector<std::vector<int> > d_nlist_a(nloc);

  assert(nloc == inlist.inum);
  for (unsigned ii = 0; ii < nloc; ++ii) {
    d_nlist_a[ii].reserve(max_nbor_size);
  }
  for (unsigned ii = 0; ii < nloc; ++ii) {
    int i_idx = inlist.ilist[ii];
    for (unsigned jj = 0; jj < inlist.numneigh[ii]; ++jj) {
      int j_idx = inlist.firstneigh[ii][jj];
      d_nlist_a[i_idx].push_back(j_idx);
    }
  }

#pragma omp parallel for
  for (int ii = 0; ii < nloc; ++ii) {
    std::vector<int> fmt_nlist_a;
    int ret = format_nlist_i_cpu(fmt_nlist_a, d_coord3, d_f_type, ii,
                                 d_nlist_a[ii], rcut, sec);
    std::vector<FPTYPE> d_em_a;
    std::vector<FPTYPE> d_em_a_deriv;
    std::vector<FPTYPE> d_em_r;
    std::vector<FPTYPE> d_em_r_deriv;
    std::vector<FPTYPE> d_rij_a;
    env_mat_a_nvnmd_quantize_cpu(d_em_a, d_em_a_deriv, d_rij_a, d_coord3,
                                 d_f_type, ii, fmt_nlist_a, sec, rcut_smth,
                                 rcut);

    // check sizes
    assert(d_em_a.size() == nem);
    assert(d_em_a_deriv.size() == nem * 3);
    assert(d_rij_a.size() == nnei * 3);
    assert(fmt_nlist_a.size() == nnei);
    // record outputs
    for (int jj = 0; jj < nem; ++jj) {
      if (type[ii] >= 0) {
        // em[ii * nem + jj] =
        //     (d_em_a[jj] - avg[type[ii] * nem + jj]) / std[type[ii] * nem +
        //     jj];
        em[ii * nem + jj] = d_em_a[jj];
      } else {
        em[ii * nem + jj] = 0;
      }
    }
    for (int jj = 0; jj < nem * 3; ++jj) {
      if (type[ii] >= 0) {
        // em_deriv[ii * nem * 3 + jj] =
        //     d_em_a_deriv[jj] / std[type[ii] * nem + jj / 3];
        em_deriv[ii * nem * 3 + jj] = d_em_a_deriv[jj];
      } else {
        em_deriv[ii * nem * 3 + jj] = 0;
      }
    }
    for (int jj = 0; jj < nnei * 3; ++jj) {
      rij[ii * nnei * 3 + jj] = d_rij_a[jj];
    }
    for (int jj = 0; jj < nnei; ++jj) {
      nlist[ii * nnei + jj] = fmt_nlist_a[jj];
    }
  }
}

template void deepmd::prod_env_mat_a_nvnmd_quantize_cpu<double>(
    double *em,
    double *em_deriv,
    double *rij,
    int *nlist,
    const double *coord,
    const int *type,
    const InputNlist &inlist,
    const int max_nbor_size,
    const double *avg,
    const double *std,
    const int nloc,
    const int nall,
    const float rcut,
    const float rcut_smth,
    const std::vector<int> sec,
    const int *f_type);

template void deepmd::prod_env_mat_a_nvnmd_quantize_cpu<float>(
    float *em,
    float *em_deriv,
    float *rij,
    int *nlist,
    const float *coord,
    const int *type,
    const InputNlist &inlist,
    const int max_nbor_size,
    const float *avg,
    const float *std,
    const int nloc,
    const int nall,
    const float rcut,
    const float rcut_smth,
    const std::vector<int> sec,
    const int *f_type);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// UNDEFINE
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
