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

#include "env_mat_nvnmd.h"

#include "switcher.h"

// env_mat_a_nvnmd_cpu
// have been remove for the same function

/*
//==================================================
  env_mat_a_nvnmd_quantize_cpu
//==================================================
*/

template <typename FPTYPE>
void deepmd::env_mat_a_nvnmd_quantize_cpu(std::vector<FPTYPE>& descrpt_a,
                                          std::vector<FPTYPE>& descrpt_a_deriv,
                                          std::vector<FPTYPE>& rij_a,
                                          const std::vector<FPTYPE>& posi,
                                          const std::vector<int>& type,
                                          const int& i_idx,
                                          const std::vector<int>& fmt_nlist_a,
                                          const std::vector<int>& sec_a,
                                          const float& rmin,
                                          const float& rmax) {
  // compute the diff of the neighbors
  rij_a.resize(sec_a.back() * 3);
  fill(rij_a.begin(), rij_a.end(), (FPTYPE)0.0);
  for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii) {
    for (int jj = sec_a[ii]; jj < sec_a[ii + 1]; ++jj) {
      if (fmt_nlist_a[jj] < 0) {
        break;
      }
      const int& j_idx = fmt_nlist_a[jj];
      for (int dd = 0; dd < 3; ++dd) {
        rij_a[jj * 3 + dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
      }
    }
  }
  // 1./rr, cos(theta), cos(phi), sin(phi)
  descrpt_a.resize(sec_a.back() * 4);
  fill(descrpt_a.begin(), descrpt_a.end(), (FPTYPE)0.0);
  // deriv wrt center: 3
  descrpt_a_deriv.resize(sec_a.back() * 4 * 3);
  fill(descrpt_a_deriv.begin(), descrpt_a_deriv.end(), (FPTYPE)0.0);
  U_Flt64_Int64 ufi;
  int64_t expo_max;

  for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
    for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter + 1];
         ++nei_iter) {
      if (fmt_nlist_a[nei_iter] < 0) {
        break;
      }
      const FPTYPE* rr = &rij_a[nei_iter * 3];

      // NVNMD
      FPTYPE rij[3];
      ufi.nflt = rr[0];
      ufi.nint &= FLT_MASK;
      rij[0] = ufi.nflt;
      ufi.nflt = rr[1];
      ufi.nint &= FLT_MASK;
      rij[1] = ufi.nflt;
      ufi.nflt = rr[2];
      ufi.nint &= FLT_MASK;
      rij[2] = ufi.nflt;

      FPTYPE nr2;
      dotmul_flt_nvnmd(nr2, rij, rij, 3);

      int idx_deriv = nei_iter * 4 * 3;  // 4 components time 3 directions
      int idx_value = nei_iter * 4;      // 4 components
      // 4 value components
      descrpt_a[idx_value + 0] = nr2;
      descrpt_a[idx_value + 1] = rij[0];
      descrpt_a[idx_value + 2] = rij[1];
      descrpt_a[idx_value + 3] = rij[2];
      // deriv of component 1/r
      descrpt_a_deriv[idx_deriv + 0] = -2 * rij[0];
      descrpt_a_deriv[idx_deriv + 1] = -2 * rij[1];
      descrpt_a_deriv[idx_deriv + 2] = -2 * rij[2];
      /*
      d(sw*x/r)_d(x) = x * d(sw/r)_d(x) + sw/r
      d(sw*y/r)_d(x) = y * d(sw/r)_d(x)
      */
      // deriv of component x/r
      descrpt_a_deriv[idx_deriv + 3] = -1;
      descrpt_a_deriv[idx_deriv + 4] = 0;
      descrpt_a_deriv[idx_deriv + 5] = 0;
      // deriv of component y/r2
      descrpt_a_deriv[idx_deriv + 6] = 0;
      descrpt_a_deriv[idx_deriv + 7] = -1;
      descrpt_a_deriv[idx_deriv + 8] = 0;
      // deriv of component z/r2
      descrpt_a_deriv[idx_deriv + 9] = 0;
      descrpt_a_deriv[idx_deriv + 10] = 0;
      descrpt_a_deriv[idx_deriv + 11] = -1;
    }
  }
}

template void deepmd::env_mat_a_nvnmd_quantize_cpu<double>(
    std::vector<double>& descrpt_a,
    std::vector<double>& descrpt_a_deriv,
    std::vector<double>& rij_a,
    const std::vector<double>& posi,
    const std::vector<int>& type,
    const int& i_idx,
    const std::vector<int>& fmt_nlist,
    const std::vector<int>& sec,
    const float& rmin,
    const float& rmax);

template void deepmd::env_mat_a_nvnmd_quantize_cpu<float>(
    std::vector<float>& descrpt_a,
    std::vector<float>& descrpt_a_deriv,
    std::vector<float>& rij_a,
    const std::vector<float>& posi,
    const std::vector<int>& type,
    const int& i_idx,
    const std::vector<int>& fmt_nlist,
    const std::vector<int>& sec,
    const float& rmin,
    const float& rmax);
