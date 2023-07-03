
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

#pragma once

#include <cmath>
#include <vector>

#include "env_mat_nvnmd.h"
#include "utilities.h"

namespace deepmd {

template <typename FPTYPE>
void env_mat_a_nvnmd_quantize_cpu(std::vector<FPTYPE> &descrpt_a,
                                  std::vector<FPTYPE> &descrpt_a_deriv,
                                  std::vector<FPTYPE> &rij_a,
                                  const std::vector<FPTYPE> &posi,
                                  const std::vector<int> &type,
                                  const int &i_idx,
                                  const std::vector<int> &fmt_nlist,
                                  const std::vector<int> &sec,
                                  const float &rmin,
                                  const float &rmax);
}

union U_Flt64_Int64 {
  double nflt;
  int64_t nint;
};

/* 21-bit fraction */
// #define NBIT_FLTF 21
// #define NBIT_CUTF (52 - NBIT_FLTF)
// #define FLT_MASK 0xffffffff80000000

/* 20-bit fraction */
#define NBIT_FLTF 20
#define NBIT_CUTF (52 - NBIT_FLTF)
#define FLT_MASK 0xffffffff00000000

/*
  split double into sign, expo, and frac
*/
template <class T>  // float and double
void split_flt(T x, int64_t &sign, int64_t &expo, int64_t &mant) {
  U_Flt64_Int64 ufi;
  ufi.nflt = x;
  sign = (ufi.nint >> 63) & 0x01;
  expo = ((ufi.nint >> 52) & 0x7ff) - 1023;
  mant = (ufi.nint & 0xfffffffffffff) | 0x10000000000000;  // 1+52
}

/*
 find the max exponent for float array x
*/
template <class T>  // float and double
void find_max_expo(int64_t &max_expo, T *x, int64_t M) {
  int ii, jj, kk;
  U_Flt64_Int64 ufi;
  int64_t expo;
  max_expo = -100;
  for (jj = 0; jj < M; jj++) {
    ufi.nflt = x[jj];
    expo = ((ufi.nint >> 52) & 0x7ff) - 1023;
    max_expo = (expo > max_expo) ? expo : max_expo;
  }
};

/*
 find the max exponent for float array x
*/
template <class T>  // float and double
void find_max_expo(int64_t &max_expo, T *x, int64_t N, int64_t M) {
  int ii, jj, kk;
  U_Flt64_Int64 ufi;
  int64_t expo;
  max_expo = -100;
  for (ii = 0; ii < N; ii++) {
    ufi.nflt = x[ii * M];
    expo = ((ufi.nint >> 52) & 0x7ff) - 1023;
    max_expo = (expo > max_expo) ? expo : max_expo;
  }
};

/*
 dot multiply
*/
template <class T>  // float and double
void dotmul_flt_nvnmd(T &y, T *x1, T *x2, int64_t M) {
  int ii, jj, kk;
  U_Flt64_Int64 ufi;
  //
  int64_t sign1, sign2, sign3;
  int64_t expo1, expo2, expo3;
  int64_t mant1, mant2, mant3;
  int64_t expos;
  //
  int64_t expo_max1, expo_max2;
  //
  find_max_expo(expo_max1, x1, M);
  find_max_expo(expo_max2, x2, M);
  //
  int64_t s = 0;
  for (jj = 0; jj < M; jj++) {
    // x1
    split_flt(x1[jj], sign1, expo1, mant1);
    mant1 >>= NBIT_CUTF;
    expos = expo_max1 - expo1;
    expos = (expos > 63) ? 63 : expos;
    mant1 >>= expos;
    // x2
    split_flt(x2[jj], sign2, expo2, mant2);
    mant2 >>= NBIT_CUTF;
    expos = expo_max2 - expo2;
    expos = (expos > 63) ? 63 : expos;
    mant2 >>= expos;
    // multiply
    mant3 = mant1 * mant2;
    mant3 = (sign1 ^ sign2) ? -mant3 : mant3;
    s += mant3;
  }
  // y * 2^(e_a+e_b)
  ufi.nflt = T(s) * pow(2.0, expo_max1 + expo_max2 - NBIT_FLTF - NBIT_FLTF);
  ufi.nint &= FLT_MASK;
  y = ufi.nflt;
}

/*
  multiply
*/
template <class T>  // float and double
void mul_flt_nvnmd(T &y, T x1, T x2) {
  U_Flt64_Int64 ufi1, ufi2, ufi3;
  ufi1.nflt = x1;
  ufi1.nint &= FLT_MASK;
  ufi2.nflt = x2;
  ufi2.nint &= FLT_MASK;
  ufi3.nflt = ufi2.nflt * ufi1.nflt;
  ufi3.nint &= FLT_MASK;
  y = ufi3.nflt;
}

/*
  add
*/
template <class T>  // float and double
void add_flt_nvnmd(T &y, T x1, T x2) {
  U_Flt64_Int64 ufi1, ufi2, ufi3;
  int64_t sign1, sign2, sign3;
  int64_t expo1, expo2, expo3;
  int64_t mant1, mant2, mant3;
  int64_t expos;
  // convert data
  split_flt(x1, sign1, expo1, mant1);
  mant1 >>= NBIT_CUTF;

  split_flt(x2, sign2, expo2, mant2);
  mant2 >>= NBIT_CUTF;

  // shift
  if (expo1 >= expo2) {
    expo3 = expo1;
    expos = expo1 - expo2;
    expos = (expos > 63) ? 63 : expos;
    mant2 >>= expos;
  } else {
    expo3 = expo2;
    expos = expo2 - expo1;
    expos = (expos > 63) ? 63 : expos;
    mant1 >>= expos;
  }

  // add
  mant1 = (sign1) ? -mant1 : mant1;
  mant2 = (sign2) ? -mant2 : mant2;
  mant3 = mant1 + mant2;
  // fix2flt
  ufi3.nflt = double(mant3) * pow(2.0, expo3 - NBIT_FLTF);
  ufi3.nint &= FLT_MASK;
  y = ufi3.nflt;
}
