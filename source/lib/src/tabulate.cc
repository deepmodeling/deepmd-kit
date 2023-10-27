// SPDX-License-Identifier: LGPL-3.0-or-later
#include "tabulate.h"

#include <string.h>

#include <cassert>
#include <iostream>
#include <vector>
/*
    This inline function was designed to get the table info and bias value for
   current input xx! lower:      indicate the lower boundary of the first table;
    upper:      indicate the upper boundary of the first table as well as the
   lower boundary of the second table; max:        indicate the upper boundary
   of the second table; stride0:    indicate the stride of the first table;
    stride1:    indicate the stride of the second table;
    xx:         indicate the inputs value;
    table_idx:  indicate the location of table info of input value xx;
*/
template <typename FPTYPE>
inline void locate_xx(const FPTYPE& lower,
                      const FPTYPE& upper,
                      const FPTYPE& max,
                      const FPTYPE& stride0,
                      const FPTYPE& stride1,
                      FPTYPE& xx,
                      int& table_idx) {
  if (xx < lower) {
    table_idx = 0;
    xx = (FPTYPE)0.;
  } else if (xx < upper) {
    table_idx = (int)((xx - lower) / stride0);
    xx -= (table_idx * stride0 + lower);
  } else if (xx < max) {
    int first_stride = int((upper - lower) / stride0);
    table_idx = first_stride + (int)((xx - upper) / stride1);
    xx -= ((table_idx - first_stride) * stride1 + upper);
  } else {
    table_idx =
        int((upper - lower) / stride0) + (int)((max - upper) / stride1) - 1;
    xx = (FPTYPE)0.;
  }
}

template <typename FPTYPE>
inline void locate_xx_se_t(const FPTYPE& lower,
                           const FPTYPE& upper,
                           const FPTYPE& min,
                           const FPTYPE& max,
                           const FPTYPE& stride0,
                           const FPTYPE& stride1,
                           FPTYPE& xx,
                           int& table_idx) {
  if (xx < min) {
    table_idx = 0;
    xx = (FPTYPE)0.;
  } else if (xx < lower) {
    table_idx = (int)((xx - min) / stride1);
    xx -= (table_idx * stride1 + min);
  } else if (xx < upper) {
    int first_stride = int((lower - min) / stride1);
    table_idx = first_stride + (int)((xx - lower) / stride0);
    xx -= ((table_idx - first_stride) * stride0 + lower);
  } else if (xx < max) {
    int first_stride =
        int((lower - min) / stride1) + int((upper - lower) / stride0);
    table_idx = first_stride + (int)((xx - upper) / stride1);
    xx -= ((table_idx - first_stride) * stride1 + upper);
  } else {
    table_idx = int((lower - min) / stride1) + int((upper - lower) / stride0) +
                (int)((max - upper) / stride1) - 1;
    xx = (FPTYPE)0.;
  }
}

template <typename FPTYPE>
inline FPTYPE dot(FPTYPE a[4], FPTYPE b[4]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_a_cpu(FPTYPE* out,
                                      const FPTYPE* table,
                                      const FPTYPE* table_info,
                                      const FPTYPE* em_x,
                                      const FPTYPE* em,
                                      const FPTYPE* two_embed,
                                      const int nloc,
                                      const int nnei,
                                      const int last_layer_size,
                                      const bool is_sorted) {
  bool enable_se_atten = two_embed != nullptr;
  memset(out, 0, sizeof(FPTYPE) * nloc * 4 * last_layer_size);
  const FPTYPE lower = table_info[0];
  const FPTYPE upper = table_info[1];
  const FPTYPE _max = table_info[2];
  const FPTYPE stride0 = table_info[3];
  const FPTYPE stride1 = table_info[4];
// for every atom, execute a small manual gemm ~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    FPTYPE ll[4] = {0};
    FPTYPE ago = em_x[ii * nnei + nnei - 1];
    bool unloop = false;
    for (int jj = 0; jj < nnei; jj++) {
      ll[0] = em[ii * nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * nnei + jj];
      if (ago == xx && ll[1] == 0. && ll[2] == 0. && ll[3] == 0. && is_sorted) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      for (int kk = 0; kk < last_layer_size; kk++) {
        FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * kk + 0];
        FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * kk + 1];
        FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * kk + 2];
        FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
        FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
        FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
        FPTYPE var =
            a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
        if (enable_se_atten) {
          FPTYPE t = two_embed[ii * nnei * last_layer_size +
                               jj * last_layer_size + kk];
          var = var * t + var;
        }

        if (unloop) {
          out[ii * last_layer_size * 4 + 0 * last_layer_size + kk] +=
              (nnei - jj) * var * ll[0];
          out[ii * last_layer_size * 4 + 1 * last_layer_size + kk] +=
              (nnei - jj) * var * ll[1];
          out[ii * last_layer_size * 4 + 2 * last_layer_size + kk] +=
              (nnei - jj) * var * ll[2];
          out[ii * last_layer_size * 4 + 3 * last_layer_size + kk] +=
              (nnei - jj) * var * ll[3];
        } else {
          out[ii * last_layer_size * 4 + 0 * last_layer_size + kk] +=
              var * ll[0];
          out[ii * last_layer_size * 4 + 1 * last_layer_size + kk] +=
              var * ll[1];
          out[ii * last_layer_size * 4 + 2 * last_layer_size + kk] +=
              var * ll[2];
          out[ii * last_layer_size * 4 + 3 * last_layer_size + kk] +=
              var * ll[3];
        }
      }
      if (unloop) {
        break;
      }
    }
  }
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_a_grad_cpu(FPTYPE* dy_dem_x,
                                           FPTYPE* dy_dem,
                                           FPTYPE* dy_dtwo,
                                           const FPTYPE* table,
                                           const FPTYPE* table_info,
                                           const FPTYPE* em_x,
                                           const FPTYPE* em,
                                           const FPTYPE* two_embed,
                                           const FPTYPE* dy,
                                           const int nloc,
                                           const int nnei,
                                           const int last_layer_size,
                                           const bool is_sorted) {
  bool enable_se_atten = two_embed != nullptr;
  memset(dy_dem_x, 0, sizeof(FPTYPE) * nloc * nnei);
  memset(dy_dem, 0, sizeof(FPTYPE) * nloc * nnei * 4);
  if (enable_se_atten) {
    memset(dy_dtwo, 0, sizeof(FPTYPE) * nloc * nnei * last_layer_size);
  }
  FPTYPE const lower = table_info[0];
  FPTYPE const upper = table_info[1];
  FPTYPE const _max = table_info[2];
  FPTYPE const stride0 = table_info[3];
  FPTYPE const stride1 = table_info[4];
// for every atom, execute a small gemm~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    FPTYPE ll[4];
    FPTYPE rr[4];
    FPTYPE ago = em_x[ii * nnei + nnei - 1];
    bool unloop = false;
    for (int jj = 0; jj < nnei; jj++) {
      // construct the dy/dx
      ll[0] = em[ii * nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * nnei + jj];
      if (ago == xx && ll[1] == 0. && ll[2] == 0. && ll[3] == 0. && is_sorted) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      FPTYPE grad = (FPTYPE)0.0;
      for (int kk = 0; kk < last_layer_size; kk++) {
        rr[0] = dy[ii * last_layer_size * 4 + 0 * last_layer_size + kk];
        rr[1] = dy[ii * last_layer_size * 4 + 1 * last_layer_size + kk];
        rr[2] = dy[ii * last_layer_size * 4 + 2 * last_layer_size + kk];
        rr[3] = dy[ii * last_layer_size * 4 + 3 * last_layer_size + kk];
        FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * kk + 0];
        FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * kk + 1];
        FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * kk + 2];
        FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
        FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
        FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
        FPTYPE res =
            a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
        FPTYPE g =
            (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx);
        FPTYPE resold = res;
        if (enable_se_atten) {
          FPTYPE t = two_embed[ii * nnei * last_layer_size +
                               jj * last_layer_size + kk];
          res = res * t + res;
          g += t * g;
        }

        FPTYPE dotllrr = dot(ll, rr);
        if (unloop) {
          grad += g * dotllrr * (nnei - jj);
          dy_dem[ii * nnei * 4 + jj * 4 + 0] += res * rr[0] * (nnei - jj);
          dy_dem[ii * nnei * 4 + jj * 4 + 1] += res * rr[1] * (nnei - jj);
          dy_dem[ii * nnei * 4 + jj * 4 + 2] += res * rr[2] * (nnei - jj);
          dy_dem[ii * nnei * 4 + jj * 4 + 3] += res * rr[3] * (nnei - jj);
          if (enable_se_atten) {
            // fill from jj to nnei
            for (int jj2 = jj; jj2 < nnei; jj2++) {
              dy_dtwo[ii * nnei * last_layer_size + jj2 * last_layer_size +
                      kk] += resold * dotllrr;
            }
          }
        } else {
          grad += g * dotllrr;
          dy_dem[ii * nnei * 4 + jj * 4 + 0] += res * rr[0];
          dy_dem[ii * nnei * 4 + jj * 4 + 1] += res * rr[1];
          dy_dem[ii * nnei * 4 + jj * 4 + 2] += res * rr[2];
          dy_dem[ii * nnei * 4 + jj * 4 + 3] += res * rr[3];
          if (enable_se_atten) {
            dy_dtwo[ii * nnei * last_layer_size + jj * last_layer_size + kk] +=
                resold * dotllrr;
          }
        }
      }
      dy_dem_x[ii * nnei + jj] = grad;
      if (unloop) {
        break;
      }
    }
  }
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_a_grad_grad_cpu(FPTYPE* dz_dy,
                                                const FPTYPE* table,
                                                const FPTYPE* table_info,
                                                const FPTYPE* em_x,
                                                const FPTYPE* em,
                                                const FPTYPE* two_embed,
                                                const FPTYPE* dz_dy_dem_x,
                                                const FPTYPE* dz_dy_dem,
                                                const FPTYPE* dz_dy_dtwo,
                                                const int nloc,
                                                const int nnei,
                                                const int last_layer_size,
                                                const bool is_sorted) {
  bool enable_se_atten = two_embed != nullptr;
  memset(dz_dy, 0, sizeof(FPTYPE) * nloc * 4 * last_layer_size);
  const FPTYPE lower = table_info[0];
  const FPTYPE upper = table_info[1];
  const FPTYPE _max = table_info[2];
  const FPTYPE stride0 = table_info[3];
  const FPTYPE stride1 = table_info[4];
// for every atom, execute a small manual gemm ~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    FPTYPE ll[4];
    FPTYPE hh[4];
    FPTYPE ago = em_x[ii * nnei + nnei - 1];
    bool unloop = false;
    for (int jj = 0; jj < nnei; jj++) {
      ll[0] = em[ii * nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * nnei * 4 + jj * 4 + 3];
      hh[0] = dz_dy_dem[ii * nnei * 4 + jj * 4 + 0];
      hh[1] = dz_dy_dem[ii * nnei * 4 + jj * 4 + 1];
      hh[2] = dz_dy_dem[ii * nnei * 4 + jj * 4 + 2];
      hh[3] = dz_dy_dem[ii * nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * nnei + jj];
      FPTYPE dz_xx = dz_dy_dem_x[ii * nnei + jj];
      if (ago == xx && ll[1] == 0. && ll[2] == 0. && ll[3] == 0. && is_sorted) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      for (int kk = 0; kk < last_layer_size; kk++) {
        FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * kk + 0];
        FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * kk + 1];
        FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * kk + 2];
        FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
        FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
        FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
        FPTYPE var =
            a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
        FPTYPE var_grad =
            a1 +
            ((FPTYPE)2. * a2 +
             ((FPTYPE)3. * a3 + ((FPTYPE)4. * a4 + (FPTYPE)5. * a5 * xx) * xx) *
                 xx) *
                xx;
        FPTYPE two_grad = 0.;
        if (enable_se_atten) {
          FPTYPE t = two_embed[ii * nnei * last_layer_size +
                               jj * last_layer_size + kk];
          // dz_dy_dtwo * var * ll
          // var above should be used instead of var + var * t below
          two_grad = dz_dy_dtwo[ii * nnei * last_layer_size +
                                jj * last_layer_size + kk] *
                     var;
          var += var * t;
          var_grad += var_grad * t;
        }

        /*
         * `dz_dy` represents the derivative of the variable `out` in the
         * function `deepmd::tabulate_fusion_se_a_cpu`.
         *
         * The expression `var * hh[0] + dz_xx * var_grad * ll[0]` utilizes the
         * product rule of derivatives: `(f * g)' = f' * g + f * g'`.
         *
         * This expression can be alternatively expressed as:
         * `hh[0] * var + ll[0] * (dz_xx * var_grad)`.
         * Note that `hh[0]` is one element of `em`, and `ll[0]` is one element
         * of `dz_dy_dem` which is `em'`.
         *
         * Therefore, we can rewrite this expression as: `em' * var + em *
         * var'`, where `em'` is the derivative of `em` and `var'` is the
         * derivative of `var`. Additionally, `var'` can be further represented
         * as: `var_grad * dz_xx`.
         *
         * If `enable_se_atten` is true, `var` will be `var * t + var`, and
         * `var'` will be `(var_grad * t + var_grad) * dz_xx`.
         */
        if (unloop) {
          dz_dy[ii * last_layer_size * 4 + 0 * last_layer_size + kk] +=
              (nnei - jj) *
              (var * hh[0] + (dz_xx * var_grad + two_grad) * ll[0]);
          dz_dy[ii * last_layer_size * 4 + 1 * last_layer_size + kk] +=
              (nnei - jj) *
              (var * hh[1] + (dz_xx * var_grad + two_grad) * ll[1]);
          dz_dy[ii * last_layer_size * 4 + 2 * last_layer_size + kk] +=
              (nnei - jj) *
              (var * hh[2] + (dz_xx * var_grad + two_grad) * ll[2]);
          dz_dy[ii * last_layer_size * 4 + 3 * last_layer_size + kk] +=
              (nnei - jj) *
              (var * hh[3] + (dz_xx * var_grad + two_grad) * ll[3]);
        } else {
          dz_dy[ii * last_layer_size * 4 + 0 * last_layer_size + kk] +=
              var * hh[0] + (dz_xx * var_grad + two_grad) * ll[0];
          dz_dy[ii * last_layer_size * 4 + 1 * last_layer_size + kk] +=
              var * hh[1] + (dz_xx * var_grad + two_grad) * ll[1];
          dz_dy[ii * last_layer_size * 4 + 2 * last_layer_size + kk] +=
              var * hh[2] + (dz_xx * var_grad + two_grad) * ll[2];
          dz_dy[ii * last_layer_size * 4 + 3 * last_layer_size + kk] +=
              var * hh[3] + (dz_xx * var_grad + two_grad) * ll[3];
        }
      }
      if (unloop) {
        break;
      }
    }
  }
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_t_cpu(FPTYPE* out,
                                      const FPTYPE* table,
                                      const FPTYPE* table_info,
                                      const FPTYPE* em_x,
                                      const FPTYPE* em,
                                      const int nloc,
                                      const int nnei_i,
                                      const int nnei_j,
                                      const int last_layer_size) {
  memset(out, 0, sizeof(FPTYPE) * nloc * last_layer_size);
  const FPTYPE lower = table_info[0];
  const FPTYPE upper = table_info[1];
  const FPTYPE _max = table_info[2];
  const FPTYPE stride0 = table_info[3];
  const FPTYPE stride1 = table_info[4];
// for every atom, execute a small manual gemm ~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    for (int jj = 0; jj < nnei_i; jj++) {
      // unloop not work as em_x is not sorted
      for (int kk = 0; kk < nnei_j; kk++) {
        FPTYPE xx = em_x[ii * nnei_i * nnei_j + jj * nnei_j + kk];
        FPTYPE ll = xx;
        int table_idx = 0;
        locate_xx_se_t(lower, upper, -_max, _max, stride0, stride1, xx,
                       table_idx);
        for (int mm = 0; mm < last_layer_size; mm++) {
          FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * mm + 0];
          FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * mm + 1];
          FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * mm + 2];
          FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * mm + 3];
          FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * mm + 4];
          FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * mm + 5];
          FPTYPE var =
              a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
          out[ii * last_layer_size + mm] += var * ll;
        }
      }
    }
  }
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_t_grad_cpu(FPTYPE* dy_dem_x,
                                           FPTYPE* dy_dem,
                                           const FPTYPE* table,
                                           const FPTYPE* table_info,
                                           const FPTYPE* em_x,
                                           const FPTYPE* em,
                                           const FPTYPE* dy,
                                           const int nloc,
                                           const int nnei_i,
                                           const int nnei_j,
                                           const int last_layer_size) {
  memset(dy_dem_x, 0, sizeof(FPTYPE) * nloc * nnei_i * nnei_j);
  memset(dy_dem, 0, sizeof(FPTYPE) * nloc * nnei_i * nnei_j);
  FPTYPE const lower = table_info[0];
  FPTYPE const upper = table_info[1];
  FPTYPE const _max = table_info[2];
  FPTYPE const stride0 = table_info[3];
  FPTYPE const stride1 = table_info[4];
// for every atom, execute a small gemm~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    FPTYPE ll = (FPTYPE)0.;
    FPTYPE rr = (FPTYPE)0.;
    for (int jj = 0; jj < nnei_i; jj++) {
      for (int kk = 0; kk < nnei_j; kk++) {
        // construct the dy/dx
        FPTYPE xx = em_x[ii * nnei_i * nnei_j + jj * nnei_j + kk];
        ll = xx;
        int table_idx = 0;
        locate_xx_se_t(lower, upper, -_max, _max, stride0, stride1, xx,
                       table_idx);
        FPTYPE grad = (FPTYPE)0.0;
        for (int mm = 0; mm < last_layer_size; mm++) {
          rr = dy[ii * last_layer_size + mm];
          FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * mm + 0];
          FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * mm + 1];
          FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * mm + 2];
          FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * mm + 3];
          FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * mm + 4];
          FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * mm + 5];
          FPTYPE res =
              a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;

          grad += (a1 + ((FPTYPE)2. * a2 +
                         ((FPTYPE)3. * a3 +
                          ((FPTYPE)4. * a4 + (FPTYPE)5. * a5 * xx) * xx) *
                             xx) *
                            xx) *
                  ll * rr;
          dy_dem[ii * nnei_i * nnei_j + jj * nnei_j + kk] += res * rr;
        }
        dy_dem_x[ii * nnei_i * nnei_j + jj * nnei_j + kk] = grad;
      }
    }
  }
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_t_grad_grad_cpu(FPTYPE* dz_dy,
                                                const FPTYPE* table,
                                                const FPTYPE* table_info,
                                                const FPTYPE* em_x,
                                                const FPTYPE* em,
                                                const FPTYPE* dz_dy_dem_x,
                                                const FPTYPE* dz_dy_dem,
                                                const int nloc,
                                                const int nnei_i,
                                                const int nnei_j,
                                                const int last_layer_size) {
  memset(dz_dy, 0, sizeof(FPTYPE) * nloc * last_layer_size);
  const FPTYPE lower = table_info[0];
  const FPTYPE upper = table_info[1];
  const FPTYPE _max = table_info[2];
  const FPTYPE stride0 = table_info[3];
  const FPTYPE stride1 = table_info[4];
// for every atom, execute a small manual gemm ~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    for (int jj = 0; jj < nnei_i; jj++) {
      for (int kk = 0; kk < nnei_j; kk++) {
        FPTYPE xx = em_x[ii * nnei_i * nnei_j + jj * nnei_j + kk];
        FPTYPE tmp = xx;
        FPTYPE dz_em = dz_dy_dem[ii * nnei_i * nnei_j + jj * nnei_j + kk];
        FPTYPE dz_xx = dz_dy_dem_x[ii * nnei_i * nnei_j + jj * nnei_j + kk];

        int table_idx = 0;
        locate_xx_se_t(lower, upper, -_max, _max, stride0, stride1, xx,
                       table_idx);
        for (int mm = 0; mm < last_layer_size; mm++) {
          FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * mm + 0];
          FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * mm + 1];
          FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * mm + 2];
          FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * mm + 3];
          FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * mm + 4];
          FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * mm + 5];
          FPTYPE var =
              a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
          FPTYPE var_grad =
              a1 + ((FPTYPE)2. * a2 +
                    ((FPTYPE)3. * a3 +
                     ((FPTYPE)4. * a4 + (FPTYPE)5. * a5 * xx) * xx) *
                        xx) *
                       xx;

          dz_dy[ii * last_layer_size + mm] +=
              var * dz_em + dz_xx * var_grad * tmp;
        }
      }
    }
  }
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_r_cpu(FPTYPE* out,
                                      const FPTYPE* table,
                                      const FPTYPE* table_info,
                                      const FPTYPE* em,
                                      const int nloc,
                                      const int nnei,
                                      const int last_layer_size) {
  memset(out, 0, sizeof(FPTYPE) * nloc * nnei * last_layer_size);
  const FPTYPE lower = table_info[0];
  const FPTYPE upper = table_info[1];
  const FPTYPE _max = table_info[2];
  const FPTYPE stride0 = table_info[3];
  const FPTYPE stride1 = table_info[4];
// for every atom, execute a small manual gemm ~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    for (int jj = 0; jj < nnei; jj++) {
      FPTYPE xx = em[ii * nnei + jj];
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      for (int kk = 0; kk < last_layer_size; kk++) {
        FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * kk + 0];
        FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * kk + 1];
        FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * kk + 2];
        FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
        FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
        FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
        out[ii * last_layer_size * nnei + jj * last_layer_size + kk] =
            a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
      }
    }
  }
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_r_grad_cpu(FPTYPE* dy_dem,
                                           const FPTYPE* table,
                                           const FPTYPE* table_info,
                                           const FPTYPE* em,
                                           const FPTYPE* dy,
                                           const int nloc,
                                           const int nnei,
                                           const int last_layer_size) {
  memset(dy_dem, 0, sizeof(FPTYPE) * nloc * nnei);
  FPTYPE const lower = table_info[0];
  FPTYPE const upper = table_info[1];
  FPTYPE const _max = table_info[2];
  FPTYPE const stride0 = table_info[3];
  FPTYPE const stride1 = table_info[4];
// for every atom, execute a small gemm~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    for (int jj = 0; jj < nnei; jj++) {
      // construct the dy/dx
      FPTYPE xx = em[ii * nnei + jj];
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      FPTYPE grad = (FPTYPE)0.0;
      for (int kk = 0; kk < last_layer_size; kk++) {
        FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * kk + 0];
        FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * kk + 1];
        FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * kk + 2];
        FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
        FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
        FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
        grad += (a1 + ((FPTYPE)2. * a2 +
                       ((FPTYPE)3. * a3 +
                        ((FPTYPE)4. * a4 + (FPTYPE)5. * a5 * xx) * xx) *
                           xx) *
                          xx) *
                dy[ii * last_layer_size * nnei + jj * last_layer_size + kk];
      }
      dy_dem[ii * nnei + jj] = grad;
    }
  }
}

template <typename FPTYPE>
void deepmd::tabulate_fusion_se_r_grad_grad_cpu(FPTYPE* dz_dy,
                                                const FPTYPE* table,
                                                const FPTYPE* table_info,
                                                const FPTYPE* em,
                                                const FPTYPE* dz_dy_dem,
                                                const int nloc,
                                                const int nnei,
                                                const int last_layer_size) {
  memset(dz_dy, 0, sizeof(FPTYPE) * nloc * nnei * last_layer_size);
  const FPTYPE lower = table_info[0];
  const FPTYPE upper = table_info[1];
  const FPTYPE _max = table_info[2];
  const FPTYPE stride0 = table_info[3];
  const FPTYPE stride1 = table_info[4];
// for every atom, execute a small manual gemm ~
// FPTYPE * res = new FPTYPE[4 * last_layer_size];
#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    for (int jj = 0; jj < nnei; jj++) {
      FPTYPE xx = em[ii * nnei + jj];
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      for (int kk = 0; kk < last_layer_size; kk++) {
        FPTYPE a0 = table[table_idx * last_layer_size * 6 + 6 * kk + 0];
        FPTYPE a1 = table[table_idx * last_layer_size * 6 + 6 * kk + 1];
        FPTYPE a2 = table[table_idx * last_layer_size * 6 + 6 * kk + 2];
        FPTYPE a3 = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
        FPTYPE a4 = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
        FPTYPE a5 = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
        FPTYPE var_grad =
            a1 +
            ((FPTYPE)2. * a2 +
             ((FPTYPE)3. * a3 + ((FPTYPE)4. * a4 + (FPTYPE)5. * a5 * xx) * xx) *
                 xx) *
                xx;
        dz_dy[ii * last_layer_size * nnei + jj * last_layer_size + kk] =
            dz_dy_dem[ii * nnei + jj] * var_grad;
      }
    }
  }
}

template void deepmd::tabulate_fusion_se_a_cpu<float>(float* out,
                                                      const float* table,
                                                      const float* table_info,
                                                      const float* em_x,
                                                      const float* em,
                                                      const float* two_embed,
                                                      const int nloc,
                                                      const int nnei,
                                                      const int last_layer_size,
                                                      const bool is_sorted);
template void deepmd::tabulate_fusion_se_a_cpu<double>(
    double* out,
    const double* table,
    const double* table_info,
    const double* em_x,
    const double* em,
    const double* two_embed,
    const int nloc,
    const int nnei,
    const int last_layer_size,
    const bool is_sorted);
template void deepmd::tabulate_fusion_se_a_grad_cpu<float>(
    float* dy_dem_x,
    float* dy_dem,
    float* dy_dtwo,
    const float* table,
    const float* table_info,
    const float* em_x,
    const float* em,
    const float* two_embed,
    const float* dy,
    const int nloc,
    const int nnei,
    const int last_layer_size,
    const bool is_sorted);
template void deepmd::tabulate_fusion_se_a_grad_cpu<double>(
    double* dy_dem_x,
    double* dy_dem,
    double* dy_dtwo,
    const double* table,
    const double* table_info,
    const double* em_x,
    const double* em,
    const double* two_embed,
    const double* dy,
    const int nloc,
    const int nnei,
    const int last_layer_size,
    const bool is_sorted);
template void deepmd::tabulate_fusion_se_a_grad_grad_cpu<float>(
    float* dz_dy,
    const float* table,
    const float* table_info,
    const float* em_x,
    const float* em,
    const float* two_embed,
    const float* dz_dy_dem_x,
    const float* dz_dy_dem,
    const float* dz_dy_dtwo,
    const int nloc,
    const int nnei,
    const int last_layer_size,
    const bool is_sorted);
template void deepmd::tabulate_fusion_se_a_grad_grad_cpu<double>(
    double* dz_dy,
    const double* table,
    const double* table_info,
    const double* em_x,
    const double* em,
    const double* two_embed,
    const double* dz_dy_dem_x,
    const double* dz_dy_dem,
    const double* dz_dy_dtwo,
    const int nloc,
    const int nnei,
    const int last_layer_size,
    const bool is_sorted);

template void deepmd::tabulate_fusion_se_t_cpu<float>(
    float* out,
    const float* table,
    const float* table_info,
    const float* em_x,
    const float* em,
    const int nloc,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_t_cpu<double>(
    double* out,
    const double* table,
    const double* table_info,
    const double* em_x,
    const double* em,
    const int nloc,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_t_grad_cpu<float>(
    float* dy_dem_x,
    float* dy_dem,
    const float* table,
    const float* table_info,
    const float* em_x,
    const float* em,
    const float* dy,
    const int nloc,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_t_grad_cpu<double>(
    double* dy_dem_x,
    double* dy_dem,
    const double* table,
    const double* table_info,
    const double* em_x,
    const double* em,
    const double* dy,
    const int nloc,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_t_grad_grad_cpu<float>(
    float* dz_dy,
    const float* table,
    const float* table_info,
    const float* em_x,
    const float* em,
    const float* dz_dy_dem_x,
    const float* dz_dy_dem,
    const int nloc,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_t_grad_grad_cpu<double>(
    double* dz_dy,
    const double* table,
    const double* table_info,
    const double* em_x,
    const double* em,
    const double* dz_dy_dem_x,
    const double* dz_dy_dem,
    const int nloc,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size);

template void deepmd::tabulate_fusion_se_r_cpu<float>(
    float* out,
    const float* table,
    const float* table_info,
    const float* em,
    const int nloc,
    const int nnei,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_r_cpu<double>(
    double* out,
    const double* table,
    const double* table_info,
    const double* em,
    const int nloc,
    const int nnei,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_r_grad_cpu<float>(
    float* dy_dem,
    const float* table,
    const float* table_info,
    const float* em,
    const float* dy,
    const int nloc,
    const int nnei,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_r_grad_cpu<double>(
    double* dy_dem,
    const double* table,
    const double* table_info,
    const double* em,
    const double* dy,
    const int nloc,
    const int nnei,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_r_grad_grad_cpu<float>(
    float* dz_dy,
    const float* table,
    const float* table_info,
    const float* em,
    const float* dz_dy_dem,
    const int nloc,
    const int nnei,
    const int last_layer_size);
template void deepmd::tabulate_fusion_se_r_grad_grad_cpu<double>(
    double* dz_dy,
    const double* table,
    const double* table_info,
    const double* em,
    const double* dz_dy_dem,
    const int nloc,
    const int nnei,
    const int last_layer_size);
