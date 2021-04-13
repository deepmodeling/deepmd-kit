#include <vector>
#include <cassert>
#include <iostream>
#include <string.h>
#include "tabulate.h"
/*
    This inline function was designed to get the table info and bias value for current input xx!
    lower:      indicate the lower boundary of the first table;
    upper:      indicate the upper boundary of the first table as well as the lower boundary of the second table;
    max:        indicate the upper boundary of the second table;
    stride0:    indicate the stride of the first table;
    stride1:    indicate the stride of the second table;
    xx:         indicate the inputs value;
    table_idx:  indicate the location of table info of input value xx;
*/
template <typename FPTYPE>
inline void locate_xx(
    const FPTYPE& lower, 
    const FPTYPE& upper,
    const FPTYPE& max, 
    const FPTYPE& stride0, 
    const FPTYPE& stride1, 
    FPTYPE& xx, 
    int& table_idx) 
{
  if (xx < lower) {
    table_idx = 0;
    xx = 0;
  }
  else if (xx < upper) {
    table_idx = (int)((xx - lower) / stride0);
    xx -= (table_idx * stride0 + lower);
  }
  else if (xx < max) {
    int first_stride = int((upper - lower) / stride0);
    table_idx = first_stride + (int)((xx - upper) / stride1);
    xx -= ((table_idx - first_stride) * stride1 + upper);
  }
  else {
    table_idx = int((upper - lower) / stride0) + (int)((max - upper) / stride1) - 1;
    xx = 0;
  }
}

template <typename FPTYPE>
inline FPTYPE dot(
    FPTYPE a[4], 
    FPTYPE b[4]) 
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; 
}

template<typename FPTYPE>
void deepmd::tabulate_fusion_cpu(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size)
{
  memset(out, 0.0, sizeof(FPTYPE) * nloc * 4 * last_layer_size);
  const FPTYPE lower   = table_info[0];
  const FPTYPE upper   = table_info[1];
  const FPTYPE _max    = table_info[2];
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
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      for (int kk = 0; kk < last_layer_size; kk++) {
        FPTYPE a0  = table[table_idx * last_layer_size * 6 + 6 * kk + 0]; 
        FPTYPE a1  = table[table_idx * last_layer_size * 6 + 6 * kk + 1]; 
        FPTYPE a2  = table[table_idx * last_layer_size * 6 + 6 * kk + 2]; 
        FPTYPE a3  = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
        FPTYPE a4  = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
        FPTYPE a5  = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
        FPTYPE var = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
        if (unloop) {
          out[ii * last_layer_size * 4 + 0 * last_layer_size + kk] += (nnei - jj) * var * ll[0];
          out[ii * last_layer_size * 4 + 1 * last_layer_size + kk] += (nnei - jj) * var * ll[1];
          out[ii * last_layer_size * 4 + 2 * last_layer_size + kk] += (nnei - jj) * var * ll[2];
          out[ii * last_layer_size * 4 + 3 * last_layer_size + kk] += (nnei - jj) * var * ll[3];
        }
        else {
          out[ii * last_layer_size * 4 + 0 * last_layer_size + kk] += var * ll[0];
          out[ii * last_layer_size * 4 + 1 * last_layer_size + kk] += var * ll[1];
          out[ii * last_layer_size * 4 + 2 * last_layer_size + kk] += var * ll[2];
          out[ii * last_layer_size * 4 + 3 * last_layer_size + kk] += var * ll[3];
        }
      }
      if (unloop) break;
    }
  }
}

template<typename FPTYPE>
void deepmd::tabulate_fusion_grad_cpu(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size) 
{
  memset(dy_dem_x, 0.0, sizeof(FPTYPE) * nloc * nnei);
  memset(dy_dem, 0.0, sizeof(FPTYPE) * nloc * nnei * 4);
  FPTYPE const lower   = table_info[0];
  FPTYPE const upper   = table_info[1];
  FPTYPE const _max    = table_info[2];
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
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      FPTYPE grad = 0.0;
      for (int kk = 0; kk < last_layer_size; kk++) {
        rr[0] = dy[ii * last_layer_size * 4 + 0 * last_layer_size + kk];
        rr[1] = dy[ii * last_layer_size * 4 + 1 * last_layer_size + kk];
        rr[2] = dy[ii * last_layer_size * 4 + 2 * last_layer_size + kk];
        rr[3] = dy[ii * last_layer_size * 4 + 3 * last_layer_size + kk];
        FPTYPE a0  = table[table_idx * last_layer_size * 6 + 6 * kk + 0]; 
        FPTYPE a1  = table[table_idx * last_layer_size * 6 + 6 * kk + 1]; 
        FPTYPE a2  = table[table_idx * last_layer_size * 6 + 6 * kk + 2]; 
        FPTYPE a3  = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
        FPTYPE a4  = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
        FPTYPE a5  = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
        FPTYPE res = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;

        if (unloop) {
          grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr) * (nnei - jj);
          dy_dem[ii * nnei * 4 + jj * 4 + 0] += res * rr[0] * (nnei - jj);
          dy_dem[ii * nnei * 4 + jj * 4 + 1] += res * rr[1] * (nnei - jj);
          dy_dem[ii * nnei * 4 + jj * 4 + 2] += res * rr[2] * (nnei - jj);
          dy_dem[ii * nnei * 4 + jj * 4 + 3] += res * rr[3] * (nnei - jj);
        }
        else {
          grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr);
          dy_dem[ii * nnei * 4 + jj * 4 + 0] += res * rr[0];
          dy_dem[ii * nnei * 4 + jj * 4 + 1] += res * rr[1];
          dy_dem[ii * nnei * 4 + jj * 4 + 2] += res * rr[2];
          dy_dem[ii * nnei * 4 + jj * 4 + 3] += res * rr[3];
        }
      }
      dy_dem_x[ii * nnei + jj] = grad;
      if (unloop) break;
    }
  }
}

template void deepmd::tabulate_fusion_cpu<float>(float * out, const float * table, const float * table_info, const float * em_x, const float * em, const int nloc, const int nnei, const int last_layer_size);
template void deepmd::tabulate_fusion_cpu<double>(double * out, const double * table, const double * table_info, const double * em_x, const double * em, const int nloc, const int nnei, const int last_layer_size);
template void deepmd::tabulate_fusion_grad_cpu<float> (float * dy_dem_x, float * dy_dem, const float * table, const float * table_info, const float * em_x, const float * em, const float * dy, const int nloc, const int nnei, const int last_layer_size); 
template void deepmd::tabulate_fusion_grad_cpu<double> (double * dy_dem_x, double * dy_dem, const double * table, const double * table_info, const double * em_x, const double * em, const double * dy, const int nloc, const int nnei, const int last_layer_size);
