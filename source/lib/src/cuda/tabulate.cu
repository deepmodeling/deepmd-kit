#include "device.h"
#include "tabulate.h"

#define MM 4
#define KK 4
#define TPB 256
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

template <typename FPTYPE>
__forceinline__ __device__
void locate_xx(
    FPTYPE& xx, 
    int& table_idx,
    const FPTYPE& lower, 
    const FPTYPE& upper,  
    const FPTYPE& max, 
    const FPTYPE& stride0, 
    const FPTYPE& stride1)
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
__forceinline__ __device__ 
FPTYPE dot(
    FPTYPE ll[4], 
    FPTYPE rr[4]) 
{
  return ll[0] * rr[0] + ll[1] * rr[1] + ll[2] * rr[2] + ll[3] * rr[3];
}

template <typename FPTYPE>
__forceinline__ 
__device__
void warp_reduce(
    FPTYPE & val) 
{
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(FULL_MASK, val, offset);
}

template <
    typename FPTYPE,
    int      MTILE,
    int      KTILE> 
__global__ void tabulate_fusion_fifth_order_polynomial(
    FPTYPE * out, 
    const FPTYPE * table, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE lower, 
    const FPTYPE upper, 
    const FPTYPE max, 
    const FPTYPE stride0, 
    const FPTYPE stride1, 
    const int nnei, 
    const int last_layer_size) 
{
  const int block_idx = blockIdx.x;   // nloc
  const int thread_idx = threadIdx.x; // last_layer_size
  FPTYPE ago = __shfl_sync(0xffffffff, em_x[block_idx * nnei + nnei - 1], 0);
  bool unloop = false;
  int breakpoint = nnei - 1;

  FPTYPE sum[MTILE] = {0.f};
  for (int ii = 0; ii < nnei; ii++) {
    FPTYPE var[6]; 
    FPTYPE xx = em_x[block_idx * nnei + ii];
    if (xx == ago) {
      unloop = true;
      breakpoint = ii;
    }
    int table_idx = 0;
    locate_xx(xx, table_idx, lower, upper, max, stride0, stride1);
    var[0] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 0];
    var[1] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 1];
    var[2] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 2];
    var[3] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 3];
    var[4] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 4];
    var[5] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 5];
    FPTYPE res = var[0] + (var[1] + (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) * xx;
    
    for (int kk = 0; kk < MTILE; kk++) {
      sum[kk] += (nnei - breakpoint) * em[block_idx * nnei * MTILE + ii * MTILE + kk] * res;
    }
    if (unloop) break;
  }
  for (int ii = 0; ii < MTILE; ii++) {
    out[block_idx * MTILE * last_layer_size + ii * last_layer_size + thread_idx] = sum[ii];
  }
}

template <
    typename FPTYPE,
    int      MTILE,
    int      KTILE> 
__global__ void tabulate_fusion_grad_fifth_order_polynomial(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,   
    const FPTYPE * table, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const FPTYPE lower, 
    const FPTYPE upper, 
    const FPTYPE max, 
    const FPTYPE stride0, 
    const FPTYPE stride1, 
    const int nnei, 
    const int last_layer_size) 
{
  extern __shared__ int _data[];
  const int block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x; // KTILE * WARP_SIZE, usally 128 here~
  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);
  int lane_idx = threadIdx.x % WARP_SIZE;
  int breakpoint = nnei - 1;
  bool unloop = false;
  FPTYPE * iteratorA = (FPTYPE *)&_data[0]; // dy
  for (int ii = 0; ii < MTILE; ii++) {
    for (int jj = thread_idx; jj < last_layer_size; jj += blockDim.x) {
      iteratorA[ii * last_layer_size + jj] = dy[block_idx * MTILE * last_layer_size + ii * last_layer_size + jj];
    }
  }
  __syncthreads();
  FPTYPE ago = __shfl_sync(0xffffffff, em_x[block_idx * nnei + nnei - 1], 0);
  for (int ii = warp_idx; ii < nnei; ii += KTILE) {
    FPTYPE xx = em_x[block_idx * nnei + ii];
    if (ago == xx) { 
      unloop = true;
      breakpoint = ii;
    }
    
    int table_idx = 0;
    locate_xx(xx, table_idx, lower, upper, max, stride0, stride1);
    FPTYPE sum[MTILE] = {0.f};
    FPTYPE Csub = 0.f;
    for (int jj = lane_idx; jj < last_layer_size; jj += WARP_SIZE) {
      FPTYPE var[6]; 
      // load iteratorB through table 
      var[0]  = table[table_idx * last_layer_size * 6 + 6 * jj + 0]; 
      var[1]  = table[table_idx * last_layer_size * 6 + 6 * jj + 1]; 
      var[2]  = table[table_idx * last_layer_size * 6 + 6 * jj + 2]; 
      var[3]  = table[table_idx * last_layer_size * 6 + 6 * jj + 3];
      var[4]  = table[table_idx * last_layer_size * 6 + 6 * jj + 4];
      var[5]  = table[table_idx * last_layer_size * 6 + 6 * jj + 5];
      FPTYPE res = var[0] + (var[1] + (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) * xx;
      
      for (int kk = 0; kk < MTILE; kk++) {
        sum[kk] += (nnei - breakpoint) * iteratorA[kk * last_layer_size + jj] * res;
      }
      res  = em[block_idx * nnei * MTILE + ii * 4 + 0] * iteratorA[0 * last_layer_size + jj];
      res += em[block_idx * nnei * MTILE + ii * 4 + 1] * iteratorA[1 * last_layer_size + jj];
      res += em[block_idx * nnei * MTILE + ii * 4 + 2] * iteratorA[2 * last_layer_size + jj];
      res += em[block_idx * nnei * MTILE + ii * 4 + 3] * iteratorA[3 * last_layer_size + jj];
      Csub += (nnei - breakpoint) * (var[1] + (2 * var[2] + (3 * var[3] + (4 * var[4] + 5 * var[5] * xx) * xx) * xx) * xx) * res;
    }
    __syncwarp();
    for (int kk = 0; kk < MTILE; kk++) {
      warp_reduce(sum[kk]);
    }
    warp_reduce(Csub);
    if (lane_idx == 0) {
      for (int kk = 0; kk < MTILE; kk++) {
        dy_dem[block_idx * nnei * MTILE + ii * 4 + kk] = sum[kk];
      }
      dy_dem_x[block_idx * nnei + ii] = Csub;
    }
    if (unloop) break;
  }
}

template <
    typename FPTYPE,
    int      MTILE,
    int      KTILE>
__global__ void tabulate_fusion_grad_grad_fifth_order_polynomial(
    FPTYPE * dz_dy,
    const FPTYPE * table,
    const FPTYPE * em_x,
    const FPTYPE * em,
    const FPTYPE * dz_dy_dem_x,
    const FPTYPE * dz_dy_dem,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei,
    const int last_layer_size)
{
  extern __shared__ int _data[];
  const int block_idx = blockIdx.x;   // nloc
  const int thread_idx = threadIdx.x; // last_layer_size
  FPTYPE ago = __shfl_sync(0xffffffff, em_x[block_idx * nnei + nnei - 1], 0);
  bool unloop = false;
  int breakpoint = nnei - 1;
  FPTYPE * iteratorC = (FPTYPE*) &_data[0];
  for (int kk = 0; kk < MTILE; kk++)
    iteratorC[kk * last_layer_size + thread_idx] = 0.f;
  __syncthreads();

  for (int ii = 0; ii < nnei; ii++) {
    FPTYPE var[6];
    FPTYPE xx = em_x[block_idx * nnei + ii];
    FPTYPE dz_xx = dz_dy_dem_x[block_idx * nnei + ii];
    if (xx == ago) {
      unloop = true;
      breakpoint = ii;
    }
    int table_idx = 0;
    locate_xx(xx, table_idx, lower, upper, max, stride0, stride1);
    var[0] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 0];
    var[1] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 1];
    var[2] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 2];
    var[3] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 3];
    var[4] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 4];
    var[5] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 5];
    FPTYPE res = var[0] + (var[1] + (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) * xx;
    FPTYPE res_grad = var[1] + (2 * var[2] + (3 * var[3] + (4 * var[4] + 5 * var[5] * xx) * xx) * xx) * xx;

    for (int kk = 0; kk < MTILE; kk++) {
      int em_index = block_idx * nnei * MTILE + ii * MTILE + kk;
      iteratorC[kk * last_layer_size + thread_idx] += (nnei - breakpoint) * (em[em_index] * res_grad * dz_xx + dz_dy_dem[em_index] * res);
    }
    if (unloop) break;
  }
  for (int ii = 0; ii < MTILE; ii++) {
    dz_dy[block_idx * MTILE * last_layer_size + ii * last_layer_size + thread_idx] = iteratorC[ii * last_layer_size + thread_idx];
  }
}

namespace deepmd {
template<typename FPTYPE>
void tabulate_fusion_gpu_cuda(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc,
    const int nnei, 
    const int last_layer_size) 
{
  if (nloc <= 0) {return;}
  tabulate_fusion_fifth_order_polynomial<FPTYPE, MM, KK> <<<nloc, last_layer_size>>>(
      out, 
      table, em_x, em, table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template<typename FPTYPE>
void tabulate_fusion_grad_gpu_cuda(
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
  if (nloc <= 0) {return;}
  DPErrcheck(cudaMemset(
      dy_dem_x,
      0.0, sizeof(FPTYPE) * nloc * nnei));
  DPErrcheck(cudaMemset(
      dy_dem,
      0.0, sizeof(FPTYPE) * nloc * nnei * 4));

  tabulate_fusion_grad_fifth_order_polynomial<FPTYPE, MM, KK> <<<nloc, KK * WARP_SIZE, sizeof(FPTYPE) * MM * last_layer_size>>>(
      dy_dem_x, dy_dem,
      table, em_x, em, dy,  table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template<typename FPTYPE>
void tabulate_fusion_grad_grad_gpu_cuda(
    FPTYPE * dz_dy,
    const FPTYPE * table,
    const FPTYPE * table_info,
    const FPTYPE * em_x,
    const FPTYPE * em,
    const FPTYPE * dz_dy_dem_x,
    const FPTYPE * dz_dy_dem,
    const int nloc,
    const int nnei,
    const int last_layer_size)
{
  if (nloc <= 0) {return;}
  DPErrcheck(cudaMemset(
    dz_dy,
    0.0, sizeof(FPTYPE) * nloc * 4 * last_layer_size));
  tabulate_fusion_grad_grad_fifth_order_polynomial<FPTYPE, MM, KK> <<<nloc, last_layer_size, sizeof(FPTYPE) * MM * last_layer_size>>>(
      dz_dy,
      table, em_x, em, dz_dy_dem_x, dz_dy_dem, table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template void tabulate_fusion_gpu_cuda<float>(float * out, const float * table, const float * table_info, const float * em_x, const float * em, const int nloc, const int nnei, const int last_layer_size);
template void tabulate_fusion_gpu_cuda<double>(double * out, const double * table, const double * table_info, const double * em_x, const double * em, const int nloc, const int nnei, const int last_layer_size);
template void tabulate_fusion_grad_gpu_cuda<float> (float * dy_dem_x, float * dy_dem, const float * table, const float * table_info, const float * em_x, const float * em, const float * dy, const int nloc, const int nnei, const int last_layer_size); 
template void tabulate_fusion_grad_gpu_cuda<double> (double * dy_dem_x, double * dy_dem, const double * table, const double * table_info, const double * em_x, const double * em, const double * dy, const int nloc, const int nnei, const int last_layer_size);
template void tabulate_fusion_grad_grad_gpu_cuda<float> (float * dz_dy, const float * table, const float * table_info, const float * em_x, const float * em, const float * dz_dy_dem_x, const float * dz_dy_dem, const int nloc, const int nnei, const int last_layer_size);
template void tabulate_fusion_grad_grad_gpu_cuda<double> (double * dz_dy, const double * table, const double * table_info, const double * em_x, const double * em, const double * dz_dy_dem_x, const double * dz_dy_dem, const int nloc, const int nnei, const int last_layer_size);
}
