#include "device.h"
#include "tabulate.h"

#define MM 4
#define KK 4
#define TPB 256
#if GOOGLE_CUDA
#define WARP_SIZE 32
#elif TENSORFLOW_USE_ROCM
// See https://github.com/pytorch/pytorch/pull/64302
#define WARP_SIZE warpSize  // = 64 or 32 (Defined in hip_runtime.h)
#else
#error "should not touch here"
#endif
#define FULL_MASK 0xffffffff

#if GOOGLE_CUDA
#define GPU_DYNAMIC_SHARED_MEM_DECL(TYPE, NAME) extern __shared__ TYPE NAME[]
#elif TENSORFLOW_USE_ROCM
#define GPU_DYNAMIC_SHARED_MEM_DECL(TYPE, NAME) HIP_DYNAMIC_SHARED(TYPE, NAME)
#else
#error "should not touch here"
#endif

// Copyright 2017 The TensorFlow Authors.
// Licensed under the Apache License, Version 2.0
template <typename T>
__device__ T
GpuShuffleSync(unsigned mask, T value, int src_lane, int width = warpSize) {
#if GOOGLE_CUDA
  return __shfl_sync(mask, value, src_lane, width);
#elif TENSORFLOW_USE_ROCM
  return __shfl(value, src_lane, width);
#else
#error "should not touch here"
#endif
}

__device__ void GpuSyncThreads() {
#if GOOGLE_CUDA
  __syncwarp();
#elif TENSORFLOW_USE_ROCM
  //__syncwarp();->syncwrap
  __syncthreads();
#else
#error "should not touch here"
#endif
}

template <typename FPTYPE>
__forceinline__ __device__ void locate_xx_se_a(FPTYPE& xx,
                                               int& table_idx,
                                               const FPTYPE& lower,
                                               const FPTYPE& upper,
                                               const FPTYPE& max,
                                               const FPTYPE& stride0,
                                               const FPTYPE& stride1) {
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
__forceinline__ __device__ void locate_xx_se_t(FPTYPE& xx,
                                               int& table_idx,
                                               const FPTYPE& lower,
                                               const FPTYPE& upper,
                                               const FPTYPE& min,
                                               const FPTYPE& max,
                                               const FPTYPE& stride0,
                                               const FPTYPE& stride1) {
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
__forceinline__ __device__ void locate_xx_se_r(FPTYPE& xx,
                                               int& table_idx,
                                               const FPTYPE& lower,
                                               const FPTYPE& upper,
                                               const FPTYPE& max,
                                               const FPTYPE& stride0,
                                               const FPTYPE& stride1) {
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
__forceinline__ __device__ void load_polynomial_params(
    FPTYPE var[6],
    const FPTYPE* table,
    const int& table_idx,
    const int& idx,
    const int& last_layer_size) {
  var[0] = table[table_idx * last_layer_size * 6 + idx * 6 + 0];
  var[1] = table[table_idx * last_layer_size * 6 + idx * 6 + 1];
  var[2] = table[table_idx * last_layer_size * 6 + idx * 6 + 2];
  var[3] = table[table_idx * last_layer_size * 6 + idx * 6 + 3];
  var[4] = table[table_idx * last_layer_size * 6 + idx * 6 + 4];
  var[5] = table[table_idx * last_layer_size * 6 + idx * 6 + 5];
}

template <typename FPTYPE>
__forceinline__ __device__ FPTYPE dot(FPTYPE ll[4], FPTYPE rr[4]) {
  return ll[0] * rr[0] + ll[1] * rr[1] + ll[2] * rr[2] + ll[3] * rr[3];
}

template <typename FPTYPE>
__forceinline__ __device__ void warp_reduce(FPTYPE& val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#if GOOGLE_CUDA
    val += __shfl_down_sync(FULL_MASK, val, offset);
#elif TENSORFLOW_USE_ROCM
    val += __shfl_down(val, offset);  // ########????
#else
#error "should not touch here"
#endif
  }
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_a_fifth_order_polynomial(
    FPTYPE* out,
    const FPTYPE* table,
    const FPTYPE* em_x,
    const FPTYPE* em,
    const FPTYPE* two_embed,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei,
    const int last_layer_size,
    const bool is_sorted) {
  bool enable_se_atten = two_embed != nullptr;
#if TENSORFLOW_USE_ROCM
  GPU_DYNAMIC_SHARED_MEM_DECL(int, _data)
#endif
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // last_layer_size
  FPTYPE ago = GpuShuffleSync(0xffffffff, em_x[block_idx * nnei + nnei - 1], 0);
  bool unloop = false;
  int breakpoint = nnei - 1;
#if GOOGLE_CUDA
  FPTYPE sum[MTILE] = {(FPTYPE)0.};
#elif TENSORFLOW_USE_ROCM
  FPTYPE* iteratorC = (FPTYPE*)&_data[0];
  for (int kk = 0; kk < MTILE; kk++) {
    iteratorC[kk * last_layer_size + thread_idx] = (FPTYPE)0.;
  }
  __syncthreads();
#else
#error "should not touch here"
#endif
  int mark_table_idx = -1;
  FPTYPE var[6];
  for (int ii = 0; ii < nnei; ii++) {
    FPTYPE xx = em_x[block_idx * nnei + ii];
    if (xx == ago && em[block_idx * nnei * 4 + ii * 4 + 1] == 0. &&
        em[block_idx * nnei * 4 + ii * 4 + 2] == 0. &&
        em[block_idx * nnei * 4 + ii * 4 + 3] == 0. && is_sorted) {
      unloop = true;
      breakpoint = ii;
    }
    int table_idx = 0;
    locate_xx_se_a(xx, table_idx, lower, upper, max, stride0, stride1);
    if (table_idx != mark_table_idx) {
      load_polynomial_params(var, table, table_idx, thread_idx,
                             last_layer_size);
    }
    FPTYPE res =
        var[0] +
        (var[1] + (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) *
            xx;
    if (enable_se_atten) {
      FPTYPE t = two_embed[block_idx * nnei * last_layer_size +
                           ii * last_layer_size + thread_idx];
      res = res * t + res;
    }

    for (int kk = 0; kk < MTILE; kk++) {
#if GOOGLE_CUDA
      sum[kk]
#elif TENSORFLOW_USE_ROCM
      iteratorC[kk * last_layer_size + thread_idx]
#else
#error "should not touch here"
#endif
          += (nnei - breakpoint) *
             em[block_idx * nnei * MTILE + ii * MTILE + kk] * res;
    }
    if (unloop) {
      break;
    }
    mark_table_idx = table_idx;
  }
  for (int ii = 0; ii < MTILE; ii++) {
    out[block_idx * MTILE * last_layer_size + ii * last_layer_size +
        thread_idx] =
#if GOOGLE_CUDA
        sum[ii];
#elif TENSORFLOW_USE_ROCM
        iteratorC[ii * last_layer_size + thread_idx];
#else
#error "should not touch here"
#endif
  }
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_a_grad_fifth_order_polynomial(
    FPTYPE* dy_dem_x,
    FPTYPE* dy_dem,
    FPTYPE* dy_dtwo,
    const FPTYPE* table,
    const FPTYPE* em_x,
    const FPTYPE* em,
    const FPTYPE* two_embed,
    const FPTYPE* dy,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei,
    const int last_layer_size,
    const bool is_sorted) {
  bool enable_se_atten = two_embed != nullptr;
  GPU_DYNAMIC_SHARED_MEM_DECL(int, _data);
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // KTILE * WARP_SIZE, usally 128 here~
  int warp_idx = GpuShuffleSync(0xffffffff, threadIdx.x / WARP_SIZE, 0);
  int lane_idx = threadIdx.x % WARP_SIZE;
  int breakpoint = nnei - 1;
  bool unloop = false;
  FPTYPE* iteratorA = (FPTYPE*)&_data[0];  // dy
  for (int ii = 0; ii < MTILE; ii++) {
    for (int jj = thread_idx; jj < last_layer_size; jj += blockDim.x) {
      iteratorA[ii * last_layer_size + jj] =
          dy[block_idx * MTILE * last_layer_size + ii * last_layer_size + jj];
    }
  }
  __syncthreads();
  FPTYPE ago = GpuShuffleSync(0xffffffff, em_x[block_idx * nnei + nnei - 1], 0);
  for (int ii = warp_idx; ii < nnei; ii += KTILE) {
    FPTYPE xx = em_x[block_idx * nnei + ii];
    if (ago == xx && em[block_idx * nnei * 4 + ii * 4 + 1] == 0. &&
        em[block_idx * nnei * 4 + ii * 4 + 2] == 0. &&
        em[block_idx * nnei * 4 + ii * 4 + 3] == 0. && is_sorted) {
      unloop = true;
      breakpoint = ii;
    }

    int table_idx = 0;
    FPTYPE reg_em[MTILE] = {em[block_idx * nnei * MTILE + ii * 4 + 0],
                            em[block_idx * nnei * MTILE + ii * 4 + 1],
                            em[block_idx * nnei * MTILE + ii * 4 + 2],
                            em[block_idx * nnei * MTILE + ii * 4 + 3]};
    FPTYPE Csub = (FPTYPE)0.;
    FPTYPE sum[MTILE] = {(FPTYPE)0.};
    locate_xx_se_a(xx, table_idx, lower, upper, max, stride0, stride1);

    FPTYPE var[6];
    for (int jj = lane_idx; jj < last_layer_size; jj += WARP_SIZE) {
      load_polynomial_params(var, table, table_idx, jj, last_layer_size);
      FPTYPE res =
          var[0] +
          (var[1] +
           (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) *
              xx;
      FPTYPE oldres = res;
      FPTYPE t;
      if (enable_se_atten) {
        t = two_embed[block_idx * nnei * last_layer_size +
                      ii * last_layer_size + jj];
        res = res * t + res;
      }

      for (int kk = 0; kk < MTILE; kk++) {
        sum[kk] +=
            (nnei - breakpoint) * iteratorA[kk * last_layer_size + jj] * res;
      }
      res = reg_em[0] * iteratorA[0 * last_layer_size + jj];
      res += reg_em[1] * iteratorA[1 * last_layer_size + jj];
      res += reg_em[2] * iteratorA[2 * last_layer_size + jj];
      res += reg_em[3] * iteratorA[3 * last_layer_size + jj];
      Csub +=
          (nnei - breakpoint) *
          (var[1] + ((FPTYPE)2. * var[2] +
                     ((FPTYPE)3. * var[3] +
                      ((FPTYPE)4. * var[4] + (FPTYPE)5. * var[5] * xx) * xx) *
                         xx) *
                        xx) *
          (enable_se_atten ? res * t + res : res);
      if (enable_se_atten) {
        // from ii to ii + (nnei - breakpoint)
        for (int ii2 = ii; ii2 < ii + nnei - breakpoint; ii2++) {
          dy_dtwo[block_idx * nnei * last_layer_size + ii2 * last_layer_size +
                  jj] = oldres * res;
        }
      }
    }
    GpuSyncThreads();
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
    if (unloop) {
      break;
    }
  }
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_a_grad_grad_fifth_order_polynomial(
    FPTYPE* dz_dy,
    const FPTYPE* table,
    const FPTYPE* em_x,
    const FPTYPE* em,
    const FPTYPE* two_embed,
    const FPTYPE* dz_dy_dem_x,
    const FPTYPE* dz_dy_dem,
    const FPTYPE* dz_dy_dtwo,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei,
    const int last_layer_size,
    const bool is_sorted) {
  bool enable_se_atten = two_embed != nullptr;
  GPU_DYNAMIC_SHARED_MEM_DECL(int, _data);
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // last_layer_size
  FPTYPE ago = GpuShuffleSync(0xffffffff, em_x[block_idx * nnei + nnei - 1], 0);
  bool unloop = false;
  int breakpoint = nnei - 1;
  FPTYPE* iteratorC = (FPTYPE*)&_data[0];
  for (int kk = 0; kk < MTILE; kk++) {
    iteratorC[kk * last_layer_size + thread_idx] = (FPTYPE)0.;
  }
  __syncthreads();

  int mark_table_idx = -1;
  FPTYPE var[6];
  for (int ii = 0; ii < nnei; ii++) {
    FPTYPE xx = em_x[block_idx * nnei + ii];
    FPTYPE dz_xx = dz_dy_dem_x[block_idx * nnei + ii];
    if (xx == ago && em[block_idx * nnei * 4 + ii * 4 + 1] == 0. &&
        em[block_idx * nnei * 4 + ii * 4 + 2] == 0. &&
        em[block_idx * nnei * 4 + ii * 4 + 3] == 0. && is_sorted) {
      unloop = true;
      breakpoint = ii;
    }
    int table_idx = 0;
    locate_xx_se_a(xx, table_idx, lower, upper, max, stride0, stride1);
    if (table_idx != mark_table_idx) {
      load_polynomial_params(var, table, table_idx, thread_idx,
                             last_layer_size);
    }

    FPTYPE res =
        var[0] +
        (var[1] + (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) *
            xx;
    FPTYPE res_grad =
        var[1] + ((FPTYPE)2. * var[2] +
                  ((FPTYPE)3. * var[3] +
                   ((FPTYPE)4. * var[4] + (FPTYPE)5. * var[5] * xx) * xx) *
                      xx) *
                     xx;
    FPTYPE two_grad = 0.;
    if (enable_se_atten) {
      FPTYPE t = two_embed[block_idx * nnei * last_layer_size +
                           ii * last_layer_size + thread_idx];
      // dz_dy_dtwo * res * em
      // res above should be used instead of res + res * t below
      two_grad = dz_dy_dtwo[block_idx * nnei * last_layer_size +
                            ii * last_layer_size + thread_idx] *
                 res;
      res += res * t;
      res_grad += res_grad * t;
    }

    /*
     * `dz_dy`(or `iteratorC`) represents the derivative of the variable `out`
     * in the function `tabulate_fusion_se_a_fifth_order_polynomial`.
     *
     * The expression `em[em_index] * res_grad * dz_xx + dz_dy_dem[em_index] *
     * res` utilizes the product rule of derivatives: `(f * g)' = f' * g + f *
     * g'`.
     *
     * This expression can be alternatively expressed as:
     * `dz_dy_dem[em_index] * res + em[em_index] * (res_grad * dz_xx)`.
     * Note that we can refer to `dz_dy_dem` as `em'`
     *
     * Therefore, we can rewrite this expression as: `em' * res + em * res'`,
     * where `em'` is the derivative of `em` and `res'` is the derivative of
     * `res`. Additionally, `res'` can be further represented as: `res_grad *
     * dz_xx`.
     *
     * If `enable_se_atten` is true, `res` will be `res * t + res`, and `res'`
     * will become `(res_grad * t + res_grad) * dz_xx`.
     */
    for (int kk = 0; kk < MTILE; kk++) {
      int em_index = block_idx * nnei * MTILE + ii * MTILE + kk;
      iteratorC[kk * last_layer_size + thread_idx] +=
          (nnei - breakpoint) * (em[em_index] * (res_grad * dz_xx + two_grad) +
                                 dz_dy_dem[em_index] * res);
    }
    mark_table_idx = table_idx;
    if (unloop) {
      break;
    }
  }
  for (int ii = 0; ii < MTILE; ii++) {
    dz_dy[block_idx * MTILE * last_layer_size + ii * last_layer_size +
          thread_idx] = iteratorC[ii * last_layer_size + thread_idx];
  }
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_t_fifth_order_polynomial(
    FPTYPE* out,
    const FPTYPE* table,
    const FPTYPE* em_x,
    const FPTYPE* em,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size) {
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // last_layer_size

  FPTYPE sum = (FPTYPE)0.;
  for (int ii = 0; ii < nnei_i; ii++) {
    FPTYPE var[6];
    int mark_table_idx = -1;
    for (int jj = 0; jj < nnei_j; jj++) {
      FPTYPE xx = em_x[block_idx * nnei_i * nnei_j + ii * nnei_j + jj];
      FPTYPE tmp = xx;
      int table_idx = 0;
      locate_xx_se_t(xx, table_idx, lower, upper, -max, max, stride0, stride1);
      if (table_idx != mark_table_idx) {
        load_polynomial_params(var, table, table_idx, thread_idx,
                               last_layer_size);
      }
      FPTYPE res =
          var[0] +
          (var[1] +
           (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) *
              xx;

      sum += tmp * res;
      mark_table_idx = table_idx;
    }
  }
  out[block_idx * last_layer_size + thread_idx] = sum;
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_t_grad_fifth_order_polynomial(
    FPTYPE* dy_dem_x,
    FPTYPE* dy_dem,
    const FPTYPE* table,
    const FPTYPE* em_x,
    const FPTYPE* em,
    const FPTYPE* dy,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size) {
  GPU_DYNAMIC_SHARED_MEM_DECL(int, _data);
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // KTILE * WARP_SIZE, usally 128 here~
  int warp_idx = GpuShuffleSync(0xffffffff, threadIdx.x / WARP_SIZE, 0);
  int lane_idx = threadIdx.x % WARP_SIZE;
  FPTYPE* iteratorA = (FPTYPE*)&_data[0];  // dy
  for (int ii = thread_idx; ii < last_layer_size; ii += blockDim.x) {
    iteratorA[ii] = dy[block_idx * last_layer_size + ii];
  }
  __syncthreads();

  for (int ii = 0; ii < nnei_i; ii++) {
    for (int jj = warp_idx; jj < nnei_j; jj += KTILE) {
      FPTYPE xx = em_x[block_idx * nnei_i * nnei_j + ii * nnei_j + jj];
      FPTYPE tmp = xx;
      int table_idx = 0;
      locate_xx_se_t(xx, table_idx, lower, upper, -max, max, stride0, stride1);
      FPTYPE sum = (FPTYPE)0.;
      FPTYPE Csub = (FPTYPE)0.;
      for (int kk = lane_idx; kk < last_layer_size; kk += WARP_SIZE) {
        FPTYPE var[6];
        load_polynomial_params(var, table, table_idx, kk, last_layer_size);
        FPTYPE res =
            var[0] +
            (var[1] +
             (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) *
                xx;

        sum += iteratorA[kk] * res;
        Csub +=
            iteratorA[kk] * tmp *
            (var[1] + ((FPTYPE)2. * var[2] +
                       ((FPTYPE)3. * var[3] +
                        ((FPTYPE)4. * var[4] + (FPTYPE)5. * var[5] * xx) * xx) *
                           xx) *
                          xx);
      }
      GpuSyncThreads();
      warp_reduce(sum);
      warp_reduce(Csub);
      if (lane_idx == 0) {
        dy_dem[block_idx * nnei_i * nnei_j + ii * nnei_j + jj] = sum;
        dy_dem_x[block_idx * nnei_i * nnei_j + ii * nnei_j + jj] = Csub;
      }
    }
  }
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_t_grad_grad_fifth_order_polynomial(
    FPTYPE* dz_dy,
    const FPTYPE* table,
    const FPTYPE* em_x,
    const FPTYPE* em,
    const FPTYPE* dz_dy_dem_x,
    const FPTYPE* dz_dy_dem,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei_i,
    const int nnei_j,
    const int last_layer_size) {
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // last_layer_size

  FPTYPE sum = (FPTYPE)0.;
  for (int ii = 0; ii < nnei_i; ii++) {
    int mark_table_idx = -1;
    for (int jj = 0; ii < nnei_j; jj++) {
      FPTYPE xx = em_x[block_idx * nnei_i * nnei_j + ii * nnei_j + jj];
      FPTYPE tmp = xx;
      FPTYPE dz_xx =
          dz_dy_dem_x[block_idx * nnei_i * nnei_j + ii * nnei_j + jj];
      FPTYPE dz_em = dz_dy_dem[block_idx * nnei_i * nnei_j + ii * nnei_j + jj];
      FPTYPE var[6];

      int table_idx = 0;
      locate_xx_se_t(xx, table_idx, lower, upper, -max, max, stride0, stride1);
      if (table_idx != mark_table_idx) {
        load_polynomial_params(var, table, table_idx, thread_idx,
                               last_layer_size);
      }
      FPTYPE res =
          var[0] +
          (var[1] +
           (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) *
              xx;
      FPTYPE res_grad =
          var[1] + (2 * var[2] +
                    (3 * var[3] + (4 * var[4] + 5 * var[5] * xx) * xx) * xx) *
                       xx;

      sum += (tmp * res_grad * dz_xx + dz_em * res);
      mark_table_idx = table_idx;
    }
  }
  dz_dy[block_idx * last_layer_size + thread_idx] = sum;
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_r_fifth_order_polynomial(
    FPTYPE* out,
    const FPTYPE* table,
    const FPTYPE* em,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei,
    const int last_layer_size) {
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // last_layer_size

  int mark_table_idx = -1;
  FPTYPE var[6];
  for (int ii = 0; ii < nnei; ii++) {
    FPTYPE xx = em[block_idx * nnei + ii];
    int table_idx = 0;
    locate_xx_se_r(xx, table_idx, lower, upper, max, stride0, stride1);
    if (table_idx != mark_table_idx) {
      load_polynomial_params(var, table, table_idx, thread_idx,
                             last_layer_size);
    }
    out[block_idx * nnei * last_layer_size + ii * last_layer_size +
        thread_idx] =
        var[0] +
        (var[1] + (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) *
            xx;
    mark_table_idx = table_idx;
  }
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_r_grad_fifth_order_polynomial(
    FPTYPE* dy_dem,
    const FPTYPE* table,
    const FPTYPE* em,
    const FPTYPE* dy,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei,
    const int last_layer_size) {
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // KTILE * WARP_SIZE, usally 128 here~
  int warp_idx = GpuShuffleSync(0xffffffff, thread_idx / WARP_SIZE, 0);
  int lane_idx = thread_idx % WARP_SIZE;
  __syncthreads();
  for (int ii = warp_idx; ii < nnei; ii += KTILE) {
    FPTYPE xx = em[block_idx * nnei + ii];

    int table_idx = 0;
    FPTYPE Csub = (FPTYPE)0.;
    locate_xx_se_r(xx, table_idx, lower, upper, max, stride0, stride1);

    FPTYPE var[6];
    for (int jj = lane_idx; jj < last_layer_size; jj += WARP_SIZE) {
      load_polynomial_params(var, table, table_idx, jj, last_layer_size);
      Csub +=
          (var[1] + ((FPTYPE)2. * var[2] +
                     ((FPTYPE)3. * var[3] +
                      ((FPTYPE)4. * var[4] + (FPTYPE)5. * var[5] * xx) * xx) *
                         xx) *
                        xx) *
          dy[block_idx * nnei * last_layer_size + ii * last_layer_size + jj];
    }
    GpuSyncThreads();

    warp_reduce(Csub);
    if (lane_idx == 0) {
      dy_dem[block_idx * nnei + ii] = Csub;
    }
  }
}

template <typename FPTYPE, int MTILE, int KTILE>
__global__ void tabulate_fusion_se_r_grad_grad_fifth_order_polynomial(
    FPTYPE* dz_dy,
    const FPTYPE* table,
    const FPTYPE* em,
    const FPTYPE* dz_dy_dem,
    const FPTYPE lower,
    const FPTYPE upper,
    const FPTYPE max,
    const FPTYPE stride0,
    const FPTYPE stride1,
    const int nnei,
    const int last_layer_size) {
  const int_64 block_idx = blockIdx.x;  // nloc
  const int thread_idx = threadIdx.x;   // last_layer_size

#if TENSORFLOW_USE_ROCM
  __syncthreads();
#endif

  int mark_table_idx = -1;
  FPTYPE var[6];
  for (int ii = 0; ii < nnei; ii++) {
    FPTYPE xx = em[block_idx * nnei + ii];
    int table_idx = 0;
    locate_xx_se_r(xx, table_idx, lower, upper, max, stride0, stride1);
    if (table_idx != mark_table_idx) {
      load_polynomial_params(var, table, table_idx, thread_idx,
                             last_layer_size);
    }
    FPTYPE res_grad =
        var[1] + ((FPTYPE)2. * var[2] +
                  ((FPTYPE)3. * var[3] +
                   ((FPTYPE)4. * var[4] + (FPTYPE)5. * var[5] * xx) * xx) *
                      xx) *
                     xx;
    mark_table_idx = table_idx;
    dz_dy[block_idx * nnei * last_layer_size + ii * last_layer_size +
          thread_idx] = dz_dy_dem[block_idx * nnei + ii] * res_grad;
  }
}

namespace deepmd {
template <typename FPTYPE>
void tabulate_fusion_se_a_gpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em_x,
                              const FPTYPE* em,
                              const FPTYPE* two_embed,
                              const int nloc,
                              const int nnei,
                              const int last_layer_size,
                              const bool is_sorted) {
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  tabulate_fusion_se_a_fifth_order_polynomial<FPTYPE, MM, KK>
#if GOOGLE_CUDA
      <<<nloc, last_layer_size>>>
#elif TENSORFLOW_USE_ROCM
      <<<nloc, last_layer_size, sizeof(FPTYPE) * MM * last_layer_size>>>
#else
#error "should not touch here"
#endif
      (out, table, em_x, em, two_embed, table_info[0], table_info[1],
       table_info[2], table_info[3], table_info[4], nnei, last_layer_size,
       is_sorted);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void tabulate_fusion_se_a_grad_gpu(FPTYPE* dy_dem_x,
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
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  DPErrcheck(gpuMemset(dy_dem_x, 0, sizeof(FPTYPE) * nloc * nnei));
  DPErrcheck(gpuMemset(dy_dem, 0, sizeof(FPTYPE) * nloc * nnei * 4));

  tabulate_fusion_se_a_grad_fifth_order_polynomial<FPTYPE, MM, KK>
      <<<nloc, KK * WARP_SIZE, sizeof(FPTYPE) * MM * last_layer_size>>>(
          dy_dem_x, dy_dem, dy_dtwo, table, em_x, em, two_embed, dy,
          table_info[0], table_info[1], table_info[2], table_info[3],
          table_info[4], nnei, last_layer_size, is_sorted);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void tabulate_fusion_se_a_grad_grad_gpu(FPTYPE* dz_dy,
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
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  DPErrcheck(gpuMemset(dz_dy, 0, sizeof(FPTYPE) * nloc * 4 * last_layer_size));
  tabulate_fusion_se_a_grad_grad_fifth_order_polynomial<FPTYPE, MM, KK>
      <<<nloc, last_layer_size, sizeof(FPTYPE) * MM * last_layer_size>>>(
          dz_dy, table, em_x, em, two_embed, dz_dy_dem_x, dz_dy_dem, dz_dy_dtwo,
          table_info[0], table_info[1], table_info[2], table_info[3],
          table_info[4], nnei, last_layer_size, is_sorted);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void tabulate_fusion_se_t_gpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em_x,
                              const FPTYPE* em,
                              const int nloc,
                              const int nnei_i,
                              const int nnei_j,
                              const int last_layer_size) {
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  tabulate_fusion_se_t_fifth_order_polynomial<FPTYPE, MM, KK>
      <<<nloc, last_layer_size>>>(
          out, table, em_x, em, table_info[0], table_info[1], table_info[2],
          table_info[3], table_info[4], nnei_i, nnei_j, last_layer_size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void tabulate_fusion_se_t_grad_gpu(FPTYPE* dy_dem_x,
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
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  DPErrcheck(gpuMemset(dy_dem_x, 0, sizeof(FPTYPE) * nloc * nnei_i * nnei_j));
  DPErrcheck(gpuMemset(dy_dem, 0, sizeof(FPTYPE) * nloc * nnei_i * nnei_j));

  tabulate_fusion_se_t_grad_fifth_order_polynomial<FPTYPE, MM, KK>
      <<<nloc, KK * WARP_SIZE, sizeof(FPTYPE) * last_layer_size>>>(
          dy_dem_x, dy_dem, table, em_x, em, dy, table_info[0], table_info[1],
          table_info[2], table_info[3], table_info[4], nnei_i, nnei_j,
          last_layer_size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void tabulate_fusion_se_t_grad_grad_gpu(FPTYPE* dz_dy,
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
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  DPErrcheck(gpuMemset(dz_dy, 0, sizeof(FPTYPE) * nloc * last_layer_size));

  tabulate_fusion_se_t_grad_grad_fifth_order_polynomial<FPTYPE, MM, KK>
      <<<nloc, last_layer_size>>>(dz_dy, table, em_x, em, dz_dy_dem_x,
                                  dz_dy_dem, table_info[0], table_info[1],
                                  table_info[2], table_info[3], table_info[4],
                                  nnei_i, nnei_j, last_layer_size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void tabulate_fusion_se_r_gpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em,
                              const int nloc,
                              const int nnei,
                              const int last_layer_size) {
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  tabulate_fusion_se_r_fifth_order_polynomial<FPTYPE, MM, KK>
      <<<nloc, last_layer_size>>>(out, table, em, table_info[0], table_info[1],
                                  table_info[2], table_info[3], table_info[4],
                                  nnei, last_layer_size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void tabulate_fusion_se_r_grad_gpu(FPTYPE* dy_dem,
                                   const FPTYPE* table,
                                   const FPTYPE* table_info,
                                   const FPTYPE* em,
                                   const FPTYPE* dy,
                                   const int nloc,
                                   const int nnei,
                                   const int last_layer_size) {
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  DPErrcheck(gpuMemset(dy_dem, 0, sizeof(FPTYPE) * nloc * nnei));

  tabulate_fusion_se_r_grad_fifth_order_polynomial<FPTYPE, MM, KK>
      <<<nloc, KK * WARP_SIZE, sizeof(FPTYPE) * MM * last_layer_size>>>(
          dy_dem, table, em, dy, table_info[0], table_info[1], table_info[2],
          table_info[3], table_info[4], nnei, last_layer_size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void tabulate_fusion_se_r_grad_grad_gpu(FPTYPE* dz_dy,
                                        const FPTYPE* table,
                                        const FPTYPE* table_info,
                                        const FPTYPE* em,
                                        const FPTYPE* dz_dy_dem,
                                        const int nloc,
                                        const int nnei,
                                        const int last_layer_size) {
  if (nloc <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  DPErrcheck(
      gpuMemset(dz_dy, 0, sizeof(FPTYPE) * nloc * nnei * last_layer_size));
  tabulate_fusion_se_r_grad_grad_fifth_order_polynomial<FPTYPE, MM, KK>
      <<<nloc, last_layer_size, sizeof(FPTYPE) * MM * last_layer_size>>>(
          dz_dy, table, em, dz_dy_dem, table_info[0], table_info[1],
          table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template void tabulate_fusion_se_a_gpu<float>(float* out,
                                              const float* table,
                                              const float* table_info,
                                              const float* em_x,
                                              const float* em,
                                              const float* two_embed,
                                              const int nloc,
                                              const int nnei,
                                              const int last_layer_size,
                                              const bool is_sorted);
template void tabulate_fusion_se_a_gpu<double>(double* out,
                                               const double* table,
                                               const double* table_info,
                                               const double* em_x,
                                               const double* em,
                                               const double* two_embed,
                                               const int nloc,
                                               const int nnei,
                                               const int last_layer_size,
                                               const bool is_sorted);
template void tabulate_fusion_se_a_grad_gpu<float>(float* dy_dem_x,
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
template void tabulate_fusion_se_a_grad_gpu<double>(double* dy_dem_x,
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
template void tabulate_fusion_se_a_grad_grad_gpu<float>(
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
template void tabulate_fusion_se_a_grad_grad_gpu<double>(
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

template void tabulate_fusion_se_t_gpu<float>(float* out,
                                              const float* table,
                                              const float* table_info,
                                              const float* em_x,
                                              const float* em,
                                              const int nloc,
                                              const int nnei_i,
                                              const int nnei_j,
                                              const int last_layer_size);
template void tabulate_fusion_se_t_gpu<double>(double* out,
                                               const double* table,
                                               const double* table_info,
                                               const double* em_x,
                                               const double* em,
                                               const int nloc,
                                               const int nnei_i,
                                               const int nnei_j,
                                               const int last_layer_size);
template void tabulate_fusion_se_t_grad_gpu<float>(float* dy_dem_x,
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
template void tabulate_fusion_se_t_grad_gpu<double>(double* dy_dem_x,
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
template void tabulate_fusion_se_t_grad_grad_gpu<float>(
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
template void tabulate_fusion_se_t_grad_grad_gpu<double>(
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

template void tabulate_fusion_se_r_gpu<float>(float* out,
                                              const float* table,
                                              const float* table_info,
                                              const float* em,
                                              const int nloc,
                                              const int nnei,
                                              const int last_layer_size);
template void tabulate_fusion_se_r_gpu<double>(double* out,
                                               const double* table,
                                               const double* table_info,
                                               const double* em,
                                               const int nloc,
                                               const int nnei,
                                               const int last_layer_size);
template void tabulate_fusion_se_r_grad_gpu<float>(float* dy_dem,
                                                   const float* table,
                                                   const float* table_info,
                                                   const float* em,
                                                   const float* dy,
                                                   const int nloc,
                                                   const int nnei,
                                                   const int last_layer_size);
template void tabulate_fusion_se_r_grad_gpu<double>(double* dy_dem,
                                                    const double* table,
                                                    const double* table_info,
                                                    const double* em,
                                                    const double* dy,
                                                    const int nloc,
                                                    const int nnei,
                                                    const int last_layer_size);
template void tabulate_fusion_se_r_grad_grad_gpu<float>(
    float* dz_dy,
    const float* table,
    const float* table_info,
    const float* em,
    const float* dz_dy_dem,
    const int nloc,
    const int nnei,
    const int last_layer_size);
template void tabulate_fusion_se_r_grad_grad_gpu<double>(
    double* dz_dy,
    const double* table,
    const double* table_info,
    const double* em,
    const double* dz_dy_dem,
    const int nloc,
    const int nnei,
    const int last_layer_size);

}  // namespace deepmd
