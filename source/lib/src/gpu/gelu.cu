#include "device.h"
#include "gelu.h"

__device__ inline double _tanh(double x) { return tanh(x); }
__device__ inline float _tanh(float x) { return tanhf(x); }

template <typename FPTYPE>
__global__ void gelu(FPTYPE* out, const FPTYPE* xx, const int_64 size) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  out[idx] = xx[idx] * (FPTYPE)0.5 *
             ((FPTYPE)1.0 +
              _tanh((FPTYPE)SQRT_2_PI * (xx[idx] + (FPTYPE)0.044715 * xx[idx] *
                                                       xx[idx] * xx[idx])));
}

template <typename FPTYPE>
__global__ void gelu_grad(FPTYPE* out,
                          const FPTYPE* xx,
                          const FPTYPE* dy,
                          const int_64 size) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  // out[idx] = xx[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (xx[idx] + 0.044715 *
  // xx[idx] * xx[idx] *xx[idx])));
  const FPTYPE var =
      _tanh((FPTYPE)SQRT_2_PI *
            (xx[idx] + (FPTYPE)0.044715 * xx[idx] * xx[idx] * xx[idx]));
  out[idx] =
      dy[idx] * ((FPTYPE)0.5 * SQRT_2_PI * xx[idx] * ((FPTYPE)1. - var * var) *
                     ((FPTYPE)0.134145 * xx[idx] * xx[idx] + (FPTYPE)1.) +
                 (FPTYPE)0.5 * var + (FPTYPE)0.5);
}

template <typename FPTYPE>
__global__ void gelu_grad_grad(FPTYPE* out,
                               const FPTYPE* xx,
                               const FPTYPE* dy,
                               const FPTYPE* dy_2,
                               const int_64 size) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  // out[idx] = xx[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (xx[idx] + 0.044715 *
  // xx[idx] * xx[idx] *xx[idx])));
  const FPTYPE var1 =
      _tanh((FPTYPE)SQRT_2_PI *
            (xx[idx] + (FPTYPE)0.044715 * xx[idx] * xx[idx] * xx[idx]));
  const FPTYPE var2 = (FPTYPE)SQRT_2_PI * ((FPTYPE)1. - var1 * var1) *
                      ((FPTYPE)0.134145 * xx[idx] * xx[idx] + (FPTYPE)1.);
  out[idx] = dy[idx] * dy_2[idx] *
             ((FPTYPE)0.134145 * (FPTYPE)SQRT_2_PI * xx[idx] * xx[idx] *
                  ((FPTYPE)1. - var1 * var1) -
              (FPTYPE)SQRT_2_PI * xx[idx] * var2 *
                  ((FPTYPE)0.134145 * xx[idx] * xx[idx] + (FPTYPE)1.) * var1 +
              var2);
}

namespace deepmd {
template <typename FPTYPE>
void gelu_gpu(FPTYPE* out, const FPTYPE* xx, const int_64 size) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  gelu<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void gelu_grad_gpu(FPTYPE* out,
                   const FPTYPE* xx,
                   const FPTYPE* dy,
                   const int_64 size) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  gelu_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, dy, size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void gelu_grad_grad_gpu(FPTYPE* out,
                        const FPTYPE* xx,
                        const FPTYPE* dy,
                        const FPTYPE* dy_2,
                        const int_64 size) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  gelu_grad_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, dy, dy_2, size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template void gelu_gpu<float>(float* out, const float* x, const int_64 size);
template void gelu_gpu<double>(double* out, const double* x, const int_64 size);
template void gelu_grad_gpu<float>(float* out,
                                   const float* x,
                                   const float* dy,
                                   const int_64 size);
template void gelu_grad_gpu<double>(double* out,
                                    const double* x,
                                    const double* dy,
                                    const int_64 size);
template void gelu_grad_grad_gpu<float>(float* out,
                                        const float* x,
                                        const float* dy,
                                        const float* dy_2,
                                        const int_64 size);
template void gelu_grad_grad_gpu<double>(double* out,
                                         const double* x,
                                         const double* dy,
                                         const double* dy_2,
                                         const int_64 size);
}  // namespace deepmd
