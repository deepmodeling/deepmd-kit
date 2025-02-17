#include "cdsilu.h"

__device__ inline double _exp(double x) { return exp(x); }
__device__ inline float _exp(float x) { return expf(x); }

// silu
template <typename FPTYPE>
__device__ inline FPTYPE silu(const FPTYPE x) {
  return x / (_exp(-x) + (FPTYPE)1.0);
}

template <typename FPTYPE>
__device__ inline FPTYPE silugrad(const FPTYPE x) {
  const FPTYPE sig = (FPTYPE)1.0 / ((FPTYPE)1.0 + _exp(-x));
  return sig * ((FPTYPE)1.0 + x * ((FPTYPE)1.0 - sig));
}

template <typename FPTYPE>
__device__ inline FPTYPE silugradgrad(const FPTYPE x) {
  const FPTYPE sig = (FPTYPE)1.0 / ((FPTYPE)1.0 + _exp(-x));
  const FPTYPE sig_prime = sig * (1 - sig);
  return sig_prime * (2 + x * (1 - 2 * sig));
}

// cdsilu

template <typename FPTYPE>
__device__ inline FPTYPE customdsilu(const FPTYPE x,
                                     const FPTYPE a,
                                     const FPTYPE b) {
  return ((FPTYPE)1.0 -
          (-a + x + (FPTYPE)1.0) / (_exp(a - x - (FPTYPE)1.0) + (FPTYPE)1.0)) /
             (_exp(a + b - x - (FPTYPE)1.0) + (FPTYPE)1.0) +
         silu(x);
}

template <typename FPTYPE>
__device__ inline FPTYPE customdsilugrad(const FPTYPE x,
                                         const FPTYPE a,
                                         const FPTYPE b) {
  FPTYPE xbar = -a + x + (FPTYPE)1.0;
  FPTYPE eax1 = _exp(-xbar);
  FPTYPE eax1p1 = eax1 + (FPTYPE)1.0;
  FPTYPE eax1p1r = (FPTYPE)1.0 / eax1p1;
  FPTYPE eax1eax1p1r = 1 - eax1p1r;
  FPTYPE eaxb1 = _exp(-xbar + b);
  FPTYPE eaxb1p1 = eaxb1 + (FPTYPE)1.0;
  FPTYPE eaxb1p1r = (FPTYPE)1.0 / eaxb1p1;
  FPTYPE eaxb1eaxb1p1r = 1 - eaxb1p1r;
  return (-xbar * eax1eax1p1r * eax1p1r - eax1p1r) * eaxb1p1r +
         ((FPTYPE)1.0 - xbar * eax1p1r) * eaxb1eaxb1p1r * eaxb1p1r +
         silugrad(x);
}

template <typename FPTYPE>
__device__ inline FPTYPE customdsilugradgrad(const FPTYPE x,
                                             const FPTYPE a,
                                             const FPTYPE b) {
  FPTYPE xbar = -a + x + (FPTYPE)1.0;
  FPTYPE eax1 = _exp(-xbar);
  FPTYPE eax1p1 = eax1 + (FPTYPE)1.0;
  FPTYPE eax1p1r = (FPTYPE)1.0 / eax1p1;
  FPTYPE eax1eax1p1r = 1 - eax1p1r;
  FPTYPE eaxb1 = _exp(-xbar + b);
  FPTYPE eaxb1p1 = eaxb1 + (FPTYPE)1.0;
  FPTYPE eaxb1p1r = (FPTYPE)1.0 / eaxb1p1;
  FPTYPE eaxb1eaxb1p1r = 1 - eaxb1p1r;
  return ((FPTYPE)2.0 * (-xbar * eax1eax1p1r * eax1p1r - eax1p1r) -
          ((FPTYPE)1.0 - xbar * eax1p1r)) *
             eaxb1eaxb1p1r * eaxb1p1r +
         (xbar - (FPTYPE)2.0 * xbar * eax1eax1p1r - (FPTYPE)2.0) * eax1eax1p1r *
             eax1p1r * eaxb1p1r +
         (FPTYPE)2.0 * ((FPTYPE)1.0 - xbar * eax1p1r) * eaxb1eaxb1p1r *
             eaxb1eaxb1p1r * eaxb1p1r +
         silugradgrad(x);
}

template <typename FPTYPE>
__global__ void cdsilu(FPTYPE* out,
                       const FPTYPE* xx,
                       const int_64 size,
                       const FPTYPE a,
                       const FPTYPE b) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  out[idx] = customdsilu(xx[idx], a, b);
}

template <typename FPTYPE>
__global__ void cdsilu_grad(FPTYPE* out,
                            const FPTYPE* xx,
                            const FPTYPE* dy,
                            const int_64 size,
                            const FPTYPE a,
                            const FPTYPE b) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  out[idx] = dy[idx] * customdsilugrad(xx[idx], a, b);
}

template <typename FPTYPE>
__global__ void cdsilu_grad_grad(FPTYPE* out,
                                 const FPTYPE* xx,
                                 const FPTYPE* dy,
                                 const FPTYPE* dy_2,
                                 const int_64 size,
                                 const FPTYPE a,
                                 const FPTYPE b) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  out[idx] = dy[idx] * dy_2[idx] * customdsilugradgrad(xx[idx], a, b);
}

namespace deepmd {
template <typename FPTYPE>
void cdsilu_gpu(FPTYPE* out,
                const FPTYPE* xx,
                const int_64 size,
                const FPTYPE a,
                const FPTYPE b) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  cdsilu<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, size, a, b);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void cdsilu_grad_gpu(FPTYPE* out,
                     const FPTYPE* xx,
                     const FPTYPE* dy,
                     const int_64 size,
                     const FPTYPE a,
                     const FPTYPE b) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  cdsilu_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, dy, size, a, b);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void cdsilu_grad_grad_gpu(FPTYPE* out,
                          const FPTYPE* xx,
                          const FPTYPE* dy,
                          const FPTYPE* dy_2,
                          const int_64 size,
                          const FPTYPE a,
                          const FPTYPE b) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  cdsilu_grad_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, dy, dy_2, size, a, b);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template void cdsilu_gpu<float>(float* out,
                                const float* x,
                                const int_64 size,
                                const float a,
                                const float b);
template void cdsilu_gpu<double>(double* out,
                                 const double* x,
                                 const int_64 size,
                                 const double a,
                                 const double b);
template void cdsilu_grad_gpu<float>(float* out,
                                     const float* x,
                                     const float* dy,
                                     const int_64 size,
                                     const float a,
                                     const float b);
template void cdsilu_grad_gpu<double>(double* out,
                                      const double* x,
                                      const double* dy,
                                      const int_64 size,
                                      const double a,
                                      const double b);
template void cdsilu_grad_grad_gpu<float>(float* out,
                                          const float* x,
                                          const float* dy,
                                          const float* dy_2,
                                          const int_64 size,
                                          const float a,
                                          const float b);
template void cdsilu_grad_grad_gpu<double>(double* out,
                                           const double* x,
                                           const double* dy,
                                           const double* dy_2,
                                           const int_64 size,
                                           const double a,
                                           const double b);
}  // namespace deepmd
