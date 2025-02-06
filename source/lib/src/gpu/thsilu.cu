#include "thsilu.h"

__device__ inline double _tanh(double x) { return tanh(x); }
__device__ inline float _tanh(float x) { return tanhf(x); }

__device__ inline double _cosh(double x) { return cosh(x); }
__device__ inline float _cosh(float x) { return coshf(x); }

__device__ inline double _exp(double x) { return exp(x); }
__device__ inline float _exp(float x) { return expf(x); }

// custom tanh
template <typename FPTYPE>
__device__ inline FPTYPE ctanh(const FPTYPE x,
                               const FPTYPE w,
                               const FPTYPE a,
                               const FPTYPE b) {
  return _tanh(w * (x - a)) + b;
}

template <typename FPTYPE>
__device__ inline FPTYPE ctanhgrad(const FPTYPE x,
                                   const FPTYPE w,
                                   const FPTYPE a) {
  const FPTYPE coshwxa = _cosh(w * (x - a));
  return w / (coshwxa * coshwxa);
}

template <typename FPTYPE>
__device__ inline FPTYPE ctanhgradgrad(const FPTYPE x,
                                       const FPTYPE w,
                                       const FPTYPE a) {
  const FPTYPE wxa = w * (x - a);
  const FPTYPE coshwxa = _cosh(wxa);
  const FPTYPE tanhwxa = _tanh(wxa);
  return (FPTYPE)-2.0 * w * w * tanhwxa / (coshwxa * coshwxa);
}

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

// thsilu

template <typename FPTYPE>
__device__ inline FPTYPE tanhsilu(const FPTYPE x,
                                  const FPTYPE w,
                                  const FPTYPE a,
                                  const FPTYPE b) {
  return x < a ? silu(x) : ctanh(x, w, a, b);
}

template <typename FPTYPE>
__device__ inline FPTYPE tanhsilugrad(const FPTYPE x,
                                      const FPTYPE w,
                                      const FPTYPE a) {
  return x < a ? silugrad(x) : ctanhgrad(x, w, a);
}

template <typename FPTYPE>
__device__ inline FPTYPE tanhsilugradgrad(const FPTYPE x,
                                          const FPTYPE w,
                                          const FPTYPE a) {
  return x < a ? silugradgrad(x) : ctanhgradgrad(x, w, a);
}

template <typename FPTYPE>
__global__ void thsilu(FPTYPE* out,
                       const FPTYPE* xx,
                       const int_64 size,
                       const FPTYPE w,
                       const FPTYPE a,
                       const FPTYPE b) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  out[idx] = tanhsilu(xx[idx], w, a, b);
}

template <typename FPTYPE>
__global__ void thsilu_grad(FPTYPE* out,
                            const FPTYPE* xx,
                            const FPTYPE* dy,
                            const int_64 size,
                            const FPTYPE w,
                            const FPTYPE a) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  out[idx] = dy[idx] * tanhsilugrad(xx[idx], w, a);
}

template <typename FPTYPE>
__global__ void thsilu_grad_grad(FPTYPE* out,
                                 const FPTYPE* xx,
                                 const FPTYPE* dy,
                                 const FPTYPE* dy_2,
                                 const int_64 size,
                                 const FPTYPE w,
                                 const FPTYPE a) {
  const int_64 idx = int_64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  out[idx] = dy_2[idx] * tanhsilugradgrad(xx[idx], w, a);
}

namespace deepmd {
template <typename FPTYPE>
void thsilu_gpu(FPTYPE* out,
                const FPTYPE* xx,
                const int_64 size,
                const FPTYPE w,
                const FPTYPE a,
                const FPTYPE b) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  thsilu<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, size, w, a, b);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void thsilu_grad_gpu(FPTYPE* out,
                     const FPTYPE* xx,
                     const FPTYPE* dy,
                     const int_64 size,
                     const FPTYPE w,
                     const FPTYPE a) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  thsilu_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, dy, size, w, a);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void thsilu_grad_grad_gpu(FPTYPE* out,
                          const FPTYPE* xx,
                          const FPTYPE* dy,
                          const FPTYPE* dy_2,
                          const int_64 size,
                          const FPTYPE w,
                          const FPTYPE a) {
  if (size <= 0) {
    return;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  thsilu_grad_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, dy, dy_2, size, w, a);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template void thsilu_gpu<float>(float* out,
                                const float* x,
                                const int_64 size,
                                const float w,
                                const float a,
                                const float b);
template void thsilu_gpu<double>(double* out,
                                 const double* x,
                                 const int_64 size,
                                 const double w,
                                 const double a,
                                 const double b);
template void thsilu_grad_gpu<float>(float* out,
                                     const float* x,
                                     const float* dy,
                                     const int_64 size,
                                     const float w,
                                     const float a);
template void thsilu_grad_gpu<double>(double* out,
                                      const double* x,
                                      const double* dy,
                                      const int_64 size,
                                      const double w,
                                      const double a);
template void thsilu_grad_grad_gpu<float>(float* out,
                                          const float* x,
                                          const float* dy,
                                          const float* dy_2,
                                          const int_64 size,
                                          const float w,
                                          const float a);
template void thsilu_grad_grad_gpu<double>(double* out,
                                           const double* x,
                                           const double* dy,
                                           const double* dy_2,
                                           const int_64 size,
                                           const double w,
                                           const double a);
}  // namespace deepmd
