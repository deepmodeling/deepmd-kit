#pragma once
#include "device.h"

namespace deepmd {

template <typename FPTYPE>
void gelu_cpu(FPTYPE* out, const FPTYPE* xx, const int_64 size);

template <typename FPTYPE>
void gelu_grad_cpu(FPTYPE* out,
                   const FPTYPE* xx,
                   const FPTYPE* dy,
                   const int_64 size);

template <typename FPTYPE>
void gelu_grad_grad_cpu(FPTYPE* out,
                        const FPTYPE* xx,
                        const FPTYPE* dy,
                        const FPTYPE* dy_2,
                        const int_64 size);

#if GOOGLE_CUDA
template <typename FPTYPE>
void gelu_gpu_cuda(FPTYPE* out, const FPTYPE* xx, const int_64 size);

template <typename FPTYPE>
void gelu_grad_gpu_cuda(FPTYPE* out,
                        const FPTYPE* xx,
                        const FPTYPE* dy,
                        const int_64 size);

template <typename FPTYPE>
void gelu_grad_grad_gpu_cuda(FPTYPE* out,
                             const FPTYPE* xx,
                             const FPTYPE* dy,
                             const FPTYPE* dy_2,
                             const int_64 size);
#endif  // GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
template <typename FPTYPE>
void gelu_gpu_rocm(FPTYPE* out, const FPTYPE* xx, const int_64 size);

template <typename FPTYPE>
void gelu_grad_gpu_rocm(FPTYPE* out,
                        const FPTYPE* xx,
                        const FPTYPE* dy,
                        const int_64 size);

template <typename FPTYPE>
void gelu_grad_grad_gpu_rocm(FPTYPE* out,
                             const FPTYPE* xx,
                             const FPTYPE* dy,
                             const FPTYPE* dy_2,
                             const int_64 size);

#endif  // TENSORFLOW_USE_ROCM
}  // namespace deepmd
