// SPDX-License-Identifier: LGPL-3.0-or-later
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename FPTYPE>
void gelu_gpu(FPTYPE* out, const FPTYPE* xx, const int_64 size);

template <typename FPTYPE>
void gelu_grad_gpu(FPTYPE* out,
                   const FPTYPE* xx,
                   const FPTYPE* dy,
                   const int_64 size);

template <typename FPTYPE>
void gelu_grad_grad_gpu(FPTYPE* out,
                        const FPTYPE* xx,
                        const FPTYPE* dy,
                        const FPTYPE* dy_2,
                        const int_64 size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace deepmd
