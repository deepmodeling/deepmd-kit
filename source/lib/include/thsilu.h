// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "device.h"

namespace deepmd {

template <typename FPTYPE>
void thsilu_cpu(FPTYPE* out,
                const FPTYPE* xx,
                const int_64 size,
                const FPTYPE w,
                const FPTYPE a,
                const FPTYPE b);

template <typename FPTYPE>
void thsilu_grad_cpu(FPTYPE* out,
                     const FPTYPE* xx,
                     const FPTYPE* dy,
                     const int_64 size,
                     const FPTYPE w,
                     const FPTYPE a);

template <typename FPTYPE>
void thsilu_grad_grad_cpu(FPTYPE* out,
                          const FPTYPE* xx,
                          const FPTYPE* dy,
                          const FPTYPE* dy_2,
                          const int_64 size,
                          const FPTYPE w,
                          const FPTYPE a);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename FPTYPE>
void thsilu_gpu(FPTYPE* out,
                const FPTYPE* xx,
                const int_64 size,
                const FPTYPE w,
                const FPTYPE a,
                const FPTYPE b);

template <typename FPTYPE>
void thsilu_grad_gpu(FPTYPE* out,
                     const FPTYPE* xx,
                     const FPTYPE* dy,
                     const int_64 size,
                     const FPTYPE w,
                     const FPTYPE a);

template <typename FPTYPE>
void thsilu_grad_grad_gpu(FPTYPE* out,
                          const FPTYPE* xx,
                          const FPTYPE* dy,
                          const FPTYPE* dy_2,
                          const int_64 size,
                          const FPTYPE w,
                          const FPTYPE a);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace deepmd
