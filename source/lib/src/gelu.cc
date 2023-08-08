// SPDX-License-Identifier: LGPL-3.0-or-later
#include "gelu.h"

#include <cmath>

#include "device.h"

template <typename FPTYPE>
void deepmd::gelu_cpu(FPTYPE* out, const FPTYPE* xx, const int_64 size) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    out[ii] = xx[ii] * (FPTYPE)0.5 *
              ((FPTYPE)1.0 +
               tanh((FPTYPE)SQRT_2_PI *
                    (xx[ii] + (FPTYPE)0.044715 * xx[ii] * xx[ii] * xx[ii])));
  }
}

template <typename FPTYPE>
void deepmd::gelu_grad_cpu(FPTYPE* out,
                           const FPTYPE* xx,
                           const FPTYPE* dy,
                           const int_64 size) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    const FPTYPE var =
        tanh((FPTYPE)SQRT_2_PI *
             (xx[ii] + (FPTYPE)0.044715 * xx[ii] * xx[ii] * xx[ii]));
    out[ii] = dy[ii] * ((FPTYPE)0.5 * (FPTYPE)SQRT_2_PI * xx[ii] *
                            ((FPTYPE)1. - var * var) *
                            ((FPTYPE)0.134145 * xx[ii] * xx[ii] + (FPTYPE)1.) +
                        (FPTYPE)0.5 * var + (FPTYPE)0.5);
  }
}

template <typename FPTYPE>
void deepmd::gelu_grad_grad_cpu(FPTYPE* out,
                                const FPTYPE* xx,
                                const FPTYPE* dy,
                                const FPTYPE* dy_2,
                                const int_64 size) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    const FPTYPE var1 =
        tanh((FPTYPE)SQRT_2_PI *
             (xx[ii] + (FPTYPE)0.044715 * xx[ii] * xx[ii] * xx[ii]));
    const FPTYPE var2 = (FPTYPE)SQRT_2_PI * ((FPTYPE)1. - var1 * var1) *
                        ((FPTYPE)0.134145 * xx[ii] * xx[ii] + (FPTYPE)1.);
    out[ii] = dy[ii] * dy_2[ii] *
              ((FPTYPE)0.134145 * (FPTYPE)SQRT_2_PI * xx[ii] * xx[ii] *
                   ((FPTYPE)1. - var1 * var1) -
               (FPTYPE)SQRT_2_PI * xx[ii] * var2 *
                   ((FPTYPE)0.134145 * xx[ii] * xx[ii] + (FPTYPE)1.) * var1 +
               var2);
  }
}

template void deepmd::gelu_cpu<float>(float* out,
                                      const float* x,
                                      const int_64 size);
template void deepmd::gelu_cpu<double>(double* out,
                                       const double* x,
                                       const int_64 size);
template void deepmd::gelu_grad_cpu<float>(float* out,
                                           const float* x,
                                           const float* dy,
                                           const int_64 size);
template void deepmd::gelu_grad_cpu<double>(double* out,
                                            const double* x,
                                            const double* dy,
                                            const int_64 size);
template void deepmd::gelu_grad_grad_cpu<float>(float* out,
                                                const float* x,
                                                const float* dy,
                                                const float* dy_2,
                                                const int_64 size);
template void deepmd::gelu_grad_grad_cpu<double>(double* out,
                                                 const double* x,
                                                 const double* dy,
                                                 const double* dy_2,
                                                 const int_64 size);
