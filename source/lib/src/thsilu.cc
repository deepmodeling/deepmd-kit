// SPDX-License-Identifier: LGPL-3.0-or-later
#include "thsilu.h"

#include <cmath>

// custom tanh
template <typename FPTYPE>
inline FPTYPE ctanh(const FPTYPE x,
                    const FPTYPE w,
                    const FPTYPE a,
                    const FPTYPE b) {
  return std::tanh(w * (x - a)) + b;
}

template <typename FPTYPE>
inline FPTYPE ctanhgrad(const FPTYPE x, const FPTYPE w, const FPTYPE a) {
  const FPTYPE coshwxa = std::cosh(w * (x - a));
  return w / (coshwxa * coshwxa);
}

template <typename FPTYPE>
inline FPTYPE ctanhgradgrad(const FPTYPE x, const FPTYPE w, const FPTYPE a) {
  const FPTYPE wxa = w * (x - a);
  const FPTYPE coshwxa = std::cosh(wxa);
  const FPTYPE tanhwxa = std::tanh(wxa);
  return (FPTYPE)-2.0 * w * w * tanhwxa / (coshwxa * coshwxa);
}

// silu
template <typename FPTYPE>
inline FPTYPE silu(const FPTYPE x) {
  return x / (std::exp(-x) + (FPTYPE)1.0);
}

template <typename FPTYPE>
inline FPTYPE silugrad(const FPTYPE x) {
  const FPTYPE sig = (FPTYPE)1.0 / ((FPTYPE)1.0 + std::exp(-x));
  return sig * ((FPTYPE)1.0 + x * ((FPTYPE)1.0 - sig));
}

template <typename FPTYPE>
inline FPTYPE silugradgrad(const FPTYPE x) {
  const FPTYPE sig = (FPTYPE)1.0 / ((FPTYPE)1.0 + std::exp(-x));
  const FPTYPE sig_prime = sig * (1 - sig);
  return sig_prime * (2 + x * (1 - 2 * sig));
}

// thsilu

template <typename FPTYPE>
inline FPTYPE tanhsilu(const FPTYPE x,
                       const FPTYPE w,
                       const FPTYPE a,
                       const FPTYPE b) {
  return x < a ? silu(x) : ctanh(x, w, a, b);
}

template <typename FPTYPE>
inline FPTYPE tanhsilugrad(const FPTYPE x, const FPTYPE w, const FPTYPE a) {
  return x < a ? silugrad(x) : ctanhgrad(x, w, a);
}

template <typename FPTYPE>
inline FPTYPE tanhsilugradgrad(const FPTYPE x, const FPTYPE w, const FPTYPE a) {
  return x < a ? silugradgrad(x) : ctanhgradgrad(x, w, a);
}

template <typename FPTYPE>
void deepmd::thsilu_cpu(FPTYPE* out,
                        const FPTYPE* xx,
                        const int_64 size,
                        const FPTYPE w,
                        const FPTYPE a,
                        const FPTYPE b) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    out[ii] = tanhsilu(xx[ii], w, a, b);
  }
}

template <typename FPTYPE>
void deepmd::thsilu_grad_cpu(FPTYPE* out,
                             const FPTYPE* xx,
                             const FPTYPE* dy,
                             const int_64 size,
                             const FPTYPE w,
                             const FPTYPE a) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    out[ii] = dy[ii] * tanhsilugrad(xx[ii], w, a);
  }
}

template <typename FPTYPE>
void deepmd::thsilu_grad_grad_cpu(FPTYPE* out,
                                  const FPTYPE* xx,
                                  const FPTYPE* dy,
                                  const FPTYPE* dy_2,
                                  const int_64 size,
                                  const FPTYPE w,
                                  const FPTYPE a) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    out[ii] = dy_2[ii] * tanhsilugradgrad(xx[ii], w, a);
  }
}

template void deepmd::thsilu_cpu<float>(float* out,
                                        const float* x,
                                        const int_64 size,
                                        const float w,
                                        const float a,
                                        const float b);
template void deepmd::thsilu_cpu<double>(double* out,
                                         const double* x,
                                         const int_64 size,
                                         const double w,
                                         const double a,
                                         const double b);
template void deepmd::thsilu_grad_cpu<float>(float* out,
                                             const float* x,
                                             const float* dy,
                                             const int_64 size,
                                             const float w,
                                             const float a);
template void deepmd::thsilu_grad_cpu<double>(double* out,
                                              const double* x,
                                              const double* dy,
                                              const int_64 size,
                                              const double w,
                                              const double a);
template void deepmd::thsilu_grad_grad_cpu<float>(float* out,
                                                  const float* x,
                                                  const float* dy,
                                                  const float* dy_2,
                                                  const int_64 size,
                                                  const float w,
                                                  const float a);
template void deepmd::thsilu_grad_grad_cpu<double>(double* out,
                                                   const double* x,
                                                   const double* dy,
                                                   const double* dy_2,
                                                   const int_64 size,
                                                   const double w,
                                                   const double a);
