// SPDX-License-Identifier: LGPL-3.0-or-later
#include "cdsilu.h"

#include <cmath>

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

// cdsilu

template <typename FPTYPE>
inline FPTYPE customdsilu(const FPTYPE x, const FPTYPE a, const FPTYPE b) {
  return ((FPTYPE)1.0 - (-a + x + (FPTYPE)1.0) /
                            (std::exp(a - x - (FPTYPE)1.0) + (FPTYPE)1.0)) /
             (std::exp(a + b - x - (FPTYPE)1.0) + (FPTYPE)1.0) +
         silu(x);
}

template <typename FPTYPE>
inline FPTYPE customdsilugrad(const FPTYPE x, const FPTYPE a, const FPTYPE b) {
  FPTYPE xbar = -a + x + (FPTYPE)1.0;
  FPTYPE eax1 = std::exp(-xbar);
  FPTYPE eax1p1 = eax1 + (FPTYPE)1.0;
  FPTYPE eax1p1r = (FPTYPE)1.0 / eax1p1;
  FPTYPE eax1eax1p1r = 1 - eax1p1r;
  FPTYPE eaxb1 = std::exp(-xbar + b);
  FPTYPE eaxb1p1 = eaxb1 + (FPTYPE)1.0;
  FPTYPE eaxb1p1r = (FPTYPE)1.0 / eaxb1p1;
  FPTYPE eaxb1eaxb1p1r = 1 - eaxb1p1r;
  return (-xbar * eax1eax1p1r * eax1p1r - eax1p1r) * eaxb1p1r +
         ((FPTYPE)1.0 - xbar * eax1p1r) * eaxb1eaxb1p1r * eaxb1p1r +
         silugrad(x);
}

template <typename FPTYPE>
inline FPTYPE customdsilugradgrad(const FPTYPE x,
                                  const FPTYPE a,
                                  const FPTYPE b) {
  FPTYPE xbar = -a + x + (FPTYPE)1.0;
  FPTYPE eax1 = std::exp(-xbar);
  FPTYPE eax1p1 = eax1 + (FPTYPE)1.0;
  FPTYPE eax1p1r = (FPTYPE)1.0 / eax1p1;
  FPTYPE eax1eax1p1r = 1 - eax1p1r;
  FPTYPE eaxb1 = std::exp(-xbar + b);
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
void deepmd::cdsilu_cpu(FPTYPE* out,
                        const FPTYPE* xx,
                        const int_64 size,
                        const FPTYPE a,
                        const FPTYPE b) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    out[ii] = customdsilu(xx[ii], a, b);
  }
}

template <typename FPTYPE>
void deepmd::cdsilu_grad_cpu(FPTYPE* out,
                             const FPTYPE* xx,
                             const FPTYPE* dy,
                             const int_64 size,
                             const FPTYPE a,
                             const FPTYPE b) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    out[ii] = dy[ii] * customdsilugrad(xx[ii], a, b);
  }
}

template <typename FPTYPE>
void deepmd::cdsilu_grad_grad_cpu(FPTYPE* out,
                                  const FPTYPE* xx,
                                  const FPTYPE* dy,
                                  const FPTYPE* dy_2,
                                  const int_64 size,
                                  const FPTYPE a,
                                  const FPTYPE b) {
#pragma omp parallel for
  for (int ii = 0; ii < size; ii++) {
    out[ii] = dy[ii] * dy_2[ii] * customdsilugradgrad(xx[ii], a, b);
  }
}

template void deepmd::cdsilu_cpu<float>(float* out,
                                        const float* x,
                                        const int_64 size,
                                        const float a,
                                        const float b);
template void deepmd::cdsilu_cpu<double>(double* out,
                                         const double* x,
                                         const int_64 size,
                                         const double a,
                                         const double b);
template void deepmd::cdsilu_grad_cpu<float>(float* out,
                                             const float* x,
                                             const float* dy,
                                             const int_64 size,
                                             const float a,
                                             const float b);
template void deepmd::cdsilu_grad_cpu<double>(double* out,
                                              const double* x,
                                              const double* dy,
                                              const int_64 size,
                                              const double a,
                                              const double b);
template void deepmd::cdsilu_grad_grad_cpu<float>(float* out,
                                                  const float* x,
                                                  const float* dy,
                                                  const float* dy_2,
                                                  const int_64 size,
                                                  const float a,
                                                  const float b);
template void deepmd::cdsilu_grad_grad_cpu<double>(double* out,
                                                   const double* x,
                                                   const double* dy,
                                                   const double* dy_2,
                                                   const int_64 size,
                                                   const double a,
                                                   const double b);
