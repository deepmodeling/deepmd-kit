// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Explicit launcher instantiations for one spherical-harmonic degree. A build
// shard defines DPA4_RMT_TYPE and DPA4_RMT_RANK to select one grid point; the
// host leaves both undefined and sets DPA4_RMT_EXTERN to declare every rank and
// dtype for its selected degree without emitting device code.

#include <c10/util/BFloat16.h>

#include "kernels.cuh"

#ifndef DPA4_RMT_L
#error "DPA4_RMT_L must name the spherical-harmonic degree"
#endif
#ifndef DPA4_RMT_EXTERN
#define DPA4_RMT_EXTERN
#endif
#if defined(DPA4_RMT_TYPE) != defined(DPA4_RMT_RANK)
#error "DPA4_RMT_TYPE and DPA4_RMT_RANK must be selected together"
#endif

namespace dpa4_sezm_kernels {

#define DPA4_RMT_ONE(T, R)                                                     \
  DPA4_RMT_EXTERN template void launch_rotate_mix_fwd<T, DPA4_RMT_L, R>(       \
      const T*, const long*, const T*, const T*, const T*, T*, long, long,     \
      long, int, int, int, cudaStream_t);                                      \
  DPA4_RMT_EXTERN template void launch_rotate_mix_fwd_pair<T, DPA4_RMT_L, R>(  \
      const T*, const T*, const long*, const T*, const T*, const T*, const T*, \
      const T*, T*, T*, long, long, long, long, long, int, int, int,           \
      cudaStream_t);                                                           \
  DPA4_RMT_EXTERN template void launch_rotate_mix_bwd<T, DPA4_RMT_L, R>(       \
      const T*, const T*, const long*, const T*, const T*, const T*, T*, T*,   \
      T*, T*, long, long, long, int, int, int, cudaStream_t);                  \
  DPA4_RMT_EXTERN template void launch_rotate_mix_bwd2<T, DPA4_RMT_L, R>(      \
      const T*, const T*, const T*, const long*, const T*, const T*, const T*, \
      const T*, const T*, T*, T*, T*, T*, long, long, long, long, long, int,   \
      int, int, cudaStream_t);

#if defined(DPA4_RMT_TYPE)
DPA4_RMT_ONE(DPA4_RMT_TYPE, DPA4_RMT_RANK)
#else
#define DPA4_RMT_ALL_RANKS(T) \
  DPA4_RMT_ONE(T, 0)          \
  DPA4_RMT_ONE(T, 1)          \
  DPA4_RMT_ONE(T, 2)          \
  DPA4_RMT_ONE(T, 3)          \
  DPA4_RMT_ONE(T, 4)

DPA4_RMT_ALL_RANKS(float)
DPA4_RMT_ALL_RANKS(double)
DPA4_RMT_ALL_RANKS(c10::BFloat16)

#undef DPA4_RMT_ALL_RANKS
#endif

#undef DPA4_RMT_ONE

}  // namespace dpa4_sezm_kernels
