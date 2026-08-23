// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Explicit launcher instantiations of the rotation / degree-mixing training
// operators for one spherical-harmonic degree. Included with DPA4_RMT_L
// defined; DPA4_RMT_EXTERN prefixes the declarations in the host unit so no
// instantiation (and no device code) lands there.

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "rotate_mix_train_kernels.cuh"

#ifndef DPA4_RMT_L
#error "DPA4_RMT_L must name the degree of this unit"
#endif
#ifndef DPA4_RMT_EXTERN
#define DPA4_RMT_EXTERN
#endif

namespace dpa4_sezm_kernels {

#define DPA4_RMT_ONE(T)                                                        \
  DPA4_RMT_EXTERN template void launch_rotate_mix_fwd<T, DPA4_RMT_L>(          \
      const T*, const long*, const T*, const T*, const T*, T*, long, long,     \
      long, int, int, int, int, cudaStream_t);                                 \
  DPA4_RMT_EXTERN template void launch_rotate_mix_fwd_pair<T, DPA4_RMT_L>(     \
      const T*, const T*, const long*, const T*, const T*, const T*, const T*, \
      const T*, T*, T*, long, long, long, long, long, int, int, int, int,      \
      cudaStream_t);                                                           \
  DPA4_RMT_EXTERN template void launch_rotate_mix_bwd<T, DPA4_RMT_L>(          \
      const T*, const T*, const long*, const T*, const T*, const T*, T*, T*,   \
      T*, T*, long, long, long, int, int, int, int, cudaStream_t);             \
  DPA4_RMT_EXTERN template void launch_rotate_mix_bwd2<T, DPA4_RMT_L>(         \
      const T*, const T*, const T*, const long*, const T*, const T*, const T*, \
      const T*, const T*, T*, T*, T*, T*, long, long, long, long, long, int,   \
      int, int, int, cudaStream_t);

DPA4_RMT_ONE(float)
DPA4_RMT_ONE(double)
DPA4_RMT_ONE(c10::Half)
DPA4_RMT_ONE(c10::BFloat16)

#undef DPA4_RMT_ONE

}  // namespace dpa4_sezm_kernels
