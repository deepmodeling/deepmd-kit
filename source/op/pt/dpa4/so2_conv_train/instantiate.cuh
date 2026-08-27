// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Explicit launcher instantiations of the fused SO(2) value-path training
// forward for one spherical-harmonic degree. A build shard defines
// DPA4_SCT_TYPE to select one dtype; the host leaves it undefined and sets
// DPA4_SCT_EXTERN to declare every dtype without emitting device code.

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "kernels.cuh"

#ifndef DPA4_SCT_L
#error "DPA4_SCT_L must name the degree of this unit"
#endif
#ifndef DPA4_SCT_EXTERN
#define DPA4_SCT_EXTERN
#endif

namespace dpa4_sezm_kernels {

#define DPA4_SCT_ONE(T)                                                        \
  DPA4_SCT_EXTERN template void launch_so2_value_fwd<T, DPA4_SCT_L>(           \
      const T*, const long*, const T*, const T*, const T*, const T*, const T*, \
      const T*, const T*, const T*, T*, T*, T*, acc_type<T>::type*, long,      \
      long, long, int, int, int, bool, bool, float, float, int, int, long,     \
      size_t, cudaStream_t);

#if defined(DPA4_SCT_TYPE)
DPA4_SCT_ONE(DPA4_SCT_TYPE)
#else
DPA4_SCT_ONE(float)
DPA4_SCT_ONE(double)
DPA4_SCT_ONE(c10::Half)
DPA4_SCT_ONE(c10::BFloat16)
#endif

#undef DPA4_SCT_ONE

}  // namespace dpa4_sezm_kernels
