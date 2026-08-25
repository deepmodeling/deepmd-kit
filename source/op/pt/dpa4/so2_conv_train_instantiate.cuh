// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Explicit launcher instantiations of the fused SO(2) value-path training
// forward for one spherical-harmonic degree. Included with DPA4_SCT_L
// defined; DPA4_SCT_EXTERN prefixes the declarations in the host unit so no
// instantiation (and no device code) lands there.

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "so2_conv_train_kernels.cuh"

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

DPA4_SCT_ONE(float)
DPA4_SCT_ONE(double)
DPA4_SCT_ONE(c10::Half)
DPA4_SCT_ONE(c10::BFloat16)

#undef DPA4_SCT_ONE

}  // namespace dpa4_sezm_kernels
