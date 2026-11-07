// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Scalar types of the SeZM training kernels.
//
// The training operators work in float32 and in bfloat16, the latter under
// automatic mixed precision. Their float64 instantiations serve only the
// numerical validation of the fused operators and are compiled when the build
// enables DEEPMD_ENABLE_DPA4_FP64. The header is free of ATen, so the generated
// instantiation shards compile without the framework headers.

#pragma once

#ifndef DEEPMD_ENABLE_DPA4_FP64
#define DEEPMD_ENABLE_DPA4_FP64 0
#endif

// Accumulator precision of the SeZM training kernels: the reduced-precision
// working types accumulate in float, and float64 keeps its own width.
template <typename scalar_t>
struct acc_type {
  using type = float;
};
template <>
struct acc_type<double> {
  using type = double;
};
