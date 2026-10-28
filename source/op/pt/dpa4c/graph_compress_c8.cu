// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Compiled specializations of the compressed DPA4C descriptor for a scalar
// width of 8 channels.

#include "graph_compress_kernel.cuh"

namespace deepmd_dpa4c {

DPA4C_DEFINE_CHANNEL(8)

}  // namespace deepmd_dpa4c
