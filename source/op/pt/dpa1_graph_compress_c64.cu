// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Compressed DPA1 CUDA specializations for 64 embedding channels.

#include "dpa1_graph_compress_kernel.cuh"

namespace deepmd_dpa1_compress {

DPA1_COMPRESS_DEFINE_CHANNEL(64)

}  // namespace deepmd_dpa1_compress
