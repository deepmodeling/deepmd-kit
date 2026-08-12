// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Compressed DPA1 CUDA specializations for 16 embedding channels.

#include "dpa1_graph_compress_kernel.cuh"

namespace deepmd_dpa1_compress {

DPA1_COMPRESS_DEFINE_CHANNEL(16)

}  // namespace deepmd_dpa1_compress
