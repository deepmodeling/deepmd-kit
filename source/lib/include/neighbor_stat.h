// SPDX-License-Identifier: LGPL-3.0-or-later
#include "neighbor_list.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace deepmd {
template <typename FPTYPE>
void neighbor_stat_gpu(const FPTYPE* coord,
                       const int* type,
                       const int nloc,
                       const deepmd::InputNlist& gpu_nlist,
                       int* max_nbor_size,
                       FPTYPE* min_nbor_dist,
                       const int ntypes,
                       const int MAX_NNEI);
}  // namespace deepmd

#endif
