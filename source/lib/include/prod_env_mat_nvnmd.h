// SPDX-License-Identifier: LGPL-3.0-or-later
/*
//==================================================
 _   _  __     __  _   _   __  __   ____
| \ | | \ \   / / | \ | | |  \/  | |  _ \
|  \| |  \ \ / /  |  \| | | |\/| | | | | |
| |\  |   \ V /   | |\  | | |  | | | |_| |
|_| \_|    \_/    |_| \_| |_|  |_| |____/

//==================================================

code: nvnmd
reference: deepmd
author: mph (pinghui_mo@outlook.com)
date: 2021-12-6

*/

#pragma once
#include <vector>

#include "device.h"
#include "neighbor_list.h"

namespace deepmd {

// prod_env_mat_a_nvnmd_cpu
// have been remove for the same function

template <typename FPTYPE>
void prod_env_mat_a_nvnmd_quantize_cpu(FPTYPE* em,
                                       FPTYPE* em_deriv,
                                       FPTYPE* rij,
                                       int* nlist,
                                       const FPTYPE* coord,
                                       const int* type,
                                       const InputNlist& inlist,
                                       const int max_nbor_size,
                                       const FPTYPE* avg,
                                       const FPTYPE* std,
                                       const int nloc,
                                       const int nall,
                                       const float rcut,
                                       const float rcut_smth,
                                       const std::vector<int> sec,
                                       const int* f_type = NULL);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// UNDEFINE
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace deepmd
