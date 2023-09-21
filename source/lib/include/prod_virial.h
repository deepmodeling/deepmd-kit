// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

namespace deepmd {

template <typename FPTYPE>
void prod_virial_a_cpu(FPTYPE* virial,
                       FPTYPE* atom_virial,
                       const FPTYPE* net_deriv,
                       const FPTYPE* env_deriv,
                       const FPTYPE* rij,
                       const int* nlist,
                       const int nloc,
                       const int nall,
                       const int nnei);

template <typename FPTYPE>
void prod_virial_r_cpu(FPTYPE* virial,
                       FPTYPE* atom_virial,
                       const FPTYPE* net_deriv,
                       const FPTYPE* env_deriv,
                       const FPTYPE* rij,
                       const int* nlist,
                       const int nloc,
                       const int nall,
                       const int nnei);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename FPTYPE>
void prod_virial_a_gpu(FPTYPE* virial,
                       FPTYPE* atom_virial,
                       const FPTYPE* net_deriv,
                       const FPTYPE* env_deriv,
                       const FPTYPE* rij,
                       const int* nlist,
                       const int nloc,
                       const int nall,
                       const int nnei);

template <typename FPTYPE>
void prod_virial_r_gpu(FPTYPE* virial,
                       FPTYPE* atom_virial,
                       const FPTYPE* net_deriv,
                       const FPTYPE* env_deriv,
                       const FPTYPE* rij,
                       const int* nlist,
                       const int nloc,
                       const int nall,
                       const int nnei);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace deepmd
