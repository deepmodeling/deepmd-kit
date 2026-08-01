// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <vector>

#include "device.h"
#include "neighbor_list.h"

namespace deepmd {

template <typename FPTYPE>
void prod_env_mat_a_cpu(FPTYPE* em,
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
                        const int nframes,
                        const float rcut,
                        const float rcut_smth,
                        const std::vector<int> sec,
                        const int* f_type = NULL);

template <typename FPTYPE>
inline void prod_env_mat_a_cpu(FPTYPE* em,
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
                               const int* f_type = NULL) {
  prod_env_mat_a_cpu(em, em_deriv, rij, nlist, coord, type, inlist,
                     max_nbor_size, avg, std, nloc, nall, 1, rcut, rcut_smth,
                     sec, f_type);
}

template <typename FPTYPE>
void prod_env_mat_r_cpu(FPTYPE* em,
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
                        const int nframes,
                        const float rcut,
                        const float rcut_smth,
                        const std::vector<int> sec);

template <typename FPTYPE>
inline void prod_env_mat_r_cpu(FPTYPE* em,
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
                               const std::vector<int> sec) {
  prod_env_mat_r_cpu(em, em_deriv, rij, nlist, coord, type, inlist,
                     max_nbor_size, avg, std, nloc, nall, 1, rcut, rcut_smth,
                     sec);
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename FPTYPE>
void prod_env_mat_a_gpu(FPTYPE* em,
                        FPTYPE* em_deriv,
                        FPTYPE* rij,
                        int* nlist,
                        const FPTYPE* coord,
                        const int* type,
                        const InputNlist& gpu_inlist,
                        int* array_int,
                        unsigned long long* array_longlong,
                        const int max_nbor_size,
                        const FPTYPE* avg,
                        const FPTYPE* std,
                        const int nloc,
                        const int nall,
                        const int nframes,
                        const float rcut,
                        const float rcut_smth,
                        const std::vector<int> sec,
                        const int* f_type = NULL);

template <typename FPTYPE>
inline void prod_env_mat_a_gpu(FPTYPE* em,
                               FPTYPE* em_deriv,
                               FPTYPE* rij,
                               int* nlist,
                               const FPTYPE* coord,
                               const int* type,
                               const InputNlist& gpu_inlist,
                               int* array_int,
                               unsigned long long* array_longlong,
                               const int max_nbor_size,
                               const FPTYPE* avg,
                               const FPTYPE* std,
                               const int nloc,
                               const int nall,
                               const float rcut,
                               const float rcut_smth,
                               const std::vector<int> sec,
                               const int* f_type = NULL) {
  prod_env_mat_a_gpu(em, em_deriv, rij, nlist, coord, type, gpu_inlist,
                     array_int, array_longlong, max_nbor_size, avg, std, nloc,
                     nall, 1, rcut, rcut_smth, sec, f_type);
}

template <typename FPTYPE>
void prod_env_mat_r_gpu(FPTYPE* em,
                        FPTYPE* em_deriv,
                        FPTYPE* rij,
                        int* nlist,
                        const FPTYPE* coord,
                        const int* type,
                        const InputNlist& gpu_inlist,
                        int* array_int,
                        unsigned long long* array_longlong,
                        const int max_nbor_size,
                        const FPTYPE* avg,
                        const FPTYPE* std,
                        const int nloc,
                        const int nall,
                        const int nframes,
                        const float rcut,
                        const float rcut_smth,
                        const std::vector<int> sec);

template <typename FPTYPE>
inline void prod_env_mat_r_gpu(FPTYPE* em,
                               FPTYPE* em_deriv,
                               FPTYPE* rij,
                               int* nlist,
                               const FPTYPE* coord,
                               const int* type,
                               const InputNlist& gpu_inlist,
                               int* array_int,
                               unsigned long long* array_longlong,
                               const int max_nbor_size,
                               const FPTYPE* avg,
                               const FPTYPE* std,
                               const int nloc,
                               const int nall,
                               const float rcut,
                               const float rcut_smth,
                               const std::vector<int> sec) {
  prod_env_mat_r_gpu(em, em_deriv, rij, nlist, coord, type, gpu_inlist,
                     array_int, array_longlong, max_nbor_size, avg, std, nloc,
                     nall, 1, rcut, rcut_smth, sec);
}

void env_mat_nbor_update(InputNlist& inlist,
                         InputNlist& gpu_inlist,
                         int& max_nbor_size,
                         int*& nbor_list_dev,
                         const int* mesh,
                         const int size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace deepmd
