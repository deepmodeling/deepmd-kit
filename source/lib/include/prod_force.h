// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

namespace deepmd {

/**
 * @brief Produce force from net_deriv and in_deriv.
 *
 * @tparam FPTYPE float or double
 * @param[out] force Atomic forces.
 * @param[in] net_deriv Net derivative.
 * @param[in] in_deriv Environmental derivative.
 * @param[in] nlist Neighbor list.
 * @param[in] nloc The number of local atoms.
 * @param[in] nall The number of all atoms, including ghost atoms.
 * @param[in] nnei The number of neighbors.
 * @param[in] nframes The number of frames.
 */
template <typename FPTYPE>
void prod_force_a_cpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes);

/**
 * @brief Produce force from net_deriv and in_deriv.
 * @details This function is used for multi-threading. Only part of atoms
 *         are computed in this thread. They will be comptued in parallel.
 *
 * @tparam FPTYPE float or double
 * @param[out] force Atomic forces.
 * @param[in] net_deriv Net derivative.
 * @param[in] in_deriv Environmental derivative.
 * @param[in] nlist Neighbor list.
 * @param[in] nloc The number of local atoms.
 * @param[in] nall The number of all atoms, including ghost atoms.
 * @param[in] nnei The number of neighbors.
 * @param[in] nframes The number of frames.
 * @param[in] thread_nloc The number of local atoms to be computed in this
 * thread.
 * @param[in] thread_start_index The start index of local atoms to be computed
 * in this thread. The index should be in [0, nloc).
 */
template <typename FPTYPE>
void prod_force_a_cpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes,
                      const int thread_nloc,
                      const int thread_start_index);

template <typename FPTYPE>
void prod_force_r_cpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename FPTYPE>
void prod_force_a_gpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes);

template <typename FPTYPE>
void prod_force_r_gpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace deepmd
