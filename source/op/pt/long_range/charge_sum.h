// SPDX-License-Identifier: LGPL-3.0-or-later
//
// The two evaluations behind deepmd::latent_charge_sum (schema and formulas in
// ops.cc): the direct sum over the pairs of non-periodic frames
// (direct_sum.cu) and the smooth particle-mesh Ewald sum of periodic frames
// (mesh_sum.cu). Both take float64 positions and float32 raw charges of the
// nodes of the frames on one CUDA device, apply the constraint, and evaluate
// the owner nodes [owner_begin, owner_begin + owner_count) against every node
// of their frames. Owners other than every node lie in a single frame, whose
// nodes are then the charged nodes of every domain of ``communicator``, over
// which the per-frame sums of the owners are added.

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <tuple>

#include "domain_exchange.h"

namespace deepmd {
namespace long_range {

// The atomic energy (owners, 1) float64, the charge gradient (owners, C), the
// force (owners, 3), the atomic virial (owners, 9) and the virial of the
// owners of every frame (F, 9), float32.
using ChargeSumOutputs = std::tuple<torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor>;

// The direct sum. ``coefficient`` holds the float64 coefficients of the
// kernel on the device.
ChargeSumOutputs direct_sum(const torch::Tensor& coord,
                            const torch::Tensor& charge,
                            const torch::Tensor& node_mask,
                            const torch::Tensor& n_node_per_frame,
                            const torch::Tensor& total_charge,
                            const torch::Tensor& coefficient,
                            int64_t kernel,
                            int64_t channel,
                            double prefactor,
                            int owner_begin,
                            int owner_count,
                            const torch::Tensor* communicator);

// The mesh sum. ``box`` holds the cell of every frame, float64 on the host;
// ``coefficient`` the float64 coefficients of the kernel on the device.
ChargeSumOutputs mesh_sum(const torch::Tensor& coord,
                          const torch::Tensor& charge,
                          const torch::Tensor& node_mask,
                          const torch::Tensor& n_node_per_frame,
                          const torch::Tensor& total_charge,
                          const torch::Tensor& box,
                          const torch::Tensor& coefficient,
                          int64_t kernel,
                          int64_t channel,
                          double prefactor,
                          int64_t order,
                          double spacing,
                          int owner_begin,
                          int owner_count,
                          const torch::Tensor* communicator);

}  // namespace long_range
}  // namespace deepmd
