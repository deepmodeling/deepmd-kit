// SPDX-License-Identifier: LGPL-3.0-or-later
//
// CUDA entry of deepmd::latent_charge_sum (schema and formulas in ops.cc).
//
// A call evaluates its frames by the direct sum when they carry no cell and
// by the mesh sum when they do (charge_sum.h). Across the domains of a
// decomposed system, the ranks gather the positions and charges of the charged
// nodes of every domain into one frame, in rank order (domain_exchange.h), and
// each rank evaluates its own charged nodes against that frame; the outputs
// return to the node axis of the rank, where the nodes it does not own carry
// no charge.

#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "charge_sum.h"
#include "domain_exchange.h"

namespace {

using deepmd::long_range::ChargeSumOutputs;

constexpr int kMaxChannels = 8;
constexpr int kMaxGaussians = 32;

enum KernelKind : int { kLes = 0, kSog = 1 };

// The evaluation of the frames, by the direct or the mesh sum, of the owner
// nodes [owner_begin, owner_begin + owner_count). ``box`` is defined for
// periodic frames only.
struct Evaluation {
  const torch::Tensor& box;
  const torch::Tensor& coefficient;
  int64_t kernel;
  int64_t channel;
  double prefactor;
  int64_t order;
  double spacing;
  bool periodic;

  ChargeSumOutputs operator()(const torch::Tensor& coord,
                              const torch::Tensor& charge,
                              const torch::Tensor& node_mask,
                              const torch::Tensor& n_node_per_frame,
                              const torch::Tensor& total_charge,
                              int owner_begin,
                              int owner_count,
                              const torch::Tensor* communicator) const {
    if (periodic) {
      return deepmd::long_range::mesh_sum(
          coord, charge, node_mask, n_node_per_frame, total_charge, box,
          coefficient, kernel, channel, prefactor, order, spacing, owner_begin,
          owner_count, communicator);
    }
    return deepmd::long_range::direct_sum(
        coord, charge, node_mask, n_node_per_frame, total_charge, coefficient,
        kernel, channel, prefactor, owner_begin, owner_count, communicator);
  }
};

// Evaluates the frame of a system decomposed into the domains of
// ``communicator``, of which this rank holds one: its charged nodes are its
// owners, the charged nodes of every domain its partners. The nodes the rank
// does not own carry no charge, and the per-frame virial is the share of its
// owners.
ChargeSumOutputs evaluate_across_domains(const Evaluation& evaluation,
                                         const torch::Tensor& coord,
                                         const torch::Tensor& charge,
                                         const torch::Tensor& node_mask,
                                         const torch::Tensor& n_node_per_frame,
                                         const torch::Tensor& total_charge,
                                         const torch::Tensor& communicator) {
  TORCH_CHECK(n_node_per_frame.size(0) == 1,
              "latent_charge_sum: an evaluation across MPI ranks takes one "
              "frame, got ",
              n_node_per_frame.size(0));
  const int64_t node_count = coord.size(0);
  // The charged nodes of the frame; the padding nodes past it carry none.
  // Their count is the one value the host has to wait for before the
  // exchange.
  const auto in_frame =
      torch::arange(node_count, n_node_per_frame.options()) < n_node_per_frame;
  const auto owned = torch::nonzero(node_mask & in_frame).squeeze(1);
  const auto rows =
      torch::cat({coord.index_select(0, owned),
                  charge.index_select(0, owned).to(torch::kFloat64)},
                 1)
          .contiguous();
  const auto [partners, owner_begin] =
      deepmd::long_range::gather_domain_rows(communicator, rows);
  const int64_t partner_count = partners.size(0);
  const auto partner_coord = partners.slice(1, 0, 3).contiguous();
  const auto partner_charge =
      partners.slice(1, 3).to(torch::kFloat32).contiguous();
  const auto partner_mask = torch::ones({partner_count}, node_mask.options());
  const auto partner_frame =
      torch::full({1}, partner_count, n_node_per_frame.options());
  const auto [owner_energy, owner_gradient, owner_force, owner_atom_virial,
              virial] =
      evaluation(partner_coord, partner_charge, partner_mask, partner_frame,
                 total_charge, static_cast<int>(owner_begin),
                 static_cast<int>(owned.size(0)), &communicator);
  return {torch::zeros({node_count, 1}, coord.options())
              .index_copy_(0, owned, owner_energy),
          torch::zeros({node_count, charge.size(1)}, charge.options())
              .index_copy_(0, owned, owner_gradient),
          torch::zeros({node_count, 3}, charge.options())
              .index_copy_(0, owned, owner_force),
          torch::zeros({node_count, 9}, charge.options())
              .index_copy_(0, owned, owner_atom_virial),
          virial};
}

// Whether the frames carry cells: true when ``box`` is given and none of its
// rows is zero, false when it is absent or all of its rows are; a call that
// mixes the two is rejected.
bool periodic_frames(const c10::optional<torch::Tensor>& box) {
  if (!box.has_value()) {
    return false;
  }
  const auto cells = box->contiguous();
  const double* value = cells.data_ptr<double>();
  const int64_t frame_count = cells.size(0);
  int64_t periodic = 0;
  for (int64_t f = 0; f < frame_count; ++f) {
    periodic += std::any_of(value + 9 * f, value + 9 * (f + 1),
                            [](double entry) { return entry != 0.0; });
  }
  TORCH_CHECK(periodic == 0 || periodic == frame_count,
              "latent_charge_sum: every frame of a call is periodic, or none "
              "is");
  return periodic > 0;
}

}  // namespace

namespace deepmd {
namespace long_range {

ChargeSumOutputs latent_charge_sum(torch::Tensor coord,
                                   torch::Tensor charge,
                                   torch::Tensor node_mask,
                                   torch::Tensor n_node_per_frame,
                                   torch::Tensor total_charge,
                                   c10::optional<torch::Tensor> box,
                                   torch::Tensor coefficient,
                                   torch::Tensor communicator,
                                   int64_t kernel,
                                   int64_t channel,
                                   double prefactor,
                                   int64_t order,
                                   double spacing) {
  TORCH_CHECK(coord.is_cuda() && charge.is_cuda() && node_mask.is_cuda() &&
                  n_node_per_frame.is_cuda() && total_charge.is_cuda() &&
                  coefficient.is_cuda(),
              "latent_charge_sum: inputs must be CUDA tensors, except box and "
              "communicator");
  TORCH_CHECK(charge.device() == coord.device() &&
                  node_mask.device() == coord.device() &&
                  n_node_per_frame.device() == coord.device() &&
                  total_charge.device() == coord.device() &&
                  coefficient.device() == coord.device(),
              "latent_charge_sum: inputs must share one device");
  TORCH_CHECK(coord.dim() == 2 && coord.size(1) == 3 &&
                  coord.scalar_type() == torch::kFloat64,
              "latent_charge_sum: coord must be float64 with shape (N, 3)");
  TORCH_CHECK(charge.dim() == 2 && charge.size(0) == coord.size(0) &&
                  charge.scalar_type() == torch::kFloat32,
              "latent_charge_sum: charge must be float32 with shape (N, C)");
  TORCH_CHECK(node_mask.dim() == 1 && node_mask.size(0) == coord.size(0) &&
                  node_mask.scalar_type() == torch::kBool,
              "latent_charge_sum: node_mask must be bool with shape (N,)");
  TORCH_CHECK(n_node_per_frame.dim() == 1,
              "latent_charge_sum: n_node_per_frame must be one-dimensional");
  TORCH_CHECK(total_charge.dim() == 1 &&
                  total_charge.size(0) == n_node_per_frame.size(0) &&
                  total_charge.scalar_type() == torch::kFloat64,
              "latent_charge_sum: total_charge must be float64 with one entry "
              "per frame");
  TORCH_CHECK(
      !box.has_value() ||
          (box->device().is_cpu() && box->scalar_type() == torch::kFloat64 &&
           box->dim() == 2 && box->size(0) == n_node_per_frame.size(0) &&
           box->size(1) == 9),
      "latent_charge_sum: box must be a float64 CPU tensor with shape "
      "(F, 9)");
  TORCH_CHECK(
      coefficient.dim() == 1 && coefficient.scalar_type() == torch::kFloat64,
      "latent_charge_sum: coefficient must be a float64 vector");
  const int64_t channels = charge.size(1);
  TORCH_CHECK(channels >= 1 && channels <= kMaxChannels,
              "latent_charge_sum: the channel count must lie in [1, ",
              kMaxChannels, "], got ", channels);
  TORCH_CHECK(channel >= -1 && channel < channels,
              "latent_charge_sum: the constrained channel must be -1 or lie "
              "in [0, ",
              channels, "), got ", channel);
  if (kernel == kLes) {
    TORCH_CHECK(coefficient.numel() == 1,
                "latent_charge_sum: the erf kernel takes one coefficient");
  } else if (kernel == kSog) {
    TORCH_CHECK(coefficient.numel() % 2 == 0 && coefficient.numel() > 0 &&
                    coefficient.numel() <= 2 * kMaxGaussians,
                "latent_charge_sum: the Gaussian kernel takes two "
                "coefficients per Gaussian, for at most ",
                kMaxGaussians, " Gaussians");
  } else {
    TORCH_CHECK(false, "latent_charge_sum: unknown kernel ", kernel);
  }
  TORCH_CHECK(coord.size(0) < std::numeric_limits<int>::max(),
              "latent_charge_sum: too many nodes");

  const c10::cuda::CUDAGuard guard(coord.device());
  coord = coord.contiguous();
  charge = charge.contiguous();
  node_mask = node_mask.contiguous();
  total_charge = total_charge.contiguous();
  const bool periodic = periodic_frames(box);
  const torch::Tensor cells = periodic ? box->contiguous() : torch::Tensor();
  coefficient = coefficient.contiguous();
  const Evaluation evaluation{cells,     coefficient, kernel,  channel,
                              prefactor, order,       spacing, periodic};
  if (deepmd::long_range::domain_of(communicator).size == 1) {
    return evaluation(coord, charge, node_mask, n_node_per_frame, total_charge,
                      0, static_cast<int>(coord.size(0)), nullptr);
  }
  return evaluate_across_domains(evaluation, coord, charge, node_mask,
                                 n_node_per_frame, total_charge, communicator);
}

}  // namespace long_range
}  // namespace deepmd

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.impl("latent_charge_sum", torch::kCUDA,
               &deepmd::long_range::latent_charge_sum);
}
