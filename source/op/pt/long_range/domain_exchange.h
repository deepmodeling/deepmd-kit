// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Exchange of the long-range direct sum between the domains of a decomposed
// system.
//
// A domain is one MPI rank of the communicator that a caller hands to
// deepmd::latent_charge_direct_sum as an int64 tensor holding the bytes of its
// MPI_Comm handle, the convention of deepmd_export::border_op. A zero handle
// stands for a single domain. Both exchanges follow the rank order, so every
// rank receives the same rows and the same sums, independent of the
// reduction algorithm of the MPI library.

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <utility>

namespace deepmd {
namespace long_range {

struct Domain {
  int rank;
  int size;
};

// The rank of this process and the number of ranks of ``communicator``; one
// rank for a zero handle.
Domain domain_of(const torch::Tensor& communicator);

// Concatenates, in rank order, the rows of every rank. ``rows`` is a
// contiguous float64 matrix of shape (M_r, W) on the CUDA device, with the
// same W on every rank. Returns the rows of all ranks, of shape (sum_r M_r,
// W) on the device of ``rows``, and the index of the first row of this rank.
std::pair<torch::Tensor, std::int64_t> gather_domain_rows(
    const torch::Tensor& communicator, const torch::Tensor& rows);

// Sums the float64 vector ``values`` over the ranks, adding the vectors of
// the ranks in rank order. Returns the sum on the device of ``values``.
torch::Tensor sum_over_domains(const torch::Tensor& communicator,
                               const torch::Tensor& values);

}  // namespace long_range
}  // namespace deepmd
