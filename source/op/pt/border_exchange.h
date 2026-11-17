// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Device-resident border exchange of a domain-decomposed system: the forward
// communication of border_op, with the send lists on the device, and its
// reverse, which accumulates the halo rows onto their atoms.
//
// The node set of a domain holds its owned rows first, then its halo rows,
// which images of atoms that other swaps deliver. A plan describes the swaps
// of a LAMMPS brick decomposition in model-node space:
//
//   swaps       int32 CPU tensor of shape (nswap, kBorderSwapColumns). Swap s
//               sends ``swaps[s, kBorderSendCount]`` rows to rank
//               ``swaps[s, kBorderSendRank]`` and receives
//               ``swaps[s, kBorderReceiveCount]`` rows from rank
//               ``swaps[s, kBorderReceiveRank]`` into the consecutive rows
//               from ``swaps[s, kBorderFirstReceive]``. The first swap
//               receives into the first halo row, every further swap into the
//               row after those of the swap before it, and the swaps together
//               fill every halo row.
//   send_nodes  int32 CUDA tensor holding the rows sent in every swap, the
//               swaps concatenated in order. A swap may send a halo row that
//               an earlier swap received.
//   communicator int64 CPU tensor of one element holding the address of the
//               MPI_Comm handle of the caller (communicator.h), zero for a
//               single process, whose swaps are all self swaps.
//
// A self swap, whose two ranks are this rank, copies the rows on the device.
// The swaps run in order, so the halo rows of an image of an image are filled
// after the rows they copy. Messages of zero rows are never posted, so a rank
// whose swaps carry no rows need not take part.

#pragma once

#include <torch/types.h>

#include <cstdint>

namespace deepmd {

/// Columns of a row of the swap table of a border plan.
enum BorderSwapColumn : int {
  kBorderSendRank = 0,
  kBorderReceiveRank = 1,
  kBorderSendCount = 2,
  kBorderReceiveCount = 3,
  kBorderFirstReceive = 4,
  kBorderSwapColumns = 5,
};

/// Check that a border plan fills the halo rows of a node set of
/// ``node_count`` rows, that its send list holds the rows of every swap and
/// that its communicator is well formed, and return the owned row count of
/// the node set: the first row the first swap receives into, or
/// ``node_count`` for a plan without swaps. ``op`` names the caller in the
/// error messages.
std::int64_t check_border_plan(const torch::Tensor& send_nodes,
                               const torch::Tensor& swaps,
                               const torch::Tensor& communicator,
                               std::int64_t node_count,
                               const char* op);

/// Replace every halo row of ``rows``, a contiguous float32 CUDA matrix of
/// shape (node_count, W), by the row of the atom it images, swap by swap. The
/// host waits for the current stream once before every swap that MPI carries,
/// so that MPI reads complete rows; the received rows reach the device in
/// stream order.
void border_exchange(const torch::Tensor& rows,
                     const torch::Tensor& send_nodes,
                     const torch::Tensor& swaps,
                     const torch::Tensor& communicator);

/// The reverse of border_exchange: add every halo row of ``rows`` onto the row
/// of the atom it images, swap by swap in reverse order, so that a halo row
/// that a later swap sent on receives the additions of its own images before
/// it adds onto the row it was copied from. The halo rows keep their values.
/// A swap of a LAMMPS plan sends every row at most once, so every row receives
/// at most one addition per swap and the sums do not depend on the order of
/// the device atomics.
void border_accumulate(const torch::Tensor& rows,
                       const torch::Tensor& send_nodes,
                       const torch::Tensor& swaps,
                       const torch::Tensor& communicator);

}  // namespace deepmd
