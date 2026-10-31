// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Device-resident border exchange (contract in border_exchange.h).
//
// The rows a swap sends are gathered on the device into one contiguous
// buffer. With an MPI library that reads device memory they travel from that
// buffer into the halo rows directly; otherwise they are staged through pinned
// host buffers. A swap of rows of this rank copies them on the device.

#include "border_exchange.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "communicator.h"

namespace deepmd {
namespace {

/// Tag of the messages of a border exchange. Messages between two ranks are
/// received in the order they are sent, so one tag serves every swap.
constexpr int kBorderTag = 0x4450;

struct Swap {
  int send_rank;
  int receive_rank;
  std::int64_t send_count;
  std::int64_t receive_count;
  std::int64_t first_receive;
};

Swap swap_at(const torch::Tensor& swaps, std::int64_t index) {
  const int* row = swaps.data_ptr<int>() + index * kBorderSwapColumns;
  return {row[kBorderSendRank], row[kBorderReceiveRank], row[kBorderSendCount],
          row[kBorderReceiveCount], row[kBorderFirstReceive]};
}

}  // namespace

std::int64_t check_border_plan(const torch::Tensor& send_nodes,
                               const torch::Tensor& swaps,
                               const torch::Tensor& communicator,
                               std::int64_t node_count,
                               const char* op) {
  TORCH_CHECK(swaps.device().is_cpu() && swaps.scalar_type() == torch::kInt32 &&
                  swaps.dim() == 2 && swaps.size(1) == kBorderSwapColumns &&
                  swaps.is_contiguous(),
              op,
              ": the border swaps must be a contiguous int32 CPU tensor of "
              "shape (nswap, ",
              static_cast<int>(kBorderSwapColumns), ")");
  TORCH_CHECK(send_nodes.is_cuda() && send_nodes.dim() == 1 &&
                  send_nodes.scalar_type() == torch::kInt32 &&
                  send_nodes.is_contiguous(),
              op,
              ": the border send list must be a contiguous int32 CUDA "
              "vector");
  communicator_address(communicator, op);
  TORCH_CHECK(node_count <= std::numeric_limits<int>::max(), op,
              ": a border exchange addresses its rows with int32 indices, but "
              "the node set holds ",
              node_count, " rows");
  const std::int64_t owned =
      swaps.size(0) == 0 ? node_count : swap_at(swaps, 0).first_receive;
  std::int64_t next = owned;
  std::int64_t sent = 0;
  for (std::int64_t s = 0; s < swaps.size(0); ++s) {
    const Swap swap = swap_at(swaps, s);
    TORCH_CHECK(swap.send_count >= 0 && swap.receive_count >= 0, op, ": swap ",
                s, " of the border plan has a negative row count");
    TORCH_CHECK(swap.first_receive == next, op, ": swap ", s,
                " of the border plan receives into the rows from ",
                swap.first_receive, ", but the halo rows before it end at ",
                next);
    next += swap.receive_count;
    sent += swap.send_count;
  }
  TORCH_CHECK(owned >= 0 && next == node_count, op,
              ": the swaps of the border plan fill the rows from ", owned,
              " up to ", next, ", but the node set holds ", node_count,
              " rows");
  TORCH_CHECK(send_nodes.numel() >= sent, op,
              ": the swaps of the border plan "
              "send ",
              sent, " rows, but the send list holds ", send_nodes.numel());
  return owned;
}

void border_exchange(const torch::Tensor& rows,
                     const torch::Tensor& send_nodes,
                     const torch::Tensor& swaps,
                     const torch::Tensor& communicator) {
  const std::int64_t nswap = swaps.size(0);
  if (nswap == 0) {
    return;
  }
  TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 &&
                  rows.scalar_type() == torch::kFloat32 && rows.is_contiguous(),
              "border_exchange: the rows must be a contiguous float32 CUDA "
              "matrix");
  const std::int64_t width = rows.size(1);
  const std::int64_t address =
      communicator_address(communicator, "border_exchange");
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Datatype row_type = MPI_DATATYPE_NULL;
  if (address != 0) {
    comm = communicator_handle(communicator, "border_exchange");
    MPI_Comm_rank(comm, &rank);
  }
#endif
  std::int64_t largest = 0;
  for (std::int64_t s = 0; s < nswap; ++s) {
    const Swap swap = swap_at(swaps, s);
    largest = std::max({largest, swap.send_count, swap.receive_count});
  }
  auto packed = torch::empty({largest, width}, rows.options());
  torch::Tensor host_send;
  torch::Tensor host_receive;
  std::int64_t offset = 0;
  for (std::int64_t s = 0; s < nswap; ++s) {
    const Swap swap = swap_at(swaps, s);
    const torch::Tensor sent = send_nodes.narrow(0, offset, swap.send_count);
    offset += swap.send_count;
    auto send_rows = packed.narrow(0, 0, swap.send_count);
    auto receive_rows = rows.narrow(0, swap.first_receive, swap.receive_count);
    if (swap.send_rank == rank && swap.receive_rank == rank) {
      TORCH_CHECK(swap.send_count == swap.receive_count,
                  "border_exchange: self swap ", s, " sends ", swap.send_count,
                  " rows but receives ", swap.receive_count);
      at::index_select_out(send_rows, rows, 0, sent);
      receive_rows.copy_(send_rows);
      continue;
    }
#ifdef USE_MPI
    TORCH_CHECK(address != 0, "border_exchange: swap ", s, " involves rank ",
                swap.send_rank, " or ", swap.receive_rank,
                ", but no communicator was given");
    if (row_type == MPI_DATATYPE_NULL) {
      MPI_Type_contiguous(static_cast<int>(width), MPI_FLOAT, &row_type);
      MPI_Type_commit(&row_type);
    }
    TORCH_CHECK(swap.send_count <= std::numeric_limits<int>::max() &&
                    swap.receive_count <= std::numeric_limits<int>::max(),
                "border_exchange: a swap exceeds the int range of an MPI "
                "count");
    if (swap.send_count > 0) {
      at::index_select_out(send_rows, rows, 0, sent);
    }
    MPI_Request requests[2];
    int pending = 0;
    if (mpi_reads_device_memory()) {
      // MPI reads the rows of the device once the stream has written them.
      at::cuda::getCurrentCUDAStream().synchronize();
      if (swap.receive_count > 0) {
        MPI_Irecv(receive_rows.data_ptr<float>(),
                  static_cast<int>(swap.receive_count), row_type,
                  swap.receive_rank, kBorderTag, comm, &requests[pending++]);
      }
      if (swap.send_count > 0) {
        MPI_Isend(send_rows.data_ptr<float>(),
                  static_cast<int>(swap.send_count), row_type, swap.send_rank,
                  kBorderTag, comm, &requests[pending++]);
      }
      MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
      continue;
    }
    if (!host_send.defined()) {
      const auto pinned =
          rows.options().device(torch::kCPU).pinned_memory(true);
      host_send = torch::empty({largest, width}, pinned);
      host_receive = torch::empty({largest, width}, pinned);
    }
    auto staged_send = host_send.narrow(0, 0, swap.send_count);
    auto staged_receive = host_receive.narrow(0, 0, swap.receive_count);
    if (swap.send_count > 0) {
      // A blocking copy waits for the stream, so the staged rows are complete.
      staged_send.copy_(send_rows);
    }
    if (swap.receive_count > 0) {
      MPI_Irecv(staged_receive.data_ptr<float>(),
                static_cast<int>(swap.receive_count), row_type,
                swap.receive_rank, kBorderTag, comm, &requests[pending++]);
    }
    if (swap.send_count > 0) {
      MPI_Isend(staged_send.data_ptr<float>(),
                static_cast<int>(swap.send_count), row_type, swap.send_rank,
                kBorderTag, comm, &requests[pending++]);
    }
    MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
    if (swap.receive_count > 0) {
      receive_rows.copy_(staged_receive, /*non_blocking=*/true);
      // The pinned buffer is rewritten by the next swap, so its copy to the
      // device completes before the host reuses it.
      at::cuda::getCurrentCUDAStream().synchronize();
    }
#else
    TORCH_CHECK(false, "border_exchange: swap ", s, " involves rank ",
                swap.send_rank, " or ", swap.receive_rank,
                ", which requires DeePMD-kit built with MPI");
#endif
  }
#ifdef USE_MPI
  if (row_type != MPI_DATATYPE_NULL) {
    MPI_Type_free(&row_type);
  }
#endif
}

void border_accumulate(const torch::Tensor& rows,
                       const torch::Tensor& send_nodes,
                       const torch::Tensor& swaps,
                       const torch::Tensor& communicator) {
  const std::int64_t nswap = swaps.size(0);
  if (nswap == 0) {
    return;
  }
  TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 &&
                  rows.scalar_type() == torch::kFloat32 && rows.is_contiguous(),
              "border_accumulate: the rows must be a contiguous float32 CUDA "
              "matrix");
  const std::int64_t width = rows.size(1);
  const std::int64_t address =
      communicator_address(communicator, "border_accumulate");
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Datatype row_type = MPI_DATATYPE_NULL;
  if (address != 0) {
    comm = communicator_handle(communicator, "border_accumulate");
    MPI_Comm_rank(comm, &rank);
  }
#endif
  std::int64_t largest = 0;
  std::vector<std::int64_t> offsets(nswap + 1, 0);
  for (std::int64_t s = 0; s < nswap; ++s) {
    const Swap swap = swap_at(swaps, s);
    largest = std::max({largest, swap.send_count, swap.receive_count});
    offsets[s + 1] = offsets[s] + swap.send_count;
  }
  auto packed = torch::empty({largest, width}, rows.options());
  torch::Tensor host_send;
  torch::Tensor host_receive;
  for (std::int64_t s = nswap - 1; s >= 0; --s) {
    const Swap swap = swap_at(swaps, s);
    const torch::Tensor sent =
        send_nodes.narrow(0, offsets[s], swap.send_count);
    const auto halo_rows =
        rows.narrow(0, swap.first_receive, swap.receive_count);
    if (swap.send_rank == rank && swap.receive_rank == rank) {
      TORCH_CHECK(swap.send_count == swap.receive_count,
                  "border_accumulate: self swap ", s, " sends ",
                  swap.send_count, " rows but receives ", swap.receive_count);
      // The addition reads its source apart from the rows it adds onto, so
      // the halo rows are copied first.
      auto returned = packed.narrow(0, 0, swap.receive_count);
      returned.copy_(halo_rows);
      rows.index_add_(0, sent, returned);
      continue;
    }
#ifdef USE_MPI
    TORCH_CHECK(address != 0, "border_accumulate: swap ", s, " involves rank ",
                swap.send_rank, " or ", swap.receive_rank,
                ", but no communicator was given");
    if (row_type == MPI_DATATYPE_NULL) {
      MPI_Type_contiguous(static_cast<int>(width), MPI_FLOAT, &row_type);
      MPI_Type_commit(&row_type);
    }
    TORCH_CHECK(swap.send_count <= std::numeric_limits<int>::max() &&
                    swap.receive_count <= std::numeric_limits<int>::max(),
                "border_accumulate: a swap exceeds the int range of an MPI "
                "count");
    // The halo rows travel back to the rank they came from, and the rows this
    // rank sent return from the rank it sent them to.
    auto returned = packed.narrow(0, 0, swap.send_count);
    MPI_Request requests[2];
    int pending = 0;
    if (mpi_reads_device_memory()) {
      at::cuda::getCurrentCUDAStream().synchronize();
      if (swap.send_count > 0) {
        MPI_Irecv(returned.data_ptr<float>(), static_cast<int>(swap.send_count),
                  row_type, swap.send_rank, kBorderTag, comm,
                  &requests[pending++]);
      }
      if (swap.receive_count > 0) {
        MPI_Isend(halo_rows.data_ptr<float>(),
                  static_cast<int>(swap.receive_count), row_type,
                  swap.receive_rank, kBorderTag, comm, &requests[pending++]);
      }
      MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
    } else {
      if (!host_send.defined()) {
        const auto pinned =
            rows.options().device(torch::kCPU).pinned_memory(true);
        host_send = torch::empty({largest, width}, pinned);
        host_receive = torch::empty({largest, width}, pinned);
      }
      auto staged_send = host_send.narrow(0, 0, swap.receive_count);
      auto staged_receive = host_receive.narrow(0, 0, swap.send_count);
      if (swap.receive_count > 0) {
        // A blocking copy waits for the stream, so the staged rows are
        // complete.
        staged_send.copy_(halo_rows);
      }
      if (swap.send_count > 0) {
        MPI_Irecv(staged_receive.data_ptr<float>(),
                  static_cast<int>(swap.send_count), row_type, swap.send_rank,
                  kBorderTag, comm, &requests[pending++]);
      }
      if (swap.receive_count > 0) {
        MPI_Isend(staged_send.data_ptr<float>(),
                  static_cast<int>(swap.receive_count), row_type,
                  swap.receive_rank, kBorderTag, comm, &requests[pending++]);
      }
      MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
      if (swap.send_count > 0) {
        returned.copy_(staged_receive, /*non_blocking=*/true);
        // The pinned buffer is rewritten by the next swap, so its copy to the
        // device completes before the host reuses it.
        at::cuda::getCurrentCUDAStream().synchronize();
      }
    }
    if (swap.send_count > 0) {
      rows.index_add_(0, sent, returned);
    }
#else
    TORCH_CHECK(false, "border_accumulate: swap ", s, " involves rank ",
                swap.send_rank, " or ", swap.receive_rank,
                ", which requires DeePMD-kit built with MPI");
#endif
  }
#ifdef USE_MPI
  if (row_type != MPI_DATATYPE_NULL) {
    MPI_Type_free(&row_type);
  }
#endif
}

}  // namespace deepmd
