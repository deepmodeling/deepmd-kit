// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Border exchange of a domain-decomposed system (contract in
// border_exchange.h), and the differentiable operators deepmd::border_exchange
// and deepmd::border_exchange_backward built on it.
//
// The rows a swap sends are gathered into one contiguous buffer. Host rows,
// and device rows with an MPI library that reads device memory, travel from
// that buffer into the halo rows directly; other device rows are staged
// through pinned host buffers. A swap of rows of this rank copies them in
// place.

#include "border_exchange.h"

#ifdef DEEPMD_TORCH_HAS_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif
#include <ATen/ops/_assert_async.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

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

void check_rows(const torch::Tensor& rows, const char* op) {
  TORCH_CHECK(rows.dim() == 2 && rows.is_contiguous() &&
                  (rows.scalar_type() == torch::kFloat32 ||
                   rows.scalar_type() == torch::kFloat64) &&
                  (rows.is_cuda() || rows.device().is_cpu()),
              op,
              ": the rows must be a contiguous float32 or float64 matrix on "
              "the host or a CUDA device");
}

#ifdef USE_MPI
/// Wait until the stream of device rows has written them, before MPI or a
/// blocking copy reads them; host rows are complete already, and a library
/// built without CUDA registers the host kernels alone.
void complete(const torch::Tensor& rows) {
#ifdef DEEPMD_TORCH_HAS_CUDA
  if (rows.is_cuda()) {
    at::cuda::getCurrentCUDAStream().synchronize();
  }
#else
  (void)rows;
#endif
}

/// The MPI type of one row of ``rows``.
MPI_Datatype row_type_of(const torch::Tensor& rows) {
  MPI_Datatype row_type = MPI_DATATYPE_NULL;
  MPI_Type_contiguous(
      static_cast<int>(rows.size(1)),
      rows.scalar_type() == torch::kFloat64 ? MPI_DOUBLE : MPI_FLOAT,
      &row_type);
  MPI_Type_commit(&row_type);
  return row_type;
}

/// Whether MPI reads and writes the memory of ``rows`` itself.
bool mpi_reads(const torch::Tensor& rows) {
  return !rows.is_cuda() || mpi_reads_device_memory();
}
#endif

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
  TORCH_CHECK(send_nodes.dim() == 1 &&
                  send_nodes.scalar_type() == torch::kInt32 &&
                  send_nodes.is_contiguous(),
              op, ": the border send list must be a contiguous int32 vector");
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
  std::int64_t offset = 0;
  for (std::int64_t s = 0; s < swaps.size(0); ++s) {
    const Swap swap = swap_at(swaps, s);
    const auto indices = send_nodes.slice(0, offset, offset + swap.send_count);
    at::_assert_async(
        ((indices >= 0) & (indices < swap.first_receive)).all(),
        "a border swap must send only owned or earlier-received rows");
    offset += swap.send_count;
  }
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
  check_rows(rows, "border_exchange");
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
      row_type = row_type_of(rows);
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
    if (mpi_reads(rows)) {
      complete(rows);
      if (swap.receive_count > 0) {
        MPI_Irecv(receive_rows.data_ptr(), static_cast<int>(swap.receive_count),
                  row_type, swap.receive_rank, kBorderTag, comm,
                  &requests[pending++]);
      }
      if (swap.send_count > 0) {
        MPI_Isend(send_rows.data_ptr(), static_cast<int>(swap.send_count),
                  row_type, swap.send_rank, kBorderTag, comm,
                  &requests[pending++]);
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
      MPI_Irecv(staged_receive.data_ptr(), static_cast<int>(swap.receive_count),
                row_type, swap.receive_rank, kBorderTag, comm,
                &requests[pending++]);
    }
    if (swap.send_count > 0) {
      MPI_Isend(staged_send.data_ptr(), static_cast<int>(swap.send_count),
                row_type, swap.send_rank, kBorderTag, comm,
                &requests[pending++]);
    }
    MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
    if (swap.receive_count > 0) {
      receive_rows.copy_(staged_receive, /*non_blocking=*/true);
      // The pinned buffer is rewritten by the next swap, so its copy to the
      // device completes before the host reuses it.
      complete(rows);
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
  check_rows(rows, "border_accumulate");
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
      row_type = row_type_of(rows);
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
    if (mpi_reads(rows)) {
      complete(rows);
      if (swap.send_count > 0) {
        MPI_Irecv(returned.data_ptr(), static_cast<int>(swap.send_count),
                  row_type, swap.send_rank, kBorderTag, comm,
                  &requests[pending++]);
      }
      if (swap.receive_count > 0) {
        MPI_Isend(halo_rows.data_ptr(), static_cast<int>(swap.receive_count),
                  row_type, swap.receive_rank, kBorderTag, comm,
                  &requests[pending++]);
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
        MPI_Irecv(staged_receive.data_ptr(), static_cast<int>(swap.send_count),
                  row_type, swap.send_rank, kBorderTag, comm,
                  &requests[pending++]);
      }
      if (swap.receive_count > 0) {
        MPI_Isend(staged_send.data_ptr(), static_cast<int>(swap.receive_count),
                  row_type, swap.receive_rank, kBorderTag, comm,
                  &requests[pending++]);
      }
      MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
      if (swap.send_count > 0) {
        returned.copy_(staged_receive, /*non_blocking=*/true);
        // The pinned buffer is rewritten by the next swap, so its copy to the
        // device completes before the host reuses it.
        complete(rows);
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

namespace {

/// Check a border plan of ``rows`` and return its owned row count.
std::int64_t plan_rows(const torch::Tensor& rows,
                       const torch::Tensor& send_nodes,
                       const torch::Tensor& swaps,
                       const torch::Tensor& communicator,
                       const char* op) {
  TORCH_CHECK(rows.dim() == 2, op, ": the rows must form a matrix");
  TORCH_CHECK(send_nodes.device() == rows.device(), op,
              ": the border send list must be on the device of the rows");
  return check_border_plan(send_nodes, swaps, communicator, rows.size(0), op);
}

/// The rows with every halo row replaced by the row of the atom it images.
torch::Tensor border_exchange_op(const torch::Tensor& rows,
                                 const torch::Tensor& send_nodes,
                                 const torch::Tensor& swaps,
                                 const torch::Tensor& communicator) {
  plan_rows(rows, send_nodes, swaps, communicator, "border_exchange");
  auto refreshed = rows.clone(at::MemoryFormat::Contiguous);
  border_exchange(refreshed, send_nodes, swaps, communicator);
  return refreshed;
}

/// The gradient of border_exchange_op: the gradient of every halo row added
/// onto the row of the atom it images, and zero on the halo rows, whose input
/// values the exchange overwrites.
torch::Tensor border_exchange_backward_op(const torch::Tensor& gradient,
                                          const torch::Tensor& send_nodes,
                                          const torch::Tensor& swaps,
                                          const torch::Tensor& communicator) {
  const std::int64_t owned = plan_rows(
      gradient, send_nodes, swaps, communicator, "border_exchange_backward");
  auto accumulated = gradient.clone(at::MemoryFormat::Contiguous);
  border_accumulate(accumulated, send_nodes, swaps, communicator);
  accumulated.narrow(0, owned, accumulated.size(0) - owned).zero_();
  return accumulated;
}

}  // namespace
}  // namespace deepmd

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "border_exchange(Tensor rows, Tensor send_nodes, Tensor swaps, "
      "Tensor communicator) -> Tensor");
  m.def(
      "border_exchange_backward(Tensor gradient, Tensor send_nodes, "
      "Tensor swaps, Tensor communicator) -> Tensor");
}

TORCH_LIBRARY_IMPL(deepmd, CPU, m) {
  m.impl("border_exchange", &deepmd::border_exchange_op);
  m.impl("border_exchange_backward", &deepmd::border_exchange_backward_op);
}

// Device rows are ordered through the CUDA stream before MPI reads them, so
// only a library built against a CUDA-enabled PyTorch serves them.
#ifdef DEEPMD_TORCH_HAS_CUDA
TORCH_LIBRARY_IMPL(deepmd, CUDA, m) {
  m.impl("border_exchange", &deepmd::border_exchange_op);
  m.impl("border_exchange_backward", &deepmd::border_exchange_backward_op);
}
#endif
