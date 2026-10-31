// SPDX-License-Identifier: LGPL-3.0-or-later
//
// MPI exchanges of the latent-charge long range (contract in
// domain_exchange.h).
//
// With an MPI library that reads device memory (MPIX_Query_cuda_support), the
// rows travel between device buffers once the current stream has finished
// writing them; otherwise they are staged through pinned host buffers. Either
// way the host waits for the device once per exchange, before MPI reads the
// data; the results return to the device without a further wait.

#include "domain_exchange.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

#include <limits>
#include <vector>

#include "../communicator.h"

namespace deepmd {
namespace long_range {

/// The operator that names an exchange in its diagnostics.
constexpr const char* kOperator = "latent_charge_sum";

int domain_count(const torch::Tensor& communicator) {
  if (communicator_address(communicator, kOperator) == 0) {
    return 1;
  }
#ifdef USE_MPI
  int size = 1;
  MPI_Comm_size(communicator_handle(communicator, kOperator), &size);
  return size;
#else
  TORCH_CHECK(false,
              "latent_charge_sum: an evaluation across MPI ranks "
              "requires DeePMD-kit built with MPI");
#endif
}

std::pair<torch::Tensor, std::int64_t> gather_domain_rows(
    const torch::Tensor& communicator, const torch::Tensor& rows) {
#ifdef USE_MPI
  TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 &&
                  rows.scalar_type() == torch::kFloat64 && rows.is_contiguous(),
              "gather_domain_rows: rows must be a contiguous float64 CUDA "
              "matrix");
  const MPI_Comm comm = communicator_handle(communicator, kOperator);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  const std::int64_t width = rows.size(1);
  std::int64_t local = rows.size(0);
  std::vector<std::int64_t> counts(size);
  MPI_Allgather(&local, 1, MPI_INT64_T, counts.data(), 1, MPI_INT64_T, comm);
  std::vector<int> receive_counts(size);
  std::vector<int> displacements(size);
  std::int64_t total = 0;
  std::int64_t offset = 0;
  for (int r = 0; r < size; ++r) {
    if (r == rank) {
      offset = total;
    }
    TORCH_CHECK((total + counts[r]) * width <= std::numeric_limits<int>::max(),
                "gather_domain_rows: the rows of all ranks exceed the int "
                "range of an MPI count");
    receive_counts[r] = static_cast<int>(counts[r] * width);
    displacements[r] = static_cast<int>(total * width);
    total += counts[r];
  }
  const int send_count = static_cast<int>(local * width);
  auto gathered = torch::empty({total, width}, rows.options());
  if (mpi_reads_device_memory()) {
    at::cuda::getCurrentCUDAStream().synchronize();
    MPI_Allgatherv(rows.data_ptr<double>(), send_count, MPI_DOUBLE,
                   gathered.data_ptr<double>(), receive_counts.data(),
                   displacements.data(), MPI_DOUBLE, comm);
  } else {
    const auto host = rows.options().device(torch::kCPU).pinned_memory(true);
    auto host_rows = torch::empty({local, width}, host);
    host_rows.copy_(rows);
    auto host_gathered = torch::empty({total, width}, host);
    MPI_Allgatherv(host_rows.data_ptr<double>(), send_count, MPI_DOUBLE,
                   host_gathered.data_ptr<double>(), receive_counts.data(),
                   displacements.data(), MPI_DOUBLE, comm);
    gathered.copy_(host_gathered, /*non_blocking=*/true);
  }
  return {gathered, offset};
#else
  TORCH_CHECK(false,
              "gather_domain_rows: an evaluation across MPI ranks requires "
              "DeePMD-kit built with MPI");
#endif
}

torch::Tensor sum_over_domains(const torch::Tensor& communicator,
                               const torch::Tensor& values) {
#ifdef USE_MPI
  TORCH_CHECK(values.dim() == 1 && values.scalar_type() == torch::kFloat64,
              "sum_over_domains: values must be a float64 vector");
  const MPI_Comm comm = communicator_handle(communicator, kOperator);
  int size = 1;
  MPI_Comm_size(comm, &size);
  const torch::Tensor local = values.to(torch::kCPU).contiguous();
  const int count = static_cast<int>(local.numel());
  std::vector<double> every(static_cast<std::size_t>(size) * count);
  MPI_Allgather(local.data_ptr<double>(), count, MPI_DOUBLE, every.data(),
                count, MPI_DOUBLE, comm);
  auto total = torch::zeros({count}, local.options().pinned_memory(true));
  double* sum = total.data_ptr<double>();
  for (int r = 0; r < size; ++r) {
    for (int k = 0; k < count; ++k) {
      sum[k] += every[static_cast<std::size_t>(r) * count + k];
    }
  }
  return total.to(values.device(), /*non_blocking=*/true);
#else
  TORCH_CHECK(false,
              "sum_over_domains: an evaluation across MPI ranks requires "
              "DeePMD-kit built with MPI");
#endif
}

}  // namespace long_range
}  // namespace deepmd
