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

#ifdef USE_MPI
#include <mpi.h>
#ifdef OMPI_MPI_H
#include <mpi-ext.h>
#endif
#endif
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

#include <limits>
#include <vector>

namespace deepmd {
namespace long_range {
namespace {

std::int64_t address_of(const torch::Tensor& communicator) {
  TORCH_CHECK(communicator.device().is_cpu() &&
                  communicator.scalar_type() == torch::kInt64 &&
                  communicator.numel() == 1,
              "latent_charge_sum: communicator must be an int64 CPU "
              "tensor with one element");
  return communicator.contiguous().data_ptr<std::int64_t>()[0];
}

#ifdef USE_MPI
// The handle of the caller, of the width of MPI_Comm in this MPI library: an
// int in MPICH, a pointer in Open MPI.
MPI_Comm communicator_of(const torch::Tensor& communicator) {
  static_assert(sizeof(std::uintptr_t) <= sizeof(std::int64_t),
                "an address must fit in an int64 tensor");
  return *reinterpret_cast<const MPI_Comm*>(
      static_cast<std::uintptr_t>(address_of(communicator)));
}

bool reads_device_memory() {
#ifdef NO_CUDA_AWARE
  return false;
#else
  return MPIX_Query_cuda_support() != 0;
#endif
}
#endif

}  // namespace

int domain_count(const torch::Tensor& communicator) {
  if (address_of(communicator) == 0) {
    return 1;
  }
#ifdef USE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  TORCH_CHECK(initialized != 0,
              "latent_charge_sum: a communicator was given, but MPI is "
              "not initialized");
  int size = 1;
  MPI_Comm_size(communicator_of(communicator), &size);
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
  const MPI_Comm comm = communicator_of(communicator);
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
  if (reads_device_memory()) {
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
  const MPI_Comm comm = communicator_of(communicator);
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
