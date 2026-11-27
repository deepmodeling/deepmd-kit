// SPDX-License-Identifier: LGPL-3.0-or-later
//
// The MPI communicator of a decomposed system, as the operators receive it: an
// int64 CPU tensor of one element that holds the address of the caller's
// MPI_Comm handle, or zero for a single process. The handle is read with the
// width of MPI_Comm of the MPI library the operators are built against: an int
// in MPICH, a pointer in Open MPI.

#pragma once

#include <c10/util/Exception.h>
#include <torch/types.h>

#include <cstdint>

#ifdef USE_MPI
#include <mpi.h>
#ifdef OMPI_MPI_H
#include <mpi-ext.h>
#endif
#endif

namespace deepmd {

/// The address held by ``communicator``, zero for a single process.
inline std::int64_t communicator_address(const torch::Tensor& communicator,
                                         const char* op) {
  TORCH_CHECK(communicator.device().is_cpu() &&
                  communicator.scalar_type() == torch::kInt64 &&
                  communicator.numel() == 1,
              op,
              ": the communicator must be an int64 CPU tensor of one element");
  return communicator.contiguous().data_ptr<std::int64_t>()[0];
}

#ifdef USE_MPI
/// The MPI_Comm handle at the address held by ``communicator``.
inline MPI_Comm communicator_handle(const torch::Tensor& communicator,
                                    const char* op) {
  static_assert(sizeof(std::uintptr_t) <= sizeof(std::int64_t),
                "an address must fit in an int64 tensor");
  const std::int64_t address = communicator_address(communicator, op);
  TORCH_CHECK(address != 0, op, ": the communicator holds no handle");
  int initialized = 0;
  MPI_Initialized(&initialized);
  TORCH_CHECK(initialized != 0, op,
              ": a communicator was given, but MPI is not initialized");
  return *reinterpret_cast<const MPI_Comm*>(
      static_cast<std::uintptr_t>(address));
}

/// Whether the MPI library reads and writes device memory.
inline bool mpi_reads_device_memory() {
#ifdef NO_CUDA_AWARE
  return false;
#else
  return MPIX_Query_cuda_support() != 0;
#endif
}
#endif

}  // namespace deepmd
