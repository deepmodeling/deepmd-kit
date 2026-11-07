// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/public/version.h"

#if (TF_MAJOR_VERSION > 2) || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 20)
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#endif

#include "device.h"
#include "neighbor_list.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// functions used in custom ops
struct DeviceFunctor {
  void operator()(std::string& device, const CPUDevice& d) { device = "CPU"; }
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  void operator()(std::string& device, const GPUDevice& d) { device = "GPU"; }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
};

namespace deepmd {
void safe_compute(OpKernelContext* context,
                  std::function<void(OpKernelContext*)> ff);
};

namespace deepmd {
namespace tf_compat {
#if (TF_MAJOR_VERSION > 2) || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 20)
using Status = absl::Status;
#else
using Status = tensorflow::Status;
#endif

template <typename... Args>
inline Status InvalidArgument(Args&&... args) {
#if (TF_MAJOR_VERSION > 2) || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 20)
  return absl::InvalidArgumentError(absl::StrCat(std::forward<Args>(args)...));
#else
  return tensorflow::errors::InvalidArgument(std::forward<Args>(args)...);
#endif
}

/**
 * @brief Derive a dense tensor's per-atom width without truncating division.
 *
 * Several low-level TensorFlow ops flatten atom and feature dimensions into a
 * single axis.  Validate the flattened width before dividing by `nloc`; raw
 * CPU/GPU kernels cannot safely consume a leftover partial atom row.
 *
 * @param per_atom_width Receives the validated feature width for one atom.
 * @param shape Rank-two tensor shape whose second dimension is flattened.
 * @param nloc Number of local atoms encoded in the flattened dimension.
 * @param tensor_name Human-readable input name used in validation errors.
 * @return An OK status, or InvalidArgument when the width is incompatible.
 */
inline Status GetPerAtomWidth(int* per_atom_width,
                              const TensorShape& shape,
                              const int nloc,
                              const char* tensor_name) {
  const int64_t flattened_width = shape.dim_size(1);
  if (nloc < 0) {
    return InvalidArgument("number of local atoms should be non-negative");
  }
  if (nloc == 0) {
    if (flattened_width != 0) {
      return InvalidArgument(tensor_name,
                             " width should be zero when nloc is zero");
    }
    *per_atom_width = 0;
    return Status();
  }
  if (flattened_width % nloc != 0) {
    return InvalidArgument(tensor_name, " width ", flattened_width,
                           " should be divisible by nloc ", nloc);
  }
  const int64_t width = flattened_width / nloc;
  if (width > std::numeric_limits<int>::max()) {
    return InvalidArgument(tensor_name,
                           " width per atom exceeds the supported int range");
  }
  *per_atom_width = static_cast<int>(width);
  return Status();
}
}  // namespace tf_compat
}  // namespace deepmd

template <typename FPTYPE>
tensorflow::Status _prepare_coord_nlist_gpu(OpKernelContext* context,
                                            Tensor* tensor_list,
                                            FPTYPE const** coord,
                                            FPTYPE*& coord_cpy,
                                            int const** type,
                                            int*& type_cpy,
                                            int*& idx_mapping,
                                            deepmd::InputNlist& inlist,
                                            int*& ilist,
                                            int*& numneigh,
                                            int**& firstneigh,
                                            int*& jlist,
                                            int*& nbor_list_dev,
                                            int& new_nall,
                                            int& mem_cpy,
                                            int& mem_nnei,
                                            int& max_nbor_size,
                                            const FPTYPE* box,
                                            const int* mesh_tensor_data,
                                            const int mesh_tensor_size,
                                            const int& nloc,
                                            const int& nei_mode,
                                            const float& rcut_r,
                                            const int& max_cpy_trial,
                                            const int& max_nnei_trial);
