// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <iostream>
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
