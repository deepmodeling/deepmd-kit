// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <iostream>
#include <string>
#include <vector>

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

template <typename FPTYPE>
void _prepare_coord_nlist_gpu(OpKernelContext* context,
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
