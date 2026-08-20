// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Host-side launch interface for the geometrically compressed DPA1 CUDA
// descriptor.
//
// The Torch-facing translation unit validates tensors and converts them into
// this plain pointer bundle. Each supported channel width owns one CUDA
// translation unit that instantiates the corresponding forward and backward
// kernels. Keeping this interface free of ATen headers avoids reparsing the
// PyTorch C++ API in every specialization.

#pragma once

#include <cuda_runtime.h>

namespace deepmd_dpa1_compress {

enum class IndexKind : int {
  kInt32 = 0,
  kInt64 = 1,
  kUInt32 = 2,
};

struct Arguments {
  long node_count = 0;
  long edge_count = 0;
  int device = 0;
  int device_major = 0;
  int multiprocessor_count = 0;
  int basis_dim = 4;
  int type_count = 0;
  int axis = 0;
  int type_embedding_dim = 0;
  int descriptor_stride = 0;
  bool one_side = false;
  bool smooth = false;
  bool canonical = false;
  bool masked = false;
  bool concatenate_type_embedding = false;
  bool write_rotation = false;
  IndexKind index_kind = IndexKind::kInt64;

  float rcut = 0.0f;
  float rcut_smooth = 0.0f;
  float protection = 0.0f;
  float inverse_neighbors = 0.0f;
  float lower = 0.0f;
  float upper = 0.0f;
  float table_max = 0.0f;
  float stride0 = 0.0f;
  float stride1 = 0.0f;

  const float* edge_vec = nullptr;
  const void* edge_index = nullptr;
  const bool* edge_mask = nullptr;
  const void* destination_order = nullptr;
  const long* destination_row_ptr = nullptr;
  const long* atype = nullptr;
  const float* type_embedding = nullptr;
  const float* average = nullptr;
  const float* inverse_stddev = nullptr;
  const float* degree_gain = nullptr;
  const float* table = nullptr;
  const float* gate_table = nullptr;

  const float* descriptor_gradient = nullptr;
  const float* rotation_gradient = nullptr;
  const float* moment = nullptr;

  float* descriptor = nullptr;
  float* rotation = nullptr;
  float* moment_out = nullptr;
  float* edge_gradient = nullptr;
};

#define DPA1_COMPRESS_FOR_EACH_CHANNEL(macro) \
  macro(8) macro(16) macro(32) macro(64) macro(128) macro(256)

#define DPA1_COMPRESS_DECLARE_CHANNEL(width)                       \
  cudaError_t launch_forward_c##width(const Arguments& arguments,  \
                                      cudaStream_t stream);        \
  cudaError_t launch_backward_c##width(const Arguments& arguments, \
                                       cudaStream_t stream);

DPA1_COMPRESS_FOR_EACH_CHANNEL(DPA1_COMPRESS_DECLARE_CHANNEL)

#undef DPA1_COMPRESS_DECLARE_CHANNEL

}  // namespace deepmd_dpa1_compress
