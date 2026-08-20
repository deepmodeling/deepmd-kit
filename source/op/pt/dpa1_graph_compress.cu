// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Torch bindings and runtime dispatch of the compressed DPA1 CUDA descriptor.
//
// The CUDA kernels are instantiated in one translation unit per channel width.
// This translation unit only validates tensors, builds the plain launch
// argument bundle, dispatches by width, and registers the operators.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>

#include "dpa1_graph_compress_launch.h"

#ifndef DEEPMD_ENABLE_DPA1_HIGH_LMAX
#define DEEPMD_ENABLE_DPA1_HIGH_LMAX 0
#endif

namespace {

using deepmd_dpa1_compress::Arguments;
using deepmd_dpa1_compress::IndexKind;

cudaError_t dispatch_width(int width,
                           bool backward,
                           const Arguments& arguments,
                           cudaStream_t stream) {
#define DPA1_COMPRESS_DISPATCH(width_value)                                 \
  if (width == width_value) {                                               \
    return backward ? deepmd_dpa1_compress::launch_backward_c##width_value( \
                          arguments, stream)                                \
                    : deepmd_dpa1_compress::launch_forward_c##width_value(  \
                          arguments, stream);                               \
  }
  DPA1_COMPRESS_FOR_EACH_CHANNEL(DPA1_COMPRESS_DISPATCH)
#undef DPA1_COMPRESS_DISPATCH
  TORCH_CHECK(false, "dpa1_graph_compress: unsupported width ", width);
  return cudaErrorInvalidValue;
}

void check_launch(const char* operation, const cudaError_t error) {
  TORCH_CHECK(error == cudaSuccess, operation, ": ", cudaGetErrorString(error));
}

IndexKind index_kind_from_tensor(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == torch::kInt32) {
    return IndexKind::kInt32;
  }
  if (tensor.scalar_type() == torch::kUInt32) {
    return IndexKind::kUInt32;
  }
  return IndexKind::kInt64;
}

const void* index_data_ptr(const torch::Tensor& tensor, IndexKind kind) {
  switch (kind) {
    case IndexKind::kInt32:
      return static_cast<const void*>(tensor.data_ptr<int>());
    case IndexKind::kUInt32:
      return static_cast<const void*>(tensor.data_ptr<std::uint32_t>());
    case IndexKind::kInt64:
      return static_cast<const void*>(tensor.data_ptr<long>());
  }
  TORCH_CHECK(false, "dpa1_graph_compress: unsupported index kind");
  return nullptr;
}

Arguments make_common_arguments(long node_count,
                                int basis_dim,
                                int type_count,
                                bool one_side,
                                bool smooth,
                                int axis,
                                bool canonical,
                                float rcut,
                                float rcut_smooth,
                                float protection,
                                float inverse_neighbors,
                                float lower,
                                float upper,
                                float table_max,
                                float stride0,
                                float stride1,
                                const torch::Tensor& edge_vec,
                                const torch::Tensor& edge_index,
                                const torch::Tensor& edge_mask,
                                const torch::Tensor& destination_order,
                                const torch::Tensor& destination_row_ptr,
                                const torch::Tensor& atype,
                                const torch::Tensor& average,
                                const torch::Tensor& inverse_stddev,
                                const torch::Tensor& degree_gain,
                                const torch::Tensor& table,
                                const torch::Tensor& gate_table) {
  const int device = edge_vec.get_device();
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(device);
  TORCH_CHECK(properties != nullptr,
              "dpa1_graph_compress: cannot query CUDA device properties");

  Arguments arguments;
  arguments.node_count = node_count;
  arguments.edge_count = edge_vec.size(0);
  arguments.device = device;
  arguments.device_major = properties->major;
  arguments.multiprocessor_count = properties->multiProcessorCount;
  arguments.basis_dim = basis_dim;
  arguments.type_count = type_count;
  arguments.axis = axis;
  arguments.one_side = one_side;
  arguments.smooth = smooth;
  arguments.canonical = canonical;
  arguments.masked = edge_mask.numel() != 0;
  arguments.index_kind = index_kind_from_tensor(edge_index);
  arguments.rcut = rcut;
  arguments.rcut_smooth = rcut_smooth;
  arguments.protection = protection;
  arguments.inverse_neighbors = inverse_neighbors;
  arguments.lower = lower;
  arguments.upper = upper;
  arguments.table_max = table_max;
  arguments.stride0 = stride0;
  arguments.stride1 = stride1;
  arguments.edge_vec = edge_vec.data_ptr<float>();
  arguments.edge_index = index_data_ptr(edge_index, arguments.index_kind);
  arguments.edge_mask = arguments.masked ? edge_mask.data_ptr<bool>() : nullptr;
  arguments.destination_order =
      destination_order.numel() == 0
          ? nullptr
          : index_data_ptr(destination_order, arguments.index_kind);
  arguments.destination_row_ptr = destination_row_ptr.data_ptr<long>();
  arguments.atype = atype.data_ptr<long>();
  arguments.average = average.data_ptr<float>();
  arguments.inverse_stddev = inverse_stddev.data_ptr<float>();
  arguments.degree_gain =
      degree_gain.numel() == 0 ? nullptr : degree_gain.data_ptr<float>();
  arguments.table = table.data_ptr<float>();
  arguments.gate_table = gate_table.data_ptr<float>();
  return arguments;
}

void validate_inputs(const torch::Tensor& edge_vec,
                     const torch::Tensor& edge_index,
                     const torch::Tensor& edge_mask,
                     const torch::Tensor& destination_order,
                     const torch::Tensor& destination_row_ptr,
                     const torch::Tensor& atype,
                     const torch::Tensor& average,
                     const torch::Tensor& inverse_stddev,
                     const torch::Tensor& table,
                     const torch::Tensor& gate_table,
                     int width,
                     int axis) {
  TORCH_CHECK(edge_vec.is_cuda() && edge_index.is_cuda() &&
                  edge_mask.is_cuda() && destination_order.is_cuda() &&
                  destination_row_ptr.is_cuda() && atype.is_cuda() &&
                  average.is_cuda() && inverse_stddev.is_cuda() &&
                  table.is_cuda() && gate_table.is_cuda(),
              "dpa1_graph_compress: inputs must be CUDA tensors");
  TORCH_CHECK(
      edge_vec.is_contiguous() && edge_index.is_contiguous() &&
          edge_mask.is_contiguous() && destination_order.is_contiguous() &&
          destination_row_ptr.is_contiguous() && atype.is_contiguous() &&
          average.is_contiguous() && inverse_stddev.is_contiguous() &&
          table.is_contiguous() && gate_table.is_contiguous(),
      "dpa1_graph_compress: inputs must be contiguous");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt32 ||
                  edge_index.scalar_type() == torch::kUInt32 ||
                  edge_index.scalar_type() == torch::kInt64,
              "dpa1_graph_compress: edge_index must be int32, uint32, or "
              "int64");
  TORCH_CHECK(destination_order.scalar_type() == edge_index.scalar_type(),
              "dpa1_graph_compress: destination_order must match the "
              "edge_index dtype");
  TORCH_CHECK(edge_mask.scalar_type() == torch::kBool,
              "dpa1_graph_compress: edge_mask must be bool");
  TORCH_CHECK(atype.scalar_type() == torch::kInt64,
              "dpa1_graph_compress: atype must be int64");
  TORCH_CHECK(destination_row_ptr.scalar_type() == torch::kInt64,
              "dpa1_graph_compress: destination_row_ptr must be int64");
  TORCH_CHECK(average.scalar_type() == torch::kFloat32 &&
                  inverse_stddev.scalar_type() == torch::kFloat32 &&
                  table.scalar_type() == torch::kFloat32 &&
                  gate_table.scalar_type() == torch::kFloat32,
              "dpa1_graph_compress: statistics and tables must be fp32");
  TORCH_CHECK(axis > 0 && axis <= 16 && axis <= width,
              "dpa1_graph_compress: axis must be in [1, min(16, width)]");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dpa1_graph_compress(
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor type_embedding,
    torch::Tensor average,
    torch::Tensor inverse_stddev,
    torch::Tensor degree_gain,
    torch::Tensor table,
    torch::Tensor gate_table,
    int64_t type_one_side,
    int64_t concatenate_type_embedding,
    int64_t write_rotation,
    int64_t smooth,
    int64_t axis,
    bool canonical,
    double lower,
    double upper,
    double table_max,
    double stride0,
    double stride1,
    double rcut,
    double rcut_smooth,
    double protection,
    double neighbors,
    int64_t basis_dim) {
  const long node_count = atype.size(0);
  const int width = static_cast<int>(table.size(1) / 6);
  validate_inputs(edge_vec, edge_index, edge_mask, destination_order,
                  destination_row_ptr, atype, average, inverse_stddev, table,
                  gate_table, width, static_cast<int>(axis));
  TORCH_CHECK(type_embedding.is_cuda() && type_embedding.is_contiguous() &&
                  type_embedding.scalar_type() == torch::kFloat32,
              "dpa1_graph_compress: type_embedding must be contiguous fp32 "
              "on CUDA");
#if DEEPMD_ENABLE_DPA1_HIGH_LMAX
  TORCH_CHECK(
      basis_dim == 4 || basis_dim == 9 || basis_dim == 16 || basis_dim == 25,
      "dpa1_graph_compress: basis_dim must be 4, 9, 16, or 25");
  TORCH_CHECK(basis_dim <= 9 || (width >= 16 && width <= 128),
              "dpa1_graph_compress: basis_dim 16 and 25 require width 16, 32, "
              "64, or 128");
#else
  TORCH_CHECK(
      basis_dim == 4,
      "dpa1_graph_compress: this build instantiates only lmax=1; rebuild "
      "with DEEPMD_ENABLE_DPA1_HIGH_LMAX=ON for lmax=2/3/4");
#endif
  const int degree_gain_size =
      basis_dim == 4 ? 0 : (basis_dim == 9 ? 1 : (basis_dim == 16 ? 2 : 3));
  TORCH_CHECK(degree_gain.is_contiguous() &&
                  degree_gain.scalar_type() == torch::kFloat32 &&
                  degree_gain.numel() == degree_gain_size,
              "dpa1_graph_compress: degree_gain has an invalid shape or dtype");
  const int ntypes = static_cast<int>(type_embedding.size(0));
  const int type_embedding_dim = static_cast<int>(type_embedding.size(1));
  const int output_dim = width * static_cast<int>(axis) +
                         (concatenate_type_embedding ? type_embedding_dim : 0);
  auto options = edge_vec.options().dtype(torch::kFloat32);
  auto descriptor = torch::empty({node_count, output_dim}, options);
  auto rotation =
      torch::empty({write_rotation ? node_count : 0, width, 3}, options);
  auto moment = torch::empty({node_count, basis_dim, width}, options);
  if (node_count == 0) {
    return {descriptor, rotation, moment};
  }
  const auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  const auto stream = at::cuda::getCurrentCUDAStream();
  Arguments arguments = make_common_arguments(
      node_count, static_cast<int>(basis_dim), ntypes, type_one_side != 0,
      smooth != 0, static_cast<int>(axis), canonical, static_cast<float>(rcut),
      static_cast<float>(rcut_smooth), static_cast<float>(protection),
      static_cast<float>(1.0 / neighbors), static_cast<float>(lower),
      static_cast<float>(upper), static_cast<float>(table_max),
      static_cast<float>(stride0), static_cast<float>(stride1), edge_vec_float,
      edge_index, edge_mask, destination_order, destination_row_ptr, atype,
      average, inverse_stddev, degree_gain, table, gate_table);
  arguments.concatenate_type_embedding = concatenate_type_embedding != 0;
  arguments.write_rotation = write_rotation != 0;
  arguments.type_embedding_dim = type_embedding_dim;
  arguments.type_embedding = type_embedding.data_ptr<float>();
  arguments.descriptor = descriptor.data_ptr<float>();
  arguments.rotation = write_rotation ? rotation.data_ptr<float>() : nullptr;
  arguments.moment_out = moment.data_ptr<float>();
  check_launch("dpa1_graph_compress forward",
               dispatch_width(width, false, arguments, stream));
  return {descriptor, rotation, moment};
}

torch::Tensor dpa1_graph_compress_backward(
    torch::Tensor descriptor_gradient,
    std::optional<torch::Tensor> rotation_gradient,
    torch::Tensor moment,
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor average,
    torch::Tensor inverse_stddev,
    torch::Tensor degree_gain,
    torch::Tensor table,
    torch::Tensor gate_table,
    int64_t type_one_side,
    int64_t smooth,
    int64_t axis,
    bool canonical,
    double lower,
    double upper,
    double table_max,
    double stride0,
    double stride1,
    double rcut,
    double rcut_smooth,
    double protection,
    double neighbors) {
  const long node_count = atype.size(0);
  const int width = static_cast<int>(table.size(1) / 6);
  const int basis_dim = static_cast<int>(moment.size(1));
  validate_inputs(edge_vec, edge_index, edge_mask, destination_order,
                  destination_row_ptr, atype, average, inverse_stddev, table,
                  gate_table, width, static_cast<int>(axis));
#if DEEPMD_ENABLE_DPA1_HIGH_LMAX
  TORCH_CHECK(
      basis_dim == 4 || basis_dim == 9 || basis_dim == 16 || basis_dim == 25,
      "dpa1_graph_compress_backward: basis dimension must be 4, 9, "
      "16, or 25");
  TORCH_CHECK(
      basis_dim <= 9 || (width >= 16 && width <= 128),
      "dpa1_graph_compress_backward: basis dimensions 16 and 25 require "
      "width 16, 32, 64, or 128");
#else
  TORCH_CHECK(
      basis_dim == 4,
      "dpa1_graph_compress_backward: this build instantiates only lmax=1; "
      "rebuild with DEEPMD_ENABLE_DPA1_HIGH_LMAX=ON for lmax=2/3/4");
#endif
  if (node_count == 0) {
    return torch::zeros_like(edge_vec);
  }
  const int ntypes = type_one_side
                         ? static_cast<int>(gate_table.size(0))
                         : static_cast<int>(llround(
                               sqrt(static_cast<double>(gate_table.size(0)))));
  auto descriptor_gradient_float =
      descriptor_gradient.to(torch::kFloat32).contiguous();
  torch::Tensor rotation_gradient_float;
  const float* rotation_gradient_ptr = nullptr;
  if (rotation_gradient.has_value() && rotation_gradient->defined() &&
      rotation_gradient->numel() > 0) {
    rotation_gradient_float =
        rotation_gradient->to(torch::kFloat32).contiguous();
    rotation_gradient_ptr = rotation_gradient_float.data_ptr<float>();
  }
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  auto edge_gradient = torch::empty_like(edge_vec_float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  Arguments arguments = make_common_arguments(
      node_count, basis_dim, ntypes, type_one_side != 0, smooth != 0,
      static_cast<int>(axis), canonical, static_cast<float>(rcut),
      static_cast<float>(rcut_smooth), static_cast<float>(protection),
      static_cast<float>(1.0 / neighbors), static_cast<float>(lower),
      static_cast<float>(upper), static_cast<float>(table_max),
      static_cast<float>(stride0), static_cast<float>(stride1), edge_vec_float,
      edge_index, edge_mask, destination_order, destination_row_ptr, atype,
      average, inverse_stddev, degree_gain, table, gate_table);
  arguments.descriptor_stride =
      static_cast<int>(descriptor_gradient_float.size(1));
  arguments.descriptor_gradient = descriptor_gradient_float.data_ptr<float>();
  arguments.rotation_gradient = rotation_gradient_ptr;
  arguments.moment = moment.data_ptr<float>();
  arguments.edge_gradient = edge_gradient.data_ptr<float>();
  check_launch("dpa1_graph_compress backward",
               dispatch_width(width, true, arguments, stream));
  return edge_gradient.to(edge_vec.scalar_type());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dpa1_canonical_compress(
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor type_embedding,
    torch::Tensor average,
    torch::Tensor inverse_stddev,
    torch::Tensor degree_gain,
    torch::Tensor table,
    torch::Tensor gate_table,
    int64_t type_one_side,
    int64_t concatenate_type_embedding,
    int64_t write_rotation,
    int64_t smooth,
    int64_t axis,
    double lower,
    double upper,
    double table_max,
    double stride0,
    double stride1,
    double rcut,
    double rcut_smooth,
    double protection,
    double neighbors,
    int64_t basis_dim) {
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa1_canonical_compress: source and edge_vec storage must "
              "share the edge axis");
  TORCH_CHECK(destination_row_ptr.numel() == atype.size(0) + 1,
              "dpa1_canonical_compress: destination_row_ptr must have N + 1 "
              "entries");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa1_graph_compress(
      edge_vec, source, edge_mask, destination_order, destination_row_ptr,
      atype, type_embedding, average, inverse_stddev, degree_gain, table,
      gate_table, type_one_side, concatenate_type_embedding, write_rotation,
      smooth, axis, true, lower, upper, table_max, stride0, stride1, rcut,
      rcut_smooth, protection, neighbors, basis_dim);
}

torch::Tensor dpa1_canonical_compress_backward(
    torch::Tensor descriptor_gradient,
    std::optional<torch::Tensor> rotation_gradient,
    torch::Tensor moment,
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor average,
    torch::Tensor inverse_stddev,
    torch::Tensor degree_gain,
    torch::Tensor table,
    torch::Tensor gate_table,
    int64_t type_one_side,
    int64_t smooth,
    int64_t axis,
    double lower,
    double upper,
    double table_max,
    double stride0,
    double stride1,
    double rcut,
    double rcut_smooth,
    double protection,
    double neighbors) {
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa1_canonical_compress_backward: source and edge_vec storage "
              "must share the edge axis");
  TORCH_CHECK(destination_row_ptr.numel() == atype.size(0) + 1,
              "dpa1_canonical_compress_backward: destination_row_ptr must "
              "have N + 1 entries");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa1_graph_compress_backward(
      descriptor_gradient, rotation_gradient, moment, edge_vec, source,
      edge_mask, destination_order, destination_row_ptr, atype, average,
      inverse_stddev, degree_gain, table, gate_table, type_one_side, smooth,
      axis, true, lower, upper, table_max, stride0, stride1, rcut, rcut_smooth,
      protection, neighbors);
}

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "dpa1_graph_compress(Tensor edge_vec, Tensor edge_index, "
      "Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor atype, "
      "Tensor type_embedding, Tensor average, Tensor inverse_stddev, "
      "Tensor degree_gain, Tensor table, Tensor gate_table, int type_one_side, "
      "int concatenate_type_embedding, int write_rotation, int smooth, "
      "int axis, bool canonical, float lower, float upper, float table_max, "
      "float stride0, float stride1, "
      "float rcut, float rcut_smooth, float protection, float neighbors, "
      "int basis_dim) "
      "-> (Tensor descriptor, Tensor rotation, Tensor moment)");
  library.impl("dpa1_graph_compress", torch::kCUDA, &dpa1_graph_compress);
  library.def(
      "dpa1_graph_compress_backward(Tensor descriptor_gradient, "
      "Tensor? rotation_gradient, Tensor moment, Tensor edge_vec, "
      "Tensor edge_index, Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor atype, Tensor average, "
      "Tensor inverse_stddev, Tensor degree_gain, Tensor table, "
      "Tensor gate_table, int type_one_side, int smooth, int axis, "
      "bool canonical, float lower, float upper, float table_max, float "
      "stride0, "
      "float stride1, float rcut, float rcut_smooth, float protection, "
      "float neighbors) -> Tensor");
  library.impl("dpa1_graph_compress_backward", torch::kCUDA,
               &dpa1_graph_compress_backward);
  library.def(
      "dpa1_canonical_compress(Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor type_embedding, "
      "Tensor average, Tensor inverse_stddev, Tensor degree_gain, Tensor "
      "table, "
      "Tensor gate_table, int type_one_side, int concatenate_type_embedding, "
      "int write_rotation, int smooth, int axis, float lower, float upper, "
      "float table_max, float stride0, float stride1, float rcut, "
      "float rcut_smooth, float protection, float neighbors, int basis_dim) -> "
      "(Tensor descriptor, Tensor rotation, Tensor moment)");
  library.impl("dpa1_canonical_compress", torch::kCUDA,
               &dpa1_canonical_compress);
  library.def(
      "dpa1_canonical_compress_backward(Tensor descriptor_gradient, "
      "Tensor? rotation_gradient, Tensor moment, Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor atype, "
      "Tensor average, Tensor inverse_stddev, Tensor degree_gain, Tensor "
      "table, "
      "Tensor gate_table, int type_one_side, int smooth, int axis, "
      "float lower, float upper, float table_max, float stride0, "
      "float stride1, float rcut, float rcut_smooth, float protection, "
      "float neighbors) -> Tensor");
  library.impl("dpa1_canonical_compress_backward", torch::kCUDA,
               &dpa1_canonical_compress_backward);
}
