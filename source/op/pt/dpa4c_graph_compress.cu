// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Torch bindings of the compressed degree-wise DPA4C descriptor.
//
// This translation unit owns argument validation, the scalar-width dispatch,
// and the operator registration. The kernels themselves are instantiated in
// one translation unit per scalar width.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "dpa4c_graph_compress_launch.h"
#include "graph_ops.h"

namespace {

using deepmd_dpa4c::Arguments;
using deepmd_dpa4c::IndexKind;
using deepmd_dpa4c::Profile;

#define DPA4C_CHECK_LAUNCH(name)                                              \
  do {                                                                        \
    const cudaError_t error = cudaGetLastError();                             \
    TORCH_CHECK(error == cudaSuccess, name, ": ", cudaGetErrorString(error)); \
  } while (0)

/// Every width the host must reproduce to validate one operator invocation.
struct Dimensions {
  int moment_width;
  int output_width;
  int degree_one;
  int coupling_records;
  int spin_channels;
};

template <int Channels, int Lmax, bool HasSpin>
Dimensions dimensions_of() {
  using P = Profile<Channels, Lmax, HasSpin>;
  return {P::MomentWidth, P::OutputWidth, P::C1,
          deepmd_dpa4c::coupling_record_count(Lmax), P::Cs};
}

template <int Channels, bool HasSpin>
Dimensions dimensions_for(int lmax) {
  switch (lmax) {
    case 2:
      return dimensions_of<Channels, 2, HasSpin>();
    case 3:
      return dimensions_of<Channels, 3, HasSpin>();
    case 4:
      return dimensions_of<Channels, 4, HasSpin>();
    default:
      TORCH_CHECK(false, "dpa4c_graph_compress: unsupported lmax ", lmax);
  }
}

Dimensions profile_dimensions(int channels, int lmax, bool has_spin) {
#define DPA4C_DIMENSIONS(width)                           \
  if (channels == width) {                                \
    return has_spin ? dimensions_for<width, true>(lmax)   \
                    : dimensions_for<width, false>(lmax); \
  }
  DPA4C_FOR_EACH_CHANNEL(DPA4C_DIMENSIONS)
#undef DPA4C_DIMENSIONS
  TORCH_CHECK(false, "dpa4c_graph_compress: unsupported channels ", channels);
}

void dispatch(int channels,
              bool backward,
              const Arguments& arguments,
              cudaStream_t stream) {
#define DPA4C_DISPATCH(width)                                    \
  if (channels == width) {                                       \
    if (backward) {                                              \
      deepmd_dpa4c::launch_backward_c##width(arguments, stream); \
    } else {                                                     \
      deepmd_dpa4c::launch_forward_c##width(arguments, stream);  \
    }                                                            \
    DPA4C_CHECK_LAUNCH("dpa4c_graph_compress");                  \
    return;                                                      \
  }
  DPA4C_FOR_EACH_CHANNEL(DPA4C_DISPATCH)
#undef DPA4C_DISPATCH
  TORCH_CHECK(false, "dpa4c_graph_compress: unsupported channels ", channels);
}

/// Immutable tensors and scalars shared by both directions of the operator.
struct Payload {
  torch::Tensor edge_index;
  torch::Tensor edge_mask;
  torch::Tensor destination_order;
  torch::Tensor destination_row_ptr;
  torch::Tensor atype;
  torch::Tensor table;
  torch::Tensor pair_film;
  torch::Tensor pair_mixing;
  torch::Tensor type_embedding;
  torch::Tensor readout_matrices;
  torch::Tensor coupling_meta;
  torch::Tensor coupling_entry;
  torch::Tensor coupling_value;
  torch::Tensor output_mean;
  torch::Tensor output_inv_std;
  // Native spin inputs. Empty together when the descriptor is spin free.
  torch::Tensor spin;
  torch::Tensor spin_pair;
  torch::Tensor spin_type;
  bool canonical;
  int64_t lmax;
  double table_stride;
  double table_max;
  double rcut;
  double eps;
  double degree_floor;
  int64_t node_begin = 0;
};

IndexKind index_kind_of(const torch::Tensor& edge_index) {
  switch (edge_index.scalar_type()) {
    case torch::kInt32:
    case torch::kUInt32:
      return IndexKind::Bits32;
    case torch::kInt64:
      return IndexKind::Bits64;
    default:
      TORCH_CHECK(false,
                  "dpa4c_graph_compress: edge indices must be int32, uint32, "
                  "or int64");
  }
}

Arguments build_arguments(const Payload& payload,
                          const torch::Tensor& edge_vec,
                          int channels,
                          const Dimensions& widths) {
  // A tiled caller passes a slice of the destination row pointer, so the run
  // length follows the slice while the type table stays system-wide.
  const long node_count = payload.destination_row_ptr.numel() - 1;
  const int type_count = static_cast<int>(payload.type_embedding.size(0));
  const int radial_modes = payload.pair_mixing.numel() == 0
                               ? 0
                               : static_cast<int>(payload.pair_mixing.size(2));

  const torch::Device device = edge_vec.device();
  for (const torch::Tensor* tensor :
       {&edge_vec, &payload.edge_index, &payload.edge_mask,
        &payload.destination_order, &payload.destination_row_ptr,
        &payload.atype, &payload.table, &payload.pair_film,
        &payload.pair_mixing, &payload.type_embedding,
        &payload.readout_matrices, &payload.coupling_meta,
        &payload.coupling_entry, &payload.coupling_value, &payload.output_mean,
        &payload.output_inv_std}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == device,
                "dpa4c_graph_compress: every tensor input must be a CUDA "
                "tensor on the device of edge_vec");
    TORCH_CHECK(tensor->is_contiguous(),
                "dpa4c_graph_compress: all tensor inputs must be contiguous");
  }
  TORCH_CHECK(payload.destination_order.scalar_type() ==
                  payload.edge_index.scalar_type(),
              "dpa4c_graph_compress: destination_order dtype must match "
              "edge_index");
  TORCH_CHECK(payload.canonical ||
                  payload.destination_order.numel() == edge_vec.size(0),
              "dpa4c_graph_compress: non-canonical input requires one "
              "destination_order entry per edge");
  TORCH_CHECK(payload.edge_mask.scalar_type() == torch::kBool,
              "dpa4c_graph_compress: edge_mask must be bool");
  TORCH_CHECK(
      payload.destination_row_ptr.scalar_type() == torch::kInt64 &&
          payload.atype.scalar_type() == torch::kInt64,
      "dpa4c_graph_compress: row pointers and atom types must be int64");
  TORCH_CHECK(payload.coupling_meta.scalar_type() == torch::kInt32 &&
                  payload.coupling_entry.scalar_type() == torch::kInt32,
              "dpa4c_graph_compress: the coupling layout must be int32");
  for (const torch::Tensor* tensor :
       {&payload.table, &payload.pair_film, &payload.pair_mixing,
        &payload.type_embedding, &payload.readout_matrices,
        &payload.coupling_value, &payload.output_mean,
        &payload.output_inv_std}) {
    TORCH_CHECK(tensor->scalar_type() == torch::kFloat32,
                "dpa4c_graph_compress: tables and weights must be fp32");
  }
  TORCH_CHECK(payload.lmax >= 2 && payload.lmax <= 4,
              "dpa4c_graph_compress: lmax must be 2, 3, or 4");
  TORCH_CHECK(payload.table_max >= payload.rcut,
              "dpa4c_graph_compress: table_max must cover rcut");
  // The shared mode cache is sized from a compile-time maximum and the split
  // spline row assumes an even table width, so the rank set is closed.
  TORCH_CHECK(radial_modes == 0 || radial_modes == 2 || radial_modes == 4 ||
                  radial_modes == 8,
              "dpa4c_graph_compress: radial_modes must be 0, 2, 4, or 8, got ",
              radial_modes);
  TORCH_CHECK(edge_vec.dim() == 2 && edge_vec.size(1) == 3,
              "dpa4c_graph_compress: edge_vec must have shape (E, 3)");
  TORCH_CHECK(payload.table.dim() == 2 && payload.table.size(0) > 0 &&
                  payload.table.size(1) == 6 * (channels + radial_modes),
              "dpa4c_graph_compress: invalid radial table shape");
  TORCH_CHECK(payload.type_embedding.dim() == 2 && type_count > 1,
              "dpa4c_graph_compress: invalid type embedding shape");
  TORCH_CHECK(
      payload.pair_film.sizes() ==
          torch::IntArrayRef(
              {static_cast<long>(type_count) * type_count, channels, 2}),
      "dpa4c_graph_compress: invalid PairFiLM cache shape");
  TORCH_CHECK(
      radial_modes == 0 ||
          payload.pair_mixing.sizes() ==
              torch::IntArrayRef({static_cast<long>(type_count) * type_count,
                                  channels, radial_modes}),
      "dpa4c_graph_compress: invalid mode-mixing cache shape");
  // The kernel addresses the packed projections with a fixed degree-one
  // stride, so the padded block must have exactly that extent.
  TORCH_CHECK(payload.readout_matrices.sizes() ==
                  torch::IntArrayRef({8, widths.degree_one, widths.degree_one}),
              "dpa4c_graph_compress: invalid readout matrix shape");
  TORCH_CHECK(payload.coupling_meta.dim() == 2 &&
                  payload.coupling_meta.size(1) == 8 &&
                  payload.coupling_meta.size(0) == widths.coupling_records,
              "dpa4c_graph_compress: the coupling layout must describe ",
              widths.coupling_records, " degree triples");
  TORCH_CHECK(payload.coupling_entry.numel() == payload.coupling_value.numel(),
              "dpa4c_graph_compress: coupling coordinates and values must "
              "have equal length");
  TORCH_CHECK(payload.output_mean.numel() == widths.output_width &&
                  payload.output_inv_std.numel() == widths.output_width,
              "dpa4c_graph_compress: invalid output calibration shape");
  // A run covers ``node_count`` destination rows starting at ``node_begin``;
  // the type table is system-wide because neighbor lookups index it with
  // absolute source indices. Whole-system entry points additionally require
  // the two to describe the same node axis.
  TORCH_CHECK(payload.atype.size(0) >= payload.node_begin + node_count,
              "dpa4c_graph_compress: atype does not cover the destination "
              "node window");

  // === Native spin ===
  // The three inputs are present together or not at all. ``spin`` spans the
  // absolute node axis because neighbour lookups address it with source
  // indices, and ``spin_type`` packs the four per-type scalars a node reads.
  const bool has_spin = payload.spin.dim() == 2;
  if (has_spin) {
    for (const torch::Tensor* tensor :
         {&payload.spin, &payload.spin_pair, &payload.spin_type}) {
      TORCH_CHECK(tensor->is_cuda() && tensor->device() == device &&
                      tensor->is_contiguous() &&
                      tensor->scalar_type() == torch::kFloat32,
                  "dpa4c_graph_compress: spin inputs must be contiguous fp32 "
                  "CUDA tensors on the device of edge_vec");
    }
    TORCH_CHECK(payload.spin.dim() == 2 && payload.spin.size(1) == 3 &&
                    payload.spin.size(0) == payload.atype.size(0),
                "dpa4c_graph_compress: spin must have shape (N_all, 3)");
    TORCH_CHECK(
        payload.spin_pair.sizes() ==
            torch::IntArrayRef({static_cast<long>(type_count) * type_count,
                                widths.spin_channels, 2}),
        "dpa4c_graph_compress: invalid ordered spin cache shape");
    TORCH_CHECK(payload.spin_type.sizes() ==
                    torch::IntArrayRef({static_cast<long>(type_count), 4}),
                "dpa4c_graph_compress: invalid per-type spin table shape");
  } else {
    TORCH_CHECK(
        payload.spin.dim() == 1 && payload.spin.numel() == 0 &&
            payload.spin_pair.numel() == 0 && payload.spin_type.numel() == 0,
        "dpa4c_graph_compress: absent spin must be a rank-one empty tensor, "
        "and spin tables require a spin input");
  }

  Arguments arguments;
  arguments.has_spin = has_spin;
  arguments.spin = has_spin ? payload.spin.data_ptr<float>() : nullptr;
  arguments.spin_pair =
      has_spin ? payload.spin_pair.data_ptr<float>() : nullptr;
  arguments.spin_type =
      has_spin ? payload.spin_type.data_ptr<float>() : nullptr;
  arguments.node_count = node_count;
  arguments.edge_count = edge_vec.size(0);
  arguments.lmax = static_cast<int>(payload.lmax);
  arguments.interval_count = static_cast<int>(payload.table.size(0));
  arguments.type_count = type_count;
  arguments.table_width = channels + radial_modes;
  arguments.radial_modes = radial_modes;
  arguments.coupling_count = static_cast<int>(payload.coupling_meta.size(0));
  arguments.table_stride = static_cast<float>(payload.table_stride);
  arguments.table_max = static_cast<float>(payload.table_max);
  arguments.rcut = static_cast<float>(payload.rcut);
  arguments.eps = static_cast<float>(payload.eps);
  arguments.degree_floor = static_cast<float>(payload.degree_floor);
  arguments.canonical = payload.canonical;
  arguments.index_kind = index_kind_of(payload.edge_index);
  arguments.edge_index = payload.edge_index.data_ptr();
  arguments.destination_order = payload.destination_order.numel() != 0
                                    ? payload.destination_order.data_ptr()
                                    : nullptr;
  arguments.edge_vec = edge_vec.data_ptr<float>();
  arguments.edge_mask = payload.edge_mask.numel() != 0
                            ? payload.edge_mask.data_ptr<bool>()
                            : nullptr;
  arguments.destination_row_ptr = payload.destination_row_ptr.data_ptr<long>();
  arguments.node_begin = payload.node_begin;
  arguments.atype = payload.atype.data_ptr<long>();
  arguments.table = payload.table.data_ptr<float>();
  arguments.pair_film = payload.pair_film.data_ptr<float>();
  arguments.pair_mixing =
      radial_modes != 0 ? payload.pair_mixing.data_ptr<float>() : nullptr;
  arguments.type_embedding = payload.type_embedding.data_ptr<float>();
  arguments.readout_matrices = payload.readout_matrices.data_ptr<float>();
  arguments.coupling_meta = payload.coupling_meta.data_ptr<int>();
  arguments.coupling_entry = payload.coupling_entry.data_ptr<int>();
  arguments.coupling_value = payload.coupling_value.data_ptr<float>();
  arguments.output_mean = payload.output_mean.data_ptr<float>();
  arguments.output_inv_std = payload.output_inv_std.data_ptr<float>();
  return arguments;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> dpa4c_graph_compress(
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor pair_film,
    torch::Tensor pair_mixing,
    torch::Tensor type_embedding,
    torch::Tensor readout_matrices,
    torch::Tensor coupling_meta,
    torch::Tensor coupling_entry,
    torch::Tensor coupling_value,
    torch::Tensor output_mean,
    torch::Tensor output_inv_std,
    torch::Tensor spin,
    torch::Tensor spin_pair,
    torch::Tensor spin_type,
    bool canonical,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  const Payload payload{edge_index,
                        edge_mask,
                        destination_order,
                        destination_row_ptr,
                        atype,
                        table,
                        pair_film,
                        pair_mixing,
                        type_embedding,
                        readout_matrices,
                        coupling_meta,
                        coupling_entry,
                        coupling_value,
                        output_mean,
                        output_inv_std,
                        spin,
                        spin_pair,
                        spin_type,
                        canonical,
                        lmax,
                        table_stride,
                        table_max,
                        rcut,
                        eps,
                        degree_floor};
  const long node_count = destination_row_ptr.numel() - 1;
  // The destination row pointer defines the node axis; the type table must
  // describe exactly that axis, or the two disagree on how many nodes exist.
  TORCH_CHECK(atype.size(0) == destination_row_ptr.numel() - 1,
              "dpa4c_graph_compress: atype and destination_row_ptr describe "
              "different node counts");
  const int channels = static_cast<int>(type_embedding.size(1));
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), spin.dim() == 2);
  TORCH_CHECK(edge_vec.is_cuda(),
              "dpa4c_graph_compress: edge_vec must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  auto options = edge_vec.options().dtype(torch::kFloat32);
  auto descriptor = torch::empty({node_count, widths.output_width}, options);
  auto state = torch::empty({node_count, widths.moment_width + 2}, options);
  if (node_count == 0) {
    return {descriptor, state};
  }
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  Arguments arguments =
      build_arguments(payload, edge_vec_float, channels, widths);
  arguments.descriptor = descriptor.data_ptr<float>();
  arguments.state_out = state.data_ptr<float>();
  dispatch(channels, false, arguments, at::cuda::getCurrentCUDAStream());
  return {descriptor, state};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_graph_compress_backward_impl(torch::Tensor descriptor_gradient,
                                   torch::Tensor state,
                                   torch::Tensor edge_vec,
                                   torch::Tensor edge_index,
                                   torch::Tensor edge_mask,
                                   torch::Tensor destination_order,
                                   torch::Tensor destination_row_ptr,
                                   torch::Tensor atype,
                                   torch::Tensor table,
                                   torch::Tensor pair_film,
                                   torch::Tensor pair_mixing,
                                   torch::Tensor type_embedding,
                                   torch::Tensor readout_matrices,
                                   torch::Tensor coupling_meta,
                                   torch::Tensor coupling_entry,
                                   torch::Tensor coupling_value,
                                   torch::Tensor output_mean,
                                   torch::Tensor output_inv_std,
                                   torch::Tensor spin,
                                   torch::Tensor spin_pair,
                                   torch::Tensor spin_type,
                                   bool canonical,
                                   int64_t lmax,
                                   double table_stride,
                                   double table_max,
                                   double rcut,
                                   double eps,
                                   double degree_floor,
                                   bool reuse_state) {
  const Payload payload{edge_index,
                        edge_mask,
                        destination_order,
                        destination_row_ptr,
                        atype,
                        table,
                        pair_film,
                        pair_mixing,
                        type_embedding,
                        readout_matrices,
                        coupling_meta,
                        coupling_entry,
                        coupling_value,
                        output_mean,
                        output_inv_std,
                        spin,
                        spin_pair,
                        spin_type,
                        canonical,
                        lmax,
                        table_stride,
                        table_max,
                        rcut,
                        eps,
                        degree_floor};
  const bool has_spin = spin.dim() == 2;
  const long node_count = destination_row_ptr.numel() - 1;
  // The destination row pointer defines the node axis; the type table must
  // describe exactly that axis, or the two disagree on how many nodes exist.
  TORCH_CHECK(
      atype.size(0) == destination_row_ptr.numel() - 1,
      "dpa4c_graph_compress_backward: atype and destination_row_ptr describe "
      "different node counts");
  const int channels = static_cast<int>(type_embedding.size(1));
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), has_spin);
  TORCH_CHECK(edge_vec.is_cuda(),
              "dpa4c_graph_compress_backward: edge_vec must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  TORCH_CHECK(state.is_cuda() && state.device() == edge_vec.device() &&
                  state.is_contiguous() &&
                  state.scalar_type() == torch::kFloat32 &&
                  state.sizes() ==
                      torch::IntArrayRef({node_count, widths.moment_width + 2}),
              "dpa4c_graph_compress_backward: invalid saved state");
  TORCH_CHECK(
      descriptor_gradient.is_cuda() &&
          descriptor_gradient.device() == edge_vec.device() &&
          descriptor_gradient.numel() == node_count * widths.output_width,
      "dpa4c_graph_compress_backward: invalid descriptor gradient");
  auto float_options = edge_vec.options().dtype(torch::kFloat32);
  // Every absent output receives its own allocation. The schema declares three
  // unannotated results, so returning one empty tensor in two slots would
  // introduce an alias the schema does not describe, which is undefined under
  // functionalization for all three inputs rather than only for spin.
  const auto absent = [&float_options] {
    return torch::empty({0}, float_options);
  };
  if (node_count == 0) {
    if (has_spin) {
      return {torch::zeros_like(edge_vec),
              torch::empty({node_count, 3}, float_options),
              torch::empty(edge_vec.sizes(), float_options)};
    }
    return {torch::zeros_like(edge_vec), absent(), absent()};
  }
  auto descriptor_gradient_float =
      descriptor_gradient.to(torch::kFloat32).contiguous();
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  auto edge_gradient = torch::empty_like(edge_vec_float);
  // The moment cotangent has exactly the layout of the saved state, so an
  // inference caller that no longer needs the state can reuse its storage.
  auto moment_gradient = reuse_state ? state : torch::empty_like(state);
  // The on-site magnetic gradient closes in the node kernel; the neighbour
  // part is emitted per edge and reduced onto source nodes by the shared edge
  // assembly, which already walks the source CSR for the conservative force.
  auto spin_gradient =
      has_spin ? torch::empty({node_count, 3}, float_options) : absent();
  auto edge_spin_gradient =
      has_spin ? torch::empty_like(edge_vec_float) : absent();
  Arguments arguments =
      build_arguments(payload, edge_vec_float, channels, widths);
  arguments.descriptor_gradient = descriptor_gradient_float.data_ptr<float>();
  arguments.state = state.data_ptr<float>();
  arguments.moment_gradient = moment_gradient.data_ptr<float>();
  arguments.edge_gradient = edge_gradient.data_ptr<float>();
  if (has_spin) {
    arguments.spin_gradient = spin_gradient.data_ptr<float>();
    arguments.edge_spin_gradient = edge_spin_gradient.data_ptr<float>();
  }
  dispatch(channels, true, arguments, at::cuda::getCurrentCUDAStream());
  return {edge_gradient.to(edge_vec.scalar_type()), spin_gradient,
          edge_spin_gradient};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_graph_compress_backward(torch::Tensor descriptor_gradient,
                              torch::Tensor state,
                              torch::Tensor edge_vec,
                              torch::Tensor edge_index,
                              torch::Tensor edge_mask,
                              torch::Tensor destination_order,
                              torch::Tensor destination_row_ptr,
                              torch::Tensor atype,
                              torch::Tensor table,
                              torch::Tensor pair_film,
                              torch::Tensor pair_mixing,
                              torch::Tensor type_embedding,
                              torch::Tensor readout_matrices,
                              torch::Tensor coupling_meta,
                              torch::Tensor coupling_entry,
                              torch::Tensor coupling_value,
                              torch::Tensor output_mean,
                              torch::Tensor output_inv_std,
                              torch::Tensor spin,
                              torch::Tensor spin_pair,
                              torch::Tensor spin_type,
                              bool canonical,
                              int64_t lmax,
                              double table_stride,
                              double table_max,
                              double rcut,
                              double eps,
                              double degree_floor) {
  return dpa4c_graph_compress_backward_impl(
      descriptor_gradient, state, edge_vec, edge_index, edge_mask,
      destination_order, destination_row_ptr, atype, table, pair_film,
      pair_mixing, type_embedding, readout_matrices, coupling_meta,
      coupling_entry, coupling_value, output_mean, output_inv_std, spin,
      spin_pair, spin_type, canonical, lmax, table_stride, table_max, rcut, eps,
      degree_floor, false);
}

std::tuple<torch::Tensor, torch::Tensor> dpa4c_canonical_compress(
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor pair_film,
    torch::Tensor pair_mixing,
    torch::Tensor type_embedding,
    torch::Tensor readout_matrices,
    torch::Tensor coupling_meta,
    torch::Tensor coupling_entry,
    torch::Tensor coupling_value,
    torch::Tensor output_mean,
    torch::Tensor output_inv_std,
    torch::Tensor spin,
    torch::Tensor spin_pair,
    torch::Tensor spin_type,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress: source and edge_vec must share the "
              "edge axis");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa4c_graph_compress(
      edge_vec, source, edge_mask, destination_order, destination_row_ptr,
      atype, table, pair_film, pair_mixing, type_embedding, readout_matrices,
      coupling_meta, coupling_entry, coupling_value, output_mean,
      output_inv_std, spin, spin_pair, spin_type, true, lmax, table_stride,
      table_max, rcut, eps, degree_floor);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward_common(torch::Tensor descriptor_gradient,
                                         torch::Tensor state,
                                         torch::Tensor edge_vec,
                                         torch::Tensor source,
                                         torch::Tensor destination_row_ptr,
                                         torch::Tensor atype,
                                         torch::Tensor table,
                                         torch::Tensor pair_film,
                                         torch::Tensor pair_mixing,
                                         torch::Tensor type_embedding,
                                         torch::Tensor readout_matrices,
                                         torch::Tensor coupling_meta,
                                         torch::Tensor coupling_entry,
                                         torch::Tensor coupling_value,
                                         torch::Tensor output_mean,
                                         torch::Tensor output_inv_std,
                                         torch::Tensor spin,
                                         torch::Tensor spin_pair,
                                         torch::Tensor spin_type,
                                         int64_t lmax,
                                         double table_stride,
                                         double table_max,
                                         double rcut,
                                         double eps,
                                         double degree_floor,
                                         bool reuse_state) {
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress_backward: source and edge_vec must "
              "share the edge axis");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa4c_graph_compress_backward_impl(
      descriptor_gradient, state, edge_vec, source, edge_mask,
      destination_order, destination_row_ptr, atype, table, pair_film,
      pair_mixing, type_embedding, readout_matrices, coupling_meta,
      coupling_entry, coupling_value, output_mean, output_inv_std, spin,
      spin_pair, spin_type, true, lmax, table_stride, table_max, rcut, eps,
      degree_floor, reuse_state);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward(torch::Tensor descriptor_gradient,
                                  torch::Tensor state,
                                  torch::Tensor edge_vec,
                                  torch::Tensor source,
                                  torch::Tensor destination_row_ptr,
                                  torch::Tensor atype,
                                  torch::Tensor table,
                                  torch::Tensor pair_film,
                                  torch::Tensor pair_mixing,
                                  torch::Tensor type_embedding,
                                  torch::Tensor readout_matrices,
                                  torch::Tensor coupling_meta,
                                  torch::Tensor coupling_entry,
                                  torch::Tensor coupling_value,
                                  torch::Tensor output_mean,
                                  torch::Tensor output_inv_std,
                                  torch::Tensor spin,
                                  torch::Tensor spin_pair,
                                  torch::Tensor spin_type,
                                  int64_t lmax,
                                  double table_stride,
                                  double table_max,
                                  double rcut,
                                  double eps,
                                  double degree_floor) {
  return dpa4c_canonical_compress_backward_common(
      descriptor_gradient, state, edge_vec, source, destination_row_ptr, atype,
      table, pair_film, pair_mixing, type_embedding, readout_matrices,
      coupling_meta, coupling_entry, coupling_value, output_mean,
      output_inv_std, spin, spin_pair, spin_type, lmax, table_stride, table_max,
      rcut, eps, degree_floor, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward_inplace(torch::Tensor descriptor_gradient,
                                          torch::Tensor state,
                                          torch::Tensor edge_vec,
                                          torch::Tensor source,
                                          torch::Tensor destination_row_ptr,
                                          torch::Tensor atype,
                                          torch::Tensor table,
                                          torch::Tensor pair_film,
                                          torch::Tensor pair_mixing,
                                          torch::Tensor type_embedding,
                                          torch::Tensor readout_matrices,
                                          torch::Tensor coupling_meta,
                                          torch::Tensor coupling_entry,
                                          torch::Tensor coupling_value,
                                          torch::Tensor output_mean,
                                          torch::Tensor output_inv_std,
                                          torch::Tensor spin,
                                          torch::Tensor spin_pair,
                                          torch::Tensor spin_type,
                                          int64_t lmax,
                                          double table_stride,
                                          double table_max,
                                          double rcut,
                                          double eps,
                                          double degree_floor) {
  return dpa4c_canonical_compress_backward_common(
      descriptor_gradient, state, edge_vec, source, destination_row_ptr, atype,
      table, pair_film, pair_mixing, type_embedding, readout_matrices,
      coupling_meta, coupling_entry, coupling_value, output_mean,
      output_inv_std, spin, spin_pair, spin_type, lmax, table_stride, table_max,
      rcut, eps, degree_floor, true);
}

// Energy and edge cotangent of one compressed inference step, evaluated over
// runs of consecutive destination nodes.
//
// Destination-sorted CSR gives a run a contiguous span of the edge axis and
// assigns every edge to exactly one destination, so the runs partition the
// work rather than splitting any reduction. Folding the descriptor, the
// fitting and the descriptor backward into one operator lets a run retire its
// descriptor, its cotangent and its moment state before the next run starts,
// which leaves only the graph and the edge cotangent at system scale. Nothing
// is recomputed: the fitting seed is the ownership mask, known up front.
//
// The loop lives here rather than in Python because its trip count follows a
// dynamic node count, which export cannot trace.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_energy_gradient(torch::Tensor edge_vec,
                                         torch::Tensor source,
                                         torch::Tensor destination_row_ptr,
                                         torch::Tensor atype,
                                         torch::Tensor table,
                                         torch::Tensor pair_film,
                                         torch::Tensor pair_mixing,
                                         torch::Tensor type_embedding,
                                         torch::Tensor readout_matrices,
                                         torch::Tensor coupling_meta,
                                         torch::Tensor coupling_entry,
                                         torch::Tensor coupling_value,
                                         torch::Tensor output_mean,
                                         torch::Tensor output_inv_std,
                                         torch::Tensor spin,
                                         torch::Tensor spin_pair,
                                         torch::Tensor spin_type,
                                         int64_t lmax,
                                         double table_stride,
                                         double table_max,
                                         double rcut,
                                         double eps,
                                         double degree_floor,
                                         std::vector<torch::Tensor> ws,
                                         std::vector<torch::Tensor> bs,
                                         std::vector<int64_t> resnets,
                                         torch::Tensor w_head,
                                         torch::Tensor b_head,
                                         torch::Tensor bias_atom_e,
                                         int64_t act,
                                         torch::Tensor seed,
                                         int64_t tile) {
  TORCH_CHECK(edge_vec.is_cuda(),
              "dpa4c_canonical_compress_energy_gradient: edge_vec must be a "
              "CUDA tensor");
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress_energy_gradient: source and edge_vec "
              "must share the edge axis");
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  const long node_count = atype.size(0);
  TORCH_CHECK(destination_row_ptr.numel() == node_count + 1,
              "dpa4c_canonical_compress_energy_gradient: atype and "
              "destination_row_ptr describe different node counts");
  const int channels = static_cast<int>(type_embedding.size(1));
  const bool has_spin = spin.dim() == 2;
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), has_spin);
  auto f32 = edge_vec.options().dtype(torch::kFloat32);
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  auto energy =
      torch::empty({node_count, 1}, edge_vec.options().dtype(torch::kFloat64));
  // With no destination rows every edge slot is padding, and the loop that
  // would clear it never runs.
  auto edge_gradient = node_count == 0 ? torch::zeros_like(edge_vec_float)
                                       : torch::empty_like(edge_vec_float);
  // Every absent output receives its own allocation, so that no two of the
  // four unannotated results share storage.
  const auto absent = [&f32] { return torch::empty({0}, f32); };
  // The on-site magnetic gradient is node local, so a run writes only its own
  // rows. The neighbour part belongs to source nodes that other runs own, so
  // it is materialized over the whole edge axis and reduced once afterwards,
  // exactly like the conservative edge cotangent.
  auto spin_gradient = has_spin ? torch::empty({node_count, 3}, f32) : absent();
  auto edge_spin_gradient =
      has_spin ? (node_count == 0 ? torch::zeros_like(edge_vec_float)
                                  : torch::empty_like(edge_vec_float))
               : absent();
  if (node_count == 0) {
    return {energy, edge_gradient.to(edge_vec.scalar_type()), spin_gradient,
            edge_spin_gradient};
  }
  auto seed_c = seed.contiguous();
  TORCH_CHECK(
      seed_c.numel() == node_count && seed_c.scalar_type() == torch::kFloat64,
      "dpa4c_canonical_compress_energy_gradient: seed must be fp64 "
      "with one entry per node");

  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.n_layer > 0 && ws[0].size(0) == widths.output_width,
              "dpa4c_canonical_compress_energy_gradient: first fitting weight "
              "does not match the descriptor width");
  const long run = tile > 0
                       ? std::max<long>(1, std::min<long>(tile, node_count))
                       : node_count;
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto descriptor = torch::empty({run, widths.output_width}, f32);
  auto state = torch::empty({run, widths.moment_width + 2}, f32);
  auto saved = torch::empty({run * plan.saved_width()}, f32);
  auto scratch = torch::empty({slots, run, plan.width_max}, f32);
  float* slot[2] = {
      scratch[0].data_ptr<float>(),
      slots > 1 ? scratch[1].data_ptr<float>() : scratch[0].data_ptr<float>()};

  auto empty_index = torch::empty({0}, source.options());
  auto empty_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto stream = at::cuda::getCurrentCUDAStream();
  for (long begin = 0; begin < node_count; begin += run) {
    const long count = std::min(run, node_count - begin);
    const Payload payload{
        source,         empty_mask,
        empty_index,    destination_row_ptr.slice(0, begin, begin + count + 1),
        atype,          table,
        pair_film,      pair_mixing,
        type_embedding, readout_matrices,
        coupling_meta,  coupling_entry,
        coupling_value, output_mean,
        output_inv_std, spin,
        spin_pair,      spin_type,
        true,           lmax,
        table_stride,   table_max,
        rcut,           eps,
        degree_floor,   begin};
    Arguments arguments =
        build_arguments(payload, edge_vec_float, channels, widths);
    arguments.descriptor = descriptor.data_ptr<float>();
    arguments.state_out = state.data_ptr<float>();
    dispatch(channels, false, arguments, stream);

    fitting_forward_range(stream, plan, descriptor.data_ptr<float>(),
                          widths.output_width, atype.data_ptr<long>() + begin,
                          ws, bs, resnets, w_head, b_head, bias_atom_e, act,
                          count, saved.data_ptr<float>(), slot,
                          energy.data_ptr<double>() + begin);
    // The cotangent replaces the descriptor, which the run no longer needs.
    fitting_backward_range(
        stream, plan, seed_c.data_ptr<double>() + begin,
        saved.data_ptr<float>(), ws, bs, resnets, w_head, act, count, slot[0],
        plan.n_layer > 1 ? slot[1] : nullptr, descriptor.data_ptr<float>());

    arguments.descriptor_gradient = descriptor.data_ptr<float>();
    arguments.state = state.data_ptr<float>();
    arguments.moment_gradient = state.data_ptr<float>();
    arguments.edge_gradient = edge_gradient.data_ptr<float>();
    if (has_spin) {
      // Node-indexed like the energy, so the run addresses its own slice.
      arguments.spin_gradient = spin_gradient.data_ptr<float>() + begin * 3;
      arguments.edge_spin_gradient = edge_spin_gradient.data_ptr<float>();
    }
    // Only the final run reaches the reserved edge slots; its row pointer ends
    // at the last physical edge, which is exactly where the padding begins.
    arguments.clear_padding = begin + count == node_count;
    dispatch(channels, true, arguments, stream);
  }
  return {energy, edge_gradient.to(edge_vec.scalar_type()), spin_gradient,
          edge_spin_gradient};
}

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "dpa4c_graph_compress(Tensor edge_vec, Tensor edge_index, "
      "Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "bool canonical, int lmax, float table_stride, float table_max, "
      "float rcut, float eps, float degree_floor) "
      "-> (Tensor descriptor, Tensor state)");
  library.impl("dpa4c_graph_compress", torch::kCUDA, &dpa4c_graph_compress);
  library.def(
      "dpa4c_graph_compress_backward(Tensor descriptor_gradient, "
      "Tensor state, Tensor edge_vec, Tensor edge_index, Tensor edge_mask, "
      "Tensor destination_order, Tensor destination_row_ptr, Tensor atype, "
      "Tensor table, Tensor pair_film, Tensor pair_mixing, "
      "Tensor type_embedding, Tensor readout_matrices, Tensor coupling_meta, "
      "Tensor coupling_entry, Tensor coupling_value, Tensor output_mean, "
      "Tensor output_inv_std, Tensor spin, Tensor spin_pair, "
      "Tensor spin_type, bool canonical, int lmax, float table_stride, "
      "float table_max, float rcut, float eps, float degree_floor) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.impl("dpa4c_graph_compress_backward", torch::kCUDA,
               &dpa4c_graph_compress_backward);
  library.def(
      "dpa4c_canonical_compress(Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor) -> (Tensor descriptor, Tensor state)");
  library.impl("dpa4c_canonical_compress", torch::kCUDA,
               &dpa4c_canonical_compress);
  library.def(
      "dpa4c_canonical_compress_backward(Tensor descriptor_gradient, "
      "Tensor state, Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.impl("dpa4c_canonical_compress_backward", torch::kCUDA,
               &dpa4c_canonical_compress_backward);
  library.def(
      "dpa4c_canonical_compress_backward_inplace("
      "Tensor descriptor_gradient, Tensor(a!) state, Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.impl("dpa4c_canonical_compress_backward_inplace", torch::kCUDA,
               &dpa4c_canonical_compress_backward_inplace);
  library.def(
      "dpa4c_canonical_compress_energy_gradient(Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, Tensor[] ws, Tensor[] bs, int[] resnets, "
      "Tensor w_head, Tensor b_head, Tensor bias_atom_e, int act, "
      "Tensor seed, int tile) "
      "-> (Tensor energy, Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.impl("dpa4c_canonical_compress_energy_gradient", torch::kCUDA,
               &dpa4c_canonical_compress_energy_gradient);
}
