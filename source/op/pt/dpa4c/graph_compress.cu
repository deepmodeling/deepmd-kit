// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Torch bindings of the compressed degree-wise DPA4C descriptor.
//
// This translation unit owns argument validation, the dispatch over the scalar
// width and the angular degree, the node-type scan, the prologue, epilogue and
// per-node kernels of the fused energy-force operator, and the operator
// registration. The descriptor kernels are instantiated in the generated
// shards of shard.cu.in: one per angular degree for the widest scalar widths
// and one per width for the others.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "../border_exchange.h"
#include "../canonical_index.h"
#include "../communicator.h"
#include "../graph_ops.h"
#include "../long_range/charge_sum.h"
#include "block_launch.h"
#include "graph_compress.cuh"
#include "graph_compress_launch.h"

namespace {

using deepmd_dpa4c::Arguments;
using deepmd_dpa4c::BlockArguments;
using deepmd_dpa4c::BlockStage;
using deepmd_dpa4c::FirstStage;
using deepmd_dpa4c::IndexKind;
using deepmd_dpa4c::kBlockHeads;
using deepmd_dpa4c::kFoldedModes;
using deepmd_dpa4c::kForceRangeWord;
using deepmd_dpa4c::kForceWords;
using deepmd_dpa4c::kTypeWords;
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
  int basis;
};

template <int Channels, int Lmax, bool HasSpin>
Dimensions dimensions_of() {
  using P = Profile<Channels, Lmax, HasSpin>;
  return {P::MomentWidth, P::OutputWidth,
          P::C1,          deepmd_dpa4c::coupling_record_count(Lmax),
          P::Cs,          P::Basis};
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

/// Whether the edge scans prefetch the bond vectors of a canonical graph whose
/// edge storage holds `bytes` of them: when they exceed the L2 cache of the
/// current device, as those of a graph that fills the storage then stream from
/// DRAM.
bool prefetches_bonds(long bytes) {
  return bytes > at::cuda::getCurrentDeviceProperties()->l2CacheSize;
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

template <int Channels, int Lmax>
void launch(bool backward, const Arguments& arguments, cudaStream_t stream) {
  if (backward) {
    deepmd_dpa4c::launch_backward<Channels, Lmax>(arguments, stream);
  } else {
    deepmd_dpa4c::launch_forward<Channels, Lmax>(arguments, stream);
  }
}

// The arguments are validated before they reach this dispatch. The default is
// still checked rather than folded into the highest degree, so that a degree
// outside the compiled set fails loudly instead of running a kernel for a
// different model.
template <int Channels>
void dispatch_degree(bool backward,
                     const Arguments& arguments,
                     cudaStream_t stream) {
  switch (arguments.lmax) {
    case 2:
      launch<Channels, 2>(backward, arguments, stream);
      return;
    case 3:
      launch<Channels, 3>(backward, arguments, stream);
      return;
    case 4:
      launch<Channels, 4>(backward, arguments, stream);
      return;
    default:
      TORCH_CHECK(false, "dpa4c_graph_compress: uncompiled angular degree ",
                  arguments.lmax);
  }
}

void dispatch(int channels,
              bool backward,
              const Arguments& arguments,
              cudaStream_t stream) {
#define DPA4C_DISPATCH(width)                            \
  if (channels == width) {                               \
    dispatch_degree<width>(backward, arguments, stream); \
    DPA4C_CHECK_LAUNCH("dpa4c_graph_compress");          \
    return;                                              \
  }
  DPA4C_FOR_EACH_CHANNEL(DPA4C_DISPATCH)
#undef DPA4C_DISPATCH
  TORCH_CHECK(false, "dpa4c_graph_compress: unsupported channels ", channels);
}

template <int Channels>
void dispatch_first_stage_degree(FirstStage stage,
                                 const Arguments& arguments,
                                 cudaStream_t stream) {
  switch (arguments.lmax) {
    case 2:
      deepmd_dpa4c::launch_first_stage<Channels, 2>(stage, arguments, stream);
      return;
    case 3:
      deepmd_dpa4c::launch_first_stage<Channels, 3>(stage, arguments, stream);
      return;
    case 4:
      deepmd_dpa4c::launch_first_stage<Channels, 4>(stage, arguments, stream);
      return;
    default:
      TORCH_CHECK(false, "dpa4c_graph_compress: uncompiled angular degree ",
                  arguments.lmax);
  }
}

/// One stage of the first block of a descriptor of several blocks.
void dispatch_first_stage(int channels,
                          FirstStage stage,
                          const Arguments& arguments,
                          cudaStream_t stream) {
#define DPA4C_DISPATCH(width)                                     \
  if (channels == width) {                                        \
    dispatch_first_stage_degree<width>(stage, arguments, stream); \
    DPA4C_CHECK_LAUNCH("dpa4c first-block stage");                \
    return;                                                       \
  }
  DPA4C_FOR_EACH_CHANNEL(DPA4C_DISPATCH)
#undef DPA4C_DISPATCH
  TORCH_CHECK(false, "dpa4c_graph_compress: unsupported channels ", channels);
}

template <int Channels>
void dispatch_block_degree(BlockStage stage,
                           const BlockArguments& arguments,
                           cudaStream_t stream) {
  switch (arguments.edges.lmax) {
    case 2:
      deepmd_dpa4c::launch_block<Channels, 2>(stage, arguments, stream);
      return;
    case 3:
      deepmd_dpa4c::launch_block<Channels, 3>(stage, arguments, stream);
      return;
    case 4:
      deepmd_dpa4c::launch_block<Channels, 4>(stage, arguments, stream);
      return;
    default:
      TORCH_CHECK(false, "dpa4c_graph_compress: uncompiled angular degree ",
                  arguments.edges.lmax);
  }
}

/// One stage of a further message-passing block.
void dispatch_block(int channels,
                    BlockStage stage,
                    const BlockArguments& arguments,
                    cudaStream_t stream) {
#define DPA4C_DISPATCH(width)                               \
  if (channels == width) {                                  \
    dispatch_block_degree<width>(stage, arguments, stream); \
    DPA4C_CHECK_LAUNCH("dpa4c further-block stage");        \
    return;                                                 \
  }
  DPA4C_FOR_EACH_CHANNEL(DPA4C_DISPATCH)
#undef DPA4C_DISPATCH
  TORCH_CHECK(false, "dpa4c_graph_compress: unsupported channels ", channels);
}

/// Widths of the buffers and of the record of a further block.
struct BlockWidths {
  int source;
  int cotangent;
  int transport;
  int node;
  int record;
};

template <int Channels, int Lmax, bool HasSpin>
BlockWidths block_widths_of() {
  using B = deepmd_dpa4c::BlockProfile<Channels, Lmax, HasSpin>;
  return {B::SourceWidth, B::CotangentWidth, B::TransportWidth, B::NodeWidth,
          B::Width};
}

template <int Channels, bool HasSpin>
BlockWidths block_widths_for(int lmax) {
  switch (lmax) {
    case 2:
      return block_widths_of<Channels, 2, HasSpin>();
    case 3:
      return block_widths_of<Channels, 3, HasSpin>();
    case 4:
      return block_widths_of<Channels, 4, HasSpin>();
    default:
      TORCH_CHECK(false, "dpa4c_graph_compress: unsupported lmax ", lmax);
  }
}

BlockWidths block_widths(int channels, int lmax, bool has_spin) {
#define DPA4C_WIDTHS(width)                                 \
  if (channels == width) {                                  \
    return has_spin ? block_widths_for<width, true>(lmax)   \
                    : block_widths_for<width, false>(lmax); \
  }
  DPA4C_FOR_EACH_CHANNEL(DPA4C_WIDTHS)
#undef DPA4C_WIDTHS
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
  // Low-rank table of the single-type and slotted scans and its expansion,
  // empty together when the descriptor carries none.
  torch::Tensor basis_table;
  torch::Tensor basis_matrix;
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
  // Per-element length scale in Å with one entry per type-table row.
  torch::Tensor contact_radius;
  bool canonical;
  int64_t lmax;
  double table_stride;
  double table_max;
  double rcut;
  double eps;
  double degree_floor;
  // Bounds of the inner bridging switch as fractions of a pair's own length
  // scale; equal fractions mean no window.
  double f_inner;
  double f_outer;
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

/// Call ``body`` with a value of the integer type that source indices of
/// ``kind`` take in the kernels: ``long`` for 64-bit indices and
/// ``std::uint32_t`` for 32-bit ones.
template <class Body>
void with_index_type(IndexKind kind, const Body& body) {
  if (kind == IndexKind::Bits64) {
    body(long{});
  } else {
    body(std::uint32_t{});
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
        &payload.atype, &payload.table, &payload.basis_table,
        &payload.basis_matrix, &payload.pair_film, &payload.pair_mixing,
        &payload.type_embedding, &payload.readout_matrices,
        &payload.coupling_meta, &payload.coupling_entry,
        &payload.coupling_value, &payload.output_mean, &payload.output_inv_std,
        &payload.contact_radius}) {
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
  TORCH_CHECK(payload.atype.dim() == 1,
              "dpa4c_graph_compress: atype must be one-dimensional");
  TORCH_CHECK(payload.coupling_meta.scalar_type() == torch::kInt32 &&
                  payload.coupling_entry.scalar_type() == torch::kInt32,
              "dpa4c_graph_compress: the coupling layout must be int32");
  for (const torch::Tensor* tensor :
       {&payload.table, &payload.basis_table, &payload.basis_matrix,
        &payload.pair_film, &payload.pair_mixing, &payload.type_embedding,
        &payload.readout_matrices, &payload.coupling_value,
        &payload.output_mean, &payload.output_inv_std,
        &payload.contact_radius}) {
    TORCH_CHECK(tensor->scalar_type() == torch::kFloat32,
                "dpa4c_graph_compress: tables and weights must be fp32");
  }
  TORCH_CHECK(payload.lmax >= 2 && payload.lmax <= 4,
              "dpa4c_graph_compress: lmax must be 2, 3, or 4");
  TORCH_CHECK(std::isfinite(payload.table_stride) && payload.table_stride > 0.0,
              "dpa4c_graph_compress: table_stride must be positive and "
              "finite, got ",
              payload.table_stride);
  TORCH_CHECK(payload.f_inner >= 0.0 && payload.f_outer >= payload.f_inner,
              "dpa4c_graph_compress: the bridging switch needs 0 <= f_inner <= "
              "f_outer, got ",
              payload.f_inner, " and ", payload.f_outer);
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
  // The scans bound the bond-length windows of a row by the table reach, so
  // the reach lies within the knots of the table.
  const double table_extent =
      static_cast<double>(payload.table.size(0)) * payload.table_stride;
  TORCH_CHECK(payload.table_max >= payload.rcut &&
                  payload.table_max <= table_extent * (1.0 + 1e-12),
              "dpa4c_graph_compress: table_max must cover rcut within the ",
              payload.table.size(0), " intervals of the radial table, got ",
              payload.table_max, " for rcut ", payload.rcut, " and stride ",
              payload.table_stride);
  // A low-rank table shares the knots of the radial table and holds the
  // degree-one channels, the basis functions and the modes.
  if (payload.basis_table.numel() != 0) {
    TORCH_CHECK(widths.basis > 0, "dpa4c_graph_compress: ",
                payload.spin.dim() == 2
                    ? std::string("a spin-conditioned descriptor admits")
                    : std::to_string(channels) + " channels admit",
                " no low-rank radial table");
    TORCH_CHECK(payload.basis_table.dim() == 2 &&
                    payload.basis_table.size(0) == payload.table.size(0) &&
                    payload.basis_table.size(1) ==
                        6 * (widths.degree_one + widths.basis + radial_modes),
                "dpa4c_graph_compress: invalid low-rank radial table shape");
    TORCH_CHECK(
        payload.basis_matrix.sizes() ==
            torch::IntArrayRef({widths.basis, channels - widths.degree_one}),
        "dpa4c_graph_compress: the low-rank expansion must have "
        "shape (",
        widths.basis, ", ", channels - widths.degree_one, ")");
  } else {
    TORCH_CHECK(payload.basis_matrix.numel() == 0,
                "dpa4c_graph_compress: a low-rank expansion requires its "
                "table");
  }
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
  TORCH_CHECK(payload.contact_radius.sizes() ==
                  torch::IntArrayRef({static_cast<long>(type_count)}),
              "dpa4c_graph_compress: contact_radius must have shape "
              "(type_count,)");
  // The scans read the radial tables, the low-rank expansion, the mode mixing
  // and the per-type spin table as float4, and the PairFiLM and ordered spin
  // caches as float2.
  const auto aligned = [](const torch::Tensor& tensor, std::uintptr_t bytes) {
    return tensor.numel() == 0 ||
           reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % bytes == 0;
  };
  TORCH_CHECK(
      aligned(payload.table, 16) && aligned(payload.basis_table, 16) &&
          aligned(payload.basis_matrix, 16) &&
          aligned(payload.pair_mixing, 16) && aligned(payload.spin_type, 16) &&
          aligned(payload.pair_film, 8) && aligned(payload.spin_pair, 8),
      "dpa4c_graph_compress: the radial tables, the low-rank "
      "expansion, the mode mixing and the per-type spin table must be "
      "16-byte aligned, and the PairFiLM and ordered spin caches "
      "8-byte aligned");
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
  arguments.system_node_count = node_count;
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
  if (payload.f_outer > payload.f_inner) {
    arguments.f_inner = static_cast<float>(payload.f_inner);
    arguments.f_outer = static_cast<float>(payload.f_outer);
    arguments.inverse_f_width =
        static_cast<float>(1.0 / (payload.f_outer - payload.f_inner));
    arguments.bridged = true;
  }
  arguments.canonical = payload.canonical;
  arguments.prefetch_bonds =
      payload.canonical &&
      prefetches_bonds(edge_vec.size(0) * 3 * static_cast<long>(sizeof(float)));
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
  if (payload.basis_table.numel() != 0) {
    arguments.basis_table = payload.basis_table.data_ptr<float>();
    arguments.basis_matrix = payload.basis_matrix.data_ptr<float>();
  }
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
  arguments.contact_radius = payload.contact_radius.data_ptr<float>();
  return arguments;
}

/// Attach the analytical pair potential to one backward launch.
///
/// \param arguments Launch bundle of the run.
/// \param pair_table Series constants of every ordered type pair with shape
///                   `((T + 1)^2, 8)`, fp32.
/// \param device Device of the edge vectors, which the table must share.
/// \param pair_seed Energy cotangent of the first node of the run, fp64.
/// \param pair_energy Energy of the first node of the run, fp64; the scan
///                    adds the pair energy onto it.
void attach_pair_term(Arguments& arguments,
                      const torch::Tensor& pair_table,
                      const torch::Device& device,
                      const double* pair_seed,
                      double* pair_energy) {
  const long pairs =
      static_cast<long>(arguments.type_count) * arguments.type_count;
  TORCH_CHECK(pair_table.is_cuda() && pair_table.device() == device &&
                  pair_table.is_contiguous() &&
                  pair_table.scalar_type() == torch::kFloat32 &&
                  pair_table.sizes() == torch::IntArrayRef({pairs, 8}),
              "dpa4c_graph_compress: pair_table must be a contiguous fp32 "
              "CUDA tensor on the device of edge_vec with shape "
              "(type_count^2, 8)");
  arguments.pair_table = pair_table.data_ptr<float>();
  arguments.pair_seed = pair_seed;
  arguments.pair_energy = pair_energy;
  arguments.bridged = true;
}

/// Words that one scan of the node types of a canonical system writes,
/// allocated cleared, for the edge kernels that follow it: whether any node
/// carries a type other than that of node zero, which selects the folded scans
/// (`Arguments::type_mismatch`), the largest length-scale radius among the
/// node types, which bounds the bridging windows of every row
/// (`Arguments::largest_radius`), and the bitmap of the node types, which
/// numbers the types of a system for the slotted scans
/// (`Arguments::type_present`).
struct NodeTypeWords {
  int type_mismatch;
  float largest_radius;
  unsigned type_present[kTypeWords];
};

/// Set the bit of `type` in a bitmap of node types. A negative type, and a
/// type beyond the words of the bitmap, which a launch of more types never
/// reads, set none.
__device__ __forceinline__ void mark_type(long type,
                                          unsigned (&present)[kTypeWords]) {
#pragma unroll
  for (int word = 0; word < kTypeWords; ++word) {
    if (type >> 5 == word) {
      present[word] |= 1u << (type & 31);
    }
  }
}

/// Fold the node types that the threads of a block have scanned into `words`.
/// Every thread of the block takes part: `same` tells whether all of the
/// thread's nodes carry the type of node zero, `largest` is the largest
/// radius among them, and `present` the bitmap of their types. Radii are
/// positive, so their bit patterns order as the integers do and the integer
/// maximum is the floating-point one.
__device__ __forceinline__ void publish_node_types(
    bool same,
    float largest,
    const unsigned (&present)[kTypeWords],
    NodeTypeWords* words) {
  constexpr int kLanes = 32;
#pragma unroll
  for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
    largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
  }
  if (threadIdx.x % kLanes == 0 && largest > 0.0f) {
    atomicMax(reinterpret_cast<int*>(&words->largest_radius),
              __float_as_int(largest));
  }
#pragma unroll
  for (int word = 0; word < kTypeWords; ++word) {
    unsigned bits = present[word];
#pragma unroll
    for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
      bits |= __shfl_xor_sync(0xffffffffu, bits, offset);
    }
    if (threadIdx.x % kLanes == 0 && bits != 0) {
      atomicOr(&words->type_present[word], bits);
    }
  }
  if (!__syncthreads_and(same) && threadIdx.x == 0) {
    atomicOr(&words->type_mismatch, 1);
  }
}

// Scan the node types of a canonical system into `words`; a null
// `contact_radius` leaves the largest radius at zero.
__global__ void node_type_kernel(long count,
                                 const long* __restrict__ atype,
                                 const float* __restrict__ contact_radius,
                                 NodeTypeWords* __restrict__ words) {
  const long first = atype[0];
  bool same = true;
  float largest = 0.0f;
  unsigned present[kTypeWords] = {};
  for (long node = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
       node < count; node += static_cast<long>(blockDim.x) * gridDim.x) {
    const long type = atype[node];
    same &= type == first;
    mark_type(type, present);
    if (contact_radius != nullptr && type >= 0) {
      largest = fmaxf(largest, __ldg(contact_radius + type));
    }
  }
  publish_node_types(same, largest, present, words);
}

/// Radius table of the node-type scan: that of a launch with a bridging
/// window, whose edge scans bound the window by the largest radius, and none
/// otherwise.
const float* windowed_radii(const Arguments& arguments) {
  return arguments.f_outer > arguments.f_inner ? arguments.contact_radius
                                               : nullptr;
}

/// Bitmap of the node types of `words` for a launch that its slotted scans
/// may serve: one whose types, the padding type included, the bitmap covers,
/// of at most `kFoldedModes` radial modes, whose descriptor carries modes or
/// a low-rank table for the slots to hold. Null for every other launch, whose
/// systems the other kernels of the profile scan.
const unsigned* present_types(const Arguments& arguments,
                              const NodeTypeWords* words) {
  const bool slotted_features =
      arguments.radial_modes > 0 || arguments.basis_table != nullptr;
  return slotted_features && arguments.type_count <= 32 * kTypeWords &&
                 arguments.radial_modes <= kFoldedModes
             ? words->type_present
             : nullptr;
}

/// Scan the node types of a canonical system (see `NodeTypeWords`) and point
/// the launch at the words.
torch::Tensor scan_node_types(const torch::Tensor& atype,
                              Arguments& arguments,
                              cudaStream_t stream) {
  auto words =
      torch::zeros({static_cast<long>(sizeof(NodeTypeWords) / sizeof(int))},
                   atype.options().dtype(torch::kInt32));
  auto* node_types = reinterpret_cast<NodeTypeWords*>(words.data_ptr<int>());
  const long count = atype.numel();
  if (count > 0) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(
        std::min<long>((count + kThreads - 1) / kThreads, 1024));
    node_type_kernel<<<blocks, kThreads, 0, stream>>>(
        count, atype.data_ptr<long>(), windowed_radii(arguments), node_types);
    DPA4C_CHECK_LAUNCH("dpa4c node types");
  }
  arguments.type_mismatch = &node_types->type_mismatch;
  arguments.largest_radius = &node_types->largest_radius;
  arguments.type_present = present_types(arguments, node_types);
  return words;
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
    torch::Tensor basis_table,
    torch::Tensor basis_matrix,
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
    torch::Tensor contact_radius,
    bool canonical,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor,
    double f_inner,
    double f_outer) {
  const Payload payload{edge_index,          edge_mask,      destination_order,
                        destination_row_ptr, atype,          table,
                        basis_table,         basis_matrix,   pair_film,
                        pair_mixing,         type_embedding, readout_matrices,
                        coupling_meta,       coupling_entry, coupling_value,
                        output_mean,         output_inv_std, spin,
                        spin_pair,           spin_type,      contact_radius,
                        canonical,           lmax,           table_stride,
                        table_max,           rcut,           eps,
                        degree_floor,        f_inner,        f_outer};
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
  const auto stream = at::cuda::getCurrentCUDAStream();
  Arguments arguments =
      build_arguments(payload, edge_vec_float, channels, widths);
  arguments.descriptor = descriptor.data_ptr<float>();
  arguments.state_out = state.data_ptr<float>();
  const torch::Tensor node_types =
      canonical ? scan_node_types(atype, arguments, stream) : torch::Tensor();
  dispatch(channels, false, arguments, stream);
  return {descriptor, state};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_graph_compress_backward_impl(torch::Tensor descriptor_gradient,
                                   torch::Tensor state,
                                   torch::Tensor edge_vec,
                                   torch::Tensor edge_index,
                                   torch::Tensor edge_mask,
                                   torch::Tensor destination_order,
                                   torch::Tensor destination_row_ptr,
                                   torch::Tensor atype,
                                   torch::Tensor table,
                                   torch::Tensor basis_table,
                                   torch::Tensor basis_matrix,
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
                                   torch::Tensor contact_radius,
                                   bool canonical,
                                   int64_t lmax,
                                   double table_stride,
                                   double table_max,
                                   double rcut,
                                   double eps,
                                   double degree_floor,
                                   double f_inner,
                                   double f_outer,
                                   torch::Tensor pair_table,
                                   torch::Tensor pair_seed,
                                   bool reuse_state) {
  const Payload payload{edge_index,          edge_mask,      destination_order,
                        destination_row_ptr, atype,          table,
                        basis_table,         basis_matrix,   pair_film,
                        pair_mixing,         type_embedding, readout_matrices,
                        coupling_meta,       coupling_entry, coupling_value,
                        output_mean,         output_inv_std, spin,
                        spin_pair,           spin_type,      contact_radius,
                        canonical,           lmax,           table_stride,
                        table_max,           rcut,           eps,
                        degree_floor,        f_inner,        f_outer};
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
  // Every absent output receives its own allocation. The schema declares four
  // unannotated results, so returning one empty tensor in two slots would
  // introduce an alias the schema does not describe, which is undefined under
  // functionalization whether or not the spin branch is active.
  const auto absent = [&float_options] {
    return torch::empty({0}, float_options);
  };
  // The pair energy accumulates onto zeros here; a model without an analytical
  // pair potential passes an empty table and receives an empty energy.
  const bool has_pair = pair_table.numel() != 0;
  auto pair_energy =
      has_pair ? torch::zeros({node_count, 1},
                              edge_vec.options().dtype(torch::kFloat64))
               : torch::empty({0}, edge_vec.options().dtype(torch::kFloat64));
  if (node_count == 0) {
    if (has_spin) {
      return {torch::zeros_like(edge_vec),
              torch::empty({node_count, 3}, float_options),
              torch::empty(edge_vec.sizes(), float_options), pair_energy};
    }
    return {torch::zeros_like(edge_vec), absent(), absent(), pair_energy};
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
  auto pair_seed_c = pair_seed.contiguous();
  if (has_pair) {
    TORCH_CHECK(pair_seed_c.is_cuda() &&
                    pair_seed_c.device() == edge_vec.device() &&
                    pair_seed_c.numel() == node_count &&
                    pair_seed_c.scalar_type() == torch::kFloat64,
                "dpa4c_graph_compress_backward: pair_seed must be an fp64 "
                "CUDA tensor on the device of edge_vec with one entry per "
                "node");
    attach_pair_term(arguments, pair_table, edge_vec.device(),
                     pair_seed_c.data_ptr<double>(),
                     pair_energy.data_ptr<double>());
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const torch::Tensor node_types =
      canonical ? scan_node_types(atype, arguments, stream) : torch::Tensor();
  dispatch(channels, true, arguments, stream);
  return {edge_gradient.to(edge_vec.scalar_type()), spin_gradient,
          edge_spin_gradient, pair_energy};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_graph_compress_backward(torch::Tensor descriptor_gradient,
                              torch::Tensor state,
                              torch::Tensor edge_vec,
                              torch::Tensor edge_index,
                              torch::Tensor edge_mask,
                              torch::Tensor destination_order,
                              torch::Tensor destination_row_ptr,
                              torch::Tensor atype,
                              torch::Tensor table,
                              torch::Tensor basis_table,
                              torch::Tensor basis_matrix,
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
                              torch::Tensor contact_radius,
                              bool canonical,
                              int64_t lmax,
                              double table_stride,
                              double table_max,
                              double rcut,
                              double eps,
                              double degree_floor,
                              double f_inner,
                              double f_outer,
                              torch::Tensor pair_table,
                              torch::Tensor pair_seed) {
  return dpa4c_graph_compress_backward_impl(
      descriptor_gradient, state, edge_vec, edge_index, edge_mask,
      destination_order, destination_row_ptr, atype, table, basis_table,
      basis_matrix, pair_film, pair_mixing, type_embedding, readout_matrices,
      coupling_meta, coupling_entry, coupling_value, output_mean,
      output_inv_std, spin, spin_pair, spin_type, contact_radius, canonical,
      lmax, table_stride, table_max, rcut, eps, degree_floor, f_inner, f_outer,
      pair_table, pair_seed, false);
}

std::tuple<torch::Tensor, torch::Tensor> dpa4c_canonical_compress(
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor basis_table,
    torch::Tensor basis_matrix,
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
    torch::Tensor contact_radius,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor,
    double f_inner,
    double f_outer) {
  source = deepmd::canonical_index_view(source);
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress: source and edge_vec must share the "
              "edge axis");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa4c_graph_compress(
      edge_vec, source, edge_mask, destination_order, destination_row_ptr,
      atype, table, basis_table, basis_matrix, pair_film, pair_mixing,
      type_embedding, readout_matrices, coupling_meta, coupling_entry,
      coupling_value, output_mean, output_inv_std, spin, spin_pair, spin_type,
      contact_radius, true, lmax, table_stride, table_max, rcut, eps,
      degree_floor, f_inner, f_outer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward_common(torch::Tensor descriptor_gradient,
                                         torch::Tensor state,
                                         torch::Tensor edge_vec,
                                         torch::Tensor source,
                                         torch::Tensor destination_row_ptr,
                                         torch::Tensor atype,
                                         torch::Tensor table,
                                         torch::Tensor basis_table,
                                         torch::Tensor basis_matrix,
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
                                         torch::Tensor contact_radius,
                                         int64_t lmax,
                                         double table_stride,
                                         double table_max,
                                         double rcut,
                                         double eps,
                                         double degree_floor,
                                         double f_inner,
                                         double f_outer,
                                         bool reuse_state) {
  source = deepmd::canonical_index_view(source);
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress_backward: source and edge_vec must "
              "share the edge axis");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  // The compact descriptor operators carry no analytical pair potential; the
  // fused energy-force operator evaluates it on this ABI.
  auto no_pair = torch::empty({0}, edge_vec.options().dtype(torch::kFloat32));
  auto [edge_gradient, spin_gradient, edge_spin_gradient, pair_energy] =
      dpa4c_graph_compress_backward_impl(
          descriptor_gradient, state, edge_vec, source, edge_mask,
          destination_order, destination_row_ptr, atype, table, basis_table,
          basis_matrix, pair_film, pair_mixing, type_embedding,
          readout_matrices, coupling_meta, coupling_entry, coupling_value,
          output_mean, output_inv_std, spin, spin_pair, spin_type,
          contact_radius, true, lmax, table_stride, table_max, rcut, eps,
          degree_floor, f_inner, f_outer, no_pair, no_pair, reuse_state);
  return {edge_gradient, spin_gradient, edge_spin_gradient};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward(torch::Tensor descriptor_gradient,
                                  torch::Tensor state,
                                  torch::Tensor edge_vec,
                                  torch::Tensor source,
                                  torch::Tensor destination_row_ptr,
                                  torch::Tensor atype,
                                  torch::Tensor table,
                                  torch::Tensor basis_table,
                                  torch::Tensor basis_matrix,
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
                                  torch::Tensor contact_radius,
                                  int64_t lmax,
                                  double table_stride,
                                  double table_max,
                                  double rcut,
                                  double eps,
                                  double degree_floor,
                                  double f_inner,
                                  double f_outer) {
  return dpa4c_canonical_compress_backward_common(
      descriptor_gradient, state, edge_vec, source, destination_row_ptr, atype,
      table, basis_table, basis_matrix, pair_film, pair_mixing, type_embedding,
      readout_matrices, coupling_meta, coupling_entry, coupling_value,
      output_mean, output_inv_std, spin, spin_pair, spin_type, contact_radius,
      lmax, table_stride, table_max, rcut, eps, degree_floor, f_inner, f_outer,
      false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward_inplace(torch::Tensor descriptor_gradient,
                                          torch::Tensor state,
                                          torch::Tensor edge_vec,
                                          torch::Tensor source,
                                          torch::Tensor destination_row_ptr,
                                          torch::Tensor atype,
                                          torch::Tensor table,
                                          torch::Tensor basis_table,
                                          torch::Tensor basis_matrix,
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
                                          torch::Tensor contact_radius,
                                          int64_t lmax,
                                          double table_stride,
                                          double table_max,
                                          double rcut,
                                          double eps,
                                          double degree_floor,
                                          double f_inner,
                                          double f_outer) {
  return dpa4c_canonical_compress_backward_common(
      descriptor_gradient, state, edge_vec, source, destination_row_ptr, atype,
      table, basis_table, basis_matrix, pair_film, pair_mixing, type_embedding,
      readout_matrices, coupling_meta, coupling_entry, coupling_value,
      output_mean, output_inv_std, spin, spin_pair, spin_type, contact_radius,
      lmax, table_stride, table_max, rcut, eps, degree_floor, f_inner, f_outer,
      true);
}

namespace {

// Threads of one block of the prologue and of the per-node results of a
// batch, and the logical threads whose nodes one block of the single-frame
// epilogue reduces. The epilogue reduces its totals in double precision,
// which most devices without a data-centre FP64 rate execute at a small
// fraction of the single-precision rate, so its node slices are kept to two
// warps and spread over as many multiprocessors as the node count allows.
constexpr int kEpilogueThreads = 64;
// Blocks of the epilogue reduction. The count follows the node count alone,
// so the summation order, and with it every rounding, is the same on every
// device.
constexpr long kEpilogueBlocks = 512;

/// Words of one evaluation, allocated cleared: those of the scan of the node
/// types, which the prologue performs, and the arrivals counted by the
/// last-block reduction of the epilogue.
struct EvaluationWords {
  NodeTypeWords node_types;
  unsigned int arrivals;
};

__device__ __forceinline__ double fixed_point_value(unsigned long long word) {
  return static_cast<double>(static_cast<long long>(word)) *
         (1.0 / static_cast<double>(deepmd_dpa4c::kForceFixedScale));
}

// Per-node preparation of one evaluation: the node types with virtual atoms
// (negative types) mapped onto type zero, the ownership mask that selects the
// energy-contributing nodes -- a real atom among the leading ``n_local``
// nodes of its frame -- as the int32 output mask, as the fp64 cotangent that
// seeds the fitting backward and, unless ``charged`` is null, as the boolean
// mask of the nodes that carry the charges of a long-range module, and
// cleared fixed-point force totals. The frame of a node comes from
// ``frame_row_ptr`` when the batch holds several frames and is frame zero
// otherwise; nodes past the last frame are padding, which no frame owns.
__global__ void energy_force_prologue_kernel(
    long node_count,
    long frame_count,
    const long* __restrict__ n_local,
    const long* __restrict__ frame_row_ptr,
    const long* __restrict__ atype,
    long* __restrict__ node_type,
    double* __restrict__ seed,
    int* __restrict__ mask,
    bool* __restrict__ charged,
    unsigned long long* __restrict__ force,
    unsigned long long* __restrict__ spin_force,
    const float* __restrict__ contact_radius,
    EvaluationWords* __restrict__ words) {
  const long first = atype[0] < 0 ? 0 : atype[0];
  bool same = true;
  float largest = 0.0f;
  unsigned present[kTypeWords] = {};
  for (long node = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
       node < node_count; node += static_cast<long>(blockDim.x) * gridDim.x) {
    const long raw = atype[node];
    const long type = raw < 0 ? 0 : raw;
    node_type[node] = type;
    same &= type == first;
    mark_type(type, present);
    if (contact_radius != nullptr) {
      largest = fmaxf(largest, __ldg(contact_radius + type));
    }
    long frame = 0;
    long frame_begin = 0;
    if (frame_count > 1) {
      // Last frame whose span begins at or before the node.
      long low = 0;
      long high = frame_count - 1;
      while (low < high) {
        const long middle = (low + high + 1) / 2;
        if (frame_row_ptr[middle] <= node) {
          low = middle;
        } else {
          high = middle - 1;
        }
      }
      frame = low;
      frame_begin = frame_row_ptr[frame];
    }
    const bool owned = raw >= 0 && node - frame_begin < n_local[frame];
    seed[node] = owned ? 1.0 : 0.0;
    mask[node] = owned ? 1 : 0;
    if (charged != nullptr) {
      charged[node] = owned;
    }
#pragma unroll
    for (int word = 0; word < kForceWords; ++word) {
      force[node * kForceWords + word] = 0ull;
      if (spin_force != nullptr) {
        spin_force[node * kForceWords + word] = 0ull;
      }
    }
  }
  publish_node_types(same, largest, present, &words->node_types);
}

// Frame totals the single-frame epilogue reduces: the energy and the nine
// virial components.
constexpr int kEpilogueTotals = 10;

// Per-node results of one evaluation of a single frame, and its totals: the
// fixed-point force (and magnetic force) totals in fp64, the node energies
// masked by ownership, zeros for a per-atom virial nobody asked for, and the
// frame energy and virial. Each block reduces a fixed slice of the nodes in a
// fixed order, the last block to finish folds the block partials in a fixed
// order, and only the nodes of the frame, ``[0, n_node[0])``, contribute.
//
// The slice of a block is that of kEpilogueThreads logical threads, and every
// total of a logical thread is carried by a warp of its own: thread ``t`` of
// the warps of total ``k`` accumulates total ``k`` of the nodes of logical
// thread ``t`` in node order, each warp reduces its 32 values by the same
// butterfly, and the warp sums and the block partials combine in the same
// order whichever warp holds them. The order of every sum is therefore the
// one of a block of kEpilogueThreads threads carrying all totals, while the
// double-precision work, which most devices run at a small fraction of the
// single-precision rate, spreads over kEpilogueTotals times as many warps.
// The warps of a total also write the per-node results that pair with it.
//
// A long-range module adds its node energies, its forces at fixed charges and
// its per-node virials, ``extra_energy``, ``extra_force`` and
// ``extra_virial``, onto those of the descriptor and the fitting; all three
// are null for a model without one.
__global__ void __launch_bounds__(kEpilogueThreads* kEpilogueTotals)
    energy_force_epilogue_kernel(
        long node_count,
        const long* __restrict__ n_node,
        const unsigned long long* __restrict__ force_fixed,
        const unsigned long long* __restrict__ spin_fixed,
        const float* __restrict__ spin_gradient,
        const double* __restrict__ raw_energy,
        const double* __restrict__ seed,
        const float* __restrict__ node_virial,
        const double* __restrict__ extra_energy,
        const float* __restrict__ extra_force,
        const float* __restrict__ extra_virial,
        double* __restrict__ force,
        double* __restrict__ force_mag,
        double* __restrict__ atom_energy,
        float* __restrict__ atom_virial_zero,
        double* __restrict__ partial,
        EvaluationWords* __restrict__ words,
        double* __restrict__ energy,
        double* __restrict__ virial) {
  constexpr int kWarps = kEpilogueThreads / 32;
  const long frame_end = n_node[0];
  const int total = threadIdx.x / kEpilogueThreads;
  const int logical = threadIdx.x % kEpilogueThreads;
  const int lane = threadIdx.x % 32;
  const int warp = logical / 32;
  // Total 0 is the energy and total 1 + c the virial component c; the warps
  // of totals 1 to 3 also write force component total - 1.
  const int component = total - 1;
  double sum = 0.0;
  for (long node = blockIdx.x * static_cast<long>(kEpilogueThreads) + logical;
       node < node_count;
       node += static_cast<long>(kEpilogueThreads) * gridDim.x) {
    if (total == 0) {
      // A node whose force contributions left the range of the fixed-point
      // totals reports its energy as not a number, which marks the forces of
      // its frame.
      double value = force_fixed[node * kForceWords + kForceRangeWord] != 0
                         ? CUDART_NAN
                         : raw_energy[node] * seed[node];
      if (extra_energy != nullptr) {
        value += extra_energy[node];
      }
      atom_energy[node] = value;
      if (node < frame_end) {
        sum += value;
      }
      continue;
    }
    if (component < 3) {
      double value =
          fixed_point_value(force_fixed[node * kForceWords + component]);
      if (extra_force != nullptr) {
        value += static_cast<double>(extra_force[node * 3 + component]);
      }
      force[node * 3 + component] = value;
      if (force_mag != nullptr) {
        // The magnetic force is minus the energy gradient with respect to the
        // moment: its on-site part and the neighbour part of its sources.
        force_mag[node * 3 + component] =
            -(static_cast<double>(spin_gradient[node * 3 + component]) +
              fixed_point_value(spin_fixed[node * kForceWords + component]));
      }
    }
    if (atom_virial_zero != nullptr) {
      atom_virial_zero[node * 9 + component] = 0.0f;
    }
    if (node < frame_end) {
      sum += static_cast<double>(node_virial[node * 9 + component]);
      if (extra_virial != nullptr) {
        sum += static_cast<double>(extra_virial[node * 9 + component]);
      }
    }
  }

  // === Block totals in a fixed order ===
  __shared__ double warp_totals[kWarps][kEpilogueTotals];
  __shared__ bool last_block;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) {
    warp_totals[warp][total] = sum;
  }
  __syncthreads();
  if (threadIdx.x < kEpilogueTotals) {
    double value = 0.0;
    for (int w = 0; w < kWarps; ++w) {
      value += warp_totals[w][threadIdx.x];
    }
    partial[blockIdx.x * kEpilogueTotals + threadIdx.x] = value;
  }
  // === The last block to arrive folds the block totals ===
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    last_block = atomicAdd(&words->arrivals, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last_block) {
    return;
  }
  __threadfence();
  // Warp k of the block folds total k.
  __shared__ double totals[kEpilogueTotals];
  const int fold = threadIdx.x / 32;
  if (fold < kEpilogueTotals) {
    double value = 0.0;
    for (int block = lane; block < static_cast<int>(gridDim.x); block += 32) {
      value += __ldcg(partial + block * kEpilogueTotals + fold);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
      totals[fold] = value;
      if (fold == 0) {
        energy[0] = value;
      } else {
        virial[fold - 1] = value;
      }
    }
  }
  __syncthreads();
  // A fixed-point force word cannot hold a non-finite value. A non-finite
  // descriptor makes the energy non-finite, and a non-finite edge gradient,
  // from which every force follows, the virial of its destination, so a
  // frame whose totals are not finite reports every force of its nodes as
  // not a number. Nodes past the frame belong to none and keep their forces,
  // as in the multi-frame path.
  bool finite = true;
#pragma unroll
  for (int index = 0; index < kEpilogueTotals; ++index) {
    finite = finite && isfinite(totals[index]);
  }
  if (!finite) {
    const long frame_entries =
        3 * (frame_end < node_count ? frame_end : node_count);
    for (long index = threadIdx.x; index < frame_entries; index += blockDim.x) {
      force[index] = CUDART_NAN;
      if (force_mag != nullptr) {
        force_mag[index] = CUDART_NAN;
      }
    }
  }
}

// Per-node results of a batch of several frames; the frame totals follow
// from the segment reductions of the caller. ``extra_energy`` and
// ``extra_force`` are the contributions of a long-range module, as in
// energy_force_epilogue_kernel.
__global__ void energy_force_nodes_kernel(
    long node_count,
    const unsigned long long* __restrict__ force_fixed,
    const unsigned long long* __restrict__ spin_fixed,
    const float* __restrict__ spin_gradient,
    const double* __restrict__ raw_energy,
    const double* __restrict__ seed,
    const double* __restrict__ extra_energy,
    const float* __restrict__ extra_force,
    double* __restrict__ force,
    double* __restrict__ force_mag,
    double* __restrict__ atom_energy,
    float* __restrict__ atom_virial_zero) {
  for (long node = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
       node < node_count; node += static_cast<long>(blockDim.x) * gridDim.x) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      double value =
          fixed_point_value(force_fixed[node * kForceWords + component]);
      if (extra_force != nullptr) {
        value += static_cast<double>(extra_force[node * 3 + component]);
      }
      force[node * 3 + component] = value;
      if (force_mag != nullptr) {
        force_mag[node * 3 + component] =
            -(static_cast<double>(spin_gradient[node * 3 + component]) +
              fixed_point_value(spin_fixed[node * kForceWords + component]));
      }
    }
    double value = force_fixed[node * kForceWords + kForceRangeWord] != 0
                       ? CUDART_NAN
                       : raw_energy[node] * seed[node];
    if (extra_energy != nullptr) {
      value += extra_energy[node];
    }
    atom_energy[node] = value;
    if (atom_virial_zero != nullptr) {
#pragma unroll
      for (int component = 0; component < 9; ++component) {
        atom_virial_zero[node * 9 + component] = 0.0f;
      }
    }
  }
}

int epilogue_blocks(long node_count) {
  return static_cast<int>(std::max<long>(
      1, std::min<long>((node_count + kEpilogueThreads - 1) / kEpilogueThreads,
                        kEpilogueBlocks)));
}

// Validates the graph inputs of a fused energy-force operator named ``op``
// that its prologue reads before the runs validate the rest of the payload:
// the frame counts, the node types and, for a windowed descriptor, the
// contact radii, and the source CSR, which is either complete, requesting the
// per-atom virial, or the two-entry placeholder that declines it. ``source``
// is replaced by its typed view. Returns whether the per-atom virial is
// requested.
bool check_energy_force_inputs(const char* op,
                               const torch::Tensor& edge_vec,
                               torch::Tensor& source,
                               const torch::Tensor& destination_row_ptr,
                               const torch::Tensor& source_row_ptr,
                               const torch::Tensor& source_order,
                               const torch::Tensor& n_node,
                               const torch::Tensor& n_local,
                               const torch::Tensor& atype,
                               const torch::Tensor& type_embedding,
                               const torch::Tensor& contact_radius,
                               double f_inner,
                               double f_outer) {
  TORCH_CHECK(edge_vec.is_cuda(), op, ": edge_vec must be a CUDA tensor");
  source = deepmd::canonical_index_view(source);
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0), op,
              ": source and edge_vec must share the edge axis");
  const long node_count = atype.size(0);
  const long edge_count = edge_vec.size(0);
  TORCH_CHECK(destination_row_ptr.numel() == node_count + 1, op,
              ": atype and destination_row_ptr describe different node "
              "counts");
  const torch::Device device = edge_vec.device();
  const auto on_device = [&](const torch::Tensor& tensor) {
    return tensor.is_cuda() && tensor.device() == device &&
           tensor.is_contiguous();
  };
  TORCH_CHECK(on_device(n_node) && n_node.dim() == 1 &&
                  n_node.scalar_type() == torch::kInt64 && on_device(n_local) &&
                  n_local.dim() == 1 &&
                  n_local.scalar_type() == torch::kInt64 &&
                  n_local.numel() == n_node.numel() && n_node.numel() > 0,
              op,
              ": n_node and n_local must be contiguous int64 tensors on the "
              "device of edge_vec with one shared shape (nf,), nf >= 1");
  TORCH_CHECK(on_device(atype) && atype.dim() == 1 &&
                  atype.scalar_type() == torch::kInt64,
              op,
              ": atype must be a contiguous int64 tensor of shape (N,) on the "
              "device of edge_vec");
  TORCH_CHECK(
      f_outer <= f_inner || (on_device(contact_radius) &&
                             contact_radius.scalar_type() == torch::kFloat32 &&
                             contact_radius.dim() == 1 &&
                             contact_radius.size(0) == type_embedding.size(0)),
      op,
      ": a windowed descriptor needs contiguous fp32 contact radii of shape "
      "(type_count,) on the device of edge_vec");
  const bool atom_virial = source_row_ptr.numel() == node_count + 1 &&
                           source_order.numel() == edge_count;
  TORCH_CHECK(
      atom_virial || (source_row_ptr.numel() == 2 && source_order.numel() == 2),
      op, ": the source CSR must hold ", node_count + 1, " row pointers and ",
      edge_count,
      " positions, or two entries each to decline the per-atom virial; got ",
      source_row_ptr.numel(), " and ", source_order.numel());
  if (atom_virial) {
    TORCH_CHECK((source.scalar_type() == torch::kUInt32 ||
                 source.scalar_type() == torch::kInt64) &&
                    on_device(source_row_ptr) &&
                    source_row_ptr.scalar_type() == torch::kInt64 &&
                    on_device(source_order) &&
                    source_order.scalar_type() == torch::kInt64,
                op,
                ": the per-atom virial needs uint32 or int64 sources and a "
                "source CSR of contiguous int64 row pointers and positions on "
                "the device of edge_vec");
  }
  return atom_virial;
}

// A fixed-point force word cannot hold a non-finite value, so a frame of a
// batch whose energy or virial is not finite reports every force of its
// nodes, and every magnetic force unless ``force_mag`` is null, as not a
// number (see energy_force_epilogue_kernel). Nodes past the last frame
// belong to none and keep their forces.
void mark_broken_frames(const torch::Tensor& energy,
                        const torch::Tensor& virial,
                        const torch::Tensor& frame_row_ptr,
                        long node_count,
                        torch::Tensor& force,
                        torch::Tensor* force_mag) {
  const auto broken =
      torch::cat({torch::logical_not(torch::isfinite(energy).all(1).logical_and(
                      torch::isfinite(virial).flatten(1).all(1))),
                  torch::zeros({1}, energy.options().dtype(torch::kBool))});
  const auto frame =
      torch::searchsorted(frame_row_ptr,
                          torch::arange(node_count, frame_row_ptr.options()),
                          /*out_int32=*/false, /*right=*/true) -
      1;
  const auto node_broken = broken.index_select(0, frame).unsqueeze(1);
  force.masked_fill_(node_broken, std::numeric_limits<double>::quiet_NaN());
  if (force_mag != nullptr) {
    force_mag->masked_fill_(node_broken,
                            std::numeric_limits<double>::quiet_NaN());
  }
}

// A fitting network of the fused energy-force operators -- the energy fitting
// or the charge network of a long-range module: its layer plan, weights,
// head, per-type bias and activation.
struct FittingNetwork {
  const FittingLayerPlan& plan;
  const std::vector<torch::Tensor>& ws;
  const std::vector<torch::Tensor>& bs;
  const std::vector<int64_t>& resnets;
  const torch::Tensor& w_head;
  const torch::Tensor& b_head;
  const torch::Tensor& bias_atom_e;
  int64_t act;
};

// The fitting routes of the runs of ``run`` nodes into which a fused operator
// splits its node axis, and the buffers they need. The full runs and the tail
// run each take the route of their own size; an absent table, like an empty
// one, leaves both to the rules. A route of the fused network needs the
// transposed weights, and its kernel writes the pre-activations of a run and
// the sheet through which the CTAs of a cluster exchange their columns into
// the pre-activation buffer; the per-layer kernels need the activation
// ping-pong, sized for the largest run that takes them, and the
// pre-activations of a run between their forward and their backward. With
// ``deferred``, every run of the per-layer kernels keeps its pre-activations
// at its own place on the node axis, so that its backward may follow the
// forward of other runs and other work of the operator; the buffer then holds
// every node, as the descriptor of the long-range operator does.
class FittingRuns {
 public:
  FittingRuns(cudaStream_t stream,
              const FittingNetwork& fitting,
              long run,
              long node_count,
              std::optional<std::vector<int64_t>> table,
              const torch::TensorOptions& f32,
              bool deferred = false)
      : fitting_(fitting), run_(run) {
    const std::vector<int64_t> routes =
        std::move(table).value_or(std::vector<int64_t>{});
    const long tail = node_count % run;
    const long full_count = node_count / run;
    full_route_ = fitting_route(fitting.plan, fitting.ws, run, routes);
    tail_route_ = tail > 0
                      ? fitting_route(fitting.plan, fitting.ws, tail, routes)
                      : full_route_;
    const long per_layer_nodes =
        std::max(full_count > 0 && full_route_ == kFittingPerLayer ? run : 0L,
                 tail > 0 && tail_route_ == kFittingPerLayer ? tail : 0L);
    if ((full_count > 0 && full_route_ != kFittingPerLayer) ||
        (tail > 0 && tail_route_ != kFittingPerLayer)) {
      transposed_ =
          torch::empty({fitting_fused_transposed_floats(fitting.ws)}, f32);
      fitting_fused_transpose(stream, fitting.plan, fitting.ws,
                              transposed_.data_ptr<float>());
    }
    if (per_layer_nodes > 0) {
      const int slots = fitting.plan.n_layer > 1 ? 2 : 1;
      scratch_ =
          torch::empty({slots, per_layer_nodes, fitting.plan.width_max}, f32);
      slot_[0] = scratch_[0].data_ptr<float>();
      slot_[1] = slots > 1 ? scratch_[1].data_ptr<float>() : slot_[0];
    }
    // The kept pre-activations of the run from node ``begin`` start at
    // ``begin * saved_width``, a 16-byte boundary, since every hidden width is
    // a multiple of four (check_weights of graph_fitting.cu).
    kept_ = deferred && per_layer_nodes > 0;
    saved_ = torch::empty(
        {(kept_ ? node_count : run) * fitting.plan.saved_width()}, f32);
  }

  // The outputs ``output`` of the ``count`` nodes of the run from node
  // ``begin``, whose descriptor rows are ``x``, and the cotangent ``d_x`` of
  // those rows from the output cotangent ``seed``; ``d_x`` may take the place
  // of ``x``.
  void output_and_cotangent(cudaStream_t stream,
                            long begin,
                            long count,
                            const float* x,
                            float* d_x,
                            long width,
                            const long* node_type,
                            const double* seed,
                            double* output) const {
    forward(stream, begin, count, x, d_x, width, node_type, seed, output);
    backward(stream, begin, count, seed, d_x);
  }

  // The outputs of a run as in output_and_cotangent, and on a route of the
  // fused network its cotangent as well, which that network forms in the same
  // pass; a run of the per-layer kernels saves its pre-activations for
  // backward().
  void forward(cudaStream_t stream,
               long begin,
               long count,
               const float* x,
               float* d_x,
               long width,
               const long* node_type,
               const double* seed,
               double* output) const {
    const FittingNetwork& fitting = fitting_;
    const int64_t route = route_of(count);
    if (route != kFittingPerLayer) {
      fitting_energy_gradient_range(
          stream, fitting.plan, x, width, node_type, fitting.ws, fitting.bs,
          fitting.resnets, fitting.w_head, fitting.b_head, fitting.bias_atom_e,
          fitting.act, transposed_.data_ptr<float>(), seed, count, saved(begin),
          output, d_x, route);
      return;
    }
    fitting_forward_range(stream, fitting.plan, x, width, node_type, fitting.ws,
                          fitting.bs, fitting.resnets, fitting.w_head,
                          fitting.b_head, fitting.bias_atom_e, fitting.act,
                          count, saved(begin), slot_, output);
  }

  // The cotangent ``d_x`` of a run of the per-layer kernels from ``seed``,
  // after its forward(); nothing for a run of the fused network, whose
  // forward() has formed it. Without ``deferred`` it follows the forward of
  // its own run.
  void backward(cudaStream_t stream,
                long begin,
                long count,
                const double* seed,
                float* d_x) const {
    if (route_of(count) != kFittingPerLayer) {
      return;
    }
    const FittingNetwork& fitting = fitting_;
    fitting_backward_range(stream, fitting.plan, seed, saved(begin), fitting.ws,
                           fitting.bs, fitting.resnets, fitting.w_head,
                           fitting.act, count, slot_[0],
                           fitting.plan.n_layer > 1 ? slot_[1] : nullptr, d_x);
  }

 private:
  int64_t route_of(long count) const {
    return count == run_ ? full_route_ : tail_route_;
  }

  float* saved(long begin) const {
    return saved_.data_ptr<float>() +
           (kept_ ? begin * fitting_.plan.saved_width() : 0);
  }

  const FittingNetwork& fitting_;
  long run_;
  int64_t full_route_ = 0;
  int64_t tail_route_ = 0;
  bool kept_ = false;
  torch::Tensor transposed_;
  torch::Tensor scratch_;
  torch::Tensor saved_;
  float* slot_[2] = {nullptr, nullptr};
};

// The state of a fused energy-force evaluation that its epilogue reads.
struct EpilogueInputs {
  const torch::Tensor& n_node;
  const torch::Tensor& frame_row_ptr;
  const torch::Tensor& force_fixed;
  const torch::Tensor& raw_energy;
  const torch::Tensor& seed;
  torch::Tensor& node_virial;
  EvaluationWords* evaluation_words;
};

// The magnetic terms of a native-spin model: the fixed-point totals and the
// on-site gradient that form the magnetic force.
struct SpinTerms {
  const torch::Tensor& fixed;
  const torch::Tensor& gradient;
  torch::Tensor& force_mag;
};

// The inputs of the per-atom virial, which the source CSR attributes to the
// source nodes of the edges.
struct SourceVirialTerms {
  const torch::Tensor& edge_gradient;
  const torch::Tensor& edge_vec;
  const torch::Tensor& row_ptr;
  const torch::Tensor& order;
};

// The terms of a long-range module: node energies (N, 1) float64, forces at
// fixed charges (N, 3) and per-node virials (N, 9) float32.
struct LongRangeTerms {
  const torch::Tensor& energy;
  const torch::Tensor& force;
  const torch::Tensor& atom_virial;
};

// The node results and the frame totals of a fused energy-force evaluation
// (energy_force_epilogue_kernel for a single frame, energy_force_nodes_kernel
// and segment reductions for several), and the per-atom virial
// ``atom_virial_output``, which is zero without ``source_virial``. ``spin`` is
// null for a model without spin and ``long_range`` for a model without a
// long-range module.
void finish_energy_force(cudaStream_t stream,
                         long node_count,
                         int blocks,
                         const EpilogueInputs& step,
                         const SpinTerms* spin,
                         const SourceVirialTerms* source_virial,
                         const LongRangeTerms* long_range,
                         torch::Tensor& energy,
                         torch::Tensor& atom_energy,
                         torch::Tensor& force,
                         torch::Tensor& virial,
                         torch::Tensor& atom_virial_output) {
  const auto* force_words = reinterpret_cast<const unsigned long long*>(
      step.force_fixed.data_ptr<int64_t>());
  const auto* spin_words = spin != nullptr
                               ? reinterpret_cast<const unsigned long long*>(
                                     spin->fixed.data_ptr<int64_t>())
                               : nullptr;
  const float* spin_gradient =
      spin != nullptr ? spin->gradient.data_ptr<float>() : nullptr;
  double* force_mag =
      spin != nullptr ? spin->force_mag.data_ptr<double>() : nullptr;
  const double* extra_energy =
      long_range != nullptr ? long_range->energy.data_ptr<double>() : nullptr;
  const float* extra_force =
      long_range != nullptr ? long_range->force.data_ptr<float>() : nullptr;
  float* atom_virial_zero =
      source_virial != nullptr ? nullptr : atom_virial_output.data_ptr<float>();
  if (step.n_node.size(0) == 1) {
    auto partial = torch::empty({blocks, kEpilogueTotals},
                                energy.options().dtype(torch::kFloat64));
    energy_force_epilogue_kernel<<<blocks, kEpilogueThreads * kEpilogueTotals,
                                   0, stream>>>(
        node_count, step.n_node.data_ptr<long>(), force_words, spin_words,
        spin_gradient, step.raw_energy.data_ptr<double>(),
        step.seed.data_ptr<double>(), step.node_virial.data_ptr<float>(),
        extra_energy, extra_force,
        long_range != nullptr ? long_range->atom_virial.data_ptr<float>()
                              : nullptr,
        force.data_ptr<double>(), force_mag, atom_energy.data_ptr<double>(),
        atom_virial_zero, partial.data_ptr<double>(), step.evaluation_words,
        energy.data_ptr<double>(), virial.data_ptr<double>());
    DPA4C_CHECK_LAUNCH("dpa4c energy-force epilogue");
  } else {
    energy_force_nodes_kernel<<<blocks, kEpilogueThreads, 0, stream>>>(
        node_count, force_words, spin_words, spin_gradient,
        step.raw_energy.data_ptr<double>(), step.seed.data_ptr<double>(),
        extra_energy, extra_force, force.data_ptr<double>(), force_mag,
        atom_energy.data_ptr<double>(), atom_virial_zero);
    DPA4C_CHECK_LAUNCH("dpa4c energy-force nodes");
    energy.copy_(frame_scalar_sum(atom_energy, step.n_node));
    if (long_range != nullptr) {
      step.node_virial.add_(long_range->atom_virial);
    }
    virial.copy_(frame_virial_sum(step.node_virial, step.n_node));
    mark_broken_frames(energy, virial, step.frame_row_ptr, node_count, force,
                       spin != nullptr ? &spin->force_mag : nullptr);
  }
  if (source_virial != nullptr) {
    atom_virial_output.copy_(canonical_source_virial(
        source_virial->edge_gradient, source_virial->edge_vec,
        source_virial->row_ptr, source_virial->order, node_count));
    if (long_range != nullptr) {
      atom_virial_output.add_(
          long_range->atom_virial.reshape({node_count, 3, 3}));
    }
  }
}

// === Further message-passing blocks ===

/// The border plan of a node set whose halo rows the further blocks of a
/// descriptor read (border_exchange.h), and the rows the operator evaluates;
/// a node set without a plan evaluates every row.
struct HaloPlan {
  torch::Tensor send_nodes;
  torch::Tensor swaps;
  torch::Tensor communicator;
  long owned = 0;

  /// Refresh the halo rows of a node feature from the ranks that own them.
  void exchange(const torch::Tensor& rows) const {
    if (swaps.defined()) {
      deepmd::border_exchange(rows, send_nodes, swaps, communicator);
    }
  }

  /// Whether the swaps of the plan cross processes, whose messages the host
  /// waits for, rather than copy the rows of a single process on the device.
  bool crosses_processes() const {
    return swaps.defined() &&
           deepmd::communicator_address(communicator, "border_exchange") != 0;
  }

  /// As exchange, on the stream ``side`` once the current stream reaches
  /// ``written``, so that the kernels the current stream runs meanwhile
  /// overlap the transfers. The current stream then waits for the exchange.
  void exchange_beside(const torch::Tensor& rows,
                       at::cuda::CUDAEvent& written,
                       const c10::cuda::CUDAStream& side) const {
    written.block(side);
    {
      const c10::cuda::CUDAStreamGuard guard(side);
      exchange(rows);
    }
    at::cuda::CUDAEvent exchanged;
    exchanged.record(side);
    exchanged.block(at::cuda::getCurrentCUDAStream());
  }

  /// Add the halo rows of a node quantity onto the rows of the atoms they
  /// image, on the ranks that own them.
  void accumulate(const torch::Tensor& rows) const {
    if (swaps.defined()) {
      deepmd::border_accumulate(rows, send_nodes, swaps, communicator);
    }
  }
};

// === Source gate of a bridged descriptor of several blocks ===
//
// The envelope of an edge carries the leave-one-out gate of its source, the
// product of the switches of the other pairs of the source (Arguments::
// source_gate). A canonical graph holds both directions of every bond, so
// the row of a node holds every pair of the node, and the totals of the
// switches of an owned node follow from its own row; those of a halo row
// arrive from its owner.

constexpr int kGateThreads = 256;
constexpr int kGateWarps = kGateThreads / deepmd_dpa4c::kWarpSize;

/// The switch of the pair of an entry of the row of ``row_node`` and the
/// slope of the switch with respect to the length; the length, the unit
/// vector and the vector of the entry are returned as well.
struct PairSwitch {
  float weight;
  float slope;
  float radius;
  float vector[3];
};

template <typename index_t>
__device__ __forceinline__ PairSwitch pair_switch(const Arguments& args,
                                                  long row_node,
                                                  long position) {
  PairSwitch pair;
  const float x = args.edge_vec[position * 3 + 0];
  const float y = args.edge_vec[position * 3 + 1];
  const float z = args.edge_vec[position * 3 + 2];
  pair.vector[0] = x;
  pair.vector[1] = y;
  pair.vector[2] = z;
  // The regularized length of load_geometry, so that the switch of the
  // totals is that of the edge scans.
  const float square = x * x + y * y + z * z + args.eps * args.eps;
  pair.radius = square * rsqrtf(square);
  pair.weight = 1.0f;
  pair.slope = 0.0f;
  const long source =
      static_cast<long>(static_cast<const index_t*>(args.edge_index)[position]);
  const float contact = __ldg(args.contact_radius + args.atype[row_node]) +
                        __ldg(args.contact_radius + args.atype[source]);
  if (pair.radius < args.f_outer * contact) {
    const float t = deepmd_dpa4c::window_parameter(
        __fdividef(pair.radius, contact), args.f_inner, args.inverse_f_width);
    pair.weight = deepmd_dpa4c::bridging_switch(t);
    pair.slope = __fdividef(
        deepmd_dpa4c::bridging_switch_derivative(t, args.inverse_f_width),
        contact);
  }
  return pair;
}

/// The totals of the switches of the pairs of every owned node: the sum of
/// the logarithms of the nonzero switches and the number of zero ones. One
/// warp scans one row, and the lanes combine their partial totals in a fixed
/// order.
template <typename index_t>
__global__ void __launch_bounds__(kGateThreads)
    source_gate_kernel(const Arguments args, float2* totals) {
  const long row =
      static_cast<long>(blockIdx.x) * kGateWarps + threadIdx.x / 32;
  if (row >= args.node_count) {
    return;
  }
  const int lane = threadIdx.x % 32;
  const long node = args.node_begin + row;
  float log_sum = 0.0f;
  float frozen = 0.0f;
  for (long position = args.destination_row_ptr[row] + lane;
       position < args.destination_row_ptr[row + 1]; position += 32) {
    const float weight = pair_switch<index_t>(args, node, position).weight;
    if (weight <= 0.0f) {
      frozen += 1.0f;
    } else {
      log_sum += __logf(weight);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    log_sum += __shfl_xor_sync(0xffffffffu, log_sum, offset);
    frozen += __shfl_xor_sync(0xffffffffu, frozen, offset);
  }
  if (lane == 0) {
    totals[node] = make_float2(log_sum, frozen);
  }
}

/// The slope of the switches under a source gate. The switch ``w`` of a pair
/// of node ``j`` enters the gate of every other edge that ``j`` emits, so with
/// the gate total ``A_j``, the sum over the edges of ``j`` of the envelope
/// cotangent times the envelope, the energy changes through those gates with
/// the length of the pair by ``(dw/dr / w) (A_j - a)``, where ``a`` is the
/// term of the edge along the pair itself. That term cancels the slope of the
/// switch in the envelope of the edge, which the edge scans therefore leave
/// out (radial_chain), so this pass adds ``(dw/dr / w) A_j`` to every edge of
/// ``j`` whose switch lies strictly between zero and one. One warp scans the
/// row of one owned node ``j``, whose entries are the pairs of ``j``, each read
/// as the edge ``j`` emits: the entry gains the force ``+g``, the row ``-g``
/// and the virial ``-g (x) v``, which is attributed to ``j``; the lanes combine
/// the share of the row in a fixed order.
template <typename index_t>
__global__ void __launch_bounds__(kGateThreads)
    gate_closure_kernel(const Arguments args,
                        const float2* totals,
                        const float* gate_sums,
                        unsigned long long* force,
                        float* virial) {
  const long row =
      static_cast<long>(blockIdx.x) * kGateWarps + threadIdx.x / 32;
  if (row >= args.node_count) {
    return;
  }
  const long node = args.node_begin + row;
  if (!deepmd_dpa4c::has_window_pair(totals[node])) {
    return;
  }
  const int lane = threadIdx.x % 32;
  const float total = gate_sums[node];
  float share[12] = {};
  for (long position = args.destination_row_ptr[row] + lane;
       position < args.destination_row_ptr[row + 1]; position += 32) {
    const PairSwitch pair = pair_switch<index_t>(args, node, position);
    if (pair.weight <= 0.0f || pair.weight >= 1.0f) {
      continue;
    }
    // The stored vector points from the entry to the row; the edge the row
    // emits points the other way.
    const float scale =
        __fdividef(pair.slope, pair.weight) * total / pair.radius;
    const long entry = static_cast<long>(
        static_cast<const index_t*>(args.edge_index)[position]);
    unsigned long long* const words = force + entry * deepmd_dpa4c::kForceWords;
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      const float gradient = -scale * pair.vector[component];
      deepmd_dpa4c::check_force_range(gradient, words);
      atomicAdd(words + component, deepmd_dpa4c::fixed_point(gradient));
      share[component] -= gradient;
    }
#pragma unroll
    for (int component = 0; component < 9; ++component) {
      share[3 + component] -=
          scale * pair.vector[component / 3] * pair.vector[component % 3];
    }
  }
#pragma unroll
  for (int component = 0; component < 12; ++component) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      share[component] +=
          __shfl_xor_sync(0xffffffffu, share[component], offset);
    }
  }
  if (lane == 0) {
    unsigned long long* const words = force + node * deepmd_dpa4c::kForceWords;
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      deepmd_dpa4c::check_force_range(share[component], words);
      atomicAdd(words + component, deepmd_dpa4c::fixed_point(share[component]));
    }
#pragma unroll
    for (int component = 0; component < 9; ++component) {
      virial[node * 9 + component] += share[3 + component];
    }
  }
}

int gate_blocks(const Arguments& args) {
  return static_cast<int>((args.node_count + kGateWarps - 1) / kGateWarps);
}

void check_gate_launch(const char* what) {
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "dpa4c_canonical_compress_energy_force: ", what,
              " failed: ", cudaGetErrorString(status));
}

/// The switch totals of the owned rows of ``args`` into ``totals``.
void launch_source_gate(const Arguments& args,
                        float2* totals,
                        cudaStream_t stream) {
  if (args.node_count == 0) {
    return;
  }
  with_index_type(args.index_kind, [&](auto index) {
    source_gate_kernel<decltype(index)>
        <<<gate_blocks(args), kGateThreads, 0, stream>>>(args, totals);
  });
  check_gate_launch("the switch totals of the source gate");
}

/// The slope of the switches of the owned rows of ``args`` under a source
/// gate, onto the fixed-point forces and the virial of every node.
void launch_gate_closure(const Arguments& args,
                         const float2* totals,
                         const float* gate_sums,
                         unsigned long long* force,
                         float* virial,
                         cudaStream_t stream) {
  if (args.node_count == 0) {
    return;
  }
  with_index_type(args.index_kind, [&](auto index) {
    gate_closure_kernel<decltype(index)>
        <<<gate_blocks(args), kGateThreads, 0, stream>>>(
            args, totals, gate_sums, force, virial);
  });
  check_gate_launch("the switch slope of the source gate");
}

/// The validated border plan of an energy-force evaluation. The send list,
/// the swaps and the communicator come together, for a descriptor of further
/// blocks; a plan with halo rows serves a single frame.
HaloPlan halo_plan(const char* op,
                   const std::optional<torch::Tensor>& send_nodes,
                   const std::optional<torch::Tensor>& swaps,
                   const std::optional<torch::Tensor>& communicator,
                   long node_count,
                   long frame_count,
                   long block_count) {
  const int given = static_cast<int>(send_nodes.has_value()) +
                    static_cast<int>(swaps.has_value()) +
                    static_cast<int>(communicator.has_value());
  if (given == 0) {
    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), node_count};
  }
  TORCH_CHECK(given == 3, op,
              ": the send list, the swaps and the communicator of a border "
              "plan are given together");
  TORCH_CHECK(block_count > 0, op,
              ": a border plan refreshes the halo rows that further "
              "message-passing blocks read, but the descriptor has one block");
  const long owned = deepmd::check_border_plan(*send_nodes, *swaps,
                                               *communicator, node_count, op);
  TORCH_CHECK(owned == node_count || frame_count == 1, op,
              ": a border plan with halo rows serves a node set of one "
              "frame, got ",
              frame_count, " frames");
  return {*send_nodes, *swaps, *communicator, owned};
}

// === Edges of a further block ===
//
// A further block reads the edges of the graph inside its own cutoff, which
// may lie below that of the graph: every message of an edge beyond it carries
// the envelope of the block, which vanishes there with its slope. A block of
// a smaller cutoff therefore scans a graph of its own, which holds the edges
// of every row inside the cutoff in the order of the row, so that its scans
// visit no edge whose messages vanish.

constexpr int kSelectThreads = 256;
constexpr int kSelectWarps = kSelectThreads / deepmd_dpa4c::kWarpSize;

/// A graph of compact rows: the row pointers, (rows + 1), the source of every
/// edge and its vector, (storage) and (storage, 3), whose storage may exceed
/// the edges of the rows.
struct EdgeGraph {
  torch::Tensor row_ptr;
  torch::Tensor source;
  torch::Tensor edge_vec;
};

/// Whether an edge lies inside the cutoff whose square, widened by a relative
/// 1e-5, is ``bound``. The regularized length of block_geometry is at least
/// the length of the vector, so every edge that the envelope of the cutoff
/// reads inside it is kept, and the edges kept beyond it carry an envelope of
/// zero.
__device__ __forceinline__ bool inside_cutoff(const float* edge_vec,
                                              long position,
                                              float bound) {
  const float x = edge_vec[position * 3 + 0];
  const float y = edge_vec[position * 3 + 1];
  const float z = edge_vec[position * 3 + 2];
  return x * x + y * y + z * z < bound;
}

/// The number of edges of every row inside the cutoff, into
/// ``counts[row + 1]``. One warp counts one row.
__global__ void __launch_bounds__(kSelectThreads)
    count_block_edges_kernel(const long* row_ptr,
                             const float* edge_vec,
                             long rows,
                             float bound,
                             long* counts) {
  const long row =
      static_cast<long>(blockIdx.x) * kSelectWarps + threadIdx.x / 32;
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x % 32;
  int count = 0;
  for (long position = row_ptr[row] + lane; position < row_ptr[row + 1];
       position += 32) {
    count += inside_cutoff(edge_vec, position, bound) ? 1 : 0;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    count += __shfl_xor_sync(0xffffffffu, count, offset);
  }
  if (lane == 0) {
    counts[row + 1] = count;
  }
}

/// The edges of every row inside the cutoff, in the order of the row, from
/// ``block_row_ptr[row]`` on. One warp moves one row, 32 edges at a time, each
/// kept edge to its rank among the kept edges before it.
template <typename index_t>
__global__ void __launch_bounds__(kSelectThreads)
    gather_block_edges_kernel(const long* row_ptr,
                              const index_t* source,
                              const float* edge_vec,
                              long rows,
                              float bound,
                              const long* block_row_ptr,
                              index_t* block_source,
                              float* block_vec) {
  const long row =
      static_cast<long>(blockIdx.x) * kSelectWarps + threadIdx.x / 32;
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x % 32;
  const unsigned before = (1u << lane) - 1u;
  const long end = row_ptr[row + 1];
  long target = block_row_ptr[row];
  for (long first = row_ptr[row]; first < end; first += 32) {
    const long position = first + lane;
    const bool keep =
        position < end && inside_cutoff(edge_vec, position, bound);
    const unsigned kept = __ballot_sync(0xffffffffu, keep);
    if (keep) {
      const long slot = target + __popc(kept & before);
      block_source[slot] = source[position];
      block_vec[slot * 3 + 0] = edge_vec[position * 3 + 0];
      block_vec[slot * 3 + 1] = edge_vec[position * 3 + 1];
      block_vec[slot * 3 + 2] = edge_vec[position * 3 + 2];
    }
    target += __popc(kept);
  }
}

/// The edges of ``graph`` inside ``cutoff`` in Å, in the storage of ``graph``.
EdgeGraph select_block_edges(const EdgeGraph& graph,
                             double cutoff,
                             cudaStream_t stream) {
  const long rows = graph.row_ptr.numel() - 1;
  const float bound = static_cast<float>(cutoff * cutoff * (1.0 + 1e-5));
  auto counts = torch::zeros({rows + 1}, graph.row_ptr.options());
  EdgeGraph selected{torch::Tensor(), torch::empty_like(graph.source),
                     torch::empty_like(graph.edge_vec)};
  if (rows == 0) {
    selected.row_ptr = counts;
    return selected;
  }
  const int blocks = static_cast<int>((rows + kSelectWarps - 1) / kSelectWarps);
  count_block_edges_kernel<<<blocks, kSelectThreads, 0, stream>>>(
      graph.row_ptr.data_ptr<long>(), graph.edge_vec.data_ptr<float>(), rows,
      bound, counts.data_ptr<long>());
  DPA4C_CHECK_LAUNCH("dpa4c count of the edges of a further block");
  selected.row_ptr = counts.cumsum(0);
  with_index_type(index_kind_of(graph.source), [&](auto index) {
    using index_t = decltype(index);
    gather_block_edges_kernel<index_t><<<blocks, kSelectThreads, 0, stream>>>(
        graph.row_ptr.data_ptr<long>(),
        static_cast<const index_t*>(graph.source.data_ptr()),
        graph.edge_vec.data_ptr<float>(), rows, bound,
        selected.row_ptr.data_ptr<long>(),
        static_cast<index_t*>(selected.source.data_ptr()),
        selected.edge_vec.data_ptr<float>());
  });
  DPA4C_CHECK_LAUNCH("dpa4c gather of the edges of a further block");
  return selected;
}

// === Rows that read halo rows ===
//
// On a node set whose halo rows arrive from other processes, an edge scan
// first visits the rows whose entries are all owned rows, which read nothing
// of the halo, while the halo rows are exchanged, and the other rows once the
// exchange completes. A canonical graph holds both directions of every bond,
// so the same rows serve the forward, which reads the source features of the
// entries, and the backward, which reads their destination cotangents.

/// The rows of a run of ``count`` rows of a graph, ascending: those whose
/// entries are all owned (inner), then the others (border), each list in a
/// row of ``rows`` (2, count), and their numbers in ``counts`` (2), both on
/// the device.
struct RowSplit {
  torch::Tensor rows;
  torch::Tensor counts;
};

/// Whether every row of the run reads only owned rows, into ``inner``: one
/// warp checks the entries of one row.
template <typename index_t>
__global__ void __launch_bounds__(kSelectThreads)
    inner_rows_kernel(const long* row_ptr,
                      const index_t* source,
                      long rows,
                      long owned,
                      int* inner) {
  const long row =
      static_cast<long>(blockIdx.x) * kSelectWarps + threadIdx.x / 32;
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x % 32;
  bool reads_halo = false;
  for (long position = row_ptr[row] + lane; position < row_ptr[row + 1];
       position += 32) {
    reads_halo |= static_cast<long>(source[position]) >= owned;
  }
  const bool any = __any_sync(0xffffffffu, reads_halo);
  if (lane == 0) {
    inner[row] = any ? 0 : 1;
  }
}

/// Every row of the run into its list, at the rank the inclusive counts
/// ``inner_rank`` give it, and the numbers of both lists.
__global__ void split_rows_kernel(const int* inner,
                                  const int* inner_rank,
                                  long rows,
                                  int* lists,
                                  int* counts) {
  const long row = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int before = inner_rank[row] - inner[row];
  if (inner[row] != 0) {
    lists[before] = static_cast<int>(row);
  } else {
    lists[rows + row - before] = static_cast<int>(row);
  }
  if (row == rows - 1) {
    counts[0] = inner_rank[row];
    counts[1] = static_cast<int>(rows) - inner_rank[row];
  }
}

/// The inner and border rows of the run of ``count`` rows from ``begin`` of
/// ``graph`` on a node set of ``owned`` owned rows.
RowSplit split_rows(const EdgeGraph& graph,
                    long begin,
                    long count,
                    long owned,
                    cudaStream_t stream) {
  const auto i32 = graph.row_ptr.options().dtype(torch::kInt32);
  auto inner = torch::empty({count}, i32);
  RowSplit split{torch::empty({2, count}, i32), torch::empty({2}, i32)};
  const int blocks =
      static_cast<int>((count + kSelectWarps - 1) / kSelectWarps);
  const long* row_ptr = graph.row_ptr.data_ptr<long>() + begin;
  with_index_type(index_kind_of(graph.source), [&](auto index) {
    using index_t = decltype(index);
    inner_rows_kernel<index_t><<<blocks, kSelectThreads, 0, stream>>>(
        row_ptr, static_cast<const index_t*>(graph.source.data_ptr()), count,
        owned, inner.data_ptr<int>());
  });
  DPA4C_CHECK_LAUNCH("dpa4c inner rows of a further block");
  const auto rank = inner.cumsum(0, torch::kInt32);
  split_rows_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
      inner.data_ptr<int>(), rank.data_ptr<int>(), count,
      split.rows.data_ptr<int>(), split.counts.data_ptr<int>());
  DPA4C_CHECK_LAUNCH("dpa4c row split of a further block");
  return split;
}

/// What the further blocks of a descriptor share with the run loop of
/// dpa4c_canonical_compress_energy_force.
struct BlockPass {
  const torch::Tensor& records;
  // Cutoff in Å of the first block, that of the graph, and of every further
  // block, one per record and none above the one before.
  double rcut;
  const std::vector<double>& block_rcut;
  // The graph of the first block, its row pointers over the whole node set.
  const EdgeGraph& graph;
  const HaloPlan& halo;
  int channels;
  int lmax;
  bool has_spin;
  // Whether the descriptor has a bridging window, whose switches gate the
  // edges of every source (Arguments::source_gate).
  bool bridged;
  const Dimensions& widths;
  long node_count;
  long run;
  const FittingRuns& runs;
  // Run buffers of the operator.
  const torch::Tensor& descriptor;
  const torch::Tensor& state;
  // Node buffers of the operator.
  const torch::Tensor& node_type;
  const torch::Tensor& seed;
  const torch::Tensor& raw_energy;
  const torch::Tensor& spin_gradient;
  unsigned long long* force;
  cudaStream_t stream;
};

/// The descriptor of several blocks (block_launch.h) with its fitting and its
/// force, over the owned rows of the node set in runs of ``pass.run`` rows.
///
/// The forward evaluates the first block of every run, keeping its moments
/// for the whole node set, and the node maps of the second block. Block after
/// block, it then refreshes the source features of the halo rows and
/// evaluates the edge scan and the readout of every run, followed by the node
/// maps of the next block or, for the last block, by the fitting with its
/// cotangent, the readout VJP and the destination cotangents of the block.
/// The backward visits the blocks in reverse: it refreshes the destination
/// cotangents of the halo rows, scans the source rows of every run, and pulls
/// the source cotangent back through the node maps and the readout of the
/// block before, into the destination cotangents of that block or, for the
/// first block, into the moment gradient from which its edge backward
/// assembles the force. The edge scans of a further block whose cutoff lies
/// below that of the block before read the edges inside it alone, selected
/// once from the edges of the block before.
///
/// A bridged descriptor gates the edges of every source by the switches of
/// its other pairs: the switch totals of every node precede the first block,
/// every edge backward adds the envelope cotangents times the envelopes of the
/// edges of a gated source to its gate total, and once the totals of the halo
/// rows have reached their owners, a last pass adds the slope of the
/// switches through the gates.
///
/// ``first_arguments(begin, count)`` returns the launch of the first block
/// over a run, and ``attach_force(arguments, begin, count)`` adds the force
/// assembly to it. Returns the virial of the edges of the further blocks and
/// of the gates, attributed to the node that emits each, (node_count, 9).
template <class FirstArguments, class AttachForce>
torch::Tensor evaluate_further_blocks(const BlockPass& pass,
                                      const FirstArguments& launch_arguments,
                                      const AttachForce& attach_force) {
  const BlockWidths sizes =
      block_widths(pass.channels, pass.lmax, pass.has_spin);
  const torch::Tensor& records = pass.records;
  TORCH_CHECK(records.device() == pass.descriptor.device() &&
                  records.scalar_type() == torch::kFloat32 &&
                  records.dim() == 2 && records.size(1) == sizes.record &&
                  records.is_contiguous(),
              "dpa4c_canonical_compress_energy_force: the records of the "
              "further blocks must be a contiguous float32 matrix of ",
              sizes.record, " columns on the device of the graph, got ",
              records.sizes());
  const long block_count = records.size(0);
  const long node_count = pass.node_count;
  const long owned = pass.halo.owned;
  const long run = pass.run;
  const long state_width = pass.widths.moment_width + 2;
  const int channels = pass.channels;
  const cudaStream_t stream = pass.stream;
  const auto f32 = pass.descriptor.options();
  const auto i64 = f32.dtype(torch::kInt64);
  float* const descriptor = pass.descriptor.data_ptr<float>();
  float* const state = pass.state.data_ptr<float>();

  // === Source gate ===
  // The switch totals of the owned rows, and of the halo rows from their
  // owners, which every launch of the first block and of the further blocks
  // reads.
  torch::Tensor gate_totals;
  if (pass.bridged) {
    gate_totals = torch::zeros({node_count, 2}, f32);
    launch_source_gate(launch_arguments(0, owned),
                       reinterpret_cast<float2*>(gate_totals.data_ptr<float>()),
                       stream);
    pass.halo.exchange(gate_totals);
  }
  const float2* const source_gate =
      pass.bridged
          ? reinterpret_cast<const float2*>(gate_totals.data_ptr<float>())
          : nullptr;
  const auto first_arguments = [&](long begin, long count) {
    Arguments arguments = launch_arguments(begin, count);
    arguments.source_gate = source_gate;
    return arguments;
  };

  // === Edges of every further block ===
  std::vector<EdgeGraph> graphs;
  graphs.reserve(block_count);
  for (long b = 0; b < block_count; ++b) {
    const double previous = b == 0 ? pass.rcut : pass.block_rcut[b - 1];
    const EdgeGraph& before = b == 0 ? pass.graph : graphs.back();
    graphs.push_back(
        pass.block_rcut[b] < previous
            ? select_block_edges(before, pass.block_rcut[b], stream)
            : before);
  }
  // The launch of the edge scans of further block ``b`` over a run: that of
  // the first block on the edges of the block.
  const auto block_edges = [&](long b, long begin, long count) {
    Arguments arguments = first_arguments(begin, count);
    const EdgeGraph& graph = graphs[b];
    arguments.destination_row_ptr = graph.row_ptr.data_ptr<long>() + begin;
    arguments.edge_index = graph.source.data_ptr();
    arguments.edge_vec = graph.edge_vec.data_ptr<float>();
    return arguments;
  };
  // A node set whose halo rows other processes deliver splits the rows of
  // its first run, on the edges of every block, into the rows that read no
  // halo row, which the scans of the run visit while the halo rows of the
  // block are exchanged, and the others, which they visit afterwards.
  const bool overlap = pass.halo.crosses_processes() && owned > 0;
  std::vector<RowSplit> splits;
  // One side stream of high priority carries every exchange of the call.
  const c10::cuda::CUDAStream side =
      overlap ? at::cuda::getStreamFromPool(/*isHighPriority=*/true,
                                            pass.descriptor.get_device())
              : at::cuda::getCurrentCUDAStream();
  if (overlap) {
    const long count = std::min(run, owned);
    splits.reserve(block_count);
    for (long b = 0; b < block_count; ++b) {
      splits.push_back(b > 0 && graphs[b].row_ptr.is_same(graphs[b - 1].row_ptr)
                           ? splits.back()
                           : split_rows(graphs[b], 0, count, owned, stream));
    }
  }
  // The edge scan ``stage`` of block ``b`` over a run, and the exchange of
  // the halo rows of ``exchanged`` that the scans of the block read, before
  // the scan of the first run: beside the inner rows of that run when the
  // rows cross processes.
  const auto scan_run = [&](BlockStage stage, const BlockArguments& block,
                            long b, long begin,
                            const torch::Tensor& exchanged) {
    if (begin != 0) {
      dispatch_block(channels, stage, block, stream);
      return;
    }
    if (!overlap) {
      pass.halo.exchange(exchanged);
      dispatch_block(channels, stage, block, stream);
      return;
    }
    at::cuda::CUDAEvent written;
    written.record(at::cuda::getCurrentCUDAStream());
    for (int part = 0; part < 2; ++part) {
      BlockArguments rows = block;
      rows.scan_rows = splits[b].rows[part].data_ptr<int>();
      rows.scan_row_count = splits[b].counts.data_ptr<int>() + part;
      dispatch_block(channels, stage, rows, stream);
      if (part == 0) {
        pass.halo.exchange_beside(exchanged, written, side);
      }
    }
  };
  // A node set without owned rows has no runs, and still takes part in the
  // exchange of every block, whose swaps may forward rows of other ranks.
  const auto exchange_without_runs = [&](const torch::Tensor& exchanged) {
    if (owned == 0) {
      pass.halo.exchange(exchanged);
    }
  };

  // === Node buffers ===
  // The moments of the first block and of every block before the last, and
  // the transport of every such block, which the backward reads again.
  auto first_state = torch::empty({node_count, state_width}, f32);
  auto states = torch::empty({block_count - 1, node_count, state_width}, f32);
  auto transports =
      torch::empty({block_count - 1, node_count, sizes.transport}, f32);
  auto sources = torch::empty({block_count, node_count, sizes.source}, f32);
  auto centers = torch::empty({block_count, node_count, kBlockHeads}, f32);
  auto divisors = torch::empty({node_count, 2}, f32);
  // The edges of a block read the destination cotangents of every node while
  // the block before forms its own, so two buffers alternate.
  auto cotangents = torch::empty({2, node_count, sizes.cotangent}, f32);
  auto center_gradient = torch::empty({node_count, kBlockHeads}, f32);
  auto carried = torch::empty({node_count, state_width}, f32);
  auto block_virial = torch::zeros({node_count, 9}, f32);
  // The gate totals of every node: those of the first block in fixed point,
  // those of the further blocks, whose backward scans the rows of the sources,
  // as row sums.
  torch::Tensor gate_fixed;
  torch::Tensor gate_rows;
  if (pass.bridged) {
    gate_fixed = torch::zeros({node_count}, i64);
    gate_rows = torch::zeros({node_count}, f32);
  }
  // === Run buffers ===
  auto transport = torch::empty({run, sizes.transport}, f32);
  auto source_gradient = torch::empty({run, sizes.source}, f32);
  auto keys_gradient = torch::empty({run, state_width}, f32);
  auto map_cotangent = torch::empty({run, sizes.node}, f32);
  auto onsite = pass.has_spin ? torch::empty({run, 3}, f32) : torch::Tensor();

  const auto rows = [](const torch::Tensor& buffer, long begin) {
    return buffer.data_ptr<float>() + begin * buffer.size(-1);
  };
  // The launch of further block ``b`` over the run of ``edges``.
  const auto block_arguments = [&](const Arguments& edges, long b) {
    BlockArguments block;
    block.edges = edges;
    block.record = records[b].data_ptr<float>();
    block.first_state = first_state.data_ptr<float>();
    block.previous_state = b == 0 ? first_state.data_ptr<float>()
                                  : states[b - 1].data_ptr<float>();
    block.sources = sources[b].data_ptr<float>();
    block.centers = centers[b].data_ptr<float>();
    block.divisors = divisors.data_ptr<float>();
    block.cotangents = cotangents[b % 2].data_ptr<float>();
    block.carried = carried.data_ptr<float>();
    block.center_gradient = rows(center_gradient, edges.node_begin);
    return block;
  };
  // Every readout VJP of a run adds the on-site magnetic gradient of its
  // invariants.
  const auto node_backward = [&](Arguments node, long begin, long count) {
    if (pass.has_spin) {
      node.spin_gradient = onsite.data_ptr<float>();
    }
    dispatch_first_stage(channels, FirstStage::NodeBackward, node, stream);
    if (pass.has_spin) {
      pass.spin_gradient.narrow(0, begin, count)
          .add_(onsite.narrow(0, 0, count));
    }
  };

  // === Forward: the first block and the node maps of the second ===
  for (long begin = 0; begin < owned; begin += run) {
    const long count = std::min(run, owned - begin);
    Arguments first = first_arguments(begin, count);
    first.state_out = rows(first_state, begin);
    dispatch(channels, false, first, stream);
    BlockArguments maps = block_arguments(first, 0);
    maps.descriptor = descriptor;
    dispatch_block(channels, BlockStage::Maps, maps, stream);
  }

  // === Forward: the further blocks ===
  for (long b = 0; b < block_count; ++b) {
    const bool last = b + 1 == block_count;
    exchange_without_runs(sources[b]);
    for (long begin = 0; begin < owned; begin += run) {
      const long count = std::min(run, owned - begin);
      Arguments edges = block_edges(b, begin, count);
      BlockArguments block = block_arguments(edges, b);
      block.state_out = last ? state : rows(states[b], begin);
      block.transport =
          last ? transport.data_ptr<float>() : rows(transports[b], begin);
      scan_run(BlockStage::Forward, block, b, begin, sources[b]);
      edges.state_out = block.state_out;
      edges.divisors = rows(divisors, begin);
      dispatch_first_stage(channels, FirstStage::Readout, edges, stream);
      if (!last) {
        BlockArguments maps = block_arguments(edges, b + 1);
        maps.descriptor = descriptor;
        dispatch_block(channels, BlockStage::Maps, maps, stream);
        continue;
      }
      // The cotangent replaces the descriptor, which the run no longer needs.
      pass.runs.output_and_cotangent(
          stream, begin, count, descriptor, descriptor,
          pass.widths.output_width, pass.node_type.data_ptr<long>() + begin,
          pass.seed.data_ptr<double>() + begin,
          pass.raw_energy.data_ptr<double>() + begin);
      edges.descriptor_gradient = descriptor;
      edges.state = state;
      edges.moment_gradient = state;
      node_backward(edges, begin, count);
      block.moment_gradient = state;
      block.accumulate_carried = false;
      dispatch_block(channels, BlockStage::Cotangents, block, stream);
    }
  }

  // === Backward: the further blocks in reverse, then the first block ===
  for (long b = block_count - 1; b >= 0; --b) {
    const torch::Tensor input = b == 0 ? first_state : states[b - 1];
    exchange_without_runs(cotangents[b % 2]);
    for (long begin = 0; begin < owned; begin += run) {
      const long count = std::min(run, owned - begin);
      Arguments edges = block_edges(b, begin, count);
      edges.force = pass.force;
      BlockArguments block = block_arguments(edges, b);
      block.source_gradient = source_gradient.data_ptr<float>();
      block.source_virial = rows(block_virial, begin);
      if (pass.bridged) {
        block.gate_sums = gate_rows.data_ptr<float>();
      }
      scan_run(BlockStage::Backward, block, b, begin, cotangents[b % 2]);

      // The invariants of the block before, which the node maps of the
      // block read, then their cotangent in their place.
      edges.state_out = rows(input, begin);
      edges.divisors = rows(divisors, begin);
      dispatch_first_stage(channels, FirstStage::Readout, edges, stream);
      block.descriptor = descriptor;
      block.descriptor_gradient = descriptor;
      block.keys_gradient = keys_gradient.data_ptr<float>();
      block.map_cotangent = map_cotangent.data_ptr<float>();
      dispatch_block(channels, BlockStage::MapBackward, block, stream);
      edges.descriptor_gradient = descriptor;
      edges.state = rows(input, begin);
      edges.moment_gradient = state;
      node_backward(edges, begin, count);
      auto gradient = pass.state.narrow(0, 0, count);
      gradient.add_(keys_gradient.narrow(0, 0, count));
      if (b > 0) {
        BlockArguments before = block_arguments(edges, b - 1);
        before.moment_gradient = state;
        before.transport = rows(transports[b - 1], begin);
        before.accumulate_carried = true;
        dispatch_block(channels, BlockStage::Cotangents, before, stream);
        continue;
      }
      gradient.add_(carried.narrow(0, begin, count));
      Arguments first = first_arguments(begin, count);
      first.moment_gradient = state;
      attach_force(first, begin, count);
      if (pass.bridged) {
        first.gate_sums = reinterpret_cast<unsigned long long*>(
            gate_fixed.data_ptr<int64_t>());
      }
      dispatch_first_stage(channels, FirstStage::EdgeBackward, first, stream);
    }
  }

  // === Slope of the switches through the gates ===
  if (pass.bridged) {
    auto gate_sums =
        gate_fixed.to(torch::kFloat64)
            .mul_(1.0 / static_cast<double>(deepmd_dpa4c::kForceFixedScale))
            .add_(gate_rows)
            .to(torch::kFloat32)
            .reshape({node_count, 1})
            .contiguous();
    pass.halo.accumulate(gate_sums);
    launch_gate_closure(first_arguments(0, owned), source_gate,
                        gate_sums.data_ptr<float>(), pass.force,
                        block_virial.data_ptr<float>(), stream);
  }
  return block_virial;
}

}  // namespace

// Energy, force and virial of one compressed inference step, evaluated over
// runs of consecutive destination nodes.
//
// Destination-sorted CSR gives a run a contiguous span of the edge axis and
// assigns every edge to exactly one destination, so the runs partition the
// work rather than splitting any reduction. Folding the descriptor, the
// fitting, the descriptor backward and the force assembly into one operator
// lets a run retire its descriptor, its cotangent and its moment state before
// the next run starts, and unless the per-atom virial is requested no
// edge-scale cotangent is materialized: the edge backward adds the source
// half of every force onto the fixed-point totals and reduces the
// destination half and the virial over its own row.
// Nothing is recomputed: the fitting seed is the ownership mask, known up
// front.
//
// The operator also owns the node bookkeeping around the network: the
// ownership mask that selects the energy-contributing nodes, the masked node
// energies, the frame totals and the conversion of the fixed-point force
// totals, so that one evaluation of a single frame adds a prologue, an
// epilogue and the fill that clears their words to its descriptor and fitting
// kernels.
//
// The per-atom virial is attributed to source nodes and is the one output
// that reads the source CSR. A caller that does not want it passes two-entry
// placeholders of zeros in its place; the per-atom virial is then returned as
// zeros and the edge cotangents are never stored. A graph of one node and two
// edge slots reads such placeholders as a complete source CSR of no edge, with
// the same result.
//
// A force contribution of 2^30 eV/Å or more lies outside the range of the
// fixed-point force totals, and marks its frame like a non-finite total (see
// kForceFixedLimit for the scope of the check).
//
// Every reduction has a fixed order for a given node count and kernel
// configuration, so an evaluation is reproducible bit for bit on one device;
// the configuration follows from the device, so another device may differ in
// the last bits.
//
// The frame counts ``n_node`` sum to at most the node count, and ``n_local``
// does not exceed them; nodes past their sum belong to no frame and enter no
// total.
//
// The loop lives here rather than in Python because its trip count follows a
// dynamic node count, which export cannot trace.
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
dpa4c_canonical_compress_energy_force(
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor source_row_ptr,
    torch::Tensor source_order,
    torch::Tensor n_node,
    torch::Tensor n_local,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor basis_table,
    torch::Tensor basis_matrix,
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
    torch::Tensor contact_radius,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor,
    double f_inner,
    double f_outer,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> bs,
    std::vector<int64_t> resnets,
    torch::Tensor w_head,
    torch::Tensor b_head,
    torch::Tensor bias_atom_e,
    int64_t act,
    int64_t tile,
    torch::Tensor pair_table,
    std::optional<std::vector<int64_t>> fitting_routes,
    std::optional<torch::Tensor> blocks,
    std::optional<torch::Tensor> halo_send,
    std::optional<torch::Tensor> halo_swaps,
    std::optional<torch::Tensor> communicator,
    std::optional<std::vector<double>> block_rcut) {
  const bool atom_virial = check_energy_force_inputs(
      "dpa4c_canonical_compress_energy_force", edge_vec, source,
      destination_row_ptr, source_row_ptr, source_order, n_node, n_local, atype,
      type_embedding, contact_radius, f_inner, f_outer);
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  const long node_count = atype.size(0);
  const int channels = static_cast<int>(type_embedding.size(1));
  const bool has_spin = spin.dim() == 2;
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), has_spin);
  const long frame_count = n_node.size(0);
  // One record per further message-passing block; none for a descriptor of
  // one block.
  const torch::Tensor records = blocks.value_or(torch::Tensor());
  TORCH_CHECK(!records.defined() || records.dim() == 2,
              "dpa4c_canonical_compress_energy_force: the records of the "
              "further blocks must form a matrix, got ",
              records.sizes());
  const long block_count = records.defined() ? records.size(0) : 0;
  // Without the cutoffs of the further blocks, every block scans the edges of
  // the first, whose messages beyond its own cutoff vanish with its envelope.
  const std::vector<double> cutoffs =
      block_rcut.value_or(std::vector<double>(block_count, rcut));
  TORCH_CHECK(static_cast<long>(cutoffs.size()) == block_count,
              "dpa4c_canonical_compress_energy_force: block_rcut must hold "
              "the cutoff of every further block, one per record of blocks; "
              "got ",
              cutoffs.size(), " cutoff(s) for ", block_count, " record(s)");
  for (long b = 0; b < block_count; ++b) {
    const double previous = b == 0 ? rcut : cutoffs[b - 1];
    TORCH_CHECK(cutoffs[b] > 0.0 && cutoffs[b] <= previous,
                "dpa4c_canonical_compress_energy_force: the cutoff of further "
                "block ",
                b + 1, " must be positive and at most ", previous,
                " Å, that of the block before; got ", cutoffs[b], " Å");
  }
  const HaloPlan halo =
      halo_plan("dpa4c_canonical_compress_energy_force", halo_send, halo_swaps,
                communicator, node_count, frame_count, block_count);
  // A node set with halo rows evaluates its owned rows: the halo rows carry
  // no edges, and every feature of theirs that an edge reads arrives from the
  // rank that owns them.
  const long owned = halo.owned;
  const auto f32 = edge_vec.options().dtype(torch::kFloat32);
  const auto f64 = edge_vec.options().dtype(torch::kFloat64);
  const auto i64 = edge_vec.options().dtype(torch::kInt64);
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Every output receives its own allocation, so that no two of the seven
  // unannotated results share storage.
  auto energy = torch::empty({frame_count, 1}, f64);
  auto atom_energy = torch::empty({node_count, 1}, f64);
  auto force = torch::empty({node_count, 3}, f64);
  auto virial = torch::empty({frame_count, 3, 3}, f64);
  auto atom_virial_output = torch::empty({node_count, 3, 3}, f32);
  auto force_mag =
      has_spin ? torch::empty({node_count, 3}, f64) : torch::empty({0}, f32);
  auto mask =
      torch::empty({node_count}, edge_vec.options().dtype(torch::kInt32));

  // With further blocks the runs cover the owned rows alone, and every node
  // backward adds its on-site magnetic gradient, so the node buffers the
  // epilogue reads start cleared.
  const auto node_buffer = [&](std::initializer_list<long> shape,
                               const torch::TensorOptions& options) {
    return block_count > 0 ? torch::zeros(shape, options)
                           : torch::empty(shape, options);
  };
  auto raw_energy = node_buffer({node_count}, f64);
  auto seed = torch::empty({node_count}, f64);
  auto node_type = torch::empty({node_count}, i64);
  auto force_fixed = torch::empty({node_count, kForceWords}, i64);
  auto node_virial = node_buffer({node_count, 9}, f32);
  // The on-site magnetic gradient is node local, so a run writes only its own
  // rows. The neighbour part belongs to source nodes that other runs own and
  // accumulates in fixed point like the force.
  auto spin_gradient =
      has_spin ? node_buffer({node_count, 3}, f32) : torch::Tensor();
  auto spin_fixed =
      has_spin ? torch::empty({node_count, kForceWords}, i64) : torch::Tensor();
  auto edge_gradient =
      atom_virial ? torch::empty_like(edge_vec_float) : torch::Tensor();
  const int blocks_of_epilogue = epilogue_blocks(node_count);
  auto words =
      torch::zeros({static_cast<long>(sizeof(EvaluationWords) / sizeof(int))},
                   edge_vec.options().dtype(torch::kInt32));
  auto* evaluation_words =
      reinterpret_cast<EvaluationWords*>(words.data_ptr<int>());
  if (node_count == 0) {
    energy.zero_();
    virial.zero_();
    return {energy,    atom_energy, force, virial, atom_virial_output,
            force_mag, mask};
  }
  const torch::Tensor frame_row_ptr =
      frame_count > 1 ? frame_row_pointer(n_node) : torch::Tensor();
  energy_force_prologue_kernel<<<blocks_of_epilogue, kEpilogueThreads, 0,
                                 stream>>>(
      node_count, frame_count, n_local.data_ptr<long>(),
      frame_count > 1 ? frame_row_ptr.data_ptr<long>() : nullptr,
      atype.data_ptr<long>(), node_type.data_ptr<long>(),
      seed.data_ptr<double>(), mask.data_ptr<int>(), nullptr,
      reinterpret_cast<unsigned long long*>(force_fixed.data_ptr<int64_t>()),
      has_spin ? reinterpret_cast<unsigned long long*>(
                     spin_fixed.data_ptr<int64_t>())
               : nullptr,
      f_outer > f_inner ? contact_radius.data_ptr<float>() : nullptr,
      evaluation_words);
  DPA4C_CHECK_LAUNCH("dpa4c energy-force prologue");

  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.n_layer > 0 && ws[0].size(0) == widths.output_width,
              "dpa4c_canonical_compress_energy_force: first fitting weight "
              "does not match the descriptor width");
  const long run = tile > 0 ? std::max<long>(1, std::min<long>(tile, owned))
                            : std::max<long>(1, owned);
  auto descriptor = torch::empty({run, widths.output_width}, f32);
  auto state = torch::empty({run, widths.moment_width + 2}, f32);
  const FittingNetwork fitting{plan,   ws,     bs,          resnets,
                               w_head, b_head, bias_atom_e, act};
  const FittingRuns runs(stream, fitting, run, owned, std::move(fitting_routes),
                         f32);

  auto empty_index = torch::empty({0}, source.options());
  auto empty_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  // The launch bundle of the first block over the run of ``count`` owned rows
  // from ``begin``.
  const auto first_arguments = [&](long begin, long count) {
    const Payload payload{
        source,
        empty_mask,
        empty_index,
        destination_row_ptr.slice(0, begin, begin + count + 1),
        node_type,
        table,
        basis_table,
        basis_matrix,
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
        contact_radius,
        true,
        lmax,
        table_stride,
        table_max,
        rcut,
        eps,
        degree_floor,
        f_inner,
        f_outer,
        begin};
    Arguments arguments =
        build_arguments(payload, edge_vec_float, channels, widths);
    arguments.system_node_count = owned;
    arguments.descriptor = descriptor.data_ptr<float>();
    arguments.state_out = state.data_ptr<float>();
    arguments.type_mismatch = &evaluation_words->node_types.type_mismatch;
    arguments.largest_radius = &evaluation_words->node_types.largest_radius;
    arguments.type_present =
        present_types(arguments, &evaluation_words->node_types);
    return arguments;
  };
  // The force assembly of the edge backward of the first block over a run,
  // and the analytical pair term: the fitting forward has assigned the energy
  // rows of the run, and the seed is the cotangent the fitting backward
  // consumed.
  const auto attach_force = [&](Arguments& arguments, long begin, long count) {
    arguments.force =
        reinterpret_cast<unsigned long long*>(force_fixed.data_ptr<int64_t>());
    arguments.node_virial = node_virial.data_ptr<float>() + begin * 9;
    arguments.edge_gradient =
        atom_virial ? edge_gradient.data_ptr<float>() : nullptr;
    if (has_spin) {
      // Node-indexed like the energy, so the run addresses its own slice.
      arguments.spin_gradient = spin_gradient.data_ptr<float>() + begin * 3;
      arguments.spin_force =
          reinterpret_cast<unsigned long long*>(spin_fixed.data_ptr<int64_t>());
    }
    // Only the final run reaches the reserved edge slots; its row pointer ends
    // at the last physical edge, which is exactly where the padding begins.
    arguments.clear_padding = begin + count == owned;
    if (pair_table.numel() != 0) {
      attach_pair_term(arguments, pair_table, edge_vec.device(),
                       seed.data_ptr<double>() + begin,
                       raw_energy.data_ptr<double>() + begin);
    }
  };

  torch::Tensor block_virial;
  if (block_count > 0) {
    const EdgeGraph graph{destination_row_ptr, source, edge_vec_float};
    const BlockPass pass{
        records,
        rcut,
        cutoffs,
        graph,
        halo,
        channels,
        static_cast<int>(lmax),
        has_spin,
        f_outer > f_inner,
        widths,
        node_count,
        run,
        runs,
        descriptor,
        state,
        node_type,
        seed,
        raw_energy,
        spin_gradient,
        reinterpret_cast<unsigned long long*>(force_fixed.data_ptr<int64_t>()),
        stream};
    block_virial = evaluate_further_blocks(pass, first_arguments, attach_force);
  } else {
    for (long begin = 0; begin < owned; begin += run) {
      const long count = std::min(run, owned - begin);
      Arguments arguments = first_arguments(begin, count);
      dispatch(channels, false, arguments, stream);

      // The cotangent replaces the descriptor, which the run no longer needs.
      runs.output_and_cotangent(
          stream, begin, count, descriptor.data_ptr<float>(),
          descriptor.data_ptr<float>(), widths.output_width,
          node_type.data_ptr<long>() + begin, seed.data_ptr<double>() + begin,
          raw_energy.data_ptr<double>() + begin);

      arguments.descriptor_gradient = descriptor.data_ptr<float>();
      arguments.state = state.data_ptr<float>();
      arguments.moment_gradient = state.data_ptr<float>();
      attach_force(arguments, begin, count);
      dispatch(channels, true, arguments, stream);
    }
  }

  // The further blocks attribute the virial of every edge to the node that
  // emits it, its source, which is how the per-atom virial attributes it, so
  // the frame virial and the per-atom virial take it as it is.
  if (block_virial.defined()) {
    node_virial.add_(block_virial);
  }
  const EpilogueInputs step{n_node, frame_row_ptr, force_fixed,     raw_energy,
                            seed,   node_virial,   evaluation_words};
  const SpinTerms spin_terms{spin_fixed, spin_gradient, force_mag};
  const SourceVirialTerms source_terms{edge_gradient, edge_vec_float,
                                       source_row_ptr, source_order};
  finish_energy_force(stream, node_count, blocks_of_epilogue, step,
                      has_spin ? &spin_terms : nullptr,
                      atom_virial ? &source_terms : nullptr, nullptr, energy,
                      atom_energy, force, virial, atom_virial_output);
  if (block_virial.defined() && atom_virial) {
    atom_virial_output.add_(block_virial.reshape({node_count, 3, 3}));
  }
  return {energy,    atom_energy, force, virial, atom_virial_output,
          force_mag, mask};
}

// Energy, force and virial of one compressed inference step of a model with a
// latent-charge long-range module.
//
// The module reads charges that its network predicts from the descriptor of
// every node, and the energy of the charges depends on all of them, so the
// descriptor backward cannot start before every charge is known. The
// operator evaluates the descriptor of every node with its moment state, and
// the charges from the descriptor; then the long range of the charges at
// fixed charges with its derivative with respect to them
// (deepmd::latent_charge_sum); then the energy fitting with its cotangent, to
// which the Jacobian row of every charge with respect to the descriptor of
// its node, the cotangent of a unit seed, scaled by that derivative adds the
// cotangent of the charge network; and one descriptor backward from the sum.
// As in dpa4c_canonical_compress_energy_force, the two networks evaluate the
// node axis in runs of ``tile`` nodes, and each takes the fused network or
// the per-layer kernels by a fitting-route table of its own
// (``fitting_routes`` and ``charge_fitting_routes``). A run of the fused
// charge network forms its Jacobian rows in the pass of its charges, and a
// run of the per-layer kernels after the energy fitting, from the
// pre-activations that it keeps for every node; on a V100 at 512 atoms that
// order shortens the LAMMPS step by 3.3% against forming them before the
// long range. The Jacobian rows take the width of the descriptor for every
// node. The edge backward adds every force onto the
// fixed-point totals as in dpa4c_canonical_compress_energy_force, and the
// epilogue adds the energies, the forces at fixed charges and the per-node
// virials of the long range.
//
// The charge network has one channel and the layout of an eligible energy
// fitting; its per-type bias is ``charge_bias``. ``coord``, ``total_charge``,
// ``box``, ``coefficient``, ``communicator``, ``kernel``, ``channel``,
// ``prefactor``, ``order`` and ``spacing`` are those of
// deepmd::latent_charge_sum, whose charged nodes are the owned real nodes of
// the frames. The per-atom virial and the source CSR follow
// dpa4c_canonical_compress_energy_force; the descriptor carries no spin
// condition and the model no analytical pair potential.
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
dpa4c_canonical_compress_long_range_energy_force(
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor source_row_ptr,
    torch::Tensor source_order,
    torch::Tensor n_node,
    torch::Tensor n_local,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor basis_table,
    torch::Tensor basis_matrix,
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
    torch::Tensor contact_radius,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor,
    double f_inner,
    double f_outer,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> bs,
    std::vector<int64_t> resnets,
    torch::Tensor w_head,
    torch::Tensor b_head,
    torch::Tensor bias_atom_e,
    int64_t act,
    int64_t tile,
    std::vector<torch::Tensor> charge_ws,
    std::vector<torch::Tensor> charge_bs,
    std::vector<int64_t> charge_resnets,
    torch::Tensor charge_w_head,
    torch::Tensor charge_b_head,
    torch::Tensor charge_bias,
    int64_t charge_act,
    torch::Tensor coord,
    torch::Tensor total_charge,
    std::optional<torch::Tensor> box,
    torch::Tensor coefficient,
    torch::Tensor communicator,
    int64_t kernel,
    int64_t channel,
    double prefactor,
    int64_t order,
    double spacing,
    std::optional<std::vector<int64_t>> fitting_routes,
    std::optional<std::vector<int64_t>> charge_fitting_routes) {
  constexpr const char* kOp =
      "dpa4c_canonical_compress_long_range_energy_force";
  const bool atom_virial = check_energy_force_inputs(
      kOp, edge_vec, source, destination_row_ptr, source_row_ptr, source_order,
      n_node, n_local, atype, type_embedding, contact_radius, f_inner, f_outer);
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  TORCH_CHECK(spin.dim() != 2, kOp,
              ": a descriptor with a spin condition is not supported");
  const long node_count = atype.size(0);
  TORCH_CHECK(coord.is_cuda() && coord.device() == edge_vec.device() &&
                  coord.dim() == 2 && coord.size(0) == node_count &&
                  coord.size(1) == 3 && coord.scalar_type() == torch::kFloat64,
              kOp,
              ": coord must be a float64 tensor of shape (N, 3) on the device "
              "of edge_vec");
  const int channels = static_cast<int>(type_embedding.size(1));
  const Dimensions widths = profile_dimensions(channels, static_cast<int>(lmax),
                                               /*has_spin=*/false);
  const long frame_count = n_node.size(0);
  const auto f32 = edge_vec.options().dtype(torch::kFloat32);
  const auto f64 = edge_vec.options().dtype(torch::kFloat64);
  const auto i64 = edge_vec.options().dtype(torch::kInt64);
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Every output receives its own allocation, so that no two of the six
  // unannotated results share storage.
  auto energy = torch::empty({frame_count, 1}, f64);
  auto atom_energy = torch::empty({node_count, 1}, f64);
  auto force = torch::empty({node_count, 3}, f64);
  auto virial = torch::empty({frame_count, 3, 3}, f64);
  auto atom_virial_output = torch::empty({node_count, 3, 3}, f32);
  auto mask =
      torch::empty({node_count}, edge_vec.options().dtype(torch::kInt32));
  if (node_count == 0) {
    energy.zero_();
    virial.zero_();
    return {energy, atom_energy, force, virial, atom_virial_output, mask};
  }

  auto raw_energy = torch::empty({node_count}, f64);
  auto seed = torch::empty({node_count}, f64);
  auto charged =
      torch::empty({node_count}, edge_vec.options().dtype(torch::kBool));
  auto node_type = torch::empty({node_count}, i64);
  auto force_fixed = torch::empty({node_count, kForceWords}, i64);
  auto node_virial = torch::empty({node_count, 9}, f32);
  auto edge_gradient =
      atom_virial ? torch::empty_like(edge_vec_float) : torch::Tensor();
  const int blocks = epilogue_blocks(node_count);
  auto words =
      torch::zeros({static_cast<long>(sizeof(EvaluationWords) / sizeof(int))},
                   edge_vec.options().dtype(torch::kInt32));
  auto* evaluation_words =
      reinterpret_cast<EvaluationWords*>(words.data_ptr<int>());
  const torch::Tensor frame_row_ptr =
      frame_count > 1 ? frame_row_pointer(n_node) : torch::Tensor();
  energy_force_prologue_kernel<<<blocks, kEpilogueThreads, 0, stream>>>(
      node_count, frame_count, n_local.data_ptr<long>(),
      frame_count > 1 ? frame_row_ptr.data_ptr<long>() : nullptr,
      atype.data_ptr<long>(), node_type.data_ptr<long>(),
      seed.data_ptr<double>(), mask.data_ptr<int>(), charged.data_ptr<bool>(),
      reinterpret_cast<unsigned long long*>(force_fixed.data_ptr<int64_t>()),
      nullptr, f_outer > f_inner ? contact_radius.data_ptr<float>() : nullptr,
      evaluation_words);
  DPA4C_CHECK_LAUNCH("dpa4c long-range energy-force prologue");

  const FittingLayerPlan plan = fitting_layer_plan(ws);
  const FittingLayerPlan charge_plan = fitting_layer_plan(charge_ws);
  TORCH_CHECK(plan.n_layer > 0 && ws[0].size(0) == widths.output_width &&
                  charge_plan.n_layer > 0 &&
                  charge_ws[0].size(0) == widths.output_width,
              kOp,
              ": the first weights of the energy fitting and of the charge "
              "network must match the descriptor width");
  TORCH_CHECK(charge_w_head.numel() == charge_ws.back().size(1) &&
                  charge_bias.dim() == 1 &&
                  charge_bias.size(0) == bias_atom_e.size(0),
              kOp,
              ": the charge network must have a scalar head and one bias per "
              "type, as the energy fitting has");

  // === Step 1. Descriptor and moment state of every node ===
  auto descriptor = torch::empty({node_count, widths.output_width}, f32);
  auto state = torch::empty({node_count, widths.moment_width + 2}, f32);
  auto empty_index = torch::empty({0}, source.options());
  auto empty_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  const Payload payload{source,
                        empty_mask,
                        empty_index,
                        destination_row_ptr,
                        node_type,
                        table,
                        basis_table,
                        basis_matrix,
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
                        contact_radius,
                        true,
                        lmax,
                        table_stride,
                        table_max,
                        rcut,
                        eps,
                        degree_floor,
                        f_inner,
                        f_outer,
                        0};
  Arguments arguments =
      build_arguments(payload, edge_vec_float, channels, widths);
  arguments.system_node_count = node_count;
  arguments.descriptor = descriptor.data_ptr<float>();
  arguments.state_out = state.data_ptr<float>();
  arguments.type_mismatch = &evaluation_words->node_types.type_mismatch;
  arguments.largest_radius = &evaluation_words->node_types.largest_radius;
  arguments.type_present =
      present_types(arguments, &evaluation_words->node_types);
  dispatch(channels, false, arguments, stream);

  // === Step 2. Charges of the nodes and their descriptor Jacobian ===
  // The charge of a node is a function of the descriptor of that node alone,
  // so the cotangent of its descriptor through the charge network is the
  // derivative of the energy with respect to the charge times the Jacobian
  // row of the charge, the input cotangent of a unit seed. A run of the fused
  // network forms the charges and those rows in one pass, before the long
  // range forms the derivative and the energy fitting overwrites the
  // descriptor; a run of the per-layer kernels forms its charges here and
  // its rows from their saved pre-activations after the energy fitting.
  const long run = tile > 0
                       ? std::max<long>(1, std::min<long>(tile, node_count))
                       : node_count;
  const long width = widths.output_width;
  const auto charge_bias_double = charge_bias.to(torch::kFloat64).contiguous();
  const FittingNetwork charge_fitting{
      charge_plan,   charge_ws,     charge_bs,          charge_resnets,
      charge_w_head, charge_b_head, charge_bias_double, charge_act};
  const FittingRuns charge_runs(stream, charge_fitting, run, node_count,
                                std::move(charge_fitting_routes), f32, true);
  const auto unit_seed = torch::ones({run}, f64);
  auto raw_charge = torch::empty({node_count}, f64);
  auto charge_jacobian = torch::empty({node_count, width}, f32);
  for (long begin = 0; begin < node_count; begin += run) {
    const long count = std::min(run, node_count - begin);
    charge_runs.forward(
        stream, begin, count, descriptor.data_ptr<float>() + begin * width,
        charge_jacobian.data_ptr<float>() + begin * width, width,
        node_type.data_ptr<long>() + begin, unit_seed.data_ptr<double>(),
        raw_charge.data_ptr<double>() + begin);
  }

  // === Step 3. Long range of the charges ===
  // The per-node virials of the charged nodes sum to the virial of each frame,
  // which the frame totals form with those of the descriptor.
  const auto long_range = deepmd::long_range::latent_charge_sum(
      coord, raw_charge.to(torch::kFloat32).unsqueeze(1), charged, n_node,
      total_charge, box, coefficient, communicator, kernel, channel, prefactor,
      order, spacing);
  const torch::Tensor& long_range_energy = std::get<0>(long_range);
  const torch::Tensor& charge_gradient = std::get<1>(long_range);
  const torch::Tensor& long_range_force = std::get<2>(long_range);
  const torch::Tensor& long_range_atom_virial = std::get<3>(long_range);

  // === Step 4. Cotangent of the descriptor ===
  // The cotangent of the energy fitting replaces the descriptor run by run;
  // the runs of the per-layer kernels of the charge network then form their
  // Jacobian rows, and the rows, scaled by the derivative of the energy with
  // respect to the charges, add the cotangent of the charge network.
  const FittingNetwork fitting{plan,   ws,     bs,          resnets,
                               w_head, b_head, bias_atom_e, act};
  const FittingRuns runs(stream, fitting, run, node_count,
                         std::move(fitting_routes), f32);
  for (long begin = 0; begin < node_count; begin += run) {
    const long count = std::min(run, node_count - begin);
    float* const run_descriptor = descriptor.data_ptr<float>() + begin * width;
    runs.output_and_cotangent(
        stream, begin, count, run_descriptor, run_descriptor, width,
        node_type.data_ptr<long>() + begin, seed.data_ptr<double>() + begin,
        raw_energy.data_ptr<double>() + begin);
  }
  for (long begin = 0; begin < node_count; begin += run) {
    const long count = std::min(run, node_count - begin);
    charge_runs.backward(stream, begin, count, unit_seed.data_ptr<double>(),
                         charge_jacobian.data_ptr<float>() + begin * width);
  }
  descriptor.addcmul_(charge_jacobian, charge_gradient);

  // === Step 5. Descriptor backward and force assembly ===
  arguments.descriptor_gradient = descriptor.data_ptr<float>();
  arguments.state = state.data_ptr<float>();
  arguments.moment_gradient = state.data_ptr<float>();
  arguments.force =
      reinterpret_cast<unsigned long long*>(force_fixed.data_ptr<int64_t>());
  arguments.node_virial = node_virial.data_ptr<float>();
  arguments.edge_gradient =
      atom_virial ? edge_gradient.data_ptr<float>() : nullptr;
  arguments.clear_padding = true;
  dispatch(channels, true, arguments, stream);

  // === Step 6. Node results and frame totals ===
  const EpilogueInputs step{n_node, frame_row_ptr, force_fixed,     raw_energy,
                            seed,   node_virial,   evaluation_words};
  const SourceVirialTerms source_terms{edge_gradient, edge_vec_float,
                                       source_row_ptr, source_order};
  const LongRangeTerms long_range_terms{long_range_energy, long_range_force,
                                        long_range_atom_virial};
  finish_energy_force(stream, node_count, blocks, step, nullptr,
                      atom_virial ? &source_terms : nullptr, &long_range_terms,
                      energy, atom_energy, force, virial, atom_virial_output);
  return {energy, atom_energy, force, virial, atom_virial_output, mask};
}

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.impl("dpa4c_graph_compress", torch::kCUDA, &dpa4c_graph_compress);
  library.impl("dpa4c_graph_compress_backward", torch::kCUDA,
               &dpa4c_graph_compress_backward);
  library.impl("dpa4c_canonical_compress", torch::kCUDA,
               &dpa4c_canonical_compress);
  library.impl("dpa4c_canonical_compress_backward", torch::kCUDA,
               &dpa4c_canonical_compress_backward);
  library.impl("dpa4c_canonical_compress_backward_inplace", torch::kCUDA,
               &dpa4c_canonical_compress_backward_inplace);
  library.impl("dpa4c_canonical_compress_energy_force", torch::kCUDA,
               &dpa4c_canonical_compress_energy_force);
  library.impl("dpa4c_canonical_compress_long_range_energy_force", torch::kCUDA,
               &dpa4c_canonical_compress_long_range_energy_force);
}
