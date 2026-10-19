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

#include "../graph_ops.h"
#include "graph_compress_launch.h"

namespace {

using deepmd_dpa4c::Arguments;
using deepmd_dpa4c::IndexKind;
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
// nodes of its frame -- as the int32 output mask and as the fp64 cotangent
// that seeds the fitting backward, and cleared fixed-point force totals. The
// frame of a node comes from ``frame_row_ptr`` when the batch holds several
// frames and is frame zero otherwise; nodes past the last frame are padding,
// which no frame owns.
__global__ void energy_force_prologue_kernel(
    long node_count,
    long frame_count,
    const long* __restrict__ n_local,
    const long* __restrict__ frame_row_ptr,
    const long* __restrict__ atype,
    long* __restrict__ node_type,
    double* __restrict__ seed,
    int* __restrict__ mask,
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
      const double value =
          force_fixed[node * kForceWords + kForceRangeWord] != 0
              ? CUDART_NAN
              : raw_energy[node] * seed[node];
      atom_energy[node] = value;
      if (node < frame_end) {
        sum += value;
      }
      continue;
    }
    if (component < 3) {
      force[node * 3 + component] =
          fixed_point_value(force_fixed[node * kForceWords + component]);
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
// from the segment reductions of the caller.
__global__ void energy_force_nodes_kernel(
    long node_count,
    const unsigned long long* __restrict__ force_fixed,
    const unsigned long long* __restrict__ spin_fixed,
    const float* __restrict__ spin_gradient,
    const double* __restrict__ raw_energy,
    const double* __restrict__ seed,
    double* __restrict__ force,
    double* __restrict__ force_mag,
    double* __restrict__ atom_energy,
    float* __restrict__ atom_virial_zero) {
  for (long node = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
       node < node_count; node += static_cast<long>(blockDim.x) * gridDim.x) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      force[node * 3 + component] =
          fixed_point_value(force_fixed[node * kForceWords + component]);
      if (force_mag != nullptr) {
        force_mag[node * 3 + component] =
            -(static_cast<double>(spin_gradient[node * 3 + component]) +
              fixed_point_value(spin_fixed[node * kForceWords + component]));
      }
    }
    atom_energy[node] = force_fixed[node * kForceWords + kForceRangeWord] != 0
                            ? CUDART_NAN
                            : raw_energy[node] * seed[node];
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
    std::optional<std::vector<int64_t>> fitting_routes) {
  TORCH_CHECK(edge_vec.is_cuda(),
              "dpa4c_canonical_compress_energy_force: edge_vec must be a CUDA "
              "tensor");
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress_energy_force: source and edge_vec "
              "must share the edge axis");
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  const long node_count = atype.size(0);
  const long edge_count = edge_vec.size(0);
  TORCH_CHECK(destination_row_ptr.numel() == node_count + 1,
              "dpa4c_canonical_compress_energy_force: atype and "
              "destination_row_ptr describe different node counts");
  // The prologue reads the frame counts, the node types and, for a windowed
  // descriptor, the contact radii before the runs validate the rest of the
  // payload, so those inputs are validated here.
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
              "dpa4c_canonical_compress_energy_force: n_node and n_local must "
              "be contiguous int64 tensors on the device of edge_vec with one "
              "shared shape (nf,), nf >= 1");
  TORCH_CHECK(on_device(atype) && atype.dim() == 1 &&
                  atype.scalar_type() == torch::kInt64,
              "dpa4c_canonical_compress_energy_force: atype must be a "
              "contiguous int64 tensor of shape (N,) on the device of "
              "edge_vec");
  TORCH_CHECK(
      f_outer <= f_inner || (on_device(contact_radius) &&
                             contact_radius.scalar_type() == torch::kFloat32 &&
                             contact_radius.dim() == 1 &&
                             contact_radius.size(0) == type_embedding.size(0)),
      "dpa4c_canonical_compress_energy_force: a windowed descriptor "
      "needs contiguous fp32 contact radii of shape (type_count,) on "
      "the device of edge_vec");
  // The source CSR is either complete, which requests the per-atom virial,
  // or the two-entry placeholder, which declines it.
  const bool atom_virial = source_row_ptr.numel() == node_count + 1 &&
                           source_order.numel() == edge_count;
  TORCH_CHECK(
      atom_virial || (source_row_ptr.numel() == 2 && source_order.numel() == 2),
      "dpa4c_canonical_compress_energy_force: the source CSR must "
      "hold ",
      node_count + 1, " row pointers and ", edge_count,
      " positions, or two entries each to decline the per-atom "
      "virial; got ",
      source_row_ptr.numel(), " and ", source_order.numel());
  if (atom_virial) {
    TORCH_CHECK(source.scalar_type() == torch::kUInt32 &&
                    on_device(source_row_ptr) &&
                    source_row_ptr.scalar_type() == torch::kInt64 &&
                    on_device(source_order) &&
                    source_order.scalar_type() == torch::kUInt32,
                "dpa4c_canonical_compress_energy_force: the per-atom virial "
                "needs uint32 sources and a source CSR of contiguous int64 "
                "row pointers and uint32 positions on the device of edge_vec");
  }
  const int channels = static_cast<int>(type_embedding.size(1));
  const bool has_spin = spin.dim() == 2;
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), has_spin);
  const long frame_count = n_node.size(0);
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

  auto raw_energy = torch::empty({node_count}, f64);
  auto seed = torch::empty({node_count}, f64);
  auto node_type = torch::empty({node_count}, i64);
  auto force_fixed = torch::empty({node_count, kForceWords}, i64);
  auto node_virial = torch::empty({node_count, 9}, f32);
  // The on-site magnetic gradient is node local, so a run writes only its own
  // rows. The neighbour part belongs to source nodes that other runs own and
  // accumulates in fixed point like the force.
  auto spin_gradient =
      has_spin ? torch::empty({node_count, 3}, f32) : torch::Tensor();
  auto spin_fixed =
      has_spin ? torch::empty({node_count, kForceWords}, i64) : torch::Tensor();
  auto edge_gradient =
      atom_virial ? torch::empty_like(edge_vec_float) : torch::Tensor();
  const int blocks = epilogue_blocks(node_count);
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
  energy_force_prologue_kernel<<<blocks, kEpilogueThreads, 0, stream>>>(
      node_count, frame_count, n_local.data_ptr<long>(),
      frame_count > 1 ? frame_row_ptr.data_ptr<long>() : nullptr,
      atype.data_ptr<long>(), node_type.data_ptr<long>(),
      seed.data_ptr<double>(), mask.data_ptr<int>(),
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
  const long run = tile > 0
                       ? std::max<long>(1, std::min<long>(tile, node_count))
                       : node_count;
  auto descriptor = torch::empty({run, widths.output_width}, f32);
  auto state = torch::empty({run, widths.moment_width + 2}, f32);
  auto saved = torch::empty({run * plan.saved_width()}, f32);
  // The full runs and the tail run each take the fitting route of their own
  // size; an absent table, like an empty one, leaves both to the rules. A
  // route of the fused network needs only the transposed weights besides the
  // saved pre-activations; the per-layer kernels need the activation
  // ping-pong, sized for the largest run that takes them.
  const std::vector<int64_t> routes =
      std::move(fitting_routes).value_or(std::vector<int64_t>{});
  const long tail = node_count % run;
  const int64_t full_route = fitting_route(plan, ws, run, routes);
  const int64_t tail_route =
      tail > 0 ? fitting_route(plan, ws, tail, routes) : full_route;
  const long full_count = node_count / run;
  const long per_layer_nodes =
      std::max(full_count > 0 && full_route == kFittingPerLayer ? run : 0L,
               tail > 0 && tail_route == kFittingPerLayer ? tail : 0L);
  torch::Tensor transposed;
  torch::Tensor scratch;
  float* slot[2] = {nullptr, nullptr};
  if ((full_count > 0 && full_route != kFittingPerLayer) ||
      (tail > 0 && tail_route != kFittingPerLayer)) {
    transposed = torch::empty({fitting_fused_transposed_floats(ws)}, f32);
    fitting_fused_transpose(stream, plan, ws, transposed.data_ptr<float>());
  }
  if (per_layer_nodes > 0) {
    const int slots = plan.n_layer > 1 ? 2 : 1;
    scratch = torch::empty({slots, per_layer_nodes, plan.width_max}, f32);
    slot[0] = scratch[0].data_ptr<float>();
    slot[1] = slots > 1 ? scratch[1].data_ptr<float>() : slot[0];
  }

  auto empty_index = torch::empty({0}, source.options());
  auto empty_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  for (long begin = 0; begin < node_count; begin += run) {
    const long count = std::min(run, node_count - begin);
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
    arguments.system_node_count = node_count;
    arguments.descriptor = descriptor.data_ptr<float>();
    arguments.state_out = state.data_ptr<float>();
    arguments.type_mismatch = &evaluation_words->node_types.type_mismatch;
    arguments.largest_radius = &evaluation_words->node_types.largest_radius;
    arguments.type_present =
        present_types(arguments, &evaluation_words->node_types);
    dispatch(channels, false, arguments, stream);

    // The cotangent replaces the descriptor, which the run no longer needs.
    const int64_t route = count == run ? full_route : tail_route;
    if (route != kFittingPerLayer) {
      fitting_energy_gradient_range(
          stream, plan, descriptor.data_ptr<float>(), widths.output_width,
          node_type.data_ptr<long>() + begin, ws, bs, resnets, w_head, b_head,
          bias_atom_e, act, transposed.data_ptr<float>(),
          seed.data_ptr<double>() + begin, count, saved.data_ptr<float>(),
          raw_energy.data_ptr<double>() + begin, descriptor.data_ptr<float>(),
          route);
    } else {
      fitting_forward_range(
          stream, plan, descriptor.data_ptr<float>(), widths.output_width,
          node_type.data_ptr<long>() + begin, ws, bs, resnets, w_head, b_head,
          bias_atom_e, act, count, saved.data_ptr<float>(), slot,
          raw_energy.data_ptr<double>() + begin);
      fitting_backward_range(
          stream, plan, seed.data_ptr<double>() + begin,
          saved.data_ptr<float>(), ws, bs, resnets, w_head, act, count, slot[0],
          plan.n_layer > 1 ? slot[1] : nullptr, descriptor.data_ptr<float>());
    }

    arguments.descriptor_gradient = descriptor.data_ptr<float>();
    arguments.state = state.data_ptr<float>();
    arguments.moment_gradient = state.data_ptr<float>();
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
    arguments.clear_padding = begin + count == node_count;
    if (pair_table.numel() != 0) {
      // The fitting forward has assigned the energy rows of this run, and the
      // seed is the cotangent the fitting backward just consumed.
      attach_pair_term(arguments, pair_table, edge_vec.device(),
                       seed.data_ptr<double>() + begin,
                       raw_energy.data_ptr<double>() + begin);
    }
    dispatch(channels, true, arguments, stream);
  }

  const auto* force_words = reinterpret_cast<const unsigned long long*>(
      force_fixed.data_ptr<int64_t>());
  const auto* spin_words = has_spin
                               ? reinterpret_cast<const unsigned long long*>(
                                     spin_fixed.data_ptr<int64_t>())
                               : nullptr;
  float* atom_virial_zero =
      atom_virial ? nullptr : atom_virial_output.data_ptr<float>();
  if (frame_count == 1) {
    auto partial = torch::empty({blocks, kEpilogueTotals}, f64);
    energy_force_epilogue_kernel<<<blocks, kEpilogueThreads * kEpilogueTotals,
                                   0, stream>>>(
        node_count, n_node.data_ptr<long>(), force_words, spin_words,
        has_spin ? spin_gradient.data_ptr<float>() : nullptr,
        raw_energy.data_ptr<double>(), seed.data_ptr<double>(),
        node_virial.data_ptr<float>(), force.data_ptr<double>(),
        has_spin ? force_mag.data_ptr<double>() : nullptr,
        atom_energy.data_ptr<double>(), atom_virial_zero,
        partial.data_ptr<double>(), evaluation_words, energy.data_ptr<double>(),
        virial.data_ptr<double>());
    DPA4C_CHECK_LAUNCH("dpa4c energy-force epilogue");
  } else {
    energy_force_nodes_kernel<<<blocks, kEpilogueThreads, 0, stream>>>(
        node_count, force_words, spin_words,
        has_spin ? spin_gradient.data_ptr<float>() : nullptr,
        raw_energy.data_ptr<double>(), seed.data_ptr<double>(),
        force.data_ptr<double>(),
        has_spin ? force_mag.data_ptr<double>() : nullptr,
        atom_energy.data_ptr<double>(), atom_virial_zero);
    DPA4C_CHECK_LAUNCH("dpa4c energy-force nodes");
    energy.copy_(frame_scalar_sum(atom_energy, n_node));
    virial.copy_(frame_virial_sum(node_virial, n_node));
    // A fixed-point force word cannot hold a non-finite value, so a frame
    // whose energy or virial is not finite reports every force as not a
    // number (see energy_force_epilogue_kernel).
    // Nodes past the last frame belong to none and keep their forces.
    const auto broken = torch::cat(
        {torch::logical_not(torch::isfinite(energy).all(1).logical_and(
             torch::isfinite(virial).flatten(1).all(1))),
         torch::zeros({1}, energy.options().dtype(torch::kBool))});
    const auto frame =
        torch::searchsorted(frame_row_ptr, torch::arange(node_count, i64),
                            /*out_int32=*/false, /*right=*/true) -
        1;
    const auto node_broken = broken.index_select(0, frame);
    force.masked_fill_(node_broken.unsqueeze(1),
                       std::numeric_limits<double>::quiet_NaN());
    if (has_spin) {
      force_mag.masked_fill_(node_broken.unsqueeze(1),
                             std::numeric_limits<double>::quiet_NaN());
    }
  }
  if (atom_virial) {
    atom_virial_output.copy_(
        canonical_source_virial(edge_gradient, edge_vec_float, source_row_ptr,
                                source_order, node_count));
  }
  return {energy,    atom_energy, force, virial, atom_virial_output,
          force_mag, mask};
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
}
