// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Device templates of the further message-passing blocks of the compressed
// DPA4C descriptor (stages, buffers and record layout in block_launch.h; the
// equations in deepmd/pt_expt/kernels/dpa4c/attention.py).
//
// The edge stages scan panels of consecutive edges of a block of destination
// rows. Every lane of a block owns the same tile of edges and channels in every
// per-edge quantity of a panel, so the amplitudes, the activations and their
// cotangents stay in registers from the table lookup through both edge maps,
// whose contractions read their operand, feature-major, from shared memory:
// the hidden layer reads the fields and the saturated amplitudes through
// [W_F; W_phi], the logit the fields and the activated hidden layer through
// [G; W_O]. The sums over the edges of a row take a few consecutive lanes per
// value, each summing a fixed residue class of the edges of the row, and add
// the lanes of a value in a fixed butterfly (add_row_sums).
//
// The backward scans the rows of the sources. A compact canonical graph lists
// every bond in both directions with opposite vectors, so the entries of the
// row of node j are the edges j emits, read with the reversed vector and the
// reversed type pair. The row then owns the cotangent of its source features
// and accumulates it in fixed order, while the destination cotangents that the
// edges read are gathered from their nodes and the force on the destination
// lands in fixed point, as in the first block.

#pragma once

#include <c10/util/Exception.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

#include "../simt_gemm.cuh"
#include "block_launch.h"
#include "graph_compress.cuh"

namespace deepmd_dpa4c {

constexpr int kBlockThreads = 256;
constexpr int kBlockWarps = kBlockThreads / kWarpSize;
/// Destination rows of a block of the edge scans at most; a small system
/// takes fewer (block_rows).
constexpr int kBlockRows = 4;
/// Items of the channel-contiguous amplitude order whose radial table rows a
/// lane loads together, ahead of their evaluation (panel_radial).
constexpr int kAmplitudeBatch = 4;

/// Shared memory that one block of an edge stage may hold: the 64 KB that a
/// block of compute capability 7.5 may opt in to, the least of the supported
/// devices (96 KB on 7.0, 99 KB or more from 8.0 on).
constexpr size_t kPanelSharedLimit = 64 * 1024;
/// Shared memory up to which an edge stage stages its edge maps: two blocks
/// of that size share a multiprocessor of 100 KB, so the staging never costs
/// a resident block.
constexpr size_t kResidentSharedBudget = 49 * 1024;

/// The geometry of the panels of an edge scan: the edges of a panel, the tile
/// of edges and channels of a lane, and the resident blocks per
/// multiprocessor for which the scan is compiled, zero leaving its register
/// budget to the compiler.
template <int EdgeCount, int TileEdges, int TileChannels, int Blocks>
struct PanelShape {
  static constexpr int Edges = EdgeCount;
  static constexpr int ThreadEdges = TileEdges;
  static constexpr int ThreadChannels = TileChannels;
  static constexpr int MinBlocks = Blocks;
};

/// The panel geometry of the forward and the backward (``Reverse``) edge scan
/// of every scalar width and angular degree. A lane tile of 4 x 4 issues half
/// the shared-memory reads per fused multiply-add of a tile of 2 x 4, and a
/// panel of 64 edges reads the streamed weights of the products half as often
/// per edge as a panel of 32, so a scan takes the widest tile and panel that
/// keep two blocks per multiprocessor. At 64 channels that holds for both
/// scans of degree two and for the forward of degree three; the backward from
/// degree three, whose register tile and per-edge state grow with the degree,
/// and both scans of degree four keep panels of 32 edges. At 128 channels a
/// panel holds 32 edges. The compiler gives a backward more registers than two
/// blocks leave, and the budget of two blocks, at the cost of a small spill,
/// runs it much faster than one block at the compiler's own count, so every
/// backward is compiled for two blocks; the forward scans fit two blocks on
/// the compiler's count.
template <int Edges, int TileEdges, int TileChannels, bool Reverse>
using ScanShape = PanelShape<Edges, TileEdges, TileChannels, Reverse ? 2 : 0>;
template <int Channels, int Lmax, bool Reverse>
struct BlockPanel;
template <int Lmax, bool Reverse>
struct BlockPanel<8, Lmax, Reverse> : ScanShape<64, 1, 2, Reverse> {};
template <int Lmax, bool Reverse>
struct BlockPanel<16, Lmax, Reverse> : ScanShape<64, 1, 4, Reverse> {};
template <int Lmax, bool Reverse>
struct BlockPanel<32, Lmax, Reverse> : ScanShape<64, 2, 4, Reverse> {};
template <int Lmax, bool Reverse>
struct BlockPanel<64, Lmax, Reverse> : ScanShape<64, 4, 4, Reverse> {};
template <>
struct BlockPanel<64, 3, true> : ScanShape<32, 2, 4, true> {};
template <bool Reverse>
struct BlockPanel<64, 4, Reverse> : ScanShape<32, 2, 4, Reverse> {};
template <int Lmax, bool Reverse>
struct BlockPanel<128, Lmax, Reverse> : ScanShape<32, 4, 4, Reverse> {};

/// The layout of the panels of one edge scan: the profile of the block, the
/// direction of the scan, and its panel geometry (BlockPanel). ``Reverse``
/// marks the backward, which reads every entry of a row as the edge the row
/// emits.
template <int ChannelCount, int Degree, bool Spin, bool Modes, bool Backward>
struct PanelLayout {
  static constexpr int Channels = ChannelCount;
  static constexpr int Lmax = Degree;
  static constexpr bool HasSpin = Spin;
  static constexpr bool HasModes = Modes;
  static constexpr bool Reverse = Backward;
  using B = BlockProfile<Channels, Lmax, HasSpin>;
  using S = BlockPanel<Channels, Lmax, Reverse>;
  static constexpr int Edges = S::Edges;
  static_assert(Edges == kWarpSize || Edges == 2 * kWarpSize,
                "a panel holds one or two warps of edges");
  static constexpr int TE = S::ThreadEdges;
  static constexpr int TC = S::ThreadChannels;
  static_assert(TC <= 4 && B::P::C2 % TC == 0,
                "a channel tile lies inside or outside every degree of two or "
                "more channels");
  static constexpr int ChannelTiles = Channels / TC;
  static constexpr int EdgeTiles = Edges / TE;
  static_assert(ChannelTiles * EdgeTiles == kBlockThreads,
                "the tiles of a panel cover the lanes of a block");
  /// A warp covers 8 edge tiles of 4 channel tiles each, so that each warp
  /// reads a slice of the weights of a product rather than all of them; the
  /// channel tiles of an edge are dealt in groups of four to
  /// ``ChannelGroups`` warps (see panel_tile for the lanes of a warp).
  static constexpr int WarpChannelTiles = 4;
  static constexpr int WarpEdgeTiles = kWarpSize / WarpChannelTiles;
  static_assert(ChannelTiles % WarpChannelTiles == 0,
                "the channel tiles of an edge fill whole warps");
  static constexpr int ChannelGroups = ChannelTiles / WarpChannelTiles;
  /// Row stride of the feature-major operand, padded to a 16-byte multiple.
  static constexpr int Stride = Edges + 4;
  static constexpr int BasisWidth = (Lmax + 1) * (Lmax + 1);
};

/// The tile of a lane in a panel: its channel group, whether it leads the
/// lanes of its warp that share its edges, and the first edge and channel of
/// its tile.
struct PanelTile {
  int group;
  bool leader;
  int e0;
  int c0;
};

/// Lane ``l`` of a warp holds the channel tile (l / 2) % 4 and the edge tile
/// 2 (l / 8) + l % 2 of the warp: every quarter-warp holds two edge tiles and
/// all four channel tiles, and the lanes that read one weight word of a
/// product are adjacent in pairs. On sm_120 this order needs fewer shared
/// wavefronts per contraction step than one with the channel tile as the
/// fastest lane index, by about a third more fused multiply-adds per cycle at
/// the tiles of BlockPanel. The lanes of an edge tile are l, l ^ 2, l ^ 4 and
/// l ^ 6, of which the one of channel tile zero leads.
template <class L>
__device__ __forceinline__ PanelTile panel_tile() {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int group = warp % L::ChannelGroups;
  const int warp_channel_tile = lane / 2 % L::WarpChannelTiles;
  const int channel_tile = group * L::WarpChannelTiles + warp_channel_tile;
  const int edge_tile =
      (warp / L::ChannelGroups) * L::WarpEdgeTiles + 2 * (lane / 8) + lane % 2;
  return {group, warp_channel_tile == 0, edge_tile * L::TE,
          channel_tile * L::TC};
}

/// Edge and channel of one item of the channel-contiguous order of
/// panel_radial.
struct RadialItem {
  int edge;
  int channel;
};

/// The channel-contiguous order of the amplitudes of a panel: tiles of eight
/// channels, an octet, by four edges, a quad, one item of a tile per lane. The
/// warps of a block form a grid of WarpOctets by WarpQuads, and warp (o, q)
/// holds the tiles of the octets o + WarpOctets a and the quads q + WarpQuads
/// b, item b + QuadItems a of each lane. Every item of a lane then lies at a
/// fixed offset from the first, so its addresses cost no registers of their
/// own.
template <class L>
struct RadialPanel {
  static constexpr int Octets = L::Channels / 8;
  static constexpr int Quads = L::Edges / 4;
  static constexpr int WarpOctets = Octets < kBlockWarps ? Octets : kBlockWarps;
  static constexpr int WarpQuads = kBlockWarps / WarpOctets;
  static_assert(L::Channels % 8 == 0 && Octets % WarpOctets == 0 &&
                    Quads % WarpQuads == 0,
                "the octets and edge quads of a panel tile its warps evenly");
  static constexpr int OctetItems = Octets / WarpOctets;
  static constexpr int QuadItems = Quads / WarpQuads;
  static constexpr int Items = OctetItems * QuadItems;

  __device__ static __forceinline__ RadialItem item(int n) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int octet = warp % WarpOctets + WarpOctets * (n / QuadItems);
    const int quad = warp / WarpOctets + WarpQuads * (n % QuadItems);
    return {4 * quad + lane / 8, 8 * octet + lane % 8};
  }
};

// === Small dense helpers ===

__device__ __forceinline__ float soft_saturate(float value, float knee) {
  const float ratio = value / knee;
  return value * rsqrtf(1.0f + ratio * ratio);
}

/// Derivative of `soft_saturate`, (1 + (x / k)^2)^(-3/2).
__device__ __forceinline__ float soft_saturate_slope(float value, float knee) {
  const float ratio = value / knee;
  const float inverse = rsqrtf(1.0f + ratio * ratio);
  return inverse * inverse * inverse;
}

__device__ __forceinline__ float block_silu(float value) {
  return value / (1.0f + __expf(-value));
}

__device__ __forceinline__ float block_silu_slope(float value) {
  const float gate = 1.0f / (1.0f + __expf(-value));
  return gate * (1.0f + value * (1.0f - gate));
}

__device__ __forceinline__ float softplus(float value) {
  return value > 0.0f ? value + log1pf(__expf(-value)) : log1pf(__expf(value));
}

/// Invoke ``body`` with the rank of the radial mode residual as a compile-time
/// constant, zero for a profile without modes and otherwise one of the ranks
/// 2, 4 and 8 that the operator admits. The loops over the modes then unroll,
/// so the loads of every channel of a tile issue ahead of its sums.
template <bool HasModes, class Body>
__device__ __forceinline__ void with_mode_rank(int rank, const Body& body) {
  if constexpr (!HasModes) {
    body(std::integral_constant<int, 0>{});
  } else if (rank == 2) {
    body(std::integral_constant<int, 2>{});
  } else if (rank == 4) {
    body(std::integral_constant<int, 4>{});
  } else {
    body(std::integral_constant<int, kMaxRadialModes>{});
  }
}

template <int Count>
__device__ __forceinline__ void load_floats(const float* source,
                                            float (&target)[Count]) {
  if constexpr (Count % 4 == 0) {
#pragma unroll
    for (int i = 0; i < Count; i += 4) {
      const float4 value = *reinterpret_cast<const float4*>(source + i);
      target[i] = value.x;
      target[i + 1] = value.y;
      target[i + 2] = value.z;
      target[i + 3] = value.w;
    }
  } else if constexpr (Count == 2) {
    const float2 value = *reinterpret_cast<const float2*>(source);
    target[0] = value.x;
    target[1] = value.y;
  } else {
#pragma unroll
    for (int i = 0; i < Count; ++i) {
      target[i] = source[i];
    }
  }
}

template <int Count>
__device__ __forceinline__ void load_floats_cached(const float* source,
                                                   float (&target)[Count]) {
  if constexpr (Count % 4 == 0) {
#pragma unroll
    for (int i = 0; i < Count; i += 4) {
      const float4 value = __ldg(reinterpret_cast<const float4*>(source + i));
      target[i] = value.x;
      target[i + 1] = value.y;
      target[i + 2] = value.z;
      target[i + 3] = value.w;
    }
  } else if constexpr (Count == 2) {
    const float2 value = __ldg(reinterpret_cast<const float2*>(source));
    target[0] = value.x;
    target[1] = value.y;
  } else {
#pragma unroll
    for (int i = 0; i < Count; ++i) {
      target[i] = __ldg(source + i);
    }
  }
}

/// Copy 16 bytes from global into shared memory. In an image compiled for
/// compute capability 8.0 or later the copy is asynchronous and completes at
/// the matching ``copy_async_wait``; an image compiled for compute
/// capability 7.x copies at once, which the barriers that follow the wait
/// order the same way.
__device__ __forceinline__ void copy_async16(float* dst, const float* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address),
               "l"(src));
#else
  *reinterpret_cast<float4*>(dst) = __ldg(reinterpret_cast<const float4*>(src));
#endif
}

/// Copy 4 bytes from global into shared memory, asynchronously in an image
/// compiled for compute capability 8.0 or later, as copy_async16 does.
__device__ __forceinline__ void copy_async4(float* dst, const float* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(address),
               "l"(src));
#else
  *dst = __ldg(src);
#endif
}

__device__ __forceinline__ void copy_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void copy_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

/// Floats of a weight chunk that a streamed product stages at a time: rows of
/// 1024 floats, sixteen rows of 64 columns.
constexpr int kWeightChunkFloats = 1024;

template <int Columns>
constexpr int weight_chunk_rows() {
  return kWeightChunkFloats / Columns;
}

/// Floats of the staging buffer of a streamed product of ``Columns`` columns:
/// two chunks.
template <int Columns>
constexpr int weight_stage_floats() {
  return 2 * weight_chunk_rows<Columns>() * Columns;
}

/// Chunks of a streamed product of depth ``Depth`` and ``Columns`` columns.
template <int Depth, int Columns>
constexpr int weight_chunks() {
  return (Depth + weight_chunk_rows<Columns>() - 1) /
         weight_chunk_rows<Columns>();
}

/// acc[i][j] += sum over k < Depth of operand[k][e0 + i] * weight[k][c0 + j],
/// the operand feature-major in shared memory with row stride ``Stride`` and
/// the weight input-major with ``Columns`` columns. A ``Resident`` weight is
/// in shared memory. Otherwise it is in global memory and streams through
/// ``stage`` (weight_stage_floats) in chunks of weight_chunk_rows, which the
/// threads of the block move cooperatively one chunk ahead of the rows the
/// product reads, so that the latency of the moves hides behind the
/// contraction; every thread of the block then takes part. A chunk of 16 rows
/// of 64 columns moves through the registers of the threads, one vector each,
/// loaded before the contraction of the chunk before and stored after it, so
/// that the contraction reads shared memory undisturbed by asynchronous copies
/// into it; a chunk of 8 rows of 128 columns contracts too briefly to hide
/// the load behind it and is copied asynchronously instead. The chunks take
/// the two buffers of the stage in turn from buffer ``First`` on. The first
/// chunk is already on its way when ``filled``, and the stage is otherwise
/// free on entry; given ``next``, the weight of a streamed product of the same
/// columns and at least one chunk of depth that follows, the last chunk of the
/// product moves the first chunk of ``next`` into the buffer after its own, so
/// that the moves of consecutive products overlap their contractions. A full
/// chunk is contracted in a loop of its compile-time length, whose loads the
/// compiler issues ahead of their products. The contraction runs in ascending
/// k with one fused multiply-add per term.
template <int Depth,
          int Stride,
          int Columns,
          int TE,
          int TC,
          bool Resident,
          int First = 0>
__device__ __forceinline__ void panel_product(const float* operand,
                                              const float* weight,
                                              float* stage,
                                              int e0,
                                              int c0,
                                              float (&acc)[TE][TC],
                                              bool filled = false,
                                              const float* next = nullptr) {
  const auto step = [&](const float* row, int k) {
    float a[TE];
    float b[TC];
    load_floats<TE>(operand + k * Stride + e0, a);
    load_floats<TC>(row + c0, b);
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      }
    }
  };
  const auto contract = [&](const float* rows, int first, int count) {
#pragma unroll 4
    for (int k = 0; k < count; ++k) {
      step(rows + k * Columns, first + k);
    }
  };
  if constexpr (Resident) {
    contract(weight, 0, Depth);
  } else if constexpr (Columns == 64) {
    constexpr int kWeightChunk = weight_chunk_rows<Columns>();
    constexpr int Chunks = weight_chunks<Depth, Columns>();
    static_assert(kWeightChunk * Columns == 4 * kBlockThreads,
                  "a chunk moves as one vector of every thread");
    const auto buffer = [&](int chunk) {
      return stage + (First + chunk) % 2 * kWeightChunk * Columns;
    };
    const int index = threadIdx.x * 4;
    const auto load = [&](const float* source, int floats) {
      return index < floats
                 ? __ldg(reinterpret_cast<const float4*>(source + index))
                 : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    const auto rows_of = [&](int chunk) {
      return min(kWeightChunk, Depth - chunk * kWeightChunk) * Columns;
    };
    if (!filled) {
      *reinterpret_cast<float4*>(buffer(0) + index) = load(weight, rows_of(0));
    }
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      // The chunk is in its buffer, and every thread has left the chunk
      // before, whose buffer the next chunk takes.
      __syncthreads();
      const bool ahead = chunk + 1 < Chunks || next != nullptr;
      const float4 moved =
          chunk + 1 < Chunks
              ? load(weight + (chunk + 1) * kWeightChunk * Columns,
                     rows_of(chunk + 1))
              : (next != nullptr ? load(next, kWeightChunk * Columns)
                                 : make_float4(0.0f, 0.0f, 0.0f, 0.0f));
      const float* rows = buffer(chunk);
      const int first = chunk * kWeightChunk;
      if (Depth - first >= kWeightChunk) {
#pragma unroll
        for (int k = 0; k < kWeightChunk; ++k) {
          step(rows + k * Columns, first + k);
        }
      } else {
        contract(rows, first, Depth - first);
      }
      if (ahead) {
        *reinterpret_cast<float4*>(buffer(chunk + 1) + index) = moved;
      }
    }
  } else {
    constexpr int kWeightChunk = weight_chunk_rows<Columns>();
    constexpr int Chunks = weight_chunks<Depth, Columns>();
    const auto buffer = [&](int chunk) {
      return stage + (First + chunk) % 2 * kWeightChunk * Columns;
    };
    const auto copy = [&](float* target, const float* source, int floats) {
      for (int index = threadIdx.x * 4; index < floats;
           index += kBlockThreads * 4) {
        copy_async16(target + index, source + index);
      }
      copy_async_commit();
    };
    const auto fill = [&](int chunk) {
      copy(buffer(chunk), weight + chunk * kWeightChunk * Columns,
           min(kWeightChunk, Depth - chunk * kWeightChunk) * Columns);
    };
    if (!filled) {
      fill(0);
    }
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      // The chunk has landed, and every thread has left the chunk before,
      // whose buffer the next copy reuses.
      copy_async_wait_all();
      __syncthreads();
      if (chunk + 1 < Chunks) {
        fill(chunk + 1);
      } else if (next != nullptr) {
        copy(buffer(Chunks), next, kWeightChunk * Columns);
      }
      const float* rows = buffer(chunk);
      const int first = chunk * kWeightChunk;
      if (Depth - first >= kWeightChunk) {
#pragma unroll
        for (int k = 0; k < kWeightChunk; ++k) {
          step(rows + k * Columns, first + k);
        }
      } else {
        contract(rows, first, Depth - first);
      }
    }
  }
}

/// Real Cartesian harmonics through ``Lmax`` in the packed layout of
/// ``build_angular_basis``.
template <int Lmax>
__device__ __forceinline__ void fill_block_basis(const EdgeGeometry& geometry,
                                                 float* basis) {
  float low[9];
  fill_angular_basis(geometry, low);
#pragma unroll
  for (int index = 0; index < 9; ++index) {
    basis[index] = low[index];
  }
  if constexpr (Lmax >= 3) {
#pragma unroll
    for (int index = 0; index < (Lmax >= 4 ? 16 : 7); ++index) {
      basis[9 + index] = high_basis_value(geometry, index);
    }
  }
}

/// Unit-direction geometry of one edge vector for the basis helpers.
__device__ __forceinline__ EdgeGeometry block_geometry(float x,
                                                       float y,
                                                       float z,
                                                       float eps) {
  EdgeGeometry geometry;
  const float square = x * x + y * y + z * z + eps * eps;
  geometry.inverse_radius = rsqrtf(square);
  geometry.radius = square * geometry.inverse_radius;
  geometry.ux = x * geometry.inverse_radius;
  geometry.uy = y * geometry.inverse_radius;
  geometry.uz = z * geometry.inverse_radius;
  geometry.seen = geometry.radius;
  geometry.envelope = 0.0f;
  geometry.source_type = 0;
  geometry.source = 0;
  return geometry;
}

/// Coordinate of the moment layout of one degree, component and channel.
template <class B>
__device__ __forceinline__ int moment_coordinate(int degree,
                                                 int component,
                                                 int channel) {
  return B::moment_offset(degree) + component * B::width_of(degree) + channel;
}

/// Degree, component and channel of an angular moment coordinate.
template <class B, int Lmax>
__device__ __forceinline__ void decode_angular(int coordinate,
                                               int& degree,
                                               int& component,
                                               int& channel) {
  degree = Lmax;
#pragma unroll
  for (int l = Lmax; l >= 1; --l) {
    if (coordinate >= B::moment_offset(l)) {
      degree = l;
      break;
    }
  }
  const int offset = coordinate - B::moment_offset(degree);
  component = offset / B::width_of(degree);
  channel = offset % B::width_of(degree);
}

// === Node maps ===
//
// The node maps are one dense layer over the saturated invariants of every
// node, pre = sat(D) W + b, whose columns hold the gate and the value of the
// SwiGLU state encoder, the neighbor term of the edge hidden layer and the
// null-mass logits (BlockLayout.node_weight of attention.py). The layer runs on
// the GEMM tiles of simt_gemm.cuh, one CTA per column tile of 32 nodes; the
// CTA of the first column tile, which holds the gate and the value, also
// finishes the state encoder and the keys of its nodes. The backward
// recomputes the first column tile, forms the cotangent of every column of the
// layer and the moment gradient of the keys there, and contracts the
// cotangent with the transpose of the layer in a second GEMM, whose epilogue
// applies the slope of the saturation.

/// GEMM tile of the node maps: 32 nodes by 64 columns, 4 x 4 accumulators per
/// lane over 32-deep slices, the small tile of the fitting network.
using NodeTile = deepmd_gemm::GemmTile<32, 64, 4, 4, 32, 2, 2, 3>;
static_assert(NodeTile::cols >= 2 * kBlockStateHidden,
              "the first column tile holds the gate and the value");

/// The A operand of the node input layer, the invariants of a run saturated
/// while they are staged.
template <bool Vector>
struct SaturatedStager : deepmd_gemm::InputStager<NodeTile, Vector> {
  using Base = deepmd_gemm::InputStager<NodeTile, Vector>;
  float knee;

  __device__ SaturatedStager(
      const float* descriptor, long row0, long nodes, int width, float knee_)
      : Base(descriptor, row0, nodes, width), knee(knee_) {}

  __device__ __forceinline__ void commit(
      float (*a)[NodeTile::rows + deepmd_gemm::kSkew]) {
#pragma unroll
    for (auto& value : this->v) {
      if constexpr (Vector) {
        value = make_float4(
            soft_saturate(value.x, knee), soft_saturate(value.y, knee),
            soft_saturate(value.z, knee), soft_saturate(value.w, knee));
      } else {
        value = soft_saturate(value, knee);
      }
    }
    Base::commit(a);
  }
};

/// The accumulators of the node input layer of the tile of a CTA, pre - b,
/// for the column tile at ``tile.col0``.
template <int Channels, int Lmax, bool HasSpin>
__device__ __forceinline__ void node_input_layer(
    const BlockArguments& block,
    const deepmd_gemm::TileIndex& tile,
    const deepmd_gemm::LaneOrigin& origin,
    deepmd_gemm::GemmStorage<NodeTile>& smem,
    deepmd_gemm::Accumulators<NodeTile>& acc) {
  using P = Profile<Channels, Lmax, HasSpin>;
  using B = BlockProfile<Channels, Lmax, HasSpin>;
  SaturatedStager<P::OutputWidth % 4 == 0> a(
      block.descriptor, tile.row0, block.edges.node_count, P::OutputWidth,
      __ldg(block.record + 1));
  deepmd_gemm::WeightStager<NodeTile> b(block.record + B::NodeWeight, tile.col0,
                                        P::OutputWidth, B::NodeWidth);
  deepmd_gemm::gemm_mainloop<NodeTile>(smem, a, b, P::OutputWidth, origin, acc);
}

/// The gate and the value of the state encoder of the nodes of a first column
/// tile, pre = acc + b, into ``gate`` (rows, 2 StateHidden); the stores
/// precede a barrier.
template <class B>
__device__ __forceinline__ void store_gate(
    const float* record,
    const deepmd_gemm::LaneOrigin& origin,
    const deepmd_gemm::Accumulators<NodeTile>& acc,
    float (*gate)[B::NodeHidden]) {
#pragma unroll
  for (int i = 0; i < NodeTile::tm; ++i) {
    const int r = deepmd_gemm::accumulator_row<NodeTile>(origin, i);
#pragma unroll
    for (int g = 0; g < NodeTile::tn / 4; ++g) {
      const int n = deepmd_gemm::accumulator_col<NodeTile>(origin, g);
      if (n < B::NodeHidden) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          gate[r][n + j] =
              acc[i][4 * g + j] + __ldg(record + B::NodeBias + n + j);
        }
      }
    }
  }
}

/// The node maps of the nodes of one tile of the run: the neighbor term of the
/// edge hidden layer and the null-mass logits from the column tile of the CTA,
/// and, in the CTA of the first column tile, the encoded state, the saturated
/// keys of the moments of block k - 1 and the divisor columns of the
/// descriptor, which every further block carries.
template <int Channels, int Lmax, bool HasSpin>
__global__ __launch_bounds__(
    NodeTile::threads,
    NodeTile::min_blocks) void block_maps_kernel(BlockArguments block) {
  using P = Profile<Channels, Lmax, HasSpin>;
  using B = BlockProfile<Channels, Lmax, HasSpin>;
  const Arguments& args = block.edges;
  const float* record = block.record;
  __shared__ __align__(16) deepmd_gemm::GemmStorage<NodeTile> smem;
  const deepmd_gemm::LaneOrigin origin = deepmd_gemm::lane_origin<NodeTile>();
  const deepmd_gemm::TileIndex tile =
      deepmd_gemm::tile_index<NodeTile>(B::NodeWidth);
  deepmd_gemm::Accumulators<NodeTile> acc;
  node_input_layer<Channels, Lmax, HasSpin>(block, tile, origin, smem, acc);

  // === Hidden-layer term and null-mass logits ===
#pragma unroll
  for (int i = 0; i < NodeTile::tm; ++i) {
    const long row =
        tile.row0 + deepmd_gemm::accumulator_row<NodeTile>(origin, i);
#pragma unroll
    for (int g = 0; g < NodeTile::tn / 4; ++g) {
      const int n =
          tile.col0 + deepmd_gemm::accumulator_col<NodeTile>(origin, g);
      if (row >= args.node_count || n < B::NodeHidden ||
          n >= B::NodeNull + B::Heads) {
        continue;
      }
      const long node = args.node_begin + row;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float value =
            acc[i][4 * g + j] + __ldg(record + B::NodeBias + n + j);
        if (n + j < B::NodeNull) {
          block.sources[node * B::SourceWidth + B::SourceHidden + n + j -
                        B::NodeHidden] = value;
        } else if (n + j < B::NodeNull + B::Heads) {
          block.centers[node * B::Heads + n + j - B::NodeNull] = value;
        }
      }
    }
  }
  if (tile.col0 != 0) {
    return;
  }

  // === Encoded state ===
  auto* gate = reinterpret_cast<float (*)[B::NodeHidden]>(&smem.a[0][0][0]);
  store_gate<B>(record, origin, acc, gate);
  __syncthreads();
  const int count = static_cast<int>(
      min(static_cast<long>(NodeTile::rows), args.node_count - tile.row0));
  for (int index = threadIdx.x; index < count * Channels;
       index += NodeTile::threads) {
    const int r = index / Channels;
    const int c = index % Channels;
    float state = 0.0f;
#pragma unroll
    for (int q = 0; q < B::StateHidden; ++q) {
      state = fmaf(block_silu(gate[r][q]) * gate[r][B::StateHidden + q],
                   __ldg(record + B::StateOutput + q * Channels + c), state);
    }
    const long node = args.node_begin + tile.row0 + r;
    block.sources[node * B::SourceWidth + B::SourceState + c] = state;
  }
  // === Divisor columns ===
  for (int index = threadIdx.x; index < count * 2; index += NodeTile::threads) {
    const long row = tile.row0 + index / 2;
    block.divisors[(args.node_begin + row) * 2 + index % 2] =
        block.descriptor[row * P::OutputWidth + P::OutputDivisor + index % 2];
  }
  // === Saturated keys of the moments of block k - 1 ===
  const float key_knee = __ldg(record + 2);
  for (int index = threadIdx.x; index < count * Lmax * B::Heads;
       index += NodeTile::threads) {
    const int r = index / (Lmax * B::Heads);
    const int degree = 1 + (index / B::Heads) % Lmax;
    const int h = index % B::Heads;
    const long node = args.node_begin + tile.row0 + r;
    const float* moments = block.previous_state + node * P::StateWidth;
    const float* projection = record + B::key_offset(degree);
    const int width = B::width_of(degree);
    float key[9];
    float norm = 0.0f;
    for (int m = 0; m < 2 * degree + 1; ++m) {
      float value = 0.0f;
      for (int c = 0; c < width; ++c) {
        value = fmaf(moments[moment_coordinate<B>(degree, m, c)],
                     __ldg(projection + c * B::Heads + h), value);
      }
      key[m] = value;
      norm = fmaf(value, value, norm);
    }
    const float scale = rsqrtf(1.0f + norm / (key_knee * key_knee));
    float* keys = block.sources + node * B::SourceWidth;
    for (int m = 0; m < 2 * degree + 1; ++m) {
      keys[(B::harmonic_offset(degree) + m) * B::Heads + h] = key[m] * scale;
    }
  }
}

/// The first half of the backward of the node maps, over the nodes of one tile
/// of the run: the cotangent of every column of the node input layer, from the
/// cotangent of the source features and of the null-mass logits, into
/// ``map_cotangent``, and the moment gradient of the keys, in the
/// node-backward layout of block k - 1. The SwiGLU state encoder pulls the
/// state cotangent back onto its gate and value, which the first column tile
/// of the layer recomputes; the hidden-layer term and the null-mass logits
/// pass theirs through.
template <int Channels, int Lmax, bool HasSpin>
__global__ __launch_bounds__(
    NodeTile::threads,
    NodeTile::min_blocks) void block_map_cotangent_kernel(BlockArguments
                                                              block) {
  using P = Profile<Channels, Lmax, HasSpin>;
  using B = BlockProfile<Channels, Lmax, HasSpin>;
  const Arguments& args = block.edges;
  const float* record = block.record;
  __shared__ __align__(16) deepmd_gemm::GemmStorage<NodeTile> smem;
  const deepmd_gemm::LaneOrigin origin = deepmd_gemm::lane_origin<NodeTile>();
  const deepmd_gemm::TileIndex tile =
      deepmd_gemm::tile_index<NodeTile>(NodeTile::cols);
  deepmd_gemm::Accumulators<NodeTile> acc;
  node_input_layer<Channels, Lmax, HasSpin>(block, tile, origin, smem, acc);
  auto* gate = reinterpret_cast<float (*)[B::NodeHidden]>(&smem.a[0][0][0]);
  auto* d_product = reinterpret_cast<float (*)[B::StateHidden]>(
      &smem.a[0][0][0] + NodeTile::rows * B::NodeHidden);
  store_gate<B>(record, origin, acc, gate);
  const int count = static_cast<int>(
      min(static_cast<long>(NodeTile::rows), args.node_count - tile.row0));

  // === Cotangent of the SwiGLU products ===
  for (int index = threadIdx.x; index < count * B::StateHidden;
       index += NodeTile::threads) {
    const int r = index / B::StateHidden;
    const int q = index % B::StateHidden;
    const float* gradient =
        block.source_gradient + (tile.row0 + r) * B::SourceWidth;
    float value = 0.0f;
    for (int c = 0; c < Channels; ++c) {
      value = fmaf(gradient[B::SourceState + c],
                   __ldg(record + B::StateOutput + q * Channels + c), value);
    }
    d_product[r][q] = value;
  }
  __syncthreads();

  // === Cotangent of the node input layer ===
  for (int index = threadIdx.x; index < count * B::NodeWidth;
       index += NodeTile::threads) {
    const int r = index / B::NodeWidth;
    const int n = index % B::NodeWidth;
    const long row = tile.row0 + r;
    const float* gradient = block.source_gradient + row * B::SourceWidth;
    float value = 0.0f;
    if (n < B::NodeHidden) {
      const int q = n % B::StateHidden;
      const float pre = gate[r][q];
      value = n < B::StateHidden
                  ? d_product[r][q] * gate[r][B::StateHidden + q] *
                        block_silu_slope(pre)
                  : d_product[r][q] * block_silu(pre);
    } else if (n < B::NodeNull) {
      value = gradient[B::SourceHidden + n - B::NodeHidden];
    } else if (n < B::NodeNull + B::Heads) {
      value = block.center_gradient[row * B::Heads + n - B::NodeNull];
    }
    block.map_cotangent[row * B::NodeWidth + n] = value;
  }

  // === Moment gradient of the keys ===
  // One warp per node. For every degree the lanes form the keys of both
  // heads, the cotangent of the raw keys through the saturation, and then
  // the moment gradient of every coordinate of the degree, with the angular
  // mass term reduced over the warp in a fixed order.
  constexpr int Warps = NodeTile::threads / kWarpSize;
  __shared__ float key_scratch[Warps][2][B::Heads * 9];
  const float key_knee = __ldg(record + 2);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int r = warp; r < count; r += Warps) {
    const long row = tile.row0 + r;
    const long node = args.node_begin + row;
    const float* moments = block.previous_state + node * P::StateWidth;
    const float angular_norm = moments[P::MomentWidth + 1];
    const float* gradient = block.source_gradient + row * B::SourceWidth;
    float* output = block.keys_gradient + row * P::StateWidth;
    float* key = key_scratch[warp][0];
    float* d_key = key_scratch[warp][1];
    // Only the angular coordinates carry a key gradient.
    for (int coordinate = lane; coordinate < P::StateWidth;
         coordinate += kWarpSize) {
      if (coordinate < P::VectorOffset || coordinate >= B::GeometricWidth) {
        output[coordinate] = 0.0f;
      }
    }
    float mass = 0.0f;
#pragma unroll
    for (int degree = 1; degree <= Lmax; ++degree) {
      const float* projection = record + B::key_offset(degree);
      const int width = B::width_of(degree);
      const int components = 2 * degree + 1;
      // key[h][m] = sum_c X[m][c] P[c][h]
      if (lane < components * B::Heads) {
        const int m = lane / B::Heads;
        const int h = lane % B::Heads;
        float value = 0.0f;
        for (int c = 0; c < width; ++c) {
          value = fmaf(moments[moment_coordinate<B>(degree, m, c)],
                       __ldg(projection + c * B::Heads + h), value);
        }
        key[h * 9 + m] = value;
      }
      __syncwarp();
      // k = b / s with s = sqrt(1 + |b|^2 / kappa^2), so
      // db = dk / s - b (b . dk) / (kappa^2 s^3).
      if (lane < components * B::Heads) {
        const int m = lane / B::Heads;
        const int h = lane % B::Heads;
        float norm = 0.0f;
        float dot = 0.0f;
        for (int n = 0; n < components; ++n) {
          const float value = key[h * 9 + n];
          norm = fmaf(value, value, norm);
          dot = fmaf(value,
                     gradient[(B::harmonic_offset(degree) + n) * B::Heads + h],
                     dot);
        }
        const float inverse = rsqrtf(1.0f + norm / (key_knee * key_knee));
        const float shrink =
            dot * inverse * inverse * inverse / (key_knee * key_knee);
        d_key[h * 9 + m] =
            gradient[(B::harmonic_offset(degree) + m) * B::Heads + h] *
                inverse -
            key[h * 9 + m] * shrink;
      }
      __syncwarp();
      for (int index = lane; index < components * width; index += kWarpSize) {
        const int m = index / width;
        const int c = index % width;
        const int coordinate = moment_coordinate<B>(degree, m, c);
        float d_moment = 0.0f;
#pragma unroll
        for (int h = 0; h < B::Heads; ++h) {
          d_moment = fmaf(d_key[h * 9 + m],
                          __ldg(projection + c * B::Heads + h), d_moment);
        }
        // The node-backward layout holds the gradient of the raw sums, the
        // normalized one times the norm.
        output[coordinate] = d_moment * angular_norm;
        mass = fmaf(d_moment, moments[coordinate], mass);
      }
      __syncwarp();
    }
    // A normalized moment X = rho n depends on the angular mass through n,
    // dn / dmass = -n^3 / 2.
    mass = warp_sum(mass);
    if (lane == 0) {
      output[P::MomentWidth + 1] = -0.5f * angular_norm * angular_norm * mass;
    }
  }
}

/// The second half of the backward of the node maps, over one tile of the
/// invariants of the run: their cotangent, the cotangent of the node input
/// layer contracted with the transpose of the layer, times the slope of the
/// saturation, into ``descriptor_gradient``, which may take the place of
/// ``descriptor``.
template <int Channels, int Lmax, bool HasSpin>
__global__ __launch_bounds__(
    NodeTile::threads,
    NodeTile::min_blocks) void block_map_input_kernel(BlockArguments block) {
  using P = Profile<Channels, Lmax, HasSpin>;
  using B = BlockProfile<Channels, Lmax, HasSpin>;
  const Arguments& args = block.edges;
  __shared__ __align__(16) deepmd_gemm::GemmStorage<NodeTile> smem;
  const deepmd_gemm::LaneOrigin origin = deepmd_gemm::lane_origin<NodeTile>();
  const deepmd_gemm::TileIndex tile =
      deepmd_gemm::tile_index<NodeTile>(P::OutputWidth);
  deepmd_gemm::Accumulators<NodeTile> acc;
  {
    deepmd_gemm::InputStager<NodeTile, true> a(block.map_cotangent, tile.row0,
                                               args.node_count, B::NodeWidth);
    deepmd_gemm::WeightTransposeStager<NodeTile> b(
        block.record + B::NodeWeight, tile.col0, B::NodeWidth, P::OutputWidth);
    deepmd_gemm::gemm_mainloop<NodeTile>(smem, a, b, B::NodeWidth, origin, acc);
  }
  const float knee = __ldg(block.record + 1);
#pragma unroll
  for (int i = 0; i < NodeTile::tm; ++i) {
    const long row =
        tile.row0 + deepmd_gemm::accumulator_row<NodeTile>(origin, i);
#pragma unroll
    for (int g = 0; g < NodeTile::tn / 4; ++g) {
      const int k =
          tile.col0 + deepmd_gemm::accumulator_col<NodeTile>(origin, g);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (row < args.node_count && k + j < P::OutputWidth) {
          const long index = row * P::OutputWidth + k + j;
          const float invariant = block.descriptor[index];
          block.descriptor_gradient[index] =
              acc[i][4 * g + j] * soft_saturate_slope(invariant, knee);
        }
      }
    }
  }
}

// === Node part of the backward ===

/// From the node-backward output at the moments of block k, the destination
/// cotangents that the edges of block k read, the cotangent of the null-mass
/// logits, and the moment gradient carried back to the first block. One warp
/// owns one node.
template <int Channels, int Lmax, bool HasSpin>
__global__ __launch_bounds__(kBlockThreads) void block_cotangent_kernel(
    BlockArguments block) {
  using P = Profile<Channels, Lmax, HasSpin>;
  using B = BlockProfile<Channels, Lmax, HasSpin>;
  const Arguments& args = block.edges;
  const float* record = block.record;
  const int lane = threadIdx.x % kWarpSize;
  const long row = static_cast<long>(blockIdx.x) * (kBlockThreads / kWarpSize) +
                   threadIdx.x / kWarpSize;
  if (row >= args.node_count) {
    return;
  }
  const long node = args.node_begin + row;
  const float* gradient = block.moment_gradient + row * P::StateWidth;
  const float angular_norm =
      block.first_state[node * P::StateWidth + P::MomentWidth + 1];
  const float* transport = block.transport + row * B::TransportWidth;
  float* cotangents = block.cotangents + node * B::CotangentWidth;
  float* carried = block.carried + node * P::StateWidth;

  // The transported keys enter the angular moments after the division, so the
  // mass gradient of the node backward, formed against the whole moment,
  // overcounts their share by -n^2/2 d.T, which the correction restores.
  float correction = 0.0f;
  for (int coordinate = P::VectorOffset + lane; coordinate < B::GeometricWidth;
       coordinate += kWarpSize) {
    int degree, component, channel;
    decode_angular<B, Lmax>(coordinate, degree, component, channel);
    const float* output = record + B::output_offset(degree);
    const int harmonic = B::harmonic_offset(degree) + component;
    float transported = 0.0f;
#pragma unroll
    for (int h = 0; h < B::Heads; ++h) {
      transported =
          fmaf(transport[harmonic * B::Heads + h],
               __ldg(output + h * B::width_of(degree) + channel), transported);
    }
    correction = fmaf(gradient[coordinate], transported, correction);
  }
  correction = 0.5f * angular_norm * warp_sum(correction);

  for (int coordinate = lane; coordinate < B::GeometricWidth;
       coordinate += kWarpSize) {
    cotangents[coordinate] = gradient[coordinate];
  }
  // dE/dT = d / n for an angular coordinate, and T = (R / den) O.
  float null_gradient[B::Heads] = {};
  for (int q = lane; q < B::KeyWidth; q += kWarpSize) {
    const int harmonic = q / B::Heads;
    const int h = q % B::Heads;
    int degree = 1;
    while (B::harmonic_offset(degree + 1) <= harmonic && degree < Lmax) {
      ++degree;
    }
    const int component = harmonic - B::harmonic_offset(degree);
    const float* output = record + B::output_offset(degree);
    float value = 0.0f;
    for (int c = 0; c < B::width_of(degree); ++c) {
      value = fmaf(gradient[moment_coordinate<B>(degree, component, c)],
                   __ldg(output + h * B::width_of(degree) + c), value);
    }
    const float denominator = transport[B::KeyWidth + h];
    const float d_transport = value / (angular_norm * denominator);
    cotangents[B::CotangentTransport + q] = d_transport;
#pragma unroll
    for (int head = 0; head < B::Heads; ++head) {
      if (head == h) {
        null_gradient[head] -= d_transport * transport[q];
      }
    }
  }
#pragma unroll
  for (int h = 0; h < B::Heads; ++h) {
    null_gradient[h] = warp_sum(null_gradient[h]);
  }
  if (lane < B::Heads) {
    const float mass_gradient = null_gradient[lane];
    cotangents[B::CotangentMass + lane] = mass_gradient;
    const float logit = block.centers[node * B::Heads + lane];
    block.center_gradient[row * B::Heads + lane] =
        mass_gradient / (1.0f + __expf(-logit));
  }
  for (int coordinate = lane; coordinate < P::StateWidth;
       coordinate += kWarpSize) {
    const float value = gradient[coordinate] +
                        (coordinate == P::MomentWidth + 1 ? correction : 0.0f);
    carried[coordinate] =
        block.accumulate_carried ? carried[coordinate] + value : value;
  }
}

// === Edge stages ===

/// The rows of a block of an edge scan -- destination rows in the forward,
/// source rows in the backward -- and the positions of their edges in the
/// panels of the block: row ``r`` of the block is row ``row[r]`` of the run,
/// whose edges, from ``begin[r]`` in the edge arrays on, take the positions
/// from ``offset[r]`` to ``offset[r + 1]``. The rows of a block are either
/// consecutive rows of the run, whose edges are then contiguous, or
/// consecutive entries of the row list of the launch
/// (BlockArguments::scan_rows), each row with edges of its own. Its
/// alignment, a whole cycle of the shared-memory banks, leaves the banks of
/// the panel arrays of a scan, which follow it, where they would lie without
/// it.
struct __align__(128) ScanRows {
  int count;
  long row[kBlockRows];
  long begin[kBlockRows];
  long offset[kBlockRows + 1];

  /// Edges of the rows of the block.
  __device__ __forceinline__ long edges() const { return offset[count]; }

  /// The row of the block that holds the edge at ``position``.
  __device__ __forceinline__ int row_of(long position) const {
    int r = 0;
    while (r + 1 < count && offset[r + 1] <= position) {
      ++r;
    }
    return r;
  }
};

/// The number of rows of the block of an edge scan, zero for a block past the
/// rows of the launch, which then retires as a whole.
__device__ __forceinline__ int scan_row_count(const BlockArguments& block) {
  const long visited = block.scan_rows != nullptr
                           ? static_cast<long>(*block.scan_row_count)
                           : block.edges.node_count;
  const long first = static_cast<long>(blockIdx.x) * block.block_rows;
  return static_cast<int>(
      max(min(static_cast<long>(block.block_rows), visited - first), 0L));
}

/// The ``count`` rows of the block of an edge scan into ``scan``, which the
/// lanes of the first warp locate together, one row each; a barrier precedes
/// their use.
__device__ __forceinline__ void locate_scan_rows(const BlockArguments& block,
                                                 int count,
                                                 ScanRows& scan) {
  if (threadIdx.x >= kWarpSize) {
    return;
  }
  const Arguments& args = block.edges;
  const long first = static_cast<long>(blockIdx.x) * block.block_rows;
  const int lane = threadIdx.x;
  long row = 0;
  long begin = 0;
  long length = 0;
  if (lane < count) {
    row = block.scan_rows != nullptr ? block.scan_rows[first + lane]
                                     : first + lane;
    begin = args.destination_row_ptr[row];
    length = args.destination_row_ptr[row + 1] - begin;
  }
  // The offset of a row is the sum of the lengths of the rows before it.
  long offset = length;
#pragma unroll
  for (int step = 1; step < kBlockRows; step *= 2) {
    const long before = __shfl_up_sync(0xffffffffu, offset, step);
    if (lane >= step) {
      offset += before;
    }
  }
  if (lane < count) {
    scan.row[lane] = row;
    scan.begin[lane] = begin;
    scan.offset[lane + 1] = offset;
  }
  if (lane == 0) {
    scan.offset[0] = 0;
    scan.count = count;
  }
}

/// The row segments of a panel. Its edges are sorted by row, so the edges of
/// every row form one segment: segment ``s`` holds the edges from
/// ``first[s]`` to ``first[s + 1]`` of row ``row[s]``, and ``count`` segments
/// cover the edges of the panel.
struct PanelRows {
  int count;
  int first[kBlockRows + 1];
  int row[kBlockRows];
};

/// Per-edge state of a panel, at the head of the shared memory of an edge
/// stage. Arrays of several values per edge are stored value-major, so that
/// the lanes of a warp, which read consecutive edges, read consecutive words.
/// The forward holds the keys of the source of every edge; the backward reads
/// the keys of its rows, the sources of all their entries, and holds the
/// radial derivatives and the vectors of its edges, which the forward does not
/// read.
template <class L>
struct __align__(16) PanelEdges {
  using B = typename L::B;
  static constexpr int Edges = L::Edges;
  static constexpr int BackwardEdges = L::Reverse ? Edges : 1;
  static constexpr int ModeEdges = L::HasModes ? Edges : 1;
  static constexpr int KeyRows = L::Reverse ? 1 : B::KeyWidth;
  long node[Edges];
  int row[Edges];
  int pair[Edges];
  int interval[Edges];
  float coordinate[Edges];
  float envelope[Edges];
  float block_envelope[Edges];
  float envelope_slope[BackwardEdges];
  float block_slope[BackwardEdges];
  /// Slope of the length the table reads with respect to the true one: the
  /// switch inside a bridging window, one elsewhere.
  float seen_slope[BackwardEdges];
  long source[Edges];
  float vector[3][BackwardEdges];
  float basis[L::BasisWidth][Edges];
  __align__(16) float modes[ModeEdges][kMaxRadialModes];
  __align__(16) float mode_slopes[L::Reverse ? ModeEdges : 1][kMaxRadialModes];
  float keys[KeyRows][Edges + 1];
  float tau[B::Heads][Edges];
  float mass[B::Heads][Edges];
  PanelRows rows;
};

/// Find the row segments of a panel of ``Edges`` edges from the rows of its
/// edges, -1 past its last edge. The first warp of the block finds them, and
/// a barrier precedes their use.
template <int Edges>
__device__ __forceinline__ void find_panel_rows(const int* rows,
                                                PanelRows& segments) {
  if (threadIdx.x >= kWarpSize) {
    return;
  }
  const int lane = threadIdx.x;
  int count = 0;
  int end = 0;
#pragma unroll
  for (int part = 0; part < Edges / kWarpSize; ++part) {
    const int e = part * kWarpSize + lane;
    const int row = rows[e];
    const bool head = row >= 0 && (e == 0 || rows[e - 1] != row);
    const unsigned heads = __ballot_sync(0xffffffffu, head);
    if (head) {
      const int s = count + __popc(heads & ((1u << lane) - 1u));
      segments.first[s] = e;
      segments.row[s] = row;
    }
    count += __popc(heads);
    end += __popc(__ballot_sync(0xffffffffu, row >= 0));
  }
  if (lane == 0) {
    segments.count = count;
    segments.first[count] = end;
  }
}

/// Consecutive lanes that share the sums of each of ``count`` values in
/// add_row_sums: the most, up to a warp, with which the values fill no more
/// than the lanes of a block.
constexpr int row_sum_parts(int count) {
  int parts = 1;
  while (parts < kWarpSize && 2 * parts * count <= kBlockThreads) {
    parts *= 2;
  }
  return parts;
}

/// The lanes that the sums of ``count`` values in ``Parts`` take in
/// add_row_sums, rounded up to whole warps: the first lane of a call that
/// follows them.
template <int Parts>
constexpr int row_sum_lanes(int count) {
  return (count * Parts + kWarpSize - 1) / kWarpSize * kWarpSize;
}

/// A value of add_row_sums that is the product of one, two or three rows of
/// shared memory at every edge, from the first factor on; a missing factor is
/// null.
struct RowProduct {
  const float* first;
  const float* second;
  const float* third;

  __device__ __forceinline__ float operator()(int e) const {
    float value = first[e];
    if (second != nullptr) {
      value *= second[e];
    }
    if (third != nullptr) {
      value *= third[e];
    }
    return value;
  }
};

/// Add to ``total[row * stride + k]`` the sum over the edges ``e`` of every
/// row of a panel of ``term(e)``, ``term = term_of(k)``, for every
/// ``k < count``. ``term_of`` forms the term of a value once, ahead of its
/// sums. Every value takes ``Parts`` consecutive lanes of a warp
/// (row_sum_parts), the lanes from ``first_lane`` of the block on (a multiple
/// of the warp width) and cyclically, so that the sums of several calls
/// without a barrier between them fall on different lanes. Part ``p`` of a
/// value sums the edges of a row whose offsets from the head of the row are
/// ``p`` modulo ``Parts``, in four interleaved partial sums that are combined
/// pairwise, and butterfly steps add the parts, so the order of every sum is
/// fixed and its dependency chains are short. The lanes of a warp read their
/// values at ``Parts`` consecutive edges, which spreads the rows of the
/// operand, whose stride is four words past a multiple of the banks, over
/// more banks.
template <int Parts, class TermOf>
__device__ __forceinline__ void add_row_sums(const PanelRows& segments,
                                             int count,
                                             const TermOf& term_of,
                                             float* total,
                                             int stride,
                                             int first_lane = 0) {
  static_assert(Parts >= 1 && Parts <= kWarpSize && kWarpSize % Parts == 0,
                "the parts of a value share a warp");
  const int lane =
      (static_cast<int>(threadIdx.x) + kBlockThreads - first_lane) %
      kBlockThreads;
  for (int base = lane - lane % kWarpSize; base < count * Parts;
       base += kBlockThreads) {
    const int slot = base + lane % kWarpSize;
    const int k = slot / Parts;
    const int part = slot % Parts;
    const bool active = k < count;
    const auto term = term_of(active ? k : 0);
    for (int s = 0; s < segments.count; ++s) {
      const int tail = segments.first[s + 1];
      float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (active) {
        int e = segments.first[s] + part;
        for (; e + 3 * Parts < tail; e += 4 * Parts) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            partial[i] += term(e + i * Parts);
          }
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          if (e + i * Parts < tail) {
            partial[i] += term(e + i * Parts);
          }
        }
      }
      float sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
#pragma unroll
      for (int offset = 1; offset < Parts; offset *= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
      }
      if (active && part == 0) {
        total[segments.row[s] * stride + k] += sum;
      }
    }
  }
}

/// The sum of ``value`` over the four lanes of a warp that share the edges of
/// a lane (panel_tile), which every one of them receives: a butterfly whose
/// two steps add commuted operands on partner lanes, so that all four hold
/// the same bits.
__device__ __forceinline__ float tile_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 4);
  return value;
}

/// The sums over the channel groups of the partial sums ``partial`` of
/// ``count`` values of every edge, ``(ChannelGroups, count, Edges)``, into
/// ``total``, ``(count, Edges)``, in group order.
template <class L>
__device__ __forceinline__ void close_channel_groups(const float* partial,
                                                     float* total,
                                                     int count) {
  for (int index = threadIdx.x; index < count * L::Edges;
       index += kBlockThreads) {
    float value = 0.0f;
#pragma unroll
    for (int group = 0; group < L::ChannelGroups; ++group) {
      value += partial[group * count * L::Edges + index];
    }
    total[index] = value;
  }
}

/// The cotangent of the fields of every edge of a panel through one edge
/// map, ``(Fields, Edges)`` into ``total``, which it replaces or, with
/// ``Add``, accumulates: the sum over the channels of the cotangents in the
/// channel rows of the operand times the field rows of the map, the leading
/// ``Fields`` rows of ``map`` (Operand, Channels), in shared memory when
/// ``Resident`` and in global memory otherwise. One thread forms the sum of
/// one field and edge, the edges of a warp consecutive, so that the lanes of a
/// warp read consecutive words of the operand and one word of the map; the
/// sum runs over the channels in four interleaved partial sums, combined
/// pairwise, so its order is fixed.
template <class L, bool Resident, bool Add>
__device__ __forceinline__ void field_cotangent(const float* operand,
                                                const float* map,
                                                float* total) {
  using B = typename L::B;
  constexpr int Channels = L::Channels;
  for (int index = threadIdx.x; index < B::Fields * L::Edges;
       index += kBlockThreads) {
    const float* column = operand + B::Fields * L::Stride + index % L::Edges;
    const float* row = map + index / L::Edges * Channels;
    float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int c = 0; c < Channels; c += 4) {
      float weight[4];
      if constexpr (Resident) {
        load_floats<4>(row + c, weight);
      } else {
        load_floats_cached<4>(row + c, weight);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        partial[i] = fmaf(column[(c + i) * L::Stride], weight[i], partial[i]);
      }
    }
    const float sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    total[index] = Add ? total[index] + sum : sum;
  }
}

/// The degree of a non-scalar harmonic in the key layout.
template <int Lmax>
__device__ __forceinline__ int harmonic_degree(int harmonic) {
  int degree = 1;
#pragma unroll
  for (int l = 2; l <= Lmax; ++l) {
    degree += harmonic >= l * l - 1 ? 1 : 0;
  }
  return degree;
}

/// Request the bond vectors and the source indices of the edges of the rows
/// of a block into L2 when the launch streams its bonds (see
/// `Arguments::prefetch_bonds`), dealing their lines to the threads of the
/// block, so that its panels find them in L2 as the rows of the first-block
/// scans do (prefetch_row_bonds).
template <typename index_t>
__device__ __forceinline__ void prefetch_block_bonds(const Arguments& args,
                                                     const ScanRows& scan) {
  if (args.prefetch_bonds) {
    for (int r = 0; r < scan.count; ++r) {
      const long begin = scan.begin[r];
      const long length = scan.offset[r + 1] - scan.offset[r];
      prefetch_lines<CacheLevel::L2>(args.edge_vec + 3 * begin, 3 * length,
                                     threadIdx.x, kBlockThreads);
      prefetch_lines<CacheLevel::L2>(
          reinterpret_cast<const float*>(
              static_cast<const index_t*>(args.edge_index) + begin),
          length * static_cast<long>(sizeof(index_t) / sizeof(float)),
          threadIdx.x, kBlockThreads);
    }
  }
}

/// The first-block envelope of an edge, the length its radial table reads,
/// and, for the backward, their slopes with respect to the true length.
struct BridgedEnvelope {
  float envelope;
  float seen;
  float slope;
  float seen_slope;
};

/// The envelope and the table length of the first block for an edge of length
/// ``radius`` between nodes of the types ``center_type`` and
/// ``source_type``, as load_geometry forms them: inside the bridging window of
/// the pair the table reads the clamped length and the envelope carries the
/// switch, and the source gate of a bridged descriptor of several blocks
/// multiplies the envelope of every edge. Under the gate the slope of the
/// switch enters through the gates of the other edges of the source, which
/// gate_closure_kernel adds, so the slope keeps c3'(r̃) w^2 alone (see
/// radial_chain). A bridged launch of the further blocks always carries the
/// source gate (evaluate_further_blocks), so the slope of a window without
/// one is not formed here.
__device__ __forceinline__ BridgedEnvelope
bridged_envelope(const Arguments& args,
                 float radius,
                 int center_type,
                 int source_type,
                 long source) {
  BridgedEnvelope result{c3_envelope(radius, args.rcut), radius,
                         c3_envelope_derivative(radius, args.rcut), 1.0f};
  if (!args.bridged) {
    return result;
  }
  float weight = 1.0f;
  const float contact = __ldg(args.contact_radius + center_type) +
                        __ldg(args.contact_radius + source_type);
  if (radius < args.f_outer * contact) {
    const float t = window_parameter(__fdividef(radius, contact), args.f_inner,
                                     args.inverse_f_width);
    weight = bridging_switch(t);
    result.seen = clamped_length(t, contact, args.f_inner, args.f_outer);
    result.envelope = c3_envelope(result.seen, args.rcut) * weight;
    result.slope =
        c3_envelope_derivative(result.seen, args.rcut) * weight * weight;
    result.seen_slope = weight;
  }
  if (args.source_gate != nullptr) {
    const float gate = leave_one_out_gate(args.source_gate[source], weight);
    result.envelope *= gate;
    result.slope *= gate;
  }
  return result;
}

/// Load the geometry, the harmonics and the mode profiles of the edges of a
/// panel, the node row that every edge reads, and the fields of every edge
/// into the leading rows of the operand. ``Reverse`` reads every entry as the
/// edge its row emits: the vector reversed and the entry's node as the
/// destination. The node row of an edge is, in the forward, the keys of its
/// source and, in the backward, the destination cotangents of the transport
/// and of the denominators of the entry's node, which ``gathered``
/// (count, Edges + 1) receives value-major. One lane per edge evaluates its
/// geometry and requests the node row of the edge with copies of one word as
/// soon as it reads the entry, so that the row lands while the lane evaluates
/// the geometry; then every lane forms the fields. The panel holds the edges
/// of the rows of ``scan`` from position ``panel_begin`` on.
template <class L, typename index_t>
__device__ __forceinline__ void load_panel_edges(
    const BlockArguments& block,
    const ScanRows& scan,
    long panel_begin,
    PanelEdges<L>& edges,
    const float (*row_keys)[L::B::KeyWidth],
    float* operand,
    float* gathered) {
  using B = typename L::B;
  constexpr bool Reverse = L::Reverse;
  constexpr int Gathered = Reverse ? B::TransportWidth : B::KeyWidth;
  constexpr int GatherStride = L::Edges + 1;
  const Arguments& args = block.edges;
  const float* record = block.record;
  const float cutoff = __ldg(record);
  // === Geometry, envelopes, harmonics and mode profiles of every edge ===
  for (int e = threadIdx.x; e < L::Edges; e += kBlockThreads) {
    const long panel_position = panel_begin + e;
    const bool active = panel_position < scan.edges();
    const int row = active ? scan.row_of(panel_position) : -1;
    edges.row[e] = row;
    float basis[L::BasisWidth] = {};
    if (active) {
      const long position =
          scan.begin[row] + (panel_position - scan.offset[row]);
      const long row_node = args.node_begin + scan.row[row];
      const long entry = static_cast<long>(
          static_cast<const index_t*>(args.edge_index)[position]);
      // The forward reads an edge into its destination row from its source;
      // the backward reads the entry of a source row as the edge into the
      // entry's node.
      const long source = Reverse ? row_node : entry;
      const long destination = Reverse ? entry : row_node;
      const float* node_row = Reverse ? block.cotangents +
                                            entry * B::CotangentWidth +
                                            B::CotangentTransport
                                      : block.sources + source * B::SourceWidth;
#pragma unroll
      for (int q = 0; q < Gathered; ++q) {
        copy_async4(gathered + q * GatherStride + e, node_row + q);
      }
      edges.node[e] = entry;
      edges.source[e] = source;
      edges.pair[e] =
          static_cast<int>(args.atype[destination]) * args.type_count +
          static_cast<int>(args.atype[source]);
      const float sign = Reverse ? -1.0f : 1.0f;
      const float x = sign * args.edge_vec[position * 3 + 0];
      const float y = sign * args.edge_vec[position * 3 + 1];
      const float z = sign * args.edge_vec[position * 3 + 2];
      const EdgeGeometry geometry = block_geometry(x, y, z, args.eps);
      // The envelope and the radial table of the first block, under the
      // bridging window of its pair and the source gate (load_geometry); the
      // envelope of the block reads the true length.
      const BridgedEnvelope first = bridged_envelope(
          args, geometry.radius, static_cast<int>(args.atype[destination]),
          static_cast<int>(args.atype[source]), source);
      edges.envelope[e] = first.envelope;
      edges.block_envelope[e] = c3_envelope(geometry.radius, cutoff);
      if constexpr (Reverse) {
        edges.vector[0][e] = x;
        edges.vector[1][e] = y;
        edges.vector[2][e] = z;
        edges.envelope_slope[e] = first.slope;
        edges.seen_slope[e] = first.seen_slope;
        edges.block_slope[e] = c3_envelope_derivative(geometry.radius, cutoff);
      }
      const TableLocation location = locate_table<true>(
          first.seen, args.table_stride, args.table_max, args.interval_count);
      edges.interval[e] = location.index;
      edges.coordinate[e] = location.coordinate;
      fill_block_basis<L::Lmax>(geometry, basis);
      with_mode_rank<L::HasModes>(args.radial_modes, [&](auto rank) {
        constexpr int Modes = decltype(rank)::value;
        const TableRow table =
            table_row(args.table, location, args.table_width);
#pragma unroll
        for (int mode = 0; mode < Modes; ++mode) {
          const float2 value = evaluate_table_with_derivative(
              table, L::Channels + mode, location.coordinate, false);
          edges.modes[e][mode] = value.x;
          if constexpr (Reverse) {
            edges.mode_slopes[e][mode] = value.y;
          }
        }
      });
    } else {
#pragma unroll
      for (int q = 0; q < Gathered; ++q) {
        gathered[q * GatherStride + e] = 0.0f;
      }
      edges.node[e] = 0;
      edges.source[e] = -1;
      edges.pair[e] = 0;
      edges.interval[e] = 0;
      edges.coordinate[e] = 0.0f;
      edges.envelope[e] = 0.0f;
      edges.block_envelope[e] = 0.0f;
      if constexpr (Reverse) {
        edges.envelope_slope[e] = 0.0f;
        edges.seen_slope[e] = 0.0f;
        edges.block_slope[e] = 0.0f;
        edges.vector[0][e] = edges.vector[1][e] = edges.vector[2][e] = 0.0f;
      }
    }
#pragma unroll
    for (int index = 0; index < L::BasisWidth; ++index) {
      edges.basis[index][e] = basis[index];
    }
  }
  copy_async_commit();
  copy_async_wait_all();
  __syncthreads();
  find_panel_rows<L::Edges>(edges.row, edges.rows);

  // === Fields along the bond towards the centre ===
  // B_l(-u) = (-1)^l B_l(u).
  for (int index = threadIdx.x; index < L::Edges * B::Fields;
       index += kBlockThreads) {
    const int e = index % L::Edges;
    const int q = index / L::Edges;
    const int degree = 1 + q / B::Heads;
    const int h = q % B::Heads;
    const float reflection = degree % 2 == 0 ? 1.0f : -1.0f;
    float field = 0.0f;
    for (int m = 0; m < 2 * degree + 1; ++m) {
      const int key = (B::harmonic_offset(degree) + m) * B::Heads + h;
      float value;
      if constexpr (Reverse) {
        value = row_keys[max(edges.row[e], 0)][key];
      } else {
        value = edges.keys[key][e];
      }
      field = fmaf(edges.basis[degree * degree + m][e], value, field);
    }
    operand[q * L::Stride + e] = reflection * field;
  }
}

/// The enveloped first-block amplitudes of the edges of a panel into the
/// channel rows of the operand, which a barrier then hands to the tiles of the
/// lanes, and, in the backward, the radial derivatives of the items of a lane
/// into ``slope``. The amplitudes are evaluated in channel-contiguous order:
/// lane ``l`` of a warp evaluates channel ``8 k + l % 8`` of edge
/// ``4 m + l / 8``, so that the eight lanes of an edge read 128 contiguous
/// bytes of its table row, its pair film and its mode mixing, which the tile
/// of a lane in the products, whose channels are contiguous per lane, would
/// read at a stride (RadialPanel). The items of a lane are evaluated in
/// batches of kAmplitudeBatch, all loads of a batch ahead of its first
/// evaluation; an edge past the panel reads the first interval and the first
/// pair, whose finite coefficients its zero amplitude then replaces.
template <class L>
__device__ __forceinline__ void panel_radial(
    const BlockArguments& block,
    const PanelEdges<L>& edges,
    float* operand,
    float (&slope)[RadialPanel<L>::Items]) {
  using R = RadialPanel<L>;
  constexpr int Channels = L::Channels;
  const Arguments& args = block.edges;
  constexpr int Batch = R::Items < kAmplitudeBatch ? R::Items : kAmplitudeBatch;
  static_assert(R::Items % Batch == 0, "the batches cover the items of a lane");
  with_mode_rank<L::HasModes>(args.radial_modes, [&](auto rank) {
    constexpr int Modes = decltype(rank)::value;
#pragma unroll
    for (int n0 = 0; n0 < R::Items; n0 += Batch) {
      TableCoefficients coefficients[Batch];
      float2 film[Batch];
#pragma unroll
      for (int b = 0; b < Batch; ++b) {
        const RadialItem item = R::item(n0 + b);
        const TableRow row =
            table_row(args.table, {edges.interval[item.edge], 0.0f, false},
                      args.table_width);
        coefficients[b] = table_coefficients(row, item.channel);
        film[b] = __ldg(reinterpret_cast<const float2*>(args.pair_film) +
                        static_cast<long>(edges.pair[item.edge]) * Channels +
                        item.channel);
      }
#pragma unroll
      for (int b = 0; b < Batch; ++b) {
        const RadialItem item = R::item(n0 + b);
        const int e = item.edge;
        const bool active = edges.row[e] >= 0;
        const float x = edges.coordinate[e];
        const float envelope = edges.envelope[e];
        const float* mixing =
            args.pair_mixing +
            (static_cast<long>(edges.pair[e]) * Channels + item.channel) *
                Modes;
        float value;
        if constexpr (L::Reverse) {
          const float2 radial =
              spline_value_with_derivative(coefficients[b], x, false);
          value = fmaf(radial.x, film[b].x, film[b].y);
          float derivative = radial.y * film[b].x;
          if constexpr (Modes > 0) {
            accumulate_modes_with_derivative(mixing, edges.modes[e],
                                             edges.mode_slopes[e], Modes, value,
                                             derivative);
          }
          if (args.bridged) {
            derivative *= edges.seen_slope[e];
          }
          slope[n0 + b] = active ? fmaf(edges.envelope_slope[e], value,
                                        envelope * derivative)
                                 : 0.0f;
        } else {
          value = fmaf(spline_value(coefficients[b], x), film[b].x, film[b].y);
          if constexpr (Modes > 0) {
            accumulate_modes(mixing, edges.modes[e], Modes, value);
          }
        }
        operand[(L::B::Fields + item.channel) * L::Stride + e] =
            active ? value * envelope : 0.0f;
      }
    }
  });
}

/// The radial cotangent of every edge of a panel, the sum over its channels of
/// the amplitude cotangent times the radial derivative of the amplitude, into
/// ``total`` (Edges). The amplitude cotangents are in the channel rows of the
/// operand and the derivatives in the items of the lanes (panel_radial). The
/// eight channels of an octet are summed by a butterfly over the lanes of the
/// octet into ``partial`` (Octets, Edges), and the octets in ascending order;
/// a barrier precedes the use of ``total``.
template <class L>
__device__ __forceinline__ void radial_cotangent(
    const float* operand,
    const float (&slope)[RadialPanel<L>::Items],
    float* partial,
    float* total) {
  using R = RadialPanel<L>;
#pragma unroll
  for (int n = 0; n < R::Items; ++n) {
    const RadialItem item = R::item(n);
    float value =
        operand[(L::B::Fields + item.channel) * L::Stride + item.edge] *
        slope[n];
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    value += __shfl_xor_sync(0xffffffffu, value, 2);
    value += __shfl_xor_sync(0xffffffffu, value, 4);
    if (threadIdx.x % 8 == 0) {
      partial[(item.channel / 8) * L::Edges + item.edge] = value;
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < L::Edges; e += kBlockThreads) {
    float value = 0.0f;
#pragma unroll
    for (int octet = 0; octet < R::Octets; ++octet) {
      value += partial[octet * L::Edges + e];
    }
    total[e] = value;
  }
}

/// The values of the tile of a lane in the channel rows of the operand.
template <class L>
__device__ __forceinline__ void load_tile(const float* operand,
                                          int e0,
                                          int c0,
                                          float (&values)[L::TE][L::TC]) {
#pragma unroll
  for (int i = 0; i < L::TE; ++i) {
#pragma unroll
    for (int j = 0; j < L::TC; ++j) {
      values[i][j] = operand[(L::B::Fields + c0 + j) * L::Stride + e0 + i];
    }
  }
}

/// The source terms that a product of the tile of a lane adds to its
/// accumulators: the ``TC`` source features from ``offset`` of the source of
/// every edge of the tile, zero past the panel. A product requests them ahead
/// of its contraction and adds them after it, so that the contraction does not
/// wait for them.
template <class L>
__device__ __forceinline__ void load_source_terms(
    const BlockArguments& block,
    const PanelEdges<L>& edges,
    int offset,
    int e0,
    int c0,
    float (&terms)[L::TE][L::TC]) {
  using B = typename L::B;
#pragma unroll
  for (int i = 0; i < L::TE; ++i) {
    const long source = edges.source[e0 + i];
    if (source >= 0) {
      load_floats_cached<L::TC>(
          block.sources + source * B::SourceWidth + offset + c0, terms[i]);
    } else {
#pragma unroll
      for (int j = 0; j < L::TC; ++j) {
        terms[i][j] = 0.0f;
      }
    }
  }
}

/// The amplitudes of the tile of a lane from the channel rows of the operand,
/// where panel_radial left them, replaced there by their saturation. The
/// transport weight of every edge, a sum over its channels, is summed over
/// the channel tiles of the warp into the partial sum of its channel group,
/// ``(ChannelGroups, Heads, Edges)``, which close_channel_groups completes.
template <class L>
__device__ __forceinline__ void panel_amplitudes(const float* record,
                                                 int e0,
                                                 int c0,
                                                 float (&phi)[L::TE][L::TC],
                                                 float* operand,
                                                 const PanelTile& tile,
                                                 float* tau_partial) {
  using B = typename L::B;
  const float knee = __ldg(record + 3);
  load_tile<L>(operand, e0, c0, phi);
#pragma unroll
  for (int i = 0; i < L::TE; ++i) {
    float tau[B::Heads] = {};
#pragma unroll
    for (int j = 0; j < L::TC; ++j) {
      const int c = c0 + j;
      const float saturated = soft_saturate(phi[i][j], knee);
      operand[(B::Fields + c) * L::Stride + e0 + i] = saturated;
#pragma unroll
      for (int h = 0; h < B::Heads; ++h) {
        tau[h] =
            fmaf(saturated, __ldg(record + B::EdgeTransport + c * B::Heads + h),
                 tau[h]);
      }
    }
#pragma unroll
    for (int h = 0; h < B::Heads; ++h) {
      const float total = tile_sum(tau[h]);
      if (tile.leader) {
        tau_partial[(tile.group * B::Heads + h) * L::Edges + e0 + i] = total;
      }
    }
  }
}

/// Stage both edge maps of a block record in shared memory: [W_F; W_phi] and
/// [G; W_O], each (Operand, Channels) and contiguous in the record.
template <class B>
__device__ __forceinline__ void stage_edge_maps(const float* record,
                                                float* hidden,
                                                float* logit) {
  constexpr int Floats = B::Operand * B::P::C0;
  for (int index = threadIdx.x * 4; index < Floats;
       index += kBlockThreads * 4) {
    *reinterpret_cast<float4*>(hidden + index) =
        __ldg(reinterpret_cast<const float4*>(record + B::EdgeField + index));
    *reinterpret_cast<float4*>(logit + index) =
        __ldg(reinterpret_cast<const float4*>(record + B::EdgeGain + index));
  }
}

/// Shared memory of the forward scan: the per-edge state, the operand, the
/// partial transport weights of the channel groups and the row sums, then
/// both edge maps when ``Resident`` and the stage of the streamed weights
/// otherwise.
template <class L, bool Resident>
constexpr size_t block_forward_shared_bytes() {
  using B = typename L::B;
  return sizeof(PanelEdges<L>) +
         sizeof(float) *
             (B::Operand * L::Stride + L::ChannelGroups * B::Heads * L::Edges +
              kBlockRows * (B::GeometricWidth + B::TransportWidth) +
              (Resident ? 2 * B::Operand * L::Channels
                        : weight_stage_floats<L::Channels>()));
}

/// Whether the forward scan stages its edge maps in shared memory, which it
/// does when they fit the budget of residency beside the panel.
template <class L>
constexpr bool forward_resident() {
  return block_forward_shared_bytes<L, true>() <= kResidentSharedBudget;
}

/// The edge scan of a further block over a block of destination rows: the
/// moments of the first block corrected by the edges inside the cutoff of the
/// block, the transported keys, and their denominators, in the state layout
/// of the first block.
template <int Channels, int Lmax, bool HasModes, bool HasSpin, typename index_t>
__global__ __launch_bounds__(
    kBlockThreads,
    BlockPanel<Channels, Lmax, false>::
        MinBlocks) void block_forward_kernel(BlockArguments block) {
  using P = Profile<Channels, Lmax, HasSpin>;
  using L = PanelLayout<Channels, Lmax, HasSpin, HasModes, false>;
  using B = typename L::B;
  using R = RadialPanel<L>;
  constexpr int TE = L::TE;
  constexpr int TC = L::TC;
  constexpr int Stride = L::Stride;
  // Row sums: the corrections of the geometric moments, the softmax
  // denominators, then the transported keys.
  constexpr int Sums = B::GeometricWidth + B::TransportWidth;
  constexpr int DenominatorSum = B::GeometricWidth;
  constexpr int TransportSum = B::GeometricWidth + B::Heads;
  const Arguments& args = block.edges;
  const float* record = block.record;
  const float logit_cap = __ldg(record + 4);
  const float null_floor = __ldg(record + 5);
  const PanelTile tile = panel_tile<L>();
  const int c0 = tile.c0;
  const int e0 = tile.e0;
  __shared__ ScanRows scan;
  const int rows = scan_row_count(block);
  if (rows == 0) {
    return;
  }
  locate_scan_rows(block, rows, scan);

  constexpr bool Resident = forward_resident<L>();
  extern __shared__ __align__(16) float block_shared[];
  auto& edges = *reinterpret_cast<PanelEdges<L>*>(block_shared);
  float* operand = block_shared + sizeof(edges) / sizeof(float);
  float* tau_partial = operand + B::Operand * Stride;
  auto* sums = reinterpret_cast<float (*)[Sums]>(
      tau_partial + L::ChannelGroups * B::Heads * L::Edges);
  const float* hidden_map = record + B::EdgeField;
  const float* logit_map = record + B::EdgeGain;
  float* const weights = reinterpret_cast<float*>(sums) + kBlockRows * Sums;
  if constexpr (Resident) {
    stage_edge_maps<B>(record, weights, weights + B::Operand * Channels);
    hidden_map = weights;
    logit_map = weights + B::Operand * Channels;
  }
  for (int index = threadIdx.x; index < kBlockRows * Sums;
       index += kBlockThreads) {
    sums[index / Sums][index % Sums] = 0.0f;
  }
  __syncthreads();
  const long edge_end = scan.edges();
  prefetch_block_bonds<index_t>(args, scan);

  for (long panel = 0; panel < edge_end; panel += L::Edges) {
    __syncthreads();
    load_panel_edges<L, index_t>(block, scan, panel, edges, nullptr, operand,
                                 edges.keys[0]);
    __syncthreads();
    float no_slope[R::Items];
    panel_radial<L>(block, edges, operand, no_slope);
    __syncthreads();
    float phi[TE][TC];
    panel_amplitudes<L>(record, e0, c0, phi, operand, tile, tau_partial);
    __syncthreads();
    close_channel_groups<L>(tau_partial, edges.tau[0], B::Heads);

    // === Hidden layer: [F, sat(phi)] [W_F; W_phi] + h_j ===
    float terms[TE][TC];
    load_source_terms<L>(block, edges, B::SourceHidden, e0, c0, terms);
    float acc[TE][TC] = {};
    panel_product<B::Operand, Stride, Channels, TE, TC, Resident>(
        operand, hidden_map, weights, e0, c0, acc, panel != 0, logit_map);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        operand[(B::Fields + c0 + j) * Stride + e0 + i] =
            block_silu(acc[i][j] + terms[i][j]);
      }
    }
    __syncthreads();

    // === Logit: [F, silu(hid)] [G; W_O] + s_j ===
    load_source_terms<L>(block, edges, B::SourceState, e0, c0, terms);
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        acc[i][j] = 0.0f;
      }
    }
    panel_product<B::Operand, Stride, Channels, TE, TC, Resident,
                  weight_chunks<B::Operand, Channels>() % 2>(
        operand, logit_map, weights, e0, c0, acc, true,
        panel + L::Edges < edge_end ? hidden_map : nullptr);
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        acc[i][j] += terms[i][j];
      }
    }
    __syncthreads();

    // === Correction amplitudes and softmax masses ===
#pragma unroll
    for (int i = 0; i < TE; ++i) {
      const int e = e0 + i;
      const float block_envelope = edges.block_envelope[e];
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        const int c = c0 + j;
        const float logit = acc[i][j];
        operand[(B::Fields + c) * Stride + e] =
            phi[i][j] * block_envelope * tanhf(logit);
        if (c < B::Heads) {
          const float weight = edges.envelope[e] * block_envelope;
          edges.mass[c][e] =
              weight * weight * __expf(logit_cap * tanhf(logit / logit_cap));
        }
      }
    }
    __syncthreads();

    // === Row sums ===
    add_row_sums<row_sum_parts(Sums)>(
        edges.rows, Sums,
        [&](int component) {
          if (component < Channels) {
            return RowProduct{operand + (B::Fields + component) * Stride,
                              nullptr, nullptr};
          }
          if (component < B::GeometricWidth) {
            int degree, harmonic, channel;
            decode_angular<B, Lmax>(component, degree, harmonic, channel);
            return RowProduct{edges.envelope,
                              operand + (B::Fields + channel) * Stride,
                              edges.basis[degree * degree + harmonic]};
          }
          if (component < TransportSum) {
            return RowProduct{edges.mass[component - DenominatorSum], nullptr,
                              nullptr};
          }
          const int q = component - TransportSum;
          const int head = q % B::Heads;
          return RowProduct{edges.mass[head], edges.tau[head], edges.keys[q]};
        },
        &sums[0][0], Sums);
  }
  __syncthreads();

  // === Moments of the block in the state layout of the first block ===
  for (int index = threadIdx.x; index < rows * P::StateWidth;
       index += kBlockThreads) {
    const int r = index / P::StateWidth;
    const int coordinate = index % P::StateWidth;
    const long row = scan.row[r];
    const long node = args.node_begin + row;
    const float* first = block.first_state + node * P::StateWidth;
    float value = first[coordinate];
    if (coordinate < Channels) {
      value = fmaf(sums[r][coordinate], first[P::MomentWidth], value);
    } else if (coordinate < B::GeometricWidth) {
      int degree, component, channel;
      decode_angular<B, Lmax>(coordinate, degree, component, channel);
      const float* output = record + B::output_offset(degree);
      const int harmonic = B::harmonic_offset(degree) + component;
      float transported = 0.0f;
#pragma unroll
      for (int h = 0; h < B::Heads; ++h) {
        const float denominator = sums[r][DenominatorSum + h] +
                                  softplus(block.centers[node * B::Heads + h]) +
                                  null_floor;
        transported = fmaf(
            sums[r][TransportSum + harmonic * B::Heads + h] / denominator,
            __ldg(output + h * B::width_of(degree) + channel), transported);
      }
      value = fmaf(sums[r][coordinate], first[P::MomentWidth + 1], value) +
              transported;
    }
    block.state_out[row * P::StateWidth + coordinate] = value;
  }
  for (int index = threadIdx.x; index < rows * B::TransportWidth;
       index += kBlockThreads) {
    const int r = index / B::TransportWidth;
    const int q = index % B::TransportWidth;
    const long row = scan.row[r];
    const long node = args.node_begin + row;
    const int h = q < B::KeyWidth ? q % B::Heads : q - B::KeyWidth;
    const float denominator = sums[r][DenominatorSum + h] +
                              softplus(block.centers[node * B::Heads + h]) +
                              null_floor;
    block.transport[row * B::TransportWidth + q] =
        q < B::KeyWidth ? sums[r][TransportSum + q] / denominator : denominator;
  }
}

/// Per-edge arrays of the backward scan after the operand, each value-major:
/// the radial and block-envelope cotangents of the amplitudes, their part
/// that the source gate reads, the field cotangents, the correction
/// amplitudes of the channels of degree one, the destination cotangents of
/// the transport and of the denominators (padded rows), and the
/// transport-weight cotangents, the key products and the weight cotangents of
/// the masses. The partial sums of the channel groups, of the per-head sums
/// and of the octets of the radial cotangents (radial_cotangent), form a
/// section of their own, which a scan that streams its weights keeps in the
/// stage of the weights: they are written and read between the products only.
template <class L>
struct BackwardPanel {
  using B = typename L::B;
  static constexpr int Edges = L::Edges;
  static constexpr int Radial = 0;
  static constexpr int Envelope = Radial + Edges;
  /// Sum over the channels of the amplitude cotangents times the amplitudes,
  /// for the source gate of a bridged descriptor.
  static constexpr int GateTotal = Envelope + Edges;
  static constexpr int FieldGradient = GateTotal + Edges;
  static constexpr int DeltaLow = FieldGradient + B::Fields * Edges;
  static constexpr int TransportCotangent = DeltaLow + B::P::C1 * Edges;
  static constexpr int TransportCotangentWidth = B::KeyWidth + B::Heads;
  // The padded rows end on a 16-byte boundary, as every array does.
  static constexpr int TauGradient = aligned_floats(
      TransportCotangent + TransportCotangentWidth * (Edges + 1));
  static constexpr int Product = TauGradient + B::Heads * Edges;
  static constexpr int MassWeight = Product + B::Heads * Edges;
  static constexpr int Floats = MassWeight + B::Heads * Edges;

  static constexpr int GroupPartial = 0;
  static constexpr int OctetPartial =
      GroupPartial + L::ChannelGroups * B::Heads * Edges;
  static constexpr int PartialFloats =
      OctetPartial + RadialPanel<L>::Octets * Edges;
};

/// Shared memory of the backward scan: the per-edge state, the operand, the
/// per-edge arrays of BackwardPanel, the keys of the rows and the row sums,
/// then the partial sums of the channel groups and both edge maps with the
/// transposes of their square parts when ``Resident``, and otherwise the
/// stage of the streamed weights, which holds the partial sums between the
/// products.
template <class L, bool Resident>
constexpr size_t block_backward_shared_bytes() {
  using B = typename L::B;
  using A = BackwardPanel<L>;
  constexpr int Channels = L::Channels;
  constexpr int Stage = weight_stage_floats<Channels>();
  return sizeof(PanelEdges<L>) +
         sizeof(float) *
             (B::Operand * L::Stride + A::Floats +
              kBlockRows * (B::KeyWidth + B::SourceWidth + 13) +
              (Resident
                   ? A::PartialFloats + 2 * B::Operand * Channels +
                         2 * Channels * Channels
                   : (Stage > A::PartialFloats ? Stage : A::PartialFloats)));
}

/// Whether the backward scan stages its edge maps in shared memory.
template <class L>
constexpr bool backward_resident() {
  return block_backward_shared_bytes<L, true>() <= kResidentSharedBudget;
}

/// The backward of a further block over a block of source rows: every entry
/// of a row is read as the edge the row emits, the forward of the edge is
/// recomputed from the source features of the row, and its cotangents follow
/// from the destination cotangents of the entry's node. The row accumulates
/// the cotangent of its source features and its share of the force and the
/// virial; the destination share of the force lands in fixed point.
template <int Channels, int Lmax, bool HasModes, bool HasSpin, typename index_t>
__global__ __launch_bounds__(
    kBlockThreads,
    BlockPanel<Channels, Lmax, true>::
        MinBlocks) void block_backward_kernel(BlockArguments block) {
  using P = Profile<Channels, Lmax, HasSpin>;
  using L = PanelLayout<Channels, Lmax, HasSpin, HasModes, true>;
  using B = typename L::B;
  using A = BackwardPanel<L>;
  using R = RadialPanel<L>;
  constexpr int TE = L::TE;
  constexpr int TC = L::TC;
  constexpr int Stride = L::Stride;
  constexpr int Edges = L::Edges;
  // Row sums: the cotangent of the source features, then the force and the
  // virial of the edges the row emits, and the gate total of the row under a
  // source gate.
  constexpr int Sums = B::SourceWidth + 13;
  constexpr int ForceSum = B::SourceWidth;
  constexpr int VirialSum = B::SourceWidth + 3;
  constexpr int GateSum = B::SourceWidth + 12;
  const Arguments& args = block.edges;
  const float* record = block.record;
  const float amplitude_knee = __ldg(record + 3);
  const float logit_cap = __ldg(record + 4);
  const PanelTile tile = panel_tile<L>();
  const int c0 = tile.c0;
  const int e0 = tile.e0;
  __shared__ ScanRows scan;
  const int rows = scan_row_count(block);
  if (rows == 0) {
    return;
  }
  locate_scan_rows(block, rows, scan);

  constexpr bool Resident = backward_resident<L>();
  extern __shared__ __align__(16) float block_shared[];
  auto& edges = *reinterpret_cast<PanelEdges<L>*>(block_shared);
  float* operand = block_shared + sizeof(edges) / sizeof(float);
  float* panel = operand + B::Operand * Stride;
  float* radial_total = panel + A::Radial;
  float* envelope_total = panel + A::Envelope;
  float* gate_total = panel + A::GateTotal;
  const bool gated = args.source_gate != nullptr;
  auto* field_gradient =
      reinterpret_cast<float (*)[Edges]>(panel + A::FieldGradient);
  auto* delta_low = reinterpret_cast<float (*)[Edges]>(panel + A::DeltaLow);
  auto* transport_cotangent =
      reinterpret_cast<float (*)[Edges + 1]>(panel + A::TransportCotangent);
  auto* tau_gradient =
      reinterpret_cast<float (*)[Edges]>(panel + A::TauGradient);
  // P_i . K_j, the destination cotangent of the transport against the keys.
  auto* product = reinterpret_cast<float (*)[Edges]>(panel + A::Product);
  auto* mass_weight = reinterpret_cast<float (*)[Edges]>(panel + A::MassWeight);
  // The keys of the rows, the sources of every edge of the backward.
  auto* row_keys = reinterpret_cast<float (*)[B::KeyWidth]>(panel + A::Floats);
  auto* sums = reinterpret_cast<float (*)[Sums]>(row_keys + kBlockRows);
  const float* hidden_map = record + B::EdgeField;
  const float* logit_map = record + B::EdgeGain;
  const float* hidden_map_t = record + B::EdgeAmplitudeT;
  const float* logit_map_t = record + B::EdgeOutputT;
  // The partial sums of the channel groups, reused for every per-head sum
  // over channels, and those of the octets of the radial cotangents: before
  // the edge maps when they are resident, and in the stage of the streamed
  // weights otherwise, whose products a barrier separates from their use.
  float* const partials = reinterpret_cast<float*>(sums + kBlockRows);
  float* const weights = Resident ? partials + A::PartialFloats : partials;
  float* group_partial = partials + A::GroupPartial;
  float* octet_partial = partials + A::OctetPartial;
  // A barrier between a product and a use of the partial sums when they share
  // the stage of its weights.
  const auto release_stage = [] {
    if constexpr (!Resident) {
      __syncthreads();
    }
  };
  if constexpr (Resident) {
    float* staged = weights;
    float* staged_t = staged + 2 * B::Operand * Channels;
    stage_edge_maps<B>(record, staged, staged + B::Operand * Channels);
    for (int index = threadIdx.x * 4; index < Channels * Channels;
         index += kBlockThreads * 4) {
      *reinterpret_cast<float4*>(staged_t + index) = __ldg(
          reinterpret_cast<const float4*>(record + B::EdgeAmplitudeT + index));
      *reinterpret_cast<float4*>(staged_t + Channels * Channels + index) =
          __ldg(
              reinterpret_cast<const float4*>(record + B::EdgeOutputT + index));
    }
    hidden_map = staged;
    logit_map = staged + B::Operand * Channels;
    hidden_map_t = staged_t;
    logit_map_t = staged_t + Channels * Channels;
  }
  for (int index = threadIdx.x; index < kBlockRows * Sums;
       index += kBlockThreads) {
    sums[index / Sums][index % Sums] = 0.0f;
  }
  __syncthreads();
  const long edge_end = scan.edges();
  prefetch_block_bonds<index_t>(args, scan);
  for (int index = threadIdx.x; index < rows * B::KeyWidth;
       index += kBlockThreads) {
    const int r = index / B::KeyWidth;
    const int q = index % B::KeyWidth;
    row_keys[r][q] =
        block.sources[(args.node_begin + scan.row[r]) * B::SourceWidth + q];
  }

  for (long panel_begin = 0; panel_begin < edge_end; panel_begin += Edges) {
    __syncthreads();
    // The destination cotangents of the transport and of the denominators of
    // every edge arrive with its geometry.
    load_panel_edges<L, index_t>(block, scan, panel_begin, edges, row_keys,
                                 operand, transport_cotangent[0]);
    for (int index = threadIdx.x; index < Edges * B::Heads;
         index += kBlockThreads) {
      const int e = index % Edges;
      const int h = index / Edges;
      const float* keys = row_keys[max(edges.row[e], 0)];
      float value = 0.0f;
      for (int harmonic = 0; harmonic < B::Harmonics; ++harmonic) {
        const int q = harmonic * B::Heads + h;
        value = fmaf(transport_cotangent[q][e], keys[q], value);
      }
      product[h][e] = value;
    }
    float slope[R::Items];
    panel_radial<L>(block, edges, operand, slope);
    __syncthreads();
    float phi[TE][TC];
    panel_amplitudes<L>(record, e0, c0, phi, operand, tile, group_partial);
    __syncthreads();
    close_channel_groups<L>(group_partial, edges.tau[0], B::Heads);
    release_stage();

    // === Recompute the hidden layer and the logit ===
    // The source of every entry is the row, as in the forward of its edge.
    float terms[TE][TC];
    load_source_terms<L>(block, edges, B::SourceHidden, e0, c0, terms);
    float acc[TE][TC] = {};
    float activation_slope[TE][TC];
    panel_product<B::Operand, Stride, Channels, TE, TC, Resident>(
        operand, hidden_map, weights, e0, c0, acc, false, logit_map);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        const float hidden = acc[i][j] + terms[i][j];
        activation_slope[i][j] = block_silu_slope(hidden);
        operand[(B::Fields + c0 + j) * Stride + e0 + i] = block_silu(hidden);
      }
    }
    __syncthreads();
    load_source_terms<L>(block, edges, B::SourceState, e0, c0, terms);
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        acc[i][j] = 0.0f;
      }
    }
    panel_product<B::Operand, Stride, Channels, TE, TC, Resident,
                  weight_chunks<B::Operand, Channels>() % 2>(
        operand, logit_map, weights, e0, c0, acc, true);
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        acc[i][j] += terms[i][j];
      }
    }

    // === Cotangent of the logit ===
    float amplitude_gradient[TE][TC];
    float block_part[TE] = {};
#pragma unroll
    for (int i = 0; i < TE; ++i) {
      const int e = e0 + i;
      const bool active = edges.row[e] >= 0;
      const float* cotangent =
          block.cotangents + (active ? edges.node[e] : 0) * B::CotangentWidth;
      const float envelope = edges.envelope[e];
      const float block_envelope = edges.block_envelope[e];
      // dE/d(delta_c) of the channels of the tile: the scalar moment and
      // every angular moment that carries the channel, each with its harmonic
      // and second envelope. The channels of a component are contiguous and a
      // tile lies inside or outside a degree of two or more channels, so the
      // tile loads as one vector per component.
      float d_delta[TC] = {};
      if (active) {
        load_floats_cached<TC>(cotangent + c0, d_delta);
#pragma unroll
        for (int degree = 1; degree <= Lmax; ++degree) {
          const int width = B::width_of(degree);
          if (c0 < width) {
            float angular[TC] = {};
            for (int m = 0; m < 2 * degree + 1; ++m) {
              const float* moment =
                  cotangent + moment_coordinate<B>(degree, m, c0);
              float values[TC] = {};
              if (width == 1) {
                values[0] = __ldg(moment);
              } else {
                load_floats_cached<TC>(moment, values);
              }
              const float harmonic = edges.basis[degree * degree + m][e];
#pragma unroll
              for (int j = 0; j < TC; ++j) {
                angular[j] = fmaf(values[j], harmonic, angular[j]);
              }
            }
#pragma unroll
            for (int j = 0; j < TC; ++j) {
              if (c0 + j < width) {
                d_delta[j] = fmaf(envelope, angular[j], d_delta[j]);
              }
            }
          }
        }
      }
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        const int c = c0 + j;
        const float logit = acc[i][j];
        const float slope_tanh = tanhf(logit);
        const float t = block_envelope * slope_tanh;
        const float d_t = d_delta[j] * phi[i][j];
        block_part[i] = fmaf(d_t, slope_tanh, block_part[i]);
        float d_logit = d_t * block_envelope * (1.0f - slope_tanh * slope_tanh);
        if (c < B::Heads) {
          const float capped = tanhf(logit / logit_cap);
          const float weight = envelope * block_envelope;
          const float exponential = __expf(logit_cap * capped);
          const float mass = active ? weight * weight * exponential : 0.0f;
          const float d_mass = transport_cotangent[B::KeyWidth + c][e] +
                               edges.tau[c][e] * product[c][e];
          d_logit = fmaf(d_mass * mass, 1.0f - capped * capped, d_logit);
          tau_gradient[c][e] = mass * product[c][e];
          edges.mass[c][e] = mass;
          // dE/dw for the weight w = chi chi_k of the mass.
          mass_weight[c][e] =
              active ? d_mass * 2.0f * weight * exponential : 0.0f;
        }
        amplitude_gradient[i][j] = d_delta[j] * t;
        acc[i][j] = active ? d_logit : 0.0f;
        if (c < P::C1) {
          delta_low[c][e] = phi[i][j] * t;
        }
      }
    }
    release_stage();
#pragma unroll
    for (int i = 0; i < TE; ++i) {
      const float total = tile_sum(block_part[i]);
      if (tile.leader) {
        group_partial[tile.group * Edges + e0 + i] = total;
      }
    }
    __syncthreads();
    close_channel_groups<L>(group_partial, envelope_total, 1);
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        operand[(B::Fields + c0 + j) * Stride + e0 + i] = acc[i][j];
      }
    }
    __syncthreads();

    // === State cotangent of the row ===
    add_row_sums<row_sum_parts(Channels)>(
        edges.rows, Channels,
        [&](int c) {
          return RowProduct{operand + (B::Fields + c) * Stride, nullptr,
                            nullptr};
        },
        &sums[0][B::SourceState], Sums);

    // === Field cotangent ===
    // The fields enter both maps, so their cotangent is the sum of the logit
    // and the hidden-layer cotangents through the field rows of [G; W_O] and
    // [W_F; W_phi], each formed while the operand holds its cotangent.
    field_cotangent<L, Resident, false>(operand, logit_map, field_gradient[0]);

    // === dsilu = dlogit W_O^T, dhid = dsilu silu'(hid) ===
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        acc[i][j] = 0.0f;
      }
    }
    panel_product<Channels, Stride, Channels, TE, TC, Resident>(
        operand + B::Fields * Stride, logit_map_t, weights, e0, c0, acc, false,
        hidden_map_t);
    __syncthreads();
    // The hidden-layer cotangent replaces the logit cotangent of the tile of
    // the lane in the operand.
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        operand[(B::Fields + c0 + j) * Stride + e0 + i] =
            edges.row[e0 + i] >= 0 ? acc[i][j] * activation_slope[i][j] : 0.0f;
      }
    }
    __syncthreads();
    field_cotangent<L, Resident, true>(operand, hidden_map, field_gradient[0]);

    // === Hidden-layer cotangent of the row ===
    add_row_sums<row_sum_parts(Channels)>(
        edges.rows, Channels,
        [&](int c) {
          return RowProduct{operand + (B::Fields + c) * Stride, nullptr,
                            nullptr};
        },
        &sums[0][B::SourceHidden], Sums);

    // === dsat = dhid W_phi^T + dtau W_T^T, then the radial cotangent ===
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        acc[i][j] = 0.0f;
      }
    }
    panel_product<Channels, Stride, Channels, TE, TC, Resident,
                  weight_chunks<Channels, Channels>() % 2>(
        operand + B::Fields * Stride, hidden_map_t, weights, e0, c0, acc, true);
#pragma unroll
    for (int i = 0; i < TE; ++i) {
      const int e = e0 + i;
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        const int c = c0 + j;
        float d_saturated = acc[i][j];
#pragma unroll
        for (int h = 0; h < B::Heads; ++h) {
          d_saturated = fmaf(
              tau_gradient[h][e],
              __ldg(record + B::EdgeTransport + c * B::Heads + h), d_saturated);
        }
        acc[i][j] =
            fmaf(d_saturated, soft_saturate_slope(phi[i][j], amplitude_knee),
                 amplitude_gradient[i][j]);
      }
    }
    // The amplitude cotangents replace the hidden-layer cotangents in the
    // channel rows of the operand once every lane has read them.
    __syncthreads();
#pragma unroll
    for (int i = 0; i < TE; ++i) {
#pragma unroll
      for (int j = 0; j < TC; ++j) {
        operand[(B::Fields + c0 + j) * Stride + e0 + i] = acc[i][j];
      }
    }
    if (gated) {
      // Under a source gate the envelope of an edge enters the gates of the
      // other edges of its source: the cotangent of the envelope times the
      // envelope takes the sum over the channels of the amplitude cotangents
      // times the amplitudes, which carry the envelope.
#pragma unroll
      for (int i = 0; i < TE; ++i) {
        float value = 0.0f;
#pragma unroll
        for (int j = 0; j < TC; ++j) {
          value = fmaf(acc[i][j], phi[i][j], value);
        }
        const float total = tile_sum(value);
        if (tile.leader) {
          group_partial[tile.group * Edges + e0 + i] = total;
        }
      }
    }
    __syncthreads();
    radial_cotangent<L>(operand, slope, octet_partial, radial_total);
    if (gated) {
      close_channel_groups<L>(group_partial, gate_total, 1);
    }
    __syncthreads();

    // === Per-edge gradient, force and virial ===
    for (int e = threadIdx.x; e < Edges; e += kBlockThreads) {
      const int row = edges.row[e];
      if (row < 0) {
        continue;
      }
      const float* cotangent =
          block.cotangents + edges.node[e] * B::CotangentWidth;
      const float envelope = edges.envelope[e];
      const float block_envelope = edges.block_envelope[e];
      float block_total = envelope_total[e];
      float weight_total = 0.0f;
#pragma unroll
      for (int h = 0; h < B::Heads; ++h) {
        weight_total += mass_weight[h][e];
      }
      // The angular moments carry the harmonic and the second envelope of the
      // correction amplitude of every low channel.
      float d_basis[L::BasisWidth] = {};
      float envelope_part = weight_total * block_envelope;
      block_total = fmaf(weight_total, envelope, block_total);
#pragma unroll
      for (int degree = 1; degree <= Lmax; ++degree) {
        const float reflection = degree % 2 == 0 ? 1.0f : -1.0f;
        const int width = B::width_of(degree);
        for (int m = 0; m < 2 * degree + 1; ++m) {
          const float* moment = cotangent + moment_coordinate<B>(degree, m, 0);
          float payload = 0.0f;
          if (width == 1) {
            payload = __ldg(moment) * delta_low[0][e];
          } else {
            for (int c = 0; c < width; c += 4) {
              float values[4];
              load_floats_cached<4>(moment + c, values);
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                payload = fmaf(values[k], delta_low[c + k][e], payload);
              }
            }
          }
          const float harmonic = edges.basis[degree * degree + m][e];
          envelope_part = fmaf(payload, harmonic, envelope_part);
          float field = 0.0f;
#pragma unroll
          for (int h = 0; h < B::Heads; ++h) {
            field = fmaf(
                row_keys[row][(B::harmonic_offset(degree) + m) * B::Heads + h],
                field_gradient[(degree - 1) * B::Heads + h][e], field);
          }
          d_basis[degree * degree + m] =
              fmaf(envelope, payload, reflection * field);
        }
      }
      const float x = edges.vector[0][e];
      const float y = edges.vector[1][e];
      const float z = edges.vector[2][e];
      const EdgeGeometry geometry = block_geometry(x, y, z, args.eps);
      const float radial_gradient = radial_total[e] +
                                    envelope_part * edges.envelope_slope[e] +
                                    block_total * edges.block_slope[e];
      if (gated) {
        // The cotangent of the envelope times the envelope, which joins the
        // gate total of the row, the source of every entry; the radial
        // cotangent above was its last reader.
        radial_total[e] = fmaf(envelope, envelope_part, gate_total[e]);
      }
      float low[9];
#pragma unroll
      for (int index = 0; index < 9; ++index) {
        low[index] = index < L::BasisWidth ? d_basis[index] : 0.0f;
      }
      float direction[3] = {0.0f, 0.0f, 0.0f};
      if constexpr (Lmax >= 3) {
#pragma unroll
        for (int index = 0; index < (Lmax >= 4 ? 16 : 7); ++index) {
          high_basis_gradient(geometry, index, d_basis[9 + index], direction);
        }
      }
      float gradient[3];
      basis_vjp(geometry, low, direction, radial_gradient, gradient);
      // The edge emitted by the row points from its destination to the row:
      // the destination gains +g and the row -g, and the virial -g (x) v.
      unsigned long long* const words =
          args.force + edges.node[e] * kForceWords;
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        check_force_range(gradient[component], words);
        atomicAdd(words + component, fixed_point(gradient[component]));
      }
      // The row share of the edge waits in the leading rows of the operand,
      // which the panel no longer reads, for the row sums below.
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        operand[component * Stride + e] = -gradient[component];
      }
#pragma unroll
      for (int component = 0; component < 9; ++component) {
        operand[(3 + component) * Stride + e] =
            -gradient[component / 3] * edges.vector[component % 3][e];
      }
    }
    __syncthreads();

    // === Key cotangent, force and virial of every row ===
    // The fields read the row's keys along the reversed bond, and the
    // transport scales them by the mass, the transport weight and the
    // destination cotangent. The force, the virial and the gate total of the
    // rows follow on the warps after those of the keys.
    constexpr int KeyParts = row_sum_parts(B::KeyWidth + 13);
    add_row_sums<KeyParts>(
        edges.rows, B::KeyWidth,
        [&](int component) {
          const int harmonic = component / B::Heads;
          const int h = component % B::Heads;
          const int degree = harmonic_degree<Lmax>(harmonic);
          const float reflection = degree % 2 == 0 ? 1.0f : -1.0f;
          const float* basis = edges.basis[1 + harmonic];
          const float* field = field_gradient[(degree - 1) * B::Heads + h];
          const float* mass = edges.mass[h];
          const float* tau = edges.tau[h];
          const float* cotangent = transport_cotangent[component];
          return [=](int e) {
            return fmaf(reflection * basis[e], field[e],
                        mass[e] * tau[e] * cotangent[e]);
          };
        },
        &sums[0][0], Sums);
    add_row_sums<KeyParts>(
        edges.rows, gated ? 13 : 12,
        [&](int component) {
          return RowProduct{
              component < 12 ? operand + component * Stride : radial_total,
              nullptr, nullptr};
        },
        &sums[0][ForceSum], Sums, row_sum_lanes<KeyParts>(B::KeyWidth));
  }
  __syncthreads();

  // === Source cotangents, row force and virial ===
  for (int index = threadIdx.x; index < rows * B::SourceWidth;
       index += kBlockThreads) {
    const int r = index / B::SourceWidth;
    const int q = index % B::SourceWidth;
    block.source_gradient[scan.row[r] * B::SourceWidth + q] = sums[r][q];
  }
  for (int index = threadIdx.x; index < rows * 12; index += kBlockThreads) {
    const int r = index / 12;
    const int q = index % 12;
    const long row = scan.row[r];
    if (q < 3) {
      const long node = args.node_begin + row;
      const float value = sums[r][ForceSum + q];
      check_force_range(value, args.force + node * kForceWords);
      atomicAdd(args.force + node * kForceWords + q, fixed_point(value));
    } else {
      block.source_virial[row * 9 + q - 3] += sums[r][VirialSum + q - 3];
    }
  }
  if (gated) {
    for (int r = threadIdx.x; r < rows; r += kBlockThreads) {
      block.gate_sums[args.node_begin + scan.row[r]] += sums[r][GateSum];
    }
  }
}

// === Launch dispatch ===

inline void check_block_status(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    cudaGetLastError();
    TORCH_CHECK(false, "dpa4c: ", what,
                " failed: ", cudaGetErrorString(status));
  }
}

/// Raise the dynamic shared memory of ``kernel`` to ``bytes`` once per kernel
/// and device, and return the blocks of ``kernel`` that the device keeps
/// resident at once.
inline long resident_blocks(void (*kernel)(BlockArguments), size_t bytes) {
  static std::mutex mutex;
  // Keyed by the address of the kernel as an integer, which orders totally.
  static std::map<std::pair<std::uintptr_t, int>, long> capacities;
  int device = 0;
  check_block_status(cudaGetDevice(&device), "the device query");
  const std::pair<std::uintptr_t, int> key{
      reinterpret_cast<std::uintptr_t>(kernel), device};
  const std::lock_guard<std::mutex> lock(mutex);
  const auto known = capacities.find(key);
  if (known != capacities.end()) {
    return known->second;
  }
  check_block_status(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(bytes)),
      "the shared memory of a further-block scan");
  int resident = 0;
  check_block_status(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                         &resident, kernel, kBlockThreads, bytes),
                     "the occupancy query of a further-block scan");
  int multiprocessors = 0;
  check_block_status(
      cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                             device),
      "the multiprocessor count");
  const long capacity = static_cast<long>(resident) * multiprocessors;
  capacities.emplace(key, capacity);
  return capacity;
}

/// Launch an edge scan over the run of ``arguments``. A block takes the most
/// destination rows among kBlockRows, kBlockRows / 2, ..., 1 with which the
/// blocks of the whole system still fill the ``capacity`` blocks the device
/// keeps resident: fewer rows shorten the panels every block scans in turn
/// when the system cannot fill the device, and more spread the partly filled
/// last panel of a block over more edges otherwise. The choice follows from
/// the node count of the whole system, so every run of a tiled system sums
/// its rows in the same order.
template <class Kernel>
void launch_panel_scan(Kernel kernel,
                       size_t bytes,
                       const BlockArguments& arguments,
                       cudaStream_t stream) {
  const long capacity = resident_blocks(kernel, bytes);
  int rows = kBlockRows;
  while (rows > 1 &&
         (arguments.edges.system_node_count + rows - 1) / rows < capacity) {
    rows /= 2;
  }
  BlockArguments launched = arguments;
  launched.block_rows = rows;
  const int blocks =
      static_cast<int>((arguments.edges.node_count + rows - 1) / rows);
  if (blocks > 0) {
    kernel<<<blocks, kBlockThreads, bytes, stream>>>(launched);
  }
}

template <int Channels, int Lmax, bool HasModes, bool HasSpin, typename index_t>
void launch_block_scan(BlockStage stage,
                       const BlockArguments& arguments,
                       cudaStream_t stream) {
  if (stage == BlockStage::Forward) {
    using L = PanelLayout<Channels, Lmax, HasSpin, HasModes, false>;
    constexpr size_t bytes =
        block_forward_shared_bytes<L, forward_resident<L>()>();
    static_assert(bytes <= kPanelSharedLimit,
                  "the panel of the forward scan exceeds its shared memory");
    launch_panel_scan(
        block_forward_kernel<Channels, Lmax, HasModes, HasSpin, index_t>, bytes,
        arguments, stream);
  } else {
    using L = PanelLayout<Channels, Lmax, HasSpin, HasModes, true>;
    constexpr size_t bytes =
        block_backward_shared_bytes<L, backward_resident<L>()>();
    static_assert(bytes <= kPanelSharedLimit,
                  "the panel of the backward scan exceeds its shared memory");
    launch_panel_scan(
        block_backward_kernel<Channels, Lmax, HasModes, HasSpin, index_t>,
        bytes, arguments, stream);
  }
}

template <int Channels, int Lmax, bool HasSpin>
void launch_block_nodes(BlockStage stage,
                        const BlockArguments& arguments,
                        cudaStream_t stream) {
  using P = Profile<Channels, Lmax, HasSpin>;
  using B = BlockProfile<Channels, Lmax, HasSpin>;
  const long count = arguments.edges.node_count;
  if (count == 0) {
    return;
  }
  // The GEMM tiles of the node maps: tile_index walks the column tiles of
  // every row tile of the run.
  const auto tiles = [&](int width) {
    return static_cast<int>((count + NodeTile::rows - 1) / NodeTile::rows *
                            ((width + NodeTile::cols - 1) / NodeTile::cols));
  };
  switch (stage) {
    case BlockStage::Maps:
      block_maps_kernel<Channels, Lmax, HasSpin>
          <<<tiles(B::NodeWidth), NodeTile::threads, 0, stream>>>(arguments);
      return;
    case BlockStage::MapBackward:
      block_map_cotangent_kernel<Channels, Lmax, HasSpin>
          <<<tiles(NodeTile::cols), NodeTile::threads, 0, stream>>>(arguments);
      block_map_input_kernel<Channels, Lmax, HasSpin>
          <<<tiles(P::OutputWidth), NodeTile::threads, 0, stream>>>(arguments);
      return;
    case BlockStage::Cotangents: {
      constexpr int Warps = kBlockThreads / kWarpSize;
      block_cotangent_kernel<Channels, Lmax, HasSpin>
          <<<static_cast<int>((count + Warps - 1) / Warps), kBlockThreads, 0,
             stream>>>(arguments);
      return;
    }
    default:
      return;
  }
}

template <int Channels, int Lmax>
void launch_block(BlockStage stage,
                  const BlockArguments& arguments,
                  cudaStream_t stream) {
  const Arguments& args = arguments.edges;
  const bool scan =
      stage == BlockStage::Forward || stage == BlockStage::Backward;
  const bool wide = args.index_kind == IndexKind::Bits64;
  const auto dispatch = [&](auto modes, auto spin) {
    constexpr bool HasModes = decltype(modes)::value;
    constexpr bool HasSpin = decltype(spin)::value;
    if (!scan) {
      launch_block_nodes<Channels, Lmax, HasSpin>(stage, arguments, stream);
    } else if (wide) {
      launch_block_scan<Channels, Lmax, HasModes, HasSpin, long>(
          stage, arguments, stream);
    } else {
      launch_block_scan<Channels, Lmax, HasModes, HasSpin, std::uint32_t>(
          stage, arguments, stream);
    }
  };
  const bool modes = args.radial_modes > 0;
  if (args.has_spin) {
    modes ? dispatch(std::true_type{}, std::true_type{})
          : dispatch(std::false_type{}, std::true_type{});
  } else {
    modes ? dispatch(std::true_type{}, std::false_type{})
          : dispatch(std::false_type{}, std::false_type{});
  }
}

}  // namespace deepmd_dpa4c
