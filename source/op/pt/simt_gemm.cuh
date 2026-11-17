// SPDX-License-Identifier: LGPL-3.0-or-later
//
// FP32 SIMT GEMM tiles shared by the node-scale CUDA operators: the fitting
// network (graph_fitting.cu) and the node maps of the further DPA4C blocks
// (dpa4c/block_kernel.cuh). A CTA computes one tile of C = A @ B in
// double-buffered depth slices; the operand stagers read A and B from
// global memory, and a caller supplies stagers of its own that transform an
// operand while staging it.

#pragma once

#include <cuda_runtime.h>

namespace deepmd_gemm {

// ---------------------------------------------------------------------------
// Vector helpers. An operand addressed in float4 quads is 16-byte aligned.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float4 zero4() {
  return make_float4(0.f, 0.f, 0.f, 0.f);
}

__device__ __forceinline__ float4 load4(const float* p) {
  return __ldg(reinterpret_cast<const float4*>(p));
}

__device__ __forceinline__ float4 lds4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ float2 lds2(const float* p) {
  return *reinterpret_cast<const float2*>(p);
}

__device__ __forceinline__ void store4(float* p, float4 v) {
  *reinterpret_cast<float4*>(p) = v;
}

/// Request the line holding ``p`` into L2.
__device__ __forceinline__ void prefetch_l2(const float* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

/// Depth slices by which the streamed node-major operands are requested into
/// L2 ahead of their register load. The register stage hides one slice of
/// latency; the prefetch covers the DRAM latency beyond it, which dominates
/// once the run outgrows L2.
constexpr int kLookahead = 2;

// ---------------------------------------------------------------------------
// GEMM tiles.
//
// A CTA computes a ``rows`` x ``cols`` block of C = A @ B, A the node-major
// (nodes, K) operand and B the (K, cols) weight operand, in ``depth``-deep
// slices staged through double-buffered shared memory: the global loads of
// slice s + 1 are in flight in registers while slice s is multiplied. A warp
// is 4 x 8 lanes and a lane owns tm x tn accumulators as blocks of
// row_group x 4, at rows r + h * row_stride + [0, row_group) and columns
// c + g * col_stride + [0, 4) from its origin (r, c), where a row group is a
// quad, or a pair for a lane of two rows. The four lane rows of a warp read
// adjacent 16- or 8-byte words of a shared A row and its eight lane columns
// adjacent 16-byte words of a shared B row. The lane rows and columns are
// lane / 8 and lane % 8, or, for a tile with ``paired_columns``, lane / 2 % 4
// and 2 * (lane / 8) + lane % 2, which gives each quarter-warp two adjacent
// B words instead of all eight. On sm_120 a 16-byte B read then takes two
// shared wavefronts instead of four, while the A reads keep theirs: one per
// 8-byte and two per 16-byte read in either layout.
// ---------------------------------------------------------------------------

/// Shared rows carry four floats of skew. Every tile extent is a multiple of
/// 32 floats, so the row stride is congruent to 4 modulo the 32 banks, which
/// spreads the transposing scalar stores of a warp over 32 banks while every
/// row stays 16-byte aligned.
constexpr int kSkew = 4;

template <int kRows,
          int kCols,
          int kThreadRows,
          int kThreadCols,
          int kDepth,
          int kWarpRows,
          int kWarpCols,
          int kMinBlocks>
struct GemmTile {
  static constexpr int rows = kRows;
  static constexpr int cols = kCols;
  static constexpr int tm = kThreadRows;
  static constexpr int tn = kThreadCols;
  static constexpr int depth = kDepth;
  static constexpr int warp_cols = kWarpCols;
  static constexpr int threads = 32 * kWarpRows * kWarpCols;
  static constexpr int min_blocks = kMinBlocks;
  static constexpr int row_group = 4;
  static constexpr bool paired_columns = false;
  static constexpr int row_stride = kRows / (kThreadRows / 4);
  static constexpr int col_stride = kCols / (kThreadCols / 4);
  static_assert(kThreadRows % 4 == 0 && kThreadCols % 4 == 0,
                "a lane owns whole row and column quads");
  static_assert(kRows == 4 * kThreadRows * kWarpRows &&
                    kCols == 8 * kThreadCols * kWarpCols,
                "a warp covers 4 tm rows and 8 tn columns as 4 x 8 lanes");
  static_assert(kRows % 32 == 0 && kCols % 32 == 0 && kDepth % 8 == 0,
                "tile extents are whole 32-row, 32-column, 8-deep blocks");
  static_assert(kCols / (kThreadCols / 4) % 32 == 0,
                "the eight lane columns of a quad cover one 32-column group");
};

template <class Tile>
struct GemmStorage {
  float a[2][Tile::depth][Tile::rows + kSkew];
  float b[2][Tile::depth][Tile::cols + kSkew];
};

template <class Tile>
using Accumulators = float[Tile::tm][Tile::tn];

/// Tile-local row and column of a lane's first accumulator.
struct LaneOrigin {
  int row;
  int col;
};

/// Lane row and lane column of ``lane`` within the 4 x 8 lanes of its warp.
template <class Tile>
__device__ __forceinline__ int lane_row(int lane) {
  return Tile::paired_columns ? lane / 2 % 4 : lane / 8;
}

template <class Tile>
__device__ __forceinline__ int lane_col(int lane) {
  return Tile::paired_columns ? 2 * (lane / 8) + lane % 2 : lane % 8;
}

/// Lane distance between lane columns ``d`` apart, for d of 1, 2 or 4 within
/// one aligned group of eight, which a butterfly over the columns uses.
template <class Tile>
__device__ __forceinline__ constexpr int lane_col_stride(int d) {
  return Tile::paired_columns && d > 1 ? 4 * d : d;
}

template <class Tile>
__device__ __forceinline__ LaneOrigin lane_origin() {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  return {
      ((warp / Tile::warp_cols) * 4 + lane_row<Tile>(lane)) * Tile::row_group,
      ((warp % Tile::warp_cols) * 8 + lane_col<Tile>(lane)) * 4};
}

/// Tile-local row of accumulator row ``i``.
template <class Tile>
__device__ __forceinline__ int accumulator_row(const LaneOrigin& o, int i) {
  return o.row + (i / Tile::row_group) * Tile::row_stride + i % Tile::row_group;
}

/// Tile-local column of the first accumulator of column quad ``g``.
template <class Tile>
__device__ __forceinline__ int accumulator_col(const LaneOrigin& o, int g) {
  return o.col + g * Tile::col_stride;
}

template <class Tile>
__device__ __forceinline__ float4
accumulator_quad(const Accumulators<Tile>& acc, int i, int g) {
  return make_float4(acc[i][4 * g], acc[i][4 * g + 1], acc[i][4 * g + 2],
                     acc[i][4 * g + 3]);
}

/// CTA tile of a one-dimensional grid: consecutive CTAs walk the column tiles
/// of one row tile, so they share their A rows while those are hot in L2.
struct TileIndex {
  long row0;
  int col0;
};

template <class Tile>
__device__ __forceinline__ TileIndex tile_index(int width) {
  const int col_tiles = (width + Tile::cols - 1) / Tile::cols;
  return {(long)(blockIdx.x / col_tiles) * Tile::rows,
          (int)(blockIdx.x % col_tiles) * Tile::cols};
}

// ---------------------------------------------------------------------------
// Operand stagers. ``fetch(k0)`` issues the global loads of the slice
// [k0, k0 + Tile::depth) of the contraction axis, of length ``extent``, into
// registers and ``commit`` writes them to shared memory in the depth-major
// layout the fragments read. Depths past the extent are zero-filled; rows and
// columns past the matrix are clamped to its last row or column, whose finite
// values reach only accumulators that are never stored.
// ---------------------------------------------------------------------------

/// Thread map of a ``kLines`` x depth slice read as float4 along depth and
/// stored depth-major. Lane pairs cover the 32-byte segment of one line in
/// each 8-deep group of the slice, so a warp reads 16 lines and its scalar
/// stores land on 32 distinct banks.
template <class Tile, int kLines>
struct TransposedSlice {
  static constexpr int count = kLines * Tile::depth / 4 / Tile::threads;
  static_assert(count >= 1 && count * 4 * Tile::threads == kLines * Tile::depth,
                "the threads must tile the slice evenly");
  static_assert(2 * kLines % 32 == 0, "a warp stays within one 8-deep group");
  __device__ static int line(int s) {
    const int i = threadIdx.x + s * Tile::threads;
    return (i / 2) % kLines;
  }
  __device__ static int quad(int s) {
    const int i = threadIdx.x + s * Tile::threads;
    return i % 2 + 2 * (i / (2 * kLines));
  }
};

template <class Tile, int kLines, int kCount>
__device__ __forceinline__ void store_transposed(float (*dst)[kLines + kSkew],
                                                 const float4 (&v)[kCount]) {
  using Map = TransposedSlice<Tile, kLines>;
#pragma unroll
  for (int s = 0; s < kCount; ++s) {
    const int q = 4 * Map::quad(s);
    const int l = Map::line(s);
    dst[q][l] = v[s].x;
    dst[q + 1][l] = v[s].y;
    dst[q + 2][l] = v[s].z;
    dst[q + 3][l] = v[s].w;
  }
}

/// A operand read as stored from a row-major (nodes, K) tensor: the layer
/// input of the forward, or a dpre emitted for the backward.
template <class Tile, bool kVector>
struct InputStager;

/// float4 loads, for 16-byte aligned rows.
template <class Tile>
struct InputStager<Tile, true> {
  using Map = TransposedSlice<Tile, Tile::rows>;
  const float* src[Map::count];
  float4 v[Map::count];
  int extent;

  __device__ InputStager(const float* x, long row0, long nodes, int extent_)
      : extent(extent_) {
#pragma unroll
    for (int s = 0; s < Map::count; ++s) {
      const long row = min(row0 + Map::line(s), nodes - 1);
      src[s] = x + row * extent + 4 * Map::quad(s);
    }
  }

  __device__ __forceinline__ void fetch(int k0) {
#pragma unroll
    for (int s = 0; s < Map::count; ++s) {
      const int k = k0 + 4 * Map::quad(s);
      v[s] = k < extent ? load4(src[s] + k0) : zero4();
      if (k + kLookahead * Tile::depth < extent) {
        prefetch_l2(src[s] + k0 + kLookahead * Tile::depth);
      }
    }
  }

  __device__ __forceinline__ void commit(float (*a)[Tile::rows + kSkew]) {
    store_transposed<Tile, Tile::rows>(a, v);
  }
};

/// Scalar loads, for any row stride: eight lanes cover the 32-byte segment of
/// one line and the warp's stores land on 32 distinct banks.
template <class Tile>
struct InputStager<Tile, false> {
  static constexpr int count = Tile::rows * Tile::depth / Tile::threads;
  static_assert(count >= 1 && count * Tile::threads == Tile::rows * Tile::depth,
                "the threads must tile the slice evenly");
  const float* src[count];
  float v[count];
  int extent;

  __device__ static int line(int s) {
    const int i = threadIdx.x + s * Tile::threads;
    return (i / 8) % Tile::rows;
  }
  __device__ static int col(int s) {
    const int i = threadIdx.x + s * Tile::threads;
    return i % 8 + 8 * (i / (8 * Tile::rows));
  }

  __device__ InputStager(const float* x, long row0, long nodes, int extent_)
      : extent(extent_) {
#pragma unroll
    for (int s = 0; s < count; ++s) {
      const long row = min(row0 + line(s), nodes - 1);
      src[s] = x + row * extent + col(s);
    }
  }

  __device__ __forceinline__ void fetch(int k0) {
#pragma unroll
    for (int s = 0; s < count; ++s) {
      const int k = k0 + col(s);
      v[s] = k < extent ? __ldg(src[s] + k0) : 0.f;
      if (k + kLookahead * Tile::depth < extent) {
        prefetch_l2(src[s] + k0 + kLookahead * Tile::depth);
      }
    }
  }

  __device__ __forceinline__ void commit(float (*a)[Tile::rows + kSkew]) {
#pragma unroll
    for (int s = 0; s < count; ++s) {
      a[col(s)][line(s)] = v[s];
    }
  }
};

/// Forward B operand: the row-major (K, width) weight, read as float4
/// along its output channels and stored as is.
template <class Tile>
struct WeightStager {
  static constexpr int quads = Tile::cols / 4;
  static constexpr int count = Tile::depth * quads / Tile::threads;
  static_assert(count >= 1 && count * Tile::threads == Tile::depth * quads &&
                    Tile::threads % quads == 0,
                "the threads must tile the slice evenly");
  const float* src;
  float4 v[count];
  int extent;
  int width;

  __device__ static int line(int s) {
    return (threadIdx.x + s * Tile::threads) / quads;
  }

  __device__ WeightStager(const float* w, int col0, int extent_, int width_)
      : src(w + min(col0 + 4 * (int)(threadIdx.x % quads), width_ - 4)),
        extent(extent_),
        width(width_) {}

  __device__ __forceinline__ void fetch(int k0) {
#pragma unroll
    for (int s = 0; s < count; ++s) {
      const int k = k0 + line(s);
      v[s] = k < extent ? load4(src + (long)k * width) : zero4();
    }
  }

  __device__ __forceinline__ void commit(float (*b)[Tile::cols + kSkew]) {
#pragma unroll
    for (int s = 0; s < count; ++s) {
      store4(&b[line(s)][4 * (threadIdx.x % quads)], v[s]);
    }
  }
};

/// Backward B operand: W^T for the row-major (width, K) weight W, read
/// as float4 along W's rows (the contraction axis) and stored transposed.
template <class Tile>
struct WeightTransposeStager {
  using Map = TransposedSlice<Tile, Tile::cols>;
  const float* src[Map::count];
  float4 v[Map::count];
  int extent;

  __device__ WeightTransposeStager(const float* w,
                                   int col0,
                                   int extent_,
                                   int width)
      : extent(extent_) {
#pragma unroll
    for (int s = 0; s < Map::count; ++s) {
      const int n = min(col0 + Map::line(s), width - 1);
      src[s] = w + (long)n * extent + 4 * Map::quad(s);
    }
  }

  __device__ __forceinline__ void fetch(int k0) {
#pragma unroll
    for (int s = 0; s < Map::count; ++s) {
      v[s] = k0 + 4 * Map::quad(s) < extent ? load4(src[s] + k0) : zero4();
    }
  }

  __device__ __forceinline__ void commit(float (*b)[Tile::cols + kSkew]) {
    store_transposed<Tile, Tile::cols>(b, v);
  }
};

// ---------------------------------------------------------------------------
// Main loop. Every accumulator sums the contraction axis in ascending order
// with one fused multiply-add per term, independent of the tile shape.
// ---------------------------------------------------------------------------

template <class Tile>
__device__ __forceinline__ void multiply_slice(
    const float (*a)[Tile::rows + kSkew],
    const float (*b)[Tile::cols + kSkew],
    const LaneOrigin& o,
    Accumulators<Tile>& acc) {
#pragma unroll
  for (int k = 0; k < Tile::depth; ++k) {
    float av[Tile::tm];
    float bv[Tile::tn];
#pragma unroll
    for (int h = 0; h < Tile::tm / Tile::row_group; ++h) {
      const float* row = &a[k][o.row + h * Tile::row_stride];
      if constexpr (Tile::row_group == 4) {
        const float4 v = lds4(row);
        av[4 * h] = v.x;
        av[4 * h + 1] = v.y;
        av[4 * h + 2] = v.z;
        av[4 * h + 3] = v.w;
      } else {
        const float2 v = lds2(row);
        av[2 * h] = v.x;
        av[2 * h + 1] = v.y;
      }
    }
#pragma unroll
    for (int g = 0; g < Tile::tn / 4; ++g) {
      const float4 v = lds4(&b[k][o.col + g * Tile::col_stride]);
      bv[4 * g] = v.x;
      bv[4 * g + 1] = v.y;
      bv[4 * g + 2] = v.z;
      bv[4 * g + 3] = v.w;
    }
#pragma unroll
    for (int i = 0; i < Tile::tm; ++i) {
#pragma unroll
      for (int j = 0; j < Tile::tn; ++j) {
        acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
      }
    }
  }
}

template <class S>
__device__ __forceinline__ void zero_accumulators(Accumulators<S>& acc) {
#pragma unroll
  for (int i = 0; i < S::tm; ++i) {
#pragma unroll
    for (int j = 0; j < S::tn; ++j) {
      acc[i][j] = 0.f;
    }
  }
}

template <class Tile, class StagerA, class StagerB>
__device__ __forceinline__ void gemm_mainloop(GemmStorage<Tile>& smem,
                                              StagerA& sa,
                                              StagerB& sb,
                                              int extent,
                                              const LaneOrigin& o,
                                              Accumulators<Tile>& acc) {
  zero_accumulators<Tile>(acc);
  const int slices = (extent + Tile::depth - 1) / Tile::depth;
  sa.fetch(0);
  sb.fetch(0);
  sa.commit(smem.a[0]);
  sb.commit(smem.b[0]);
  __syncthreads();
  for (int s = 0; s < slices; ++s) {
    const int buf = s & 1;
    sa.fetch((s + 1) * Tile::depth);
    sb.fetch((s + 1) * Tile::depth);
    multiply_slice<Tile>(smem.a[buf], smem.b[buf], o, acc);
    sa.commit(smem.a[buf ^ 1]);
    sb.commit(smem.b[buf ^ 1]);
    __syncthreads();
  }
}

}  // namespace deepmd_gemm
