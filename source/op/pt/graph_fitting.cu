// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused energy fitting network for graph-lower inference. The operator is
// descriptor-agnostic: any graph-lowered energy model whose fitting is a
// plain MLP over the flat node axis dispatches here.
//   h_0 = act(x @ W_0 + b_0)                      (+ identity residual when
//   h_l = act(h_{l-1} @ W_l + b_l)                  the layer is square)
//   e   = h_{L-1} @ w_head + b_head + bias_atom_e[atype]   (fp64 output)
// Each layer is one FP32 SIMT GEMM kernel with the elementwise work fused into
// it: the epilogue stores the pre-activation, adds the bias, applies the
// activation and the identity residual, and stores the activation; the last
// layer reduces its activation against the head weight instead, so no hidden
// tensor is read back for the energy.
// The backward (upstream d_e, a unit vector for the energy reduction) chains
//   dh_{L-1} = d_e * w_head^T
//   dpre_l   = dh_l * act'(pre_l + b_l)
//   dh_{l-1} = dpre_l @ W_l^T  (+ dh_l identity residual)
//   d_x      = dpre_0 @ W_0^T
// with dpre formed while the GEMM stages its A operand -- the seed layer forms
// dh_{L-1} there as well, so it never exists as a tensor -- and the identity
// residual added in the GEMM epilogue. A layer without identity residual
// consumes only dpre, so when its GEMM spans several column tiles, each of
// which would form the same dpre again, the epilogue of the layer above emits
// that dpre directly and it is formed once.
//
// The GEMMs accumulate in fp32 on the SIMT cores with one fused multiply-add
// per term, without TF32 or split-K reordering, so the fp32 potential-energy
// surface is deterministic for MD. Every reduction runs in a fixed order
// without atomics: each GEMM output sums its contraction in ascending order,
// and the energy head reduces each 32-channel group of its dot product by a
// fixed butterfly and sums the groups in ascending order. Neither order
// depends on the tile shape. The tile routing also decides whether a dpre is
// emitted or formed while staging, and both evaluate the same expression on
// the same operands in the same order, so the results are bitwise independent
// of the tile routing and of how a caller splits the node axis into runs.
//
// The saved state is the pre-activation of every layer, which each GEMM writes
// directly. The backward re-derives the activation derivative from it, so
// besides it the forward stores only the activation, one full-tensor store per
// layer less than a formulation that also materializes the derivative.
//
// Inference, which knows the energy cotangent before the forward runs, can
// take a fused network instead: one kernel carries a panel of nodes through
// every layer, the head and the backward with the activations in shared
// memory, and reproduces the energy and the input cotangent of the per-layer
// path bit for bit. It serves the networks of one hidden width it is compiled
// for, on runs whose panels split over clusters of CTAs that the device keeps
// resident at once or, as single CTAs, cover enough of the device, except
// where the per-layer kernels are compute-bound on the device;
// dispatch_fused_shape lists the shapes of each width. A fitting-route table
// measured at freeze time on the deployment device names another route for
// the run sizes at which the measurement departs from these rules
// (fitting_route).
//
// All tensors here are node-scale (atoms, not edges); the fusion removes the
// full-tensor elementwise passes rather than FLOPs. The head bias arrives as a
// device tensor so that symbolic tracing never reads a value host-side.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "fp32_ridge.h"
#include "graph_ops.h"

namespace {

#define FITTING_CHECK_LAUNCH(what)                                        \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

// sigma(z) = (1 + tanh(z/2)) / 2. The identity is not the detour it looks
// like: CUDA's tanhf is a branch-free sequence of one MUFU.EX2 and one
// MUFU.RCP, whereas an accurate expf carries a branchy slow path and the
// intrinsic __expf still needs a reciprocal to finish the sigmoid. Measured
// per kernel instance at one million nodes, the identity beats both, and the
// only faster form is fully approximate, worth 0.5% of the fitting forward
// against an approximate division and exponential.
__device__ __forceinline__ float sigmoid(float z) {
  return 0.5f * (1.f + tanhf(0.5f * z));
}

// Activation codes follow
// deepmd.pt_expt.kernels.triton.dpa1.activation.ACT_CODES (0 = tanh, 1 = silu).
// Value and derivative are separate because the forward needs only the former
// and the backward only the latter.
template <int ACT>
__device__ __forceinline__ float act_value(float z) {
  if constexpr (ACT == 0) {
    return tanhf(z);
  }
  return z * sigmoid(z);
}

template <int ACT>
__device__ __forceinline__ float act_derivative(float z) {
  if constexpr (ACT == 0) {
    const float a = tanhf(z);
    return 1.f - a * a;
  }
  const float s = sigmoid(z);
  return s * (1.f + z * (1.f - s));
}

long ceil_div(long a, long b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Vector helpers. Every hidden width is a positive multiple of four and every
// weight, bias and head weight 16-byte aligned (check_weights,
// check_parameters), so every hidden row, bias and head weight is addressed in
// aligned float4 quads.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float4 zero4() {
  return make_float4(0.f, 0.f, 0.f, 0.f);
}

__device__ __forceinline__ float4 load4(const float* p) {
  return __ldg(reinterpret_cast<const float4*>(p));
}

/// Load through L2 only, for data that the kernel itself writes while it runs,
/// which the read-only path of ``load4`` must not serve.
__device__ __forceinline__ float4 load4_coherent(const float* p) {
  return __ldcg(reinterpret_cast<const float4*>(p));
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

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/// a + b per element as explicitly rounded additions. The forward identity
/// residual follows the activation product, which a compiler may contract
/// with the addition into one FMA in one kernel and not in another; both
/// forward epilogues add it with this function, so the fused network and the
/// per-layer kernels round the same operations whatever the code generation.
/// The backward residual adds two values that no product produces, so a plain
/// addition rounds it identically in every kernel.
__device__ __forceinline__ float4 add4_rn(float4 a, float4 b) {
  return make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y),
                     __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
}

/// s * a + b per element, each with a single rounding.
__device__ __forceinline__ float4 fma4(float s, float4 a, float4 b) {
  return make_float4(fmaf(s, a.x, b.x), fmaf(s, a.y, b.y), fmaf(s, a.z, b.z),
                     fmaf(s, a.w, b.w));
}

__device__ __forceinline__ float4 scale4(float s, float4 v) {
  return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}

template <int ACT>
__device__ __forceinline__ float4 act_value4(float4 z) {
  return make_float4(act_value<ACT>(z.x), act_value<ACT>(z.y),
                     act_value<ACT>(z.z), act_value<ACT>(z.w));
}

template <int ACT>
__device__ __forceinline__ float4 act_derivative4(float4 z) {
  return make_float4(act_derivative<ACT>(z.x), act_derivative<ACT>(z.y),
                     act_derivative<ACT>(z.z), act_derivative<ACT>(z.w));
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

/// Large tile, 8 x 8 accumulators per lane, for grids of several waves.
using LargeTile = GemmTile<64, 128, 8, 8, 8, 2, 2, 3>;
/// Small tile with a four times finer grid for runs that would leave the
/// device partly idle under the large one. Its 27 KB of shared memory keep
/// three CTAs resident per SM on sm_120.
using SmallTile = GemmTile<32, 64, 4, 4, 32, 2, 2, 3>;

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

/// Where a backward layer takes its A operand dpre = dh * act'(pre + b) from.
enum class Cotangent {
  kHead,    //!< Seed layer: dh = d_e[n] * w_head, dpre formed while staging.
  kChain,   //!< dh read from the layer above, dpre formed while staging.
  kFormed,  //!< dpre emitted by the epilogue of the layer above.
};

/// Backward A operand formed while staging, so the derivative never
/// round-trips through memory. The seed layer forms dh = d_e[n] * w_head on the
/// fly as well, so no head cotangent tensor exists.
template <class Tile, int ACT, bool kSeed>
struct CotangentStager {
  using Map = TransposedSlice<Tile, Tile::rows>;
  const float* pre_src[Map::count];
  const float* dh_src[Map::count];
  float de[Map::count];
  float4 pre[Map::count];
  float4 g[Map::count];
  float4 bias[Map::count];
  const float* b;
  const float* w_head;
  int extent;

  __device__ CotangentStager(const float* pre_,
                             const float* dh,
                             const double* d_e,
                             const float* w_head_,
                             const float* b_,
                             long row0,
                             long nodes,
                             int extent_)
      : b(b_), w_head(w_head_), extent(extent_) {
#pragma unroll
    for (int s = 0; s < Map::count; ++s) {
      const long row = min(row0 + Map::line(s), nodes - 1);
      const long offset = row * extent + 4 * Map::quad(s);
      pre_src[s] = pre_ + offset;
      if constexpr (kSeed) {
        de[s] = (float)d_e[row];
      } else {
        dh_src[s] = dh + offset;
      }
    }
  }

  __device__ __forceinline__ void fetch(int k0) {
#pragma unroll
    for (int s = 0; s < Map::count; ++s) {
      const int k = k0 + 4 * Map::quad(s);
      const bool valid = k < extent;
      pre[s] = valid ? load4(pre_src[s] + k0) : zero4();
      if constexpr (kSeed) {
        g[s] = valid ? load4(w_head + k) : zero4();
      } else {
        g[s] = valid ? load4(dh_src[s] + k0) : zero4();
      }
      bias[s] = valid && b ? load4(b + k) : zero4();
      if (k + kLookahead * Tile::depth < extent) {
        prefetch_l2(pre_src[s] + k0 + kLookahead * Tile::depth);
        if constexpr (!kSeed) {
          prefetch_l2(dh_src[s] + k0 + kLookahead * Tile::depth);
        }
      }
    }
  }

  __device__ __forceinline__ void commit(float (*a)[Tile::rows + kSkew]) {
    float4 d[Map::count];
#pragma unroll
    for (int s = 0; s < Map::count; ++s) {
      const float4 dh = kSeed ? scale4(de[s], g[s]) : g[s];
      const float4 slope = act_derivative4<ACT>(add4(pre[s], bias[s]));
      d[s] = make_float4(dh.x * slope.x, dh.y * slope.y, dh.z * slope.z,
                         dh.w * slope.w);
    }
    store_transposed<Tile, Tile::rows>(a, d);
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

// ---------------------------------------------------------------------------
// Forward layer: pre = x @ W, y = act(pre + b) (+ x).
//
// The last layer stores no activation. Its head dot products are reduced in
// 32-channel groups: each lane takes the dot of its own four channels, and
// the eight lanes of a group combine by a butterfly, which yields the same
// value on every lane and in every tile shape. The groups are then summed in
// ascending order, inside the CTA when it spans the whole width and otherwise
// by head_finalize_kernel from the per-group partials.
// ---------------------------------------------------------------------------

constexpr int kHeadGroup = 32;

struct LayerForward {
  const float* x;             //!< (nodes, din) layer input.
  const float* w;             //!< (din, dout) weight.
  const float* b;             //!< (dout,) bias or nullptr.
  float* pre;                 //!< (nodes, dout) saved pre-activation.
  float* y;                   //!< (nodes, dout) activation, null for the last.
  const float* w_head;        //!< (dout,) head weight.
  const float* b_head;        //!< (1,) head bias or nullptr.
  const double* bias_atom_e;  //!< (ntypes,) per-type energy bias.
  const long* atype;          //!< (nodes,) atom types.
  float* head_partial;        //!< (groups, nodes) partial dots of the head.
  double* e;                  //!< (nodes,) atomic energy.
  long nodes;
  int din;
  int dout;
  int residual;
};

/// e = (sum of the group partials in ascending order + b_head) +
/// bias_atom_e[atype], in fp64 after the fp32 head.
__device__ __forceinline__ double head_energy(const float* partial,
                                              long stride,
                                              int groups,
                                              const float* b_head,
                                              double bias) {
  float acc = 0.f;
  for (int c = 0; c < groups; ++c) {
    acc += partial[c * stride];
  }
  return (double)(acc + (b_head ? b_head[0] : 0.f)) + bias;
}

template <class Tile, int ACT, bool kVector>
__global__ void __launch_bounds__(Tile::threads, Tile::min_blocks)
    layer_forward_kernel(const LayerForward p) {
  __shared__ __align__(16) GemmStorage<Tile> smem;
  const LaneOrigin o = lane_origin<Tile>();
  const TileIndex tile = tile_index<Tile>(p.dout);
  Accumulators<Tile> acc;
  {
    InputStager<Tile, kVector> sa(p.x, tile.row0, p.nodes, p.din);
    WeightStager<Tile> sb(p.w, tile.col0, p.din, p.dout);
    gemm_mainloop<Tile>(smem, sa, sb, p.din, o, acc);
  }

  // === Step 1. Pre-activation, activation and the lane head dots ===
  const bool last = p.y == nullptr;
  float head[Tile::tm][Tile::tn / 4];
#pragma unroll
  for (int i = 0; i < Tile::tm; ++i) {
    const long m = tile.row0 + accumulator_row<Tile>(o, i);
#pragma unroll
    for (int g = 0; g < Tile::tn / 4; ++g) {
      const int n = tile.col0 + accumulator_col<Tile>(o, g);
      head[i][g] = 0.f;
      if (m >= p.nodes || n >= p.dout) {
        continue;
      }
      const long t = m * p.dout + n;
      const float4 z = accumulator_quad<Tile>(acc, i, g);
      store4(p.pre + t, z);
      float4 v = act_value4<ACT>(add4(z, p.b ? load4(p.b + n) : zero4()));
      if (p.residual) {
        // A square layer reads its input row as aligned quads unless the
        // input itself is misaligned.
        const float4 r =
            kVector ? load4(p.x + t)
                    : make_float4(__ldg(p.x + t), __ldg(p.x + t + 1),
                                  __ldg(p.x + t + 2), __ldg(p.x + t + 3));
        v = add4_rn(v, r);
      }
      if (!last) {
        store4(p.y + t, v);
      } else {
        const float4 wh = load4(p.w_head + n);
        float q = v.x * wh.x;
        q = fmaf(v.y, wh.y, q);
        q = fmaf(v.z, wh.z, q);
        head[i][g] = fmaf(v.w, wh.w, q);
      }
    }
  }
  if (!last) {
    return;
  }

  // === Step 2. 32-channel group partials ===
  const int groups = (p.dout + kHeadGroup - 1) / kHeadGroup;
  const bool whole = p.dout <= Tile::cols;
  constexpr int kTileGroups = Tile::cols / kHeadGroup;
  float* partial = &smem.a[0][0][0];  // (kTileGroups, rows), whole tiles
#pragma unroll
  for (int i = 0; i < Tile::tm; ++i) {
    const int r = accumulator_row<Tile>(o, i);
#pragma unroll
    for (int g = 0; g < Tile::tn / 4; ++g) {
      float q = head[i][g];
      q += __shfl_xor_sync(0xffffffffu, q, lane_col_stride<Tile>(1));
      q += __shfl_xor_sync(0xffffffffu, q, lane_col_stride<Tile>(2));
      q += __shfl_xor_sync(0xffffffffu, q, lane_col_stride<Tile>(4));
      const int group = (tile.col0 + accumulator_col<Tile>(o, g)) / kHeadGroup;
      if (lane_col<Tile>(threadIdx.x % 32) != 0 || group >= groups) {
        continue;
      }
      if (whole) {
        partial[group * Tile::rows + r] = q;
      } else if (tile.row0 + r < p.nodes) {
        p.head_partial[group * p.nodes + tile.row0 + r] = q;
      }
    }
  }
  if (!whole) {
    return;
  }

  // === Step 3. Energy of a CTA spanning the whole width ===
  static_assert(
      kTileGroups * Tile::rows <= sizeof(GemmStorage<Tile>::a) / sizeof(float),
      "the group partials fit the A buffers");
  __syncthreads();
  for (int r = threadIdx.x; r < Tile::rows; r += Tile::threads) {
    const long m = tile.row0 + r;
    if (m < p.nodes) {
      p.e[m] = head_energy(partial + r, Tile::rows, groups, p.b_head,
                           p.bias_atom_e[p.atype[m]]);
    }
  }
}

/// Energy of a head whose groups span several column tiles.
__global__ void head_finalize_kernel(long nodes,
                                     int groups,
                                     const float* __restrict__ partial,
                                     const float* __restrict__ b_head,
                                     const double* __restrict__ bias_atom_e,
                                     const long* __restrict__ atype,
                                     double* __restrict__ e) {
  const long n = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (n >= nodes) {
    return;
  }
  e[n] = head_energy(partial + n, nodes, groups, b_head, bias_atom_e[atype[n]]);
}

// ---------------------------------------------------------------------------
// Backward layer: dh_in = dpre @ W^T (+ dh) with dpre = dh * act'(pre + b),
// where dpre is formed while staging or read as emitted by the layer above.
// The layer stores dh_in or, when it emits for the layer below, that layer's
// dpre dh_in * act'(below_pre + below_b).
// ---------------------------------------------------------------------------

struct LayerBackward {
  const float* dh;         //!< (nodes, dout) output cotangent (kChain) or the
                           //!< layer's dpre (kFormed); unused by the seed.
  const double* d_e;       //!< (nodes,) energy cotangent.
  const float* w_head;     //!< (dout,) head weight.
  const float* pre;        //!< (nodes, dout) saved pre-activation.
  const float* b;          //!< (dout,) bias or nullptr.
  const float* w;          //!< (din, dout) weight.
  float* out;              //!< (nodes, din) cotangent of the layer input, or
                           //!< the dpre of the layer below if below_pre is set.
  const float* below_pre;  //!< (nodes, din) pre-activation of the layer below
                           //!< to emit dpre for, or nullptr.
  const float* below_b;    //!< (din,) bias of the layer below or nullptr.
  long nodes;
  int din;
  int dout;
  int residual;
};

template <class Tile, int ACT, Cotangent kSource>
__global__ void __launch_bounds__(Tile::threads, Tile::min_blocks)
    layer_backward_kernel(const LayerBackward p) {
  constexpr bool kSeed = kSource == Cotangent::kHead;
  __shared__ __align__(16) GemmStorage<Tile> smem;
  const LaneOrigin o = lane_origin<Tile>();
  const TileIndex tile = tile_index<Tile>(p.din);
  Accumulators<Tile> acc;
  {
    using StagerA = std::conditional_t<kSource == Cotangent::kFormed,
                                       InputStager<Tile, true>,
                                       CotangentStager<Tile, ACT, kSeed>>;
    StagerA sa = [&] {
      if constexpr (kSource == Cotangent::kFormed) {
        return StagerA(p.dh, tile.row0, p.nodes, p.dout);
      } else {
        return StagerA(p.pre, p.dh, p.d_e, p.w_head, p.b, tile.row0, p.nodes,
                       p.dout);
      }
    }();
    WeightTransposeStager<Tile> sb(p.w, tile.col0, p.dout, p.din);
    gemm_mainloop<Tile>(smem, sa, sb, p.dout, o, acc);
  }

  // Rows of a width that is not a multiple of four, such as the d_x of an
  // unpadded descriptor, and a misaligned d_x are stored per element; hidden
  // rows are whole aligned quads. A dpre emitted for the layer below is
  // dh_in * act'(below_pre + below_b), rounded exactly as the stager of that
  // layer would form it from a stored dh_in.
  const bool vector =
      p.din % 4 == 0 && reinterpret_cast<std::uintptr_t>(p.out) % 16 == 0;
#pragma unroll
  for (int i = 0; i < Tile::tm; ++i) {
    const long m = tile.row0 + accumulator_row<Tile>(o, i);
    if (m >= p.nodes) {
      continue;
    }
    const float de = kSeed && p.residual ? (float)p.d_e[m] : 0.f;
#pragma unroll
    for (int g = 0; g < Tile::tn / 4; ++g) {
      const int n = tile.col0 + accumulator_col<Tile>(o, g);
      if (n >= p.din) {
        continue;
      }
      const long t = m * p.din + n;
      float4 v = accumulator_quad<Tile>(acc, i, g);
      // Emission targets only layers without identity residual, so a layer
      // that reads an emitted dpre has no residual to add.
      if constexpr (kSource != Cotangent::kFormed) {
        if (p.residual) {
          v = kSeed ? fma4(de, load4(p.w_head + n), v)
                    : add4(v, load4(p.dh + t));
        }
      }
      if (p.below_pre) {
        const float4 bias = p.below_b ? load4(p.below_b + n) : zero4();
        const float4 slope =
            act_derivative4<ACT>(add4(load4(p.below_pre + t), bias));
        v = make_float4(v.x * slope.x, v.y * slope.y, v.z * slope.z,
                        v.w * slope.w);
      }
      if (vector) {
        store4(p.out + t, v);
      } else {
        const float c[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (n + j < p.din) {
            p.out[t + j] = c[j];
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Fused network. One CTA carries a panel of ``rows`` nodes through the whole
// network and back in a single launch: every hidden layer, the energy head,
// and the cotangent of the input. The panel's activations stay in shared
// memory, feature-major so that they are the A operand of the next GEMM as
// stored, and each layer replaces its input in place once every lane has read
// it. The weights of every GEMM, forward and backward in the order they are
// multiplied, form one stream of depth slices that flows through a ring filled
// by asynchronous copies ahead of the slice being multiplied, across the layer
// boundaries, so no GEMM waits for its first slices; the backward reads the
// transposed weights, so both directions copy rows as stored. A shape that
// serves many panels per multiprocessor keeps a short ring and lets the other
// panels hide the latency of the stream; a shape for a run of fewer panels
// than multiprocessors keeps a deep one, which then hides it alone. The
// backward reads the pre-activations of every layer but the last,
// which the forward stores; the last layer's derivative and the head
// cotangent are formed from its accumulators. The output cotangent of a
// layer, which the identity residual of the layer below adds, replaces the
// pre-activation sheet the backward has just consumed.
//
// A run of no more panels than multiprocessors splits each panel over a
// cluster of CTAs when the device keeps a cluster per panel resident at once:
// CTA ``rank`` computes the columns
// [rank * cols, (rank + 1) * cols) of every GEMM and streams only those
// columns of each weight, while it keeps the whole activation panel, which
// the next GEMM contracts over. After each epilogue the CTAs assemble the
// next A operand from their column slices through an exchange sheet in global
// memory: each CTA stores its slice there and, behind a cluster barrier, the
// next GEMM loads the slices of the others into its shared rows, each while
// the column slice before it is multiplied. The slices travel through L2,
// the fastest path between the CTAs of a cluster on sm_120, which delivers 16
// to 21 B/clk to each multiprocessor. Every column a CTA owns -- its saved
// pre-activations and its output cotangents -- stays with that CTA through
// the whole pass, so between GEMMs only the A operands move; the head
// partials of its groups move once, into the first CTA, which forms the
// energy.
//
// Every GEMM sums its contraction in ascending order with one fused
// multiply-add per term, the head reduces by the 32-channel butterfly and the
// ascending group sum of the per-layer kernels, and every elementwise step
// evaluates the same expression on the same operands, so the energy and the
// input cotangent are bitwise those of the per-layer path for every cluster
// size.
// ---------------------------------------------------------------------------

/// Hidden layers the fused kernel carries.
constexpr int kFusedLayers = 8;

template <int kWidth,
          int kCluster,
          int kThreadRows,
          int kThreadCols,
          int kWarpRows,
          int kWarpCols,
          int kMinBlocks,
          int kStages,
          int kDepth = 8,
          int kServeQuarters = 2>
struct FusedShape {
  static constexpr int width = kWidth;
  /// CTAs that split the columns of each panel between them.
  static constexpr int cluster = kCluster;
  /// Quarters of the multiprocessors that the 32-node panels of a run must
  /// cover before a single CTA per panel serves it; a smaller run leaves the
  /// device idle enough that the per-layer kernels, which spread every layer
  /// over it, finish first.
  static constexpr int serve_quarters = kServeQuarters;
  static constexpr int min_blocks = kMinBlocks;
  /// Weight slices in the ring: the one being multiplied and those ahead.
  static constexpr int stages = kStages;
  static constexpr int rows = 4 * kThreadRows * kWarpRows;
  /// Columns of every GEMM that one CTA computes.
  static constexpr int cols = 8 * kThreadCols * kWarpCols;
  static constexpr int tm = kThreadRows;
  static constexpr int tn = kThreadCols;
  /// Contraction rows of a weight slice.
  static constexpr int depth = kDepth;
  static constexpr int warp_cols = kWarpCols;
  static constexpr int threads = 32 * kWarpRows * kWarpCols;
  /// Consecutive rows of a lane: a quad, or a pair for a lane of two rows,
  /// which gives a narrow CTA twice the lanes for the same columns.
  static constexpr int row_group = kThreadRows < 4 ? kThreadRows : 4;
  static constexpr bool paired_columns = true;
  static constexpr int row_stride = rows / (kThreadRows / row_group);
  static constexpr int col_stride = cols / (kThreadCols / 4);
  static constexpr int act_stride = rows + kSkew;
  static constexpr int weight_stride = cols + kSkew;
  static constexpr int act_floats = width * act_stride;
  static constexpr int weight_floats = kStages * depth * weight_stride;
  static constexpr int input_floats = 2 * depth * act_stride;
  static constexpr size_t bytes =
      sizeof(float) * (act_floats + weight_floats + input_floats);
  static_assert(cols * kCluster == width,
                "the CTAs of a cluster split the hidden width");
  static_assert(cols % kHeadGroup == 0,
                "every CTA of a cluster owns whole head groups");
  static_assert((kThreadRows == 2 || kThreadRows % 4 == 0) &&
                    kThreadCols % 4 == 0,
                "a lane owns whole row groups and column quads");
  static_assert(rows % 32 == 0 && width % 32 == 0,
                "the panel and the head groups are whole 32-blocks");
  static_assert(kStages >= 2, "the ring holds at least one slice ahead");
  static_assert(kDepth % 8 == 0 && width % kDepth == 0,
                "the hidden contraction is whole slices");
  static_assert((width / 32) * rows <= input_floats,
                "the head partials fit the input staging");
  // Every shape fits the 99 KB of shared memory that compute capability 8.6,
  // 8.9 and 12.x let one CTA opt in to, and so every device of compute
  // capability 8.0 or later. A device that grants less, 96 KB on 7.0 and 64 KB
  // on 7.5, takes only the shapes it admits (launch_first_fit).
  static_assert(bytes <= 101376,
                "a shape fits the opt-in shared memory of 8.6, 8.9 and 12.x");
  /// The template parameters packed into one positive integer, which names
  /// the shape in a fitting-route table (fitting_route).
  static constexpr int64_t key =
      static_cast<int64_t>(kWidth) | static_cast<int64_t>(kCluster) << 9 |
      static_cast<int64_t>(kThreadRows) << 15 |
      static_cast<int64_t>(kThreadCols) << 21 |
      static_cast<int64_t>(kWarpRows) << 27 |
      static_cast<int64_t>(kWarpCols) << 33 |
      static_cast<int64_t>(kMinBlocks) << 39 |
      static_cast<int64_t>(kStages) << 45 | static_cast<int64_t>(kDepth) << 51 |
      static_cast<int64_t>(kServeQuarters) << 58;
  static_assert(kWidth < 512 && kCluster < 64 && kThreadRows < 64 &&
                    kThreadCols < 64 && kWarpRows < 64 && kWarpCols < 64 &&
                    kMinBlocks < 64 && kStages < 64 && kDepth < 128 &&
                    kServeQuarters < 16,
                "every template parameter fits its field of the key");
};

struct FusedNetwork {
  const float* x;                 //!< (nodes, din) network input.
  const long* atype;              //!< (nodes,) atom types.
  const float* w[kFusedLayers];   //!< (din_l, width) weights.
  const float* wt[kFusedLayers];  //!< (width, pitch_l) transposed weights.
  int pitch[kFusedLayers];        //!< Row pitch of wt, din_l rounded to 4.
  const float* b[kFusedLayers];   //!< (width,) biases or nullptr.
  int residual[kFusedLayers];     //!< Identity residual of each layer.
  const float* w_head;            //!< (width,) head weight.
  const float* b_head;            //!< (1,) head bias or nullptr.
  const double* bias_atom_e;      //!< (ntypes,) per-type energy bias.
  const double* d_e;              //!< (nodes,) energy cotangent.
  float* saved;                   //!< n_layer sheets of (nodes, width): the
                                  //!< pre-activations of every layer but the
                                  //!< last, then the exchange sheet.
  double* e;                      //!< (nodes,) atomic energy.
  float* d_x;                     //!< (nodes, din) input cotangent, which
                                  //!< may take the place of ``x``.
  long nodes;
  int din;
  int n_layer;
};

/// Copy 16 bytes into shared memory, zero-filled when the source is outside
/// the operand. In an image compiled for compute capability 8.0 or later the
/// copy is asynchronous and completes at the matching ``copy_async_wait``; an
/// image compiled for compute capability 7.x copies at once, which the
/// barriers of the ring order the same way.
__device__ __forceinline__ void copy_async16(void* dst,
                                             const void* src,
                                             bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(address),
               "l"(src), "r"(valid ? 16 : 0));
#else
  *static_cast<float4*>(dst) =
      valid ? *static_cast<const float4*>(src) : zero4();
#endif
}

__device__ __forceinline__ void copy_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int kPending>
__device__ __forceinline__ void copy_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
#endif
}

/// A operand of the first layer: depth columns of the row-major (nodes,
/// extent) input for the panel's rows, read per element, eight lanes per
/// 32-byte row segment, and stored feature-major. Rows past the run are
/// clamped to its last row and depths past the extent are zero. The input is
/// read through L2, since the input cotangent the kernel writes may take its
/// place (see load4_coherent).
template <class S>
struct FusedInputColumns {
  static constexpr int items = S::rows * S::depth;
  static constexpr int count = (items + S::threads - 1) / S::threads;
  const float* x;
  long row0;
  long nodes;
  int extent;
  float v[count];

  __device__ FusedInputColumns(const float* x_,
                               long row0_,
                               long nodes_,
                               int extent_)
      : x(x_), row0(row0_), nodes(nodes_), extent(extent_) {}

  __device__ __forceinline__ void fetch(int k0) {
#pragma unroll
    for (int s = 0; s < count; ++s) {
      const int item = threadIdx.x + s * S::threads;
      const long row = min(row0 + item / S::depth, nodes - 1);
      const int k = k0 + item % S::depth;
      v[s] = item < items && k < extent ? __ldcg(x + row * extent + k) : 0.f;
    }
  }

  __device__ __forceinline__ void commit(float (*a)[S::act_stride]) {
#pragma unroll
    for (int s = 0; s < count; ++s) {
      const int item = threadIdx.x + s * S::threads;
      if (item < items) {
        a[item % S::depth][item / S::depth] = v[s];
      }
    }
  }
};

/// B operand of one GEMM of the fused network: rows [0, rows_valid) and the
/// columns from ``col0`` below ``cols_valid`` of a row-major weight at row
/// ``pitch``, in ``slices`` depth slices.
struct FusedOperand {
  const float* base;
  int pitch;
  int col0;
  int cols_valid;
  int rows_valid;
  int slices;
};

/// GEMMs of the fused network in the order the kernel multiplies them: the
/// forward layers from the first, then the backward from the last layer down,
/// where the first layer takes one pass per block of ``width`` input columns.
__device__ __forceinline__ int fused_gemm_count(const FusedNetwork& p,
                                                int width) {
  return 2 * p.n_layer - 1 + (p.din + width - 1) / width;
}

/// Operand of GEMM ``gemm`` in the order of ``fused_gemm_count``, restricted
/// to the columns from ``col0`` of every block of ``width`` output columns.
template <class S>
__device__ __forceinline__ FusedOperand fused_operand(const FusedNetwork& p,
                                                      int gemm,
                                                      int col0) {
  const int last = p.n_layer - 1;
  if (gemm <= last) {
    const int din = gemm == 0 ? p.din : S::width;
    return {p.w[gemm], S::width, col0,
            S::width,  din,      (din + S::depth - 1) / S::depth};
  }
  const int back = gemm - last - 1;
  const int l = back < last ? last - back : 0;
  const int pass = back < last ? 0 : back - last;
  return {p.wt[l],    p.pitch[l], pass * S::width + col0,
          p.pitch[l], S::width,   S::width / S::depth};
}

/// The weight slices of every GEMM of the network as one stream through the
/// ring, each restricted to the columns of the CTA from ``col0``. Every lane
/// walks the same sequence: the issuing cursor runs ``stages - 1`` slices
/// ahead of the consuming one, and crosses into the next GEMM as soon as the
/// current one is fully issued, so the slices of a GEMM are in flight while
/// the epilogue of the previous one runs. A lane copies the same 16-byte
/// words of every slice, ``depth`` operand rows further down each time, so it
/// keeps a source offset per word and derives the offsets anew only when the
/// stream enters the next GEMM. Rows and columns outside the operand arrive
/// as zeros.
template <class S>
class WeightStream {
 public:
  using Ring = float (*)[S::depth][S::weight_stride];

  __device__ WeightStream(const FusedNetwork& p, Ring ring, int col0)
      : p_(p), ring_(ring), total_(fused_gemm_count(p, S::width)), col0_(col0) {
    aim();
#pragma unroll
    for (int s = 0; s < S::stages - 1; ++s) {
      issue(s);
    }
  }

  /// Wait for the next slice, issue the slice that the ring frees in its
  /// place, and return the ring row of the waited slice. The barrier also
  /// releases the slot of the previous slice, which every lane has
  /// multiplied by then.
 __device__ __forceinline__ const float (*next()) [S::weight_stride] {
   copy_async_wait<S::stages - 2>();
   __syncthreads();
   issue((consumed_ + S::stages - 1) % S::stages);
   return ring_[consumed_++ % S::stages];
 }

 private : static constexpr int kQuads = S::cols / 4;
  static constexpr int kItems = S::depth * kQuads;
  static_assert(kItems % S::threads == 0,
                "the lanes copy the 16-byte words of a slice in whole rounds");
  static constexpr int kWords = kItems / S::threads;

  /// Row and column of word ``w`` of a slice that the lane copies.
  __device__ static int line(int w) {
    return (threadIdx.x + w * S::threads) / kQuads;
  }
  __device__ static int column(int w) {
    return 4 * ((threadIdx.x + w * S::threads) % kQuads);
  }

  /// Point the offsets at the first slice of the GEMM under the cursor.
  __device__ __forceinline__ void aim() {
    operand_ = fused_operand<S>(p_, gemm_, col0_);
    slice_ = 0;
#pragma unroll
    for (int w = 0; w < kWords; ++w) {
      offset_[w] = line(w) * operand_.pitch + operand_.col0 + column(w);
    }
  }

  /// Issue the slice under the issuing cursor into ``slot``, or nothing past
  /// the end of the stream, and close the copy group either way, so that the
  /// group count of every lane stays aligned with the slices.
  __device__ __forceinline__ void issue(int slot) {
    if (gemm_ < total_) {
      const int k0 = slice_ * S::depth;
#pragma unroll
      for (int w = 0; w < kWords; ++w) {
        const bool valid = operand_.col0 + column(w) < operand_.cols_valid &&
                           k0 + line(w) < operand_.rows_valid;
        copy_async16(&ring_[slot][line(w)][column(w)],
                     operand_.base + (valid ? offset_[w] : 0), valid);
        offset_[w] += S::depth * operand_.pitch;
      }
      if (++slice_ == operand_.slices && ++gemm_ < total_) {
        aim();
      }
    }
    copy_async_commit();
  }

  const FusedNetwork& p_;
  Ring ring_;
  int total_;
  int col0_;
  FusedOperand operand_;  // of the GEMM under the issuing cursor
  int gemm_ = 0;
  int slice_ = 0;
  int consumed_ = 0;
  int offset_[kWords];  // of each word of the lane in the next slice
};

/// C = A @ B over ``slices`` depth slices of an A operand resident in shared
/// memory, feature-major, and the next GEMM of the stream. The closing
/// barrier releases the A operand to the caller's epilogue.
template <class S>
__device__ __forceinline__ void fused_resident_gemm(
    const float (*a)[S::act_stride],
    WeightStream<S>& stream,
    int slices,
    const LaneOrigin& o,
    Accumulators<S>& acc) {
  zero_accumulators<S>(acc);
  for (int s = 0; s < slices; ++s) {
    multiply_slice<S>(a + s * S::depth, stream.next(), o, acc);
  }
  __syncthreads();
}

/// Store quad ``v`` of feature-major shared rows at (feature n, row r).
template <class S>
__device__ __forceinline__ void store_feature_quad(float (*act)[S::act_stride],
                                                   int n,
                                                   int r,
                                                   float4 v) {
  act[n][r] = v.x;
  act[n + 1][r] = v.y;
  act[n + 2][r] = v.z;
  act[n + 3][r] = v.w;
}

template <class S>
__device__ __forceinline__ float4
load_feature_quad(const float (*act)[S::act_stride], int n, int r) {
  return make_float4(act[n][r], act[n + 1][r], act[n + 2][r], act[n + 3][r]);
}

/// Arrive on the barrier of the cluster, with release semantics: the memory
/// operations of the thread before it are visible to every thread of the
/// cluster once that thread's matching ``cluster_wait`` returns. Code compiled
/// for a virtual architecture below compute_90 has no clusters, and the
/// dispatch selects a cluster shape only for a kernel image compiled for
/// compute_90 or later. The cluster primitives are the compute capability 9.0
/// intrinsics of the CUDA runtime (sm_90_rt.h).
__device__ __forceinline__ void cluster_arrive() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __cluster_barrier_arrive();
#else
  __trap();
#endif
}

/// Wait, with acquire semantics, until every thread of the cluster has
/// arrived on the barrier phase of this thread's last ``cluster_arrive``.
__device__ __forceinline__ void cluster_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __cluster_barrier_wait();
#else
  __trap();
#endif
}

/// ``p`` in the shared memory of CTA ``rank`` of the cluster.
__device__ __forceinline__ float* cluster_shared(float* p, int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  return static_cast<float*>(__cluster_map_shared_rank(p, rank));
#else
  __trap();
  return p;
#endif
}

/// The column slices of the A operands that the CTAs of a cluster exchange
/// between GEMMs, through the panel's rows of the (nodes, width) exchange
/// sheet, and the head partials that the first CTA collects. A single CTA
/// holds every column itself, and every step is empty.
///
/// One sheet serves every exchange of the pass, so no CTA may store its slice
/// of an exchange while another still loads the previous one. The cluster
/// barrier therefore alternates two phases per exchange. A CTA arrives on the
/// first once it has stored its slice and waits on it before it loads the
/// slices of the others, which orders every store of the exchange before every
/// load. It arrives on the second once its last load has landed in its shared
/// rows and waits on it only before it stores its slice of the next exchange,
/// which orders every load before the next stores. The pass opens with an
/// arrival on the second phase and closes with a wait on it, so no CTA exits
/// while the first one may still read its shared memory.
template <class S>
class PanelExchange {
  /// Feature quads of a slice per lane.
  static constexpr int kQuads = S::cols * S::rows / 4 / S::threads;
  static_assert(kQuads * 4 * S::threads == S::cols * S::rows &&
                    S::rows % 16 == 0 && S::cols % 8 == 0,
                "the lanes split a slice evenly");

 public:
  /// The quads of one slice that a lane carries from the sheet.
  using Quads = float4[kQuads];

  __device__ PanelExchange(float* sheet, long row0, int rows, int rank)
      : sheet_(sheet + row0 * S::width), rows_(rows), rank_(rank) {
    release();
  }

  /// Wait until every CTA has loaded the previous exchange out of the sheet.
  __device__ __forceinline__ void acquire() const {
    if constexpr (S::cluster > 1) {
      cluster_wait();
    }
  }

  /// Store quad ``v`` of this CTA's slice at (row r, columns [n, n + 4)).
  __device__ __forceinline__ void publish(int r, int n, float4 v) const {
    if constexpr (S::cluster > 1) {
      if (r < rows_) {
        store4(sheet_ + (long)r * S::width + n, v);
      }
    }
  }

  /// Wait until every CTA of the cluster has stored its slice.
  __device__ __forceinline__ void open() const {
    if constexpr (S::cluster > 1) {
      cluster_arrive();
      cluster_wait();
    }
  }

  /// The first peer after ``peer`` in ascending rank, skipping this CTA, or
  /// S::cluster past the last.
  __device__ __forceinline__ int after(int peer) const {
    ++peer;
    return peer == rank_ ? peer + 1 : peer;
  }

  /// Load the lane's quads of the slice of CTA ``peer``; rows past the run
  /// are zeros. Two lanes cover the 32-byte segment of one row.
  __device__ __forceinline__ void load(int peer, Quads& v) const {
#pragma unroll
    for (int s = 0; s < kQuads; ++s) {
      v[s] = row(s) < rows_ ? load4_coherent(sheet_ + (long)row(s) * S::width +
                                             peer * S::cols + feature(s))
                            : zero4();
    }
  }

  /// Store the lane's quads of the slice of CTA ``peer`` feature-major. A
  /// warp stores two quads of 16 rows, which lands the four scalar stores of
  /// each quad on 32 distinct banks.
  __device__ __forceinline__ void store(float (*act)[S::act_stride],
                                        int peer,
                                        const Quads& v) const {
#pragma unroll
    for (int s = 0; s < kQuads; ++s) {
      store_feature_quad<S>(act, peer * S::cols + feature(s), row(s), v[s]);
    }
  }

  /// In the first CTA, read the (width / 32, rows) head partials of the
  /// groups that the other CTAs own, which the barrier of the seed exchange
  /// has made visible, into its own partial rows.
  __device__ __forceinline__ void collect(float* partial) const {
    if constexpr (S::cluster > 1) {
      constexpr int kOwn = S::cols / kHeadGroup * S::rows;
      constexpr int kAll = S::width / kHeadGroup * S::rows;
      if (rank_ == 0) {
        for (int item = kOwn + threadIdx.x; item < kAll; item += S::threads) {
          partial[item] = cluster_shared(partial, item / kOwn)[item];
        }
      }
    }
  }

  /// Release the sheet to the next exchange.
  __device__ __forceinline__ void release() const {
    if constexpr (S::cluster > 1) {
      cluster_arrive();
    }
  }

 private:
  /// Row and first feature of the lane's quad ``s`` within a slice.
  __device__ static int row(int s) {
    return (threadIdx.x + s * S::threads) / 2 % S::rows;
  }
  __device__ static int feature(int s) {
    const int q = threadIdx.x + s * S::threads;
    return 4 * (q % 2) + 8 * (q / (2 * S::rows));
  }

  float* sheet_;
  int rows_;
  int rank_;
};

/// C = A @ B over a resident A operand of ``width`` features. With
/// ``gather``, the A operand is that of an open exchange: the CTA's own
/// columns are in place, and the slices of the other CTAs stream in from the
/// exchange sheet in ascending column order, the contraction order, each
/// loaded while the slices before it are multiplied and stored just before
/// its first use, so the GEMM waits on the transfer of the first slice only.
/// Storing the last of them releases the sheet. Without, the operand is
/// complete already.
template <class S>
__device__ __forceinline__ void fused_gathered_gemm(
    float (*act)[S::act_stride],
    WeightStream<S>& stream,
    const PanelExchange<S>& exchange,
    bool gather,
    const LaneOrigin& o,
    Accumulators<S>& acc) {
  constexpr int kSlices = S::width / S::depth;
  if constexpr (S::cluster == 1) {
    fused_resident_gemm<S>(act, stream, kSlices, o, acc);
  } else {
    static_assert(S::cols % S::depth == 0,
                  "every column slice is whole depth slices");
    constexpr int kPerPeer = S::cols / S::depth;
    zero_accumulators<S>(acc);
    typename PanelExchange<S>::Quads incoming;
    int peer = gather ? exchange.after(-1) : S::cluster;
    if (peer < S::cluster) {
      exchange.load(peer, incoming);
    }
    for (int s = 0; s < kSlices; ++s) {
      if (s == peer * kPerPeer) {
        exchange.store(act, peer, incoming);
        peer = exchange.after(peer);
        if (peer < S::cluster) {
          exchange.load(peer, incoming);
        } else {
          exchange.release();
        }
      }
      multiply_slice<S>(act + s * S::depth, stream.next(), o, acc);
    }
    __syncthreads();
  }
}

/// e = (sum of the group partials in ascending order + b_head) +
/// bias_atom_e[atype] for the ``rows`` rows of the panel at ``row0``.
template <class S>
__device__ __forceinline__ void panel_energy(const FusedNetwork& p,
                                             const float* partial,
                                             long row0,
                                             int rows) {
  for (int r = threadIdx.x; r < rows; r += S::threads) {
    const long m = row0 + r;
    p.e[m] = head_energy(partial + r, S::rows, S::width / kHeadGroup, p.b_head,
                         p.bias_atom_e[p.atype[m]]);
  }
}

template <class S, int ACT>
__global__ void __launch_bounds__(S::threads, S::min_blocks)
    fused_network_kernel(const FusedNetwork p) {
  extern __shared__ __align__(16) float fused_smem[];
  auto act = reinterpret_cast<float (*)[S::act_stride]>(fused_smem);
  auto ring = reinterpret_cast<float (*)[S::depth][S::weight_stride]>(
      fused_smem + S::act_floats);
  auto xbuf = reinterpret_cast<float (*)[S::depth][S::act_stride]>(
      fused_smem + S::act_floats + S::weight_floats);
  // (width / 32, rows) partials of the energy head, in the input staging of
  // the first layer, which the forward has passed by then.
  float* const partial = &xbuf[0][0][0];
  const LaneOrigin o = lane_origin<S>();
  // A cluster is S::cluster consecutive CTAs of the grid, and its CTA
  // ``rank`` owns the columns from ``col0`` of every GEMM.
  const int rank = blockIdx.x % S::cluster;
  const int col0 = rank * S::cols;
  const long row0 = (long)(blockIdx.x / S::cluster) * S::rows;
  // Rows of the panel inside the run. Panel rows past it compute on the run's
  // last row, and in a cluster on zeros in the columns of the other CTAs;
  // every GEMM keeps rows apart, so their values reach only rows that are
  // never stored.
  const int rows = (int)min((long)S::rows, p.nodes - row0);
  const int last = p.n_layer - 1;
  const long sheet = p.nodes * S::width;
  const PanelExchange<S> exchange(p.saved + last * sheet, row0, rows, rank);
  Accumulators<S> acc;

  // Energy cotangent of accumulator row ``i``; rows past the run carry none.
  const auto de = [&](int i) {
    const int r = accumulator_row<S>(o, i);
    return r < rows ? (float)p.d_e[row0 + r] : 0.f;
  };

  // === Step 1. Forward through every hidden layer ===
  WeightStream<S> stream(p, ring, col0);
  for (int l = 0; l <= last; ++l) {
    if (l == 0) {
      FusedInputColumns<S> sa(p.x, row0, p.nodes, p.din);
      zero_accumulators<S>(acc);
      const int slices = (p.din + S::depth - 1) / S::depth;
      sa.fetch(0);
      sa.commit(xbuf[0]);
      for (int s = 0; s < slices; ++s) {
        const bool next = s + 1 < slices;
        if (next) {
          sa.fetch((s + 1) * S::depth);
        }
        multiply_slice<S>(xbuf[s & 1], stream.next(), o, acc);
        if (next) {
          sa.commit(xbuf[(s + 1) & 1]);
        }
      }
      __syncthreads();
    } else {
      exchange.open();
      fused_gathered_gemm<S>(act, stream, exchange, true, o, acc);
    }

    // Pre-activation, activation and identity residual. The last layer
    // reduces its activation against the head weight and leaves the seed
    // cotangent d_e * w_head * act'(pre + b) of the backward in its place.
    // The parameter quads are loaded before the barrier and the stores of
    // the epilogue, so that their L2 round trips overlap.
    float4 bias[S::tn / 4];
    float4 wh[S::tn / 4];
#pragma unroll
    for (int g = 0; g < S::tn / 4; ++g) {
      const int n = col0 + accumulator_col<S>(o, g);
      bias[g] = p.b[l] ? load4(p.b[l] + n) : zero4();
      wh[g] = l == last ? load4(p.w_head + n) : zero4();
    }
    exchange.acquire();
    // Only the residual of the first layer reads the input, and a layer has a
    // residual only when it is square, so its input rows are whole quads.
    const bool vector_input = reinterpret_cast<std::uintptr_t>(p.x) % 16 == 0;
    float head[S::tm][S::tn / 4];
#pragma unroll
    for (int i = 0; i < S::tm; ++i) {
      const int r = accumulator_row<S>(o, i);
      const long mc = row0 + min(r, rows - 1);
#pragma unroll
      for (int g = 0; g < S::tn / 4; ++g) {
        const int n = col0 + accumulator_col<S>(o, g);
        const float4 z = accumulator_quad<S>(acc, i, g);
        if (l < last && r < rows) {
          store4(p.saved + l * sheet + mc * S::width + n, z);
        }
        float4 v = act_value4<ACT>(add4(z, bias[g]));
        if (p.residual[l]) {
          float4 rv;
          if (l > 0) {
            rv = load_feature_quad<S>(act, n, r);
          } else if (vector_input) {
            rv = load4_coherent(p.x + mc * p.din + n);
          } else {
            const float* xr = p.x + mc * p.din + n;
            rv = make_float4(__ldcg(xr), __ldcg(xr + 1), __ldcg(xr + 2),
                             __ldcg(xr + 3));
          }
          v = add4_rn(v, rv);
        }
        if (l == last) {
          float q = v.x * wh[g].x;
          q = fmaf(v.y, wh[g].y, q);
          q = fmaf(v.z, wh[g].z, q);
          head[i][g] = fmaf(v.w, wh[g].w, q);
          const float4 dh = scale4(de(i), wh[g]);
          const float4 slope = act_derivative4<ACT>(add4(z, bias[g]));
          v = make_float4(dh.x * slope.x, dh.y * slope.y, dh.z * slope.z,
                          dh.w * slope.w);
        }
        store_feature_quad<S>(act, n, r, v);
        exchange.publish(r, n, v);
      }
    }
    if (l < last) {
      continue;
    }

    // === Step 2. Energy head: 32-channel group partials, ascending groups ===
    // Each CTA reduces the groups of its own columns; the first CTA of a
    // cluster collects those of the others once the seed exchange opens and
    // sums every group in order.
#pragma unroll
    for (int i = 0; i < S::tm; ++i) {
      const int r = accumulator_row<S>(o, i);
#pragma unroll
      for (int g = 0; g < S::tn / 4; ++g) {
        float q = head[i][g];
        q += __shfl_xor_sync(0xffffffffu, q, lane_col_stride<S>(1));
        q += __shfl_xor_sync(0xffffffffu, q, lane_col_stride<S>(2));
        q += __shfl_xor_sync(0xffffffffu, q, lane_col_stride<S>(4));
        if (lane_col<S>(threadIdx.x % 32) == 0) {
          partial[((col0 + accumulator_col<S>(o, g)) / kHeadGroup) * S::rows +
                  r] = q;
        }
      }
    }
  }

  // === Step 3. Backward from the seed cotangent to the input ===
  for (int l = last; l >= 0; --l) {
    const int din = l == 0 ? p.din : S::width;
    const int passes = (din + S::width - 1) / S::width;
    for (int pass = 0; pass < passes; ++pass) {
      const int n0 = pass * S::width + col0;
      if (pass == 0) {
        exchange.open();
        if (l == last) {
          exchange.collect(partial);
          __syncthreads();
          if (rank == 0) {
            panel_energy<S>(p, partial, row0, rows);
          }
        }
      }
      fused_gathered_gemm<S>(act, stream, exchange, pass == 0, o, acc);

      // The global operands of the epilogue: the bias of the layer below,
      // and per quad the factor of the identity residual -- the head weight
      // at the last layer, which the energy cotangent scales, and below it
      // the output cotangent that replaced the pre-activation sheet of the
      // layer -- and the stored pre-activation of the layer below. A tile of
      // at most 16 accumulators in a CTA of at most 256 threads loads the
      // operands of every quad before the barrier and its stores, as the
      // forward loads its parameters, so that their round trips overlap each
      // other and the wait of a cluster; every other shape loads those of
      // each quad where it uses them. The two placements compute the same
      // values; the split keeps every shape within its register budget
      // without spills on sm_80, sm_90 and sm_120.
      const float* bb = l > 0 ? p.b[l - 1] : nullptr;
      float4 bias[S::tn / 4];
#pragma unroll
      for (int g = 0; g < S::tn / 4; ++g) {
        bias[g] = bb ? load4(bb + n0 + accumulator_col<S>(o, g)) : zero4();
      }
      float4 factor[S::tm][S::tn / 4];
      float4 pre[S::tm][S::tn / 4];
      const auto load_operands = [&](int i, int g) {
        const int n = n0 + accumulator_col<S>(o, g);
        const long mc = row0 + min(accumulator_row<S>(o, i), rows - 1);
        factor[i][g] =
            !p.residual[l] ? zero4()
            : l == last
                ? load4(p.w_head + n)
                : load4_coherent(p.saved + l * sheet + mc * S::width + n);
        pre[i][g] =
            l > 0
                ? load4_coherent(p.saved + (l - 1) * sheet + mc * S::width + n)
                : zero4();
      };
      constexpr bool kAhead = S::tm * S::tn <= 16 && S::threads <= 256;
      if constexpr (kAhead) {
#pragma unroll
        for (int i = 0; i < S::tm; ++i) {
#pragma unroll
          for (int g = 0; g < S::tn / 4; ++g) {
            load_operands(i, g);
          }
        }
      }
      if (l > 0) {
        exchange.acquire();
      }
      const bool vector_output =
          p.din % 4 == 0 && reinterpret_cast<std::uintptr_t>(p.d_x) % 16 == 0;
#pragma unroll
      for (int i = 0; i < S::tm; ++i) {
        const int r = accumulator_row<S>(o, i);
        const long mc = row0 + min(r, rows - 1);
#pragma unroll
        for (int g = 0; g < S::tn / 4; ++g) {
          const int n = n0 + accumulator_col<S>(o, g);
          if constexpr (!kAhead) {
            load_operands(i, g);
          }
          float4 v = accumulator_quad<S>(acc, i, g);
          if (p.residual[l]) {
            // v + d_e * w_head with one rounding, as the per-layer path forms
            // it; below the last layer the scale of one leaves a plain sum.
            v = fma4(l == last ? de(i) : 1.f, factor[i][g], v);
          }
          if (l > 0) {
            // dpre of the layer below, from its stored pre-activation, which
            // the cotangent then replaces.
            const float4 slope = act_derivative4<ACT>(add4(pre[i][g], bias[g]));
            const float4 dpre = make_float4(v.x * slope.x, v.y * slope.y,
                                            v.z * slope.z, v.w * slope.w);
            store_feature_quad<S>(act, n, r, dpre);
            exchange.publish(r, n, dpre);
            if (r < rows) {
              store4(p.saved + (l - 1) * sheet + mc * S::width + n, v);
            }
          } else if (r < rows && n < p.din) {
            float* out = p.d_x + mc * p.din + n;
            if (vector_output) {
              store4(out, v);
            } else {
              const float c[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if (n + q < p.din) {
                  out[q] = c[q];
                }
              }
            }
          }
        }
      }
    }
  }
  exchange.acquire();
}

/// Destinations of the transposed weight sheets of one network, passed by
/// value so that a captured launch carries them in its parameters.
struct TransposedSheets {
  float* wt[kFusedLayers];
};

/// Edge of the square tiles in which transpose_weights_kernel moves a weight.
constexpr int kTransposeTile = 32;

/// Transpose every weight of the network into the row-major (width, pitch)
/// layout the backward of the fused network streams, zero-padded past din.
/// CTA (x, y, l) of 32 x 8 threads moves the 32 x 32 tile of layer l at
/// weight rows from 32 * x and columns from 32 * y through shared memory, so
/// that each warp reads 32 consecutive floats of a weight row and writes 32
/// consecutive floats of a sheet row; the extra column of the tile keeps the
/// transposed reads on distinct banks.
__global__ void transpose_weights_kernel(const FusedNetwork p,
                                         int width,
                                         const TransposedSheets sheets) {
  __shared__ float tile[kTransposeTile][kTransposeTile + 1];
  const int l = blockIdx.z;
  const int din = l == 0 ? p.din : width;
  const int pitch = p.pitch[l];
  const int n0 = blockIdx.x * kTransposeTile;
  const int k0 = blockIdx.y * kTransposeTile;
  if (n0 >= pitch) {
    return;
  }
  for (int i = threadIdx.y; i < kTransposeTile; i += blockDim.y) {
    const int n = n0 + i;
    const int k = k0 + threadIdx.x;
    tile[i][threadIdx.x] =
        n < din && k < width ? p.w[l][(long)n * width + k] : 0.f;
  }
  __syncthreads();
  for (int i = threadIdx.y; i < kTransposeTile; i += blockDim.y) {
    const int k = k0 + i;
    const int n = n0 + threadIdx.x;
    if (k < width && n < pitch) {
      sheets.wt[l][(long)k * pitch + n] = tile[threadIdx.x][i];
    }
  }
}

// ---------------------------------------------------------------------------
// Launch.
// ---------------------------------------------------------------------------

/// Dispatch a kernel template over the two supported activation codes.
template <typename Fn>
void dispatch_activation(long act, Fn&& launch) {
  if (act == 0) {
    launch(std::integral_constant<int, 0>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }
}

/// Dispatch over the two tiles. A run small enough to leave the large tile's
/// grid short of a few waves takes the small tile; the threshold of four and
/// a half small CTAs per SM is where the two break even on the fitting shapes
/// of the DPA4C presets on sm_120.
template <typename Fn>
void dispatch_tile(long nodes, int width, Fn&& launch) {
  const long sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long small_ctas =
      ceil_div(nodes, SmallTile::rows) * ceil_div(width, SmallTile::cols);
  if (2 * small_ctas <= 9 * sms) {
    launch(SmallTile{});
  } else {
    launch(LargeTile{});
  }
}

template <class Tile>
unsigned grid_size(long nodes, int width) {
  return (unsigned)(ceil_div(nodes, Tile::rows) * ceil_div(width, Tile::cols));
}

/// Column tiles of a GEMM with ``width`` outputs under its routed tile.
long column_tiles(long nodes, int width) {
  long tiles = 0;
  dispatch_tile(nodes, width, [&](auto tile) {
    tiles = ceil_div(width, decltype(tile)::cols);
  });
  return tiles;
}

/// Launch one forward layer. Returns whether the energy still needs
/// head_finalize_kernel, which is the case when the last layer is wider than
/// a tile.
bool launch_layer_forward(cudaStream_t stream,
                          const LayerForward& p,
                          long act) {
  bool finalize = false;
  dispatch_tile(p.nodes, p.dout, [&](auto tile) {
    using Tile = decltype(tile);
    const unsigned grid = grid_size<Tile>(p.nodes, p.dout);
    const bool vector =
        p.din % 4 == 0 && reinterpret_cast<std::uintptr_t>(p.x) % 16 == 0;
    dispatch_activation(act, [&](auto tag) {
      constexpr int kAct = decltype(tag)::value;
      if (vector) {
        layer_forward_kernel<Tile, kAct, true>
            <<<grid, Tile::threads, 0, stream>>>(p);
      } else {
        layer_forward_kernel<Tile, kAct, false>
            <<<grid, Tile::threads, 0, stream>>>(p);
      }
    });
    finalize = p.y == nullptr && p.dout > Tile::cols;
  });
  return finalize;
}

void launch_layer_backward(cudaStream_t stream,
                           const LayerBackward& p,
                           Cotangent source,
                           long act) {
  // An input of zero width has no cotangent to compute.
  if (p.din == 0) {
    return;
  }
  dispatch_tile(p.nodes, p.din, [&](auto tile) {
    using Tile = decltype(tile);
    const unsigned grid = grid_size<Tile>(p.nodes, p.din);
    dispatch_activation(act, [&](auto tag) {
      constexpr int kAct = decltype(tag)::value;
      switch (source) {
        case Cotangent::kHead:
          layer_backward_kernel<Tile, kAct, Cotangent::kHead>
              <<<grid, Tile::threads, 0, stream>>>(p);
          break;
        case Cotangent::kChain:
          layer_backward_kernel<Tile, kAct, Cotangent::kChain>
              <<<grid, Tile::threads, 0, stream>>>(p);
          break;
        case Cotangent::kFormed:
          layer_backward_kernel<Tile, kAct, Cotangent::kFormed>
              <<<grid, Tile::threads, 0, stream>>>(p);
          break;
      }
    });
  });
}

}  // namespace

namespace {

FittingLayerPlan validate_fitting_forward_inputs(
    const char* operation,
    const torch::Tensor& x,
    const torch::Tensor& atype,
    const std::vector<torch::Tensor>& ws,
    const torch::Tensor& bias_atom_e) {
  TORCH_CHECK(x.dim() == 2 && x.is_cuda() && x.is_contiguous() &&
                  x.scalar_type() == torch::kFloat32,
              operation, ": x must be contiguous CUDA fp32 with shape (N, D)");
  TORCH_CHECK(atype.dim() == 1 && atype.size(0) == x.size(0) &&
                  atype.is_cuda() && atype.is_contiguous() &&
                  atype.device() == x.device() &&
                  atype.scalar_type() == torch::kInt64,
              operation,
              ": atype must be contiguous CUDA int64 with shape (N,) on "
              "the device of x");
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(
      plan.n_layer > 0 && ws[0].dim() == 2 && ws[0].size(0) == x.size(1),
      operation, ": the first fitting weight must match the input width");
  TORCH_CHECK(bias_atom_e.dim() == 1 && bias_atom_e.is_cuda() &&
                  bias_atom_e.is_contiguous() &&
                  bias_atom_e.device() == x.device() &&
                  bias_atom_e.scalar_type() == torch::kFloat64,
              operation,
              ": bias_atom_e must be contiguous CUDA fp64 on the device of x");
  return plan;
}

/// Whether `t` is a contiguous tensor of `dtype` on the current CUDA device,
/// or empty when `optional`.
bool resident_parameter(const torch::Tensor& t,
                        at::ScalarType dtype,
                        bool optional) {
  if (optional && t.numel() == 0) {
    return true;
  }
  return t.is_cuda() && t.get_device() == c10::cuda::current_device() &&
         t.is_contiguous() && t.scalar_type() == dtype;
}

bool aligned_parameter(const torch::Tensor& t) {
  return t.numel() == 0 ||
         reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0;
}

/// The kernels read the weight of every layer as aligned float4 from a
/// contiguous fp32 (din, dout) matrix on the device they run on, the layers
/// chain, and every hidden width is a positive multiple of four.
void check_weights(const std::vector<torch::Tensor>& ws) {
  TORCH_CHECK(!ws.empty(), "graph_fitting: the network has no layer");
  for (size_t l = 0; l < ws.size(); ++l) {
    TORCH_CHECK(resident_parameter(ws[l], torch::kFloat32, false) &&
                    aligned_parameter(ws[l]) && ws[l].dim() == 2,
                "graph_fitting: the weight of layer ", l,
                " must be a contiguous, 16-byte aligned fp32 matrix on the "
                "current CUDA device");
    TORCH_CHECK(ws[l].size(1) > 0 && ws[l].size(1) % 4 == 0,
                "graph_fitting: the width of layer ", l,
                " must be a positive multiple of four, got ", ws[l].size(1));
    TORCH_CHECK(l == 0 || ws[l].size(0) == ws[l - 1].size(1),
                "graph_fitting: layer ", l, " takes ", ws[l].size(0),
                " features, but layer ", l - 1, " yields ", ws[l - 1].size(1));
  }
}

/// `check_weights` together with the biases, the residual flags and the head
/// weight, which the kernels read like the weights.
void check_parameters(const std::vector<torch::Tensor>& ws,
                      const std::vector<torch::Tensor>& bs,
                      const std::vector<int64_t>& resnets,
                      const torch::Tensor& w_head) {
  check_weights(ws);
  TORCH_CHECK(bs.size() == ws.size() && resnets.size() == ws.size(),
              "graph_fitting: expected one bias and one residual flag per "
              "fitting layer, got ",
              bs.size(), " and ", resnets.size(), " for ", ws.size(),
              " layers");
  for (size_t l = 0; l < ws.size(); ++l) {
    TORCH_CHECK(resident_parameter(bs[l], torch::kFloat32, true) &&
                    aligned_parameter(bs[l]) &&
                    (bs[l].numel() == 0 || bs[l].numel() == ws[l].size(1)),
                "graph_fitting: the bias of layer ", l,
                " must be empty or a contiguous, 16-byte aligned fp32 vector "
                "of the layer width on the current CUDA device");
  }
  TORCH_CHECK(resident_parameter(w_head, torch::kFloat32, false) &&
                  aligned_parameter(w_head) &&
                  w_head.numel() == ws.back().size(1),
              "graph_fitting: the head weight must be a contiguous, 16-byte "
              "aligned fp32 tensor of ",
              ws.back().size(1), " entries on the current CUDA device");
}

/// `check_parameters` together with the head bias and the per-type energy
/// bias of a pass that forms the energy.
void check_energy_parameters(const std::vector<torch::Tensor>& ws,
                             const std::vector<torch::Tensor>& bs,
                             const std::vector<int64_t>& resnets,
                             const torch::Tensor& w_head,
                             const torch::Tensor& b_head,
                             const torch::Tensor& bias_atom_e) {
  check_parameters(ws, bs, resnets, w_head);
  TORCH_CHECK(
      resident_parameter(b_head, torch::kFloat32, true) && b_head.numel() <= 1,
      "graph_fitting: the head bias must be empty or a single fp32 "
      "value on the current CUDA device");
  TORCH_CHECK(bias_atom_e.dim() == 1 &&
                  resident_parameter(bias_atom_e, torch::kFloat64, false),
              "graph_fitting: bias_atom_e must be a contiguous fp64 vector on "
              "the current CUDA device");
}

}  // namespace

// Evaluate the network over one contiguous run of nodes. Every full-width
// tensor is indexed from the run's first node, so the same code serves the
// whole node axis and a single tile of it. ``saved`` and ``activation`` are
// 16-byte aligned and sized for the run, not for the system.
void fitting_forward_range(cudaStream_t stream,
                           const FittingLayerPlan& plan,
                           const float* x,
                           long input_width,
                           const long* atype,
                           const std::vector<torch::Tensor>& ws,
                           const std::vector<torch::Tensor>& bs,
                           const std::vector<int64_t>& resnets,
                           const torch::Tensor& w_head,
                           const torch::Tensor& b_head,
                           const torch::Tensor& bias_atom_e,
                           int64_t act,
                           long run_nodes,
                           float* saved,
                           float* const activation[2],
                           double* e) {
  check_energy_parameters(ws, bs, resnets, w_head, b_head, bias_atom_e);
  const float* cur = x;
  int din = (int)input_width;
  LayerForward p{};
  p.w_head = w_head.data_ptr<float>();
  p.b_head = b_head.numel() ? b_head.data_ptr<float>() : nullptr;
  p.bias_atom_e = bias_atom_e.data_ptr<double>();
  p.atype = atype;
  p.e = e;
  p.nodes = run_nodes;
  bool finalize = false;
  for (int l = 0; l < plan.n_layer; ++l) {
    const int dout = (int)ws[l].size(1);
    const bool last = l == plan.n_layer - 1;
    p.x = cur;
    p.w = ws[l].data_ptr<float>();
    p.b = bs[l].numel() ? bs[l].data_ptr<float>() : nullptr;
    p.pre = saved + plan.offset[l] * run_nodes;
    // The last layer stores no activation; its slot, which it does not read,
    // holds the partial head dots instead.
    p.y = last ? nullptr : activation[l & 1];
    p.head_partial = last ? activation[l & 1] : nullptr;
    p.din = din;
    p.dout = dout;
    p.residual = resnets[l] && dout == din ? 1 : 0;
    finalize = launch_layer_forward(stream, p, act);
    FITTING_CHECK_LAUNCH("graph_fitting layer");
    cur = p.y;
    din = dout;
  }
  if (finalize) {
    head_finalize_kernel<<<ceil_div(run_nodes, 256), 256, 0, stream>>>(
        run_nodes, (int)ceil_div(din, kHeadGroup), p.head_partial, p.b_head,
        p.bias_atom_e, atype, e);
    FITTING_CHECK_LAUNCH("graph_fitting head");
  }
}

// Propagate the head cotangent of one run of nodes back to the input. ``dh``
// and ``dh_next`` are 16-byte aligned (run, width_max) scratch, written from
// the last layer down: ``dh`` is needed by networks of two or more layers and
// ``dh_next`` by networks of three or more. ``d_x`` is indexed from the run's
// first node.
void fitting_backward_range(cudaStream_t stream,
                            const FittingLayerPlan& plan,
                            const double* d_e,
                            const float* saved,
                            const std::vector<torch::Tensor>& ws,
                            const std::vector<torch::Tensor>& bs,
                            const std::vector<int64_t>& resnets,
                            const torch::Tensor& w_head,
                            int64_t act,
                            long run_nodes,
                            float* dh,
                            float* dh_next,
                            float* d_x) {
  check_parameters(ws, bs, resnets, w_head);
  // Layer l writes its input cotangent to the slot its successor did not
  // write, or to d_x for the first layer. Above a layer without identity
  // residual whose GEMM spans several column tiles it writes that layer's dpre
  // instead, so the dpre is formed once rather than once per column tile. A
  // GEMM of a single column tile forms its dpre once in its pipelined main
  // loop, which costs less than forming it in this epilogue.
  const auto residual = [&](int l) {
    return resnets[l] && ws[l].size(0) == ws[l].size(1);
  };
  float* const slot[2] = {dh, dh_next};
  LayerBackward p{};
  p.d_e = d_e;
  p.w_head = w_head.data_ptr<float>();
  p.nodes = run_nodes;
  Cotangent source = Cotangent::kHead;
  for (int l = plan.n_layer - 1; l >= 0; --l) {
    const bool emit = l > 0 && !residual(l - 1) &&
                      column_tiles(run_nodes, (int)ws[l - 1].size(0)) > 1;
    p.pre = saved + plan.offset[l] * run_nodes;
    p.b = bs[l].numel() ? bs[l].data_ptr<float>() : nullptr;
    p.w = ws[l].data_ptr<float>();
    p.out = l > 0 ? slot[(plan.n_layer - 1 - l) & 1] : d_x;
    p.below_pre = emit ? saved + plan.offset[l - 1] * run_nodes : nullptr;
    p.below_b =
        emit && bs[l - 1].numel() ? bs[l - 1].data_ptr<float>() : nullptr;
    p.din = (int)ws[l].size(0);
    p.dout = (int)ws[l].size(1);
    p.residual = residual(l) ? 1 : 0;
    launch_layer_backward(stream, p, source, act);
    FITTING_CHECK_LAUNCH("graph_fitting backward layer");
    p.dh = p.out;
    source = emit ? Cotangent::kFormed : Cotangent::kChain;
  }
}

namespace {

/// Launch configuration of ``panels`` panels under shape S: one CTA per
/// panel, in clusters of S::cluster consecutive CTAs. ``cluster`` holds the
/// cluster attribute that the configuration points to.
template <class S>
cudaLaunchConfig_t fused_launch_config(long panels,
                                       cudaStream_t stream,
                                       cudaLaunchAttribute* cluster) {
  cluster->id = cudaLaunchAttributeClusterDimension;
  cluster->val.clusterDim.x = S::cluster;
  cluster->val.clusterDim.y = 1;
  cluster->val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((unsigned)(panels * S::cluster));
  config.blockDim = dim3(S::threads);
  config.dynamicSmemBytes = S::bytes;
  config.stream = stream;
  config.attrs = cluster;
  config.numAttrs = S::cluster > 1 ? 1 : 0;
  return config;
}

/// Opt the kernel of shape S into its dynamic shared memory and, for a
/// cluster above the portable size of eight, into non-portable cluster sizes,
/// up to the largest cluster that the device launches.
template <class S, int ACT>
cudaError_t set_fused_attributes() {
  const auto kernel = fused_network_kernel<S, ACT>;
  if constexpr (S::cluster > 8) {
    const cudaError_t error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (error != cudaSuccess) {
      return error;
    }
  }
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S::bytes);
}

/// `set_fused_attributes` for a launch, which fails when the device refuses
/// them.
template <class S, int ACT>
void configure_fused_kernel() {
  const cudaError_t error = set_fused_attributes<S, ACT>();
  if (error != cudaSuccess) {
    // The refusal is cleared, so that no later launch check reports it.
    (void)cudaGetLastError();
    TORCH_CHECK(false,
                "graph_fitting fused network: cannot configure "
                "clusters of ",
                S::cluster, " CTAs with ", S::bytes,
                " bytes of dynamic shared memory: ", cudaGetErrorString(error));
  }
}

/// Whether a query of the fused kernel succeeded. A refused query leaves its
/// error behind, which is cleared, so that the shape it asked about is merely
/// not used.
bool accepted(cudaError_t error) {
  if (error == cudaSuccess) {
    return true;
  }
  (void)cudaGetLastError();
  return false;
}

/// What the current device runs of shape S, which no run changes and which
/// each device therefore answers once.
struct FusedCapacity {
  /// Lower virtual architecture of the code that the device runs for the two
  /// activations: that of the native code it selected, which may be an
  /// earlier minor revision of its own (compute_80 on sm_86), or the PTX
  /// target that the driver compiled from.
  int version;
  /// Clusters of the shape that the device keeps resident at once for both
  /// activations; zero for a single-CTA shape, and zero unless the device
  /// launches clusters, the images were compiled for compute_90 or later, and
  /// it launches clusters of S::cluster CTAs of this kernel, which a device or
  /// partition with fewer multiprocessors per GPC may not. The runtime counts
  /// the resident clusters against the multiprocessors of each GPC and seats
  /// several CTAs on one multiprocessor where their registers and shared
  /// memory allow, which the narrow network does.
  int clusters;
  /// CTAs of a single-CTA shape that a multiprocessor keeps resident at once
  /// for both activations, which sets the waves in which a run of its panels
  /// passes; zero for a cluster shape and where the device refuses the query.
  int resident;
};

template <class S>
FusedCapacity fused_capacity() {
  static std::mutex mutex;
  static std::map<c10::DeviceIndex, FusedCapacity> capacities;
  const c10::DeviceIndex device = c10::cuda::current_device();
  const std::lock_guard<std::mutex> lock(mutex);
  const auto known = capacities.find(device);
  if (known != capacities.end()) {
    return known->second;
  }
  // A query the device or the driver refuses reads as a version of zero or as
  // no resident cluster, so the shape is not used and the per-layer kernels
  // serve the run instead of the call failing.
  FusedCapacity capacity{std::numeric_limits<int>::max(), 0,
                         S::cluster == 1 ? std::numeric_limits<int>::max() : 0};
  for (const long act : {0L, 1L}) {
    dispatch_activation(act, [&](auto tag) {
      constexpr int kAct = decltype(tag)::value;
      const auto kernel = fused_network_kernel<S, kAct>;
      cudaFuncAttributes attributes{};
      const bool known = accepted(cudaFuncGetAttributes(&attributes, kernel));
      capacity.version =
          std::min(capacity.version, known ? attributes.ptxVersion : 0);
      if constexpr (S::cluster == 1) {
        int resident = 0;
        if (!accepted(set_fused_attributes<S, kAct>()) ||
            !accepted(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident, kernel, S::threads, S::bytes))) {
          resident = 0;
        }
        capacity.resident = std::min(capacity.resident, resident);
      }
    });
  }
  int cluster_launch = 0;
  const bool launches_clusters =
      accepted(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch,
                                      device)) &&
      cluster_launch != 0;
  if (S::cluster > 1 && launches_clusters && capacity.version >= 90) {
    capacity.clusters = std::numeric_limits<int>::max();
    for (const long act : {0L, 1L}) {
      dispatch_activation(act, [&](auto tag) {
        constexpr int kAct = decltype(tag)::value;
        const auto kernel = fused_network_kernel<S, kAct>;
        cudaLaunchAttribute cluster;
        const cudaLaunchConfig_t config =
            fused_launch_config<S>(1, nullptr, &cluster);
        int largest = 0;
        int clusters = 0;
        if (!accepted(set_fused_attributes<S, kAct>()) ||
            !accepted(cudaOccupancyMaxPotentialClusterSize(&largest, kernel,
                                                           &config)) ||
            largest < S::cluster ||
            !accepted(
                cudaOccupancyMaxActiveClusters(&clusters, kernel, &config))) {
          clusters = 0;
        }
        capacity.clusters = std::min(capacity.clusters, clusters);
      });
    }
  }
  capacities.emplace(device, capacity);
  return capacity;
}

/// Launch the first of ``shapes`` that takes a run of ``panels`` panels: of
/// the shapes whose dynamic shared memory the device lets one CTA opt in to,
/// a cluster shape when the device keeps a cluster per panel resident at once,
/// a single-CTA shape unconditionally. Returns whether a shape launched.
template <typename Fn, class... Shapes>
bool launch_first_fit(long panels, Fn&& launch, Shapes... shapes) {
  const auto take = [&](auto shape) {
    using S = decltype(shape);
    if (S::bytes >
        at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin) {
      return false;
    }
    if constexpr (S::cluster > 1) {
      if (fused_capacity<S>().clusters < panels) {
        return false;
      }
    }
    launch(shape);
    return true;
  };
  return (take(shapes) || ...);
}

/// A list of fused shapes.
template <class... Shapes>
struct ShapeList {};

/// launch_first_fit over the shapes of a list.
template <typename Fn, class... Shapes>
bool launch_first_listed(long panels, Fn&& launch, ShapeList<Shapes...>) {
  return launch_first_fit(panels, launch, Shapes{}...);
}

/// The fused shapes compiled for a hidden width, in the order of the dispatch:
/// the latency shapes, the largest cluster first, then the throughput shapes.
template <int W>
struct FusedShapes;
template <>
struct FusedShapes<64> {
  using Latency = ShapeList<FusedShape<64, 2, 2, 4, 4, 1, 1, 4, 32>,
                            FusedShape<64, 1, 4, 4, 2, 2, 1, 8>>;
  using Throughput = ShapeList<FusedShape<64, 1, 4, 4, 2, 2, 4, 3>,
                               FusedShape<64, 1, 4, 4, 4, 2, 2, 2, 16>>;
};
template <>
struct FusedShapes<96> {
  using Latency = ShapeList<FusedShape<96, 3, 2, 4, 4, 1, 1, 4, 32>,
                            FusedShape<96, 1, 4, 4, 2, 3, 1, 8>>;
  using WidePanels = FusedShape<96, 1, 4, 4, 4, 3, 2, 2, 16>;
  using NarrowPanels = FusedShape<96, 1, 4, 4, 2, 3, 3, 3>;
  using Throughput = ShapeList<WidePanels, NarrowPanels>;
};
template <>
struct FusedShapes<192> {
  using Latency = ShapeList<FusedShape<192, 6, 2, 4, 4, 1, 1, 4, 32>,
                            FusedShape<192, 3, 4, 4, 2, 2, 1, 4, 32>,
                            FusedShape<192, 2, 4, 4, 2, 3, 1, 4, 16>,
                            FusedShape<192, 1, 4, 4, 2, 6, 1, 8>>;
  using Throughput = ShapeList<FusedShape<192, 1, 4, 8, 2, 3, 2, 3>>;
};
template <>
struct FusedShapes<256> {
  using Latency = ShapeList<FusedShape<256, 8, 2, 4, 4, 1, 1, 4, 32>,
                            FusedShape<256, 4, 4, 4, 2, 2, 1, 4, 32>,
                            FusedShape<256, 2, 4, 4, 2, 4, 1, 4, 16>,
                            FusedShape<256, 1, 4, 4, 2, 8, 1, 7, 8, 3>>;
  using Throughput = ShapeList<FusedShape<256, 1, 8, 8, 2, 4, 1, 3>>;
};
template <>
struct FusedShapes<384> {
  using Latency = ShapeList<FusedShape<384, 12, 2, 4, 4, 1, 1, 4, 32>,
                            FusedShape<384, 6, 4, 4, 2, 2, 1, 4, 32>,
                            FusedShape<384, 4, 4, 4, 2, 3, 1, 4, 16>,
                            FusedShape<384, 3, 4, 4, 2, 4, 1, 4, 16>,
                            FusedShape<384, 2, 4, 4, 2, 6, 1, 3, 16>,
                            FusedShape<384, 1, 8, 4, 1, 12, 1, 3, 8>>;
  using Throughput = ShapeList<>;
};

/// Call ``fn`` with the FusedShapes of ``width``; false for a width that no
/// fused shape is compiled for.
template <typename Fn>
bool with_fused_shapes(long width, Fn&& fn) {
  switch (width) {
    case 64:
      return fn(FusedShapes<64>{});
    case 96:
      return fn(FusedShapes<96>{});
    case 192:
      return fn(FusedShapes<192>{});
    case 256:
      return fn(FusedShapes<256>{});
    case 384:
      return fn(FusedShapes<384>{});
    default:
      return false;
  }
}

/// Call ``visit`` with the shapes of a list in order until it returns true;
/// returns whether it did.
template <typename Fn, class... Shapes>
bool visit_shapes(ShapeList<Shapes...>, [[maybe_unused]] Fn&& visit) {
  return (visit(Shapes{}) || ...);
}

/// Call ``visit`` with every fused shape compiled for ``width`` until it
/// returns true; returns whether it did.
template <typename Fn>
bool visit_fused_shapes(long width, Fn&& visit) {
  return with_fused_shapes(width, [&](auto shapes) {
    using Shapes = decltype(shapes);
    return visit_shapes(typename Shapes::Latency{}, visit) ||
           visit_shapes(typename Shapes::Throughput{}, visit);
  });
}

/// Call ``visit`` with the fused shapes of ``width`` that a measurement of
/// the fitting routes compares on runs of ``latency`` or of more panels than
/// multiprocessors: the latency shapes on the former, and on the latter the
/// throughput shapes, or the single-CTA latency shapes of a width that has
/// none; until it returns true. Returns whether it did.
template <typename Fn>
bool visit_compared_shapes(long width, bool latency, Fn&& visit) {
  return with_fused_shapes(width, [&](auto shapes) {
    using Shapes = decltype(shapes);
    if (latency) {
      return visit_shapes(typename Shapes::Latency{}, visit);
    }
    if constexpr (std::is_same_v<typename Shapes::Throughput, ShapeList<>>) {
      return visit_shapes(typename Shapes::Latency{}, [&](auto shape) {
        return decltype(shape)::cluster == 1 && visit(shape);
      });
    } else {
      return visit_shapes(typename Shapes::Throughput{}, visit);
    }
  });
}

/// Dispatch the fused network over the hidden widths it is compiled for.
/// Returns whether it passed a shape to ``launch``.
///
/// A run of more panels than multiprocessors takes the throughput shape of its
/// width, the fastest on sm_120 of those that fit the register and
/// shared-memory budgets: the narrow widths take 32-node panels, four CTAs
/// per SM at a width of 64, three at 96 and two at 192, whose small lane
/// tiles keep the most panels in flight, and a width of 256 takes 64-node
/// panels on one CTA per SM, which stream each weight slice to half as many
/// CTAs as 32-node panels and so keep the L2 traffic within the FMA rate. A
/// width of 96 moves on to 64-node panels of twelve warps, two CTAs per SM in
/// place of three, once a run fills two waves of them, or once they pass it in
/// fewer waves than the 32-node panels, each shape at the CTAs per SM that the
/// device keeps resident. Such a panel streams each weight slice once for twice
/// the nodes of a 32-node panel, and on sm_120 the larger panels take 6% less
/// time at 100,800 nodes and 11% less at 131,072, the default node tile; from
/// two waves on, this gain outweighs the cost of their last, partly filled
/// wave, and below two waves a wave fewer outweighs it: on the 92
/// multiprocessors of an L20, one wave of 64-node panels takes 0.064 ms at
/// 10,296 nodes against 0.078 ms for two waves of 32-node panels.
///
/// A width that the 128 columns of the per-layer large tile divide is left to
/// the per-layer kernels at runs of more panels than multiprocessors on a
/// device whose fp32 ridge lies below width / 4, the FLOP per byte of a hidden
/// layer of those kernels: bound there by the arithmetic rather than by the
/// activation traffic that the fusion saves, they use more of the fp32 peak
/// than the fused panels, and at a width of 256 on an H20, whose ridge is 10,
/// they shorten the LAMMPS step by 1% to 6%. At a width of 384 the per-layer
/// kernels, whose 64 x 128 tiles reuse each weight slice across twice the
/// nodes, serve such a run on every device: a 32-node panel of that width
/// takes 94,848 bytes of shared memory, a multiprocessor of sm_120 to itself,
/// so the run would pass in waves of one panel per multiprocessor.
///
/// A smaller run splits its 32-node panels over the largest cluster, of the
/// divisors of width / 32 above one, of which the device keeps one per panel
/// resident at once; the cluster divides the FMAs of each panel between its
/// CTAs. A cluster above the portable size of eight takes the non-portable
/// cluster attribute and runs where the device launches clusters of its size.
/// A run too large for any cluster in one wave, or on a device without
/// clusters, takes the latency shape: 32-node panels on the most lanes their
/// tile allows, each alone on its multiprocessor, with a ring deep enough to
/// keep the weight stream ahead of the FMAs without other panels to hide its
/// latency. The ring runs ahead through asynchronous copies, so
/// fitting_fused_serves leaves such a run to the per-layer kernels on a kernel
/// image of compute capability 7.x, whose copies are plain loads and stores.
/// The tail of a tiled call that graph_fitting_energy_gradient serves with the
/// fused network takes such a shape nevertheless, since that operator decides
/// once for the runs of the call; the fused energy-force operator of DPA4C
/// decides the tail by its own size (fitting_route).
template <typename Fn>
bool dispatch_fused_shape(long width, long run_nodes, Fn&& launch) {
  const long sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long panels = ceil_div(run_nodes, 32);
  const bool latency = panels <= sms;
  if (!latency && width % LargeTile::cols == 0 && fp32_ridge() < width / 4.0) {
    return false;
  }
  return with_fused_shapes(width, [&](auto shapes) {
    using Shapes = decltype(shapes);
    if (latency) {
      return launch_first_listed(panels, launch, typename Shapes::Latency{});
    }
    if constexpr (std::is_same_v<Shapes, FusedShapes<96>>) {
      using WidePanels = typename Shapes::WidePanels;
      using NarrowPanels = typename Shapes::NarrowPanels;
      // Panels of each shape that one wave of the device holds.
      const long wide_wave = fused_capacity<WidePanels>().resident * sms;
      const long narrow_wave = fused_capacity<NarrowPanels>().resident * sms;
      if (wide_wave > 0 &&
          (run_nodes >= 2 * wide_wave * WidePanels::rows ||
           (narrow_wave > 0 &&
            ceil_div(ceil_div(run_nodes, WidePanels::rows), wide_wave) <
                ceil_div(ceil_div(run_nodes, NarrowPanels::rows),
                         narrow_wave)))) {
        return launch_first_fit(panels, launch, WidePanels{});
      }
      return launch_first_fit(panels, launch, NarrowPanels{});
    } else {
      return launch_first_listed(panels, launch, typename Shapes::Throughput{});
    }
  });
}

/// Whether the device runs shape S on a run of ``run_nodes`` nodes: an image
/// of compute_70 or later whose shared memory the device admits and, for a
/// cluster shape, a cluster per panel resident at once. A single-CTA shape
/// runs a run of any size, in as many waves as it takes.
template <class S>
bool fused_shape_runs(long run_nodes) {
  const FusedCapacity capacity = fused_capacity<S>();
  if (capacity.version < 70) {
    return false;
  }
  if constexpr (S::cluster > 1) {
    return capacity.clusters >= ceil_div(run_nodes, S::rows);
  } else {
    return capacity.resident > 0;
  }
}

/// CTAs of the per-layer kernels of ``Tile`` that a multiprocessor of the
/// current device keeps resident, the fewest over the forward and the backward
/// kernels and both activations. Each device answers once.
template <class Tile>
int tile_resident() {
  static std::mutex mutex;
  static std::map<c10::DeviceIndex, int> residents;
  const c10::DeviceIndex device = c10::cuda::current_device();
  const std::lock_guard<std::mutex> lock(mutex);
  const auto known = residents.find(device);
  if (known != residents.end()) {
    return known->second;
  }
  int resident = std::numeric_limits<int>::max();
  for (const long act : {0L, 1L}) {
    dispatch_activation(act, [&](auto tag) {
      constexpr int kAct = decltype(tag)::value;
      int forward = 0;
      int backward = 0;
      if (!accepted(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &forward, layer_forward_kernel<Tile, kAct, true>, Tile::threads,
              0))) {
        forward = 0;
      }
      if (!accepted(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &backward, layer_backward_kernel<Tile, kAct, Cotangent::kChain>,
              Tile::threads, 0))) {
        backward = 0;
      }
      resident = std::min({resident, forward, backward});
    });
  }
  residents.emplace(device, resident);
  return resident;
}

// A fitting-route table records which route serves each run size of one
// network on one device, as measured at freeze time; the fused energy-force
// operator of DPA4C receives it as a literal argument of the frozen program.
// It holds kRouteTableMagic, the fingerprint of the device it was measured on
// (device_fingerprint), the number n of its intervals of run sizes, the
// inclusive upper bounds of the first n - 1 intervals in increasing order, and
// one choice per interval: kFittingPerLayer or the key of a fused shape
// (FusedShape::key) where the measurement departs from the rules, and
// kFittingRules, which leaves the run to them, everywhere else.

/// First word of a fitting-route table, which also names its layout.
constexpr int64_t kRouteTableMagic = 0x52545431;
/// Words of the device fingerprint of a fitting-route table.
constexpr size_t kFingerprintWords = 11;

/// What sets the waves and the speed of the fitting kernels on the current
/// device: its compute capability, multiprocessors, L2 size, registers and
/// shared memory per multiprocessor, shared memory a block may opt in to,
/// thread and block limits per multiprocessor, memory bus width, and memory and
/// SM clocks. Each device answers once.
std::vector<int64_t> device_fingerprint() {
  static std::mutex mutex;
  static std::map<int, std::vector<int64_t>> fingerprints;
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const std::lock_guard<std::mutex> lock(mutex);
  const auto known = fingerprints.find(device);
  if (known != fingerprints.end()) {
    return known->second;
  }
  const cudaDeviceAttr attributes[] = {
      cudaDevAttrComputeCapabilityMajor,
      cudaDevAttrComputeCapabilityMinor,
      cudaDevAttrMultiProcessorCount,
      cudaDevAttrL2CacheSize,
      cudaDevAttrMaxRegistersPerMultiprocessor,
      cudaDevAttrMaxSharedMemoryPerMultiprocessor,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      cudaDevAttrMaxThreadsPerMultiProcessor,
      cudaDevAttrMaxBlocksPerMultiprocessor,
      cudaDevAttrGlobalMemoryBusWidth,
      cudaDevAttrMemoryClockRate,
      cudaDevAttrClockRate};
  std::vector<int64_t> values;
  for (const cudaDeviceAttr attribute : attributes) {
    int value = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&value, attribute, device));
    values.push_back(value);
  }
  std::vector<int64_t> fingerprint{10 * values[0] + values[1]};
  fingerprint.insert(fingerprint.end(), values.begin() + 2, values.end());
  TORCH_INTERNAL_ASSERT(fingerprint.size() == kFingerprintWords);
  fingerprints.emplace(device, fingerprint);
  return fingerprint;
}

/// The choice that ``table`` makes for a run of ``run_nodes`` nodes:
/// kFittingPerLayer or the key of a fused shape where the measurement departed
/// from the rules, and kFittingRules where it agreed with them or the table is
/// empty, malformed or measured on a device of another fingerprint.
int64_t table_choice(const std::vector<int64_t>& table, long run_nodes) {
  constexpr size_t kHead = 2 + kFingerprintWords;
  if (table.size() <= kHead || table[0] != kRouteTableMagic) {
    return kFittingRules;
  }
  const std::vector<int64_t> fingerprint = device_fingerprint();
  if (!std::equal(fingerprint.begin(), fingerprint.end(), table.begin() + 1)) {
    return kFittingRules;
  }
  const int64_t intervals = table[kHead - 1];
  if (intervals < 1 ||
      table.size() != kHead + 2 * static_cast<size_t>(intervals) - 1) {
    return kFittingRules;
  }
  const int64_t* bounds = table.data() + kHead;
  const int64_t* choices = bounds + intervals - 1;
  for (int64_t i = 1; i < intervals - 1; ++i) {
    if (bounds[i - 1] >= bounds[i]) {
      return kFittingRules;
    }
  }
  int64_t interval = 0;
  while (interval < intervals - 1 && run_nodes > bounds[interval]) {
    ++interval;
  }
  return choices[interval];
}

/// Row pitch of the transposed weight ``w``: its input width rounded up to
/// whole quads.
long transposed_pitch(const torch::Tensor& w) {
  return (w.size(0) + 3) / 4 * 4;
}

/// Point the transposed weights of ``p`` into ``wt``, one (width, pitch)
/// sheet per layer in order.
void place_transposed_weights(FusedNetwork& p,
                              const std::vector<torch::Tensor>& ws,
                              const float* wt) {
  long offset = 0;
  for (size_t l = 0; l < ws.size(); ++l) {
    p.pitch[l] = static_cast<int>(transposed_pitch(ws[l]));
    p.wt[l] = wt + offset;
    offset += ws[l].size(1) * p.pitch[l];
  }
}

/// The plan of a fused network holds one layer per weight, at most
/// ``kFusedLayers`` of them, which take at least one input feature and share
/// one hidden width, the width that selects the kernel.
void check_fused_plan(const FittingLayerPlan& plan,
                      const std::vector<torch::Tensor>& ws) {
  TORCH_CHECK(plan.n_layer == static_cast<int>(ws.size()) &&
                  plan.n_layer <= kFusedLayers,
              "graph_fitting: the fused network holds one plan layer per "
              "weight and at most ",
              kFusedLayers, " layers, got ", plan.n_layer, " and ", ws.size());
  TORCH_CHECK(ws[0].size(0) > 0,
              "graph_fitting: the fused network takes at least one input "
              "feature");
  for (int l = 1; l < plan.n_layer; ++l) {
    TORCH_CHECK(ws[l].size(1) == ws[0].size(1),
                "graph_fitting: the fused network takes one hidden width, got ",
                ws[0].size(1), " and ", ws[l].size(1));
  }
}

/// Whether the fused kernel computes the network: one to eight hidden layers
/// of one width, every layer after the first square, and an input of at least
/// one feature.
bool fused_network_eligible(const FittingLayerPlan& plan,
                            const std::vector<torch::Tensor>& ws) {
  if (plan.n_layer < 1 || plan.n_layer > kFusedLayers || ws[0].size(0) < 1) {
    return false;
  }
  const long width = ws[0].size(1);
  for (int l = 0; l < plan.n_layer; ++l) {
    if (ws[l].size(1) != width || (l > 0 && ws[l].size(0) != width)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool fitting_fused_serves(const FittingLayerPlan& plan,
                          const std::vector<torch::Tensor>& ws,
                          long run_nodes) {
  if (!fused_network_eligible(plan, ws)) {
    return false;
  }
  const long width = ws[0].size(1);
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  bool serves = false;
  dispatch_fused_shape(width, run_nodes, [&](auto shape) {
    using S = decltype(shape);
    // The kernel streams its weights by asynchronous copies, which an image
    // compiled for compute capability 7.x performs as plain loads and stores.
    // The panels of a throughput shape share a multiprocessor and hide the
    // weight stream behind each other, so they serve images of compute
    // capability 7.0 and later; a latency shape runs each panel alone on its
    // multiprocessor and keeps the stream ahead of its FMAs through the
    // asynchronous copies of its ring alone, so it serves images of 8.0 and
    // later. An image of an earlier architecture leaves the run to the
    // per-layer kernels at every width.
    //
    // A run of few panels leaves each panel's pass through the whole network
    // on one multiprocessor, while the per-layer kernels spread every layer
    // over the device; a cluster splits each panel between several CTAs
    // instead. A single CTA per panel serves once the run's 32-node panels
    // cover the share of the multiprocessors its shape names -- half of them,
    // and three quarters for a width of 256 -- from which the latency shapes
    // outrun the per-layer kernels on sm_120.
    const long panels = ceil_div(run_nodes, 32);
    const long sms = properties->multiProcessorCount;
    serves = fused_capacity<S>().version >= (panels <= sms ? 80 : 70) &&
             (S::cluster > 1 || 4 * panels >= S::serve_quarters * sms);
  });
  // graph_fitting_energy_gradient decides once for the runs of a tiled call
  // and launches its tail with a shape of the tail's own size, so a run is
  // served only where every shorter run finds a shape as well. The fused
  // energy-force operator of DPA4C decides its tail by its own size but takes
  // the same rule, so that the two operators route a run of one size alike. A
  // tail of more panels than multiprocessors takes the throughput list of its
  // width, as the served run does: one shape, or at a width of 96 the 64-node
  // panels only where the device keeps them resident and otherwise the
  // 32-node panels, which take less shared memory. A shorter tail takes the
  // latency list, which ends with a single-CTA shape that launch_first_fit
  // takes at any panel count once the device admits its shared memory; a
  // device that admits less than every shape, below compute capability 8.0,
  // launches no clusters, so its latency list serves every run size or none,
  // and a run of one panel stands for every tail.
  return serves && dispatch_fused_shape(width, 1, [](auto) {});
}

int64_t fitting_route(const FittingLayerPlan& plan,
                      const std::vector<torch::Tensor>& ws,
                      long run_nodes,
                      const std::vector<int64_t>& table) {
  const int64_t choice = table_choice(table, run_nodes);
  if (choice == kFittingPerLayer) {
    return kFittingPerLayer;
  }
  if (choice > 0 && fused_network_eligible(plan, ws) &&
      visit_fused_shapes(ws[0].size(1), [&](auto shape) {
        using S = decltype(shape);
        return S::key == choice && fused_shape_runs<S>(run_nodes);
      })) {
    return choice;
  }
  return fitting_fused_serves(plan, ws, run_nodes) ? kFittingRules
                                                   : kFittingPerLayer;
}

long fitting_fused_transposed_floats(const std::vector<torch::Tensor>& ws) {
  long total = 0;
  for (const auto& w : ws) {
    total += w.size(1) * transposed_pitch(w);
  }
  return total;
}

void fitting_fused_transpose(cudaStream_t stream,
                             const FittingLayerPlan& plan,
                             const std::vector<torch::Tensor>& ws,
                             float* wt) {
  check_weights(ws);
  check_fused_plan(plan, ws);
  FusedNetwork p{};
  TransposedSheets sheets{};
  for (int l = 0; l < plan.n_layer; ++l) {
    p.w[l] = ws[l].data_ptr<float>();
  }
  p.din = (int)ws[0].size(0);
  place_transposed_weights(p, ws, wt);
  // The sheets lie in the caller's writable ``wt``.
  int pitch = 0;
  for (int l = 0; l < plan.n_layer; ++l) {
    sheets.wt[l] = const_cast<float*>(p.wt[l]);
    pitch = std::max(pitch, p.pitch[l]);
  }
  const int width = (int)ws[0].size(1);
  const dim3 grid((unsigned)ceil_div(pitch, kTransposeTile),
                  (unsigned)ceil_div(width, kTransposeTile),
                  (unsigned)plan.n_layer);
  transpose_weights_kernel<<<grid, dim3(kTransposeTile, 8), 0, stream>>>(
      p, width, sheets);
  FITTING_CHECK_LAUNCH("graph_fitting weight transpose");
}

void fitting_energy_gradient_range(cudaStream_t stream,
                                   const FittingLayerPlan& plan,
                                   const float* x,
                                   long input_width,
                                   const long* atype,
                                   const std::vector<torch::Tensor>& ws,
                                   const std::vector<torch::Tensor>& bs,
                                   const std::vector<int64_t>& resnets,
                                   const torch::Tensor& w_head,
                                   const torch::Tensor& b_head,
                                   const torch::Tensor& bias_atom_e,
                                   int64_t act,
                                   const float* wt,
                                   const double* d_e,
                                   long run_nodes,
                                   float* saved,
                                   double* e,
                                   float* d_x,
                                   int64_t route) {
  check_energy_parameters(ws, bs, resnets, w_head, b_head, bias_atom_e);
  check_fused_plan(plan, ws);
  FusedNetwork p{};
  p.x = x;
  p.atype = atype;
  for (int l = 0; l < plan.n_layer; ++l) {
    p.w[l] = ws[l].data_ptr<float>();
    p.b[l] = bs[l].numel() ? bs[l].data_ptr<float>() : nullptr;
    p.residual[l] = resnets[l] && ws[l].size(0) == ws[l].size(1) ? 1 : 0;
  }
  p.w_head = w_head.data_ptr<float>();
  p.b_head = b_head.numel() ? b_head.data_ptr<float>() : nullptr;
  p.bias_atom_e = bias_atom_e.data_ptr<double>();
  p.d_e = d_e;
  p.saved = saved;
  p.e = e;
  p.d_x = d_x;
  p.nodes = run_nodes;
  p.din = (int)input_width;
  p.n_layer = plan.n_layer;
  place_transposed_weights(p, ws, wt);
  const auto launch = [&](auto shape) {
    using S = decltype(shape);
    dispatch_activation(act, [&](auto tag) {
      constexpr int kAct = decltype(tag)::value;
      configure_fused_kernel<S, kAct>();
      cudaLaunchAttribute cluster;
      const cudaLaunchConfig_t config = fused_launch_config<S>(
          ceil_div(run_nodes, S::rows), stream, &cluster);
      C10_CUDA_CHECK(
          cudaLaunchKernelEx(&config, fused_network_kernel<S, kAct>, p));
    });
  };
  const long width = ws[0].size(1);
  const bool launched = route == kFittingRules
                            ? dispatch_fused_shape(width, run_nodes, launch)
                            : visit_fused_shapes(width, [&](auto shape) {
                                if (decltype(shape)::key != route) {
                                  return false;
                                }
                                launch(shape);
                                return true;
                              });
  TORCH_CHECK(launched, "graph_fitting fused network: no fused shape serves ",
              run_nodes, " nodes of hidden width ", width, " on route ", route,
              "; fitting_fused_serves and fitting_route select the routes");
}

// Forward: per-atom energy (fp64 (N, 1)) plus the flat saved buffer of the
// layer pre-activations -- chunk l a contiguous (N, width_l) sheet. The
// activations themselves stay in a forward-only ping-pong.
std::tuple<torch::Tensor, torch::Tensor> graph_fitting(
    torch::Tensor x,
    torch::Tensor atype,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> bs,
    std::vector<int64_t> resnets,
    torch::Tensor w_head,
    torch::Tensor b_head,
    torch::Tensor bias_atom_e,
    int64_t act) {
  const FittingLayerPlan plan = validate_fitting_forward_inputs(
      "graph_fitting", x, atype, ws, bias_atom_e);
  const c10::cuda::CUDAGuard device_guard(x.device());
  const long n_node = x.size(0);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto saved = torch::empty({n_node * plan.saved_width()}, f32);
  auto e = torch::empty({n_node, 1}, x.options().dtype(torch::kFloat64));
  if (n_node == 0) {
    return {e, saved};
  }
  // Two-slot ping-pong for the activations: layer l writes slot ``l & 1``
  // while reading the previous layer's slot, so an activation is overwritten
  // only after the next GEMM has consumed it (kernels run in stream order).
  // The last layer stores no activation; its slot holds the head partial dots,
  // so a single-layer network requires only that slot.
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto act_buf = torch::empty({slots, n_node, plan.width_max}, f32);
  float* activation[2] = {
      act_buf[0].data_ptr<float>(),
      slots > 1 ? act_buf[1].data_ptr<float>() : act_buf[0].data_ptr<float>()};
  fitting_forward_range(
      at::cuda::getCurrentCUDAStream(), plan, x.data_ptr<float>(), x.size(1),
      atype.data_ptr<long>(), ws, bs, resnets, w_head, b_head, bias_atom_e, act,
      n_node, saved.data_ptr<float>(), activation, e.data_ptr<double>());
  return {e, saved};
}

// Backward: d_x from the upstream d_e (fp64 (N, 1)). The saved pre-activation
// extent and fitting widths determine the output shape, so the descriptor is
// not retained solely for shape metadata.
void graph_fitting_backward_core(torch::Tensor d_e,
                                 torch::Tensor saved,
                                 std::vector<torch::Tensor> ws,
                                 std::vector<torch::Tensor> bs,
                                 std::vector<int64_t> resnets,
                                 torch::Tensor w_head,
                                 int64_t act,
                                 torch::Tensor d_x) {
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.saved_width() > 0 && saved.numel() % plan.saved_width() == 0,
              "graph_fitting_backward: saved buffer does not match the "
              "fitting widths");
  const long n_node = saved.numel() / plan.saved_width();
  const long input_width = ws[0].size(0);
  TORCH_CHECK(d_x.dim() == 2 && d_x.size(0) == n_node &&
                  d_x.size(1) == input_width &&
                  d_x.scalar_type() == torch::kFloat32 && d_x.is_cuda() &&
                  d_x.is_contiguous(),
              "graph_fitting_backward: output must be contiguous CUDA "
              "fp32 with shape (N, input_width)");
  // Guard the empty system before the division by ``n_node`` below.
  if (n_node == 0) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(saved.device());
  auto f32 = saved.options().dtype(torch::kFloat32);
  auto d_e_c = d_e.contiguous();
  const auto scratch = [&](bool needed) {
    return needed ? torch::empty({n_node, plan.width_max}, f32)
                  : torch::empty({0}, f32);
  };
  auto dh = scratch(plan.n_layer > 1);
  auto dh_next = scratch(plan.n_layer > 2);
  fitting_backward_range(at::cuda::getCurrentCUDAStream(), plan,
                         d_e_c.data_ptr<double>(), saved.data_ptr<float>(), ws,
                         bs, resnets, w_head, act, n_node,
                         plan.n_layer > 1 ? dh.data_ptr<float>() : nullptr,
                         plan.n_layer > 2 ? dh_next.data_ptr<float>() : nullptr,
                         d_x.data_ptr<float>());
}

torch::Tensor graph_fitting_backward(torch::Tensor d_e,
                                     torch::Tensor saved,
                                     std::vector<torch::Tensor> ws,
                                     std::vector<torch::Tensor> bs,
                                     std::vector<int64_t> resnets,
                                     torch::Tensor w_head,
                                     int64_t act) {
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.saved_width() > 0 && saved.numel() % plan.saved_width() == 0,
              "graph_fitting_backward: saved buffer does not match the "
              "fitting widths");
  const long n_node = saved.numel() / plan.saved_width();
  auto d_x = torch::empty({n_node, ws[0].size(0)}, saved.options());
  graph_fitting_backward_core(d_e, saved, std::move(ws), std::move(bs),
                              std::move(resnets), w_head, act, d_x);
  return d_x;
}

// Energy and input gradient in one pass over node tiles.
//
// Inference knows the head cotangent before the forward runs -- it is the
// ownership mask -- so the two directions need not be separated by the whole
// node axis. Walking tiles instead retires each tile's pre-activations as soon
// as its backward consumes them, which replaces the largest node-scale
// allocation of a step with one of tile size. The network is evaluated per
// node, so no tile needs anything from another and nothing is recomputed.
//
// ``tile`` of zero, or any value covering the system, degenerates to a single
// run and reproduces the untiled sequence exactly. A network and run that the
// fused network serves take it instead of the per-layer sequence, with the
// same results bit for bit.
//
// The cotangent replaces the descriptor in place. A tile reads its descriptor
// rows in the first layer and writes their cotangent in the last, and no later
// tile revisits them, so the two never need separate node-scale allocations;
// inference has no further use for the descriptor once the forward has read
// it. Only the energy is returned.
torch::Tensor graph_fitting_energy_gradient(torch::Tensor x,
                                            torch::Tensor atype,
                                            std::vector<torch::Tensor> ws,
                                            std::vector<torch::Tensor> bs,
                                            std::vector<int64_t> resnets,
                                            torch::Tensor w_head,
                                            torch::Tensor b_head,
                                            torch::Tensor bias_atom_e,
                                            int64_t act,
                                            torch::Tensor seed,
                                            int64_t tile) {
  const FittingLayerPlan plan = validate_fitting_forward_inputs(
      "graph_fitting_energy_gradient", x, atype, ws, bias_atom_e);
  const c10::cuda::CUDAGuard device_guard(x.device());
  const long n_node = x.size(0);
  const long input_width = x.size(1);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto e = torch::empty({n_node, 1}, x.options().dtype(torch::kFloat64));
  if (n_node == 0) {
    return e;
  }
  auto seed_c = seed.contiguous();
  TORCH_CHECK(
      seed_c.numel() == n_node && seed_c.scalar_type() == torch::kFloat64 &&
          seed_c.is_cuda() && seed_c.device() == x.device(),
      "graph_fitting_energy_gradient: seed must be CUDA fp64 with one entry "
      "per node on the device of x");

  const long run = tile > 0 ? std::min<long>(tile, n_node) : n_node;
  auto saved = torch::empty({run * plan.saved_width()}, f32);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (fitting_fused_serves(plan, ws, run)) {
    auto transposed = torch::empty({fitting_fused_transposed_floats(ws)}, f32);
    fitting_fused_transpose(stream, plan, ws, transposed.data_ptr<float>());
    for (long begin = 0; begin < n_node; begin += run) {
      const long count = std::min(run, n_node - begin);
      fitting_energy_gradient_range(
          stream, plan, x.data_ptr<float>() + begin * input_width, input_width,
          atype.data_ptr<long>() + begin, ws, bs, resnets, w_head, b_head,
          bias_atom_e, act, transposed.data_ptr<float>(),
          seed_c.data_ptr<double>() + begin, count, saved.data_ptr<float>(),
          e.data_ptr<double>() + begin,
          x.data_ptr<float>() + begin * input_width, kFittingRules);
    }
    return e;
  }
  // The backward reads only the saved pre-activations, never the activations,
  // so the forward ping-pong and the two cotangent buffers never hold live
  // data at the same time and share one allocation.
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto scratch = torch::empty({slots, run, plan.width_max}, f32);
  float* slot[2] = {
      scratch[0].data_ptr<float>(),
      slots > 1 ? scratch[1].data_ptr<float>() : scratch[0].data_ptr<float>()};
  for (long begin = 0; begin < n_node; begin += run) {
    const long count = std::min(run, n_node - begin);
    fitting_forward_range(
        stream, plan, x.data_ptr<float>() + begin * input_width, input_width,
        atype.data_ptr<long>() + begin, ws, bs, resnets, w_head, b_head,
        bias_atom_e, act, count, saved.data_ptr<float>(), slot,
        e.data_ptr<double>() + begin);
    fitting_backward_range(stream, plan, seed_c.data_ptr<double>() + begin,
                           saved.data_ptr<float>(), ws, bs, resnets, w_head,
                           act, count, slot[0],
                           plan.n_layer > 1 ? slot[1] : nullptr,
                           x.data_ptr<float>() + begin * input_width);
  }
  return e;
}

namespace {

// Queries of the freeze-time measurement of the fitting routes
// (deepmd/pt_expt/kernels/fitting_routes.py) and of its tests. They answer for
// CUDA device ``device``.

/// The fingerprint that a fitting-route table measured on the device carries.
std::vector<int64_t> graph_fitting_route_fingerprint(int64_t device) {
  const c10::cuda::CUDAGuard device_guard(
      static_cast<c10::DeviceIndex>(device));
  return device_fingerprint();
}

/// The keys of the fused shapes of hidden width ``width`` that a measurement
/// compares on a run of ``run_nodes`` nodes (visit_compared_shapes) and that
/// the device runs on it, in the order of the dispatch.
std::vector<int64_t> graph_fitting_route_choices(int64_t device,
                                                 int64_t width,
                                                 int64_t run_nodes) {
  const c10::cuda::CUDAGuard device_guard(
      static_cast<c10::DeviceIndex>(device));
  const long sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  std::vector<int64_t> keys;
  visit_compared_shapes(width, ceil_div(run_nodes, 32) <= sms, [&](auto shape) {
    using S = decltype(shape);
    if (fused_shape_runs<S>(run_nodes)) {
      keys.push_back(S::key);
    }
    return false;
  });
  return keys;
}

/// The route that the rules take for the network ``ws`` on a run of
/// ``run_nodes`` nodes: kFittingPerLayer or the key of the fused shape the
/// dispatch selects.
int64_t graph_fitting_route_rule(std::vector<torch::Tensor> ws,
                                 int64_t run_nodes) {
  TORCH_CHECK(!ws.empty() && ws[0].is_cuda(),
              "graph_fitting_route_rule: the weights must be CUDA tensors");
  const c10::cuda::CUDAGuard device_guard(ws[0].device());
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  if (!fitting_fused_serves(plan, ws, run_nodes)) {
    return kFittingPerLayer;
  }
  int64_t key = kFittingPerLayer;
  dispatch_fused_shape(ws[0].size(1), run_nodes,
                       [&](auto shape) { key = decltype(shape)::key; });
  return key;
}

/// The route that the fused energy-force operator of DPA4C takes for the
/// network ``ws`` on a run of ``run_nodes`` nodes under the fitting-route
/// table ``table`` (fitting_route): kFittingPerLayer, kFittingRules or the key
/// of a fused shape.
int64_t graph_fitting_route_resolve(std::vector<torch::Tensor> ws,
                                    int64_t run_nodes,
                                    std::vector<int64_t> table) {
  TORCH_CHECK(!ws.empty() && ws[0].is_cuda(),
              "graph_fitting_route_resolve: the weights must be CUDA tensors");
  const c10::cuda::CUDAGuard device_guard(ws[0].device());
  return fitting_route(fitting_layer_plan(ws), ws, run_nodes, table);
}

/// Inclusive upper bounds of the run sizes, up to ``max_nodes`` and ending
/// with it, between which none of the routes that a measurement compares
/// (visit_compared_shapes and the per-layer kernels) changes the number of
/// waves of its grids on the device, and no rule changes the route of the
/// network of hidden width ``width`` and input width ``din``: the cluster
/// capacities and the serve gates of the latency shapes, the waves of the
/// single-CTA shapes compared above one panel per multiprocessor, the waves of
/// the per-layer tiles on every column count of the network and the switch
/// between the two tiles, and the switch between the latency and the
/// throughput shapes. A route keeps its waves inside such a band, but not the
/// fill of its last wave, which the measurement samples at both ends of the
/// band.
std::vector<int64_t> graph_fitting_route_bands(int64_t device,
                                               int64_t width,
                                               int64_t din,
                                               int64_t max_nodes) {
  const c10::cuda::CUDAGuard device_guard(
      static_cast<c10::DeviceIndex>(device));
  const int64_t sms =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t latency_nodes = 32 * sms;
  std::set<int64_t> bounds;
  const auto add = [&](int64_t nodes) {
    if (nodes >= 1 && nodes < max_nodes) {
      bounds.insert(nodes);
    }
  };
  add(latency_nodes);
  visit_compared_shapes(width, true, [&](auto shape) {
    using S = decltype(shape);
    const FusedCapacity capacity = fused_capacity<S>();
    if constexpr (S::cluster > 1) {
      add(std::min<int64_t>(capacity.clusters * S::rows, latency_nodes));
    } else {
      add(32 * (ceil_div(S::serve_quarters * sms, 4) - 1));
    }
    return false;
  });
  visit_compared_shapes(width, false, [&](auto shape) {
    using S = decltype(shape);
    const int64_t wave =
        static_cast<int64_t>(fused_capacity<S>().resident) * sms * S::rows;
    for (int64_t nodes = wave; wave > 0 && nodes < max_nodes; nodes += wave) {
      if (nodes > latency_nodes) {
        add(nodes);
      }
    }
    return false;
  });
  // The per-layer kernels take the small tile up to the run size at which its
  // grid reaches four and a half CTAs per multiprocessor (dispatch_tile), and
  // the large tile beyond it, on each GEMM of the network.
  for (const int64_t columns : {width, din}) {
    const int64_t small_limit =
        SmallTile::rows * (9 * sms / (2 * ceil_div(columns, SmallTile::cols)));
    add(small_limit);
    const auto add_tile_waves = [&](auto tile, int64_t from, int64_t to) {
      using Tile = decltype(tile);
      const int64_t wave = static_cast<int64_t>(tile_resident<Tile>()) * sms;
      const int64_t tiles = ceil_div(columns, Tile::cols);
      for (int64_t waves = 1; wave > 0; ++waves) {
        const int64_t nodes = Tile::rows * (waves * wave / tiles);
        if (nodes >= to) {
          break;
        }
        if (nodes > from) {
          add(nodes);
        }
      }
    };
    add_tile_waves(SmallTile{}, 0, small_limit);
    add_tile_waves(LargeTile{}, small_limit, max_nodes);
  }
  bounds.insert(max_nodes);
  return {bounds.begin(), bounds.end()};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.impl("graph_fitting", torch::kCUDA, &graph_fitting);
  m.impl("graph_fitting_backward", torch::kCUDA, &graph_fitting_backward);
  m.impl("graph_fitting_energy_gradient", torch::kCUDA,
         &graph_fitting_energy_gradient);
  m.def("graph_fitting_route_fingerprint(int device) -> int[]",
        &graph_fitting_route_fingerprint);
  m.def(
      "graph_fitting_route_choices(int device, int width, int run_nodes) "
      "-> int[]",
      &graph_fitting_route_choices);
  m.def(
      "graph_fitting_route_bands(int device, int width, int din, "
      "int max_nodes) -> int[]",
      &graph_fitting_route_bands);
  m.def("graph_fitting_route_rule(Tensor[] ws, int run_nodes) -> int");
  m.impl("graph_fitting_route_rule", torch::kCUDA, &graph_fitting_route_rule);
  m.def(
      "graph_fitting_route_resolve(Tensor[] ws, int run_nodes, int[] table) "
      "-> int");
  m.impl("graph_fitting_route_resolve", torch::kCUDA,
         &graph_fitting_route_resolve);
}
