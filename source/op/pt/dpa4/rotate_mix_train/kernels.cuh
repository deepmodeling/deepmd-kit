// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Kernel bodies of the SeZM rotation / degree-mixing training operators.
// Included by the (degree, rank, dtype) build shards and by the host file; the
// kernels live in a named namespace so explicit instantiations link across
// translation units.

#pragma once

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dpa4_sezm_kernels {

#define DPA4_RM_CHECK_LAUNCH(what)                                        \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

constexpr int kMaxLmax = 6;
constexpr int kMaxRank = 4;
constexpr int kMaxRotateFocus = 4;
constexpr int kMediumChannelLanes = 192;
constexpr int kNarrowChannelLanes = 256;
constexpr int kWideChannelLanes = 384;

// Threads per block: one thread per channel lane, padded to a warp multiple.
__host__ inline int lane_count(int c_wide) { return ((c_wide + 31) / 32) * 32; }

// Block-wide sum over the channel lanes. Every thread contributes one value;
// lane 0 of the block receives the total. ``smem`` holds one float per warp.
__device__ __forceinline__ float block_channel_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, off);
  }
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();
  const int n_warps = (blockDim.x + 31) >> 5;
  float total = 0.0f;
  if (warp == 0) {
    total = (lane < n_warps) ? smem[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      total += __shfl_down_sync(0xffffffff, total, off);
    }
  }
  __syncthreads();
  return total;
}

// Warp-local stage of a batched block reduction: reduces ``v`` within the
// warp and parks the warp's partial in ``part[slot * n_warps + warp]``.
// The caller separates the accumulation phase from the write-out phase with
// one ``__syncthreads`` for the whole batch of slots, instead of paying two
// block-wide barriers per reduced scalar as ``block_channel_sum`` does.
__device__ __forceinline__ void warp_partial_sum(float v,
                                                 int slot,
                                                 int n_warps,
                                                 float* __restrict__ part) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, off);
  }
  if ((threadIdx.x & 31) == 0) {
    part[slot * n_warps + (threadIdx.x >> 5)] = v;
  }
}

// Cross-warp completion of one batched slot.
__device__ __forceinline__ float finish_partial_sum(
    const float* __restrict__ part, int slot, int n_warps) {
  float t = 0.0f;
  for (int w = 0; w < n_warps; ++w) {
    t += part[slot * n_warps + w];
  }
  return t;
}

// ---------------------------------------------------------------------------
// Forward: one block per edge, one thread per channel.
// ---------------------------------------------------------------------------
template <typename scalar_t, int L, int RANK>
__global__ void rotate_mix_fwd_kernel(const scalar_t* __restrict__ x,
                                      const long* __restrict__ src,
                                      const scalar_t* __restrict__ wig,
                                      const scalar_t* __restrict__ kc,
                                      const scalar_t* __restrict__ cb,
                                      scalar_t* __restrict__ u,
                                      long n_edge,
                                      long x_sn,
                                      long x_sd,
                                      int cf,
                                      int c_wide) {
  constexpr int NS0 = L + 1;
  constexpr int RED = 3 * L + 1;
  constexpr int DIM = (L + 1) * (L + 1);
  constexpr int NW = 3 * DIM - 2;
  const long edge = blockIdx.x;
  if (edge >= n_edge) {
    return;
  }
  const int c = threadIdx.x;
  const bool active = c < c_wide;
  const long row_w = (long)RED * cf;

  const long s = src[edge];
  const scalar_t* xb = x + s * x_sn + (active ? c : 0);
  const scalar_t* edge_runs = wig + edge * NW;

  // === Phase 1. Rotate to the local frame (registers) ===
  float xr[DIM];
#pragma unroll
  for (int r = 0; r < DIM; ++r) {
    xr[r] = active ? (float)xb[r * x_sd] : 0.0f;
  }
  float xl[RED];
#pragma unroll
  for (int l = 0; l <= L; ++l) {
    const int base = l * l;
    float a0 = 0.0f, am = 0.0f, ap = 0.0f;
#pragma unroll
    for (int j = 0; j < 2 * l + 1; ++j) {
      const float xv = xr[base + j];
      a0 += (float)edge_runs[base + j] * xv;
      if (l >= 1) {
        am += (float)edge_runs[DIM + base - 1 + j] * xv;
        ap += (float)edge_runs[2 * DIM + base - 2 + j] * xv;
      }
    }
    xl[l] = a0;
    if (l >= 1) {
      xl[NS0 + l - 1] = am;
      xl[NS0 + L + l - 1] = ap;
    }
  }

  if (!active) {
    return;
  }
  // Focus-major store offset for this channel.
  scalar_t* ub = u + (long)(c / cf) * n_edge * row_w + edge * row_w + (c % cf);

  // === Phase 2. Degree mixing, store focus-major ===
  if (RANK == 0) {
    const scalar_t* rad = kc + edge * (long)NS0 * c_wide + c;
#pragma unroll
    for (int o = 0; o < NS0; ++o) {
      ub[o * cf] = (scalar_t)(xl[o] * (float)rad[o * (long)c_wide]);
    }
#pragma unroll
    for (int o = 0; o < L; ++o) {
      const float r = (float)rad[(o + 1) * (long)c_wide];
      ub[(NS0 + o) * cf] = (scalar_t)(xl[NS0 + o] * r);
      ub[(NS0 + L + o) * cf] = (scalar_t)(xl[NS0 + L + o] * r);
    }
    return;
  }
  float cbv[RANK > 0 ? RANK : 1];
#pragma unroll
  for (int t = 0; t < RANK; ++t) {
    cbv[t] = (float)cb[t * (long)c_wide + c];
  }
  const scalar_t* kb = kc + edge * (long)(NS0 * NS0 + L * L) * RANK;
#pragma unroll
  for (int o = 0; o < NS0; ++o) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < NS0; ++i) {
      if (RANK == 1) {
        acc += (float)kb[i * NS0 + o] * xl[i];
      } else {
        float keff = 0.0f;
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          keff += (float)kb[(i * NS0 + o) * RANK + t] * cbv[t];
        }
        acc += keff * xl[i];
      }
    }
    if (RANK == 1) {
      acc *= cbv[0];
    }
    ub[o * cf] = (scalar_t)acc;
  }
#pragma unroll
  for (int o = 0; o < L; ++o) {
    float an = 0.0f, aq = 0.0f;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      if (RANK == 1) {
        const float k = (float)kb[NS0 * NS0 + i * L + o];
        an += k * xl[NS0 + i];
        aq += k * xl[NS0 + L + i];
      } else {
        float keff = 0.0f;
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          keff += (float)kb[(NS0 * NS0 + i * L + o) * RANK + t] * cbv[t];
        }
        an += keff * xl[NS0 + i];
        aq += keff * xl[NS0 + L + i];
      }
    }
    if (RANK == 1) {
      an *= cbv[0];
      aq *= cbv[0];
    }
    ub[(NS0 + o) * cf] = (scalar_t)an;
    ub[(NS0 + L + o) * cf] = (scalar_t)aq;
  }
}

// ---------------------------------------------------------------------------
// Rotation of one channel lane over the packed structural rows:
// xl[m-major reduced rows] from the gathered feature rows xr. The run stores
// every m = 0 row first, followed by the m = -1 and m = +1 rows. Within each
// group the degree-l row starts at l^2, so the three bases are ``l^2``,
// ``DIM + l^2 - 1`` and ``2 * DIM + l^2 - 2``.
// ---------------------------------------------------------------------------
template <typename scalar_t, int L>
__device__ __forceinline__ void rotate_lane(const float* __restrict__ xr,
                                            const scalar_t* __restrict__ runs,
                                            float* __restrict__ xl) {
  constexpr int NS0 = L + 1;
  constexpr int DIM = (L + 1) * (L + 1);
#pragma unroll
  for (int l = 0; l <= L; ++l) {
    const int base = l * l;
    float a0 = 0.0f, am = 0.0f, ap = 0.0f;
#pragma unroll
    for (int j = 0; j < 2 * l + 1; ++j) {
      const float xv = xr[base + j];
      a0 += (float)runs[base + j] * xv;
      if (l >= 1) {
        am += (float)runs[DIM + base - 1 + j] * xv;
        ap += (float)runs[2 * DIM + base - 2 + j] * xv;
      }
    }
    xl[l] = a0;
    if (l >= 1) {
      xl[NS0 + l - 1] = am;
      xl[NS0 + L + l - 1] = ap;
    }
  }
}

// Degree mixing of one lane against a compact rank-RANK kernel (RANK >= 1),
// accumulated onto the output rows. The kernel block is read in place; at
// high degree and rank it exceeds any reasonable register budget.
template <typename scalar_t, int L, int RANK>
__device__ __forceinline__ void degree_mix_acc(const float* __restrict__ xl,
                                               const scalar_t* __restrict__ kb,
                                               const float* __restrict__ cbv,
                                               float* __restrict__ out) {
  constexpr int NS0 = L + 1;
#pragma unroll
  for (int o = 0; o < NS0; ++o) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < NS0; ++i) {
      if (RANK == 1) {
        acc += (float)kb[i * NS0 + o] * xl[i];
      } else {
        float keff = 0.0f;
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          keff += (float)kb[(i * NS0 + o) * RANK + t] * cbv[t];
        }
        acc += keff * xl[i];
      }
    }
    if (RANK == 1) {
      acc *= cbv[0];
    }
    out[o] += acc;
  }
#pragma unroll
  for (int o = 0; o < L; ++o) {
    float an = 0.0f, aq = 0.0f;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      if (RANK == 1) {
        const float k = (float)kb[NS0 * NS0 + i * L + o];
        an += k * xl[NS0 + i];
        aq += k * xl[NS0 + L + i];
      } else {
        float keff = 0.0f;
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          keff += (float)kb[(NS0 * NS0 + i * L + o) * RANK + t] * cbv[t];
        }
        an += keff * xl[NS0 + i];
        aq += keff * xl[NS0 + L + i];
      }
    }
    if (RANK == 1) {
      an *= cbv[0];
      aq *= cbv[0];
    }
    out[NS0 + o] += an;
    out[NS0 + L + o] += aq;
  }
}

// ---------------------------------------------------------------------------
// Paired forward for the second order: one traversal produces both the
// rotated input u0 = M(kc) R(wig) x and the upstream cotangent of the
// rotation backward,
//
//   h_gu0 = M(kc) R(wig) h_e + M(kc) R(h_gwig) x + M(h_gkc) R(wig) x,
//
// with h_e the node cotangent gathered onto edges. The mixer is linear in
// its feature operand, so the two kc-mixed terms share one mixing pass over
// the summed rotated lanes. Loads of the feature rows, the cotangent rows
// and both Wigner blocks happen once; the four separate forward re-entries
// this replaces each re-read their operands from L2.
// ---------------------------------------------------------------------------
template <typename scalar_t, int L, int RANK, int MAX_THREADS, int MIN_BLOCKS>
__global__
__launch_bounds__(MAX_THREADS, MIN_BLOCKS) void rotate_mix_fwd_pair_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ h_gx,
    const long* __restrict__ src,
    const scalar_t* __restrict__ wig,
    const scalar_t* __restrict__ h_gwig,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ h_gkc,
    const scalar_t* __restrict__ cb,
    scalar_t* __restrict__ u0,
    scalar_t* __restrict__ hgu0,
    long n_edge,
    long x_sn,
    long x_sd,
    long h_sn,
    long h_sd,
    int cf,
    int c_wide) {
  constexpr int NS0 = L + 1;
  constexpr int RED = 3 * L + 1;
  constexpr int DIM = (L + 1) * (L + 1);
  constexpr int NW = 3 * DIM - 2;
  const long edge = blockIdx.x;
  if (edge >= n_edge) {
    return;
  }
  const int c = threadIdx.x;
  if (c >= c_wide) {
    return;
  }
  const long row_w = (long)RED * cf;
  const long s = src[edge];

  float cbv[RANK > 0 ? RANK : 1];
#pragma unroll
  for (int t = 0; t < RANK; ++t) {
    cbv[t] = (float)cb[t * (long)c_wide + c];
  }

  // === Rotated lanes: xl_x for u0 and the h_gkc term; xl_s for the summed
  // kc-mixed cotangent terms (linearity of the mixer merges them) ===
  const scalar_t* db = wig + edge * NW;
  float xl_x[RED];
  float xl_s[RED];
  {
    float xr[DIM];
    const scalar_t* xb = x + s * x_sn + c;
#pragma unroll
    for (int r = 0; r < DIM; ++r) {
      xr[r] = (float)xb[r * x_sd];
    }
    rotate_lane<scalar_t, L>(xr, db, xl_x);
    if (h_gwig != nullptr) {
      rotate_lane<scalar_t, L>(xr, h_gwig + edge * NW, xl_s);
    } else {
#pragma unroll
      for (int r = 0; r < RED; ++r) {
        xl_s[r] = 0.0f;
      }
    }
    const scalar_t* hxb = h_gx + s * h_sn + c;
#pragma unroll
    for (int r = 0; r < DIM; ++r) {
      xr[r] = (float)hxb[r * h_sd];
    }
    float xl_h[RED];
    rotate_lane<scalar_t, L>(xr, db, xl_h);
#pragma unroll
    for (int r = 0; r < RED; ++r) {
      xl_s[r] += xl_h[r];
    }
  }

  const int f = c / cf;
  const int cfi = c % cf;
  scalar_t* ub = u0 + (long)f * n_edge * row_w + edge * row_w + cfi;
  scalar_t* hb = hgu0 + (long)f * n_edge * row_w + edge * row_w + cfi;

  if (RANK == 0) {
    const scalar_t* rad = kc + edge * (long)NS0 * c_wide + c;
    const scalar_t* hrad =
        h_gkc != nullptr ? h_gkc + edge * (long)NS0 * c_wide + c : nullptr;
#pragma unroll
    for (int o = 0; o < NS0; ++o) {
      const float r = (float)rad[o * (long)c_wide];
      ub[o * cf] = (scalar_t)(xl_x[o] * r);
      float h = xl_s[o] * r;
      if (hrad != nullptr) {
        h += xl_x[o] * (float)hrad[o * (long)c_wide];
      }
      hb[o * cf] = (scalar_t)h;
    }
#pragma unroll
    for (int o = 0; o < L; ++o) {
      const float r = (float)rad[(o + 1) * (long)c_wide];
      const float hr =
          hrad != nullptr ? (float)hrad[(o + 1) * (long)c_wide] : 0.0f;
      ub[(NS0 + o) * cf] = (scalar_t)(xl_x[NS0 + o] * r);
      ub[(NS0 + L + o) * cf] = (scalar_t)(xl_x[NS0 + L + o] * r);
      hb[(NS0 + o) * cf] = (scalar_t)(xl_s[NS0 + o] * r + xl_x[NS0 + o] * hr);
      hb[(NS0 + L + o) * cf] =
          (scalar_t)(xl_s[NS0 + L + o] * r + xl_x[NS0 + L + o] * hr);
    }
    return;
  }

  const scalar_t* kb = kc + edge * (long)(NS0 * NS0 + L * L) * RANK;
  float out_u[RED];
  float out_h[RED];
#pragma unroll
  for (int r = 0; r < RED; ++r) {
    out_u[r] = 0.0f;
    out_h[r] = 0.0f;
  }
  degree_mix_acc<scalar_t, L, RANK>(xl_x, kb, cbv, out_u);
  degree_mix_acc<scalar_t, L, RANK>(xl_s, kb, cbv, out_h);
  if (h_gkc != nullptr) {
    degree_mix_acc<scalar_t, L, RANK>(
        xl_x, h_gkc + edge * (long)(NS0 * NS0 + L * L) * RANK, cbv, out_h);
  }
#pragma unroll
  for (int r = 0; r < RED; ++r) {
    ub[r * cf] = (scalar_t)out_u[r];
    hb[r * cf] = (scalar_t)out_h[r];
  }
}

// ---------------------------------------------------------------------------
// Rotation-curvature kernel for the second order: one traversal evaluates
// the three multilinear re-entries of the rotation backward,
//
//   <h_e,    RotBwd(gy; x slot)>      -> Wigner, kernel, basis curvature
//   <h_gwig, RotBwd(gy; wig slot)>    -> node, kernel, basis curvature
//   <h_gkc,  RotBwd(gy; kc slot)>     -> node, Wigner, basis curvature
//
// against the shared upstream gy = grad_u0. Terms landing on the same
// output are linear in the rotated lanes, so they merge before the block
// reductions: the kernel gradient contracts gy against the summed lanes
// rot(wig) h_e + rot(h_gwig) x, the Wigner gradient sums the kc- and
// h_gkc-mixed local gradients' outer products, and the node gradient sums
// the two projections. Every operand is read once.
// ---------------------------------------------------------------------------
template <typename scalar_t, int L, int RANK, int MAX_THREADS, int MIN_BLOCKS>
__global__ __launch_bounds__(
    MAX_THREADS,
    MIN_BLOCKS) void rotate_mix_bwd2_kernel(const scalar_t* __restrict__ gu,
                                            const scalar_t* __restrict__ x,
                                            const scalar_t* __restrict__ h_gx,
                                            const long* __restrict__ src,
                                            const scalar_t* __restrict__ wig,
                                            const scalar_t* __restrict__ h_gwig,
                                            const scalar_t* __restrict__ kc,
                                            const scalar_t* __restrict__ h_gkc,
                                            const scalar_t* __restrict__ cb,
                                            scalar_t* __restrict__ gxe,
                                            scalar_t* __restrict__ gw,
                                            scalar_t* __restrict__ gkc,
                                            scalar_t* __restrict__ pcb,
                                            long n_edge,
                                            long x_sn,
                                            long x_sd,
                                            long h_sn,
                                            long h_sd,
                                            int cf,
                                            int c_wide) {
  constexpr int NS0 = L + 1;
  constexpr int RED = 3 * L + 1;
  constexpr int DIM = (L + 1) * (L + 1);
  constexpr int NW = 3 * DIM - 2;
  // Batched-reduction scratch, as in the first-order backward.
  constexpr int KC_SLOTS = RANK > 0 ? (NS0 * NS0 + L * L) * RANK : 1;
  constexpr int WIG_SLOTS = NW;
  constexpr int MAX_WARPS = (MAX_THREADS + 31) / 32;
  __shared__ float part_kc[KC_SLOTS * MAX_WARPS];
  __shared__ float part_wig[WIG_SLOTS * MAX_WARPS];

  const long edge = blockIdx.x;
  if (edge >= n_edge) {
    return;
  }
  const int c = threadIdx.x;
  const bool active = c < c_wide;
  const int n_warps = (int)((blockDim.x + 31) >> 5);
  const long row_w = (long)RED * cf;
  const long s = src[edge];
  const scalar_t* db = wig + edge * NW;
  const scalar_t* dbh = h_gwig != nullptr ? h_gwig + edge * NW : nullptr;

  // === Phase 0. Rotated lanes (the raw rows are re-read from L2 in the
  // phase-2 outer products; two DIM-wide register arrays are what would
  // otherwise cap the residency) ===
  const scalar_t* xb = x + s * x_sn + (active ? c : 0);
  const scalar_t* hxb = h_gx + s * h_sn + (active ? c : 0);
  // xl_x: rot(wig) x, feeds the h_gkc-route basis partials.
  // xl_s: rot(wig) h_e + rot(h_gwig) x, feeds the kernel curvature.
  float xl_x[RED];
  float xl_s[RED];
  {
    float xr[DIM];
#pragma unroll
    for (int r = 0; r < DIM; ++r) {
      xr[r] = active ? (float)xb[r * x_sd] : 0.0f;
    }
    rotate_lane<scalar_t, L>(xr, db, xl_x);
    if (dbh != nullptr) {
      float xl_w[RED];
      rotate_lane<scalar_t, L>(xr, dbh, xl_w);
#pragma unroll
      for (int r = 0; r < RED; ++r) {
        xl_s[r] = xl_w[r];
      }
    } else {
#pragma unroll
      for (int r = 0; r < RED; ++r) {
        xl_s[r] = 0.0f;
      }
    }
#pragma unroll
    for (int r = 0; r < DIM; ++r) {
      xr[r] = active ? (float)hxb[r * h_sd] : 0.0f;
    }
    float xl_h[RED];
    rotate_lane<scalar_t, L>(xr, db, xl_h);
#pragma unroll
    for (int r = 0; r < RED; ++r) {
      xl_s[r] += xl_h[r];
    }
  }

  float cbv[RANK > 0 ? RANK : 1];
#pragma unroll
  for (int t = 0; t < RANK; ++t) {
    cbv[t] = active ? (float)cb[t * (long)c_wide + c] : 0.0f;
  }

  const scalar_t* gub =
      gu + (long)(c / cf) * n_edge * row_w + edge * row_w + (c % cf);
  float gy[RED];
#pragma unroll
  for (int r = 0; r < RED; ++r) {
    gy[r] = active ? (float)gub[r * cf] : 0.0f;
  }
  // === Phase 1. Kernel curvature: gy contracted against the summed lanes ===
  if (RANK == 0) {
    if (active) {
      scalar_t* gkb = gkc + edge * (long)NS0 * c_wide + c;
      gkb[0] = (scalar_t)(gy[0] * xl_s[0]);
#pragma unroll
      for (int d = 1; d < NS0; ++d) {
        gkb[d * (long)c_wide] =
            (scalar_t)(gy[d] * xl_s[d] + gy[NS0 + d - 1] * xl_s[NS0 + d - 1] +
                       gy[NS0 + L + d - 1] * xl_s[NS0 + L + d - 1]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < NS0; ++i) {
#pragma unroll
      for (int o = 0; o < NS0; ++o) {
        const float prod = gy[o] * xl_s[i];
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          warp_partial_sum(prod * cbv[t], (i * NS0 + o) * RANK + t, n_warps,
                           part_kc);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < L; ++i) {
#pragma unroll
      for (int o = 0; o < L; ++o) {
        const float prod =
            gy[NS0 + o] * xl_s[NS0 + i] + gy[NS0 + L + o] * xl_s[NS0 + L + i];
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          warp_partial_sum(prod * cbv[t], (NS0 * NS0 + i * L + o) * RANK + t,
                           n_warps, part_kc);
        }
      }
    }
  }

  // === Phase 2. Local gradients of both kernel routes; Wigner curvature,
  // node curvature and basis partials accumulate alongside ===
  scalar_t* gdb = gw + edge * NW;
  scalar_t* gxb = gxe != nullptr
                      ? gxe + edge * (long)DIM * c_wide + (active ? c : 0)
                      : nullptr;
  const scalar_t* kb = RANK == 0 ? kc + edge * (long)NS0 * c_wide
                                 : kc + edge * (long)(NS0 * NS0 + L * L) * RANK;
  const scalar_t* khb =
      h_gkc == nullptr
          ? nullptr
          : (RANK == 0 ? h_gkc + edge * (long)NS0 * c_wide
                       : h_gkc + edge * (long)(NS0 * NS0 + L * L) * RANK);
  float pcb_acc[RANK > 0 ? RANK : 1];
#pragma unroll
  for (int t = 0; t < RANK; ++t) {
    pcb_acc[t] = 0.0f;
  }
#pragma unroll
  for (int l = 0; l <= L; ++l) {
    const int base = l * l;
    // g_k: local gradient through the stored kernel (row l).
    // g_h: local gradient through the kernel cotangent (row l).
    float g0k = 0.0f, gmk = 0.0f, gpk = 0.0f;
    float g0h = 0.0f, gmh = 0.0f, gph = 0.0f;
    if (RANK == 0) {
      const float rad_l = active ? (float)kb[l * (long)c_wide + c] : 0.0f;
      g0k = gy[l] * rad_l;
      if (l >= 1) {
        gmk = gy[NS0 + l - 1] * rad_l;
        gpk = gy[NS0 + L + l - 1] * rad_l;
      }
      if (khb != nullptr) {
        const float hr = active ? (float)khb[l * (long)c_wide + c] : 0.0f;
        g0h = gy[l] * hr;
        if (l >= 1) {
          gmh = gy[NS0 + l - 1] * hr;
          gph = gy[NS0 + L + l - 1] * hr;
        }
      }
    } else {
      float raw0[RANK > 0 ? RANK : 1];
#pragma unroll
      for (int t = 0; t < RANK; ++t) {
        raw0[t] = 0.0f;
      }
#pragma unroll
      for (int o = 0; o < NS0; ++o) {
        float keff = 0.0f, heff = 0.0f;
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          const float k = (float)kb[(l * NS0 + o) * RANK + t];
          keff += k * cbv[t];
          raw0[t] += k * gy[o];
        }
        if (khb != nullptr) {
#pragma unroll
          for (int t = 0; t < RANK; ++t) {
            heff += (float)khb[(l * NS0 + o) * RANK + t] * cbv[t];
          }
          g0h += heff * gy[o];
        }
        g0k += keff * gy[o];
      }
#pragma unroll
      for (int t = 0; t < RANK; ++t) {
        pcb_acc[t] += raw0[t] * xl_s[l];
      }
      if (khb != nullptr) {
#pragma unroll
        for (int o = 0; o < NS0; ++o) {
          float hraw[RANK > 0 ? RANK : 1];
#pragma unroll
          for (int t = 0; t < RANK; ++t) {
            hraw[t] = (float)khb[(l * NS0 + o) * RANK + t] * gy[o];
            pcb_acc[t] += hraw[t] * xl_x[l];
          }
        }
      }
      if (l >= 1) {
        float rawm[RANK > 0 ? RANK : 1];
        float rawp[RANK > 0 ? RANK : 1];
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          rawm[t] = 0.0f;
          rawp[t] = 0.0f;
        }
#pragma unroll
        for (int o = 0; o < L; ++o) {
          float keff = 0.0f, heff = 0.0f;
#pragma unroll
          for (int t = 0; t < RANK; ++t) {
            const float k = (float)kb[(NS0 * NS0 + (l - 1) * L + o) * RANK + t];
            keff += k * cbv[t];
            rawm[t] += k * gy[NS0 + o];
            rawp[t] += k * gy[NS0 + L + o];
          }
          gmk += keff * gy[NS0 + o];
          gpk += keff * gy[NS0 + L + o];
          if (khb != nullptr) {
#pragma unroll
            for (int t = 0; t < RANK; ++t) {
              const float h =
                  (float)khb[(NS0 * NS0 + (l - 1) * L + o) * RANK + t];
              heff += h * cbv[t];
              pcb_acc[t] += h * (gy[NS0 + o] * xl_x[NS0 + l - 1] +
                                 gy[NS0 + L + o] * xl_x[NS0 + L + l - 1]);
            }
            gmh += heff * gy[NS0 + o];
            gph += heff * gy[NS0 + L + o];
          }
        }
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          pcb_acc[t] +=
              rawm[t] * xl_s[NS0 + l - 1] + rawp[t] * xl_s[NS0 + L + l - 1];
        }
      }
    }
    {
#pragma unroll
      for (int j = 0; j < 2 * l + 1; ++j) {
        const int col = base + j;
        const float xv = active ? (float)xb[col * x_sd] : 0.0f;
        const float hv = active ? (float)hxb[col * h_sd] : 0.0f;
        // Wigner curvature: kc-route outer product against the cotangent
        // rows plus h_gkc-route outer product against the feature rows.
        warp_partial_sum(g0k * hv + g0h * xv, base + j, n_warps, part_wig);
        float gx_row = 0.0f;
        if (dbh != nullptr) {
          gx_row += (float)dbh[base + j] * g0k;
        }
        if (khb != nullptr) {
          gx_row += (float)db[base + j] * g0h;
        }
        if (l >= 1) {
          const int minus = DIM + base - 1 + j;
          const int plus = 2 * DIM + base - 2 + j;
          warp_partial_sum(gmk * hv + gmh * xv, minus, n_warps, part_wig);
          warp_partial_sum(gpk * hv + gph * xv, plus, n_warps, part_wig);
          if (dbh != nullptr) {
            gx_row += (float)dbh[minus] * gmk + (float)dbh[plus] * gpk;
          }
          if (khb != nullptr) {
            gx_row += (float)db[minus] * gmh + (float)db[plus] * gph;
          }
        }
        if (gxb != nullptr && active) {
          gxb[col * (long)c_wide] = (scalar_t)gx_row;
        }
      }
    }
  }
  if (RANK > 0 && active) {
    scalar_t* pcb_out = pcb + edge * (long)RANK * c_wide + c;
#pragma unroll
    for (int t = 0; t < RANK; ++t) {
      pcb_out[t * (long)c_wide] = (scalar_t)pcb_acc[t];
    }
  }

  // === Phase 3. One barrier completes every batched reduction ===
  __syncthreads();
  if (RANK > 0) {
    scalar_t* gkb = gkc + edge * (long)(NS0 * NS0 + L * L) * RANK;
    for (int s2 = threadIdx.x; s2 < KC_SLOTS; s2 += blockDim.x) {
      gkb[s2] = (scalar_t)finish_partial_sum(part_kc, s2, n_warps);
    }
  }
  for (int s2 = threadIdx.x; s2 < WIG_SLOTS; s2 += blockDim.x) {
    gdb[s2] = (scalar_t)finish_partial_sum(part_wig, s2, n_warps);
  }
}

// ---------------------------------------------------------------------------
// Backward: one block per edge, one thread per channel. Recomputes the
// rotated rows, then emits the degree-kernel gradient (block channel
// reductions for the compact kernels), the Wigner gradient on the
// structural non-zeros, and the dense per-edge node gradient.
// ---------------------------------------------------------------------------
// The reduction-heavy body wants registers; unconstrained the compiler
// allocates ~165 per thread and the residency collapses to one block per
// multiprocessor with the DRAM pipe mostly idle. Capping at two 256-thread
// blocks trades a modest register spill (absorbed by the idle L2) for
// twice the latency cover, which is what the measured occupancy needed.
template <typename scalar_t, int L, int RANK, int MAX_THREADS, int MIN_BLOCKS>
__global__ __launch_bounds__(
    MAX_THREADS,
    MIN_BLOCKS) void rotate_mix_bwd_kernel(const scalar_t* __restrict__ gu,
                                           const scalar_t* __restrict__ x,
                                           const long* __restrict__ src,
                                           const scalar_t* __restrict__ wig,
                                           const scalar_t* __restrict__ kc,
                                           const scalar_t* __restrict__ cb,
                                           scalar_t* __restrict__ gxe,
                                           scalar_t* __restrict__ gw,
                                           scalar_t* __restrict__ gkc,
                                           scalar_t* __restrict__ pcb,
                                           long n_edge,
                                           long x_sn,
                                           long x_sd,
                                           int cf,
                                           int c_wide) {
  constexpr int NS0 = L + 1;
  constexpr int RED = 3 * L + 1;
  constexpr int DIM = (L + 1) * (L + 1);
  constexpr int NW = 3 * DIM - 2;
  // Batched-reduction scratch: one partial per (slot, warp). Kernel-gradient
  // slots map linearly onto the compact kernel layout; Wigner slots follow
  // the block-diagonal enumeration of phase 2.
  constexpr int KC_SLOTS = RANK > 0 ? (NS0 * NS0 + L * L) * RANK : 1;
  constexpr int WIG_SLOTS = NW;
  constexpr int MAX_WARPS = (MAX_THREADS + 31) / 32;
  __shared__ float part_kc[KC_SLOTS * MAX_WARPS];
  __shared__ float part_wig[WIG_SLOTS * MAX_WARPS];
  __shared__ scalar_t edge_runs[L == kMaxLmax ? NW : 1];
  __shared__ scalar_t edge_kernel[L == kMaxLmax ? KC_SLOTS : 1];

  const long edge = blockIdx.x;
  if (edge >= n_edge) {
    return;
  }
  const int c = threadIdx.x;
  const bool active = c < c_wide;
  const int n_warps = (int)((blockDim.x + 31) >> 5);
  const long row_w = (long)RED * cf;

  const long s = src[edge];
  const scalar_t* xb = x + s * x_sn + (active ? c : 0);
  const scalar_t* db = wig + edge * NW;
  if constexpr (L == kMaxLmax) {
    for (int i = threadIdx.x; i < NW; i += blockDim.x) {
      edge_runs[i] = db[i];
    }
    if constexpr (RANK > 0) {
      const scalar_t* global_kernel =
          kc + edge * (long)(NS0 * NS0 + L * L) * RANK;
      for (int i = threadIdx.x; i < KC_SLOTS; i += blockDim.x) {
        edge_kernel[i] = global_kernel[i];
      }
    }
    __syncthreads();
    db = edge_runs;
  }

  // === Phase 0. Recompute the rotated rows (the raw rows are re-read from
  // L2 in phase 2 rather than held: DIM registers per thread are exactly
  // what caps this kernel's residency) ===
  float xl[RED];
  {
    float xr[DIM];
#pragma unroll
    for (int r = 0; r < DIM; ++r) {
      xr[r] = active ? (float)xb[r * x_sd] : 0.0f;
    }
#pragma unroll
    for (int l = 0; l <= L; ++l) {
      const int base = l * l;
      float a0 = 0.0f, am = 0.0f, ap = 0.0f;
#pragma unroll
      for (int j = 0; j < 2 * l + 1; ++j) {
        const float xv = xr[base + j];
        a0 += (float)db[base + j] * xv;
        if (l >= 1) {
          am += (float)db[DIM + base - 1 + j] * xv;
          ap += (float)db[2 * DIM + base - 2 + j] * xv;
        }
      }
      xl[l] = a0;
      if (l >= 1) {
        xl[NS0 + l - 1] = am;
        xl[NS0 + L + l - 1] = ap;
      }
    }
  }

  float cbv[RANK > 0 ? RANK : 1];
#pragma unroll
  for (int t = 0; t < RANK; ++t) {
    cbv[t] = active ? (float)cb[t * (long)c_wide + c] : 0.0f;
  }

  // The raw upstream rows: the degree-kernel and channel-basis gradients
  // contract against them directly, while the rotation gradient of the
  // rank-1 form folds the single basis in at use.
  const scalar_t* gub =
      gu + (long)(c / cf) * n_edge * row_w + edge * row_w + (c % cf);
  float gy[RED];
#pragma unroll
  for (int r = 0; r < RED; ++r) {
    gy[r] = active ? (float)gub[r * cf] : 0.0f;
  }

  // === Phase 1. Degree-kernel (or radial-feature) gradient ===
  if (RANK == 0) {
    if (active) {
      scalar_t* gkb = gkc + edge * (long)NS0 * c_wide + c;
      gkb[0] = (scalar_t)(gy[0] * xl[0]);
#pragma unroll
      for (int d = 1; d < NS0; ++d) {
        gkb[d * (long)c_wide] =
            (scalar_t)(gy[d] * xl[d] + gy[NS0 + d - 1] * xl[NS0 + d - 1] +
                       gy[NS0 + L + d - 1] * xl[NS0 + L + d - 1]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < NS0; ++i) {
#pragma unroll
      for (int o = 0; o < NS0; ++o) {
        const float prod = gy[o] * xl[i];
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          warp_partial_sum(prod * cbv[t], (i * NS0 + o) * RANK + t, n_warps,
                           part_kc);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < L; ++i) {
#pragma unroll
      for (int o = 0; o < L; ++o) {
        const float prod =
            gy[NS0 + o] * xl[NS0 + i] + gy[NS0 + L + o] * xl[NS0 + L + i];
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          warp_partial_sum(prod * cbv[t], (NS0 * NS0 + i * L + o) * RANK + t,
                           n_warps, part_kc);
        }
      }
    }
  }

  // === Phase 2. Rotation backward with g_local formed on the fly; the
  // channel-basis partials accumulate alongside since every operand is
  // already in registers ===
  scalar_t* gdb = gw + edge * NW;
  scalar_t* gxb = gxe + edge * (long)DIM * c_wide + (active ? c : 0);
  const scalar_t* kb =
      RANK == 0
          ? kc + edge * (long)NS0 * c_wide
          : (L == kMaxLmax ? edge_kernel
                           : kc + edge * (long)(NS0 * NS0 + L * L) * RANK);
  float pcb_acc[RANK > 0 ? RANK : 1];
#pragma unroll
  for (int t = 0; t < RANK; ++t) {
    pcb_acc[t] = 0.0f;
  }
#pragma unroll
  for (int l = 0; l <= L; ++l) {
    const int base = l * l;
    float g0 = 0.0f, gm = 0.0f, gp = 0.0f;
    if (RANK == 0) {
      const float rad_l = active ? (float)kb[l * (long)c_wide + c] : 0.0f;
      g0 = gy[l] * rad_l;
      if (l >= 1) {
        gm = gy[NS0 + l - 1] * rad_l;
        gp = gy[NS0 + L + l - 1] * rad_l;
      }
    } else {
      float raw0[RANK > 0 ? RANK : 1];
#pragma unroll
      for (int t = 0; t < RANK; ++t) {
        raw0[t] = 0.0f;
      }
#pragma unroll
      for (int o = 0; o < NS0; ++o) {
        float keff = 0.0f;
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          const float k = (float)kb[(l * NS0 + o) * RANK + t];
          keff += k * cbv[t];
          raw0[t] += k * gy[o];
        }
        g0 += keff * gy[o];
      }
#pragma unroll
      for (int t = 0; t < RANK; ++t) {
        pcb_acc[t] += raw0[t] * xl[l];
      }
      if (l >= 1) {
        float rawm[RANK > 0 ? RANK : 1];
        float rawp[RANK > 0 ? RANK : 1];
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          rawm[t] = 0.0f;
          rawp[t] = 0.0f;
        }
#pragma unroll
        for (int o = 0; o < L; ++o) {
          float keff = 0.0f;
#pragma unroll
          for (int t = 0; t < RANK; ++t) {
            const float k = (float)kb[(NS0 * NS0 + (l - 1) * L + o) * RANK + t];
            keff += k * cbv[t];
            rawm[t] += k * gy[NS0 + o];
            rawp[t] += k * gy[NS0 + L + o];
          }
          gm += keff * gy[NS0 + o];
          gp += keff * gy[NS0 + L + o];
        }
#pragma unroll
        for (int t = 0; t < RANK; ++t) {
          pcb_acc[t] +=
              rawm[t] * xl[NS0 + l - 1] + rawp[t] * xl[NS0 + L + l - 1];
        }
      }
    }
    {
#pragma unroll
      for (int j = 0; j < 2 * l + 1; ++j) {
        const int col = base + j;
        const float xv = active ? (float)xb[col * x_sd] : 0.0f;
        float gx_row = (float)db[base + j] * g0;
        warp_partial_sum(g0 * xv, base + j, n_warps, part_wig);
        if (l >= 1) {
          const int minus = DIM + base - 1 + j;
          const int plus = 2 * DIM + base - 2 + j;
          gx_row += (float)db[minus] * gm + (float)db[plus] * gp;
          warp_partial_sum(gm * xv, minus, n_warps, part_wig);
          warp_partial_sum(gp * xv, plus, n_warps, part_wig);
        }
        if (active) {
          gxb[col * (long)c_wide] = (scalar_t)gx_row;
        }
      }
    }
  }
  if (RANK > 0 && active) {
    scalar_t* pcb_out = pcb + edge * (long)RANK * c_wide + c;
#pragma unroll
    for (int t = 0; t < RANK; ++t) {
      pcb_out[t * (long)c_wide] = (scalar_t)pcb_acc[t];
    }
  }

  // === Phase 3. One barrier completes every batched reduction ===
  __syncthreads();
  if (RANK > 0) {
    scalar_t* gkb = gkc + edge * (long)(NS0 * NS0 + L * L) * RANK;
    for (int s2 = threadIdx.x; s2 < KC_SLOTS; s2 += blockDim.x) {
      gkb[s2] = (scalar_t)finish_partial_sum(part_kc, s2, n_warps);
    }
  }
  for (int s2 = threadIdx.x; s2 < WIG_SLOTS; s2 += blockDim.x) {
    gdb[s2] = (scalar_t)finish_partial_sum(part_wig, s2, n_warps);
  }
}

// ---------------------------------------------------------------------------
// CSR segment sum: out[seg] = sum of rows[order[i]] over the segment's span.
// One block per (segment, feature-tile); rows are (R, F1, F2) flattened over
// the last two axes.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void segment_sum_kernel(const scalar_t* __restrict__ rows,
                                   const long* __restrict__ order,
                                   const long* __restrict__ row_ptr,
                                   scalar_t* __restrict__ out,
                                   long n_seg,
                                   long feat) {
  const long seg = blockIdx.x;
  if (seg >= n_seg) {
    return;
  }
  const long lo = row_ptr[seg];
  const long hi = row_ptr[seg + 1];
  for (long f = blockIdx.y * (long)blockDim.x + threadIdx.x; f < feat;
       f += (long)gridDim.y * blockDim.x) {
    float acc = 0.0f;
    for (long i = lo; i < hi; ++i) {
      acc += (float)rows[order[i] * feat + f];
    }
    out[seg * feat + f] = (scalar_t)acc;
  }
}

inline void check_rotate_inputs(const at::Tensor& x,
                                const at::Tensor& src,
                                const at::Tensor& runs,
                                int64_t lmax,
                                int64_t n_focus,
                                int64_t rank,
                                const char* who) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.stride(2) == 1, who,
              ": x must be (N, D, C_wide) with unit channel stride");
  TORCH_CHECK(1 <= lmax && lmax <= kMaxLmax, who, ": unsupported lmax");
  TORCH_CHECK(0 <= rank && rank <= kMaxRank, who, ": unsupported rank");
  TORCH_CHECK(1 <= n_focus && n_focus <= kMaxRotateFocus, who,
              ": unsupported focus count");
  TORCH_CHECK(x.size(1) == (lmax + 1) * (lmax + 1), who,
              ": x degree dimension does not match lmax");
  TORCH_CHECK(x.size(2) % n_focus == 0, who,
              ": channel width must split into the focus streams");
  TORCH_CHECK(x.size(2) <= kNarrowChannelLanes ||
                  (lmax == kMaxLmax && x.size(2) <= kWideChannelLanes),
              who, ": channel width exceeds the supported block lane count");
  const int64_t dim = (lmax + 1) * (lmax + 1);
  TORCH_CHECK(runs.is_contiguous() && runs.dim() == 2 &&
                  runs.size(0) == src.size(0) && runs.size(1) == 3 * dim - 2,
              who, ": runs must be contiguous (E, 3 * DIM - 2)");
  TORCH_CHECK(src.scalar_type() == at::kLong, who, ": src must be int64");
}

// Dispatch helper over the compile-time (L, RANK) grid.
template <typename F>
void dispatch_l_rank(int64_t lmax, int64_t rank, const F& f) {
  const int key = (int)lmax * 8 + (int)rank;
  switch (key) {
#define DPA4_RM_CASE(L, R)                                                 \
  case L * 8 + R:                                                          \
    f(std::integral_constant<int, L>{}, std::integral_constant<int, R>{}); \
    break;
#define DPA4_RM_CASES_FOR_L(L) \
  DPA4_RM_CASE(L, 0)           \
  DPA4_RM_CASE(L, 1)           \
  DPA4_RM_CASE(L, 2)           \
  DPA4_RM_CASE(L, 3)           \
  DPA4_RM_CASE(L, 4)
    DPA4_RM_CASES_FOR_L(1)
    DPA4_RM_CASES_FOR_L(2)
    DPA4_RM_CASES_FOR_L(3)
    DPA4_RM_CASES_FOR_L(4)
    DPA4_RM_CASES_FOR_L(5)
    DPA4_RM_CASES_FOR_L(6)
#undef DPA4_RM_CASES_FOR_L
#undef DPA4_RM_CASE
    default:
      TORCH_CHECK(false, "sezm_rotate_mix: unsupported (lmax, rank)");
  }
}

// ---------------------------------------------------------------------------
// Host launchers: one static (L, RANK, dtype) specialization per build shard.
// ---------------------------------------------------------------------------
template <typename scalar_t, int L, int RANK>
void launch_rotate_mix_fwd(const scalar_t* x,
                           const long* src,
                           const scalar_t* wig,
                           const scalar_t* kc,
                           const scalar_t* cb,
                           scalar_t* u,
                           long n_edge,
                           long x_sn,
                           long x_sd,
                           int cf,
                           int c_wide,
                           int threads,
                           cudaStream_t stream) {
  rotate_mix_fwd_kernel<scalar_t, L, RANK><<<n_edge, threads, 0, stream>>>(
      x, src, wig, kc, cb, u, n_edge, x_sn, x_sd, cf, c_wide);
}

template <typename scalar_t, int L, int RANK>
void launch_rotate_mix_fwd_pair(const scalar_t* x,
                                const scalar_t* h_gx,
                                const long* src,
                                const scalar_t* wig,
                                const scalar_t* h_gwig,
                                const scalar_t* kc,
                                const scalar_t* h_gkc,
                                const scalar_t* cb,
                                scalar_t* u0,
                                scalar_t* hgu0,
                                long n_edge,
                                long x_sn,
                                long x_sd,
                                long h_sn,
                                long h_sd,
                                int cf,
                                int c_wide,
                                int threads,
                                cudaStream_t stream) {
  if constexpr (L == kMaxLmax) {
    if (threads > kNarrowChannelLanes) {
      rotate_mix_fwd_pair_kernel<scalar_t, L, RANK, kWideChannelLanes, 1>
          <<<n_edge, threads, 0, stream>>>(x, h_gx, src, wig, h_gwig, kc, h_gkc,
                                           cb, u0, hgu0, n_edge, x_sn, x_sd,
                                           h_sn, h_sd, cf, c_wide);
      return;
    }
    if (threads == kMediumChannelLanes) {
      rotate_mix_fwd_pair_kernel<scalar_t, L, RANK, kMediumChannelLanes, 2>
          <<<n_edge, threads, 0, stream>>>(x, h_gx, src, wig, h_gwig, kc, h_gkc,
                                           cb, u0, hgu0, n_edge, x_sn, x_sd,
                                           h_sn, h_sd, cf, c_wide);
      return;
    }
  }
  rotate_mix_fwd_pair_kernel<scalar_t, L, RANK, kNarrowChannelLanes, 2>
      <<<n_edge, threads, 0, stream>>>(x, h_gx, src, wig, h_gwig, kc, h_gkc, cb,
                                       u0, hgu0, n_edge, x_sn, x_sd, h_sn, h_sd,
                                       cf, c_wide);
}

template <typename scalar_t, int L, int RANK>
void launch_rotate_mix_bwd(const scalar_t* gu,
                           const scalar_t* x,
                           const long* src,
                           const scalar_t* wig,
                           const scalar_t* kc,
                           const scalar_t* cb,
                           scalar_t* gxe,
                           scalar_t* gw,
                           scalar_t* gkc,
                           scalar_t* pcb,
                           long n_edge,
                           long x_sn,
                           long x_sd,
                           int cf,
                           int c_wide,
                           int threads,
                           cudaStream_t stream) {
  if constexpr (L == kMaxLmax) {
    if (threads > kNarrowChannelLanes) {
      rotate_mix_bwd_kernel<scalar_t, L, RANK, kWideChannelLanes, 1>
          <<<n_edge, threads, 0, stream>>>(gu, x, src, wig, kc, cb, gxe, gw,
                                           gkc, pcb, n_edge, x_sn, x_sd, cf,
                                           c_wide);
      return;
    }
    if (threads == kMediumChannelLanes) {
      rotate_mix_bwd_kernel<scalar_t, L, RANK, kMediumChannelLanes, 2>
          <<<n_edge, threads, 0, stream>>>(gu, x, src, wig, kc, cb, gxe, gw,
                                           gkc, pcb, n_edge, x_sn, x_sd, cf,
                                           c_wide);
      return;
    }
  }
  rotate_mix_bwd_kernel<scalar_t, L, RANK, kNarrowChannelLanes, 2>
      <<<n_edge, threads, 0, stream>>>(gu, x, src, wig, kc, cb, gxe, gw, gkc,
                                       pcb, n_edge, x_sn, x_sd, cf, c_wide);
}

template <typename scalar_t, int L, int RANK>
void launch_rotate_mix_bwd2(const scalar_t* gu,
                            const scalar_t* x,
                            const scalar_t* h_gx,
                            const long* src,
                            const scalar_t* wig,
                            const scalar_t* h_gwig,
                            const scalar_t* kc,
                            const scalar_t* h_gkc,
                            const scalar_t* cb,
                            scalar_t* gxe,
                            scalar_t* gw,
                            scalar_t* gkc,
                            scalar_t* pcb,
                            long n_edge,
                            long x_sn,
                            long x_sd,
                            long h_sn,
                            long h_sd,
                            int cf,
                            int c_wide,
                            int threads,
                            cudaStream_t stream) {
  if constexpr (L == kMaxLmax) {
    if (threads > kNarrowChannelLanes) {
      rotate_mix_bwd2_kernel<scalar_t, L, RANK, kWideChannelLanes, 1>
          <<<n_edge, threads, 0, stream>>>(gu, x, h_gx, src, wig, h_gwig, kc,
                                           h_gkc, cb, gxe, gw, gkc, pcb, n_edge,
                                           x_sn, x_sd, h_sn, h_sd, cf, c_wide);
      return;
    }
    if (threads == kMediumChannelLanes) {
      rotate_mix_bwd2_kernel<scalar_t, L, RANK, kMediumChannelLanes, 2>
          <<<n_edge, threads, 0, stream>>>(gu, x, h_gx, src, wig, h_gwig, kc,
                                           h_gkc, cb, gxe, gw, gkc, pcb, n_edge,
                                           x_sn, x_sd, h_sn, h_sd, cf, c_wide);
      return;
    }
  }
  rotate_mix_bwd2_kernel<scalar_t, L, RANK, kNarrowChannelLanes, 2>
      <<<n_edge, threads, 0, stream>>>(gu, x, h_gx, src, wig, h_gwig, kc, h_gkc,
                                       cb, gxe, gw, gkc, pcb, n_edge, x_sn,
                                       x_sd, h_sn, h_sd, cf, c_wide);
}

}  // namespace dpa4_sezm_kernels
