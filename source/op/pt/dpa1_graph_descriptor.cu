// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused CUDA descriptor for the DPA1 (``se_atten``) graph lower -- the
// attention-free configuration (concat or strip tebd input) evaluated on
// the flat edge stream. One forward kernel produces the per-node moment
// matrix from ``edge_vec``; one backward kernel returns ``d(edge_vec)`` for
// the analytic force / virial assembly. No ``(E, .)`` activation tensor is
// ever materialized on the autograd tape: the embedding runs on
// shared-memory tiles, and the two operands the backward cannot recompute
// cheaply are spilled once in a streaming layout (see "Saved tensors"
// below).
//
// Mathematical contract (matches ``DescrptBlockSeAtten`` at attn_layer 0):
//   rr   = ((sw/q, sw*x/q^2, sw*y/q^2, sw*z/q^2) - davg[ct]) / dstd[ct]
//          with q = |r| + protection, sw the quintic smooth switch of |r| on
//          [rcut_smth, rcut]; masked (padding) edges use |r| + 1.
//   in   = [rr0, tebd[nt], tebd[ct]]   (concat)  |  rr0   (strip)
//   h1   = act(in @ W1 + b1) * idt1
//   h2   = act(h1 @ W2 + b2) * idt2 (+ [h1, h1] doubling residual)
//   g    = act(h2 @ W3 + b3) * idt3 (+ [h2, h2] doubling residual)
//   gg   = g                                     (concat)
//        = g * (1 + gate[pair] (* sw if smooth)) (strip type-pair gate,
//          gate = embeddings_strip(tebd pairs), precomputed by the caller)
//   gr[n, i, k] = (1/nnei) * segment_sum(gg[:, i] * mask * rr[:, k], dst)
//   grrg[n, i * axis + j] = sum_k gr[n, i, k] * gr[n, j, k]     (j < axis)
//   rot_mat[n, i, c]      = gr[n, i, 1 + c]
//   grrg tail (concat_output_tebd): type_embedding[atype[n]].
// The moment is stored transposed as (N, 4, NG); both contractions above are
// invariant to that layout, so downstream results equal the reference up to
// fp32 summation order.
//
// Algebraic reductions
// --------------------
// * Pair table: the concat embedding input is [rr0 | tebd[nt] | tebd[ct]],
//   whose type-embedding block takes only ``ntypes^2`` distinct values. The
//   host folds those rows of W1 (plus b1) into a per-type-pair table of
//   layer-1 pre-activations, so layer 1 degenerates to one FMA per channel:
//   ``pre1 = rr0 * W1[0] + pair_table[ct * T + nt]``. This removes the
//   layer-1 GEMM and every per-edge type-embedding gather. Strip mode keeps
//   the same code path with the table collapsed to its single bias row, and
//   applies the type-pair gate table in the layer-3 epilogue instead.
// * Sort-free CSR: graph builders emit real edges already dst-sorted, so the
//   edge ordering is a histogram + exclusive scan + identity permutation;
//   genuinely unsorted input falls back to an atomic-cursor scatter. This
//   replaces a per-step argsort.
//
// Kernel organization
// -------------------
// * Every dense stage is an 8x8 (edges x channels) register micro-tile whose
//   activation operand is one float4 shared-memory fragment and whose weight
//   operand is a warp-uniform float4 ``__ldg`` broadcast (L1-resident,
//   one transaction per warp). Weights are never staged in shared memory;
//   the same row-major layout serves the forward k-loops and the dgrad
//   loops (which fix the input channel's row and walk the output axis).
// * The moment reduction walks the tile's CSR runs per (moment row, channel)
//   slot and flushes one global atomicAdd per run -- no shuffle reductions
//   on the forward path.
// * The backward gathers the upstream moment gradient once per (edge, run):
//   rows of dgr for the tile's first RMAX runs are staged in shared memory,
//   and an all-edges-share-one-run fast path amortizes the four scalars per
//   channel over the whole eight-edge fragment. Per-edge partials of the
//   environment gradient accumulate in per-warp shared-memory banks (a
//   shuffle fold halves the writers), avoiding shared-memory atomics on the
//   hot path.
//
// Saved tensors (forward -> backward)
// -----------------------------------
// ``pre2`` (N2, E_pad) and -- tanh: ``g``; silu: ``pre3`` -- (NG, E_pad) are
// spilled transposed with streaming (evict-first) stores so both sides read
// coalesced rows without evicting the L1-resident weights. This halves the
// backward FLOPs relative to a full recompute at the cost of
// ``(N2 + NG) * 4`` bytes per edge. E_pad rounds E up to the tile size so
// partial-tile vector stores stay in bounds.
//
// Applicability (enforced by the Python gate): three embedding layers with
// N1 in {8, 16, 32, 64}, N2 in {N1, 2*N1}, NG in {N2, 2*N2}, N2 <= 64, and
// NG <= 128. Layers 2 and 3 implement identity and width-doubling residuals.
// Layer 1 is accepted only when its native input/output shape does not form a
// residual. All layers share tanh or silu, with optional timesteps, fp32
// weights, statistics, and compute. ``edge_vec`` in fp32 or fp64 is cast to
// fp32 on entry and the leaf dtype is restored on the returned gradient.
// ``concat`` or ``strip`` tebd input (strip with or without the smooth gate),
// ``attn_layer == 0``, no excluded type pairs.

#include <algorithm>
#include <optional>
#include <tuple>

#include "dpa1_graph_common.cuh"

namespace {

// Activation codes follow deepmd.kernels.triton.dpa1.activation.ACT_CODES:
// 0 = tanh, 1 = silu. Forward and backward share this helper so energy and
// its analytic force gradient stay consistent (the potential-energy surface
// remains smooth).
//
// The sigmoid factor of silu(z) = z * sigmoid(z) is evaluated through the
// identity sigmoid(z) = 0.5 * (1 + tanh(0.5 * z)). The accurate fp32 division
// in 1 / (1 + expf(-z)) -- not the exponential -- dominates the silu activation
// cost here; the tanh form removes that division, and since this architecture
// emits no dedicated tanh hardware instruction the tanh factor is no costlier
// than the exponential it replaces. Every silu site -- the forward value, its
// derivative, and the backward reconstruction of the pre-activation -- must use
// this identical expression; a mismatched sigmoid on either side would break
// the identity force == d(energy)/d(coord) at the fp32 level.
template <int ACT>
DEV_INLINE float act_val(float z) {
  if constexpr (ACT == 0) {
    return tanhf(z);
  }
  return 0.5f * z * (1.f + tanhf(0.5f * z));
}

// Activation value and derivative at z.
template <int ACT>
DEV_INLINE float2 act_vg(float z) {
  if constexpr (ACT == 0) {
    float a = tanhf(z);
    return make_float2(a, 1.f - a * a);
  }
  const float s = 0.5f * (1.f + tanhf(0.5f * z));  // sigmoid via tanh identity
  return make_float2(z * s, s * (1.f + z * (1.f - s)));
}

// ======================================================================
// Pair table: fold the type-embedding rows of W1 (and b1) into a per-pair
// layer-1 pre-activation. Two-side pairs index as p = ct * T + nt; one-side
// as p = nt.
// ======================================================================
__global__ void pair_table_kernel(int n_pairs,
                                  int ntypes,
                                  int tebd_dim,
                                  int n1,
                                  int one_side,
                                  const float* __restrict__ tebd,
                                  const float* __restrict__ w1,
                                  const float* __restrict__ b1,
                                  float* __restrict__ pair_table) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pairs * n1) {
    return;
  }
  const int p = idx / n1, o = idx % n1;
  const int nt = one_side ? p : p % ntypes;
  const int ct = one_side ? 0 : p / ntypes;
  float acc = b1 ? b1[o] : 0.f;
  for (int t = 0; t < tebd_dim; ++t) {
    acc = fmaf(tebd[nt * tebd_dim + t], w1[(1 + t) * n1 + o], acc);
  }
  if (!one_side) {
    for (int t = 0; t < tebd_dim; ++t) {
      acc = fmaf(tebd[ct * tebd_dim + t], w1[(1 + tebd_dim + t) * n1 + o], acc);
    }
  }
  pair_table[(long)p * n1 + o] = acc;
}

// Layer-1 tile from the pair table: float4 blocks of the (BE, N1) sheet, so
// the gather streams coalesced 16-byte words. Strip mode has no per-pair
// layer-1 term (the type embedding enters through the gate instead), so the
// table degenerates to its single bias row.
template <int N1, int ACT, int TILE, int STRIDE>
DEV_INLINE void layer1_tile(int tid,
                            const EdgeTablesT<TILE>& T,
                            int strip,
                            const float* __restrict__ pair_table,
                            const float* __restrict__ w1,
                            const float* __restrict__ idt1,
                            float (*h1)[STRIDE]) {
  constexpr int kBlocks = TILE * N1 / 4;
  for (int blk = tid; blk < kBlocks; blk += kThreads) {
    const int e = (blk * 4) / N1, o = (blk * 4) % N1;
    const long row = strip ? 0 : (long)T.pair_idx[e];
    const float4 pp =
        *reinterpret_cast<const float4*>(pair_table + row * N1 + o);
    const float4 ws = weight4(w1 + o);
    const float4 it =
        idt1 ? weight4(idt1 + o) : make_float4(1.f, 1.f, 1.f, 1.f);
    const float rv = T.radial[e];
    h1[o + 0][e] = act_val<ACT>(fmaf(rv, ws.x, pp.x)) * it.x;
    h1[o + 1][e] = act_val<ACT>(fmaf(rv, ws.y, pp.y)) * it.y;
    h1[o + 2][e] = act_val<ACT>(fmaf(rv, ws.z, pp.z)) * it.z;
    h1[o + 3][e] = act_val<ACT>(fmaf(rv, ws.w, pp.w)) * it.w;
  }
}

// ======================================================================
// Forward kernel.
//
// Persistent CTAs stride over BE-edge slices of the CSR-ordered stream.
// Stages per tile (tx = tid % 16 indexes eight-edge fragments, ty = tid / 16
// indexes channel groups):
//   0. environment matrix + run scan          (stage_tile)
//   1. h1 tile from the pair table            (layer1_tile)
//   2. GEMM2 (8e x 4c micro-tiles) -> h2 tile; spill pre2
//   3. GEMM3 (8e x 8c micro-tiles) -> g tile (overlaying h1 through the
//      stage union) + spill g (tanh) or pre3 (silu)
//   4. moment walk: thread owns channel c, streams each CSR run over the
//      tile rows with all four moment components, one atomicAdd per
//      (run, component, channel)
// ======================================================================
template <int N1, int N2, int NG, int TILE>
struct FwdSmem {
  static constexpr int STRIDE = TILE + 4;
  // The g tile overlays the (dead) h1 tile; NG >= N1, so the union is sized
  // by g. h1's last readers (the GEMM2 residual epilogue) finish before the
  // barrier that precedes the first g write.
  union {
    float h1[N1][STRIDE];  // stages 1-2
    float g[NG][STRIDE];   // stages 3-4 (walk view)
  } u;
  float h2[N2][STRIDE];
  EdgeTablesT<TILE> T;
};

static_assert(sizeof(FwdSmem<32, 64, 128, 128>) > 96 * 1024);
static_assert(sizeof(FwdSmem<32, 64, 128, 64>) <= 96 * 1024);

template <int N1, int N2, int NG, int ACT, int STRIP, int EPT>
__global__ __launch_bounds__(kThreads, 2) void dpa1_graph_forward_kernel(
    long n_edge,
    int ntypes,
    int one_side,
    int resnet2,
    int resnet3,
    int smooth,
    float rcut,
    float rcut_smth,
    float protection,
    float inv_nnei,
    const float* __restrict__ edge_vec,
    const long* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const long* __restrict__ atype,
    const float* __restrict__ davg,
    const float* __restrict__ inv_dstd,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    const float* __restrict__ idt1,
    const float* __restrict__ idt2,
    const float* __restrict__ w3,
    const float* __restrict__ b3,
    const float* __restrict__ idt3,
    const float* __restrict__ pair_table,
    const float* __restrict__ gate_table,
    const int* __restrict__ order,
    float* __restrict__ gr,
    float* __restrict__ pre2_saved,
    float* __restrict__ g_saved,
    long e_pad) {
  constexpr int TILE = 16 * EPT;  // edges per tile (16 edge-lanes x EPT)
  constexpr int STRIDE = TILE + 4;
  extern __shared__ char smem_raw[];
  auto& S = *reinterpret_cast<FwdSmem<N1, N2, NG, TILE>*>(smem_raw);
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;
  const int erow = tx * EPT;
  constexpr int strip = STRIP;

  const long n_tiles = ceil_div(n_edge, TILE);
  for (long tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    const long tile_base = tile * TILE;
    const int rows = (int)min((long)TILE, n_edge - tile_base);

    stage_tile<TILE>(tid, tile_base, rows, n_edge, ntypes, one_side, rcut,
                     rcut_smth, protection, inv_nnei, edge_vec, edge_index,
                     edge_mask, atype, davg, inv_dstd, order, S.T);
    layer1_tile<N1, ACT, TILE, STRIDE>(tid, S.T, strip, pair_table, w1, idt1,
                                       S.u.h1);
    __syncthreads();

    // === Step 2. GEMM2 -> h2 tile; spill pre2 (transposed (N2, E)) ===
    // The residual reads h1[c % N1], which covers both the doubling
    // ([h1, h1], N2 == 2 * N1) and the identity (N2 == N1) layer shapes.
    {
      const int c0 = ty * 4;  // 4 channels per group; groups beyond N2 idle
      if (c0 < N2) {
        float acc[4][EPT];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            acc[j][i] = 0.f;
          }
#pragma unroll 8
        for (int k = 0; k < N1; ++k) {
          float a[EPT];
          loadN<EPT>(&S.u.h1[k][erow], a);
          const float4 b = weight4(w2 + k * N2 + c0);
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            acc[0][i] = fmaf(a[i], b.x, acc[0][i]);
            acc[1][i] = fmaf(a[i], b.y, acc[1][i]);
            acc[2][i] = fmaf(a[i], b.z, acc[2][i]);
            acc[3][i] = fmaf(a[i], b.w, acc[3][i]);
          }
        }
        const float4 bias = b2 ? weight4(b2 + c0) : make_float4(0, 0, 0, 0);
        const float4 idt = idt2 ? weight4(idt2 + c0) : make_float4(1, 1, 1, 1);
        const float bs[4] = {bias.x, bias.y, bias.z, bias.w};
        const float is[4] = {idt.x, idt.y, idt.z, idt.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int c = c0 + j;
          float pre[EPT], out[EPT];
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            pre[i] = acc[j][i] + bs[j];
            float v = act_val<ACT>(pre[i]) * is[j];
            if (resnet2) {
              v += S.u.h1[c % N1][erow + i];
            }
            out[i] = v;
          }
          storeN<EPT>(&S.h2[c][erow], out);
          storeN_streaming<EPT>(&pre2_saved[(long)c * e_pad + tile_base + erow],
                                pre);
        }
      }
    }
    __syncthreads();

    // === Step 3. GEMM3 -> g tile (over h1's storage) + spill ===
    // h1's last readers (the GEMM2 residual epilogue) finished at the
    // preceding barrier, so the union overlay is race-free. The residual
    // reads h2[c % N2] (doubling or identity, as in Step 2).
    {
      const int c0 = ty * 8;  // 8 channels per group; groups beyond NG idle
      if (c0 < NG) {
        float acc[8][EPT];  // [8 channels][EPT edges]
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            acc[j][i] = 0.f;
          }
#pragma unroll 4
        for (int k = 0; k < N2; ++k) {
          float a[EPT], b[8];  // a: EPT edges; b: 8 channel weights
          loadN<EPT>(&S.h2[k][erow], a);
          const float4 b0 = weight4(w3 + k * NG + c0);
          const float4 b1 = weight4(w3 + k * NG + c0 + 4);
          b[0] = b0.x;
          b[1] = b0.y;
          b[2] = b0.z;
          b[3] = b0.w;
          b[4] = b1.x;
          b[5] = b1.y;
          b[6] = b1.z;
          b[7] = b1.w;
#pragma unroll
          for (int j = 0; j < 8; ++j)  // 8 channels
#pragma unroll
            for (int i = 0; i < EPT; ++i) {  // EPT edges
              acc[j][i] = fmaf(a[i], b[j], acc[j][i]);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {  // 8 channels
          const int c = c0 + j;
          const float bs = b3 ? __ldg(b3 + c) : 0.f;
          const float is = idt3 ? __ldg(idt3 + c) : 1.f;
          float out[EPT], pre[EPT];
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            pre[i] = acc[j][i] + bs;
            float v = act_val<ACT>(pre[i]) * is;
            if (resnet3) {
              v += S.h2[c % N2][erow + i];
            }
            out[i] = v;
          }
          // tanh recovers act' from the saved output (1 - a^2); silu needs
          // the pre-activation, so the spill slot holds pre3 instead. The
          // spill precedes the strip gate: the backward reconstructs the
          // ungated g and regathers the gate.
          if constexpr (ACT == 0) {
            storeN_streaming<EPT>(&g_saved[(long)c * e_pad + tile_base + erow],
                                  out);
          } else {
            storeN_streaming<EPT>(&g_saved[(long)c * e_pad + tile_base + erow],
                                  pre);
          }
          // Strip mode: the walk consumes the gated gg = g * (1 + gate_eff).
          if (strip) {
#pragma unroll
            for (int i = 0; i < EPT; ++i) {
              const int e = erow + i;
              out[i] *= 1.f + gate_factor(gate_table, NG, S.T.pair_idx[e], c,
                                          S.T.sw[e], smooth);
            }
          }
          storeN<EPT>(&S.u.g[c][erow], out);
        }
      }
    }
    __syncthreads();

    // === Step 4. Moment walk over CSR runs ===
    // kThreads / NG threads share one channel and split each run's row
    // range; one g read feeds all four moment components.
    {
      constexpr int kSlices = kThreads / NG;  // 2 (NG = 128) or 4 (NG = 64)
      const int c = tid % NG;
      const int slice = tid / NG;
      const float* gcol = &S.u.g[c][0];
      const int n_runs = S.T.n_runs;
      for (int run = 0; run < n_runs; ++run) {
        const int node = S.T.run_node[run];
        const int rb = S.T.run_begin[run], re = S.T.run_begin[run + 1];
        const int span = re - rb;
        const int beg = rb + span * slice / kSlices;
        const int end = rb + span * (slice + 1) / kSlices;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int r = beg; r < end; ++r) {
          const float4 rv = *reinterpret_cast<const float4*>(&S.T.rr[r][0]);
          const float gv = gcol[r];
          a0 = fmaf(rv.x, gv, a0);
          a1 = fmaf(rv.y, gv, a1);
          a2 = fmaf(rv.z, gv, a2);
          a3 = fmaf(rv.w, gv, a3);
        }
        if (node >= 0) {
          const long base = ((long)node * 4) * NG + c;
          if (a0 != 0.f) {
            atomicAdd(&gr[base + 0 * NG], a0);
          }
          if (a1 != 0.f) {
            atomicAdd(&gr[base + 1 * NG], a1);
          }
          if (a2 != 0.f) {
            atomicAdd(&gr[base + 2 * NG], a2);
          }
          if (a3 != 0.f) {
            atomicAdd(&gr[base + 3 * NG], a3);
          }
        }
      }
    }
    __syncthreads();
  }
}

// ======================================================================
// Backward kernel.
//
// Consumes the saved pre2 / g spills instead of recomputing the embedding.
// Stages per tile:
//   0. environment matrix + run scan; stage the tile's first RMAX distinct
//      dgr rows in shared memory
//   1. h2 tile reconstructed as act(pre2) * idt2 (+ residual re-derived from
//      the pair table)
//   2. NG / N2 passes over N2-channel blocks: per (4c x 8e) micro-tile read
//      the saved g, form
//        dgv    = sum_k rr[k] * dgr[dst, k, c]   (moment backward)
//        dpre3  = dgs * idt3 * act'(pre3)        -> block tile x_t
//        drr[k] += gg * dgr[dst, k, c]           (environment backward)
//        dh2[c mod N2] += dgs[c]                 (layer-3 residual fold)
//      with dgs the (strip-gated) g gradient. The N2-wide blocking makes
//      the fold thread-local for BOTH residual shapes: block b covers
//      channels [b * N2, (b + 1) * N2), so c mod N2 always lands on the
//      owning thread's dh2 registers (doubling folds dgs[c] + dgs[c + N2]
//      over two blocks; identity folds dgs[c] over the single block). Then
//      the dgrad3 block-pass dh2 += dpre3[block] @ W3[:, block]^T. When all
//      eight edges of a fragment share one CSR run (the common case) the
//      four dgr scalars per channel are read once and amortized across the
//      fragment.
//   3. dpre2 = dh2 * act'(pre2) * idt2 -> x_t rows [0, N2); raw dh2 parks in
//      the (dead) h2 tile for the layer-2 residual fold
//   4. dgrad2 -> dh1; dpre1 through the pair table; d(radial) reduced into
//      shared memory
//   5. analytic environment backward -> d_edge_vec
// ======================================================================
template <int N1, int N2, int NG, int TILE>
struct BwdSmem {
  static constexpr int kResidentRuns = 4;
  static constexpr int STRIDE = TILE + 4;
  float h2[N2][STRIDE];         // act(pre2) tile; raw dh2 after stage 3
  float x_t[N2][STRIDE];        // dpre3 block tile; dpre2 rows [0, N2)
  float drr_banks[8][4][TILE];  // per-warp env-gradient banks
  float d_radial[TILE];
  float d_sw[TILE];  // strip gate: dE/d(sw) through gate * sw
  float dgr_rows[kResidentRuns][4 * NG];
  int run_slot[TILE];  // resident dgr slot per row (-1: global read)
  EdgeTablesT<TILE> T;
};

template <int N1, int N2, int NG, int ACT, int STRIP, int EPT, int PIPE>
__global__ __launch_bounds__(kThreads, 2) void dpa1_graph_backward_kernel(
    long n_edge,
    int ntypes,
    int one_side,
    int resnet2,
    int resnet3,
    int smooth,
    float rcut,
    float rcut_smth,
    float protection,
    float inv_nnei,
    const float* __restrict__ edge_vec,
    const long* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const long* __restrict__ atype,
    const float* __restrict__ davg,
    const float* __restrict__ inv_dstd,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    const float* __restrict__ idt1,
    const float* __restrict__ idt2,
    const float* __restrict__ w3,
    const float* __restrict__ b3,
    const float* __restrict__ idt3,
    const float* __restrict__ pair_table,
    const float* __restrict__ gate_table,
    const int* __restrict__ order,
    const float* __restrict__ dgr,
    const float* __restrict__ pre2_saved,
    const float* __restrict__ g_saved,
    float* __restrict__ d_edge_vec,
    long e_pad) {
  constexpr int kBlocks = NG / N2;  // channel blocks: 1 (identity) or 2
  constexpr int TILE = 16 * EPT;    // edges per tile (16 edge-lanes x EPT)
  constexpr int kResidentRuns = BwdSmem<N1, N2, NG, TILE>::kResidentRuns;
  extern __shared__ char smem_raw[];
  auto& S = *reinterpret_cast<BwdSmem<N1, N2, NG, TILE>*>(smem_raw);
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;
  const int erow = tx * EPT;
  constexpr int strip = STRIP;

  const long n_tiles = ceil_div(n_edge, TILE);
  for (long tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    const long tile_base = tile * TILE;
    const int rows = (int)min((long)TILE, n_edge - tile_base);

    stage_tile<TILE>(tid, tile_base, rows, n_edge, ntypes, one_side, rcut,
                     rcut_smth, protection, inv_nnei, edge_vec, edge_index,
                     edge_mask, atype, davg, inv_dstd, order, S.T);
    if (tid < TILE) {
      S.run_slot[tid] = S.T.run_of[tid] < kResidentRuns ? S.T.run_of[tid] : -1;
      S.d_radial[tid] = 0.f;
      S.d_sw[tid] = 0.f;
    }
    for (int t = tid; t < 8 * 4 * TILE; t += kThreads) {
      (&S.drr_banks[0][0][0])[t] = 0.f;
    }
    {
      const int resident = min(S.T.n_runs, kResidentRuns);
      for (int s = 0; s < resident; ++s) {
        const long d = S.T.run_node[s];
        if (d < 0) {
          continue;
        }
        for (int t = tid; t < 4 * NG; t += kThreads) {
          S.dgr_rows[s][t] = __ldg(dgr + d * 4 * NG + t);
        }
      }
    }
    // The staged dgr rows, run slots and cleared accumulator banks are
    // written cooperatively across warps; the moment stage below reads them
    // from arbitrary warps.
    __syncthreads();

    // === Step 1. h2 tile from the saved pre2 ===
    // The layer-2 doubling residual re-derives h1 per (channel, edge) from
    // the pair table (one FMA and one activation), which is cheaper than a
    // second saved tensor.
    {
      const int c0 = ty * 4;
      if (c0 < N2) {
        const float4 idt = idt2 ? weight4(idt2 + c0) : make_float4(1, 1, 1, 1);
        const float is[4] = {idt.x, idt.y, idt.z, idt.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int c = c0 + j;
          float pre[EPT], hv[EPT];
          loadN_streaming<EPT>(&pre2_saved[(long)c * e_pad + tile_base + erow],
                               pre);
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            hv[i] = act_val<ACT>(pre[i]) * is[j];
          }
          if (resnet2) {
            const int cm = c % N1;
#pragma unroll
            for (int i = 0; i < EPT; ++i) {
              const int e = erow + i;
              const long row = strip ? 0 : (long)S.T.pair_idx[e];
              const float it1 = idt1 ? __ldg(idt1 + cm) : 1.f;
              const float pre1 = fmaf(S.T.radial[e], __ldg(w1 + cm),
                                      __ldg(pair_table + row * N1 + cm));
              hv[i] += act_val<ACT>(pre1) * it1;
            }
          }
          storeN<EPT>(&S.h2[c][erow], hv);
        }
      }
    }

    // Per-edge moment weights, hoisted to registers once per tile (the dgv /
    // drr / residual-fold loops otherwise re-read the float4 per channel).
    float rr_frag[EPT][4];
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const float4 rv = *reinterpret_cast<const float4*>(&S.T.rr[erow + i][0]);
      rr_frag[i][0] = rv.x;
      rr_frag[i][1] = rv.y;
      rr_frag[i][2] = rv.z;
      rr_frag[i][3] = rv.w;
    }
#define RR_(i, k) (rr_frag[i][k])

    // === Step 2. N2-channel blocks: moment backward + dgrad3 ===
    // dh2 (4 channels x EPT edges in registers) accumulates the layer-3
    // residual fold and the dgrad3 GEMM across the blocks.
    float dh2[4][EPT];
    const int kh0 = ty * 4;
#pragma unroll
    for (int block = 0; block < kBlocks; ++block) {
      if (block == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            dh2[j][i] = 0.f;
          }
      } else {
        __syncthreads();  // x_t consumed by the previous dgrad3
      }
      {
        const int c0 = ty * 4;  // 4 channels per group; groups beyond N2 idle
        if (c0 < N2) {
          const int cg = block * N2 + c0;
          const int slot0 = S.run_slot[erow];
          const bool uniform_run =
              S.T.run_of[erow] == S.T.run_of[erow + EPT - 1] && slot0 >= 0;
          float dr0[EPT], dr1[EPT], dr2[EPT], dr3[EPT], dsw[EPT];
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            dr0[i] = dr1[i] = dr2[i] = dr3[i] = dsw[i] = 0.f;
          }
          // Optional prefetch (PIPE): issue the group's four channel fragments
          // of the g / pre3 spill together so their (L2-missing) global
          // latency overlaps, then consume from registers. The narrow EPT
          // fragment leaves the register headroom this needs. Beneficial for
          // wide NG (large spill, e.g. the doubling stacks); for narrow NG it
          // adds registers without hiding enough latency, so it is dispatched
          // per shape.
          float gv4[4][EPT];
          if constexpr (PIPE) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              loadN_streaming<EPT>(
                  &g_saved[(long)(cg + j) * e_pad + tile_base + erow], gv4[j]);
            }
          }
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int c = cg + j;
            const float is = idt3 ? __ldg(idt3 + c) : 1.f;
            const float inv_is = 1.f / is;
            float gv[EPT], dp[EPT], dgv[EPT];
            if constexpr (PIPE) {
#pragma unroll
              for (int i = 0; i < EPT; ++i) {
                gv[i] = gv4[j][i];
              }
            } else {
              loadN_streaming<EPT>(&g_saved[(long)c * e_pad + tile_base + erow],
                                   gv);
            }
            if (uniform_run) {
              const float* row = S.dgr_rows[slot0];
              const float d0 = row[0 * NG + c], d1 = row[1 * NG + c];
              const float d2 = row[2 * NG + c], d3 = row[3 * NG + c];
#pragma unroll
              for (int i = 0; i < EPT; ++i) {
                dgv[i] = RR_(i, 0) * d0 + RR_(i, 1) * d1 + RR_(i, 2) * d2 +
                         RR_(i, 3) * d3;
              }
#pragma unroll
              for (int i = 0; i < EPT; ++i) {
                float gval, aprime;
                if constexpr (ACT == 0) {
                  gval = gv[i];
                  float raw = gval;
                  if (resnet3) {
                    raw -= S.h2[c % N2][erow + i];
                  }
                  const float a = raw * inv_is;
                  aprime = 1.f - a * a;
                } else {
                  const float z = gv[i];  // silu spill holds pre3
                  const float s =
                      0.5f *
                      (1.f +
                       tanhf(0.5f * z));  // sigmoid via tanh; matches act_val
                  aprime = s * (1.f + z * (1.f - s));
                  gval = z * s * is;
                  if (resnet3) {
                    gval += S.h2[c % N2][erow + i];
                  }
                }
                // Strip gate chain: gg = g * (1 + gate_eff) feeds the moment,
                // so drr uses gg, the g-path gradient scales by (1 + gate_eff)
                // and (smooth) the raw gate contributes dE/d(sw).
                float gg = gval;
                if (strip) {
                  const int e = erow + i;
                  const float gate =
                      __ldg(gate_table + (long)S.T.pair_idx[e] * NG + c);
                  const float geff = smooth ? gate * S.T.sw[e] : gate;
                  gg = gval * (1.f + geff);
                  if (smooth) {
                    dsw[i] = fmaf(dgv[i] * gval, gate, dsw[i]);
                  }
                  dgv[i] *= 1.f + geff;
                }
                dp[i] = dgv[i] * is * aprime;
                dr0[i] = fmaf(gg, d0, dr0[i]);
                dr1[i] = fmaf(gg, d1, dr1[i]);
                dr2[i] = fmaf(gg, d2, dr2[i]);
                dr3[i] = fmaf(gg, d3, dr3[i]);
              }
            } else {
#pragma unroll
              for (int i = 0; i < EPT; ++i) {
                const int e = erow + i;
                const int slot = S.run_slot[e];
                const float* row =
                    slot >= 0 ? S.dgr_rows[slot]
                              : dgr + (long)max(S.T.dst[e], 0) * 4 * NG;
                const float d0 = row[0 * NG + c], d1 = row[1 * NG + c];
                const float d2 = row[2 * NG + c], d3 = row[3 * NG + c];
                dgv[i] = RR_(i, 0) * d0 + RR_(i, 1) * d1 + RR_(i, 2) * d2 +
                         RR_(i, 3) * d3;
                float gval, aprime;
                if constexpr (ACT == 0) {
                  gval = gv[i];
                  float raw = gval;
                  if (resnet3) {
                    raw -= S.h2[c % N2][e];
                  }
                  const float a = raw * inv_is;
                  aprime = 1.f - a * a;
                } else {
                  const float z = gv[i];
                  const float s =
                      0.5f *
                      (1.f +
                       tanhf(0.5f * z));  // sigmoid via tanh; matches act_val
                  aprime = s * (1.f + z * (1.f - s));
                  gval = z * s * is;
                  if (resnet3) {
                    gval += S.h2[c % N2][e];
                  }
                }
                float gg = gval;
                if (strip) {
                  const float gate =
                      __ldg(gate_table + (long)S.T.pair_idx[e] * NG + c);
                  const float geff = smooth ? gate * S.T.sw[e] : gate;
                  gg = gval * (1.f + geff);
                  if (smooth) {
                    dsw[i] = fmaf(dgv[i] * gval, gate, dsw[i]);
                  }
                  dgv[i] *= 1.f + geff;
                }
                dp[i] = dgv[i] * is * aprime;
                dr0[i] = fmaf(gg, d0, dr0[i]);
                dr1[i] = fmaf(gg, d1, dr1[i]);
                dr2[i] = fmaf(gg, d2, dr2[i]);
                dr3[i] = fmaf(gg, d3, dr3[i]);
              }
            }
            // Layer-3 residual fold: dh2[c mod N2] accumulates dgs over the
            // blocks. This thread's fold channels coincide with its dgv
            // channels (kh0 == c0), so block b contributes dgs[b * N2 + c]
            // -- the doubling shape folds two terms, identity folds one.
            if (resnet3) {
#pragma unroll
              for (int i = 0; i < EPT; ++i) {
                dh2[j][i] += dgv[i];
              }
            }
            storeN<EPT>(&S.x_t[c0 + j][erow], dp);
          }
          // Environment-gradient partials: fold the two channel-group
          // halves of each warp (lanes l and l + 16 share the edge
          // fragment), then accumulate race-free into this warp's bank.
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            dr0[i] += __shfl_xor_sync(0xffffffffu, dr0[i], 16);
            dr1[i] += __shfl_xor_sync(0xffffffffu, dr1[i], 16);
            dr2[i] += __shfl_xor_sync(0xffffffffu, dr2[i], 16);
            dr3[i] += __shfl_xor_sync(0xffffffffu, dr3[i], 16);
            dsw[i] += __shfl_xor_sync(0xffffffffu, dsw[i], 16);
          }
          if ((ty & 1) == 0) {
            const int w = tid >> 5;
            float* b0 = &S.drr_banks[w][0][erow];
            float* b1 = &S.drr_banks[w][1][erow];
            float* b2v = &S.drr_banks[w][2][erow];
            float* b3v = &S.drr_banks[w][3][erow];
#pragma unroll
            for (int i = 0; i < EPT; ++i) {
              b0[i] += dr0[i];
              b1[i] += dr1[i];
              b2v[i] += dr2[i];
              b3v[i] += dr3[i];
            }
            if (strip && smooth) {
#pragma unroll
              for (int i = 0; i < EPT; ++i) {
                if (dsw[i] != 0.f) {
                  atomicAdd(&S.d_sw[erow + i], dsw[i]);
                }
              }
            }
          }
        }
      }
      __syncthreads();

      // dgrad3 block-pass: dh2 += dpre3[block] @ W3[:, block]^T.
      // Four-channel steps keep the weight reads as row-contiguous float4
      // broadcasts.
      if (kh0 < N2) {
#pragma unroll 2
        for (int c = 0; c < N2; c += 4) {
          float a0[EPT], a1[EPT], a2[EPT], a3[EPT];
          loadN<EPT>(&S.x_t[c + 0][erow], a0);
          loadN<EPT>(&S.x_t[c + 1][erow], a1);
          loadN<EPT>(&S.x_t[c + 2][erow], a2);
          loadN<EPT>(&S.x_t[c + 3][erow], a3);
          const int cg = block * N2 + c;
          const float4 w0 = weight4(w3 + (kh0 + 0) * NG + cg);
          const float4 w1v = weight4(w3 + (kh0 + 1) * NG + cg);
          const float4 w2v = weight4(w3 + (kh0 + 2) * NG + cg);
          const float4 w3v = weight4(w3 + (kh0 + 3) * NG + cg);
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            dh2[0][i] +=
                a0[i] * w0.x + a1[i] * w0.y + a2[i] * w0.z + a3[i] * w0.w;
            dh2[1][i] +=
                a0[i] * w1v.x + a1[i] * w1v.y + a2[i] * w1v.z + a3[i] * w1v.w;
            dh2[2][i] +=
                a0[i] * w2v.x + a1[i] * w2v.y + a2[i] * w2v.z + a3[i] * w2v.w;
            dh2[3][i] +=
                a0[i] * w3v.x + a1[i] * w3v.y + a2[i] * w3v.z + a3[i] * w3v.w;
          }
        }
      }
    }

    // === Step 3. dpre2 -> x_t rows [0, N2); raw dh2 -> h2 rows ===
    // h2's activation values are dead once every block consumed them, so
    // the tile stores the raw dh2 for the layer-2 residual fold.
    __syncthreads();
    if (kh0 < N2) {
      const float4 idt = idt2 ? weight4(idt2 + kh0) : make_float4(1, 1, 1, 1);
      const float is[4] = {idt.x, idt.y, idt.z, idt.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = kh0 + j;
        float pre[EPT], dp[EPT];
        loadN_streaming<EPT>(&pre2_saved[(long)k * e_pad + tile_base + erow],
                             pre);
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
          dp[i] = dh2[j][i] * act_vg<ACT>(pre[i]).y * is[j];
        }
        storeN<EPT>(&S.x_t[k][erow], dp);
        storeN<EPT>(&S.h2[k][erow], dh2[j]);
      }
    }
    __syncthreads();

    // === Step 4. dgrad2 -> dh1 -> dpre1 -> d(radial) ===
    // Each channel group owns kC1 = max(N1 / 16, 2) consecutive dh1
    // channels so the 16 groups cover any N1 up to 64.
    {
      constexpr int kC1 = N1 / 16 > 2 ? N1 / 16 : 2;
      const int c0 = ty * kC1;
      if (c0 < N1) {
        float acc[kC1][EPT];
#pragma unroll
        for (int j = 0; j < kC1; ++j)
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            acc[j][i] = 0.f;
          }
#pragma unroll 2
        for (int k = 0; k < N2; k += 4) {
          float a0[EPT], a1[EPT], a2[EPT], a3[EPT];
          loadN<EPT>(&S.x_t[k + 0][erow], a0);
          loadN<EPT>(&S.x_t[k + 1][erow], a1);
          loadN<EPT>(&S.x_t[k + 2][erow], a2);
          loadN<EPT>(&S.x_t[k + 3][erow], a3);
#pragma unroll
          for (int j = 0; j < kC1; ++j) {
            const float4 wv = weight4(w2 + (c0 + j) * N2 + k);
#pragma unroll
            for (int i = 0; i < EPT; ++i) {
              acc[j][i] +=
                  a0[i] * wv.x + a1[i] * wv.y + a2[i] * wv.z + a3[i] * wv.w;
            }
          }
        }
        float ds[EPT];
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
          ds[i] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < kC1; ++j) {
          const int c = c0 + j;
          const float ws = __ldg(w1 + c);
          const float it1 = idt1 ? __ldg(idt1 + c) : 1.f;
#pragma unroll
          for (int i = 0; i < EPT; ++i) {
            const int e = erow + i;
            float dh1 = acc[j][i];
            // Layer-2 residual fold over the raw dh2 parked in the h2 tile:
            // one term for the identity shape, two for the doubling shape.
            if (resnet2) {
#pragma unroll
              for (int b = 0; b < N2 / N1; ++b) {
                dh1 += S.h2[c + b * N1][e];
              }
            }
            const long row = strip ? 0 : (long)S.T.pair_idx[e];
            const float pre1 =
                fmaf(S.T.radial[e], ws, __ldg(pair_table + row * N1 + c));
            const float a1d = act_vg<ACT>(pre1).y * it1;
            ds[i] = fmaf(dh1 * a1d, ws, ds[i]);
          }
        }
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
          atomicAdd(&S.d_radial[erow + i], ds[i]);
        }
      }
    }
    __syncthreads();

    // === Step 5. Analytic environment backward -> d_edge_vec ===
    // With v = (sw/q, sw*x/q^2, ...), the chain through sw(|r|), 1/q and the
    // per-type normalization is evaluated in closed form. The strip gate
    // contributes an additional dE/d(sw) term along the unit vector.
    if (tid < rows) {
      const int e = order[tile_base + tid];
      if (!edge_mask[e]) {
        d_edge_vec[(long)e * 3 + 0] = 0.f;
        d_edge_vec[(long)e * 3 + 1] = 0.f;
        d_edge_vec[(long)e * 3 + 2] = 0.f;
      } else {
        const float x = edge_vec[(long)e * 3 + 0];
        const float y = edge_vec[(long)e * 3 + 1];
        const float z = edge_vec[(long)e * 3 + 2];
        const float len = sqrtf(x * x + y * y + z * z);
        const float q = len + protection;
        const float sw = switch_val(len, rcut_smth, rcut);
        const float dsw = switch_deriv(len, rcut_smth, rcut);
        const float* isd = inv_dstd + (long)atype[S.T.dst[tid]] * 4;
        float dr[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int w = 0; w < 8; ++w)
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            dr[k] += S.drr_banks[w][k][tid];
          }
        const float g0 = (dr[0] * inv_nnei + S.d_radial[tid]) * isd[0];
        const float gx = dr[1] * inv_nnei * isd[1];
        const float gy = dr[2] * inv_nnei * isd[2];
        const float gz = dr[3] * inv_nnei * isd[3];
        const float inv_len = len > 0.f ? 1.f / len : 0.f;
        const float rq = 1.f / q;
        const float gdot = gx * x + gy * y + gz * z;
        const float coef =
            (g0 * rq * (dsw - sw * rq) +
             gdot * rq * rq * (dsw - 2.f * sw * rq) + S.d_sw[tid] * dsw) *
            inv_len;
        const float s2 = sw * rq * rq;
        d_edge_vec[(long)e * 3 + 0] = coef * x + s2 * gx;
        d_edge_vec[(long)e * 3 + 1] = coef * y + s2 * gy;
        d_edge_vec[(long)e * 3 + 2] = coef * z + s2 * gz;
      }
    }
    __syncthreads();
  }
#undef RR_
}

// ======================================================================
// Host-side dispatch
// ======================================================================
struct EmbeddingWidths {
  int n1, n2, ng;
};

// Supported width stacks: each layer keeps (identity residual shape) or
// doubles (concat [x, x] residual shape) the previous width, with N1 a power
// of two in [8, 64], N2 <= 64 and NG <= 128 (the bounds of the 16-group tile
// framework at 256 threads).
EmbeddingWidths check_widths(const torch::Tensor& w1,
                             const torch::Tensor& w2,
                             const torch::Tensor& w3) {
  EmbeddingWidths s{(int)w1.size(1), (int)w2.size(1), (int)w3.size(1)};
  TORCH_CHECK((int)w2.size(0) == s.n1 && (int)w3.size(0) == s.n2,
              "dpa1_graph_descriptor: inconsistent embedding widths");
  TORCH_CHECK(
      (s.n2 == s.n1 || s.n2 == 2 * s.n1) && (s.ng == s.n2 || s.ng == 2 * s.n2),
      "dpa1_graph_descriptor: each layer width must equal or double the "
      "previous one");
  TORCH_CHECK((s.n1 == 8 || s.n1 == 16 || s.n1 == 32 || s.n1 == 64) &&
                  s.n2 <= 64 && s.ng <= 128,
              "dpa1_graph_descriptor: unsupported widths (", s.n1, ", ", s.n2,
              ", ", s.ng, ")");
  return s;
}

// CSR ordering + layer-1 pair table, shared by the forward entry point (the
// backward receives both as saved tensors). Strip mode has no type term in
// layer 1: the table degenerates to the single bias row (zeros without a
// bias) and the type embedding enters through the gate table instead.
std::tuple<torch::Tensor, torch::Tensor> build_order_and_pair_table(
    const torch::Tensor& edge_index,
    const torch::Tensor& edge_mask,
    long n_edge,
    long n_node,
    int ntypes,
    int tebd_dim,
    int n1,
    int one_side,
    int strip,
    const torch::Tensor& type_embedding,
    const torch::Tensor& w1,
    const torch::Tensor& b1,
    cudaStream_t stream) {
  auto order = build_edge_order(edge_index, edge_mask, n_edge, n_node, stream);

  if (strip) {
    auto pair_table = b1.numel()
                          ? b1.reshape({1, n1}).contiguous()
                          : torch::zeros({1, n1}, type_embedding.options());
    return {order, pair_table};
  }
  const long n_pairs = one_side ? ntypes : (long)ntypes * ntypes;
  auto pair_table = torch::empty({n_pairs, n1}, type_embedding.options());
  pair_table_kernel<<<ceil_div(n_pairs * n1, 256), 256, 0, stream>>>(
      (int)n_pairs, ntypes, tebd_dim, n1, one_side,
      type_embedding.data_ptr<float>(), w1.data_ptr<float>(), optional_ptr(b1),
      pair_table.data_ptr<float>());
  DPA1_CHECK_LAUNCH("dpa1_graph_descriptor pair table");
  return {order, pair_table};
}

// Bundles the tensor arguments shared verbatim by the forward and backward
// launches, so the width / activation dispatch stays a one-line macro.
struct LaunchArgs {
  long n_edge;
  int ntypes, one_side, resnet2, resnet3, smooth;
  float rcut, rcut_smth, protection, inv_nnei;
  const torch::Tensor &edge_vec, &edge_index, &edge_mask, &atype;
  const torch::Tensor &davg, &inv_dstd;
  const torch::Tensor &w1, &w2, &b2, &idt1, &idt2, &w3, &b3, &idt3;
  const torch::Tensor &pair_table, &gate_table, &order;
  cudaStream_t stream;
};

template <int N1, int N2, int NG, int ACT, int STRIP, int EPT>
void launch_forward(const LaunchArgs& a,
                    torch::Tensor& gr,
                    torch::Tensor& pre2_saved,
                    torch::Tensor& g_saved) {
  constexpr int TILE = 16 * EPT;
  auto kernel = dpa1_graph_forward_kernel<N1, N2, NG, ACT, STRIP, EPT>;
  const size_t smem = sizeof(FwdSmem<N1, N2, NG, TILE>);
  const cudaError_t attribute_error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  TORCH_CHECK(
      attribute_error == cudaSuccess,
      "dpa1_graph_descriptor forward: cannot configure ", smem,
      " bytes of dynamic shared memory: ", cudaGetErrorString(attribute_error));
  kernel<<<persistent_grid(a.n_edge, 2, TILE), kThreads, smem, a.stream>>>(
      a.n_edge, a.ntypes, a.one_side, a.resnet2, a.resnet3, a.smooth, a.rcut,
      a.rcut_smth, a.protection, a.inv_nnei, a.edge_vec.data_ptr<float>(),
      a.edge_index.data_ptr<long>(), a.edge_mask.data_ptr<bool>(),
      a.atype.data_ptr<long>(), a.davg.data_ptr<float>(),
      a.inv_dstd.data_ptr<float>(), a.w1.data_ptr<float>(),
      a.w2.data_ptr<float>(), optional_ptr(a.b2), optional_ptr(a.idt1),
      optional_ptr(a.idt2), a.w3.data_ptr<float>(), optional_ptr(a.b3),
      optional_ptr(a.idt3), a.pair_table.data_ptr<float>(),
      optional_ptr(a.gate_table), a.order.data_ptr<int>(), gr.data_ptr<float>(),
      pre2_saved.data_ptr<float>(), g_saved.data_ptr<float>(),
      pre2_saved.size(1));
  DPA1_CHECK_LAUNCH("dpa1_graph_descriptor forward");
}

template <int N1, int N2, int NG, int ACT, int STRIP>
void launch_forward_portable(const LaunchArgs& a,
                             torch::Tensor& gr,
                             torch::Tensor& pre2_saved,
                             torch::Tensor& g_saved) {
  constexpr int kWideEdgesPerThread = 8;
  constexpr int kNarrowEdgesPerThread = 4;
  constexpr int kWideTile = 16 * kWideEdgesPerThread;
  constexpr size_t kWideSharedMemory = sizeof(FwdSmem<N1, N2, NG, kWideTile>);
  constexpr size_t kPortableSharedMemoryFloor = 48 * 1024;
  if constexpr (kWideSharedMemory <= kPortableSharedMemoryFloor) {
    launch_forward<N1, N2, NG, ACT, STRIP, kWideEdgesPerThread>(
        a, gr, pre2_saved, g_saved);
    return;
  }
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const size_t device_limit = std::max(properties->sharedMemPerBlock,
                                       properties->sharedMemPerBlockOptin);
  if (kWideSharedMemory <= device_limit) {
    launch_forward<N1, N2, NG, ACT, STRIP, kWideEdgesPerThread>(
        a, gr, pre2_saved, g_saved);
  } else {
    constexpr int kNarrowTile = 16 * kNarrowEdgesPerThread;
    constexpr size_t kNarrowSharedMemory =
        sizeof(FwdSmem<N1, N2, NG, kNarrowTile>);
    TORCH_CHECK(kNarrowSharedMemory <= device_limit,
                "dpa1_graph_descriptor forward requires ", kNarrowSharedMemory,
                " bytes of dynamic shared memory, but the device supports ",
                device_limit);
    launch_forward<N1, N2, NG, ACT, STRIP, kNarrowEdgesPerThread>(
        a, gr, pre2_saved, g_saved);
  }
}

template <int N1, int N2, int NG, int ACT, int STRIP, int EPT, int PIPE>
void launch_backward(const LaunchArgs& a,
                     const torch::Tensor& dgr,
                     const torch::Tensor& pre2_saved,
                     const torch::Tensor& g_saved,
                     torch::Tensor& d_edge_vec) {
  constexpr int TILE = 16 * EPT;
  auto kernel = dpa1_graph_backward_kernel<N1, N2, NG, ACT, STRIP, EPT, PIPE>;
  const size_t smem = sizeof(BwdSmem<N1, N2, NG, TILE>);
  const cudaError_t attribute_error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  TORCH_CHECK(
      attribute_error == cudaSuccess,
      "dpa1_graph_descriptor backward: cannot configure ", smem,
      " bytes of dynamic shared memory: ", cudaGetErrorString(attribute_error));
  kernel<<<persistent_grid(a.n_edge, 2, TILE), kThreads, smem, a.stream>>>(
      a.n_edge, a.ntypes, a.one_side, a.resnet2, a.resnet3, a.smooth, a.rcut,
      a.rcut_smth, a.protection, a.inv_nnei, a.edge_vec.data_ptr<float>(),
      a.edge_index.data_ptr<long>(), a.edge_mask.data_ptr<bool>(),
      a.atype.data_ptr<long>(), a.davg.data_ptr<float>(),
      a.inv_dstd.data_ptr<float>(), a.w1.data_ptr<float>(),
      a.w2.data_ptr<float>(), optional_ptr(a.b2), optional_ptr(a.idt1),
      optional_ptr(a.idt2), a.w3.data_ptr<float>(), optional_ptr(a.b3),
      optional_ptr(a.idt3), a.pair_table.data_ptr<float>(),
      optional_ptr(a.gate_table), a.order.data_ptr<int>(),
      dgr.data_ptr<float>(), pre2_saved.data_ptr<float>(),
      g_saved.data_ptr<float>(), d_edge_vec.data_ptr<float>(),
      pre2_saved.size(1));
  DPA1_CHECK_LAUNCH("dpa1_graph_descriptor backward");
}

// The backward uses four edges per thread once N1 reaches 16 to bound the
// register footprint of widening stacks. The N1 == 8 specialisation keeps
// eight edges per thread to avoid doubling the tile count. NG >= 128
// prefetches saved g / pre3 rows into the available registers.
constexpr int backward_edges_per_thread(int n1) { return n1 >= 16 ? 4 : 8; }
constexpr int backward_prefetch(int ng) { return ng >= 128 ? 1 : 0; }

}  // namespace

// Width / activation instantiation table shared by the two entry points:
// every "equal or doubling" stack over N1 in {8, 16, 32, 64} within the
// N2 <= 64, NG <= 128 tile bounds (see check_widths). Each stack expands to a
// runtime branch that resolves the activation and the tebd input mode (strip
// carries the type-pair gate, concat does not) to compile-time template
// arguments; the backward's per-thread edge fragment and spill prefetch follow
// from the width via backward_edges_per_thread / backward_prefetch.
#define DPA1_DISPATCH_ONE(LAUNCH, N1V, N2V, NGV)                       \
  else if (widths.n1 == N1V && widths.n2 == N2V && widths.ng == NGV) { \
    if (strip) {                                                       \
      if (act == 0)                                                    \
        LAUNCH(N1V, N2V, NGV, 0, 1);                                   \
      else                                                             \
        LAUNCH(N1V, N2V, NGV, 1, 1);                                   \
    } else {                                                           \
      if (act == 0)                                                    \
        LAUNCH(N1V, N2V, NGV, 0, 0);                                   \
      else                                                             \
        LAUNCH(N1V, N2V, NGV, 1, 0);                                   \
    }                                                                  \
  }

#define DPA1_DISPATCH_WIDTH_ACT(LAUNCH)                                 \
  do {                                                                  \
    if (false) {                                                        \
    }                                                                   \
    DPA1_DISPATCH_ONE(LAUNCH, 8, 8, 8)                                  \
    DPA1_DISPATCH_ONE(LAUNCH, 8, 8, 16)                                 \
    DPA1_DISPATCH_ONE(LAUNCH, 8, 16, 16)                                \
    DPA1_DISPATCH_ONE(LAUNCH, 8, 16, 32)                                \
    DPA1_DISPATCH_ONE(LAUNCH, 16, 16, 16)                               \
    DPA1_DISPATCH_ONE(LAUNCH, 16, 16, 32)                               \
    DPA1_DISPATCH_ONE(LAUNCH, 16, 32, 32)                               \
    DPA1_DISPATCH_ONE(LAUNCH, 16, 32, 64)                               \
    DPA1_DISPATCH_ONE(LAUNCH, 32, 32, 32)                               \
    DPA1_DISPATCH_ONE(LAUNCH, 32, 32, 64)                               \
    DPA1_DISPATCH_ONE(LAUNCH, 32, 64, 64)                               \
    DPA1_DISPATCH_ONE(LAUNCH, 32, 64, 128)                              \
    DPA1_DISPATCH_ONE(LAUNCH, 64, 64, 64)                               \
    DPA1_DISPATCH_ONE(LAUNCH, 64, 64, 128)                              \
    else {                                                              \
      TORCH_CHECK(false, "dpa1_graph_descriptor: width dispatch miss"); \
    }                                                                   \
  } while (0)

// Forward: (grrg, rot_mat) plus the tensors the backward consumes. See the
// file header for the layout invariants and the applicability gate; the
// Python wrapper (deepmd.kernels.cuda.dpa1.graph_descriptor) documents the
// argument contract. An empty gate_table selects concat mode; a populated
// one ((T or T^2, NG), the strip embedding of the type pairs) selects strip.
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
dpa1_graph_descriptor(torch::Tensor edge_vec,
                      torch::Tensor edge_index,
                      torch::Tensor edge_mask,
                      torch::Tensor atype,
                      torch::Tensor type_embedding,
                      torch::Tensor davg,
                      torch::Tensor dstd,
                      torch::Tensor w1,
                      torch::Tensor b1,
                      torch::Tensor idt1,
                      torch::Tensor w2,
                      torch::Tensor b2,
                      torch::Tensor idt2,
                      torch::Tensor w3,
                      torch::Tensor b3,
                      torch::Tensor idt3,
                      torch::Tensor gate_table,
                      int64_t act,
                      int64_t type_one_side,
                      int64_t concat_tebd,
                      int64_t write_rotation,
                      int64_t smooth,
                      int64_t axis,
                      int64_t resnet2,
                      int64_t resnet3,
                      double rcut,
                      double rcut_smth,
                      double protection,
                      double nnei) {
  const auto widths = check_widths(w1, w2, w3);
  const long n_edge = edge_vec.size(0);
  const long n_node = atype.size(0);
  const int ntypes = (int)type_embedding.size(0);
  const int tebd_dim = (int)type_embedding.size(1);
  const int NG = widths.ng;
  const bool strip = gate_table.numel() > 0;
  TORCH_CHECK(edge_vec.is_contiguous() && edge_index.is_contiguous() &&
                  edge_mask.is_contiguous() && atype.is_contiguous(),
              "dpa1_graph_descriptor: graph tensors must be contiguous");
  TORCH_CHECK(
      (int)w1.size(0) == (strip ? 1 : 1 + (type_one_side ? 1 : 2) * tebd_dim),
      "dpa1_graph_descriptor: w1 rows do not match the tebd mode");
  if (strip) {
    TORCH_CHECK(gate_table.is_contiguous() && (int)gate_table.size(1) == NG &&
                    gate_table.size(0) ==
                        (type_one_side ? ntypes : (long)ntypes * ntypes),
                "dpa1_graph_descriptor: gate_table must be a contiguous "
                "(pairs, ng) strip embedding of the type pairs");
  }
  TORCH_CHECK(edge_mask.scalar_type() == torch::kBool,
              "dpa1_graph_descriptor: edge_mask must be bool");
  TORCH_CHECK(act == 0 || act == 1,
              "dpa1_graph_descriptor: act must be 0 (tanh) or 1 (silu)");
  auto stream = at::cuda::getCurrentCUDAStream();

  auto [order, pair_table] = build_order_and_pair_table(
      edge_index, edge_mask, n_edge, n_node, ntypes, tebd_dim, widths.n1,
      (int)type_one_side, (int)strip, type_embedding, w1, b1, stream);
  auto inv_dstd = torch::reciprocal(dstd).contiguous();

  auto f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(edge_vec.device());
  auto gr = torch::zeros({n_node, 4, NG}, f32);
  // Backward operands, spilled transposed so both directions stream
  // coalesced rows. E_pad rounds up to the tile size so partial-tile float4
  // stores stay in bounds; for silu the g slot holds pre3 (its derivative
  // needs the pre-activation).
  const long e_pad = ceil_div(n_edge, kTileEdges) * kTileEdges;
  auto pre2_saved = torch::empty({(long)widths.n2, e_pad}, f32);
  auto g_saved = torch::empty({(long)NG, e_pad}, f32);
  // The operator computes in fp32 (fp32 weights and tables); the coordinate
  // input, in the model's global precision, enters through a single fp32 cast
  // at the boundary (a no-op when already fp32).
  const auto edge_vec_f = edge_vec.to(torch::kFloat32);
  const LaunchArgs args{n_edge,
                        ntypes,
                        (int)type_one_side,
                        (int)resnet2,
                        (int)resnet3,
                        (int)smooth,
                        (float)rcut,
                        (float)rcut_smth,
                        (float)protection,
                        (float)(1.0 / nnei),
                        edge_vec_f,
                        edge_index,
                        edge_mask,
                        atype,
                        davg,
                        inv_dstd,
                        w1,
                        w2,
                        b2,
                        idt1,
                        idt2,
                        w3,
                        b3,
                        idt3,
                        pair_table,
                        gate_table,
                        order,
                        stream};

#define DPA1_LAUNCH_FWD(N1, N2, NG, ACT, STRIP) \
  launch_forward_portable<N1, N2, NG, ACT, STRIP>(args, gr, pre2_saved, g_saved)
  if (n_edge > 0) {
    DPA1_DISPATCH_WIDTH_ACT(DPA1_LAUNCH_FWD);
  }
#undef DPA1_LAUNCH_FWD

  const int out_dim = NG * (int)axis + (concat_tebd ? tebd_dim : 0);
  auto grrg = torch::empty({n_node, out_dim}, f32);
  auto rot_mat = torch::empty({write_rotation ? n_node : 0, NG, 3}, f32);
  if (n_node > 0) {
    gram_kernel<<<(int)n_node, 128, 4 * NG * sizeof(float), stream>>>(
        (int)n_node, NG, (int)axis, tebd_dim, (int)concat_tebd,
        gr.data_ptr<float>(), type_embedding.data_ptr<float>(),
        atype.data_ptr<long>(), grrg.data_ptr<float>(),
        write_rotation ? rot_mat.data_ptr<float>() : nullptr);
    DPA1_CHECK_LAUNCH("dpa1_graph_descriptor gram");
  }
  return {grrg, rot_mat, gr, order, pair_table, pre2_saved, g_saved};
}

// Backward: (d_grrg, d_rot_mat) and the saved tensors -> d_edge_vec in the
// edge_vec dtype. rot_mat may be unused downstream (the energy fitting reads
// only grrg), in which case autograd passes None for its gradient.
torch::Tensor dpa1_graph_descriptor_backward(
    torch::Tensor d_grrg,
    std::optional<torch::Tensor> d_rot_mat,
    torch::Tensor gr,
    torch::Tensor order,
    torch::Tensor pair_table,
    torch::Tensor pre2_saved,
    torch::Tensor g_saved,
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor atype,
    torch::Tensor davg,
    torch::Tensor dstd,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor idt1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor idt2,
    torch::Tensor w3,
    torch::Tensor b3,
    torch::Tensor idt3,
    torch::Tensor gate_table,
    int64_t act,
    int64_t type_one_side,
    int64_t smooth,
    int64_t axis,
    int64_t resnet2,
    int64_t resnet3,
    double rcut,
    double rcut_smth,
    double protection,
    double nnei) {
  const auto widths = check_widths(w1, w2, w3);
  const long n_edge = edge_vec.size(0);
  const long n_node = atype.size(0);
  const int NG = widths.ng;
  const bool strip = gate_table.numel() > 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(edge_vec.device());

  auto dgr = torch::empty({n_node, 4, NG}, f32);
  auto d_grrg_c = d_grrg.to(torch::kFloat32).contiguous();
  torch::Tensor d_rot_c;
  const float* d_rot_ptr = nullptr;
  if (d_rot_mat.has_value() && d_rot_mat->defined() && d_rot_mat->numel()) {
    d_rot_c = d_rot_mat->to(torch::kFloat32).contiguous();
    d_rot_ptr = d_rot_c.data_ptr<float>();
  }
  if (n_node > 0) {
    const size_t smem = (4 * NG + NG * (int)axis) * sizeof(float);
    gram_backward_kernel<<<(int)n_node, 128, smem, stream>>>(
        (int)n_node, NG, (int)axis, (int)d_grrg_c.size(1),
        d_grrg_c.data_ptr<float>(), d_rot_ptr, gr.data_ptr<float>(),
        dgr.data_ptr<float>());
    DPA1_CHECK_LAUNCH("dpa1_graph_descriptor gram backward");
  }

  auto inv_dstd = torch::reciprocal(dstd).contiguous();
  // fp32 compute: cast the coordinate input in and produce the gradient in
  // fp32; it is cast back to the model's precision at the boundary (see the
  // forward).
  const auto edge_vec_f = edge_vec.to(torch::kFloat32);
  auto d_edge_vec = torch::empty({n_edge, 3}, f32);
  // ntypes is recovered from the type-pair table row count (T or T^2): the
  // layer-1 table in concat mode, the gate table in strip mode (whose
  // layer-1 table has a single row).
  const long n_pairs = strip ? gate_table.size(0) : pair_table.size(0);
  const int ntypes =
      type_one_side ? (int)n_pairs : (int)llround(sqrt((double)n_pairs));
  const LaunchArgs args{n_edge,
                        ntypes,
                        (int)type_one_side,
                        (int)resnet2,
                        (int)resnet3,
                        (int)smooth,
                        (float)rcut,
                        (float)rcut_smth,
                        (float)protection,
                        (float)(1.0 / nnei),
                        edge_vec_f,
                        edge_index,
                        edge_mask,
                        atype,
                        davg,
                        inv_dstd,
                        w1,
                        w2,
                        b2,
                        idt1,
                        idt2,
                        w3,
                        b3,
                        idt3,
                        pair_table,
                        gate_table,
                        order,
                        stream};

#define DPA1_LAUNCH_BWD(N1, N2, NG, ACT, STRIP)                          \
  launch_backward<N1, N2, NG, ACT, STRIP, backward_edges_per_thread(N1), \
                  backward_prefetch(NG)>(args, dgr, pre2_saved, g_saved, \
                                         d_edge_vec)
  if (n_edge > 0) {
    DPA1_DISPATCH_WIDTH_ACT(DPA1_LAUNCH_BWD);
  }
#undef DPA1_LAUNCH_BWD
  // Return the gradient in the coordinate's precision (a no-op when fp32).
  return d_edge_vec.to(edge_vec.scalar_type());
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "dpa1_graph_descriptor(Tensor edge_vec, Tensor edge_index, "
      "Tensor edge_mask, Tensor atype, Tensor type_embedding, Tensor davg, "
      "Tensor dstd, Tensor w1, Tensor b1, Tensor idt1, Tensor w2, Tensor b2, "
      "Tensor idt2, Tensor w3, Tensor b3, Tensor idt3, Tensor gate_table, "
      "int act, int type_one_side, int concat_tebd, int write_rotation, int "
      "smooth, int axis, int resnet2, int resnet3, float rcut, float "
      "rcut_smth, "
      "float protection, float nnei) -> (Tensor grrg, Tensor rot_mat, "
      "Tensor gr, Tensor edge_order, Tensor pair_table, Tensor pre2_saved, "
      "Tensor g_saved)");
  m.impl("dpa1_graph_descriptor", torch::kCUDA, &dpa1_graph_descriptor);
  m.def(
      "dpa1_graph_descriptor_backward(Tensor d_grrg, Tensor? d_rot_mat, "
      "Tensor gr, Tensor edge_order, Tensor pair_table, Tensor pre2_saved, "
      "Tensor g_saved, Tensor edge_vec, Tensor edge_index, Tensor edge_mask, "
      "Tensor atype, Tensor davg, Tensor dstd, Tensor w1, Tensor b1, "
      "Tensor idt1, Tensor w2, Tensor b2, Tensor idt2, Tensor w3, Tensor b3, "
      "Tensor idt3, Tensor gate_table, int act, int type_one_side, "
      "int smooth, int axis, int resnet2, int resnet3, float rcut, "
      "float rcut_smth, float protection, float nnei) -> Tensor");
  m.impl("dpa1_graph_descriptor_backward", torch::kCUDA,
         &dpa1_graph_descriptor_backward);
}
