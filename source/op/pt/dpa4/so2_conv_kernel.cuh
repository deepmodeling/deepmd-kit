// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused SO(2) convolution value path for SeZM / DPA4 inference: kernels.
//
// One operator pair spans the whole per-edge path of one ``SO2Convolution``:
//
//   x_src   = x[src[e]]                                   // (D, Cf)
//   x_local = Wigner_e @ x_src                            // (RED, Cf)
//   u0      = degree_mix(x_local; kc[e], cb)              // (RED, Cf)
//   u       = gated_stack(u0; W0, W1, Gw)                 // n_layers
//   pre[n]  = rescale * sum_{dst[e]=n} alpha[e] * Wigner_e^T @ u
//   out[n]  = pre[n] * head_gate[n]
//
// Fusing them keeps every per-edge intermediate out of device memory. The path
// they replace moves about 3.6 GB of ``(E, ROW)`` activation per convolution at
// the reference shape, which pins it to the DRAM roof at an arithmetic
// intensity of 24 FLOP/byte against a ridge point near 78.
//
// Where the activation lives
// --------------------------
// The activation is register resident, not shared. Every multiply in the stack
// assigns output column ``j * 32 + lane`` to lane ``lane`` at register slot
// ``j``, so the residual, the gate sigmoids, the inverse rotation and the node
// accumulator are all lane-local, and the only cross-lane traffic is the ``k``
// broadcast of the multiply, done with ``__shfl_sync``. That matters twice:
//
//   * Shared memory does not scale with the chunk width, so the widest
//     supported shape fits. Holding ``BE x ROW`` floats in shared memory would
//     need 106 KB at ``lmax = 4, Cf = 64`` and 32 edges, above the 100 KB
//     per-block limit.
//   * Occupancy is not capped by an activation buffer. A shared-memory
//     activation holds this kernel at 8 warps per multiprocessor and makes
//     every tile that improves arithmetic intensity cost a resident block.
//
// Weight traffic
// --------------
// Each block-chunk stages the complete weight set of the convolution through
// shared memory, so the L2 traffic is ``(E / BE) * W`` floats against ``E * W``
// products: an arithmetic intensity of ``BE / 2`` FLOP/byte independent of
// shape. ``BE = WARPS * TM`` therefore trades directly against register
// pressure, which is what the per-shape launch policy balances.
//
// Numerics are IEEE fp32 throughout. There is no reduced-precision path.

#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include "so2_conv.cuh"

namespace dpa4 {

/// Shape constants of one convolution instantiation.
///
/// The ``*B`` counts are widths in units of the 32-column group a lane owns.
template <int L, int CF>
struct ConvShape {
  static constexpr int DIM = packed_dim<L>();
  static constexpr int RED = reduced_dim<L>();
  static constexpr int NW = wigner_run<L>();
  static constexpr int KSZ = degree_kernel_size<L>();
  static constexpr int M0 = (L + 1) * CF;
  static constexpr int M1 = 2 * L * CF;
  static constexpr int ROW = M0 + M1;
  static constexpr int GATE = L * CF;
  static constexpr int NMAX = (M1 > M0) ? M1 : M0;
  static constexpr int CFB = CF / kWarp;
  static constexpr int RB = ROW / kWarp;
  static constexpr int M0B = M0 / kWarp;
  static constexpr int M1B = M1 / kWarp;
  static constexpr int GB = GATE / kWarp;
  /// Per-warp staging footprint: the row set serves the two Wigner outer
  /// products (RED + DIM rows) and the degree-kernel gradient sweep (one row
  /// per compact slot).
  static constexpr int OUTER =
      ((RED + DIM > KSZ) ? RED + DIM : KSZ) * kStageStride;

  static_assert(CF % kWarp == 0, "focus width must be a multiple of the warp");
};

/// Shared-memory plan of one chunk.
///
/// Only the per-edge geometry lives here permanently. ``scratch`` is the weight
/// panel of a block multiply, the per-warp staging tile of the two Wigner outer
/// products, and the cross-warp reduction buffer of the node store, in that
/// order of appearance; the three uses never overlap in time.
template <int L, int CF, typename T>
struct ConvSmem {
  using S = ConvShape<L, CF>;
  // Two panels: the asynchronous copy of the next fills one buffer while
  // the block computes from the other.
  static constexpr int PANEL = 2 * T::PK * S::NMAX;
  static constexpr int OUTER = T::WARPS * S::OUTER;
  static constexpr int SCRATCH_A = PANEL > OUTER ? PANEL : OUTER;
  static constexpr int SCRATCH = SCRATCH_A > T::NT ? SCRATCH_A : T::NT;

  float* scratch;
  float* astage;   // (WARPS, TM, 32) per-warp column-group broadcast stage
  float* wig;      // (BE, NW) packed block-diagonal Wigner run
  float* kc;       // (BE, kc_len) compact degree kernel
  float* alpha;    // (BE, H)
  float* rescale;  // (DIM,)
  float* softm;    // (H,) running softmax maximum, forward only
  float* softd;    // (H,) running softmax denominator, forward only
  float* softr;    // (H,) accumulator rescale of the current chunk
  float* gwig;     // (BE, NW) Wigner-gradient accumulator, backward only

  DPA4_DEV void bind(float* base, int kc_len, int n_head) {
    scratch = base;
    astage = scratch + SCRATCH;
    wig = astage + T::WARPS * T::TM * kWarp;
    kc = wig + T::BE * S::NW;
    alpha = kc + T::BE * kc_len;
    rescale = alpha + T::BE * n_head;
    softm = rescale + S::DIM;
    softd = softm + n_head;
    softr = softd + n_head;
    gwig = softr + n_head;
  }

  static int bytes(int kc_len, int n_head, bool backward) {
    return static_cast<int>(sizeof(float)) *
           (SCRATCH + T::WARPS * T::TM * kWarp +
            T::BE * (S::NW + kc_len + n_head) + S::DIM + 3 * n_head +
            (backward ? T::BE * S::NW : 0));
  }
};

/// Accumulate ``acc = A @ W`` for one register tile of a block multiply.
///
/// ``areg`` is the register-resident left operand: activation column
/// ``JBEG * 32 + k`` lives in lane ``k % 32`` at slot ``JBEG + k / 32``, so the
/// broadcast is a warp shuffle rather than a memory read. ``w`` is the
/// ``(KK, NN)`` weight matrix in ``(in, out)`` order and ``acc[i][j]`` holds
/// output column ``j * 32 + lane``, so one weight row is a coalesced
/// 128-byte transaction per column group and every warp of the block walks the
/// same lines.
///
/// Broadcast one reduction step's edge tile from the step-major stage.
///
/// Every lane reads the same address, and tile sizes of four and two map onto
/// one 16- or 8-byte load.
template <int TM>
DPA4_DEV void broadcast_tile(const float* step, float (&av)[TM]) {
  if constexpr (TM == 4) {
    const float4 v = *reinterpret_cast<const float4*>(step);
    av[0] = v.x;
    av[1] = v.y;
    av[2] = v.z;
    av[3] = v.w;
  } else if constexpr (TM == 2) {
    const float2 v = *reinterpret_cast<const float2*>(step);
    av[0] = v.x;
    av[1] = v.y;
  } else {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      av[i] = step[i];
    }
  }
}

/// Accumulate ``acc = A @ W`` for one register tile of a block multiply.
///
/// ``areg`` is the register-resident left operand: activation column
/// ``JBEG * 32 + k`` lives in lane ``k % 32`` at slot ``JBEG + k / 32``, so a
/// reduction step broadcasts it with a warp shuffle. ``acc[i][j]`` holds output
/// column ``j * 32 + lane``.
///
/// ``w`` is the ``(KK, NN)`` weight matrix in ``(in, out)`` order, repacked by
/// the host to ``(KK / 4, NN, 4)`` so that the four reduction steps of one
/// output column are contiguous. A lane then covers four steps of a column
/// group with one 16-byte shared load instead of four scalar ones, which is
/// what moves the inner loop off its issue ceiling: the step count per load
/// drops from one to four while the products per step are unchanged. The
/// packing leaves the panel decomposition alone, because a panel depth is
/// always a multiple of four, and it is conflict free: eight lanes of a load
/// phase start four banks apart and together cover all thirty-two.
///
/// The reduction is walked as ``KK / 32`` column groups of the activation, each
/// covering ``32 / PK`` staged weight panels. Three properties are load
/// bearing.
///
/// 1. The column-group loop is unrolled and the panel loop inside it stays
///    rolled. This is what makes the activation slot a compile-time index: the
///    tile is read as ``areg[i][JBEG + cg]`` rather than selected with a
///    predicated comparison chain over all ``AB`` slots, which costs ``TM *
///    AB`` instructions per column group and was half again the products it
///    fed. Unrolling the panel loop as well would put up to twelve panel bodies
///    in the instruction stream at the wider degrees and lose a third of the
///    issue slots to instruction fetch, so only the outer level is expanded.
/// 2. The panel depth sets the staging footprint ``PK * NN``. Pinning it to the
///    whole 32-row column group costs shared memory the residency needs.
/// 3. The next panel is fetched immediately after the barrier that publishes
///    the current one. The weight matrices are L2 resident, but a fetch
///    consumed by the barrier that follows it exposes several hundred cycles
///    per panel and cost this kernel a factor of 1.74 before it was moved.
///    Reading the weights straight from the cache hierarchy instead removes the
///    barriers but is a factor of 1.85 slower overall: at the eight warps per
///    multiprocessor this tile allows, nothing covers the dependent load
///    latency inside the reduction. Staging the left operand in shared memory
///    instead of registers is worse still: the tile costs 29 KB per block,
///    which drops the multiprocessor to a single resident block.
///
/// ``active`` is false for the warps a short tail chunk does not reach. They
/// still cross every barrier and carry their share of the panel staging; only
/// the products are skipped.
template <typename T, int TM, int AB, int TN, int JBEG, int KK, int NN>
DPA4_DEV void row_multiply(const float* __restrict__ w,
                           const float (&areg)[TM][AB],
                           float* w_s,
                           float* a_s,
                           int lane,
                           bool active,
                           float acc[TM][TN]) {
  static_assert(NN % kWarp == 0, "weight width must cover the column groups");
  static_assert(KK % kWarp == 0,
                "reduction depth must cover the column groups");
  static_assert(T::PK % 4 == 0, "panel depth must cover the packed step group");
  constexpr int PANEL_ELEMS = T::PK * NN;
  constexpr int PANELS = KK / T::PK;
  // Activation column groups, and the panels each of them spans.
  constexpr int GROUPS = KK / kWarp;
  constexpr int PANELS_PER_GROUP = kWarp / T::PK;
  static_assert(kWarp % T::PK == 0,
                "a column group must hold a whole number of panels");
  // A panel splits into 16-byte copies whenever its size allows, and single
  // floats otherwise; the narrow case only arises for gate projections whose
  // width is not a power of two.
  constexpr int VEC = (PANEL_ELEMS % (T::NT * 4) == 0) ? 4 : 1;
  constexpr int VPT = PANEL_ELEMS / (T::NT * VEC);
  static_assert(PANEL_ELEMS % (T::NT * VEC) == 0,
                "panel must split evenly over the block");
  const int tid = threadIdx.x;
  // One panel is staged with asynchronous copies while the previous one is
  // being consumed, so the transfer neither passes through registers nor
  // exposes its latency to the barrier that publishes it.
  const auto stage = [&](int panel) {
    const float* from = w + static_cast<long>(panel) * PANEL_ELEMS;
    float* into = w_s + (panel & 1) * PANEL_ELEMS;
#pragma unroll
    for (int p = 0; p < VPT; ++p) {
      const int at = (tid + p * T::NT) * VEC;
      __pipeline_memcpy_async(into + at, from + at, VEC * sizeof(float));
    }
    __pipeline_commit();
  };
  // The staging region is shared with the outer-product and reduction phases
  // and with the trailing panels of the previous multiply, so the first copy
  // may not be issued before every warp of the block is done with it.
  __syncthreads();
  stage(0);
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.f;
    }
  }
#pragma unroll
  for (int group = 0; group < GROUPS; ++group) {
    // Publish this column group into the warp-private stage. The slot is a
    // compile-time index, so the tile is read straight out of the registers.
    // The stage is step major, so one vector load broadcasts the whole edge
    // tile of a reduction step to every lane; a shuffle here is quarter rate
    // and costs as many issue slots as the products it feeds.
    if (active) {
      __syncwarp();
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_s[lane * TM + i] = areg[i][JBEG + group];
      }
      __syncwarp();
    }
#pragma unroll 1
    for (int p = 0; p < PANELS_PER_GROUP; ++p) {
      const int panel = group * PANELS_PER_GROUP + p;
      // The buffer the next copy fills was last read one iteration ago; the
      // barrier orders that read before the overwrite.
      __syncthreads();
      if (panel + 1 < PANELS) {
        stage(panel + 1);
        __pipeline_wait_prior(1);
      } else {
        __pipeline_wait_prior(0);
      }
      __syncthreads();
      const float* w_cur = w_s + (panel & 1) * PANEL_ELEMS;
      if (!active) {
        continue;
      }
      const int t0 = p * T::PK;
      // Two step groups, eight reduction steps, is enough to cover the
      // shared-memory latency; unrolling a deeper panel in full costs more in
      // instruction fetch than it recovers.
#pragma unroll 2
      for (int t4 = 0; t4 < T::PK / 4; ++t4) {
        // One 16-byte load per column group carries the whole step group.
        float bv[TN][4];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const float4 v = *reinterpret_cast<const float4*>(
              w_cur + (t4 * NN + j * kWarp + lane) * 4);
          bv[j][0] = v.x;
          bv[j][1] = v.y;
          bv[j][2] = v.z;
          bv[j][3] = v.w;
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          float av[TM];
          broadcast_tile<TM>(a_s + (t0 + t4 * 4 + t) * TM, av);
#pragma unroll
          for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
              acc[i][j] += av[i] * bv[j][t];
            }
          }
        }
      }
    }
  }
}

/// Stage the per-edge geometry and topology of one chunk.
template <int L, int CF, typename T>
DPA4_DEV void load_chunk(const ConvArgs& a,
                         const ConvSmem<L, CF, T>& sm,
                         long beg,
                         int ne,
                         int focus,
                         int64_t* edge_s,
                         int64_t* peer_s) {
  using S = ConvShape<L, CF>;
  // Slots a short tail chunk leaves unfilled point at edge zero so every
  // address the padded lanes form stays in range; their results are discarded
  // by the ``e < ne`` guards on the stores.
  for (int i = threadIdx.x; i < T::BE; i += T::NT) {
    const long edge = i < ne ? a.order[beg + i] : 0;
    edge_s[i] = edge;
    peer_s[i] = a.peer[edge];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < ne * S::NW; idx += T::NT) {
    const int i = idx / S::NW;
    const int t = idx - i * S::NW;
    sm.wig[idx] = a.runs[edge_s[i] * S::NW + t];
  }
  for (int idx = threadIdx.x; idx < ne * a.kc_len; idx += T::NT) {
    const int i = idx / a.kc_len;
    sm.kc[idx] = a.kc[edge_s[i] * a.kc_len + (idx - i * a.kc_len)];
  }
  if (a.alpha != nullptr) {
    for (int idx = threadIdx.x; idx < ne * a.n_head; idx += T::NT) {
      const int i = idx / a.n_head;
      sm.alpha[idx] = a.alpha[edge_s[i] * a.a_se + focus * a.a_sf +
                              (idx - i * a.n_head) * a.a_sh];
    }
  }
}

/// Attention logits and the online-softmax update of one forward chunk.
///
/// The logit of edge ``e`` and head ``h`` is the scaled query-key dot plus the
/// radial bias plus twice the log envelope; an edge outside the cutoff carries
/// no mass. The raw logit is stashed in the weight output so the epilogue can
/// normalize it once the segment maximum is final, and the staged chunk weight
/// becomes ``exp(logit - m)`` in the running frame, with the node accumulator
/// rescale published in ``softr``. Heads never split a 32-lane channel slot,
/// which the host guarantees by requiring a head width of at least one warp.
template <int L, int CF, typename T>
DPA4_DEV void attention_chunk(const ConvArgs& a,
                              const ConvSmem<L, CF, T>& sm,
                              const int64_t* edge_s,
                              const int64_t* peer_s,
                              const float (&qv)[ConvShape<L, CF>::CFB],
                              int ne,
                              int focus,
                              int warp,
                              int lane,
                              int head_dim) {
  using S = ConvShape<L, CF>;
  // === Step 1. Raw logits of this warp's edges ===
#pragma unroll
  for (int i = 0; i < T::TM; ++i) {
    const int e = warp * T::TM + i;
    if (e >= ne) {
      continue;
    }
    const long edge = edge_s[e];
    const long src_node = peer_s[e];
    const float ev = a.env[edge];
    const float log_env2 = (ev > 0.f) ? 2.f * logf(ev) : -1e30f;
    float qk = 0.f;
    float bias = 0.f;
#pragma unroll
    for (int cb = 0; cb < S::CFB; ++cb) {
      const int ca = cb * kWarp + lane;
      const float kv =
          a.k[src_node * static_cast<long>(a.c_wide) + focus * CF + ca];
      qk += qv[cb] * kv;
      const int head = ca / head_dim;
      bias += a.kc0[edge * static_cast<long>(a.c_wide) + focus * CF + ca] *
              a.logit_w[(static_cast<long>(focus) * CF + ca) * a.n_head + head];
    }
    // The operator serves one attention head, so the warp sum completes both
    // contractions over the full focus width.
    const float dot = warp_all_sum(qk) * a.inv_sqrt_ch;
    const float bias_sum = warp_all_sum(bias);
    if (lane < a.n_head) {
      const float eff = dot + bias_sum + log_env2;
      sm.alpha[e * a.n_head + lane] = eff;
      a.alpha_out[(edge * a.n_focus + focus) * a.n_head + lane] = eff;
    }
  }
  __syncthreads();

  // === Step 2. Fold the chunk into the running segment state ===
  // The scan is warp serial: a chunk holds at most ``BE`` logits per head and
  // the arithmetic is trivial next to one panel of the mixing stack.
  if (warp == 0) {
    for (int h = 0; h < a.n_head; ++h) {
      float local = -1e30f;
      for (int e = lane; e < ne; e += kWarp) {
        local = fmaxf(local, sm.alpha[e * a.n_head + h]);
      }
      const float chunk_max = warp_all_max(local);
      const float m_old = sm.softm[h];
      const float m_new = fmaxf(m_old, chunk_max);
      float part = 0.f;
      for (int e = lane; e < ne; e += kWarp) {
        const float w = expf(sm.alpha[e * a.n_head + h] - m_new);
        // The cross-focus competition scales the finished weight outside the
        // softmax, so it multiplies the staged value but not the denominator.
        const float fs = (a.fscale == nullptr)
                             ? 1.f
                             : a.fscale[edge_s[e] * a.n_focus + focus];
        sm.alpha[e * a.n_head + h] = w * fs;
        part += w;
      }
      const float chunk_sum = warp_all_sum(part);
      if (lane == 0) {
        const float r = expf(m_old - m_new);
        sm.softd[h] = sm.softd[h] * r + chunk_sum;
        sm.softm[h] = m_new;
        sm.softr[h] = r;
      }
    }
  }
  __syncthreads();
}

/// Per-channel mixer view of chunk edge ``e`` at channel slot ``cb``.///
/// Per-channel mixer view of chunk edge ``e`` at channel slot ``cb``.
template <int L, int CF, typename T>
DPA4_DEV DegreeMixer<L> mixer_of(const ConvArgs& a,
                                 const ConvSmem<L, CF, T>& sm,
                                 int e,
                                 int focus,
                                 int cb,
                                 int lane) {
  DegreeMixer<L> mix;
  mix.kc = sm.kc + e * a.kc_len;
  mix.cb = a.cb;
  mix.c_wide = a.c_wide;
  mix.channel = focus * CF + cb * kWarp + lane;
  mix.rank = a.rank;
  return mix;
}

/// Run the gated mixing stack over the register activation, saving one slot of
/// state per layer for the backward.
///
/// Gated layers save their pre-activation; the final identity layer saves the
/// finished activation, which is what lets the reverse sweep start at the
/// inverse rotation instead of replaying the forward.
template <int L, int CF, typename T>
DPA4_DEV void run_stack(const ConvArgs& a,
                        const ConvSmem<L, CF, T>& sm,
                        const int64_t* edge_s,
                        int focus,
                        int warp,
                        int lane,
                        int ne,
                        float (&ureg)[T::TM][ConvShape<L, CF>::RB]) {
  using S = ConvShape<L, CF>;
  const long w0_stride = static_cast<long>(S::M0) * S::M0;
  const long w1_stride = static_cast<long>(S::M1) * S::M1;
  const long gw_stride = static_cast<long>(CF) * S::GATE;
  const long z_stride = static_cast<long>(a.n_edge) * a.n_focus * S::ROW;
  const bool active = warp * T::TM < ne;
#pragma unroll 1
  for (int layer = 0; layer < a.n_layers; ++layer) {
    const bool gated = layer < a.n_layers - 1;
    float sg[T::TM][S::GB];
    float acc0[T::TM][S::M0B];
    row_multiply<T, T::TM, S::RB, S::M0B, 0, S::M0, S::M0>(
        a.w0 + (layer * a.n_focus + focus) * w0_stride, ureg, sm.scratch,
        sm.astage + warp * T::TM * kWarp, lane, active, acc0);
    if (gated) {
      float accg[T::TM][S::GB];
      row_multiply<T, T::TM, S::M0B, S::GB, 0, CF, S::GATE>(
          a.gw + (layer * a.n_focus + focus) * gw_stride, acc0, sm.scratch,
          sm.astage + warp * T::TM * kWarp, lane, active, accg);
#pragma unroll
      for (int i = 0; i < T::TM; ++i) {
#pragma unroll
        for (int j = 0; j < S::GB; ++j) {
          sg[i][j] = sigmoid_f(accg[i][j]);
        }
      }
    }
    if (active) {
#pragma unroll
      for (int i = 0; i < T::TM; ++i) {
        const int e = warp * T::TM + i;
        const bool store = e < ne;
        float* zp = a.z_all + layer * z_stride +
                    (edge_s[e] * a.n_focus + focus) * S::ROW;
#pragma unroll
        for (int j = 0; j < S::M0B; ++j) {
          ureg[i][j] += gated ? ((j < S::CFB) ? silu_f(acc0[i][j])
                                              : acc0[i][j] * sg[i][j - S::CFB])
                              : acc0[i][j];
          if (store) {
            zp[j * kWarp + lane] = gated ? acc0[i][j] : ureg[i][j];
          }
        }
      }
    }
    float acc1[T::TM][S::M1B];
    row_multiply<T, T::TM, S::RB, S::M1B, S::M0B, S::M1, S::M1>(
        a.w1 + (layer * a.n_focus + focus) * w1_stride, ureg, sm.scratch,
        sm.astage + warp * T::TM * kWarp, lane, active, acc1);
    if (active) {
#pragma unroll
      for (int i = 0; i < T::TM; ++i) {
        const int e = warp * T::TM + i;
        const bool store = e < ne;
        float* zp = a.z_all + layer * z_stride +
                    (edge_s[e] * a.n_focus + focus) * S::ROW + S::M0;
#pragma unroll
        for (int j = 0; j < S::M1B; ++j) {
          const int slot = S::M0B + j;
          ureg[i][slot] += gated ? acc1[i][j] * sg[i][j % S::GB] : acc1[i][j];
          if (store) {
            zp[j * kWarp + lane] = gated ? acc1[i][j] : ureg[i][slot];
          }
        }
      }
    }
  }
}

/// Reduce a per-warp, per-channel node accumulator across the block and store.
template <int L, int CF, typename T>
DPA4_DEV void reduce_node(const ConvSmem<L, CF, T>& sm,
                          int warp,
                          int lane,
                          int cb,
                          int d,
                          const float value,
                          float& out) {
  float* red = sm.scratch;
  __syncthreads();
  red[warp * kWarp + lane] = value;
  __syncthreads();
  float v = 0.f;
#pragma unroll
  for (int w = 0; w < T::WARPS; ++w) {
    v += red[w * kWarp + lane];
  }
  out = v;
  (void)cb;
  (void)d;
}

template <int L, int CF, typename T>
__global__ __launch_bounds__(T::NT,
                             T::OCC) void so2_conv_fwd_kernel(ConvArgs a,
                                                              float* out,
                                                              float* pre_gate) {
  using S = ConvShape<L, CF>;
  extern __shared__ float smem[];
  ConvSmem<L, CF, T> sm;
  sm.bind(smem, a.kc_len, a.n_head);
  __shared__ int64_t edge_s[T::BE];
  __shared__ int64_t peer_s[T::BE];
  // Transaction barriers of the panel staging, one per buffer. They are armed
  // once and carry their phase across every multiply of the block; parts
  // without the bulk copy engine stage through the asynchronous copy pipeline
  // and leave them idle.

  const int node = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid & (kWarp - 1);
  const int head_dim = CF / a.n_head;
  for (int d = tid; d < S::DIM; d += T::NT) {
    sm.rescale[d] = a.rescale[d];
  }
  const long beg = a.row_ptr[node];
  const long end = a.row_ptr[node + 1];

  // === Step 1. One focus stream at a time ===
  // The focus loop is outermost so the node accumulator stays register sized.
  // Restaging the per-edge geometry per stream costs a few hundred bytes per
  // edge, against ``n_focus`` times the accumulator registers if it were not.
  for (int focus = 0; focus < a.n_focus; ++focus) {
    float oacc[S::CFB][S::DIM];
#pragma unroll
    for (int cb = 0; cb < S::CFB; ++cb) {
#pragma unroll
      for (int d = 0; d < S::DIM; ++d) {
        oacc[cb][d] = 0.f;
      }
    }
    // The query of the owning node and the softmax state of this stream. The
    // running maximum starts at the null-mass logit, which keeps an empty or
    // fully cut segment finite without a fallback branch.
    float qv[S::CFB];
#pragma unroll
    for (int cb = 0; cb < S::CFB; ++cb) {
      qv[cb] = a.q[static_cast<long>(node) * a.c_wide + focus * CF +
                   cb * kWarp + lane];
    }
    if (tid < a.n_head) {
      sm.softm[tid] = a.null_logit[focus * a.n_head + tid];
      sm.softd[tid] = 0.f;
    }

    for (long chunk = beg; chunk < end; chunk += T::BE) {
      const long left = end - chunk;
      const int ne = left < T::BE ? static_cast<int>(left) : T::BE;
      __syncthreads();
      load_chunk<L, CF, T>(a, sm, chunk, ne, focus, edge_s, peer_s);
      __syncthreads();

      // === Step 2. Attention weights of this chunk, online softmax frame ===
      attention_chunk<L, CF, T>(a, sm, edge_s, peer_s, qv, ne, focus, warp,
                                lane, head_dim);
#pragma unroll
      for (int cb = 0; cb < S::CFB; ++cb) {
        const float r = sm.softr[(cb * kWarp + lane) / head_dim];
#pragma unroll
        for (int d = 0; d < S::DIM; ++d) {
          oacc[cb][d] *= r;
        }
      }

      // === Step 3. Rotate into the edge frame, then mix the radial degrees ===
      float ureg[T::TM][S::RB];  // (TM, ROW / 32) activation, lane-major
#pragma unroll
      for (int i = 0; i < T::TM; ++i) {
        const long src_node = peer_s[warp * T::TM + i];
#pragma unroll
        for (int cb = 0; cb < S::CFB; ++cb) {
          const float* xp = a.x + src_node * static_cast<long>(a.x_sn) +
                            focus * CF + cb * kWarp + lane;
          float xv[S::DIM];  // (DIM,) packed node coefficients of one channel
#pragma unroll
          for (int j = 0; j < S::DIM; ++j) {
            xv[j] = xp[static_cast<long>(j) * a.x_sd];
          }
          float xl[S::RED];  // (RED,) reduced local-frame coefficients
          rotate_to_local<L>(xv, sm.wig + (warp * T::TM + i) * S::NW, xl);
          float y[S::RED];
          degree_mix<L>(
              xl, mixer_of<L, CF, T>(a, sm, warp * T::TM + i, focus, cb, lane),
              y);
#pragma unroll
          for (int r = 0; r < S::RED; ++r) {
            ureg[i][r * S::CFB + cb] = y[r];
          }
        }
      }

      // === Step 4. Gated mixing stack ===
      run_stack<L, CF, T>(a, sm, edge_s, focus, warp, lane, ne, ureg);

      // === Step 5. Inverse rotation, attention weight, destination reduction
      // ===
#pragma unroll
      for (int i = 0; i < T::TM; ++i) {
        const int e = warp * T::TM + i;
        if (e >= ne) {
          continue;
        }
#pragma unroll
        for (int cb = 0; cb < S::CFB; ++cb) {
          float xl[S::RED];
#pragma unroll
          for (int r = 0; r < S::RED; ++r) {
            xl[r] = ureg[i][r * S::CFB + cb];
          }
          const int head = (cb * kWarp + lane) / head_dim;
          float rb[S::DIM];  // (DIM,) packed global-frame message
          rotate_to_global<L>(xl, sm.wig + e * S::NW,
                              sm.alpha[e * a.n_head + head], rb);
#pragma unroll
          for (int d = 0; d < S::DIM; ++d) {
            oacc[cb][d] += rb[d];
          }
        }
      }
    }

    // === Step 6. Normalize, rescale, apply the output head gate, store ===
    for (int cb = 0; cb < S::CFB; ++cb) {
      const int channel = focus * CF + cb * kWarp + lane;
      const int head = (cb * kWarp + lane) / head_dim;
      const float denom =
          sm.softd[head] +
          expf(a.null_logit[focus * a.n_head + head] - sm.softm[head]);
      const float gate =
          a.head_gate[(static_cast<long>(node) * a.n_focus + focus) * a.n_head +
                      head];
      for (int d = 0; d < S::DIM; ++d) {
        float v = 0.f;
        reduce_node<L, CF, T>(sm, warp, lane, cb, d, oacc[cb][d], v);
        if (warp == 0) {
          const long idx =
              (static_cast<long>(node) * S::DIM + d) * a.c_wide + channel;
          const float pre = v * sm.rescale[d] / denom;
          pre_gate[idx] = pre;
          out[idx] = pre * gate;
        }
      }
    }

    // === Step 7. Normalize the stashed logits into the weight output ===
    __syncthreads();
    for (long p = beg + tid; p < end; p += T::NT) {
      const long edge = a.order[p];
#pragma unroll 1
      for (int h = 0; h < a.n_head; ++h) {
        const float denom =
            sm.softd[h] +
            expf(a.null_logit[focus * a.n_head + h] - sm.softm[h]);
        const float fs =
            (a.fscale == nullptr) ? 1.f : a.fscale[edge * a.n_focus + focus];
        float* ap = a.alpha_out + (edge * a.n_focus + focus) * a.n_head + h;
        *ap = expf(*ap - sm.softm[h]) / denom * fs;
      }
    }
    __syncthreads();
  }
}

template <int L, int CF, typename T>
__global__ __launch_bounds__(T::NT, T::OCC) void so2_conv_bwd_kernel(
    ConvArgs a,
    const float* g_out,
    const float* w0t,
    const float* w1t,
    const float* gwt,
    float* g_x,
    float* g_runs,
    float* g_kc,
    float* g_alpha) {
  using S = ConvShape<L, CF>;
  extern __shared__ float smem[];
  ConvSmem<L, CF, T> sm;
  sm.bind(smem, a.kc_len, a.n_head);
  __shared__ int64_t edge_s[T::BE];
  __shared__ int64_t peer_s[T::BE];
  // Transaction barriers of the panel staging, one per buffer. They are armed
  // once and carry their phase across every multiply of the block; parts
  // without the bulk copy engine stage through the asynchronous copy pipeline
  // and leave them idle.

  const int node = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid & (kWarp - 1);
  const int head_dim = CF / a.n_head;
  const long w0_stride = static_cast<long>(S::M0) * S::M0;
  const long w1_stride = static_cast<long>(S::M1) * S::M1;
  const long gw_stride = static_cast<long>(CF) * S::GATE;
  const long z_stride = static_cast<long>(a.n_edge) * a.n_focus * S::ROW;
  for (int d = tid; d < S::DIM; d += T::NT) {
    sm.rescale[d] = a.rescale[d];
  }
  const long beg = a.row_ptr[node];
  const long end = a.row_ptr[node + 1];

  for (int focus = 0; focus < a.n_focus; ++focus) {
    // The whole segment shares one source node, so its feature is read once.
    float xnode[S::CFB][S::DIM];
#pragma unroll
    for (int cb = 0; cb < S::CFB; ++cb) {
      const float* xp = a.x + static_cast<long>(node) * a.x_sn + focus * CF +
                        cb * kWarp + lane;
#pragma unroll
      for (int j = 0; j < S::DIM; ++j) {
        xnode[cb][j] = xp[static_cast<long>(j) * a.x_sd];
      }
    }
    float gx_acc[S::CFB][S::DIM];
#pragma unroll
    for (int cb = 0; cb < S::CFB; ++cb) {
#pragma unroll
      for (int d = 0; d < S::DIM; ++d) {
        gx_acc[cb][d] = 0.f;
      }
    }

    for (long chunk = beg; chunk < end; chunk += T::BE) {
      const long left_n = end - chunk;
      const int ne = left_n < T::BE ? static_cast<int>(left_n) : T::BE;
      const bool active = warp * T::TM < ne;
      __syncthreads();
      load_chunk<L, CF, T>(a, sm, chunk, ne, focus, edge_s, peer_s);
      for (int idx = tid; idx < ne * S::NW; idx += T::NT) {
        sm.gwig[idx] = 0.f;
      }
      __syncthreads();

      // === Step 1. Load the saved final activation ===
      float ureg[T::TM][S::RB];  // final activation, then the cotangent
#pragma unroll
      for (int i = 0; i < T::TM; ++i) {
        const float* zp =
            a.z_all + (a.n_layers - 1) * z_stride +
            (edge_s[warp * T::TM + i] * a.n_focus + focus) * S::ROW;
#pragma unroll
        for (int j = 0; j < S::RB; ++j) {
          ureg[i][j] = zp[j * kWarp + lane];
        }
      }

      // === Step 2. Inverse-rotation VJP, alpha gradient, Wigner outer product
      // === One warp owns one edge's full channel set, which is what makes the
      // alpha and Wigner channel reductions warp-local.
      float* ob = sm.scratch + warp * S::OUTER;
#pragma unroll
      for (int i = 0; i < T::TM; ++i) {
        const int e = warp * T::TM + i;
        const bool live = e < ne;
        const long dst = peer_s[e];
        float ga[S::CFB];
#pragma unroll
        for (int cb = 0; cb < S::CFB; ++cb) {
          float uf[S::RED];
#pragma unroll
          for (int r = 0; r < S::RED; ++r) {
            uf[r] = ureg[i][r * S::CFB + cb];
          }
          float rb[S::DIM];
          rotate_to_global<L>(uf, sm.wig + e * S::NW, 1.f, rb);
          const int head = (cb * kWarp + lane) / head_dim;
          const float* gp =
              g_out + dst * S::DIM * a.c_wide + focus * CF + cb * kWarp + lane;
          const float gate =
              a.head_gate[(dst * a.n_focus + focus) * a.n_head + head];
          const float wgt = sm.alpha[e * a.n_head + head];
          float g_rb[S::DIM];
          ga[cb] = 0.f;
#pragma unroll
          for (int d = 0; d < S::DIM; ++d) {
            const float gv =
                sm.rescale[d] * gate * gp[static_cast<long>(d) * a.c_wide];
            g_rb[d] = wgt * gv;
            ga[cb] += gv * rb[d];
          }
          float g_uf[S::RED];
          rotate_to_global_vjp<L>(g_rb, sm.wig + e * S::NW, 1.f, g_uf);
#pragma unroll
          for (int r = 0; r < S::RED; ++r) {
            ob[r * kStageStride + lane] = uf[r];
          }
#pragma unroll
          for (int d = 0; d < S::DIM; ++d) {
            ob[(S::RED + d) * kStageStride + lane] = g_rb[d];
          }
          __syncwarp();
          accumulate_wigner_grad<L>(ob, ob + S::RED * kStageStride, lane,
                                    sm.gwig + e * S::NW, live);
          __syncwarp();
#pragma unroll
          for (int r = 0; r < S::RED; ++r) {
            ureg[i][r * S::CFB + cb] = g_uf[r];
          }
        }
        // A head spans ``head_dim`` channels, which may cross channel slots.
        for (int h = 0; h < a.n_head; ++h) {
          float part = 0.f;
#pragma unroll
          for (int cb = 0; cb < S::CFB; ++cb) {
            if ((cb * kWarp + lane) / head_dim == h) {
              part += ga[cb];
            }
          }
          const float s = warp_all_sum(part);
          if (live && lane == 0) {
            g_alpha[edge_s[e] * a.a_se + focus * a.a_sf + h * a.a_sh] = s;
          }
        }
      }

      // === Step 3. Reverse the identity layer ===
      {
        float acc0[T::TM][S::M0B];
        row_multiply<T, T::TM, S::RB, S::M0B, 0, S::M0, S::M0>(
            w0t + ((a.n_layers - 1) * a.n_focus + focus) * w0_stride, ureg,
            sm.scratch, sm.astage + warp * T::TM * kWarp, lane, active, acc0);
        float acc1[T::TM][S::M1B];
        row_multiply<T, T::TM, S::RB, S::M1B, S::M0B, S::M1, S::M1>(
            w1t + ((a.n_layers - 1) * a.n_focus + focus) * w1_stride, ureg,
            sm.scratch, sm.astage + warp * T::TM * kWarp, lane, active, acc1);
#pragma unroll
        for (int i = 0; i < T::TM; ++i) {
#pragma unroll
          for (int j = 0; j < S::M0B; ++j) {
            ureg[i][j] += acc0[i][j];
          }
#pragma unroll
          for (int j = 0; j < S::M1B; ++j) {
            ureg[i][S::M0B + j] += acc1[i][j];
          }
        }
      }

      // === Step 4. Reverse the gated layers ===
#pragma unroll 1
      for (int layer = a.n_layers - 2; layer >= 0; --layer) {
        // The saved pre-activation is fetched before the gate projection so its
        // latency is covered by that multiply rather than exposed on the
        // point-wise use that follows.
        float zr[T::TM][S::RB];
#pragma unroll
        for (int i = 0; i < T::TM; ++i) {
          const float* zp =
              a.z_all + layer * z_stride +
              (edge_s[warp * T::TM + i] * a.n_focus + focus) * S::ROW;
#pragma unroll
          for (int j = 0; j < S::RB; ++j) {
            zr[i][j] = zp[j * kWarp + lane];
          }
        }
        float z0[T::TM][S::M0B];
#pragma unroll
        for (int i = 0; i < T::TM; ++i) {
#pragma unroll
          for (int j = 0; j < S::M0B; ++j) {
            z0[i][j] = zr[i][j];
          }
        }
        float sg[T::TM][S::GB];
        {
          float accg[T::TM][S::GB];
          row_multiply<T, T::TM, S::M0B, S::GB, 0, CF, S::GATE>(
              a.gw + (layer * a.n_focus + focus) * gw_stride, z0, sm.scratch,
              sm.astage + warp * T::TM * kWarp, lane, active, accg);
#pragma unroll
          for (int i = 0; i < T::TM; ++i) {
#pragma unroll
            for (int j = 0; j < S::GB; ++j) {
              sg[i][j] = sigmoid_f(accg[i][j]);
            }
          }
        }
        float gsave[T::TM][S::RB];
        float gsp[T::TM][S::GB];
#pragma unroll
        for (int i = 0; i < T::TM; ++i) {
          float gs[S::GB];
#pragma unroll
          for (int q = 0; q < S::GB; ++q) {
            gs[q] = 0.f;
          }
#pragma unroll
          for (int j = 0; j < S::M0B; ++j) {
            const float g = ureg[i][j];
            gsave[i][j] = g;
            if (j < S::CFB) {
              ureg[i][j] = g * silu_grad(zr[i][j]);
            } else {
              const int q = j - S::CFB;
              ureg[i][j] = g * sg[i][q];
              gs[q] += g * zr[i][j];
            }
          }
#pragma unroll
          for (int j = 0; j < S::M1B; ++j) {
            const int slot = S::M0B + j;
            const float g = ureg[i][slot];
            gsave[i][slot] = g;
            const int q = j % S::GB;
            ureg[i][slot] = g * sg[i][q];
            gs[q] += g * zr[i][slot];
          }
#pragma unroll
          for (int q = 0; q < S::GB; ++q) {
            const float s = sg[i][q];
            gsp[i][q] = gs[q] * s * (1.f - s);
          }
        }
        // The gate-weight path folds into the scalar block of the cotangent.
        {
          float accs[T::TM][S::CFB];
          row_multiply<T, T::TM, S::GB, S::CFB, 0, S::GATE, CF>(
              gwt + (layer * a.n_focus + focus) * gw_stride, gsp, sm.scratch,
              sm.astage + warp * T::TM * kWarp, lane, active, accs);
#pragma unroll
          for (int i = 0; i < T::TM; ++i) {
#pragma unroll
            for (int j = 0; j < S::CFB; ++j) {
              ureg[i][j] += accs[i][j];
            }
          }
        }
        float acc0[T::TM][S::M0B];
        row_multiply<T, T::TM, S::RB, S::M0B, 0, S::M0, S::M0>(
            w0t + (layer * a.n_focus + focus) * w0_stride, ureg, sm.scratch,
            sm.astage + warp * T::TM * kWarp, lane, active, acc0);
        float acc1[T::TM][S::M1B];
        row_multiply<T, T::TM, S::RB, S::M1B, S::M0B, S::M1, S::M1>(
            w1t + (layer * a.n_focus + focus) * w1_stride, ureg, sm.scratch,
            sm.astage + warp * T::TM * kWarp, lane, active, acc1);
#pragma unroll
        for (int i = 0; i < T::TM; ++i) {
#pragma unroll
          for (int j = 0; j < S::M0B; ++j) {
            ureg[i][j] = gsave[i][j] + acc0[i][j];
          }
#pragma unroll
          for (int j = 0; j < S::M1B; ++j) {
            ureg[i][S::M0B + j] = gsave[i][S::M0B + j] + acc1[i][j];
          }
        }
      }

      // The outer-product staging below overlays the weight-panel region, so
      // every warp must be past the multiplies before the first write.
      __syncthreads();

      // === Step 5. Mixer and rotation VJP into the node and edge gradients ===
#pragma unroll
      for (int i = 0; i < T::TM; ++i) {
        const int e = warp * T::TM + i;
        const bool live = e < ne;
        // A padded slot aliases edge zero, so its cotangent would otherwise
        // enter both the node accumulator and edge zero's own gradients. The
        // whole body is warp uniform, so skipping it crosses no barrier.
        if (!live) {
          continue;
        }
#pragma unroll
        for (int cb = 0; cb < S::CFB; ++cb) {
          float xl[S::RED];
          rotate_to_local<L>(xnode[cb], sm.wig + e * S::NW, xl);
          float g_y[S::RED];
#pragma unroll
          for (int r = 0; r < S::RED; ++r) {
            g_y[r] = ureg[i][r * S::CFB + cb];
          }
          const DegreeMixer<L> mix =
              mixer_of<L, CF, T>(a, sm, e, focus, cb, lane);
          float g_xl[S::RED];
          degree_mix_vjp<L>(g_y, mix, g_xl);
          float g_packed[S::DIM];
          rotate_to_local_vjp<L>(g_xl, sm.wig + e * S::NW, g_packed);
#pragma unroll
          for (int d = 0; d < S::DIM; ++d) {
            gx_acc[cb][d] += g_packed[d];
          }
          accumulate_mixer_grad<L>(g_y, xl, mix, lane, ob,
                                   g_kc + edge_s[e] * a.kc_len);
#pragma unroll
          for (int r = 0; r < S::RED; ++r) {
            ob[r * kStageStride + lane] = g_xl[r];
          }
#pragma unroll
          for (int d = 0; d < S::DIM; ++d) {
            ob[(S::RED + d) * kStageStride + lane] = xnode[cb][d];
          }
          __syncwarp();
          accumulate_wigner_grad<L>(ob, ob + S::RED * kStageStride, lane,
                                    sm.gwig + e * S::NW, live);
          __syncwarp();
        }
      }
      // === Step 6. Scatter the packed-run cotangent ===
      // The contraction onto the quaternions is a standalone kernel; each edge
      // belongs to exactly one block and the focus streams visit it
      // sequentially, so the read-modify-write is exclusive.
      __syncthreads();
      for (int idx = tid; idx < ne * S::NW; idx += T::NT) {
        const int i = idx / S::NW;
        const int t = idx - i * S::NW;
        g_runs[edge_s[i] * S::NW + t] += sm.gwig[idx];
      }
    }

    // === Step 7. Reduce the node gradient across the block ===
    for (int cb = 0; cb < S::CFB; ++cb) {
      for (int d = 0; d < S::DIM; ++d) {
        float v = 0.f;
        reduce_node<L, CF, T>(sm, warp, lane, cb, d, gx_acc[cb][d], v);
        if (warp == 0) {
          g_x[(static_cast<long>(node) * S::DIM + d) * a.c_wide + focus * CF +
              cb * kWarp + lane] = v;
        }
      }
    }
  }
}

}  // namespace dpa4
