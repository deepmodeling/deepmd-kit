// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Layout algebra and device primitives shared by the fused DPA4 / SeZM SO(2)
// convolution kernels.
//
// The m-major reduced layout
// --------------------------
// A convolution carries ``RED = 3 * lmax + 1`` reduced coefficient rows per
// edge and focus stream, ordered
//
//   r in [0, lmax]            -> degree r,             order m = 0
//   r in (lmax, 2 * lmax]     -> degree r - lmax,      order m = -1
//   r in (2 * lmax, 3 * lmax] -> degree r - 2 * lmax,  order m = +1
//
// so one focus stream's flat activation row is ``u[r * Cf + c]`` of width
// ``ROW = RED * Cf``. The first ``M0 = (lmax + 1) * Cf`` columns are the
// ``m = 0`` block and the remaining ``M1 = 2 * lmax * Cf`` are the two
// ``|m| = 1`` blocks. The SO(2) mixing weights are block diagonal over that
// split, which is why the stack runs as two independent multiplies plus a gate
// shared between them.
//
// Column ownership
// ----------------
// Every multiply in the stack assigns output column ``j * 32 + lane`` to lane
// ``lane`` at register slot ``j``. With ``CFB = Cf / 32`` channel slots per
// lane, slot ``j`` of the reduced row decomposes as
//
//   j = r * CFB + cb,   channel = cb * 32 + lane
//
// so the same register array serves the block multiplies, the rotations and the
// gate without any cross-lane traffic. This is what lets the activation, the
// gate sigmoids and the node accumulator all live in registers.
//
// Block-diagonal Wigner-D
// -----------------------
// A production Wigner-D matrix is block diagonal in the degree, so only the
// ``2 * l + 1`` entries of the degree block of each selected row are non-zero.
// Both the forward rotation and the inverse rotation read exactly those
// entries, and the kernels stage them per edge as a packed
// ``NW = 3 * (lmax + 1)^2 - 2`` float run. The dense reference contracts the
// full packed column in the forward direction; the two agree on any block
// diagonal matrix and differ on a dense random one, which is the same contract
// the flash-attention operator already uses for its inverse rotation.

#pragma once

#include <cuda_runtime.h>

#define DPA4_DEV __device__ __forceinline__

namespace dpa4 {

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kWarp = 32;

/// Largest supported spherical-harmonic degree.
constexpr int kMaxL = 6;

/// Packed SO(3) coefficient count of a node feature, ``(lmax + 1)^2``.
template <int L>
constexpr int packed_dim() {
  return (L + 1) * (L + 1);
}

/// Reduced m-major coefficient count of an edge feature, ``3 * lmax + 1``.
template <int L>
constexpr int reduced_dim() {
  return 3 * L + 1;
}

/// Degree of reduced row ``r``.
template <int L>
constexpr int red_degree(int r) {
  return (r <= L) ? r : ((r <= 2 * L) ? (r - L) : (r - 2 * L));
}

/// Packed Wigner-D row index of reduced row ``r``.
template <int L>
constexpr int red_wigner_row(int r) {
  const int l = red_degree<L>(r);
  return (r <= L) ? (l * l + l)
                  : ((r <= 2 * L) ? (l * l + l - 1) : (l * l + l + 1));
}

/// Offset of reduced row ``r`` inside the packed block-diagonal Wigner run.
template <int L>
constexpr int red_wigner_offset(int r) {
  int off = 0;
  for (int i = 0; i < r; ++i) {
    off += 2 * red_degree<L>(i) + 1;
  }
  return off;
}

/// Total length of the packed block-diagonal Wigner run of one edge.
template <int L>
constexpr int wigner_run() {
  return red_wigner_offset<L>(reduced_dim<L>());
}

/// Reduced row carrying degree ``l`` at order ``m = 0``.
template <int L>
constexpr int row_m0(int l) {
  return l;
}

/// Reduced row carrying degree ``l`` at order ``m = -1`` (``l >= 1``).
template <int L>
constexpr int row_mm(int l) {
  return L + l;
}

/// Reduced row carrying degree ``l`` at order ``m = +1`` (``l >= 1``).
template <int L>
constexpr int row_mp(int l) {
  return 2 * L + l;
}

/// Compact degree-kernel length of the ``mmax = 1`` radial mixer, without the
/// low-rank factor: ``(lmax + 1)^2`` entries for ``m = 0`` and ``lmax^2`` for
/// ``|m| = 1``.
template <int L>
constexpr int degree_kernel_size() {
  return (L + 1) * (L + 1) + L * L;
}

DPA4_DEV float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }

DPA4_DEV float silu_f(float x) { return x * sigmoid_f(x); }

/// Derivative of ``silu`` expressed through its own sigmoid.
DPA4_DEV float silu_grad(float x) {
  const float s = sigmoid_f(x);
  return s * (1.f + x * (1.f - s));
}

/// Warp sum over all 32 lanes, result valid in every lane.
DPA4_DEV float warp_all_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, d);
  }
  return v;
}

/// Warp maximum over all 32 lanes, result valid in every lane.
DPA4_DEV float warp_all_max(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(kFullMask, v, d));
  }
  return v;
}

/// Resolve the packed block-diagonal Wigner slot ``t`` into its reduced row and
/// the offset of the entry inside that row's degree block.
template <int L>
DPA4_DEV void wigner_slot(int t, int& row, int& col) {
  int off = 0;
  row = 0;
  col = 0;
#pragma unroll
  for (int r = 0; r < reduced_dim<L>(); ++r) {
    const int len = 2 * red_degree<L>(r) + 1;
    if (t >= off && t < off + len) {
      row = r;
      col = t - off;
    }
    off += len;
  }
}

/// Rotate one channel of a packed node feature into the reduced local frame.
///
/// ``x`` holds the ``(lmax + 1)^2`` packed coefficients of one channel and
/// ``dw`` the packed block-diagonal Wigner run of the edge.
template <int L>
DPA4_DEV void rotate_to_local(const float* x, const float* dw, float* xl) {
  constexpr int RED = reduced_dim<L>();
#pragma unroll
  for (int r = 0; r < RED; ++r) {
    const int l = red_degree<L>(r);
    const int base = l * l;
    const float* d = dw + red_wigner_offset<L>(r);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j <= 2 * l; ++j) {
      acc += d[j] * x[base + j];
    }
    xl[r] = acc;
  }
}

/// Transpose of :func:`rotate_to_local`: scatter a reduced cotangent back onto
/// the packed degree blocks it was contracted from.
template <int L>
DPA4_DEV void rotate_to_local_vjp(const float* g, const float* dw, float* out) {
  constexpr int RED = reduced_dim<L>();
#pragma unroll
  for (int j = 0; j < packed_dim<L>(); ++j) {
    out[j] = 0.f;
  }
#pragma unroll
  for (int r = 0; r < RED; ++r) {
    const int l = red_degree<L>(r);
    const int base = l * l;
    const float* d = dw + red_wigner_offset<L>(r);
    const float gv = g[r];
#pragma unroll
    for (int j = 0; j <= 2 * l; ++j) {
      out[base + j] += d[j] * gv;
    }
  }
}

/// Inverse-rotate one channel of a reduced local feature back to the packed
/// global frame, scaled by ``scale``.
///
/// The per-degree amplitude rescale of the reduced basis is left to the caller:
/// applied once after the destination reduction it costs ``DIM`` multiplies per
/// node instead of per edge.
template <int L>
DPA4_DEV void rotate_to_global(const float* xl,
                               const float* dw,
                               float scale,
                               float* out) {
#pragma unroll
  for (int l = 0; l <= L; ++l) {
    const int base = l * l;
    const float v0 = scale * xl[row_m0<L>(l)];
    const float vm = (l >= 1) ? scale * xl[row_mm<L>(l)] : 0.f;
    const float vp = (l >= 1) ? scale * xl[row_mp<L>(l)] : 0.f;
    const float* d0 = dw + red_wigner_offset<L>(row_m0<L>(l));
    const float* dm = (l >= 1) ? dw + red_wigner_offset<L>(row_mm<L>(l)) : d0;
    const float* dp = (l >= 1) ? dw + red_wigner_offset<L>(row_mp<L>(l)) : d0;
#pragma unroll
    for (int j = 0; j <= 2 * l; ++j) {
      float v = d0[j] * v0;
      if (l >= 1) {
        v += dm[j] * vm + dp[j] * vp;
      }
      out[base + j] = v;
    }
  }
}

/// Transpose of :func:`rotate_to_global` acting on a packed cotangent.
template <int L>
DPA4_DEV void rotate_to_global_vjp(const float* g,
                                   const float* dw,
                                   float scale,
                                   float* out) {
#pragma unroll
  for (int l = 0; l <= L; ++l) {
    const int base = l * l;
    const float* d0 = dw + red_wigner_offset<L>(row_m0<L>(l));
    const float* dm = (l >= 1) ? dw + red_wigner_offset<L>(row_mm<L>(l)) : d0;
    const float* dp = (l >= 1) ? dw + red_wigner_offset<L>(row_mp<L>(l)) : d0;
    float a0 = 0.f;
    float am = 0.f;
    float ap = 0.f;
#pragma unroll
    for (int j = 0; j <= 2 * l; ++j) {
      const float gv = g[base + j];
      a0 += d0[j] * gv;
      if (l >= 1) {
        am += dm[j] * gv;
        ap += dp[j] * gv;
      }
    }
    out[row_m0<L>(l)] = a0 * scale;
    if (l >= 1) {
      out[row_mm<L>(l)] = am * scale;
      out[row_mp<L>(l)] = ap * scale;
    }
  }
}

/// Per-channel view of the radial degree mixer of one edge.
///
/// ``rank == 0`` is the mixer-free variant: the compact buffer is the projected
/// radial feature itself, indexed ``[degree * c_wide + channel]``, and each
/// reduced row is scaled by its own degree's entry. ``rank >= 1`` is the
/// ``degree_channel`` mixer: the compact buffer holds
/// ``degree_kernel_size * rank`` entries per edge and the effective per-channel
/// kernel is ``sum_r kc[slot * rank + r] * cb[r * c_wide + channel]``.
template <int L>
struct DegreeMixer {
  const float* kc;
  const float* cb;
  int c_wide;
  int channel;
  int rank;

  /// Effective kernel entry of compact slot ``slot`` for this channel.
  DPA4_DEV float kernel(int slot) const {
    float acc = 0.f;
    for (int r = 0; r < rank; ++r) {
      acc += kc[slot * rank + r] * cb[r * c_wide + channel];
    }
    return acc;
  }

  /// Radial scale of degree ``l`` for this channel, mixer-free variant.
  DPA4_DEV float radial(int l) const { return kc[l * c_wide + channel]; }
};

/// Apply the radial degree mixer to one rotated channel.
template <int L>
DPA4_DEV void degree_mix(const float* xl, const DegreeMixer<L>& mix, float* y) {
  constexpr int NDEG = L + 1;
  constexpr int K0 = NDEG * NDEG;
  if (mix.rank == 0) {
#pragma unroll
    for (int o = 0; o < NDEG; ++o) {
      y[o] = xl[o] * mix.radial(o);
    }
#pragma unroll
    for (int o = 0; o < L; ++o) {
      const float rad = mix.radial(o + 1);
      y[NDEG + o] = xl[NDEG + o] * rad;
      y[NDEG + L + o] = xl[NDEG + L + o] * rad;
    }
    return;
  }
#pragma unroll
  for (int o = 0; o < NDEG; ++o) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NDEG; ++i) {
      acc += mix.kernel(i * NDEG + o) * xl[i];
    }
    y[o] = acc;
  }
#pragma unroll
  for (int o = 0; o < L; ++o) {
    float accm = 0.f;
    float accp = 0.f;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const float kv = mix.kernel(K0 + i * L + o);
      accm += kv * xl[NDEG + i];
      accp += kv * xl[NDEG + L + i];
    }
    y[NDEG + o] = accm;
    y[NDEG + L + o] = accp;
  }
}

/// Launch tile of one convolution instantiation.
///
/// ``TM`` edges per warp and ``WARPS`` warps per block, so a chunk is
/// ``BE = WARPS * TM`` edges. ``TM`` sets both the weight reuse of the block
/// multiply, which issues one weight load per ``TM`` products, and the register
/// footprint of the activation, which is ``TM * RB`` per thread. ``PK`` is the
/// weight-panel depth: it divides 32 so a panel stays inside one column group
/// of the activation, and it bounds the prefetch at ``PK * NMAX / NT``
/// registers.
/// ``OCC`` is the resident-block target handed to ``__launch_bounds__``, where
/// one leaves the assembler free. The activation tile and the weight prefetch
/// together reach the 255-register cap, which leaves a single wave of eight
/// warps per multiprocessor and exposes every panel barrier; a target above one
/// makes the assembler schedule inside a smaller budget, which it meets without
/// spilling.
template <int TM_, int WARPS_, int PK_, int OCC_>
struct ConvTile {
  static constexpr int TM = TM_;
  static constexpr int WARPS = WARPS_;
  static constexpr int PK = PK_;
  static constexpr int OCC = OCC_;
  static constexpr int NT = WARPS * kWarp;
  static constexpr int BE = WARPS * TM;
  static_assert(32 % PK == 0, "panel depth must divide the column group");
};

/// Everything one convolution launch needs, in either direction.
///
/// The forward computes the attention weights itself: ``q``, ``k``,
/// ``logit_w``, ``null_logit`` and ``env`` feed its online segment softmax and
/// the normalized weights land in ``alpha_out``. The backward consumes the
/// finished weights through ``alpha`` and leaves the softmax cotangent to the
/// caller.
struct ConvArgs {
  const float* x;           // (N, D, C_wide) node features
  const int64_t* order;     // (E,) CSR permutation of the owning endpoint
  const int64_t* row_ptr;   // (N + 1,)
  const int64_t* peer;      // src[e] forward, dst[e] backward
  const float* runs;        // (E, NW) packed block-diagonal Wigner runs
  const float* kc;          // (E, kc_len) compact degree kernel
  const float* cb;          // (rank, C_wide) channel basis, unused at rank 0
  const float* w0;          // (n_layers, F, M0, M0), (in, out) convention
  const float* w1;          // (n_layers, F, M1, M1)
  const float* gw;          // (n_layers - 1, F, Cf, GATE)
  const float* q;           // (N, C_wide) attention query, forward only
  const float* k;           // (N, C_wide) attention key, forward only
  const float* logit_w;     // (F, Cf, H) radial logit projection, forward only
  const float* null_logit;  // (F, H) log null mass, forward only
  const float* env;         // (E,) cutoff envelope, forward only
  const float* kc0;         // (E, C_wide) radial scalar row, forward only
  const float* fscale;      // (E, F) post-softmax weight scale, may be null
  const float* alpha;       // (E, F, H) finished weights, backward only
  float* alpha_out;         // (E, F, H) weight output, forward only
  const float* head_gate;   // (N, F, H) output-side head gate
  const float* rescale;     // (D,) inverse-rotation amplitude rescale
  float* z_all;             // (n_layers, E, F, ROW) saved state
  long n_edge;
  int x_sn;
  int x_sd;
  int a_se;
  int a_sf;
  int a_sh;
  int n_focus;
  int n_head;
  int n_layers;
  int rank;
  int kc_len;
  int c_wide;
  float inv_sqrt_ch;  // rsqrt of the head width, forward only
};

/// Row stride of the per-warp outer-product staging tile.
///
/// Every lane reads a different row at the same channel, so an unpadded stride
/// of 32 puts all lanes on one shared-memory bank -- a 25-way conflict at
/// ``lmax = 2``.
constexpr int kStageStride = kWarp + 1;

/// Accumulate the Wigner outer product of one edge into its packed run.
///
/// ``left`` is a ``(RED, kStageStride)`` per-warp staging row set and ``right``
/// a ``(DIM, kStageStride)`` one; slot ``t`` of the packed block-diagonal run
/// receives ``sum_c left[row][c] * right[l(row)^2 + col][c]``. A lane sums the
/// 32 channels of its slot itself, which replaces ``NW`` serialized warp
/// reductions with one shared-memory sweep. The run outgrows a warp from degree
/// three on, so lanes stride over the slots.
template <int L>
DPA4_DEV void accumulate_wigner_grad(const float* left,
                                     const float* right,
                                     int lane,
                                     float* gwig_edge,
                                     bool live) {
  constexpr int NW = wigner_run<L>();
  if (!live) {
    return;
  }
  for (int t = lane; t < NW; t += kWarp) {
    int row = 0;
    int col = 0;
    wigner_slot<L>(t, row, col);
    const int base = red_degree<L>(row) * red_degree<L>(row);
    const float* lp = left + row * kStageStride;
    const float* rp = right + (base + col) * kStageStride;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < kWarp; ++c) {
      acc += lp[c] * rp[c];
    }
    gwig_edge[t] += acc;
  }
}

/// Accumulate the compact degree-kernel gradient of one edge and channel slot.
///
/// At ``rank == 0`` the compact buffer is the radial feature itself and is
/// indexed by channel, so each slot receives one lane-local product. At
/// ``rank >= 1`` the buffer is shared by every channel, so each of the
/// ``degree_kernel_size * rank`` entries needs a reduction over the channels of
/// the warp. The per-slot products are staged through ``stage``, a per-warp
/// ``(degree_kernel_size, kStageStride)`` row set, and one lane then sums the
/// 32 channels of its slot against the channel basis: a warp reduction per
/// ``(slot, rank)`` pair costs five quarter-rate shuffles each and four times
/// the issue slots of this sweep. The caller adds the contributions of the
/// remaining channel slots and focus streams into the same location.
template <int L>
DPA4_DEV void accumulate_mixer_grad(const float* g_y,
                                    const float* xl,
                                    const DegreeMixer<L>& mix,
                                    int lane,
                                    float* stage,
                                    float* gkc_edge) {
  constexpr int NDEG = L + 1;
  constexpr int K0 = NDEG * NDEG;
  constexpr int KSZ = K0 + L * L;
  if (mix.rank == 0) {
#pragma unroll
    for (int l = 0; l <= L; ++l) {
      float v = g_y[l] * xl[l];
      if (l >= 1) {
        v += g_y[row_mm<L>(l)] * xl[row_mm<L>(l)] +
             g_y[row_mp<L>(l)] * xl[row_mp<L>(l)];
      }
      gkc_edge[l * mix.c_wide + mix.channel] += v;
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < NDEG; ++i) {
#pragma unroll
    for (int o = 0; o < NDEG; ++o) {
      stage[(i * NDEG + o) * kStageStride + lane] = g_y[o] * xl[i];
    }
  }
#pragma unroll
  for (int i = 0; i < L; ++i) {
#pragma unroll
    for (int o = 0; o < L; ++o) {
      stage[(K0 + i * L + o) * kStageStride + lane] =
          g_y[NDEG + o] * xl[NDEG + i] + g_y[NDEG + L + o] * xl[NDEG + L + i];
    }
  }
  __syncwarp();
  // The channel of lane zero anchors the 32-channel basis segment this warp
  // covers; the basis read is one broadcast per channel.
  const float* basis = mix.cb + (mix.channel - lane);
  for (int s = lane; s < KSZ; s += kWarp) {
    const float* row = stage + s * kStageStride;
    for (int r = 0; r < mix.rank; ++r) {
      float acc = 0.f;
#pragma unroll
      for (int c = 0; c < kWarp; ++c) {
        acc += row[c] * basis[r * mix.c_wide + c];
      }
      gkc_edge[s * mix.rank + r] += acc;
    }
  }
  __syncwarp();
}

/// Transpose of :func:`degree_mix` acting on a cotangent.
template <int L>
DPA4_DEV void degree_mix_vjp(const float* g,
                             const DegreeMixer<L>& mix,
                             float* out) {
  constexpr int NDEG = L + 1;
  constexpr int K0 = NDEG * NDEG;
  if (mix.rank == 0) {
#pragma unroll
    for (int o = 0; o < NDEG; ++o) {
      out[o] = g[o] * mix.radial(o);
    }
#pragma unroll
    for (int o = 0; o < L; ++o) {
      const float rad = mix.radial(o + 1);
      out[NDEG + o] = g[NDEG + o] * rad;
      out[NDEG + L + o] = g[NDEG + L + o] * rad;
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < NDEG; ++i) {
    float acc = 0.f;
#pragma unroll
    for (int o = 0; o < NDEG; ++o) {
      acc += mix.kernel(i * NDEG + o) * g[o];
    }
    out[i] = acc;
  }
#pragma unroll
  for (int i = 0; i < L; ++i) {
    float accm = 0.f;
    float accp = 0.f;
#pragma unroll
    for (int o = 0; o < L; ++o) {
      const float kv = mix.kernel(K0 + i * L + o);
      accm += kv * g[NDEG + o];
      accp += kv * g[NDEG + L + o];
    }
    out[NDEG + i] = accm;
    out[NDEG + L + i] = accp;
  }
}

}  // namespace dpa4
