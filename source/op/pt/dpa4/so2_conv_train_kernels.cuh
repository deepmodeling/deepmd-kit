// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Kernel body of the fused SO(2) value-path training forward. Included by
// the per-degree instantiation units and by the host file; the kernel lives
// in a named namespace so explicit instantiations link across translation
// units.

#pragma once

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "sezm_train_ops.cuh"

namespace dpa4_sezm_kernels {

constexpr int kThreads = 256;
constexpr int kMaxFocus = 4;

template <typename acc_t>
__device__ __forceinline__ acc_t exp_a(acc_t x) {
  return exp(x);
}
template <>
__device__ __forceinline__ float exp_a<float>(float x) {
  return __expf(x);
}

template <typename acc_t>
__device__ __forceinline__ acc_t sigmoid_a(acc_t x) {
  return acc_t(1) / (acc_t(1) + exp_a(-x));
}

// ---------------------------------------------------------------------------
// Forward mega kernel: one block per tile of TE edges.
//
// The per-edge work is a chain of vector-matrix products against weights the
// whole graph shares. A single-edge block would re-read every weight column
// from L2 once per multiply-accumulate (arithmetic intensity of half a FLOP
// per byte), which pins the kernel to the L2 bandwidth an order of magnitude
// below the FP32 roof. Tiling TE edges into one block amortizes each weight
// read over TE register accumulators, multiplying the arithmetic intensity
// by TE; the activations of every edge in the tile stay resident in shared
// memory, read as broadcasts.
//
// Shared memory layout (accumulator type), per tile slot:
//   u_a, u_b [TE][F * ROW]   running activations (double buffered)
//   sig      [TE][F * L*CF]  gate sigmoids of the current layer
//   alp      [TE][F]         competition weights
//
// The competition weight leaves the kernel in accumulator precision rather
// than the working precision of the surfaces. It is the backward's anchor for
// the whole head: the closed-form logit gradient reconstructs the softmax from
// it as p = (alpha - ls/F) / (1 - ls) and divides the traversal's alpha
// gradient by it. Rounding the anchor to bfloat16 would cost about three
// decimal digits in both, which no later promotion recovers, while the tensor
// itself is (E, F) scalars -- negligible next to the (E, F, ROW) surfaces.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
template <typename scalar_t, int L, int RANK, int TE>
__global__ void so2_value_fwd_kernel(
    const scalar_t* __restrict__ x,
    const long* __restrict__ src,
    const scalar_t* __restrict__ wig,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ cb,
    const scalar_t* __restrict__ w_fc,
    const scalar_t* __restrict__ fc_bias,
    const scalar_t* __restrict__ w0_all,
    const scalar_t* __restrict__ w1_all,
    const scalar_t* __restrict__ gw_all,
    scalar_t* __restrict__ x_out,
    scalar_t* __restrict__ z_all,
    scalar_t* __restrict__ u_final,
    typename acc_type<scalar_t>::type* __restrict__ alpha_out,
    long n_edge,
    long x_sn,
    long x_sd,
    int cf,
    int n_focus,
    int n_gated,
    bool apply_alpha,
    bool has_bias,
    float inv_tau,
    float label_smooth) {
  using acc_t = typename acc_type<scalar_t>::type;
  constexpr int NS0 = L + 1;
  constexpr int RED = 3 * L + 1;
  constexpr int DIM = (L + 1) * (L + 1);
  const long edge0 = (long)blockIdx.x * TE;
  if (edge0 >= n_edge) {
    return;
  }
  const int n_here = (int)min((long)TE, n_edge - edge0);
  const int c_wide = n_focus * cf;
  const int row_w = RED * cf;
  const int m0 = NS0 * cf;
  const int lg = L * cf;
  const int frow = n_focus * row_w;
  // Tile-slot strides carry one word of padding: the row width is a
  // multiple of the bank count, so unpadded slots would land the TE-wide
  // inner reads of the weight contraction on a single bank.
  const int frow_p = frow + 1;
  const int slg_p = n_focus * lg + 1;

  extern __shared__ char smem_raw[];
  acc_t* u_a = reinterpret_cast<acc_t*>(smem_raw);  // TE * (F * ROW + 1)
  acc_t* u_b = u_a + TE * frow_p;                   // TE * (F * ROW + 1)
  acc_t* sig = u_b + TE * frow_p;                   // TE * (F * L*CF + 1)
  acc_t* alp = sig + TE * slg_p;                    // TE * F

  // === Phase R. Rotate + radial degree mixing into shared memory ===
  for (int slot = threadIdx.x; slot < TE * c_wide; slot += blockDim.x) {
    const int e = slot / c_wide;
    const int c = slot % c_wide;
    const long edge = edge0 + e;
    const int f = c / cf;
    const int cfi = c % cf;
    acc_t* ub = u_a + e * frow_p + f * row_w + cfi;
    if (e >= n_here) {
      // Inactive tile slots hold zeros so the uniform phases below stay
      // NaN-free; nothing of theirs is ever written back.
      for (int o = 0; o < RED; ++o) {
        ub[o * cf] = acc_t(0);
      }
      continue;
    }
    const long s = src[edge];
    const scalar_t* xb = x + s * x_sn + c;
    const scalar_t* db = wig + edge * DIM * DIM;
    acc_t xr[DIM];
#pragma unroll
    for (int r = 0; r < DIM; ++r) {
      xr[r] = (acc_t)xb[r * x_sd];
    }
    acc_t xl[RED];
#pragma unroll
    for (int l = 0; l <= L; ++l) {
      const int base = l * l;
      const int r0 = base + l;
      acc_t a0 = 0, am = 0, ap = 0;
#pragma unroll
      for (int j = 0; j < 2 * l + 1; ++j) {
        const acc_t xv = xr[base + j];
        a0 += (acc_t)db[r0 * DIM + base + j] * xv;
        if (l >= 1) {
          am += (acc_t)db[(r0 - 1) * DIM + base + j] * xv;
          ap += (acc_t)db[(r0 + 1) * DIM + base + j] * xv;
        }
      }
      xl[l] = a0;
      if (l >= 1) {
        xl[NS0 + l - 1] = am;
        xl[NS0 + L + l - 1] = ap;
      }
    }
    if (RANK == 0) {
      const scalar_t* rad = kc + edge * (long)NS0 * c_wide + c;
#pragma unroll
      for (int o = 0; o < NS0; ++o) {
        ub[o * cf] = xl[o] * (acc_t)rad[o * (long)c_wide];
      }
#pragma unroll
      for (int o = 0; o < L; ++o) {
        const acc_t r = (acc_t)rad[(o + 1) * (long)c_wide];
        ub[(NS0 + o) * cf] = xl[NS0 + o] * r;
        ub[(NS0 + L + o) * cf] = xl[NS0 + L + o] * r;
      }
    } else {
      acc_t cbv[RANK > 0 ? RANK : 1];
#pragma unroll
      for (int t = 0; t < RANK; ++t) {
        cbv[t] = (acc_t)cb[t * (long)c_wide + c];
      }
      const scalar_t* kb = kc + edge * (long)(NS0 * NS0 + L * L) * RANK;
#pragma unroll
      for (int o = 0; o < NS0; ++o) {
        acc_t acc = 0;
#pragma unroll
        for (int i = 0; i < NS0; ++i) {
          if (RANK == 1) {
            acc += (acc_t)kb[i * NS0 + o] * xl[i];
          } else {
            acc_t keff = 0;
#pragma unroll
            for (int t = 0; t < RANK; ++t) {
              keff += (acc_t)kb[(i * NS0 + o) * RANK + t] * cbv[t];
            }
            acc += keff * xl[i];
          }
        }
        if (RANK == 1) {
          acc *= cbv[0];
        }
        ub[o * cf] = acc;
      }
#pragma unroll
      for (int o = 0; o < L; ++o) {
        acc_t an = 0, aq = 0;
#pragma unroll
        for (int i = 0; i < L; ++i) {
          if (RANK == 1) {
            const acc_t k = (acc_t)kb[NS0 * NS0 + i * L + o];
            an += k * xl[NS0 + i];
            aq += k * xl[NS0 + L + i];
          } else {
            acc_t keff = 0;
#pragma unroll
            for (int t = 0; t < RANK; ++t) {
              keff += (acc_t)kb[(NS0 * NS0 + i * L + o) * RANK + t] * cbv[t];
            }
            an += keff * xl[NS0 + i];
            aq += keff * xl[NS0 + L + i];
          }
        }
        if (RANK == 1) {
          an *= cbv[0];
          aq *= cbv[0];
        }
        ub[(NS0 + o) * cf] = an;
        ub[(NS0 + L + o) * cf] = aq;
      }
    }
  }
  __syncthreads();

  // === Phase A. Cross-focus competition weight from the l = 0 scalars ===
  // One warp owns one (tile slot, focus) pair: the lanes stride the scalar
  // channels and reduce by shuffles, so no block-wide barrier sits between
  // the pairs.
  {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_warps = (int)(blockDim.x >> 5);
    for (int pair = warp; pair < TE * n_focus; pair += n_warps) {
      const int e = pair / n_focus;
      const int f = pair % n_focus;
      if (e >= n_here) {
        continue;
      }
      const long edge = edge0 + e;
      if (!apply_alpha) {
        if (lane == 0) {
          alp[e * n_focus + f] = acc_t(1);
          alpha_out[edge * n_focus + f] = acc_t(1);
        }
        continue;
      }
      // Lane-strided dot of the scalar row with the head column, then the
      // full softmax evaluated redundantly per pair (n_focus is at most 4).
      acc_t logits[kMaxFocus];
      for (int g = 0; g < n_focus; ++g) {
        acc_t part = 0;
        for (int i = lane; i < cf; i += 32) {
          part += u_a[e * frow_p + g * row_w + i] *
                  (acc_t)w_fc[(long)i * n_focus + g];
        }
        for (int off = 16; off > 0; off >>= 1) {
          part += __shfl_down_sync(0xffffffff, part, off);
        }
        logits[g] = part;
      }
      if (lane == 0) {
        acc_t mx = acc_t(-1e30);
        for (int g = 0; g < n_focus; ++g) {
          if (has_bias) {
            logits[g] += (acc_t)fc_bias[g];
          }
          logits[g] *= (acc_t)inv_tau;
          mx = max(mx, logits[g]);
        }
        acc_t denom = 0;
        for (int g = 0; g < n_focus; ++g) {
          logits[g] = exp_a(logits[g] - mx);
          denom += logits[g];
        }
        const acc_t a = logits[f] / denom * (acc_t(1) - (acc_t)label_smooth) +
                        (acc_t)label_smooth / (acc_t)n_focus;
        alp[e * n_focus + f] = a;
        alpha_out[edge * n_focus + f] = a;
      }
    }
  }
  __syncthreads();

  // === Phase M. Gated mixing layers, activations resident in shared memory
  // (u_cur -> u_nxt double buffer; the pre-activations stream to z_all).
  // Every weight column is read once and contracted against all TE tile
  // slots in registers. ===
  acc_t* u_cur = u_a;
  acc_t* u_nxt = u_b;
  for (int layer = 0; layer < n_gated; ++layer) {
    const scalar_t* w0 = w0_all + (long)layer * n_focus * m0 * m0;
    const scalar_t* w1 =
        w1_all + (long)layer * n_focus * (row_w - m0) * (row_w - m0);
    const scalar_t* gw = gw_all + (long)layer * n_focus * cf * lg;
    scalar_t* z_l = z_all + (long)layer * n_focus * n_edge * row_w;

    // Scalar block first: the gates need it.
    for (int col = threadIdx.x; col < n_focus * cf; col += blockDim.x) {
      const int f = col / cf;
      const int o = col % cf;
      const scalar_t* w0f = w0 + (long)f * m0 * m0;
      acc_t acc[TE];
#pragma unroll
      for (int e = 0; e < TE; ++e) {
        acc[e] = 0;
      }
#pragma unroll 4
      for (int i = 0; i < m0; ++i) {
        const acc_t w = (acc_t)w0f[(long)i * m0 + o];
#pragma unroll
        for (int e = 0; e < TE; ++e) {
          acc[e] += u_cur[e * frow_p + f * row_w + i] * w;
        }
      }
#pragma unroll
      for (int e = 0; e < TE; ++e) {
        if (e < n_here) {
          z_l[(long)f * n_edge * row_w + (edge0 + e) * row_w + o] =
              (scalar_t)acc[e];
        }
        // Staged for the gate projection below.
        u_nxt[e * frow_p + f * row_w + o] = acc[e];
      }
    }
    __syncthreads();

    // Gate sigmoids: q = z_s G, one output lane per (focus, gate column).
    for (int col = threadIdx.x; col < n_focus * lg; col += blockDim.x) {
      const int f = col / lg;
      const int g = col % lg;
      const scalar_t* gwf = gw + (long)f * cf * lg;
      acc_t acc[TE];
#pragma unroll
      for (int e = 0; e < TE; ++e) {
        acc[e] = 0;
      }
#pragma unroll 4
      for (int i = 0; i < cf; ++i) {
        const acc_t w = (acc_t)gwf[(long)i * lg + g];
#pragma unroll
        for (int e = 0; e < TE; ++e) {
          acc[e] += u_nxt[e * frow_p + f * row_w + i] * w;
        }
      }
#pragma unroll
      for (int e = 0; e < TE; ++e) {
        sig[e * slg_p + f * lg + g] = sigmoid_a(acc[e]);
      }
    }
    __syncthreads();

    // Remaining columns: pre-activation GEMV, gate, residual into u_nxt.
    for (int col = threadIdx.x; col < frow; col += blockDim.x) {
      const int f = col / row_w;
      const int o = col % row_w;
      acc_t z[TE];
      if (o < cf) {
#pragma unroll
        for (int e = 0; e < TE; ++e) {
          z[e] = u_nxt[e * frow_p + f * row_w + o];  // staged scalar
        }
      } else if (o < m0) {
        const scalar_t* w0f = w0 + (long)f * m0 * m0;
#pragma unroll
        for (int e = 0; e < TE; ++e) {
          z[e] = 0;
        }
#pragma unroll 4
        for (int i = 0; i < m0; ++i) {
          const acc_t w = (acc_t)w0f[(long)i * m0 + o];
#pragma unroll
          for (int e = 0; e < TE; ++e) {
            z[e] += u_cur[e * frow_p + f * row_w + i] * w;
          }
        }
      } else {
        const int m1 = row_w - m0;
        const scalar_t* w1f = w1 + (long)f * m1 * m1;
        const int o1 = o - m0;
#pragma unroll
        for (int e = 0; e < TE; ++e) {
          z[e] = 0;
        }
#pragma unroll 4
        for (int i = 0; i < m1; ++i) {
          const acc_t w = (acc_t)w1f[(long)i * m1 + o1];
#pragma unroll
          for (int e = 0; e < TE; ++e) {
            z[e] += u_cur[e * frow_p + f * row_w + m0 + i] * w;
          }
        }
      }
#pragma unroll
      for (int e = 0; e < TE; ++e) {
        if (o >= cf && e < n_here) {
          z_l[(long)f * n_edge * row_w + (edge0 + e) * row_w + o] =
              (scalar_t)z[e];
        }
        acc_t act;
        if (o < cf) {
          act = z[e] * sigmoid_a(z[e]);
        } else if (o < m0) {
          act = z[e] * sig[e * slg_p + f * lg + (o - cf)];
        } else {
          act = z[e] * sig[e * slg_p + f * lg + ((o - m0) % lg)];
        }
        u_nxt[e * frow_p + f * row_w + o] =
            u_cur[e * frow_p + f * row_w + o] + act;
      }
    }
    __syncthreads();
    acc_t* t = u_cur;
    u_cur = u_nxt;
    u_nxt = t;
  }

  // === Phase F. Final identity layer, edge-major store with the scale ===
  for (int col = threadIdx.x; col < frow; col += blockDim.x) {
    const int f = col / row_w;
    const int o = col % row_w;
    const scalar_t* w0 =
        w0_all + (long)n_gated * n_focus * m0 * m0 + (long)f * m0 * m0;
    const scalar_t* w1 = w1_all +
                         (long)n_gated * n_focus * (row_w - m0) * (row_w - m0) +
                         (long)f * (row_w - m0) * (row_w - m0);
    acc_t z[TE];
#pragma unroll
    for (int e = 0; e < TE; ++e) {
      z[e] = 0;
    }
    if (o < m0) {
#pragma unroll 4
      for (int i = 0; i < m0; ++i) {
        const acc_t w = (acc_t)w0[(long)i * m0 + o];
#pragma unroll
        for (int e = 0; e < TE; ++e) {
          z[e] += u_cur[e * frow_p + f * row_w + i] * w;
        }
      }
    } else {
      const int m1 = row_w - m0;
      const int o1 = o - m0;
#pragma unroll 4
      for (int i = 0; i < m1; ++i) {
        const acc_t w = (acc_t)w1[(long)i * m1 + o1];
#pragma unroll
        for (int e = 0; e < TE; ++e) {
          z[e] += u_cur[e * frow_p + f * row_w + m0 + i] * w;
        }
      }
    }
#pragma unroll
    for (int e = 0; e < TE; ++e) {
      if (e >= n_here) {
        continue;
      }
      const long edge = edge0 + e;
      const acc_t u = u_cur[e * frow_p + f * row_w + o];
      const acc_t v = (u + z[e]) * alp[e * n_focus + f];
      x_out[(edge * n_focus + f) * (long)row_w + o] = (scalar_t)v;
      u_final[(long)f * n_edge * row_w + edge * row_w + o] = (scalar_t)u;
    }
  }
}

// ---------------------------------------------------------------------------
// Host launcher: rank and tile width switch inside the per-degree unit so
// the device code of each degree is compiled and launched within one
// translation unit (no relocatable device code required).
// ---------------------------------------------------------------------------
template <typename scalar_t, int L>
void launch_so2_value_fwd(const scalar_t* x,
                          const long* src,
                          const scalar_t* wig,
                          const scalar_t* kc,
                          const scalar_t* cb,
                          const scalar_t* w_fc,
                          const scalar_t* fc_bias,
                          const scalar_t* w0_all,
                          const scalar_t* w1_all,
                          const scalar_t* gw_all,
                          scalar_t* x_out,
                          scalar_t* z_all,
                          scalar_t* u_final,
                          typename acc_type<scalar_t>::type* alpha_out,
                          long n_edge,
                          long x_sn,
                          long x_sd,
                          int cf,
                          int n_focus,
                          int n_gated,
                          bool apply_alpha,
                          bool has_bias,
                          float inv_tau,
                          float label_smooth,
                          int rank,
                          int te,
                          long n_blocks,
                          size_t smem_bytes,
                          cudaStream_t stream) {
  auto run = [&](auto rc, auto tc) {
    auto kernel = so2_value_fwd_kernel<scalar_t, L, decltype(rc)::value,
                                       decltype(tc)::value>;
    if (smem_bytes > 48 * 1024) {
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem_bytes);
    }
    kernel<<<n_blocks, 256, smem_bytes, stream>>>(
        x, src, wig, kc, cb, w_fc, fc_bias, w0_all, w1_all, gw_all, x_out,
        z_all, u_final, alpha_out, n_edge, x_sn, x_sd, cf, n_focus, n_gated,
        apply_alpha, has_bias, inv_tau, label_smooth);
  };
  auto by_te = [&](auto rc) {
    switch (te) {
      case 8:
        run(rc, std::integral_constant<int, 8>{});
        break;
      case 4:
        run(rc, std::integral_constant<int, 4>{});
        break;
      case 2:
        run(rc, std::integral_constant<int, 2>{});
        break;
      default:
        run(rc, std::integral_constant<int, 1>{});
    }
  };
  switch (rank) {
#define DPA4_SCT_CASE(R)                     \
  case R:                                    \
    by_te(std::integral_constant<int, R>{}); \
    break;
    DPA4_SCT_CASE(0)
    DPA4_SCT_CASE(1)
    DPA4_SCT_CASE(2)
    DPA4_SCT_CASE(3)
    DPA4_SCT_CASE(4)
#undef DPA4_SCT_CASE
  }
}

}  // namespace dpa4_sezm_kernels
