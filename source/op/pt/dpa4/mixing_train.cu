// SPDX-License-Identifier: LGPL-3.0-or-later
//
// SO(2) mixing stack for SeZM / DPA4 force-loss training.
//
// The stack applies ``n_gated`` gated layers followed by one identity layer
// to the focus-major activation ``u`` of shape (F, E, ROW) with
// ``ROW = (3 lmax + 1) Cf``:
//
//   z_l   = [ u_l[:, :m0] W0_l | u_l[:, m0:] W1_l ]      (block GEMMs)
//   u_l+1 = u_l + act(z_l)                               (gated activation)
//   out   = (u_n + [ u_n[:, :m0] W0_n | u_n[:, m0:] W1_n ]) * alpha
//
// where act applies SiLU to the scalar rows (l = 0) and gates every row of
// degree l >= 1 by the sigmoid of that degree's slice of the per-degree
// projection q = s G of the scalars. Expressed as graph operations the
// training-time backward of this stack materializes several surfaces per layer;
// this operator keeps the whole traversal inside one call. The forward saves
// only the stacked pre-activations ``z_all`` and the final gated activation
// ``u_final``: the backward walks the residual recursion in reverse and
// recovers every layer's input as ``u_l = u_{l+1} - act(z_l)`` from the
// saved pre-activation, so no per-layer activation is stored.
//
// Block GEMMs and the whole-edge weight-gradient contractions run through
// ATen (cuBLAS) on strided views of the (F, E, ROW) buffers -- the m0 / m1
// column blocks are legal strided batched operands, so no repacking copy
// exists anywhere in the traversal. The elementwise bodies run as the CUDA
// kernels below, one thread per (focus, edge, group, channel) site.
//
// The mathematics mirrors the fused Triton operators of
// ``so2_value_path.py`` (`_mixing_stack_reference` and
// ``_mixing_stack_backward_reference`` are the eager ground truths shared by
// both implementations).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "sezm_train_ops.cuh"

namespace {

constexpr int kThreads = 256;

#define DPA4_CHECK_LAUNCH(what)                                           \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

__device__ __forceinline__ float sigmoid_f(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_grad_f(float s, float sig) {
  return sig * (1.0f + s * (1.0f - sig));
}

__device__ __forceinline__ float silu_grad2_f(float s, float sig) {
  return sig * (1.0f - sig) * (2.0f + s * (1.0f - 2.0f * sig));
}

// ---------------------------------------------------------------------------
// Forward gate: u_next = u + act(z) with the gate sigmoids precomputed.
// Thread site (f, e, slot, c); slot 0 covers the scalar rows, slot 1..L the
// gate groups (three rows each).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void mixing_gate_fwd_kernel(const scalar_t* __restrict__ u,
                                       const scalar_t* __restrict__ z,
                                       const float* __restrict__ sig,
                                       scalar_t* __restrict__ u_next,
                                       long total,
                                       int lmax,
                                       int cf) {
  const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }
  const int c = tid % cf;
  const long rest = tid / cf;
  const int slot = rest % (lmax + 1);
  const long fe = rest / (lmax + 1);

  const int row_w = (3 * lmax + 1) * cf;
  const long base = fe * row_w;
  if (slot == 0) {
    const float zs = (float)z[base + c];
    u_next[base + c] = (scalar_t)((float)u[base + c] + zs * sigmoid_f(zs));
    return;
  }
  const int g = slot - 1;
  const float sg = sig[fe * (long)(lmax * cf) + g * cf + c];
  const long r0 = base + (long)(1 + g) * cf + c;
  const long rn = base + (long)(lmax + 1 + g) * cf + c;
  const long rp = base + (long)(2 * lmax + 1 + g) * cf + c;
  u_next[r0] = (scalar_t)((float)u[r0] + (float)z[r0] * sg);
  u_next[rn] = (scalar_t)((float)u[rn] + (float)z[rn] * sg);
  u_next[rp] = (scalar_t)((float)u[rp] + (float)z[rp] * sg);
}

// ---------------------------------------------------------------------------
// Final identity layer: out[e, f, :] = (u + z_id) * alpha[e, f], streaming
// straight into the edge-major output layout.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void mixing_final_kernel(const scalar_t* __restrict__ u,
                                    const scalar_t* __restrict__ z_id,
                                    const scalar_t* __restrict__ alpha,
                                    scalar_t* __restrict__ out,
                                    long total,
                                    long n_edge,
                                    int n_focus,
                                    int row_w,
                                    bool apply_alpha) {
  const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }
  const int r = tid % row_w;
  const long rest = tid / row_w;
  const long e = rest % n_edge;
  const int f = rest / n_edge;

  const long src = ((long)f * n_edge + e) * row_w + r;
  float v = (float)u[src] + (float)z_id[src];
  if (apply_alpha) {
    v *= (float)alpha[e * n_focus + f];
  }
  out[(e * n_focus + f) * (long)row_w + r] = (scalar_t)v;
}

// ---------------------------------------------------------------------------
// Backward gate: pre-activation gradient, gate-logit gradient and the
// recovered layer input, in one pass. The contraction of the logit gradient
// back onto the scalar rows is the caller's batched matmul. The recovered
// input u_prev = u_next - act(z) carries the forward's accumulated rounding;
// the bottom layer, whose exact input is the operator operand, does not
// consume it (see the host loop).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void mixing_gate_bwd_kernel(const scalar_t* __restrict__ g,
                                       const scalar_t* __restrict__ z,
                                       const float* __restrict__ sig,
                                       const scalar_t* __restrict__ u_next,
                                       scalar_t* __restrict__ gz,
                                       scalar_t* __restrict__ glogit,
                                       scalar_t* __restrict__ u_prev,
                                       long total,
                                       int lmax,
                                       int cf) {
  const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }
  const int c = tid % cf;
  const long rest = tid / cf;
  const int slot = rest % (lmax + 1);
  const long fe = rest / (lmax + 1);

  const int row_w = (3 * lmax + 1) * cf;
  const long base = fe * row_w;
  if (slot == 0) {
    const float zs = (float)z[base + c];
    const float gs = (float)g[base + c];
    const float s0 = sigmoid_f(zs);
    gz[base + c] = (scalar_t)(gs * silu_grad_f(zs, s0));
    u_prev[base + c] = (scalar_t)((float)u_next[base + c] - zs * s0);
    return;
  }
  const int gi = slot - 1;
  const float sg = sig[fe * (long)(lmax * cf) + gi * cf + c];
  const long r0 = base + (long)(1 + gi) * cf + c;
  const long rn = base + (long)(lmax + 1 + gi) * cf + c;
  const long rp = base + (long)(2 * lmax + 1 + gi) * cf + c;
  const float g0 = (float)g[r0], gn = (float)g[rn], gp = (float)g[rp];
  const float z0 = (float)z[r0], zn = (float)z[rn], zp = (float)z[rp];
  gz[r0] = (scalar_t)(g0 * sg);
  gz[rn] = (scalar_t)(gn * sg);
  gz[rp] = (scalar_t)(gp * sg);
  u_prev[r0] = (scalar_t)((float)u_next[r0] - z0 * sg);
  u_prev[rn] = (scalar_t)((float)u_next[rn] - zn * sg);
  u_prev[rp] = (scalar_t)((float)u_next[rp] - zp * sg);
  const float grad_sig = g0 * z0 + gn * zn + gp * zp;
  // Stored in the working precision: both consumers are batched matmuls whose
  // inputs are in the working precision anyway.
  glogit[fe * (long)(lmax * cf) + gi * cf + c] =
      (scalar_t)(grad_sig * sg * (1.0f - sg));
}

// ---------------------------------------------------------------------------
// Second order of one gated layer, pointwise part. The layer's first-order
// backward is linear in the incoming gradient; this kernel evaluates the
// adjoint of that map at the replayed linearization point. The effective
// gate-logit cotangent already carries the scalar route (hq + hz_s G, a
// caller-side matmul), and the trailing contraction dz_s += dq G^T is
// likewise the caller's. The adjoint head update dg = J^T(hz) + h runs in
// place on the head buffer: one thread owns one element.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void mixing_2nd_gate_kernel(const scalar_t* __restrict__ hz,
                                       const scalar_t* __restrict__ hq_eff,
                                       const scalar_t* __restrict__ g,
                                       const scalar_t* __restrict__ z,
                                       const float* __restrict__ sig,
                                       scalar_t* __restrict__ head,
                                       scalar_t* __restrict__ dz,
                                       scalar_t* __restrict__ dq,
                                       long total,
                                       int lmax,
                                       int cf) {
  const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }
  const int c = tid % cf;
  const long rest = tid / cf;
  const int slot = rest % (lmax + 1);
  const long fe = rest / (lmax + 1);

  const int row_w = (3 * lmax + 1) * cf;
  const long base = fe * row_w;
  if (slot == 0) {
    const float zs = (float)z[base + c];
    const float gs = (float)g[base + c];
    const float hzs = (float)hz[base + c];
    const float s0 = sigmoid_f(zs);
    head[base + c] =
        (scalar_t)((float)head[base + c] + hzs * silu_grad_f(zs, s0));
    dz[base + c] = (scalar_t)(hzs * gs * silu_grad2_f(zs, s0));
    return;
  }
  const int gi = slot - 1;
  const long q_idx = fe * (long)(lmax * cf) + gi * cf + c;
  const float sg = sig[q_idx];
  const float d_sig = sg * (1.0f - sg);
  const float dd_sig = d_sig * (1.0f - 2.0f * sg);
  const float hq = (float)hq_eff[q_idx];
  const float w = hq * d_sig;

  const long r0 = base + (long)(1 + gi) * cf + c;
  const long rn = base + (long)(lmax + 1 + gi) * cf + c;
  const long rp = base + (long)(2 * lmax + 1 + gi) * cf + c;
  const float g0 = (float)g[r0], gn = (float)g[rn], gp = (float)g[rp];
  const float z0 = (float)z[r0], zn = (float)z[rn], zp = (float)z[rp];
  const float h0 = (float)hz[r0], hn = (float)hz[rn], hp = (float)hz[rp];

  const float sum_gz = g0 * z0 + gn * zn + gp * zp;
  const float sum_hg = h0 * g0 + hn * gn + hp * gp;
  dq[q_idx] = (scalar_t)(sum_hg * d_sig + hq * sum_gz * dd_sig);

  head[r0] = (scalar_t)((float)head[r0] + h0 * sg + w * z0);
  head[rn] = (scalar_t)((float)head[rn] + hn * sg + w * zn);
  head[rp] = (scalar_t)((float)head[rp] + hp * sg + w * zp);
  dz[r0] = (scalar_t)(w * g0);
  dz[rn] = (scalar_t)(w * gn);
  dz[rp] = (scalar_t)(w * gp);
}

// ---------------------------------------------------------------------------
// Head of the second-order adjoint at the final identity layer, fused: one
// block owns one (edge, focus) row, completes h_gbar = h + h W (the GEMM
// half arrives precomputed), stores the edge-major cotangent of the raw
// output gradient (scaled by the competition weight when it was applied),
// and reduces the competition-weight cotangent sum_r h_gbar * grad_out in
// the same pass.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void mixing_2nd_final_kernel(const scalar_t* __restrict__ h,
                                        const scalar_t* __restrict__ h_gbar_w,
                                        const scalar_t* __restrict__ grad_out,
                                        const scalar_t* __restrict__ alpha,
                                        const scalar_t* __restrict__ gg_init,
                                        scalar_t* __restrict__ grad_grad_out,
                                        scalar_t* __restrict__ grad_alpha_in,
                                        long n_edge,
                                        int n_focus,
                                        int row_w,
                                        bool apply_alpha) {
  const long row = blockIdx.x;
  if (row >= n_edge * (long)n_focus) {
    return;
  }
  const long e = row / n_focus;
  const int f = row % n_focus;
  const long fm = ((long)f * n_edge + e) * row_w;
  const long em = row * (long)row_w;
  const float a = apply_alpha ? (float)alpha[row] : 1.0f;
  float acc = 0.0f;
  for (int r = threadIdx.x; r < row_w; r += blockDim.x) {
    const float hb = (float)h[fm + r] + (float)h_gbar_w[fm + r];
    acc += hb * (float)grad_out[em + r];
    // The competition head's curvature on the upstream gradient arrives as
    // an initializer, so the caller never runs a separate addition pass.
    const float init = gg_init != nullptr ? (float)gg_init[em + r] : 0.0f;
    grad_grad_out[em + r] = (scalar_t)(hb * a + init);
  }
  if (!apply_alpha) {
    return;
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, off);
  }
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = acc;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    acc = (threadIdx.x < (int)((blockDim.x + 31) >> 5)) ? warp_sums[threadIdx.x]
                                                        : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (threadIdx.x == 0) {
      grad_alpha_in[row] = (scalar_t)acc;
    }
  }
}

// ---------------------------------------------------------------------------
// Entry-side gradient of the final store: g_edge = grad_out * alpha in the
// focus-major layout. The alpha gradient is a per-(edge, focus) row
// reduction and runs as an ATen sum on the host side, where it maps to a
// single reduction kernel instead of a contended atomic per row element.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void mixing_entry_bwd_kernel(const scalar_t* __restrict__ grad_out,
                                        const scalar_t* __restrict__ alpha,
                                        scalar_t* __restrict__ g_focus,
                                        long total,
                                        long n_edge,
                                        int n_focus,
                                        int row_w,
                                        bool apply_alpha) {
  const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }
  const int r = tid % row_w;
  const long rest = tid / row_w;
  const long e = rest % n_edge;
  const int f = rest / n_edge;

  const long src = (e * n_focus + f) * (long)row_w + r;
  float gv = (float)grad_out[src];
  if (apply_alpha) {
    gv *= (float)alpha[e * n_focus + f];
  }
  g_focus[((long)f * n_edge + e) * row_w + r] = (scalar_t)gv;
}

// ---------------------------------------------------------------------------
// Alpha gradient: grad_alpha[e, f] = sum_r grad_out[e, f, r] * out[e, f, r]
// / alpha[e, f], exact because the final store is a plain scale. One block
// reduces one contiguous (edge, focus) row in fp32.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void mixing_alpha_bwd_kernel(const scalar_t* __restrict__ grad_out,
                                        const scalar_t* __restrict__ x_local,
                                        const scalar_t* __restrict__ alpha,
                                        scalar_t* __restrict__ grad_alpha,
                                        long n_rows,
                                        int row_w) {
  const long row = blockIdx.x;
  if (row >= n_rows) {
    return;
  }
  const scalar_t* g = grad_out + row * (long)row_w;
  const scalar_t* x = x_local + row * (long)row_w;
  float acc = 0.0f;
  for (int r = threadIdx.x; r < row_w; r += blockDim.x) {
    acc += (float)g[r] * (float)x[r];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, off);
  }
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = acc;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    acc = (threadIdx.x < (int)((blockDim.x + 31) >> 5)) ? warp_sums[threadIdx.x]
                                                        : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (threadIdx.x == 0) {
      grad_alpha[row] = (scalar_t)(acc / fmaxf((float)alpha[row], 1e-12f));
    }
  }
}

// ---------------------------------------------------------------------------
// Weight-gradient contraction C[b] = A[b]^T B[b] through cublasLt.
//
// The contraction reduces over the edge count, which dwarfs the output tile
// (e.g. 384x384 over K ~ 1e4); served without workspace, the library
// heuristic degrades to percent-level kernels for such shapes, while its
// top choice with ample workspace is a split-K algorithm within a factor
// ~1.5 of the traffic bound. The heuristic result is cached per shape; the
// benchmark-free top-1 choice keeps the selection deterministic across
// processes, which distributed compilation relies on.
// ---------------------------------------------------------------------------
struct LtShapeKey {
  int m, n, lda, ldb, batch;
  long k, sa, sb;
  int dtype;
  bool operator==(const LtShapeKey& o) const {
    return m == o.m && n == o.n && lda == o.lda && ldb == o.ldb &&
           batch == o.batch && k == o.k && sa == o.sa && sb == o.sb &&
           dtype == o.dtype;
  }
};

struct LtShapeKeyHash {
  size_t operator()(const LtShapeKey& s) const {
    size_t h = (size_t)s.m;
    for (long v : {(long)s.n, (long)s.lda, (long)s.ldb, (long)s.batch, s.k,
                   s.sa, s.sb, (long)s.dtype}) {
      h = h * 1000003u + (size_t)v;
    }
    return h;
  }
};

constexpr size_t kLtWorkspaceBytes = 32u << 20;

// C = A^T B with A viewed as (batch, K, m) and B as (batch, K, n), both with
// unit stride along the last axis; C is contiguous (batch, m, n). Strides and
// leading dimensions are taken from the tensors, so strided column blocks of
// a wider buffer are legal operands without repacking.
void lt_weight_grad(const at::Tensor& A,
                    const at::Tensor& B,
                    at::Tensor& C,
                    cudaStream_t stream) {
  static std::mutex mu;
  static std::unordered_map<LtShapeKey, cublasLtMatmulHeuristicResult_t,
                            LtShapeKeyHash>
      algo_cache;

  // The Lt path is a split-K accelerated fp32-compute contraction; the
  // double form (validation runs) keeps the exact dtype through ATen.
  if (A.scalar_type() == at::kDouble) {
    C.copy_(at::bmm(A.transpose(1, 2), B));
    return;
  }

  const int batch = (int)A.size(0);
  const long K = A.size(1);
  const int m = (int)A.size(2);
  const int n = (int)B.size(2);
  const LtShapeKey key{
      m, n,           (int)A.stride(1), (int)B.stride(1),    batch,
      K, A.stride(0), B.stride(0),      (int)A.scalar_type()};

  const cudaDataType_t ab_type =
      A.scalar_type() == at::kBFloat16
          ? CUDA_R_16BF
          : (A.scalar_type() == at::kHalf ? CUDA_R_16F : CUDA_R_32F);

  cublasLtMatmulDesc_t op;
  TORCH_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
              CUBLAS_STATUS_SUCCESS);
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                 sizeof(ta));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                 sizeof(tb));

  cublasLtMatrixLayout_t la, lb, lc;
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  cublasLtMatrixLayoutCreate(&la, ab_type, K, m, key.lda);
  cublasLtMatrixLayoutCreate(&lb, ab_type, K, n, key.ldb);
  cublasLtMatrixLayoutCreate(&lc, ab_type, m, n, n);
  const long sc = (long)m * n;
  for (auto [layout, stride] :
       {std::pair{la, key.sa}, std::pair{lb, key.sb}, std::pair{lc, sc}}) {
    cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                     &row_order, sizeof(row_order));
    cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batch, sizeof(batch));
    cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
        sizeof(stride));
  }

  cublasLtMatmulHeuristicResult_t algo;
  bool have_algo = false;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = algo_cache.find(key);
    if (it != algo_cache.end()) {
      algo = it->second;
      have_algo = true;
    }
  }
  // A dedicated handle rather than the framework's: the framework couples
  // its handle to its own workspace budget, under which the heuristic
  // refuses every split-K candidate and degrades to the same kernels the
  // contraction is escaping from.
  static cublasLtHandle_t handle = [] {
    cublasLtHandle_t h;
    TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS,
                "cublasLtCreate failed");
    return h;
  }();
  if (!have_algo) {
    // The heuristic's top choice is not reliable across these shapes (on
    // the non-64-aligned widths it picks a small-tile kernel ~1.5x off the
    // best candidate), so the top candidates are timed once on the live
    // operands and the fastest is cached. Every candidate computes the
    // full contraction, so the surviving output is exact regardless of
    // which one ran last.
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    const size_t ws = kLtWorkspaceBytes;
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    constexpr int kMaxCand = 8;
    cublasLtMatmulHeuristicResult_t cands[kMaxCand];
    int n_results = 0;
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        handle, op, la, lb, lc, lc, pref, kMaxCand, cands, &n_results);
    cublasLtMatmulPreferenceDestroy(pref);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS && n_results > 0,
                "cublasLt heuristic found no algorithm for the "
                "weight-gradient shape m=",
                m, " n=", n, " K=", K);
    algo = cands[0];
    if (n_results > 1) {
      auto bench_ws =
          at::empty({(long)kLtWorkspaceBytes}, A.options().dtype(at::kByte));
      const float one = 1.0f, zero = 0.0f;
      cudaEvent_t ev0, ev1;
      cudaEventCreate(&ev0);
      cudaEventCreate(&ev1);
      float best = -1.f;
      for (int cand = 0; cand < n_results; ++cand) {
        const auto run = [&] {
          return cublasLtMatmul(handle, op, &one, A.const_data_ptr(), la,
                                B.const_data_ptr(), lb, &zero, C.data_ptr(), lc,
                                C.data_ptr(), lc, &cands[cand].algo,
                                bench_ws.data_ptr(), bench_ws.numel(), stream);
        };
        if (run() != CUBLAS_STATUS_SUCCESS) {
          continue;
        }
        cudaEventRecord(ev0, stream);
        for (int rep = 0; rep < 3; ++rep) {
          run();
        }
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        if (best < 0.f || ms < best) {
          best = ms;
          algo = cands[cand];
        }
      }
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
    }
    std::lock_guard<std::mutex> lock(mu);
    algo_cache.emplace(key, algo);
  }

  auto workspace =
      at::empty({(long)std::min(algo.workspaceSize, kLtWorkspaceBytes)},
                A.options().dtype(at::kByte));
  const float one = 1.0f, zero = 0.0f;
  const cublasStatus_t st = cublasLtMatmul(
      handle, op, &one, A.const_data_ptr(), la, B.const_data_ptr(), lb, &zero,
      C.data_ptr(), lc, C.data_ptr(), lc, &algo.algo, workspace.data_ptr(),
      workspace.numel(), stream);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed (", (int)st,
              ") for weight-gradient shape m=", m, " n=", n, " K=", K);
}

void check_stack_inputs(const at::Tensor& u0,
                        const at::Tensor& w0_all,
                        const at::Tensor& w1_all,
                        const at::Tensor& gw_all,
                        int64_t lmax,
                        int64_t focus_dim,
                        const char* who) {
  TORCH_CHECK(u0.is_cuda() && u0.dim() == 3, who,
              ": u0 must be (F, E, ROW) on CUDA");
  TORCH_CHECK(u0.size(2) == (3 * lmax + 1) * focus_dim, who,
              ": row width does not match lmax and focus_dim");
  TORCH_CHECK(w0_all.size(0) == gw_all.size(0) + 1 &&
                  w1_all.size(0) == gw_all.size(0) + 1,
              who, ": block weights must carry the final identity layer");
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entries, composed by the fused SO(2) value-path operator.
// ---------------------------------------------------------------------------
namespace dpa4_sezm {

// Forward: (out, z_all, u_final).
std::tuple<at::Tensor, at::Tensor, at::Tensor> mixing_fwd(
    const at::Tensor& u0_in,
    const at::Tensor& alpha,
    const at::Tensor& w0_in,
    const at::Tensor& w1_in,
    const at::Tensor& gw_in,
    int64_t lmax,
    int64_t focus_dim,
    bool apply_alpha) {
  check_stack_inputs(u0_in, w0_in, w1_in, gw_in, lmax, focus_dim,
                     "sezm_mixing_fwd");
  // The elementwise kernels address flat contiguous rows; a caller-side
  // view (a compiled graph may forward one) is materialized here.
  const at::Tensor u0 = u0_in.contiguous();
  const at::Tensor w0_all = w0_in.contiguous();
  const at::Tensor w1_all = w1_in.contiguous();
  const at::Tensor gw_all = gw_in.contiguous();
  const c10::cuda::CUDAGuard guard(u0.device());
  const long n_focus = u0.size(0);
  const long n_edge = u0.size(1);
  const long row_w = u0.size(2);
  const long n_gated = gw_all.size(0);
  const long m0 = (lmax + 1) * focus_dim;
  const long lg = lmax * focus_dim;

  auto z_all = at::empty({n_gated, n_focus, n_edge, row_w}, u0.options());
  auto x_local = at::empty({n_edge, n_focus, row_w}, u0.options());
  if (n_edge == 0) {
    return {x_local, z_all, u0};
  }
  auto sig = at::empty({n_focus, n_edge, lg}, u0.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  const long gate_total = n_focus * n_edge * (lmax + 1) * focus_dim;
  const long gate_blocks = (gate_total + kThreads - 1) / kThreads;

  at::Tensor u = u0;
  at::Tensor u_next;
  for (long layer = 0; layer < n_gated; ++layer) {
    auto z = z_all[layer];
    // Block GEMMs write straight into the saved pre-activation slices.
    auto z0 = z.slice(2, 0, m0);
    auto z1 = z.slice(2, m0, row_w);
    at::bmm_out(z0, u.slice(2, 0, m0), w0_all[layer]);
    at::bmm_out(z1, u.slice(2, m0, row_w), w1_all[layer]);
    // Gate projection on the freshly written scalar rows.
    at::sigmoid_out(sig, at::bmm(z.slice(2, 0, focus_dim), gw_all[layer]));
    u_next = at::empty_like(u);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, u0.scalar_type(), "mixing_gate_fwd", [&] {
          mixing_gate_fwd_kernel<scalar_t>
              <<<gate_blocks, kThreads, 0, stream>>>(
                  u.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(),
                  sig.data_ptr<float>(), u_next.data_ptr<scalar_t>(),
                  gate_total, (int)lmax, (int)focus_dim);
        });
    DPA4_CHECK_LAUNCH("sezm_mixing_fwd gate");
    u = u_next;
  }
  const at::Tensor u_final = u;

  // Final identity layer; its pre-activation is transient.
  auto z_id = at::empty_like(u_final);
  {
    auto zi0 = z_id.slice(2, 0, m0);
    auto zi1 = z_id.slice(2, m0, row_w);
    at::bmm_out(zi0, u_final.slice(2, 0, m0), w0_all[n_gated]);
    at::bmm_out(zi1, u_final.slice(2, m0, row_w), w1_all[n_gated]);
  }
  const long fin_total = n_focus * n_edge * row_w;
  const long fin_blocks = (fin_total + kThreads - 1) / kThreads;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, u0.scalar_type(), "mixing_final", [&] {
        mixing_final_kernel<scalar_t><<<fin_blocks, kThreads, 0, stream>>>(
            u_final.data_ptr<scalar_t>(), z_id.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(), x_local.data_ptr<scalar_t>(), fin_total,
            n_edge, (int)n_focus, (int)row_w, apply_alpha);
      });
  DPA4_CHECK_LAUNCH("sezm_mixing_fwd final");
  return {x_local, z_all, u_final};
}

// ---------------------------------------------------------------------------
// First-order backward. Mirrors ``_mixing_stack_backward_reference``; the
// optional upstream gradients fold the outer graph's cotangents of the saved
// surfaces into the traversal (they arrive materialized under the force-loss
// trace). The optional ``u0`` is the stack input; when supplied, the bottom
// layer's weight gradients and retained input surface use the exact value
// instead of the recovered one. ``with_weights`` selects the weight-gradient
// contractions (the training step needs them, a pure gradient propagation
// does not); ``keep_state`` retains the per-layer surfaces the second order
// linearizes around, in which case every downstream input surface remains a
// rolling buffer but the adjoint heads, pre-activation gradients and gate
// logits stack per layer.
// ---------------------------------------------------------------------------
std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor>
mixing_bwd(const at::Tensor& grad_out_in,
           const at::Tensor& x_local_in,
           const at::Tensor& z_all_in,
           const at::Tensor& u_final_in,
           const at::Tensor& alpha_in,
           const at::Tensor& w0t_in,
           const at::Tensor& w1t_in,
           const at::Tensor& gw_in,
           const at::Tensor& gwt_in,
           const c10::optional<at::Tensor>& u0_in,
           const c10::optional<at::Tensor>& grad_z_up_in,
           const c10::optional<at::Tensor>& grad_u_up_in,
           int64_t lmax,
           int64_t focus_dim,
           bool apply_alpha,
           bool with_weights,
           bool keep_state) {
  // The elementwise kernels address flat contiguous rows; caller-side views
  // (a compiled graph may forward them) are materialized here.
  const at::Tensor grad_out = grad_out_in.contiguous();
  const at::Tensor x_local = x_local_in.contiguous();
  const at::Tensor z_all = z_all_in.contiguous();
  const at::Tensor u_final = u_final_in.contiguous();
  const at::Tensor alpha = alpha_in.contiguous();
  // The transposed weights feed batched matmuls only, which consume the
  // strided transpose views directly; materializing them would copy every
  // block weight once per backward call.
  const at::Tensor w0t_all = w0t_in;
  const at::Tensor w1t_all = w1t_in;
  const at::Tensor gw_all = gw_in.contiguous();
  const at::Tensor gwt_all = gwt_in;
  const c10::optional<at::Tensor> u0 =
      u0_in.has_value() ? c10::optional<at::Tensor>(u0_in->contiguous())
                        : c10::nullopt;
  const c10::optional<at::Tensor> grad_z_up =
      grad_z_up_in.has_value()
          ? c10::optional<at::Tensor>(grad_z_up_in->contiguous())
          : c10::nullopt;
  const c10::optional<at::Tensor> grad_u_up =
      grad_u_up_in.has_value()
          ? c10::optional<at::Tensor>(grad_u_up_in->contiguous())
          : c10::nullopt;
  const c10::cuda::CUDAGuard guard(u_final.device());
  const long n_focus = u_final.size(0);
  const long n_edge = u_final.size(1);
  const long row_w = u_final.size(2);
  const long n_gated = gw_all.size(0);
  const long m0 = (lmax + 1) * focus_dim;
  const long lg = lmax * focus_dim;
  auto stream = at::cuda::getCurrentCUDAStream();

  auto grad_w0 = with_weights ? at::empty(w0t_all.sizes(), w0t_all.options())
                              : at::empty({0}, w0t_all.options());
  auto grad_w1 = with_weights ? at::empty(w1t_all.sizes(), w1t_all.options())
                              : at::empty({0}, w1t_all.options());
  auto grad_gw =
      with_weights ? at::empty_like(gw_all) : at::empty({0}, gw_all.options());
  // The stacked surfaces exist only for a following second order; the
  // recovered inputs are consumed in place, so the input surface stays a
  // rolling pair of buffers.
  const long n_keep = keep_state ? n_gated : 0;
  auto upstream_all =
      at::empty({n_keep, n_focus, n_edge, row_w}, u_final.options());
  auto input_all = at::empty({0, n_focus, n_edge, row_w}, u_final.options());
  auto grad_z_all =
      at::empty({keep_state ? n_gated : std::min<long>(n_gated, 1), n_focus,
                 n_edge, row_w},
                u_final.options());
  auto grad_logit_all = at::empty(
      {keep_state ? n_gated : std::min<long>(n_gated, 1), n_focus, n_edge, lg},
      u_final.options());
  // The competition-weight gradient feeds the head's closed form, whose
  // gate-slice term enters the input gradient; it is therefore computed
  // whenever the competition is active, independent of the weight
  // contractions.
  auto grad_alpha = at::empty({n_edge, n_focus}, u_final.options());
  if (apply_alpha) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, u_final.scalar_type(), "mixing_alpha_bwd",
        [&] {
          mixing_alpha_bwd_kernel<scalar_t>
              <<<n_edge * n_focus, kThreads, 0, stream>>>(
                  grad_out.data_ptr<scalar_t>(), x_local.data_ptr<scalar_t>(),
                  alpha.data_ptr<scalar_t>(), grad_alpha.data_ptr<scalar_t>(),
                  n_edge * n_focus, (int)row_w);
        });
    DPA4_CHECK_LAUNCH("sezm_mixing_bwd alpha");
  } else {
    grad_alpha.zero_();
  }

  // === Entry: undo the competition scale and the edge-major store ===
  auto g_focus = at::empty({n_focus, n_edge, row_w}, u_final.options());
  {
    const long total = n_focus * n_edge * row_w;
    const long blocks = (total + kThreads - 1) / kThreads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, u_final.scalar_type(), "mixing_entry_bwd",
        [&] {
          mixing_entry_bwd_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
              grad_out.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(),
              g_focus.data_ptr<scalar_t>(), total, n_edge, (int)n_focus,
              (int)row_w, apply_alpha);
        });
    DPA4_CHECK_LAUNCH("sezm_mixing_bwd entry");
  }

  // === Final identity layer: dgrad and its weight gradient ===
  // The layer heads write straight into their retention slots: the
  // out-of-place residual contraction produces the snapshot the second
  // order linearizes around, so no separate copy exists.
  at::Tensor g0 = g_focus.slice(2, 0, m0);
  at::Tensor g1 = g_focus.slice(2, m0, row_w);
  auto head_buf = at::empty({n_focus, n_edge, row_w}, u_final.options());
  at::Tensor g_cur =
      (keep_state && n_gated > 0) ? upstream_all[n_gated - 1] : head_buf;
  {
    auto gc0 = g_cur.slice(2, 0, m0);
    auto gc1 = g_cur.slice(2, m0, row_w);
    at::baddbmm_out(gc0, g0, g0, w0t_all[n_gated]);
    at::baddbmm_out(gc1, g1, g1, w1t_all[n_gated]);
  }
  if (grad_u_up.has_value()) {
    g_cur.add_(grad_u_up.value());
  }
  if (with_weights) {
    auto gw0_last = grad_w0[n_gated];
    auto gw1_last = grad_w1[n_gated];
    lt_weight_grad(u_final.slice(2, 0, m0), g0, gw0_last, stream);
    lt_weight_grad(u_final.slice(2, m0, row_w), g1, gw1_last, stream);
  }

  // === Gated layers in reverse ===
  auto sig =
      at::empty({n_focus, n_edge, lg}, u_final.options().dtype(at::kFloat));
  // Two buffers alternate: the buffer written two layers ago is no longer
  // referenced once its layer's contractions are done, so the recovery
  // ping-pongs between them.
  at::Tensor u_ping, u_pong;
  if (n_gated > 0) {
    u_ping = at::empty({n_focus, n_edge, row_w}, u_final.options());
    u_pong = at::empty({n_focus, n_edge, row_w}, u_final.options());
  }
  const long gate_total = n_focus * n_edge * (lmax + 1) * focus_dim;
  const long gate_blocks = (gate_total + kThreads - 1) / kThreads;
  at::Tensor u_next = u_final;
  for (long layer = n_gated - 1; layer >= 0; --layer) {
    auto z = z_all[layer];
    at::sigmoid_out(sig, at::bmm(z.slice(2, 0, focus_dim), gw_all[layer]));
    auto gz = grad_z_all[keep_state ? layer : 0];
    auto glogit = grad_logit_all[keep_state ? layer : 0];
    at::Tensor u_prev = ((n_gated - 1 - layer) % 2 == 0) ? u_ping : u_pong;
    // The bottom layer's input is the stack input itself: when the caller
    // supplies it, the exact value replaces the recovered one, whose error
    // is the sum of the forward's per-layer rounding and grows with depth.
    const bool exact_bottom = (layer == 0 && u0.has_value());
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, u_final.scalar_type(), "mixing_gate_bwd",
        [&] {
          mixing_gate_bwd_kernel<scalar_t>
              <<<gate_blocks, kThreads, 0, stream>>>(
                  g_cur.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(),
                  sig.data_ptr<float>(), u_next.data_ptr<scalar_t>(),
                  gz.data_ptr<scalar_t>(), glogit.data_ptr<scalar_t>(),
                  u_prev.data_ptr<scalar_t>(), gate_total, (int)lmax,
                  (int)focus_dim);
        });
    DPA4_CHECK_LAUNCH("sezm_mixing_bwd gate");
    // Fold the gate-logit contraction back onto the scalar rows.
    {
      auto gz_s = gz.slice(2, 0, focus_dim);
      gz_s.baddbmm_(glogit, gwt_all[layer]);
    }
    if (grad_z_up.has_value()) {
      gz.add_(grad_z_up.value()[layer]);
    }
    if (with_weights) {
      // Weight gradients contract the layer input against gz.
      const at::Tensor u_in = exact_bottom ? u0.value() : u_prev;
      auto gw0_l = grad_w0[layer];
      auto gw1_l = grad_w1[layer];
      auto ggw_l = grad_gw[layer];
      lt_weight_grad(u_in.slice(2, 0, m0), gz.slice(2, 0, m0), gw0_l, stream);
      lt_weight_grad(u_in.slice(2, m0, row_w), gz.slice(2, m0, row_w), gw1_l,
                     stream);
      lt_weight_grad(z.slice(2, 0, focus_dim), glogit, ggw_l, stream);
    }
    // Residual recursion: g_{l-1} = g_l + gz W^T, written out of place into
    // the next head's retention slot (or the rolling buffer), which is what
    // makes the per-layer snapshot free.
    {
      at::Tensor g_next =
          (keep_state && layer > 0) ? upstream_all[layer - 1] : head_buf;
      auto gn0 = g_next.slice(2, 0, m0);
      auto gn1 = g_next.slice(2, m0, row_w);
      at::baddbmm_out(gn0, g_cur.slice(2, 0, m0), gz.slice(2, 0, m0),
                      w0t_all[layer]);
      at::baddbmm_out(gn1, g_cur.slice(2, m0, row_w), gz.slice(2, m0, row_w),
                      w1t_all[layer]);
      g_cur = g_next;
    }
    u_next = u_prev;
  }
  return {g_cur,        grad_alpha, grad_w0,    grad_w1,       grad_gw,
          upstream_all, input_all,  grad_z_all, grad_logit_all};
}

// ---------------------------------------------------------------------------
// Second order of the training backward, for the force-loss regime: the
// cotangents of the weight-gradient outputs are absent (parameter gradients
// feed the optimizer, not the force), so the input-recovery routes vanish
// and the adjoint reduces to the head recursion plus per-layer pointwise
// second orders. Replays the first order for its linearization points --
// whose input gradient rides along as the last output -- then walks the
// layers first to last.
// ---------------------------------------------------------------------------
std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor>
mixing_bwd2(const at::Tensor& grad_out_in,
            const at::Tensor& x_local_in,
            const at::Tensor& z_all_in,
            const at::Tensor& u_final_in,
            const at::Tensor& alpha_in,
            const at::Tensor& w0t_in,
            const at::Tensor& w1t_in,
            const at::Tensor& gw_in,
            const at::Tensor& gwt_in,
            const c10::optional<at::Tensor>& u0,
            const at::Tensor& h_u0_in,
            const c10::optional<at::Tensor>& h_alpha,
            const c10::optional<at::Tensor>& grad_z_up_in,
            const c10::optional<at::Tensor>& grad_u_up_in,
            const c10::optional<at::Tensor>& kept_upstream,
            const c10::optional<at::Tensor>& kept_grad_z,
            const c10::optional<at::Tensor>& kept_grad_logit,
            const c10::optional<at::Tensor>& ggout_init,
            int64_t lmax,
            int64_t focus_dim,
            bool apply_alpha) {
  // See mixing_bwd for the layout contract.
  const at::Tensor grad_out = grad_out_in.contiguous();
  const at::Tensor x_local = x_local_in.contiguous();
  const at::Tensor z_all = z_all_in.contiguous();
  const at::Tensor u_final = u_final_in.contiguous();
  const at::Tensor alpha = alpha_in.contiguous();
  // The transposed weights feed batched matmuls only, which consume the
  // strided transpose views directly; materializing them would copy every
  // block weight once per backward call.
  const at::Tensor w0t_all = w0t_in;
  const at::Tensor w1t_all = w1t_in;
  const at::Tensor gw_all = gw_in.contiguous();
  const at::Tensor gwt_all = gwt_in;
  const at::Tensor h_u0 = h_u0_in.contiguous();
  const c10::optional<at::Tensor> grad_z_up =
      grad_z_up_in.has_value()
          ? c10::optional<at::Tensor>(grad_z_up_in->contiguous())
          : c10::nullopt;
  const c10::optional<at::Tensor> grad_u_up =
      grad_u_up_in.has_value()
          ? c10::optional<at::Tensor>(grad_u_up_in->contiguous())
          : c10::nullopt;
  const c10::cuda::CUDAGuard guard(u_final.device());
  const long n_focus = u_final.size(0);
  const long n_edge = u_final.size(1);
  const long row_w = u_final.size(2);
  const long n_gated = gw_all.size(0);
  const long m0 = (lmax + 1) * focus_dim;
  auto stream = at::cuda::getCurrentCUDAStream();

  // === Linearization points: kept by the first order, or replayed ===
  // When the first-order backward retained its per-layer surfaces (force
  // regime, where the second differentiation is known to follow), they
  // arrive directly and the whole replay disappears; otherwise the
  // traversal is replayed here without the weight contractions. The
  // replayed input gradient rides along as the last output so a caller
  // needing both differentiations pays for one traversal either way.
  at::Tensor grad_u0_first, upstream_all, grad_z_all, grad_logit_all;
  if (kept_upstream.has_value() && kept_grad_z.has_value() &&
      kept_grad_logit.has_value()) {
    upstream_all = kept_upstream.value();
    grad_z_all = kept_grad_z.value();
    grad_logit_all = kept_grad_logit.value();
    grad_u0_first = at::empty({0}, u_final.options());
  } else {
    auto replay =
        mixing_bwd(grad_out, x_local, z_all, u_final, alpha, w0t_all, w1t_all,
                   gw_all, gwt_all, u0, grad_z_up, grad_u_up, lmax, focus_dim,
                   apply_alpha, /*with_weights=*/false,
                   /*keep_state=*/true);
    grad_u0_first = std::get<0>(replay);
    upstream_all = std::get<5>(replay);
    grad_z_all = std::get<7>(replay);
    grad_logit_all = std::get<8>(replay);
  }

  auto grad_z_out = at::empty_like(z_all);
  auto grad_gw_out = at::empty_like(gw_all);
  auto grad_w0t_out = at::empty(w0t_all.sizes(), w0t_all.options());
  auto grad_w1t_out = at::empty(w1t_all.sizes(), w1t_all.options());
  auto grad_gz_up = grad_z_up.has_value() ? at::empty_like(z_all)
                                          : at::empty({0}, z_all.options());

  auto h = h_u0.clone();
  auto hgz = at::empty({n_focus, n_edge, row_w}, u_final.options());
  auto dq = at::empty({n_focus, n_edge, lmax * focus_dim}, u_final.options());
  auto ggw_tmp = at::empty_like(grad_gw_out[0]);
  auto sig = at::empty({n_focus, n_edge, lmax * focus_dim},
                       u_final.options().dtype(at::kFloat));
  const long gate_total = n_focus * n_edge * (lmax + 1) * focus_dim;
  const long gate_blocks = (gate_total + kThreads - 1) / kThreads;

  // === Adjoint traversal, first gated layer to last ===
  for (long layer = 0; layer < n_gated; ++layer) {
    auto z = z_all[layer];
    auto gz = grad_z_all[layer];
    auto glogit = grad_logit_all[layer];
    at::sigmoid_out(sig, at::bmm(z.slice(2, 0, focus_dim), gw_all[layer]));

    // Cotangent of the pre-activation gradient: the residual contraction.
    {
      auto hgz0 = hgz.slice(2, 0, m0);
      auto hgz1 = hgz.slice(2, m0, row_w);
      at::bmm_out(hgz0, h.slice(2, 0, m0), w0t_all[layer].transpose(1, 2));
      at::bmm_out(hgz1, h.slice(2, m0, row_w), w1t_all[layer].transpose(1, 2));
    }
    if (grad_z_up.has_value()) {
      grad_gz_up[layer].copy_(hgz);
    }
    // Effective gate-logit cotangent: the scalar route of the first order's
    // external fold.
    auto hq_eff = at::bmm(hgz.slice(2, 0, focus_dim), gw_all[layer]);

    // The residual contraction's weight route, against the pre-update head.
    {
      auto gw0_l = grad_w0t_out[layer];
      auto gw1_l = grad_w1t_out[layer];
      lt_weight_grad(gz.slice(2, 0, m0), h.slice(2, 0, m0), gw0_l, stream);
      lt_weight_grad(gz.slice(2, m0, row_w), h.slice(2, m0, row_w), gw1_l,
                     stream);
    }

    // Pointwise second order; the head update runs in place.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, u_final.scalar_type(), "mixing_2nd_gate",
        [&] {
          mixing_2nd_gate_kernel<scalar_t>
              <<<gate_blocks, kThreads, 0, stream>>>(
                  hgz.data_ptr<scalar_t>(), hq_eff.data_ptr<scalar_t>(),
                  upstream_all[layer].data_ptr<scalar_t>(),
                  z.data_ptr<scalar_t>(), sig.data_ptr<float>(),
                  h.data_ptr<scalar_t>(),
                  grad_z_out[layer].data_ptr<scalar_t>(),
                  dq.data_ptr<scalar_t>(), gate_total, (int)lmax,
                  (int)focus_dim);
        });
    DPA4_CHECK_LAUNCH("sezm_mixing_bwd2 gate");

    // Trailing contractions of the pointwise second order.
    {
      auto dz_s = grad_z_out[layer].slice(2, 0, focus_dim);
      dz_s.baddbmm_(dq, gwt_all[layer]);
      auto ggw_l = grad_gw_out[layer];
      lt_weight_grad(z.slice(2, 0, focus_dim), dq, ggw_l, stream);
      lt_weight_grad(hgz.slice(2, 0, focus_dim), glogit, ggw_tmp, stream);
      ggw_l.add_(ggw_tmp);
    }
  }

  // The upstream final-activation gradient joined the head additively.
  auto grad_gu_up =
      grad_u_up.has_value() ? h.clone() : at::empty({0}, h.options());

  // === Final identity layer and the competition scale ===
  // The GEMM half of h_gbar = h + h W; the residual add, the edge-major
  // store and the competition-weight reduction fuse into one kernel below.
  auto h_gbar_w = at::empty_like(h);
  {
    auto hb0 = h_gbar_w.slice(2, 0, m0);
    auto hb1 = h_gbar_w.slice(2, m0, row_w);
    at::bmm_out(hb0, h.slice(2, 0, m0), w0t_all[n_gated].transpose(1, 2));
    at::bmm_out(hb1, h.slice(2, m0, row_w), w1t_all[n_gated].transpose(1, 2));
  }
  // grad_final = (grad_out * alpha) in the focus-major layout.
  auto grad_final = at::empty({n_focus, n_edge, row_w}, u_final.options());
  {
    const long total = n_focus * n_edge * row_w;
    const long blocks = (total + kThreads - 1) / kThreads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, u_final.scalar_type(), "mixing_bwd2_entry",
        [&] {
          mixing_entry_bwd_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
              grad_out.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(),
              grad_final.data_ptr<scalar_t>(), total, n_edge, (int)n_focus,
              (int)row_w, apply_alpha);
        });
    DPA4_CHECK_LAUNCH("sezm_mixing_bwd2 entry");
  }
  {
    auto gw0_n = grad_w0t_out[n_gated];
    auto gw1_n = grad_w1t_out[n_gated];
    lt_weight_grad(grad_final.slice(2, 0, m0), h.slice(2, 0, m0), gw0_n,
                   stream);
    lt_weight_grad(grad_final.slice(2, m0, row_w), h.slice(2, m0, row_w), gw1_n,
                   stream);
  }

  auto grad_grad_out = at::empty({n_edge, n_focus, row_w}, grad_out.options());
  auto grad_alpha_in =
      at::empty({apply_alpha ? n_edge : 0, n_focus}, grad_out.options());
  {
    const at::Tensor gg_init =
        ggout_init.has_value() ? ggout_init->contiguous() : at::Tensor();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, u_final.scalar_type(), "mixing_2nd_final",
        [&] {
          mixing_2nd_final_kernel<scalar_t>
              <<<n_edge * n_focus, kThreads, 0, stream>>>(
                  h.data_ptr<scalar_t>(), h_gbar_w.data_ptr<scalar_t>(),
                  grad_out.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(),
                  gg_init.defined() ? gg_init.data_ptr<scalar_t>() : nullptr,
                  grad_grad_out.data_ptr<scalar_t>(),
                  grad_alpha_in.data_ptr<scalar_t>(), n_edge, (int)n_focus,
                  (int)row_w, apply_alpha);
        });
    DPA4_CHECK_LAUNCH("sezm_mixing_bwd2 final");
  }

  at::Tensor grad_u_final;
  if (apply_alpha && h_alpha.has_value()) {
    // ``grad_alpha`` contracted the raw cotangent against the unscaled
    // output; both factors receive its cotangent in turn.
    auto y_fm = at::empty_like(u_final);
    {
      auto y0 = y_fm.slice(2, 0, m0);
      auto y1 = y_fm.slice(2, m0, row_w);
      at::bmm_out(y0, u_final.slice(2, 0, m0),
                  w0t_all[n_gated].transpose(1, 2));
      at::bmm_out(y1, u_final.slice(2, m0, row_w),
                  w1t_all[n_gated].transpose(1, 2));
      y_fm.add_(u_final);
    }
    auto ha = h_alpha.value().unsqueeze(-1);
    grad_grad_out = grad_grad_out + ha * y_fm.permute({1, 0, 2});
    auto v = (ha * grad_out).permute({1, 0, 2}).contiguous();
    auto hu = at::empty_like(v);
    {
      auto hu0 = hu.slice(2, 0, m0);
      auto hu1 = hu.slice(2, m0, row_w);
      at::bmm_out(hu0, v.slice(2, 0, m0), w0t_all[n_gated]);
      at::bmm_out(hu1, v.slice(2, m0, row_w), w1t_all[n_gated]);
      hu.add_(v);
    }
    grad_u_final = hu;
    {
      auto gw0_n = grad_w0t_out[n_gated];
      auto gw1_n = grad_w1t_out[n_gated];
      auto blk_tmp0 = at::empty_like(gw0_n);
      auto blk_tmp1 = at::empty_like(gw1_n);
      lt_weight_grad(v.slice(2, 0, m0), u_final.slice(2, 0, m0), blk_tmp0,
                     stream);
      lt_weight_grad(v.slice(2, m0, row_w), u_final.slice(2, m0, row_w),
                     blk_tmp1, stream);
      gw0_n.add_(blk_tmp0);
      gw1_n.add_(blk_tmp1);
    }
  } else {
    grad_u_final = at::empty({0}, u_final.options());
  }

  auto grad_u0_in = at::empty({0}, u_final.options());
  return {grad_grad_out.contiguous(),
          grad_z_out,
          grad_u_final,
          grad_alpha_in,
          grad_w0t_out,
          grad_w1t_out,
          grad_gw_out,
          grad_u0_in,
          grad_gz_up,
          grad_gu_up,
          grad_u0_first};
}

}  // namespace dpa4_sezm
