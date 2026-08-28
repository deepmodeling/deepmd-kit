// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Whole SO(2) value path of SeZM / DPA4 as one training operator.
//
// One launch carries an edge from the gathered source features to the final
// edge-major activation:
//
//   1. rotate to the local frame over the structural block-diagonal
//      non-zeros of the Wigner-D matrix (m-major reduced rows |m| <= 1),
//   2. apply the edge-conditioned radial degree mixing,
//   3. form the cross-focus competition weight from the l = 0 scalars
//      (identity pass-through, linear head, tempered softmax, label
//      smoothing),
//   4. run every gated mixing layer (block GEMMs against the stacked
//      weights, sigmoid gates from the scalar rows, SiLU on the scalars,
//      residual accumulation),
//   5. apply the final identity layer and store edge-major, scaled by the
//      competition weight.
//
// The rotated input u0 and every inter-layer activation live in shared
// memory for the lifetime of the block: the only surfaces written to global
// memory are the operator outputs and the backward anchors (the stacked
// pre-activations z_all, the final gated activation u_final, and the
// competition weight alpha). The backward recomputes u0 from x and the
// Wigner matrix, exactly as the standalone rotate-mix backward does.
//
// The attention span downstream of this operator (segmented softmax, flash
// aggregation, head gate) runs as the Triton operator composition inside
// the traced graph, where the compiler fuses it with its neighbours; a
// fused CUDA form of that span was built, measured slower at equal memory,
// and removed (see dpa4_cuda.md section 12).
//
// The mathematics mirrors ``_TritonSO2ValuePath.__call__`` composed of
// ``_rotate_mix_reference``, ``_focus_alpha`` (identity norm) and
// ``_mixing_stack_reference`` in ``so2_value_path.py`` / ``so2.py``.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <tuple>
#include <utility>

#include "sezm_train_ops.cuh"
#include "so2_conv_train/kernels.cuh"

// The forward kernel is instantiated in one generated source per degree and
// dtype; the declarations below keep this host unit free of device code.
#define DPA4_SCT_EXTERN extern
#define DPA4_SCT_L 1
#include "so2_conv_train/instantiate.cuh"
#undef DPA4_SCT_L
#define DPA4_SCT_L 2
#include "so2_conv_train/instantiate.cuh"
#undef DPA4_SCT_L
#define DPA4_SCT_L 3
#include "so2_conv_train/instantiate.cuh"
#undef DPA4_SCT_L
#define DPA4_SCT_L 4
#include "so2_conv_train/instantiate.cuh"
#undef DPA4_SCT_L
#define DPA4_SCT_L 5
#include "so2_conv_train/instantiate.cuh"
#undef DPA4_SCT_L
#define DPA4_SCT_L 6
#include "so2_conv_train/instantiate.cuh"
#undef DPA4_SCT_L
#undef DPA4_SCT_EXTERN

using namespace dpa4_sezm_kernels;

namespace {

constexpr int kWideChannelLanes = 384;

#define DPA4_SC_CHECK_LAUNCH(what)                                        \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

void check_value_inputs(const at::Tensor& x,
                        const at::Tensor& src,
                        const at::Tensor& runs,
                        const at::Tensor& kc,
                        const at::Tensor& w0_all,
                        int64_t lmax,
                        int64_t n_focus,
                        int64_t rank,
                        double softmax_tau,
                        double label_smoothing,
                        const char* who) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.stride(2) == 1, who,
              ": x must be (N, D, C_wide) with unit channel stride");
  TORCH_CHECK(1 <= lmax && lmax <= 6, who, ": unsupported lmax");
  TORCH_CHECK(0 <= rank && rank <= 4, who, ": unsupported rank");
  TORCH_CHECK(1 <= n_focus && n_focus <= kMaxFocus, who,
              ": unsupported focus count");
  TORCH_CHECK(x.size(1) == (lmax + 1) * (lmax + 1), who,
              ": x degree dimension does not match lmax");
  TORCH_CHECK(x.size(2) % n_focus == 0, who,
              ": channel width must split into the focus streams");
  TORCH_CHECK(
      x.size(2) <= kThreads || (lmax == 6 && x.size(2) <= kWideChannelLanes),
      who, ": channel width exceeds the supported block lane count");
  const int64_t dim = (lmax + 1) * (lmax + 1);
  TORCH_CHECK(runs.is_contiguous() && runs.dim() == 2 &&
                  runs.size(0) == src.size(0) && runs.size(1) == 3 * dim - 2,
              who, ": runs must be contiguous (E, 3 * DIM - 2)");
  TORCH_CHECK(src.scalar_type() == at::kLong, who, ": src must be int64");
  TORCH_CHECK(kc.dim() >= 1 && kc.size(0) == src.size(0), who,
              ": degree-kernel edge count must match src");
  TORCH_CHECK(w0_all.dim() == 4, who, ": stacked block weights expected");
  TORCH_CHECK(softmax_tau > 0.0, who, ": softmax_tau must be positive");
  TORCH_CHECK(0.0 <= label_smoothing && label_smoothing < 1.0, who,
              ": label_smoothing must be in [0, 1)");
}

template <typename F>
void dispatch_l_sc(int64_t lmax, const F& f) {
  switch (lmax) {
#define DPA4_SC_L_CASE(L)                \
  case L:                                \
    f(std::integral_constant<int, L>{}); \
    break;
    DPA4_SC_L_CASE(1)
    DPA4_SC_L_CASE(2)
    DPA4_SC_L_CASE(3)
    DPA4_SC_L_CASE(4)
    DPA4_SC_L_CASE(5)
    DPA4_SC_L_CASE(6)
#undef DPA4_SC_L_CASE
    default:
      TORCH_CHECK(false, "sezm_so2_value_fwd: unsupported lmax");
  }
}

// ---------------------------------------------------------------------------
// Competition-head forward. One warp owns one focus of an edge and reduces
// the scalar-channel projection directly from the focus-major rotation output.
// The block then normalizes the at-most-four logits and writes the fp32 softmax
// anchor. This avoids materializing an edge-major fp32 gate surface around a
// one-row contraction.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void competition_fwd_kernel(
    const scalar_t* __restrict__ u0,
    const scalar_t* __restrict__ w_fc,
    const scalar_t* __restrict__ bias,
    typename acc_type<scalar_t>::type* __restrict__ alpha,
    long n_edge,
    int n_focus,
    int cf,
    int row_w,
    float inv_tau,
    float label_smoothing,
    bool has_bias) {
  using acc_t = typename acc_type<scalar_t>::type;
  const long edge = blockIdx.x;
  if (edge >= n_edge) {
    return;
  }

  const int focus = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  __shared__ acc_t logits[kMaxFocus];
  if (focus < n_focus) {
    acc_t logit = 0;
    const scalar_t* gate = u0 + ((long)focus * n_edge + edge) * row_w;
    for (int channel = lane; channel < cf; channel += 32) {
      logit += (acc_t)gate[channel] * (acc_t)w_fc[channel * n_focus + focus];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      logit += __shfl_down_sync(0xffffffff, logit, offset);
    }
    if (lane == 0) {
      if (has_bias) {
        logit += (acc_t)bias[focus];
      }
      logits[focus] = logit * (acc_t)inv_tau;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    acc_t maximum = logits[0];
    for (int f = 1; f < n_focus; ++f) {
      maximum = maximum > logits[f] ? maximum : logits[f];
    }
    acc_t denominator = 0;
    acc_t weights[kMaxFocus];
    for (int f = 0; f < n_focus; ++f) {
      weights[f] = exp((acc_t)(logits[f] - maximum));
      denominator += weights[f];
    }
    const acc_t smooth = (acc_t)label_smoothing / (acc_t)n_focus;
    const acc_t scale = (acc_t)1 - (acc_t)label_smoothing;
    for (int f = 0; f < n_focus; ++f) {
      alpha[edge * (long)n_focus + f] =
          weights[f] / denominator * scale + smooth;
    }
  }
}

// ---------------------------------------------------------------------------
// Competition-head backward. One block owns one edge, reconstructs the
// smoothed softmax derivative in double precision, and immediately consumes
// the logit gradient into the focus-major traversal gradient. The optional
// (E, F) output is retained only for the parameter contractions; no
// (E, F, Cf) gate-gradient surface exists.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void competition_bwd_kernel(
    scalar_t* __restrict__ grad_u0,
    const scalar_t* __restrict__ w_fc,
    const typename acc_type<scalar_t>::type* __restrict__ alpha,
    const typename acc_type<scalar_t>::type* __restrict__ grad_alpha_mix,
    const typename acc_type<scalar_t>::type* __restrict__ h_alpha,
    double* __restrict__ grad_logit,
    long n_edge,
    int n_focus,
    int cf,
    int row_w,
    double inv_tau,
    double label_smoothing) {
  using acc_t = typename acc_type<scalar_t>::type;
  const long edge = blockIdx.x;
  if (edge >= n_edge) {
    return;
  }

  __shared__ double gl_shared[kMaxFocus];
  if (threadIdx.x == 0) {
    double p[kMaxFocus];
    double ga[kMaxFocus];
    double ga_mean = 0.0;
    const double smooth_scale = 1.0 - label_smoothing;
    for (int focus = 0; focus < n_focus; ++focus) {
      const long row = edge * (long)n_focus + focus;
      p[focus] = fmax(((double)alpha[row] - label_smoothing / (double)n_focus) /
                          smooth_scale,
                      0.0);
      ga[focus] = (double)grad_alpha_mix[row] * smooth_scale;
      if (h_alpha != nullptr) {
        ga[focus] += (double)h_alpha[row] * smooth_scale;
      }
      ga_mean += ga[focus] * p[focus];
    }
    for (int focus = 0; focus < n_focus; ++focus) {
      const double gl = (ga[focus] - ga_mean) * p[focus] * inv_tau;
      gl_shared[focus] = gl;
      if (grad_logit != nullptr) {
        grad_logit[edge * (long)n_focus + focus] = gl;
      }
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < n_focus * cf; index += blockDim.x) {
    const int focus = index / cf;
    const int channel = index - focus * cf;
    const long u_index = ((long)focus * n_edge + edge) * row_w + channel;
    const scalar_t gate =
        (scalar_t)(gl_shared[focus] * (double)w_fc[channel * n_focus + focus]);
    grad_u0[u_index] = (scalar_t)((acc_t)grad_u0[u_index] + (acc_t)gate);
  }
}

// ---------------------------------------------------------------------------
// Value span forward (rotation, degree mixing, competition, gated stack)
// ---------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> value_fwd(
    const at::Tensor& x_in,
    const at::Tensor& src,
    const at::Tensor& runs_in,
    const at::Tensor& kc_in,
    const at::Tensor& cb_in,
    const c10::optional<at::Tensor>& w_fc,
    const c10::optional<at::Tensor>& fc_bias,
    const at::Tensor& w0_in,
    const at::Tensor& w1_in,
    const at::Tensor& gw_in,
    int64_t lmax,
    int64_t n_focus,
    int64_t rank,
    bool apply_alpha,
    double softmax_tau,
    double label_smoothing) {
  check_value_inputs(x_in, src, runs_in, kc_in, w0_in, lmax, n_focus, rank,
                     softmax_tau, label_smoothing, "sezm_so2_value_fwd");
  TORCH_CHECK(!apply_alpha || w_fc.has_value(),
              "sezm_so2_value_fwd: competition weights required");
  const c10::cuda::CUDAGuard guard(x_in.device());
  const at::Tensor x = x_in.stride(2) == 1 ? x_in : x_in.contiguous();
  const at::Tensor runs = runs_in.contiguous();
  const at::Tensor kc = kc_in.contiguous();
  const at::Tensor cb = cb_in.contiguous();
  const at::Tensor w0_all = w0_in.contiguous();
  const at::Tensor w1_all = w1_in.contiguous();
  const at::Tensor gw_all = gw_in.contiguous();
  const at::Tensor w_fc_t =
      apply_alpha ? w_fc->contiguous() : at::empty({0}, x.options());
  const bool has_bias = apply_alpha && fc_bias.has_value();
  const at::Tensor fc_bias_t =
      has_bias ? fc_bias->contiguous() : at::empty({0}, x.options());

  const long n_edge = src.size(0);
  const int c_wide = (int)x.size(2);
  const int cf = c_wide / (int)n_focus;
  const long row_w = (3 * lmax + 1) * cf;
  const long n_gated = gw_all.size(0);
  const int lg = (int)lmax * cf;
  // The competition weight is the backward's anchor for the whole head, which
  // reconstructs the softmax from it and divides by it; it therefore leaves
  // the forward in accumulator precision whichever branch produced it.
  const auto alpha_opts =
      x.options().dtype(dpa4_sezm::alpha_dtype(x.scalar_type()));
  const size_t acc_bytes =
      x.scalar_type() == at::kDouble ? sizeof(double) : sizeof(float);
  // Bytes of tile-resident state per edge slot (including the bank-offset
  // padding word per surface); the tile width is the largest power of two
  // whose footprint stays inside the current device's shared-memory window,
  // which keeps the weight traffic amortized over as many register
  // accumulators as the configuration allows.
  const size_t per_edge =
      (size_t)(2 * (n_focus * row_w + 1) + (n_focus * lg + 1) + n_focus) *
      acc_bytes;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const size_t smem_ceiling = std::max(properties->sharedMemPerBlock,
                                       properties->sharedMemPerBlockOptin);
  int te = 8;
  while (te > 1 && (size_t)te * per_edge > smem_ceiling) {
    te >>= 1;
  }
  const bool resident_supported = per_edge <= smem_ceiling;

  // The resident kernel multiplies its arithmetic intensity by the tile
  // width. Where the activation footprint forces the tile below eight
  // edges, the residency also caps the occupancy at one block per
  // multiprocessor, and the plain-FMA interior falls an order of magnitude
  // behind the tensor-core GEMMs. Blackwell reaches the same crossover at a
  // per-focus width of 64: its tensor-core throughput grows faster than the
  // L2 bandwidth serving the resident kernel's scalar contractions. Those
  // shapes run the same value stream as a composition of the rotation kernel,
  // the closed-form competition head and the cuBLAS-backed mixing traversal,
  // producing identical anchor layouts for the shared backward. Double inputs
  // (the parity harnesses' ground truth) stay on the resident kernel whenever
  // the device can hold one edge slot; its accumulators follow the input
  // precision.
  const bool blackwell_wide = properties->major >= 12 && cf >= 64;
  const bool use_composed_path =
      !resident_supported ||
      (x.scalar_type() != at::kDouble && (te < 8 || blackwell_wide));
  if (use_composed_path && n_edge > 0) {
    auto u0 =
        dpa4_sezm::rotate_mix_fwd(x, src, runs, kc, cb, lmax, n_focus, rank);
    at::Tensor alpha_t;
    if (apply_alpha) {
      alpha_t = at::empty({n_edge, n_focus}, alpha_opts);
      auto stream = at::cuda::getCurrentCUDAStream();
      const int threads = 32 * (int)n_focus;
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kBFloat16, at::kHalf, x.scalar_type(), "competition_fwd", [&] {
            using acc_t = typename acc_type<scalar_t>::type;
            competition_fwd_kernel<scalar_t><<<n_edge, threads, 0, stream>>>(
                u0.data_ptr<scalar_t>(), w_fc_t.data_ptr<scalar_t>(),
                fc_bias_t.data_ptr<scalar_t>(), alpha_t.data_ptr<acc_t>(),
                n_edge, (int)n_focus, cf, (int)u0.size(2),
                (float)(1.0 / softmax_tau), (float)label_smoothing, has_bias);
          });
      DPA4_SC_CHECK_LAUNCH("sezm_so2_value_fwd competition");
    } else {
      alpha_t = at::ones({n_edge, n_focus}, alpha_opts);
    }
    auto mix = dpa4_sezm::mixing_fwd(std::move(u0), alpha_t, w0_all, w1_all,
                                     gw_all, lmax, cf, apply_alpha);
    return {std::get<0>(mix), std::get<1>(mix), std::get<2>(mix), alpha_t};
  }

  auto x_out = at::empty({n_edge, n_focus, row_w}, x.options());
  auto z_all = at::empty({n_gated, n_focus, n_edge, row_w}, x.options());
  auto u_final = at::empty({n_focus, n_edge, row_w}, x.options());
  auto alpha = at::empty({n_edge, n_focus}, alpha_opts);
  if (n_edge == 0) {
    return {x_out, z_all, u_final, alpha};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t smem_bytes = (size_t)te * per_edge;
  const long n_blocks = (n_edge + te - 1) / te;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "so2_value_fwd", [&] {
        dispatch_l_sc(lmax, [&](auto lc) {
          launch_so2_value_fwd<scalar_t, decltype(lc)::value>(
              x.data_ptr<scalar_t>(), src.data_ptr<long>(),
              runs.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
              cb.data_ptr<scalar_t>(), w_fc_t.data_ptr<scalar_t>(),
              fc_bias_t.data_ptr<scalar_t>(), w0_all.data_ptr<scalar_t>(),
              w1_all.data_ptr<scalar_t>(), gw_all.data_ptr<scalar_t>(),
              x_out.data_ptr<scalar_t>(), z_all.data_ptr<scalar_t>(),
              u_final.data_ptr<scalar_t>(),
              alpha.data_ptr<typename acc_type<scalar_t>::type>(), n_edge,
              x.stride(0), x.stride(1), cf, (int)n_focus, (int)n_gated,
              apply_alpha, has_bias, (float)(1.0 / softmax_tau),
              (float)label_smoothing, (int)rank, te, n_blocks, smem_bytes,
              stream);
        });
      });
  DPA4_SC_CHECK_LAUNCH("sezm_so2_value_fwd");
  return {x_out, z_all, u_final, alpha};
}

// ---------------------------------------------------------------------------
// First-order backward: the value-path adjoint composed from the traversal
// entries of this library. The rotated input is recomputed (the forward
// never stores it), the mixing traversal runs with its weight contractions,
// the competition head is differentiated in closed form from the stored
// weight, and the rotation gradients reduce over the source CSR view.
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
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor>
value_bwd(const at::Tensor& grad_x_local,
          const at::Tensor& x,
          const at::Tensor& src,
          const at::Tensor& src_order,
          const at::Tensor& src_rowptr,
          const at::Tensor& runs,
          const at::Tensor& kc,
          const at::Tensor& cb,
          const c10::optional<at::Tensor>& w_fc,
          const c10::optional<at::Tensor>& fc_bias,
          const at::Tensor& w0_all,
          const at::Tensor& w1_all,
          const at::Tensor& gw_all,
          const at::Tensor& x_local,
          const at::Tensor& z_all,
          const at::Tensor& u_final,
          const at::Tensor& alpha,
          const c10::optional<at::Tensor>& h_z,
          const c10::optional<at::Tensor>& h_uf,
          const c10::optional<at::Tensor>& h_alpha,
          int64_t lmax,
          int64_t n_focus,
          int64_t rank,
          bool apply_alpha,
          double softmax_tau,
          double label_smoothing,
          bool keep_state,
          bool with_weights) {
  check_value_inputs(x, src, runs, kc, w0_all, lmax, n_focus, rank, softmax_tau,
                     label_smoothing, "sezm_so2_value_bwd");
  TORCH_CHECK(!apply_alpha || w_fc.has_value(),
              "sezm_so2_value_bwd: competition weights required");
  const c10::cuda::CUDAGuard guard(x.device());
  const int cf = (int)(x.size(2) / n_focus);

  // === Step 1. Recompute the rotated input (never stored) ===
  auto u0 =
      dpa4_sezm::rotate_mix_fwd(x, src, runs, kc, cb, lmax, n_focus, rank);

  // === Step 2. Mixing traversal ===
  // Under ``keep_state`` (the force regime, where a second differentiation
  // is known to follow) the traversal retains its per-layer surfaces and
  // this operator returns them together with the total input gradient; the
  // second order then replays nothing. The weight contractions run only
  // when a parameter gradient is requested -- the force pass
  // (``autograd.grad(E, coord)``) differentiates the coordinate chain
  // alone, and its parameter-gradient GEMMs would be discarded.
  auto w0t = w0_all.transpose(2, 3);
  auto w1t = w1_all.transpose(2, 3);
  auto gwt = gw_all.transpose(2, 3);
  auto mix =
      dpa4_sezm::mixing_bwd(grad_x_local.contiguous(), x_local, z_all, u_final,
                            alpha, w0t, w1t, gw_all, gwt, u0, h_z, h_uf, lmax,
                            cf, apply_alpha, with_weights, keep_state);
  at::Tensor grad_u0 = std::get<0>(mix);
  const at::Tensor grad_alpha_mix = std::get<1>(mix);
  const at::Tensor grad_w0 = std::get<2>(mix);
  const at::Tensor grad_w1 = std::get<3>(mix);
  const at::Tensor grad_gw = std::get<4>(mix);
  const at::Tensor kept_upstream =
      keep_state ? std::get<5>(mix) : at::empty({0}, x.options());
  const at::Tensor kept_grad_z =
      keep_state ? std::get<7>(mix) : at::empty({0}, x.options());
  const at::Tensor kept_gate_logit =
      keep_state ? std::get<8>(mix) : at::empty({0}, x.options());
  const at::Tensor kept_grad_alpha_mix =
      keep_state && apply_alpha ? grad_alpha_mix
                                : at::empty({0, n_focus}, alpha.options());

  // === Step 3. Competition head, closed form from the stored weight ===
  // The gate-slice term enters the input gradient and is always applied;
  // the parameter contractions follow the weight gate.
  at::Tensor grad_w_fc = at::empty({0}, x.options());
  at::Tensor grad_bias = at::empty({0}, x.options());
  if (apply_alpha) {
    const long n_edge = alpha.size(0);
    auto grad_logit =
        with_weights
            ? at::empty({n_edge, n_focus}, alpha.options().dtype(at::kDouble))
            : at::empty({0, n_focus}, alpha.options().dtype(at::kDouble));
    const at::Tensor w_fc_t = w_fc->contiguous();
    const at::Tensor h_alpha_t = h_alpha.has_value()
                                     ? h_alpha->contiguous()
                                     : at::empty({0}, alpha.options());
    if (n_edge > 0) {
      int threads = 32;
      while (threads < n_focus * cf && threads < kThreads) {
        threads <<= 1;
      }
      auto stream = at::cuda::getCurrentCUDAStream();
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kBFloat16, at::kHalf, x.scalar_type(), "competition_bwd", [&] {
            using acc_t = typename acc_type<scalar_t>::type;
            competition_bwd_kernel<scalar_t><<<n_edge, threads, 0, stream>>>(
                grad_u0.data_ptr<scalar_t>(), w_fc_t.data_ptr<scalar_t>(),
                alpha.data_ptr<acc_t>(), grad_alpha_mix.data_ptr<acc_t>(),
                h_alpha.has_value() ? h_alpha_t.data_ptr<acc_t>() : nullptr,
                with_weights ? grad_logit.data_ptr<double>() : nullptr, n_edge,
                (int)n_focus, cf, (int)grad_u0.size(2), 1.0 / softmax_tau,
                label_smoothing);
          });
      DPA4_SC_CHECK_LAUNCH("sezm_so2_value_bwd competition");
    }
    if (with_weights) {
      auto gate = u0.narrow(2, 0, cf).permute({1, 0, 2}).to(at::kDouble);
      grad_w_fc = at::einsum("ef,efi->if", {grad_logit, gate})
                      .to(w_fc->scalar_type())
                      .contiguous();
      if (fc_bias.has_value()) {
        grad_bias = grad_logit.sum(0).to(fc_bias->scalar_type()).contiguous();
      }
    }
  }

  // === Step 4. Rotation gradients and the CSR node reduction ===
  auto rot = dpa4_sezm::rotate_mix_bwd(grad_u0, x, src, runs, kc, cb, lmax,
                                       n_focus, rank);
  auto grad_x =
      dpa4_sezm::segment_sum_csr(std::get<0>(rot), src_order, src_rowptr);

  return {grad_x,           std::get<1>(rot),
          std::get<2>(rot), std::get<3>(rot),
          grad_w_fc,        grad_bias,
          grad_w0,          grad_w1,
          grad_gw,          keep_state ? grad_u0 : at::empty({0}, x.options()),
          kept_upstream,    kept_grad_z,
          kept_gate_logit,  kept_grad_alpha_mix};
}

// ---------------------------------------------------------------------------
// Second order for the force-loss regime: the cotangent enters only through
// the node-feature gradient (the parameter gradients feed the optimizer and
// carry no cotangent). The rotation front end is multilinear, so its second
// order re-enters the forward and backward traversals with the cotangent in
// the feature slot; the competition head's softmax curvature is closed form;
// the mixing traversal delegates to its own hand-derived second order. The
// trailing outputs carry the curvature of the backward's anchor inputs
// (x_local, alpha, z_all; the u_final slot is a zero-shaped placeholder,
// since the force-regime first order never reads it), which autograd routes
// back through the forward's output slots into one anchor re-entry of the
// first-order operator.
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
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor>
value_bwd2(const at::Tensor& h_gx,
           const c10::optional<at::Tensor>& h_gruns,
           const c10::optional<at::Tensor>& h_gkc,
           const at::Tensor& grad_x_local,
           const at::Tensor& x,
           const at::Tensor& src,
           const at::Tensor& src_order,
           const at::Tensor& src_rowptr,
           const at::Tensor& runs,
           const at::Tensor& kc,
           const at::Tensor& cb,
           const c10::optional<at::Tensor>& w_fc,
           const c10::optional<at::Tensor>& fc_bias,
           const at::Tensor& w0_all,
           const at::Tensor& w1_all,
           const at::Tensor& gw_all,
           const at::Tensor& x_local,
           const at::Tensor& z_all,
           const at::Tensor& u_final,
           const at::Tensor& alpha,
           const c10::optional<at::Tensor>& kept_grad_u0,
           const c10::optional<at::Tensor>& kept_upstream,
           const c10::optional<at::Tensor>& kept_grad_z,
           const c10::optional<at::Tensor>& kept_gate_logit,
           const c10::optional<at::Tensor>& kept_grad_alpha_mix,
           int64_t lmax,
           int64_t n_focus,
           int64_t rank,
           bool apply_alpha,
           double softmax_tau,
           double label_smoothing) {
  check_value_inputs(x, src, runs, kc, w0_all, lmax, n_focus, rank, softmax_tau,
                     label_smoothing, "sezm_so2_value_bwd2");
  TORCH_CHECK(!apply_alpha || w_fc.has_value(),
              "sezm_so2_value_bwd2: competition weights required");
  const c10::cuda::CUDAGuard guard(x.device());
  const int cf = (int)(x.size(2) / n_focus);
  const bool kept = kept_grad_u0.has_value() && kept_upstream.has_value() &&
                    kept_grad_z.has_value() && kept_gate_logit.has_value();

  // === Step 1. Linearization points: rotated input and edge cotangents ===
  // The rotation backward is multilinear, so the cotangent of its upstream
  // collects one forward re-entry per differentiated output: the node
  // gradient with the node cotangent in the feature slot (gathered onto
  // edges in place), the run gradient with its cotangent in the run
  // slot, and the degree-kernel gradient with its cotangent in the kernel
  // slot. The paired kernel evaluates u0 and that sum in one traversal.
  auto pair = dpa4_sezm::rotate_mix_fwd_pair(x, h_gx, src, runs, h_gruns, kc,
                                             h_gkc, cb, lmax, n_focus, rank);
  auto u0 = std::get<0>(pair);
  auto h_gu0 = std::get<1>(pair);

  // === Step 2. Competition head curvature (feeds the traversal below) ===
  // The first-order head reads the softmax off the stored competition
  // weight, p = (alpha - ls/F) / (1 - ls), takes the traversal's alpha
  // gradient ga_mix[e,f] = <grad_out[e,f,:], x_local[e,f,:]> / alpha[e,f],
  // and emits gl = p (ga - <ga, p>) / tau with ga = (1 - ls) ga_mix and the
  // gate-slice gradient g_gate = gl w_fc^T. This second order linearizes
  // exactly that map: the cotangent of g_gate (the gate slice of
  // ``h_gu0``) lands on w_fc directly, on (grad_out, x_local, alpha)
  // through ga_mix, and on the alpha anchor again through p. The autograd
  // composition then routes the alpha and x_local cotangents back through
  // the forward's own graph, where the softmax's dependence on
  // (u0, w_fc, bias) lives; nothing of it belongs to this operator's x or
  // bias slots, and the finite-difference contract of the backward
  // confirms both are flat.
  at::Tensor gwfc2 = at::empty({0}, x.options());
  at::Tensor gbias2 = at::empty({0}, x.options());
  at::Tensor ggxl_scale;   // row scale of the upstream-gradient curvature
  at::Tensor galpha_head;  // head curvature on the alpha anchor
  at::Tensor gl_first;     // first-order logit gradient of the head
  const double ls = label_smoothing;
  const double inv_tau = 1.0 / softmax_tau;
  if (apply_alpha) {
    // The (E, F) scalar chain divides by alpha (as small as ls / F) and
    // runs in double, which keeps that conditioning out of the fp32
    // gradients at negligible cost. The row-wide products against the
    // stored surfaces stay in fp32: promoting an (E, F, ROW) surface to
    // double costs hundreds of megabytes of traffic on the wide shapes and
    // contributes nothing -- the surfaces themselves carry working
    // precision.
    auto alpha_acc = alpha.to(at::kDouble);
    auto p = ((alpha_acc - ls / (double)n_focus) / (1.0 - ls)).clamp_min(0.0);
    // The force traversal retains this scalar contraction from its first
    // order. A caller without retained state reconstructs it from the wide
    // rows, accumulating the reduction in fp32; only the (E, F) chain that
    // divides by alpha runs in double.
    auto ga_mix = kept_grad_alpha_mix.has_value()
                      ? kept_grad_alpha_mix->to(at::kDouble)
                      : (grad_x_local * x_local)
                                .sum(-1, false, at::kFloat)
                                .to(at::kDouble) /
                            alpha_acc;
    auto ga = ga_mix * (1.0 - ls);
    auto A = (ga * p).sum(1, true);
    auto gl = p * (ga - A) * inv_tau;
    gl_first = gl;

    // Gate slice of the grad_u0 cotangent, focus-major -> edge-major.
    auto hgg =
        h_gu0.narrow(2, 0, cf).permute({1, 0, 2}).to(at::kDouble);  // (E,F,Cf)
    auto wfc_acc = w_fc->to(at::kDouble);
    auto s = at::einsum("efi,if->ef", {hgg, wfc_acc});
    auto S2 = (s * p).sum(1, true);
    // VJP onto ga_mix (the gl route at fixed p), then through ga_mix's own
    // operands: the upstream rows, the stored output rows, and the alpha
    // divisor.
    auto h_ga = p * (s - S2) * (inv_tau * (1.0 - ls));
    ggxl_scale = (h_ga / alpha_acc).to(x_local.scalar_type()).contiguous();
    // VJP onto the alpha anchor: the p route of gl plus ga_mix's divisor.
    galpha_head = ((s * (ga - A) - ga * S2) * (inv_tau / (1.0 - ls)) -
                   h_ga * ga_mix / alpha_acc)
                      .to(alpha.scalar_type())
                      .contiguous();
    // Parameter curvature: g_gate is linear in w_fc at fixed (p, ga).
    gwfc2 = at::einsum("ef,efi->if", {gl, hgg})
                .to(w_fc->scalar_type())
                .contiguous();
    if (fc_bias.has_value()) {
      gbias2 = at::zeros_like(*fc_bias);
    }
  }

  // === Step 3. Mixing traversal second order ===
  // In the kept regime the first-order surfaces and the total input
  // gradient arrive from the first-order operator and no traversal
  // replays. The head curvature on the upstream gradient seeds the
  // traversal's output store, so no separate addition pass exists.
  auto w0t = w0_all.transpose(2, 3);
  auto w1t = w1_all.transpose(2, 3);
  auto gwt = gw_all.transpose(2, 3);
  auto mix2 = dpa4_sezm::mixing_bwd2(
      grad_x_local.contiguous(), x_local, z_all, u_final, alpha, w0t, w1t,
      gw_all, gwt, u0, h_gu0.contiguous(), c10::nullopt, c10::nullopt,
      c10::nullopt, kept ? kept_upstream : c10::nullopt,
      kept ? kept_grad_z : c10::nullopt, kept ? kept_gate_logit : c10::nullopt,
      apply_alpha ? c10::optional<at::Tensor>(ggxl_scale) : c10::nullopt, lmax,
      cf, apply_alpha);
  auto grad_grad_x_local = std::get<0>(mix2);
  auto gz2 = std::get<1>(mix2);
  // Zero-shaped placeholder: the force-regime first order never reads
  // u_final (weight contractions skipped, the alpha gradient contracts
  // against the stored output), so its anchor slot carries no curvature.
  auto guf2 = std::get<2>(mix2);
  auto galpha2 = std::get<3>(mix2);
  auto gw02 = std::get<4>(mix2);
  auto gw12 = std::get<5>(mix2);
  auto ggw2 = std::get<6>(mix2);
  auto gxlocal2 = std::get<11>(mix2);
  // Total first-order input gradient (head term included in the kept form;
  // added here otherwise).
  at::Tensor grad_u0 = kept ? kept_grad_u0.value() : std::get<10>(mix2);
  if (apply_alpha && !kept) {
    auto g_gate = at::einsum("ef,if->efi", {gl_first, w_fc->to(at::kDouble)})
                      .to(u0.scalar_type());
    grad_u0.narrow(2, 0, cf).add_(g_gate.permute({1, 0, 2}));
  }

  at::Tensor gxl2_out = at::empty({0}, x.options());
  if (apply_alpha) {
    galpha2.add_(galpha_head);
    gxl2_out = gxlocal2;
  }

  // === Step 4. Rotation tail ===
  // One traversal evaluates the three backward re-entries (node, run and
  // kernel cotangents each placed in the slot of the operand they
  // differentiate) against the shared upstream grad_u0; grad_u0 itself is
  // flat in x at fixed anchors, so the node curvature comes only from the
  // run and kernel cotangents.
  auto rot2 = dpa4_sezm::rotate_mix_bwd2(grad_u0, x, h_gx, src, runs, h_gruns,
                                         kc, h_gkc, cb, lmax, n_focus, rank);
  auto gruns2 = std::get<1>(rot2);
  auto gkc2 = std::get<2>(rot2);
  auto gcb2 = rank > 0 ? std::get<3>(rot2) : at::empty({0}, x.options());
  auto gx2_edge = std::get<0>(rot2);
  auto gx2 = gx2_edge.size(0) > 0
                 ? dpa4_sezm::segment_sum_csr(gx2_edge, src_order, src_rowptr)
                 : at::zeros(x.sizes(), x.options());

  return {grad_grad_x_local,
          gx2,
          gruns2,
          gkc2,
          gcb2,
          gwfc2,
          gbias2,
          gw02,
          gw12,
          ggw2,
          gxl2_out,
          galpha2,
          gz2,
          guf2};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "sezm_so2_value_fwd(Tensor x, Tensor src, Tensor runs, Tensor kc, "
      "Tensor cb, Tensor? w_fc, Tensor? fc_bias, Tensor w0_all, "
      "Tensor w1_all, Tensor gw_all, int lmax, int n_focus, int rank, "
      "bool apply_alpha, float softmax_tau, float label_smoothing) "
      "-> (Tensor x_out, Tensor z_all, Tensor u_final, Tensor alpha)");
  m.def(
      "sezm_so2_value_bwd(Tensor grad_x_local, Tensor x, Tensor src, "
      "Tensor src_order, Tensor src_rowptr, Tensor runs, Tensor kc, "
      "Tensor cb, Tensor? w_fc, Tensor? fc_bias, Tensor w0_all, "
      "Tensor w1_all, Tensor gw_all, Tensor x_local, Tensor z_all, "
      "Tensor u_final, Tensor alpha, Tensor? h_z, Tensor? h_uf, "
      "Tensor? h_alpha, int lmax, int n_focus, int rank, bool apply_alpha, "
      "float softmax_tau, float label_smoothing, bool keep_state, "
      "bool with_weights) "
      "-> (Tensor grad_x, Tensor grad_runs, Tensor grad_kc, "
      "Tensor grad_cb, Tensor grad_w_fc, Tensor grad_bias, "
      "Tensor grad_w0_all, Tensor grad_w1_all, Tensor grad_gw_all, "
      "Tensor kept_grad_u0, Tensor kept_upstream, Tensor kept_grad_z, "
      "Tensor kept_gate_logit, Tensor kept_grad_alpha_mix)");
  m.def(
      "sezm_so2_value_bwd2(Tensor h_gx, Tensor? h_gruns, Tensor? h_gkc, "
      "Tensor grad_x_local, Tensor x, "
      "Tensor src, Tensor src_order, Tensor src_rowptr, Tensor runs, "
      "Tensor kc, Tensor cb, Tensor? w_fc, Tensor? fc_bias, Tensor w0_all, "
      "Tensor w1_all, Tensor gw_all, Tensor x_local, Tensor z_all, "
      "Tensor u_final, Tensor alpha, Tensor? kept_grad_u0, "
      "Tensor? kept_upstream, Tensor? kept_grad_z, Tensor? kept_gate_logit, "
      "Tensor? kept_grad_alpha_mix, "
      "int lmax, int n_focus, int rank, "
      "bool apply_alpha, float softmax_tau, float label_smoothing) "
      "-> (Tensor grad_grad_x_local, Tensor gx2, Tensor gruns2, Tensor gkc2, "
      "Tensor gcb2, Tensor gwfc2, Tensor gbias2, Tensor gw02, Tensor gw12, "
      "Tensor ggw2, Tensor gxl2, Tensor galpha2, Tensor gz2, Tensor guf2)");
}

TORCH_LIBRARY_IMPL(deepmd, CUDA, m) {
  m.impl("sezm_so2_value_fwd", &value_fwd);
  m.impl("sezm_so2_value_bwd", &value_bwd);
  m.impl("sezm_so2_value_bwd2", &value_bwd2);
}

TORCH_LIBRARY_IMPL(deepmd, Autograd, m) {
  m.impl("sezm_so2_value_fwd", torch::CppFunction::makeFallthrough());
  m.impl("sezm_so2_value_bwd", torch::CppFunction::makeFallthrough());
  m.impl("sezm_so2_value_bwd2", torch::CppFunction::makeFallthrough());
}
