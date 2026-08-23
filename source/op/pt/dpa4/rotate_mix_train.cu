// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused rotate-to-local + radial degree mixing for SeZM / DPA4 training.
//
// The operator gathers the source-node features of every edge, applies the
// block-diagonal Wigner-D rotation over its structural non-zeros only (the
// m-major reduced rows |m| <= 1), applies the edge-conditioned radial degree
// mixing, and stores straight into the focus-major layout (F, E, ROW) with
// ROW = (3 lmax + 1) Cf that the mixing stack consumes:
//
//   x_local[e, r, c] = sum_j D_e[row_sel(r), col(r, j)] x[src[e], col(r, j), c]
//   u[f, e, r, cf]   = mix(x_local)[e, r, c],  c = f Cf + cf
//
// with the mixing one of
//   RANK == 0:  x_local[e, r, c] * rad[e, deg(r), c]
//   RANK == 1:  (sum_i k[e, i, o] x_local[e, i, c]) * cb[c]
//   RANK >= 2:  sum_i (sum_t k[e, i, o, t] cb[t, c]) x_local[e, i, c]
// where the degree kernel contracts the m = 0 rows over (L+1)^2 pairs and
// the |m| = 1 rows over L^2 pairs shared by the two signed halves.
//
// The backward recomputes the rotated rows from x and D in registers (the
// kernel reads both anyway, so the forward saves no per-edge intermediate),
// emits the per-edge node gradient densely -- the caller segment-sums it
// over the source CSR view, which this file also provides -- and writes the
// Wigner gradient on the structural non-zeros and the degree-kernel gradient
// through block-wide channel reductions.
//
// The mathematics mirrors the fused Triton operators of
// ``so2_value_path.py`` (`_rotate_mix_reference` and
// ``_rotate_mix_backward_reference`` are the eager ground truths shared by
// both implementations).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <tuple>

#include "rotate_mix_train_kernels.cuh"
#include "sezm_train_ops.cuh"

// The kernel templates are instantiated in the per-degree units
// (rotate_mix_train_l*.cu); the declarations below keep this host unit from
// re-instantiating them, which is what dominated its build time.
#define DPA4_RMT_EXTERN extern
#define DPA4_RMT_L 1
#include "rotate_mix_train_instantiate.cuh"
#undef DPA4_RMT_L
#define DPA4_RMT_L 2
#include "rotate_mix_train_instantiate.cuh"
#undef DPA4_RMT_L
#define DPA4_RMT_L 3
#include "rotate_mix_train_instantiate.cuh"
#undef DPA4_RMT_L
#define DPA4_RMT_L 4
#include "rotate_mix_train_instantiate.cuh"
#undef DPA4_RMT_L
#define DPA4_RMT_L 5
#include "rotate_mix_train_instantiate.cuh"
#undef DPA4_RMT_L
#define DPA4_RMT_L 6
#include "rotate_mix_train_instantiate.cuh"
#undef DPA4_RMT_L
#undef DPA4_RMT_EXTERN

using namespace dpa4_sezm_kernels;

namespace {

template <typename F>
void dispatch_l(int64_t lmax, const F& f) {
  switch (lmax) {
#define DPA4_RM_L_CASE(L)                \
  case L:                                \
    f(std::integral_constant<int, L>{}); \
    break;
    DPA4_RM_L_CASE(1)
    DPA4_RM_L_CASE(2)
    DPA4_RM_L_CASE(3)
    DPA4_RM_L_CASE(4)
    DPA4_RM_L_CASE(5)
    DPA4_RM_L_CASE(6)
#undef DPA4_RM_L_CASE
    default:
      TORCH_CHECK(false, "sezm_rotate_mix: unsupported lmax");
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entries, composed by the fused SO(2) value-path operator.
// ---------------------------------------------------------------------------
namespace dpa4_sezm {

at::Tensor rotate_mix_fwd(const at::Tensor& x_in,
                          const at::Tensor& src,
                          const at::Tensor& wigner_in,
                          const at::Tensor& kc_in,
                          const at::Tensor& cb_in,
                          int64_t lmax,
                          int64_t n_focus,
                          int64_t rank) {
  check_rotate_inputs(x_in, src, wigner_in, lmax, n_focus, rank,
                      "sezm_rotate_mix_fwd");
  const c10::cuda::CUDAGuard guard(x_in.device());
  const at::Tensor x = x_in.stride(2) == 1 ? x_in : x_in.contiguous();
  const at::Tensor wigner = wigner_in.contiguous();
  const at::Tensor kc = kc_in.contiguous();
  const at::Tensor cb = cb_in.contiguous();
  const long n_edge = src.size(0);
  const int c_wide = (int)x.size(2);
  const int cf = c_wide / (int)n_focus;
  const long row_w = (3 * lmax + 1) * cf;
  auto u = at::empty({n_focus, n_edge, row_w}, x.options());
  if (n_edge == 0) {
    return u;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = lane_count(c_wide);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "rotate_mix_fwd", [&] {
        dispatch_l(lmax, [&](auto lc) {
          launch_rotate_mix_fwd<scalar_t, decltype(lc)::value>(
              x.data_ptr<scalar_t>(), src.data_ptr<long>(),
              wigner.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
              cb.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), n_edge,
              x.stride(0), x.stride(1), cf, c_wide, (int)rank, threads, stream);
        });
      });
  DPA4_RM_CHECK_LAUNCH("sezm_rotate_mix_fwd");
  return u;
}

std::tuple<at::Tensor, at::Tensor> rotate_mix_fwd_pair(
    const at::Tensor& x_in,
    const at::Tensor& h_gx_in,
    const at::Tensor& src,
    const at::Tensor& wigner_in,
    const c10::optional<at::Tensor>& h_gwig,
    const at::Tensor& kc_in,
    const c10::optional<at::Tensor>& h_gkc,
    const at::Tensor& cb_in,
    int64_t lmax,
    int64_t n_focus,
    int64_t rank) {
  check_rotate_inputs(x_in, src, wigner_in, lmax, n_focus, rank,
                      "sezm_rotate_mix_fwd_pair");
  const c10::cuda::CUDAGuard guard(x_in.device());
  const at::Tensor x = x_in.stride(2) == 1 ? x_in : x_in.contiguous();
  const at::Tensor h_gx =
      h_gx_in.stride(2) == 1 ? h_gx_in : h_gx_in.contiguous();
  const at::Tensor wigner = wigner_in.contiguous();
  const at::Tensor kc = kc_in.contiguous();
  const at::Tensor cb = cb_in.contiguous();
  const at::Tensor h_gwig_t =
      h_gwig.has_value() ? h_gwig->contiguous() : at::Tensor();
  const at::Tensor h_gkc_t =
      h_gkc.has_value() ? h_gkc->contiguous() : at::Tensor();
  const long n_edge = src.size(0);
  const int c_wide = (int)x.size(2);
  const int cf = c_wide / (int)n_focus;
  const long row_w = (3 * lmax + 1) * cf;
  auto u0 = at::empty({n_focus, n_edge, row_w}, x.options());
  auto hgu0 = at::empty({n_focus, n_edge, row_w}, x.options());
  if (n_edge == 0) {
    return {u0, hgu0};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = lane_count(c_wide);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "rotate_mix_fwd_pair", [&] {
        dispatch_l(lmax, [&](auto lc) {
          launch_rotate_mix_fwd_pair<scalar_t, decltype(lc)::value>(
              x.data_ptr<scalar_t>(), h_gx.data_ptr<scalar_t>(),
              src.data_ptr<long>(), wigner.data_ptr<scalar_t>(),
              h_gwig_t.defined() ? h_gwig_t.data_ptr<scalar_t>() : nullptr,
              kc.data_ptr<scalar_t>(),
              h_gkc_t.defined() ? h_gkc_t.data_ptr<scalar_t>() : nullptr,
              cb.data_ptr<scalar_t>(), u0.data_ptr<scalar_t>(),
              hgu0.data_ptr<scalar_t>(), n_edge, x.stride(0), x.stride(1),
              h_gx.stride(0), h_gx.stride(1), cf, c_wide, (int)rank, threads,
              stream);
        });
      });
  DPA4_RM_CHECK_LAUNCH("sezm_rotate_mix_fwd_pair");
  return {u0, hgu0};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rotate_mix_bwd(
    const at::Tensor& grad_u_in,
    const at::Tensor& x_in,
    const at::Tensor& src,
    const at::Tensor& wigner_in,
    const at::Tensor& kc_in,
    const at::Tensor& cb_in,
    int64_t lmax,
    int64_t n_focus,
    int64_t rank) {
  check_rotate_inputs(x_in, src, wigner_in, lmax, n_focus, rank,
                      "sezm_rotate_mix_bwd");
  const c10::cuda::CUDAGuard guard(x_in.device());
  const at::Tensor grad_u = grad_u_in.contiguous();
  const at::Tensor x = x_in.stride(2) == 1 ? x_in : x_in.contiguous();
  const at::Tensor wigner = wigner_in.contiguous();
  const at::Tensor kc = kc_in.contiguous();
  const at::Tensor cb = cb_in.contiguous();
  const long n_edge = src.size(0);
  const int c_wide = (int)x.size(2);
  const int cf = c_wide / (int)n_focus;
  const long dim = (lmax + 1) * (lmax + 1);
  auto grad_x_edge = at::empty({n_edge, dim, c_wide}, x.options());
  auto grad_wigner = at::zeros_like(wigner);
  auto grad_kc = at::empty_like(kc);
  // Per-edge channel-basis partials; the reduction over edges runs as one
  // sum below (a direct atomic accumulation would serialize every edge on
  // the tiny (rank, C) output).
  auto pcb = at::empty({rank > 0 ? n_edge : 0, rank, c_wide}, x.options());
  auto grad_cb = at::zeros_like(cb);
  if (n_edge == 0) {
    return {grad_x_edge, grad_wigner, grad_kc, grad_cb};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = lane_count(c_wide);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "rotate_mix_bwd", [&] {
        dispatch_l(lmax, [&](auto lc) {
          launch_rotate_mix_bwd<scalar_t, decltype(lc)::value>(
              grad_u.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
              src.data_ptr<long>(), wigner.data_ptr<scalar_t>(),
              kc.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
              grad_x_edge.data_ptr<scalar_t>(),
              grad_wigner.data_ptr<scalar_t>(), grad_kc.data_ptr<scalar_t>(),
              rank > 0 ? pcb.data_ptr<scalar_t>() : nullptr, n_edge,
              x.stride(0), x.stride(1), cf, c_wide, (int)rank, threads, stream);
        });
      });
  DPA4_RM_CHECK_LAUNCH("sezm_rotate_mix_bwd");
  if (rank > 0) {
    grad_cb = pcb.sum(0, false, at::kFloat).to(cb.scalar_type()).view_as(cb);
  }
  return {grad_x_edge, grad_wigner, grad_kc, grad_cb};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rotate_mix_bwd2(
    const at::Tensor& grad_u_in,
    const at::Tensor& x_in,
    const at::Tensor& h_gx_in,
    const at::Tensor& src,
    const at::Tensor& wigner_in,
    const c10::optional<at::Tensor>& h_gwig,
    const at::Tensor& kc_in,
    const c10::optional<at::Tensor>& h_gkc,
    const at::Tensor& cb_in,
    int64_t lmax,
    int64_t n_focus,
    int64_t rank) {
  check_rotate_inputs(x_in, src, wigner_in, lmax, n_focus, rank,
                      "sezm_rotate_mix_bwd2");
  const c10::cuda::CUDAGuard guard(x_in.device());
  const at::Tensor grad_u = grad_u_in.contiguous();
  const at::Tensor x = x_in.stride(2) == 1 ? x_in : x_in.contiguous();
  const at::Tensor h_gx =
      h_gx_in.stride(2) == 1 ? h_gx_in : h_gx_in.contiguous();
  const at::Tensor wigner = wigner_in.contiguous();
  const at::Tensor kc = kc_in.contiguous();
  const at::Tensor cb = cb_in.contiguous();
  const at::Tensor h_gwig_t =
      h_gwig.has_value() ? h_gwig->contiguous() : at::Tensor();
  const at::Tensor h_gkc_t =
      h_gkc.has_value() ? h_gkc->contiguous() : at::Tensor();
  const bool wants_gxe = h_gwig_t.defined() || h_gkc_t.defined();
  const long n_edge = src.size(0);
  const int c_wide = (int)x.size(2);
  const int cf = c_wide / (int)n_focus;
  const long dim = (lmax + 1) * (lmax + 1);
  auto grad_x_edge =
      at::empty({wants_gxe ? n_edge : 0, dim, c_wide}, x.options());
  auto grad_wigner = at::zeros_like(wigner);
  auto grad_kc = at::empty_like(kc);
  auto pcb = at::empty({rank > 0 ? n_edge : 0, rank, c_wide}, x.options());
  auto grad_cb = at::zeros_like(cb);
  if (n_edge == 0) {
    return {grad_x_edge, grad_wigner, grad_kc, grad_cb};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = lane_count(c_wide);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "rotate_mix_bwd2", [&] {
        dispatch_l(lmax, [&](auto lc) {
          launch_rotate_mix_bwd2<scalar_t, decltype(lc)::value>(
              grad_u.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
              h_gx.data_ptr<scalar_t>(), src.data_ptr<long>(),
              wigner.data_ptr<scalar_t>(),
              h_gwig_t.defined() ? h_gwig_t.data_ptr<scalar_t>() : nullptr,
              kc.data_ptr<scalar_t>(),
              h_gkc_t.defined() ? h_gkc_t.data_ptr<scalar_t>() : nullptr,
              cb.data_ptr<scalar_t>(),
              wants_gxe ? grad_x_edge.data_ptr<scalar_t>() : nullptr,
              grad_wigner.data_ptr<scalar_t>(), grad_kc.data_ptr<scalar_t>(),
              rank > 0 ? pcb.data_ptr<scalar_t>() : nullptr, n_edge,
              x.stride(0), x.stride(1), h_gx.stride(0), h_gx.stride(1), cf,
              c_wide, (int)rank, threads, stream);
        });
      });
  DPA4_RM_CHECK_LAUNCH("sezm_rotate_mix_bwd2");
  if (rank > 0) {
    grad_cb = pcb.sum(0, false, at::kFloat).to(cb.scalar_type()).view_as(cb);
  }
  return {grad_x_edge, grad_wigner, grad_kc, grad_cb};
}

at::Tensor segment_sum_csr(const at::Tensor& rows_in,
                           const at::Tensor& order,
                           const at::Tensor& row_ptr) {
  TORCH_CHECK(rows_in.is_cuda() && rows_in.dim() >= 2,
              "sezm_segment_sum: rows must be at least 2D on CUDA");
  TORCH_CHECK(
      order.scalar_type() == at::kLong && row_ptr.scalar_type() == at::kLong,
      "sezm_segment_sum: CSR indices must be int64");
  const c10::cuda::CUDAGuard guard(rows_in.device());
  const at::Tensor rows = rows_in.contiguous();
  const long n_seg = row_ptr.size(0) - 1;
  auto sizes = rows.sizes().vec();
  long feat = 1;
  for (size_t i = 1; i < sizes.size(); ++i) {
    feat *= sizes[i];
  }
  sizes[0] = n_seg;
  auto out = at::empty(sizes, rows.options());
  if (n_seg == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int tiles = (int)std::min<long>((feat + threads - 1) / threads, 64);
  dim3 grid((unsigned)n_seg, (unsigned)std::max(tiles, 1));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, rows.scalar_type(), "segment_sum_csr", [&] {
        segment_sum_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            rows.data_ptr<scalar_t>(), order.data_ptr<long>(),
            row_ptr.data_ptr<long>(), out.data_ptr<scalar_t>(), n_seg, feat);
      });
  DPA4_RM_CHECK_LAUNCH("sezm_segment_sum");
  return out;
}

}  // namespace dpa4_sezm
