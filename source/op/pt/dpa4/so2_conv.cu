// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Host side of the fused DPA4 / SeZM SO(2) convolution: validation, topology,
// and the PyTorch operator schema.
//
// The kernels live in ``so2_conv_kernel.cuh`` and are instantiated per
// focus width in ``so2_conv_c{32,64}.cu``; see ``so2_conv_launch.h``
// for the launch policy and ``so2_conv.cuh`` for the layout algebra.
//
// Both directions derive their CSR view of the topology here rather than
// accepting it as an argument: a Python wrapper would need the node count as a
// Python integer to size the row pointer, which bakes the trace-time count into
// the compiled graph.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cmath>
#include <tuple>

#include "so2_conv_launch.h"

namespace {

#define DPA4_CHECK_LAUNCH(what)                                           \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

/// Runtime configuration of one convolution, resolved from the arguments.
struct ConvConfig {
  int lmax;
  int focus_dim;
  int n_focus;
  int n_head;
  int n_layers;
  int rank;
  int kc_len;
  int c_wide;
  int dim;
  int row;
  long n_edge;
  int n_node;
};

ConvConfig resolve_config(const torch::Tensor& x,
                          const torch::Tensor& runs,
                          const torch::Tensor& kc,
                          const torch::Tensor& cb,
                          const torch::Tensor& w0,
                          const torch::Tensor& head_gate,
                          int64_t lmax,
                          int64_t focus_dim,
                          int64_t rank) {
  ConvConfig c{};
  c.lmax = static_cast<int>(lmax);
  c.focus_dim = static_cast<int>(focus_dim);
  c.c_wide = static_cast<int>(x.size(2));
  c.n_focus = c.c_wide / c.focus_dim;
  c.n_head = static_cast<int>(head_gate.size(2));
  c.n_layers = static_cast<int>(w0.size(0));
  c.rank = static_cast<int>(rank);
  c.n_edge = runs.size(0);
  c.kc_len = static_cast<int>(kc.size(1));
  c.dim = (c.lmax + 1) * (c.lmax + 1);
  c.row = (3 * c.lmax + 1) * c.focus_dim;
  c.n_node = static_cast<int>(x.size(0));
  (void)cb;
  return c;
}

void check_inputs(const torch::Tensor& x,
                  const torch::Tensor& runs,
                  const torch::Tensor& kc,
                  const torch::Tensor& cb,
                  const torch::Tensor& head_gate,
                  const ConvConfig& c) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat,
              "dpa4_so2_conv: x must be cuda fp32");
  TORCH_CHECK(
      x.dim() == 3 && x.stride(2) == 1,
      "dpa4_so2_conv: x must be (N, D, C_wide) with unit channel stride");
  TORCH_CHECK(x.size(1) == c.dim,
              "dpa4_so2_conv: x degree extent does not match lmax");
  TORCH_CHECK(dpa4::conv_shape_instantiated(c.lmax, c.focus_dim),
              "dpa4_so2_conv: no instantiation for lmax=", c.lmax,
              " focus_dim=", c.focus_dim);
  TORCH_CHECK(c.c_wide == c.n_focus * c.focus_dim,
              "dpa4_so2_conv: C_wide must be a multiple of focus_dim");
  TORCH_CHECK(c.n_head == 1,
              "dpa4_so2_conv: the fused operator supports one attention head");
  TORCH_CHECK(c.n_layers >= 2,
              "dpa4_so2_conv: the stack needs at least one gated layer");
  TORCH_CHECK(
      runs.dim() == 2 && runs.size(1) == 3L * (c.lmax + 1) * (c.lmax + 1) - 2,
      "dpa4_so2_conv: the packed runs must be (E, NW)");
  TORCH_CHECK(kc.size(0) == c.n_edge,
              "dpa4_so2_conv: the degree kernel must be edge major");
  const int expect_kc =
      c.rank == 0 ? (c.lmax + 1) * c.c_wide
                  : ((c.lmax + 1) * (c.lmax + 1) + c.lmax * c.lmax) * c.rank;
  TORCH_CHECK(c.kc_len == expect_kc, "dpa4_so2_conv: degree-kernel width ",
              c.kc_len,
              " does not match "
              "rank ",
              c.rank, ", expected ", expect_kc);
  TORCH_CHECK(c.rank == 0 || cb.numel() == static_cast<long>(c.rank) * c.c_wide,
              "dpa4_so2_conv: channel basis must be (rank, C_wide)");
  TORCH_CHECK(head_gate.dim() == 3 && head_gate.size(0) == c.n_node &&
                  head_gate.size(1) == c.n_focus &&
                  head_gate.size(2) == c.n_head,
              "dpa4_so2_conv: head gate must be (N, F, H)");
}

/// Validate one precomputed CSR view of an endpoint array.
void check_csr(const torch::Tensor& order,
               const torch::Tensor& row_ptr,
               const ConvConfig& c,
               const char* what) {
  TORCH_CHECK(order.scalar_type() == torch::kLong && order.numel() == c.n_edge,
              what, ": the CSR permutation must hold one index per edge");
  TORCH_CHECK(
      row_ptr.scalar_type() == torch::kLong && row_ptr.numel() == c.n_node + 1,
      what, ": the CSR row pointer must hold n_node + 1 offsets");
}

/// Evaluate a quaternion monomial basis over all edges.
///
/// Consecutive threads cover consecutive basis elements of one edge, so the
/// quaternion row broadcasts and the store coalesces. The powers are formed by
/// repeated multiplication; the exponents are at most twice the largest
/// instantiated degree.
__global__ __launch_bounds__(256) void monomial_kernel(
    const float* __restrict__ quat,
    const signed char* __restrict__ exps,
    float* __restrict__ mono,
    long total,
    int n_mono) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const long edge = idx / n_mono;
  const int m = static_cast<int>(idx - edge * n_mono);
  const float* q = quat + edge * 4;
  float value = 1.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int e = exps[m * 4 + c];
    const float base = q[c];
    for (int p = 0; p < e; ++p) {
      value *= base;
    }
  }
  mono[idx] = value;
}

/// Monomial matrix of every edge on the given basis, shape (E, M).
torch::Tensor monomial_matrix(const torch::Tensor& quat,
                              const torch::Tensor& exps,
                              cudaStream_t stream) {
  const long n_edge = quat.size(0);
  const long n_mono = exps.size(0);
  auto mono = torch::empty({n_edge, n_mono}, quat.options());
  const long total = n_edge * n_mono;
  if (total > 0) {
    const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
    monomial_kernel<<<dim3(blocks), dim3(256), 0, stream>>>(
        quat.data_ptr<float>(), exps.data_ptr<signed char>(),
        mono.data_ptr<float>(), total, static_cast<int>(n_mono));
  }
  DPA4_CHECK_LAUNCH("dpa4_so2_conv: monomial basis");
  return mono;
}

/// Repack a mixing weight for the vectorized reduction of ``row_multiply``.
///
/// The kernel walks the reduction four steps at a time and wants those four
/// steps of one output column contiguous, so an ``(in, out)`` matrix of shape
/// ``(..., KK, NN)`` is delivered as ``(..., KK / 4, NN, 4)``. The element
/// count and therefore every per-layer stride is unchanged, and because a
/// weight panel is always a whole number of step groups the panel staging is
/// unaffected.
torch::Tensor pack_reduction(const torch::Tensor& w) {
  return w.unflatten(-2, {w.size(-2) / 4, 4}).transpose(-2, -1).contiguous();
}

/// Fill the argument block shared by both directions.
dpa4::ConvArgs make_args(const ConvConfig& c,
                         const torch::Tensor& x,
                         const torch::Tensor& order,
                         const torch::Tensor& row_ptr,
                         const torch::Tensor& peer,
                         const torch::Tensor& runs,
                         const torch::Tensor& kc,
                         const torch::Tensor& cb,
                         const torch::Tensor& w0,
                         const torch::Tensor& w1,
                         const torch::Tensor& gw,
                         const torch::Tensor& head_gate,
                         const torch::Tensor& rescale,
                         torch::Tensor& z_all) {
  dpa4::ConvArgs a{};
  a.x = x.data_ptr<float>();
  a.order = order.data_ptr<int64_t>();
  a.row_ptr = row_ptr.data_ptr<int64_t>();
  a.peer = peer.data_ptr<int64_t>();
  a.runs = runs.data_ptr<float>();
  a.kc = kc.data_ptr<float>();
  a.cb = cb.data_ptr<float>();
  a.w0 = w0.data_ptr<float>();
  a.w1 = w1.data_ptr<float>();
  a.gw = gw.data_ptr<float>();
  a.head_gate = head_gate.data_ptr<float>();
  a.rescale = rescale.data_ptr<float>();
  a.z_all = z_all.data_ptr<float>();
  a.n_edge = c.n_edge;
  a.x_sn = static_cast<int>(x.stride(0));
  a.x_sd = static_cast<int>(x.stride(1));
  a.n_focus = c.n_focus;
  a.n_head = c.n_head;
  a.n_layers = c.n_layers;
  a.rank = c.rank;
  a.kc_len = c.kc_len;
  a.c_wide = c.c_wide;
  return a;
}

/// Build the packed runs of every edge: one monomial sweep and one product.
torch::Tensor build_runs(const torch::Tensor& quat,
                         const torch::Tensor& mono_coeff,
                         const torch::Tensor& mono_exp,
                         cudaStream_t stream) {
  auto mono = monomial_matrix(quat, mono_exp, stream);  // (E, M)
  return at::mm(mono, mono_coeff.t());                  // (E, NW)
}

/// Contract the packed-run cotangent onto the quaternions.
///
/// The run is a polynomial in the quaternion, so the cotangent folds through
/// the derivative tables: one product against the slot-major table and one
/// reduction over the derivative basis. The extension of the fitted polynomial
/// off the unit sphere is immaterial, because the quaternion normalization
/// upstream projects the radial gradient component out.
torch::Tensor contract_quat_grad(const torch::Tensor& quat,
                                 const torch::Tensor& g_runs,
                                 const torch::Tensor& dmono_coeff,
                                 const torch::Tensor& dmono_exp,
                                 cudaStream_t stream) {
  const long n_edge = quat.size(0);
  const long n_dmono = dmono_coeff.size(2);
  auto dmono = monomial_matrix(quat, dmono_exp, stream);  // (E, M')
  auto partial = at::mm(g_runs, dmono_coeff.reshape({g_runs.size(1), -1}))
                     .reshape({n_edge, 4, n_dmono});  // (E, 4, M')
  return (partial * dmono.unsqueeze(1)).sum(-1);      // (E, 4)
}

/// Route a runtime shape to its instantiated forward entry point.
bool dispatch_forward(const ConvConfig& c,
                      const dpa4::ConvArgs& args,
                      float* out,
                      float* pre_gate,
                      cudaStream_t stream) {
#define DPA4_CASE(LV, CFV)                                                     \
  if (c.lmax == LV && c.focus_dim == CFV) {                                    \
    dpa4::conv_forward_launch<LV, CFV>(args, c.n_node, out, pre_gate, stream); \
    return true;                                                               \
  }
  DPA4_CONV_FOR_EACH_SHAPE(DPA4_CASE)
#undef DPA4_CASE
  return false;
}

/// Route a runtime shape to its instantiated backward entry point.
bool dispatch_backward(const ConvConfig& c,
                       const dpa4::ConvArgs& args,
                       const float* g_out,
                       const float* w0t,
                       const float* w1t,
                       const float* gwt,
                       float* g_x,
                       float* g_wigner,
                       float* g_kc,
                       float* g_alpha,
                       cudaStream_t stream) {
#define DPA4_CASE(LV, CFV)                                                     \
  if (c.lmax == LV && c.focus_dim == CFV) {                                    \
    dpa4::conv_backward_launch<LV, CFV>(args, c.n_node, g_out, w0t, w1t, gwt,  \
                                        g_x, g_wigner, g_kc, g_alpha, stream); \
    return true;                                                               \
  }
  DPA4_CONV_FOR_EACH_SHAPE(DPA4_CASE)
#undef DPA4_CASE
  return false;
}

}  // namespace

namespace dpa4 {

bool conv_shape_instantiated(int lmax, int focus_dim) {
  return 1 <= lmax && lmax <= kMaxL &&
         (focus_dim == kFocusDim32 || focus_dim == kFocusDim64);
}

}  // namespace dpa4

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dpa4_so2_conv(torch::Tensor x,
              torch::Tensor src,
              torch::Tensor dst,
              torch::Tensor dst_order,
              torch::Tensor dst_rowptr,
              torch::Tensor src_order,
              torch::Tensor src_rowptr,
              torch::Tensor runs,
              torch::Tensor kc,
              torch::Tensor cb,
              torch::Tensor w0,
              torch::Tensor w1,
              torch::Tensor gw,
              torch::Tensor q,
              torch::Tensor k,
              torch::Tensor logit_w,
              torch::Tensor null_logit,
              torch::Tensor env,
              torch::Tensor rad0,
              torch::Tensor fscale,
              torch::Tensor head_gate,
              torch::Tensor rescale,
              int64_t lmax,
              int64_t focus_dim,
              int64_t rank) {
  const at::cuda::OptionalCUDAGuard device_guard(x.device());
  x = x.contiguous();
  kc = kc.contiguous().flatten(1);
  const ConvConfig c =
      resolve_config(x, runs, kc, cb, w0, head_gate, lmax, focus_dim, rank);
  check_inputs(x, runs, kc, cb, head_gate, c);
  TORCH_CHECK(q.numel() == static_cast<long>(c.n_node) * c.c_wide &&
                  k.numel() == q.numel(),
              "dpa4_so2_conv: q and k must be (N, C_wide)");
  TORCH_CHECK(
      logit_w.numel() == static_cast<long>(c.n_focus) * c.focus_dim * c.n_head,
      "dpa4_so2_conv: the logit projection must be (F, Cf, H)");
  TORCH_CHECK(null_logit.numel() == static_cast<long>(c.n_focus) * c.n_head,
              "dpa4_so2_conv: the null logit must be (F, H)");
  TORCH_CHECK(env.numel() == c.n_edge,
              "dpa4_so2_conv: the envelope must hold one weight per edge");
  TORCH_CHECK(rad0.numel() == c.n_edge * static_cast<long>(c.c_wide),
              "dpa4_so2_conv: the radial scalar row must be (E, C_wide)");
  TORCH_CHECK(fscale.numel() == 0 ||
                  fscale.numel() == c.n_edge * static_cast<long>(c.n_focus),
              "dpa4_so2_conv: the weight scale must be (E, F) or empty");

  src = src.to(torch::kLong).contiguous();
  dst = dst.to(torch::kLong).contiguous();
  runs = runs.contiguous();
  cb = cb.contiguous();
  w0 = pack_reduction(w0);
  w1 = pack_reduction(w1);
  gw = pack_reduction(gw);
  q = q.contiguous();
  k = k.contiguous();
  logit_w = logit_w.contiguous();
  null_logit = null_logit.contiguous();
  env = env.contiguous();
  rad0 = rad0.contiguous();
  fscale = fscale.contiguous();
  head_gate = head_gate.contiguous();
  rescale = rescale.contiguous();
  auto order = dst_order.contiguous();
  auto row_ptr = dst_rowptr.contiguous();
  check_csr(order, row_ptr, c, "dpa4_so2_conv");
  (void)src_order;
  (void)src_rowptr;

  auto out = torch::empty({c.n_node, c.dim, c.c_wide}, x.options());
  auto alpha = torch::empty({c.n_edge, c.n_focus, c.n_head}, x.options());
  auto pre_gate = torch::empty_like(out);
  // One slot per layer: the gated layers keep their pre-activation and the
  // identity layer keeps the finished activation the backward starts from.
  auto z_all =
      torch::empty({c.n_layers, c.n_edge, c.n_focus, c.row}, x.options());
  if (c.n_node <= 0) {
    return {out, alpha, pre_gate, z_all};
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  auto args = make_args(c, x, order, row_ptr, src, runs, kc, cb, w0, w1, gw,
                        head_gate, rescale, z_all);
  args.q = q.data_ptr<float>();
  args.k = k.data_ptr<float>();
  args.logit_w = logit_w.data_ptr<float>();
  args.null_logit = null_logit.data_ptr<float>();
  args.env = env.data_ptr<float>();
  args.kc0 = rad0.data_ptr<float>();
  args.fscale = fscale.numel() == 0 ? nullptr : fscale.data_ptr<float>();
  args.alpha_out = alpha.data_ptr<float>();
  args.inv_sqrt_ch =
      1.0f / std::sqrt(static_cast<float>(c.focus_dim / c.n_head));
  const bool launched = dispatch_forward(c, args, out.data_ptr<float>(),
                                         pre_gate.data_ptr<float>(), stream);
  TORCH_CHECK(launched,
              "dpa4_so2_conv: no instantiation for the resolved shape");
  DPA4_CHECK_LAUNCH("dpa4_so2_conv");
  return {out, alpha, pre_gate, z_all};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dpa4_so2_conv_backward(torch::Tensor grad_out,
                       torch::Tensor z_all,
                       torch::Tensor x,
                       torch::Tensor src,
                       torch::Tensor dst,
                       torch::Tensor src_order,
                       torch::Tensor src_rowptr,
                       torch::Tensor runs,
                       torch::Tensor kc,
                       torch::Tensor cb,
                       torch::Tensor w0,
                       torch::Tensor w1,
                       torch::Tensor gw,
                       torch::Tensor alpha,
                       torch::Tensor head_gate,
                       torch::Tensor rescale,
                       int64_t lmax,
                       int64_t focus_dim,
                       int64_t rank) {
  const at::cuda::OptionalCUDAGuard device_guard(x.device());
  x = x.contiguous();
  kc = kc.contiguous().flatten(1);
  const ConvConfig c =
      resolve_config(x, runs, kc, cb, w0, head_gate, lmax, focus_dim, rank);
  check_inputs(x, runs, kc, cb, head_gate, c);
  TORCH_CHECK(alpha.dim() == 3 && alpha.size(0) == c.n_edge &&
                  alpha.size(1) == c.n_focus && alpha.size(2) == c.n_head,
              "dpa4_so2_conv_backward: alpha must be (E, F, H)");
  TORCH_CHECK(grad_out.is_cuda() && grad_out.scalar_type() == torch::kFloat,
              "dpa4_so2_conv_backward: grad_out must be cuda fp32");

  grad_out = grad_out.contiguous();
  z_all = z_all.contiguous();
  src = src.to(torch::kLong).contiguous();
  dst = dst.to(torch::kLong).contiguous();
  runs = runs.contiguous();
  cb = cb.contiguous();
  alpha = alpha.contiguous();
  head_gate = head_gate.contiguous();
  rescale = rescale.contiguous();

  // The reverse sweep multiplies by the transpose, and replays the gate with
  // the forward orientation, so both orientations are packed for the
  // vectorized reduction.
  auto w0t = pack_reduction(w0.transpose(2, 3));
  auto w1t = pack_reduction(w1.transpose(2, 3));
  auto gwt = pack_reduction(gw.transpose(2, 3));
  w0 = pack_reduction(w0);
  w1 = pack_reduction(w1);
  gw = pack_reduction(gw);

  // The run and degree-kernel cotangents are accumulated across the focus
  // streams, so they start at zero, as does the node cotangent, which is
  // accumulated across incident edges.
  auto g_x = torch::zeros_like(x);
  auto g_runs = torch::zeros_like(runs);
  auto g_kc = torch::zeros_like(kc);
  auto g_alpha = torch::empty_like(alpha);
  if (c.n_node <= 0) {
    return {g_x, g_runs, g_kc, g_alpha};
  }

  auto order = src_order.contiguous();
  auto row_ptr = src_rowptr.contiguous();
  check_csr(order, row_ptr, c, "dpa4_so2_conv_backward");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto args = make_args(c, x, order, row_ptr, dst, runs, kc, cb, w0, w1, gw,
                        head_gate, rescale, z_all);
  args.alpha = alpha.data_ptr<float>();
  args.a_se = static_cast<int>(alpha.stride(0));
  args.a_sf = static_cast<int>(alpha.stride(1));
  args.a_sh = static_cast<int>(alpha.stride(2));
  const bool launched = dispatch_backward(
      c, args, grad_out.data_ptr<float>(), w0t.data_ptr<float>(),
      w1t.data_ptr<float>(), gwt.data_ptr<float>(), g_x.data_ptr<float>(),
      g_runs.data_ptr<float>(), g_kc.data_ptr<float>(),
      g_alpha.data_ptr<float>(), stream);
  TORCH_CHECK(
      launched,
      "dpa4_so2_conv_backward: no instantiation for the resolved shape");
  DPA4_CHECK_LAUNCH("dpa4_so2_conv_backward");
  return {g_x, g_runs, g_kc, g_alpha};
}

torch::Tensor dpa4_wigner_runs(torch::Tensor quat,
                               torch::Tensor mono_coeff,
                               torch::Tensor mono_exp,
                               int64_t lmax) {
  const at::cuda::OptionalCUDAGuard device_guard(quat.device());
  TORCH_CHECK(quat.dim() == 2 && quat.size(1) == 4 &&
                  quat.scalar_type() == torch::kFloat,
              "dpa4_wigner_runs: the quaternions must be (E, 4) fp32");
  const long nw = 3L * (lmax + 1) * (lmax + 1) - 2;
  TORCH_CHECK(mono_coeff.dim() == 2 && mono_coeff.size(0) == nw,
              "dpa4_wigner_runs: the run coefficients must be (NW, M)");
  TORCH_CHECK(mono_exp.scalar_type() == torch::kChar &&
                  mono_exp.numel() == mono_coeff.size(1) * 4,
              "dpa4_wigner_runs: the monomial exponents must be (M, 4) int8");
  return build_runs(quat.contiguous(), mono_coeff.contiguous(),
                    mono_exp.contiguous(), at::cuda::getCurrentCUDAStream());
}

/// Ridge point of the current device: peak fp32 FMA throughput over DRAM
/// bandwidth, in FLOP per byte.
///
/// The fused convolution trades memory traffic for float32 SIMT arithmetic,
/// so the arithmetic budget at which it stops paying scales with this ratio.
/// The Python routing gate normalizes its measured threshold by the ridge of
/// the card it was calibrated on, which makes the decision follow the actual
/// part rather than an architecture name. Attribute queries cover every
/// supported toolkit; the FMA-per-SM width is 128 lanes on every consumer and
/// data-center part since Ampere except the A100 die (64), which the
/// compute-capability pair identifies.
double dpa4_fp32_ridge() {
  int dev = 0;
  cudaGetDevice(&dev);
  int sm_count = 0, clock_khz = 0, mem_clock_khz = 0, bus_width = 0;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  const int lanes = (major < 8 || (major == 8 && minor == 0)) ? 64 : 128;
  const double flops = 2.0 * lanes * sm_count * (clock_khz * 1e3);
  const double bytes = (mem_clock_khz * 1e3) * (bus_width / 8.0) * 2.0;
  return bytes > 0.0 ? flops / bytes : 0.0;
}

torch::Tensor dpa4_wigner_runs_backward(torch::Tensor grad_runs,
                                        torch::Tensor quat,
                                        torch::Tensor dmono_coeff,
                                        torch::Tensor dmono_exp) {
  const at::cuda::OptionalCUDAGuard device_guard(quat.device());
  TORCH_CHECK(dmono_coeff.dim() == 3 && dmono_coeff.size(1) == 4 &&
                  dmono_coeff.size(0) == grad_runs.size(1),
              "dpa4_wigner_runs_backward: the derivative coefficients must be "
              "(NW, 4, M')");
  TORCH_CHECK(dmono_exp.scalar_type() == torch::kChar &&
                  dmono_exp.numel() == dmono_coeff.size(2) * 4,
              "dpa4_wigner_runs_backward: the derivative exponents must be "
              "(M', 4) int8");
  return contract_quat_grad(quat.contiguous(), grad_runs.contiguous(),
                            dmono_coeff.contiguous(), dmono_exp.contiguous(),
                            at::cuda::getCurrentCUDAStream());
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "dpa4_so2_conv(Tensor x, Tensor src, Tensor dst, Tensor dst_order, "
      "Tensor dst_rowptr, Tensor src_order, Tensor src_rowptr, Tensor runs, "
      "Tensor kc, Tensor cb, Tensor w0, Tensor w1, Tensor gw, Tensor q, "
      "Tensor k, Tensor logit_w, Tensor null_logit, Tensor env, Tensor rad0, "
      "Tensor fscale, Tensor head_gate, Tensor rescale, "
      "int lmax, int focus_dim, int rank) "
      "-> (Tensor out, Tensor alpha, Tensor pre_gate, Tensor z_all)");
  m.impl("dpa4_so2_conv", torch::kCUDA, &dpa4_so2_conv);
  m.def(
      "dpa4_so2_conv_backward(Tensor grad_out, Tensor z_all, Tensor x, "
      "Tensor src, Tensor dst, Tensor src_order, Tensor src_rowptr, "
      "Tensor runs, Tensor kc, Tensor cb, Tensor w0, Tensor w1, Tensor gw, "
      "Tensor alpha, Tensor head_gate, Tensor rescale, "
      "int lmax, int focus_dim, int rank) "
      "-> (Tensor g_x, Tensor g_runs, Tensor g_kc, Tensor g_alpha)");
  m.impl("dpa4_so2_conv_backward", torch::kCUDA, &dpa4_so2_conv_backward);
  m.def(
      "dpa4_wigner_runs(Tensor quat, Tensor mono_coeff, Tensor mono_exp, "
      "int lmax) -> Tensor");
  m.impl("dpa4_wigner_runs", torch::kCUDA, &dpa4_wigner_runs);
  m.def(
      "dpa4_wigner_runs_backward(Tensor grad_runs, Tensor quat, "
      "Tensor dmono_coeff, Tensor dmono_exp) -> Tensor");
  m.impl("dpa4_wigner_runs_backward", torch::kCUDA, &dpa4_wigner_runs_backward);
  m.def("dpa4_fp32_ridge() -> float", &dpa4_fp32_ridge);
}

namespace dpa4 {

/// Fail with the shape, the requested size and the CUDA error of one launch
/// step of the fused convolution.
void report_launch_failure(int lmax, int focus_dim, int bytes, int error) {
  int limit = 0;
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device);
  TORCH_CHECK(
      false, "dpa4_so2_conv: launching degree ", lmax, " at focus width ",
      focus_dim, " with ", bytes,
      " bytes of dynamic shared memory (device opt-in limit ", limit,
      ") failed: ", cudaGetErrorString(static_cast<cudaError_t>(error)));
}

}  // namespace dpa4
