// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused energy fitting network for graph-lower inference. The operator is
// descriptor-agnostic: any graph-lowered energy model whose fitting is a
// plain MLP over the flat node axis dispatches here.
//   h_0 = act(x @ W_0 + b_0)                      (+ identity residual when
//   h_l = act(h_{l-1} @ W_l + b_l)                  the layer is square)
//   e   = h_{L-1} @ w_head + b_head + bias_atom_e[atype]   (fp64 output)
// The GEMMs run on cuBLAS in pedantic fp32 (TF32 off); each layer's bias,
// activation and residual collapse into one elementwise epilogue kernel.
// The backward (upstream d_e, a unit vector for the energy reduction) chains
//   dh_{L-1} = d_e * w_head^T
//   dpre_l   = dh_l * act'(pre_l + b_l)
//   dh_{l-1} = dpre_l @ W_l^T  (+ dh_l identity residual)
//   d_x      = dpre_0 @ W_0^T
// with the elementwise steps fused likewise.
//
// The saved state is the pre-activation of every layer, which each GEMM writes
// directly. The backward re-derives the activation derivative from it, so the
// epilogue writes only the activation and the layer costs one full-tensor
// store less than a formulation that also materializes the derivative.
//
// All tensors here are node-scale (atoms, not edges); the fusion removes
// kernel launches and aten glue rather than FLOPs. The head bias arrives as
// a device tensor so that symbolic tracing never reads a value host-side.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph_ops.h"

namespace {

#define FITTING_CHECK_LAUNCH(what)                                        \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

cublasHandle_t cublas_handle() {
  // A cuBLAS handle is device-bound and unsafe to share across threads, so
  // cache one per device in thread-local storage. Pedantic math keeps the fp32
  // potential-energy surface exact (no TF32, no split-K reordering) for MD.
  thread_local std::unordered_map<int, cublasHandle_t> handles;
  int device = 0;
  cudaGetDevice(&device);
  cublasHandle_t& h = handles[device];
  if (!h) {
    cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
  }
  return h;
}

// Row-major C(m, n) = A(m, k) @ B(k, n) + beta * C.
void gemm_nn(cudaStream_t stream,
             const float* a,
             const float* b,
             float* c,
             int m,
             int n,
             int k,
             float beta = 0.f) {
  cublasSetStream(cublas_handle(), stream);
  const float alpha = 1.f;
  cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n,
              a, k, &beta, c, n);
}

// Row-major C(m, n) = A(m, k) @ B(n, k)^T + beta * C.
void gemm_nt(cudaStream_t stream,
             const float* a,
             const float* b,
             float* c,
             int m,
             int n,
             int k,
             float beta = 0.f) {
  cublasSetStream(cublas_handle(), stream);
  const float alpha = 1.f;
  cublasSgemm(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, k,
              a, k, &beta, c, n);
}

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

// Activation codes follow deepmd.kernels.triton.dpa1.activation.ACT_CODES
// (0 = tanh, 1 = silu). Value and derivative are separate because the forward
// needs only the former and the backward only the latter.
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

// The node axis is long, so deriving a channel from a flat element index
// costs a 64-bit division per thread. The elementwise epilogues instead index
// (channel, node) directly: threadIdx.x selects the float4 lane within a row,
// keeping a warp's accesses contiguous, and the remaining dimensions walk the
// node axis. Layer widths are multiples of four (Python gate).
struct ElementwiseLaunch {
  dim3 grid;
  dim3 block;
};

ElementwiseLaunch elementwise_launch(long n_node, int dout) {
  const int lanes = dout / 4;
  const int rows = std::max(1, 256 / lanes);
  return {dim3((unsigned)ceil_div(n_node, rows)), dim3(lanes, rows)};
}

// y = act(pre + b) (+ x residual). The pre-activation stays where the GEMM
// wrote it, so the layer stores one tensor instead of two.
template <int ACT>
__global__ void layer_epilogue_kernel(long n_node,
                                      int dout,
                                      const float* __restrict__ pre,
                                      const float* __restrict__ b,
                                      const float* __restrict__ x,
                                      int residual,
                                      float* __restrict__ y) {
  const long node = blockIdx.x * (long)blockDim.y + threadIdx.y;
  if (node >= n_node) {
    return;
  }
  const int c = (int)threadIdx.x * 4;
  const long t = node * dout + c;
  const float4 p = *reinterpret_cast<const float4*>(pre + t);
  const float4 bb =
      b ? *reinterpret_cast<const float4*>(b + c) : make_float4(0, 0, 0, 0);
  float4 yy =
      make_float4(act_value<ACT>(p.x + bb.x), act_value<ACT>(p.y + bb.y),
                  act_value<ACT>(p.z + bb.z), act_value<ACT>(p.w + bb.w));
  if (residual) {
    const float4 xx = *reinterpret_cast<const float4*>(x + t);
    yy.x += xx.x;
    yy.y += xx.y;
    yy.z += xx.z;
    yy.w += xx.w;
  }
  *reinterpret_cast<float4*>(y + t) = yy;
}

// Energy head: e[n] = h[n] @ w_head + b_head + bias_atom_e[atype[n]].
__global__ void head_kernel(long n_node,
                            int width,
                            const float* __restrict__ h,
                            const float* __restrict__ w_head,
                            const float* __restrict__ b_head,
                            const double* __restrict__ bias_atom_e,
                            const long* __restrict__ atype,
                            double* __restrict__ e) {
  const long n = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (n >= n_node) {
    return;
  }
  const float* row = h + n * width;
  float acc = 0.f;
  for (int k = 0; k < width; k += 4) {
    const float4 hv = *reinterpret_cast<const float4*>(row + k);
    const float4 wv = *reinterpret_cast<const float4*>(w_head + k);
    acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
  }
  e[n] = (double)(acc + (b_head ? b_head[0] : 0.f)) + bias_atom_e[atype[n]];
}

// Construct the head gradient and apply the final hidden activation VJP in one
// pass. A square final layer also preserves the unmodified head gradient for
// its identity branch.
template <int ACT>
__global__ void seed_backward_epilogue_kernel(
    long n_node,
    int dout,
    const double* __restrict__ d_e,
    const float* __restrict__ w_head,
    const float* __restrict__ pre,
    const float* __restrict__ b,
    float* __restrict__ dpre,
    float* __restrict__ residual_out) {
  const long node = blockIdx.x * (long)blockDim.y + threadIdx.y;
  if (node >= n_node) {
    return;
  }
  const int c = (int)threadIdx.x * 4;
  const long t = node * dout + c;
  const float de = (float)d_e[node];
  const float4 wv = *reinterpret_cast<const float4*>(w_head + c);
  const float4 dh = make_float4(de * wv.x, de * wv.y, de * wv.z, de * wv.w);
  const float4 p = *reinterpret_cast<const float4*>(pre + t);
  const float4 bb =
      b ? *reinterpret_cast<const float4*>(b + c) : make_float4(0, 0, 0, 0);
  if (residual_out) {
    *reinterpret_cast<float4*>(residual_out + t) = dh;
  }
  *reinterpret_cast<float4*>(dpre + t) =
      make_float4(dh.x * act_derivative<ACT>(p.x + bb.x),
                  dh.y * act_derivative<ACT>(p.y + bb.y),
                  dh.z * act_derivative<ACT>(p.z + bb.z),
                  dh.w * act_derivative<ACT>(p.w + bb.w));
}

// Convert dh to dpre in place: dh *= act'(pre + b). A square residual layer
// also preserves the unmodified dh in its output buffer before the GEMM
// accumulates the weighted branch with beta = 1.
template <int ACT>
__global__ void backward_epilogue_kernel(long n_node,
                                         int dout,
                                         const float* __restrict__ dh,
                                         const float* __restrict__ pre,
                                         const float* __restrict__ b,
                                         float* __restrict__ dpre,
                                         float* __restrict__ residual_out) {
  const long node = blockIdx.x * (long)blockDim.y + threadIdx.y;
  if (node >= n_node) {
    return;
  }
  const int c = (int)threadIdx.x * 4;
  const long t = node * dout + c;
  const float4 d = *reinterpret_cast<const float4*>(dh + t);
  const float4 p = *reinterpret_cast<const float4*>(pre + t);
  const float4 bb =
      b ? *reinterpret_cast<const float4*>(b + c) : make_float4(0, 0, 0, 0);
  if (residual_out) {
    *reinterpret_cast<float4*>(residual_out + t) = d;
  }
  *reinterpret_cast<float4*>(dpre + t) =
      make_float4(d.x * act_derivative<ACT>(p.x + bb.x),
                  d.y * act_derivative<ACT>(p.y + bb.y),
                  d.z * act_derivative<ACT>(p.z + bb.z),
                  d.w * act_derivative<ACT>(p.w + bb.w));
}

/// Dispatch a kernel template over the two supported activation codes.
template <typename Fn>
void dispatch_activation(long act, Fn&& launch) {
  if (act == 0) {
    launch(std::integral_constant<int, 0>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }
}

}  // namespace

FittingLayerPlan fitting_layer_plan(const std::vector<torch::Tensor>& ws) {
  FittingLayerPlan plan{std::vector<long>(ws.size() + 1, 0), 0, (int)ws.size()};
  for (size_t l = 0; l < ws.size(); ++l) {
    plan.offset[l + 1] = plan.offset[l] + ws[l].size(1);
    plan.width_max = std::max(plan.width_max, (long)ws[l].size(1));
  }
  return plan;
}

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

}  // namespace

// Evaluate the network over one contiguous run of nodes. Every full-width
// tensor is indexed from the run's first node, so the same code serves the
// whole node axis and a single tile of it. ``saved`` and ``activation`` are
// sized for the run, not for the system.
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
  const float* cur = x;
  int din = (int)input_width;
  for (int l = 0; l < plan.n_layer; ++l) {
    const int dout = (int)ws[l].size(1);
    float* pre = saved + plan.offset[l] * run_nodes;
    float* y = activation[l & 1];
    gemm_nn(stream, cur, ws[l].data_ptr<float>(), pre, (int)run_nodes, dout,
            din);
    const ElementwiseLaunch shape = elementwise_launch(run_nodes, dout);
    const bool residual = resnets[l] && dout == din;
    dispatch_activation(act, [&](auto tag) {
      layer_epilogue_kernel<decltype(tag)::value>
          <<<shape.grid, shape.block, 0, stream>>>(
              run_nodes, dout, pre,
              bs[l].numel() ? bs[l].data_ptr<float>() : nullptr, cur,
              residual ? 1 : 0, y);
    });
    FITTING_CHECK_LAUNCH("graph_fitting layer");
    cur = y;
    din = dout;
  }
  head_kernel<<<ceil_div(run_nodes, 256), 256, 0, stream>>>(
      run_nodes, din, cur, w_head.data_ptr<float>(),
      b_head.numel() ? b_head.data_ptr<float>() : nullptr,
      bias_atom_e.data_ptr<double>(), atype, e);
  FITTING_CHECK_LAUNCH("graph_fitting head");
}

// Propagate the head cotangent of one run of nodes back to the input. ``dh``
// and ``dh_next`` are scratch of the run's size; ``d_x`` is indexed from the
// run's first node.
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
  for (int l = plan.n_layer - 1; l >= 0; --l) {
    const int dout = (int)ws[l].size(1);
    const int din = (int)ws[l].size(0);
    const float* pre = saved + plan.offset[l] * run_nodes;
    const float* b = bs[l].numel() ? bs[l].data_ptr<float>() : nullptr;
    float* out = l > 0 ? dh_next : d_x;
    const bool residual = resnets[l] && dout == din;
    const ElementwiseLaunch shape = elementwise_launch(run_nodes, dout);
    dispatch_activation(act, [&](auto tag) {
      constexpr int kAct = decltype(tag)::value;
      if (l == plan.n_layer - 1) {
        seed_backward_epilogue_kernel<kAct>
            <<<shape.grid, shape.block, 0, stream>>>(
                run_nodes, dout, d_e, w_head.data_ptr<float>(), pre, b, dh,
                residual ? out : nullptr);
      } else {
        backward_epilogue_kernel<kAct><<<shape.grid, shape.block, 0, stream>>>(
            run_nodes, dout, dh, pre, b, dh, residual ? out : nullptr);
      }
    });
    FITTING_CHECK_LAUNCH("graph_fitting backward layer");
    gemm_nt(stream, dh, ws[l].data_ptr<float>(), out, (int)run_nodes, din, dout,
            residual ? 1.f : 0.f);
    if (l > 0) {
      std::swap(dh, dh_next);
    }
  }
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
  // only after the next GEMM has consumed it (kernels run in stream order). A
  // single-layer network requires only its output slot.
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
  auto dh = torch::empty({n_node, plan.width_max}, f32);
  auto dh_next = plan.n_layer > 1 ? torch::empty({n_node, plan.width_max}, f32)
                                  : torch::empty({0}, f32);
  fitting_backward_range(at::cuda::getCurrentCUDAStream(), plan,
                         d_e_c.data_ptr<double>(), saved.data_ptr<float>(), ws,
                         bs, resnets, w_head, act, n_node, dh.data_ptr<float>(),
                         plan.n_layer > 1 ? dh_next.data_ptr<float>() : nullptr,
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
// run and reproduces the untiled sequence exactly.
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
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto saved = torch::empty({run * plan.saved_width()}, f32);
  // The backward reads only the saved pre-activations, never the activations,
  // so the forward ping-pong and the two cotangent buffers never hold live
  // data at the same time and share one allocation.
  auto scratch = torch::empty({slots, run, plan.width_max}, f32);
  float* slot[2] = {
      scratch[0].data_ptr<float>(),
      slots > 1 ? scratch[1].data_ptr<float>() : scratch[0].data_ptr<float>()};

  auto stream = at::cuda::getCurrentCUDAStream();
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

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "graph_fitting(Tensor x, Tensor atype, Tensor[] ws, Tensor[] bs, "
      "int[] resnets, Tensor w_head, Tensor b_head, Tensor bias_atom_e, "
      "int act) -> (Tensor e, Tensor saved)");
  m.impl("graph_fitting", torch::kCUDA, &graph_fitting);
  m.def(
      "graph_fitting_backward(Tensor d_e, Tensor saved, Tensor[] ws, "
      "Tensor[] bs, int[] resnets, Tensor w_head, int act) -> Tensor");
  m.impl("graph_fitting_backward", torch::kCUDA, &graph_fitting_backward);
  m.def(
      "graph_fitting_energy_gradient(Tensor(a!) x, Tensor atype, "
      "Tensor[] ws, Tensor[] bs, int[] resnets, Tensor w_head, "
      "Tensor b_head, Tensor bias_atom_e, int act, Tensor seed, int tile) "
      "-> Tensor");
  m.impl("graph_fitting_energy_gradient", torch::kCUDA,
         &graph_fitting_energy_gradient);
}
