// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused energy fitting network for graph-lower inference. The operator is
// descriptor-agnostic: any graph-lowered energy model whose fitting is a
// plain MLP over the flat node axis dispatches here.
//   h_0 = act(x @ W_0 + b_0) * idt_0              (+ identity residual when
//   h_l = act(h_{l-1} @ W_l + b_l) * idt_l          the layer is square)
//   e   = h_{L-1} @ w_head + b_head + bias_atom_e[atype]   (fp64 output)
// The GEMMs run on cuBLAS in pedantic fp32 (TF32 off); each layer's bias,
// activation, timestep and residual collapse into one elementwise epilogue
// kernel that also stores the activation derivative for the backward. The
// backward (upstream d_e, a unit vector for the energy reduction) chains
//   dh_{L-1} = d_e * w_head^T
//   dpre_l   = dh_l * act'_l
//   dh_{l-1} = dpre_l @ W_l^T  (+ dh_l identity residual)
//   d_x      = dpre_0 @ W_0^T
// with the elementwise steps fused likewise.
//
// All tensors here are node-scale (atoms, not edges); the fusion removes
// kernel launches and aten glue rather than FLOPs. The head bias arrives as
// a device tensor so that symbolic tracing never reads a value host-side.

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <unordered_map>
#include <utility>
#include <vector>

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

// Activation value and derivative; codes follow
// deepmd.kernels.triton.dpa1.activation.ACT_CODES (0 = tanh, 1 = silu).
template <int ACT>
__device__ __forceinline__ float2 act_vg(float z) {
  if constexpr (ACT == 0) {
    float a = tanhf(z);
    return make_float2(a, 1.f - a * a);
  }
  const float s = 0.5f * (1.f + tanhf(0.5f * z));  // sigmoid via tanh identity
  return make_float2(z * s, s * (1.f + z * (1.f - s)));
}

// y = act(pre + b) * idt (+ x residual); adot = act' * idt is stored for the
// backward. float4 lanes; layer widths are multiples of four (Python gate).
template <int ACT>
__global__ void layer_epilogue_kernel(long total4,
                                      int dout,
                                      const float* __restrict__ pre,
                                      const float* __restrict__ b,
                                      const float* __restrict__ idt,
                                      const float* __restrict__ x,
                                      int residual,
                                      float* __restrict__ y,
                                      float* __restrict__ adot) {
  const long t4 = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t4 >= total4) {
    return;
  }
  const long t = t4 * 4;
  const int c = (int)(t % dout);
  const float4 p = *reinterpret_cast<const float4*>(pre + t);
  const float4 bb =
      b ? *reinterpret_cast<const float4*>(b + c) : make_float4(0, 0, 0, 0);
  const float4 ii =
      idt ? *reinterpret_cast<const float4*>(idt + c) : make_float4(1, 1, 1, 1);
  const float2 v0 = act_vg<ACT>(p.x + bb.x);
  const float2 v1 = act_vg<ACT>(p.y + bb.y);
  const float2 v2 = act_vg<ACT>(p.z + bb.z);
  const float2 v3 = act_vg<ACT>(p.w + bb.w);
  float4 yy = make_float4(v0.x * ii.x, v1.x * ii.y, v2.x * ii.z, v3.x * ii.w);
  if (residual) {
    const float4 xx = *reinterpret_cast<const float4*>(x + t);
    yy.x += xx.x;
    yy.y += xx.y;
    yy.z += xx.z;
    yy.w += xx.w;
  }
  *reinterpret_cast<float4*>(y + t) = yy;
  *reinterpret_cast<float4*>(adot + t) =
      make_float4(v0.y * ii.x, v1.y * ii.y, v2.y * ii.z, v3.y * ii.w);
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

// Backward seed: dh_{L-1}[n, c] = d_e[n] * w_head[c] (fp64 upstream).
__global__ void seed_kernel(long total4,
                            int dout,
                            const double* __restrict__ d_e,
                            const float* __restrict__ w_head,
                            float* __restrict__ dh) {
  const long t4 = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t4 >= total4) {
    return;
  }
  const long t = t4 * 4;
  const long n = t / dout;
  const int c = (int)(t % dout);
  const float de = (float)d_e[n];
  const float4 wv = *reinterpret_cast<const float4*>(w_head + c);
  *reinterpret_cast<float4*>(dh + t) =
      make_float4(de * wv.x, de * wv.y, de * wv.z, de * wv.w);
}

// Convert dh to dpre in place: dh *= adot.
__global__ void backward_epilogue_kernel(long total4,
                                         const float* __restrict__ dh,
                                         const float* __restrict__ adot,
                                         float* __restrict__ dpre) {
  const long t4 = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t4 >= total4) {
    return;
  }
  const long t = t4 * 4;
  const float4 d = *reinterpret_cast<const float4*>(dh + t);
  const float4 a = *reinterpret_cast<const float4*>(adot + t);
  *reinterpret_cast<float4*>(dpre + t) =
      make_float4(d.x * a.x, d.y * a.y, d.z * a.z, d.w * a.w);
}

long ceil_div(long a, long b) { return (a + b - 1) / b; }

}  // namespace

// Forward: per-atom energy (fp64 (N, 1)) plus the flat saved buffer of the
// activation derivatives -- adot chunks, chunk l a contiguous (N, width_l)
// sheet. The backward needs only these derivatives; the activations themselves
// stay in a forward-only ping-pong.
std::tuple<torch::Tensor, torch::Tensor> graph_fitting(
    torch::Tensor x,
    torch::Tensor atype,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> bs,
    std::vector<torch::Tensor> idts,
    std::vector<int64_t> resnets,
    torch::Tensor w_head,
    torch::Tensor b_head,
    torch::Tensor bias_atom_e,
    int64_t act) {
  const long n_node = x.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n_layer = (int)ws.size();
  std::vector<long> offset(n_layer + 1, 0);
  for (int l = 0; l < n_layer; ++l) {
    offset[l + 1] = offset[l] + ws[l].size(1);
  }
  const long total_width = offset[n_layer];
  auto f32 = x.options().dtype(torch::kFloat32);
  auto saved = torch::empty({n_node * total_width}, f32);
  auto e = torch::empty({n_node, 1}, x.options().dtype(torch::kFloat64));
  if (n_node == 0) {
    return {e, saved};
  }

  long width_max = 0;
  for (int l = 0; l < n_layer; ++l) {
    width_max = std::max(width_max, (long)ws[l].size(1));
  }
  // Two-slot ping-pong for the activations: layer l writes slot ``l & 1`` while
  // reading the previous layer's slot, so an activation is overwritten only
  // after the next GEMM has consumed it (kernels run in stream order).
  auto act_buf = torch::empty({2, n_node, width_max}, f32);
  float* act_slot[2] = {act_buf[0].data_ptr<float>(),
                        act_buf[1].data_ptr<float>()};
  const float* cur = x.data_ptr<float>();
  int din = (int)x.size(1);
  for (int l = 0; l < n_layer; ++l) {
    const int dout = (int)ws[l].size(1);
    float* h = act_slot[l & 1];
    float* adot = saved.data_ptr<float>() + offset[l] * n_node;
    gemm_nn(stream, cur, ws[l].data_ptr<float>(), h, (int)n_node, dout, din);
    const long total4 = n_node * dout / 4;
    const bool residual = resnets[l] && dout == din;
    auto launch = [&](auto act_tag) {
      layer_epilogue_kernel<decltype(act_tag)::value>
          <<<ceil_div(total4, 256), 256, 0, stream>>>(
              total4, dout, h,
              bs[l].numel() ? bs[l].data_ptr<float>() : nullptr,
              idts[l].numel() ? idts[l].data_ptr<float>() : nullptr, cur,
              residual ? 1 : 0, h, adot);
    };
    if (act == 0) {
      launch(std::integral_constant<int, 0>{});
    } else {
      launch(std::integral_constant<int, 1>{});
    }
    FITTING_CHECK_LAUNCH("graph_fitting layer");
    cur = h;
    din = dout;
  }
  head_kernel<<<ceil_div(n_node, 256), 256, 0, stream>>>(
      n_node, din, cur, w_head.data_ptr<float>(),
      b_head.numel() ? b_head.data_ptr<float>() : nullptr,
      bias_atom_e.data_ptr<double>(), atype.data_ptr<long>(),
      e.data_ptr<double>());
  FITTING_CHECK_LAUNCH("graph_fitting head");
  return {e, saved};
}

// Backward: d_x from the upstream d_e (fp64 (N, 1)). The saved derivative
// extent and fitting widths determine the output shape, so the descriptor is
// not retained solely for shape metadata. Two ping-pong dh buffers walk the
// layers from the head down.
void graph_fitting_backward_core(torch::Tensor d_e,
                                 torch::Tensor saved,
                                 std::vector<torch::Tensor> ws,
                                 std::vector<int64_t> resnets,
                                 torch::Tensor w_head,
                                 torch::Tensor d_x) {
  long total_width = 0;
  for (const auto& weight : ws) {
    total_width += weight.size(1);
  }
  TORCH_CHECK(total_width > 0 && saved.numel() % total_width == 0,
              "graph_fitting_backward: saved derivative buffer does not match "
              "the fitting widths");
  const long n_node = saved.numel() / total_width;
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
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n_layer = (int)ws.size();
  std::vector<long> offset(n_layer + 1, 0);
  for (int l = 0; l < n_layer; ++l) {
    offset[l + 1] = offset[l] + ws[l].size(1);
  }
  auto f32 = saved.options().dtype(torch::kFloat32);
  auto d_e_c = d_e.contiguous();

  long width_max = 0;
  for (int l = 0; l < n_layer; ++l) {
    width_max = std::max(width_max, (long)ws[l].size(1));
  }
  auto dh = torch::empty({n_node, width_max}, f32);
  auto dh_next = torch::empty({n_node, width_max}, f32);

  {
    const int dout = (int)ws[n_layer - 1].size(1);
    seed_kernel<<<ceil_div(n_node * dout / 4, 256), 256, 0, stream>>>(
        n_node * dout / 4, dout, d_e_c.data_ptr<double>(),
        w_head.data_ptr<float>(), dh.data_ptr<float>());
    FITTING_CHECK_LAUNCH("graph_fitting seed");
  }
  for (int l = n_layer - 1; l >= 0; --l) {
    const int dout = (int)ws[l].size(1);
    const int din = (int)ws[l].size(0);
    const float* adot = saved.data_ptr<float>() + offset[l] * n_node;
    float* out = l > 0 ? dh_next.data_ptr<float>() : d_x.data_ptr<float>();
    const bool residual = resnets[l] && dout == din;
    float beta = 0.f;
    if (residual) {
      // Identity bypass: dh_{l-1} starts from dh_l.
      cudaMemcpyAsync(out, dh.data_ptr<float>(), sizeof(float) * n_node * din,
                      cudaMemcpyDeviceToDevice, stream);
      beta = 1.f;
    }
    backward_epilogue_kernel<<<ceil_div(n_node * dout / 4, 256), 256, 0,
                               stream>>>(
        n_node * dout / 4, dh.data_ptr<float>(), adot, dh.data_ptr<float>());
    FITTING_CHECK_LAUNCH("graph_fitting backward layer");
    gemm_nt(stream, dh.data_ptr<float>(), ws[l].data_ptr<float>(), out,
            (int)n_node, din, dout, beta);
    if (l > 0) {
      std::swap(dh, dh_next);
    }
  }
}

torch::Tensor graph_fitting_backward(torch::Tensor d_e,
                                     torch::Tensor saved,
                                     std::vector<torch::Tensor> ws,
                                     std::vector<int64_t> resnets,
                                     torch::Tensor w_head) {
  long total_width = 0;
  for (const auto& weight : ws) {
    total_width += weight.size(1);
  }
  TORCH_CHECK(total_width > 0 && saved.numel() % total_width == 0,
              "graph_fitting_backward: saved derivative buffer does not match "
              "the fitting widths");
  const long n_node = saved.numel() / total_width;
  auto d_x = torch::empty({n_node, ws[0].size(0)}, saved.options());
  graph_fitting_backward_core(d_e, saved, std::move(ws), std::move(resnets),
                              w_head, d_x);
  return d_x;
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "graph_fitting(Tensor x, Tensor atype, Tensor[] ws, Tensor[] bs, "
      "Tensor[] idts, int[] resnets, Tensor w_head, Tensor b_head, "
      "Tensor bias_atom_e, int act) -> (Tensor e, Tensor saved)");
  m.impl("graph_fitting", torch::kCUDA, &graph_fitting);
  m.def(
      "graph_fitting_backward(Tensor d_e, Tensor saved, "
      "Tensor[] ws, int[] resnets, Tensor w_head) -> Tensor");
  m.impl("graph_fitting_backward", torch::kCUDA, &graph_fitting_backward);
}
