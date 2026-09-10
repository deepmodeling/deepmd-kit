// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused cutoff envelope and radial basis for SeZM / DPA4 inference.
//
// Both quantities the edge cache derives from the pair distance are functions
// of that distance alone:
//
//   env[e]     = keep[e] * E_p1(r)
//   rbf[e, n]  = keep[e] * phi_n(r) * E_p2(r)
//
// with the C3 cutoff envelope written in its cancellation-free factorization
//
//   u = clamp((rcut - r) / rcut, 0, 1),  x = 1 - u,  E_p(r) = u^4 * S_p(x)
//
// where ``S_p`` is the Horner series of the positive binomial coefficients, and
// the basis either Bessel, ``phi_n(r) = sin(r f_n) / r``, or Gaussian,
// ``phi_n(r) = exp(k (r - c_n)^2)`` with ``k < 0``.
//
// Fused because the compiler otherwise inlines the whole chain into every
// consumer of ``env`` and ``rbf`` and re-evaluates it there, which turns a
// 96 MB pass into several. One thread owns one edge, evaluates both envelopes
// once, and streams the basis straight out.
//
// The backward contracts every cotangent back onto the distance analytically;
// the basis frequencies are inference-time constants and take no gradient.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <tuple>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxSeries = 16;

#define DPA4_CHECK_LAUNCH(what)                                           \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

/// Basis families with an implementation.
enum BasisType : int { kBessel = 0, kGaussian = 1 };

/// The C3 envelope and its derivative with respect to the distance.
///
/// ``u`` saturates outside the cutoff, where both the value and the derivative
/// are identically zero, which is what makes the potential energy surface C3
/// continuous at ``rcut``.
__device__ __forceinline__ void envelope_pair(float r,
                                              float inv_rcut,
                                              const float* series,
                                              int order,
                                              float& value,
                                              float& derivative) {
  const float u = fminf(fmaxf((1.f - r * inv_rcut), 0.f), 1.f);
  const float x = 1.f - u;
  float s = series[order - 1];
  float ds = 0.f;
  for (int k = order - 2; k >= 0; --k) {
    ds = fmaf(x, ds, s);
    s = fmaf(x, s, series[k]);
  }
  const float u2 = u * u;
  const float u3 = u2 * u;
  value = u3 * u * s;
  // du/dr = -1/rcut and dx/dr = +1/rcut inside the cutoff, giving
  // d/dr [u^4 S] = u^3 (u S' - 4 S) / rcut. Outside, u is clamped and both the
  // value and the derivative vanish with it, which is the C3 contact.
  derivative = (u > 0.f) ? (u3 * fmaf(u, ds, -4.f * s)) * inv_rcut : 0.f;
}

__global__ __launch_bounds__(kThreads) void edge_radial_fwd_kernel(
    const float* __restrict__ edge_len,  // (E,)
    const float* __restrict__ keep,      // (E,)
    const float* __restrict__ freqs,     // (n_radial,)
    const float* __restrict__ env_series,
    const float* __restrict__ rbf_series,
    float* __restrict__ env,  // (E,)
    float* __restrict__ rbf,  // (E, n_radial)
    long n_edge,
    int n_radial,
    int env_order,
    int rbf_order,
    float inv_rcut,
    float gaussian_coeff,
    int basis) {
  extern __shared__ float shared[];
  float* s_env = shared;
  float* s_rbf = shared + kMaxSeries;
  float* s_freq = shared + 2 * kMaxSeries;
  for (int i = threadIdx.x; i < env_order; i += kThreads) {
    s_env[i] = env_series[i];
  }
  for (int i = threadIdx.x; i < rbf_order; i += kThreads) {
    s_rbf[i] = rbf_series[i];
  }
  for (int i = threadIdx.x; i < n_radial; i += kThreads) {
    s_freq[i] = freqs[i];
  }
  __syncthreads();

  for (long e = blockIdx.x * static_cast<long>(kThreads) + threadIdx.x;
       e < n_edge; e += static_cast<long>(kThreads) * gridDim.x) {
    const float r = edge_len[e];
    const float mask = keep[e];
    float e1 = 0.f;
    float d1 = 0.f;
    envelope_pair(r, inv_rcut, s_env, env_order, e1, d1);
    env[e] = mask * e1;

    float e2 = 0.f;
    float d2 = 0.f;
    envelope_pair(r, inv_rcut, s_rbf, rbf_order, e2, d2);
    const float scale = mask * e2;
    float* row = rbf + e * static_cast<long>(n_radial);
    if (basis == kBessel) {
      const float inv_r = 1.f / r;
      for (int n = 0; n < n_radial; ++n) {
        row[n] = scale * sinf(r * s_freq[n]) * inv_r;
      }
    } else {
      for (int n = 0; n < n_radial; ++n) {
        const float dr = r - s_freq[n];
        row[n] = scale * expf(dr * dr * gaussian_coeff);
      }
    }
  }
}

__global__ __launch_bounds__(kThreads) void edge_radial_bwd_kernel(
    const float* __restrict__ grad_env,  // (E,)
    const float* __restrict__ grad_rbf,  // (E, n_radial)
    const float* __restrict__ edge_len,  // (E,)
    const float* __restrict__ keep,      // (E,)
    const float* __restrict__ freqs,     // (n_radial,)
    const float* __restrict__ env_series,
    const float* __restrict__ rbf_series,
    float* __restrict__ grad_len,  // (E,)
    long n_edge,
    int n_radial,
    int env_order,
    int rbf_order,
    float inv_rcut,
    float gaussian_coeff,
    int basis) {
  extern __shared__ float shared[];
  float* s_env = shared;
  float* s_rbf = shared + kMaxSeries;
  float* s_freq = shared + 2 * kMaxSeries;
  for (int i = threadIdx.x; i < env_order; i += kThreads) {
    s_env[i] = env_series[i];
  }
  for (int i = threadIdx.x; i < rbf_order; i += kThreads) {
    s_rbf[i] = rbf_series[i];
  }
  for (int i = threadIdx.x; i < n_radial; i += kThreads) {
    s_freq[i] = freqs[i];
  }
  __syncthreads();

  for (long e = blockIdx.x * static_cast<long>(kThreads) + threadIdx.x;
       e < n_edge; e += static_cast<long>(kThreads) * gridDim.x) {
    const float r = edge_len[e];
    const float mask = keep[e];
    float e1 = 0.f;
    float d1 = 0.f;
    envelope_pair(r, inv_rcut, s_env, env_order, e1, d1);
    float total = grad_env[e] * mask * d1;

    float e2 = 0.f;
    float d2 = 0.f;
    envelope_pair(r, inv_rcut, s_rbf, rbf_order, e2, d2);
    const float* row = grad_rbf + e * static_cast<long>(n_radial);
    const float inv_r = 1.f / r;
    for (int n = 0; n < n_radial; ++n) {
      float phi = 0.f;
      float dphi = 0.f;
      if (basis == kBessel) {
        float sine = 0.f;
        float cosine = 0.f;
        sincosf(r * s_freq[n], &sine, &cosine);
        phi = sine * inv_r;
        // d/dr [sin(r f) / r] = (f cos(r f) - sin(r f) / r) / r. The two terms
        // cancel to leading order at large ``r f``, so the difference is formed
        // with a fused multiply-add to keep the rounding to one step.
        dphi = fmaf(s_freq[n], cosine, -phi) * inv_r;
      } else {
        const float dr = r - s_freq[n];
        phi = expf(dr * dr * gaussian_coeff);
        dphi = phi * 2.f * dr * gaussian_coeff;
      }
      total = fmaf(row[n] * mask, fmaf(dphi, e2, phi * d2), total);
    }
    grad_len[e] = total;
  }
}

void check_inputs(const torch::Tensor& edge_len,
                  const torch::Tensor& keep,
                  const torch::Tensor& freqs,
                  const torch::Tensor& env_series,
                  const torch::Tensor& rbf_series) {
  TORCH_CHECK(edge_len.is_cuda() && edge_len.scalar_type() == torch::kFloat,
              "dpa4_edge_radial: the distance must be cuda fp32");
  TORCH_CHECK(keep.numel() == edge_len.numel(),
              "dpa4_edge_radial: one keep weight per edge");
  TORCH_CHECK(
      env_series.numel() <= kMaxSeries && rbf_series.numel() <= kMaxSeries,
      "dpa4_edge_radial: envelope order beyond the staged limit");
  TORCH_CHECK(env_series.numel() >= 2 && rbf_series.numel() >= 2,
              "dpa4_edge_radial: the envelope series needs at least two terms");
  TORCH_CHECK(freqs.numel() > 0,
              "dpa4_edge_radial: the basis must be non-empty");
}

unsigned block_count(long n_edge) {
  const long blocks = (n_edge + kThreads - 1) / kThreads;
  return static_cast<unsigned>(blocks > 65535 ? 65535 : blocks);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> dpa4_edge_radial(
    torch::Tensor edge_len,
    torch::Tensor keep,
    torch::Tensor freqs,
    torch::Tensor env_series,
    torch::Tensor rbf_series,
    double rcut,
    double gaussian_coeff,
    int64_t basis) {
  const at::cuda::OptionalCUDAGuard device_guard(edge_len.device());
  check_inputs(edge_len, keep, freqs, env_series, rbf_series);
  edge_len = edge_len.contiguous().reshape({-1});
  keep = keep.contiguous().reshape({-1});
  freqs = freqs.contiguous().reshape({-1});
  env_series = env_series.contiguous();
  rbf_series = rbf_series.contiguous();

  const long n_edge = edge_len.numel();
  const int n_radial = static_cast<int>(freqs.numel());
  auto env = torch::empty({n_edge, 1}, edge_len.options());
  auto rbf = torch::empty({n_edge, n_radial}, edge_len.options());
  if (n_edge == 0) {
    return {env, rbf};
  }
  const int shared =
      (2 * kMaxSeries + n_radial) * static_cast<int>(sizeof(float));
  edge_radial_fwd_kernel<<<dim3(block_count(n_edge)), dim3(kThreads), shared,
                           at::cuda::getCurrentCUDAStream()>>>(
      edge_len.data_ptr<float>(), keep.data_ptr<float>(),
      freqs.data_ptr<float>(), env_series.data_ptr<float>(),
      rbf_series.data_ptr<float>(), env.data_ptr<float>(),
      rbf.data_ptr<float>(), n_edge, n_radial,
      static_cast<int>(env_series.numel()),
      static_cast<int>(rbf_series.numel()), static_cast<float>(1.0 / rcut),
      static_cast<float>(gaussian_coeff), static_cast<int>(basis));
  DPA4_CHECK_LAUNCH("dpa4_edge_radial");
  return {env, rbf};
}

torch::Tensor dpa4_edge_radial_backward(torch::Tensor grad_env,
                                        torch::Tensor grad_rbf,
                                        torch::Tensor edge_len,
                                        torch::Tensor keep,
                                        torch::Tensor freqs,
                                        torch::Tensor env_series,
                                        torch::Tensor rbf_series,
                                        double rcut,
                                        double gaussian_coeff,
                                        int64_t basis) {
  const at::cuda::OptionalCUDAGuard device_guard(edge_len.device());
  check_inputs(edge_len, keep, freqs, env_series, rbf_series);
  grad_env = grad_env.contiguous().reshape({-1});
  grad_rbf = grad_rbf.contiguous();
  edge_len = edge_len.contiguous().reshape({-1});
  keep = keep.contiguous().reshape({-1});
  freqs = freqs.contiguous().reshape({-1});
  env_series = env_series.contiguous();
  rbf_series = rbf_series.contiguous();

  const long n_edge = edge_len.numel();
  const int n_radial = static_cast<int>(freqs.numel());
  auto grad_len = torch::empty({n_edge, 1}, edge_len.options());
  if (n_edge == 0) {
    return grad_len;
  }
  const int shared =
      (2 * kMaxSeries + n_radial) * static_cast<int>(sizeof(float));
  edge_radial_bwd_kernel<<<dim3(block_count(n_edge)), dim3(kThreads), shared,
                           at::cuda::getCurrentCUDAStream()>>>(
      grad_env.data_ptr<float>(), grad_rbf.data_ptr<float>(),
      edge_len.data_ptr<float>(), keep.data_ptr<float>(),
      freqs.data_ptr<float>(), env_series.data_ptr<float>(),
      rbf_series.data_ptr<float>(), grad_len.data_ptr<float>(), n_edge,
      n_radial, static_cast<int>(env_series.numel()),
      static_cast<int>(rbf_series.numel()), static_cast<float>(1.0 / rcut),
      static_cast<float>(gaussian_coeff), static_cast<int>(basis));
  DPA4_CHECK_LAUNCH("dpa4_edge_radial_backward");
  return grad_len;
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "dpa4_edge_radial(Tensor edge_len, Tensor keep, Tensor freqs, "
      "Tensor env_series, Tensor rbf_series, float rcut, "
      "float gaussian_coeff, int basis) -> (Tensor env, Tensor rbf)");
  m.impl("dpa4_edge_radial", torch::kCUDA, &dpa4_edge_radial);
  m.def(
      "dpa4_edge_radial_backward(Tensor grad_env, Tensor grad_rbf, "
      "Tensor edge_len, Tensor keep, Tensor freqs, Tensor env_series, "
      "Tensor rbf_series, float rcut, float gaussian_coeff, int basis) "
      "-> Tensor");
  m.impl("dpa4_edge_radial_backward", torch::kCUDA, &dpa4_edge_radial_backward);
}
