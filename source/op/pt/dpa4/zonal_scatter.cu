// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused geometric initial embedding for SeZM / DPA4 inference.
//
// The initial embedding broadcasts one radial feature per packed non-scalar
// row, scales it by the zonal coupling of that row, and reduces the result over
// the incident edges of every destination node:
//
//   out[n, r, c] = sum_{dst[e] = n} zonal[e, r] * radial[e, slot[r], c]
//
// Written with tensor operations this materializes the per-edge message, an
// ``(E, R, C)`` tensor that is 1.3 GB at the production shape and is written
// once and read once by the scatter. Here the message never leaves registers: a
// warp owns one node, walks its incidence list through the destination CSR the
// convolution already builds, and accumulates straight into the node tile. The
// device traffic drops to the two operands and the node result.
//
// Layout. ``R = (lmax + 1)^2 - 1`` packed non-scalar rows carry degrees
// ``1..lmax`` in packed order, so row ``r`` reuses radial degree ``slot[r]``
// and the rows of one degree are contiguous, which is what makes the repeated
// radial reads hit L1. A lane owns one channel of a 32-wide block and holds the
// whole row tile in registers; wider channel counts sweep the edge list once
// per block, and because a block only reads its own channels the device traffic
// is unchanged.
//
// The reduction follows the CSR order, so it is bitwise reproducible and needs
// no atomics.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <tuple>

namespace {

constexpr int kWarp = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarp * kWarps;
constexpr int kMaxLmax = 6;

#define DPA4_CHECK_LAUNCH(what)                                           \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

/// Packed non-scalar row count of degrees ``1..LMAX``.
template <int LMAX>
constexpr int row_count() {
  return (LMAX + 1) * (LMAX + 1) - 1;
}

/// Radial slot row ``r`` reuses.
///
/// Degree ``l`` owns the packed non-scalar rows from ``l^2 - 1`` upward, and
/// the radial feature carries degrees ``1..lmax`` at slots ``0..lmax - 1``.
template <int LMAX>
constexpr int row_slot(int r) {
  int l = 1;
  while (l < LMAX && (l + 1) * (l + 1) - 1 <= r) {
    ++l;
  }
  return l - 1;
}

/// One warp per node, one lane per channel of a 32-wide block.
///
/// The output carries the full packed node layout already normalized: row zero
/// is the scalar coefficient, which this embedding leaves at zero, and rows
/// ``1..R`` hold the reduction scaled by the smooth degree ``node_scale``.
/// Emitting the padded and scaled tile here saves the caller a concatenation
/// and a second full-size pass.
///
/// ``node_scale`` is not a constant: it is the inverse square root of a sum
/// over the cutoff envelope, so it carries a gradient back to the geometry.
/// The backward returns it, reconstructing the unscaled reduction from the
/// saved output, which is exact because the degree floor keeps the scale
/// strictly positive.
template <int LMAX>
__global__ __launch_bounds__(kThreads) void zonal_scatter_fwd_kernel(
    const float* __restrict__ zonal,    // (E, R)
    const float* __restrict__ radial,   // (E, L, C)
    const int64_t* __restrict__ order,  // (E,) destination CSR permutation
    const int64_t* __restrict__ row_ptr,
    const float* __restrict__ node_scale,  // (N,)
    float* __restrict__ out,               // (N, R + 1, C)
    int n_node,
    int n_channel,
    int n_slot) {
  constexpr int R = row_count<LMAX>();
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (kWarp - 1);
  const int node = blockIdx.x * kWarps + warp;
  if (node >= n_node) {
    return;
  }
  const long begin = row_ptr[node];
  const long end = row_ptr[node + 1];
  const float scale = node_scale[node];

  for (int block = 0; block < n_channel; block += kWarp) {
    const int channel = block + lane;
    if (channel >= n_channel) {
      break;
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r] = 0.f;
    }
    for (long p = begin; p < end; ++p) {
      const long edge = order[p];
      const float* rad =
          radial + edge * static_cast<long>(n_slot) * n_channel + channel;
      const float* zon = zonal + edge * R;
      // Rows of one degree are contiguous, so the repeated radial load of a
      // degree block is one L1 hit after the first.
#pragma unroll
      for (int r = 0; r < R; ++r) {
        acc[r] =
            fmaf(zon[r], rad[static_cast<long>(row_slot<LMAX>(r)) * n_channel],
                 acc[r]);
      }
    }
    float* dst = out + static_cast<long>(node) * (R + 1) * n_channel + channel;
    dst[0] = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      dst[static_cast<long>(r + 1) * n_channel] = acc[r] * scale;
    }
  }
}

/// One warp per edge: both cotangents are per-edge quantities that read one
/// destination node tile, which the node feature's small footprint keeps in
/// cache.
template <int LMAX>
__global__ __launch_bounds__(kThreads) void zonal_scatter_bwd_kernel(
    const float* __restrict__ grad_out,    // (N, R + 1, C)
    const float* __restrict__ zonal,       // (E, R)
    const float* __restrict__ radial,      // (E, L, C)
    const int64_t* __restrict__ dst,       // (E,)
    const float* __restrict__ node_scale,  // (N,)
    float* __restrict__ g_zonal,           // (E, R)
    float* __restrict__ g_radial,          // (E, L, C)
    long n_edge,
    int n_channel,
    int n_slot) {
  constexpr int R = row_count<LMAX>();
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (kWarp - 1);
  const long edge = static_cast<long>(blockIdx.x) * kWarps + warp;
  if (edge >= n_edge) {
    return;
  }
  const long node = dst[edge];
  // Row zero of the node tile is the scalar coefficient this embedding does
  // not write, so the cotangent of packed row ``r`` sits one row further on and
  // carries the normalization the forward applied.
  const float scale = node_scale[node];
  const float* gout =
      grad_out + node * static_cast<long>(R + 1) * n_channel + n_channel;
  const float* rad = radial + edge * static_cast<long>(n_slot) * n_channel;
  const float* zon = zonal + edge * R;
  float* g_rad = g_radial + edge * static_cast<long>(n_slot) * n_channel;
  float* g_zon = g_zonal + edge * R;

  // The zonal cotangent contracts the whole channel axis, so its partials
  // accumulate across the channel blocks and are written once at the end.
  float g_zon_acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    g_zon_acc[r] = 0.f;
  }
  for (int block = 0; block < n_channel; block += kWarp) {
    const int channel = block + lane;
    const bool live = channel < n_channel;
    // The radial cotangent gathers every row that shares a slot, so it is
    // accumulated per slot and written once per channel block.
    float g_slot[LMAX];
#pragma unroll
    for (int l = 0; l < LMAX; ++l) {
      g_slot[l] = 0.f;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int l = row_slot<LMAX>(r);
      const float g =
          live ? gout[static_cast<long>(r) * n_channel + channel] * scale : 0.f;
      g_slot[l] = fmaf(g, zon[r], g_slot[l]);
      float partial =
          live ? g * rad[static_cast<long>(l) * n_channel + channel] : 0.f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
      }
      g_zon_acc[r] += partial;
    }
    if (live) {
#pragma unroll
      for (int l = 0; l < LMAX; ++l) {
        if (l < n_slot) {
          g_rad[static_cast<long>(l) * n_channel + channel] = g_slot[l];
        }
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      g_zon[r] = g_zon_acc[r];
    }
  }
}

void check_inputs(const torch::Tensor& zonal,
                  const torch::Tensor& radial,
                  int lmax) {
  TORCH_CHECK(zonal.is_cuda() && zonal.scalar_type() == torch::kFloat &&
                  radial.scalar_type() == torch::kFloat,
              "dpa4_zonal_scatter: operands must be cuda fp32");
  TORCH_CHECK(zonal.dim() == 2 && radial.dim() == 3,
              "dpa4_zonal_scatter: zonal must be (E, R) and radial (E, L, C)");
  TORCH_CHECK(zonal.size(0) == radial.size(0),
              "dpa4_zonal_scatter: operands must share the edge axis");
  TORCH_CHECK(1 <= lmax && lmax <= kMaxLmax,
              "dpa4_zonal_scatter: degree out of the instantiated range");
  TORCH_CHECK(zonal.size(1) == (lmax + 1) * (lmax + 1) - 1,
              "dpa4_zonal_scatter: the row count must match the degree");
  TORCH_CHECK(radial.size(1) >= lmax,
              "dpa4_zonal_scatter: the radial feature must hold degrees "
              "1..lmax at slots 0..lmax-1");
}

/// Degree implied by the packed row count.
int lmax_of(long n_row) {
  for (int l = 1; l <= kMaxLmax; ++l) {
    if ((l + 1) * (l + 1) - 1 == n_row) {
      return l;
    }
  }
  return 0;
}

#define DPA4_ZONAL_FOR_EACH_LMAX(macro) \
  macro(1) macro(2) macro(3) macro(4) macro(5) macro(6)

}  // namespace

/// Forward entry. ``dst`` is not read here: the forward walks the CSR view,
/// while the edge-parallel backward wants the raw destination index, and an
/// input a backward needs has to appear in the operator's signature.
torch::Tensor dpa4_zonal_scatter(torch::Tensor zonal,
                                 torch::Tensor radial,
                                 torch::Tensor dst,
                                 torch::Tensor dst_order,
                                 torch::Tensor dst_rowptr,
                                 torch::Tensor node_scale,
                                 c10::SymInt node_count) {
  const at::cuda::OptionalCUDAGuard device_guard(zonal.device());
  const int lmax = lmax_of(zonal.size(1));
  check_inputs(zonal, radial, lmax);
  zonal = zonal.contiguous();
  radial = radial.contiguous();
  dst_order = dst_order.contiguous();
  dst_rowptr = dst_rowptr.contiguous();
  node_scale = node_scale.contiguous().reshape({-1});

  const int n_node = static_cast<int>(node_count.expect_int());
  const int n_row = static_cast<int>(zonal.size(1));
  const int n_slot = static_cast<int>(radial.size(1));
  const int n_channel = static_cast<int>(radial.size(2));
  // The packed node layout carries the scalar row the embedding leaves zero.
  auto out = torch::zeros({n_node, n_row + 1, n_channel}, radial.options());
  if (n_node == 0 || n_channel == 0) {
    return out;
  }
  TORCH_CHECK(dst_rowptr.numel() == n_node + 1,
              "dpa4_zonal_scatter: the row pointer must have N + 1 entries");
  TORCH_CHECK(node_scale.numel() == n_node,
              "dpa4_zonal_scatter: one degree normalization per node");

  const unsigned blocks = static_cast<unsigned>((n_node + kWarps - 1) / kWarps);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define DPA4_LAUNCH_ZONAL_FWD(LV)                                              \
  case LV:                                                                     \
    zonal_scatter_fwd_kernel<LV><<<dim3(blocks), dim3(kThreads), 0, stream>>>( \
        zonal.data_ptr<float>(), radial.data_ptr<float>(),                     \
        dst_order.data_ptr<int64_t>(), dst_rowptr.data_ptr<int64_t>(),         \
        node_scale.data_ptr<float>(), out.data_ptr<float>(), n_node,           \
        n_channel, n_slot);                                                    \
    break;
  switch (lmax) {
    DPA4_ZONAL_FOR_EACH_LMAX(DPA4_LAUNCH_ZONAL_FWD)
    default:
      break;
  }
#undef DPA4_LAUNCH_ZONAL_FWD
  DPA4_CHECK_LAUNCH("dpa4_zonal_scatter");
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> dpa4_zonal_scatter_backward(
    torch::Tensor grad_out,
    torch::Tensor zonal,
    torch::Tensor radial,
    torch::Tensor dst,
    torch::Tensor node_scale) {
  const at::cuda::OptionalCUDAGuard device_guard(zonal.device());
  const int lmax = lmax_of(zonal.size(1));
  check_inputs(zonal, radial, lmax);
  grad_out = grad_out.contiguous();
  zonal = zonal.contiguous();
  radial = radial.contiguous();
  dst = dst.to(torch::kLong).contiguous();
  node_scale = node_scale.contiguous().reshape({-1});

  auto g_zonal = torch::empty_like(zonal);
  auto g_radial = torch::zeros_like(radial);
  const long n_edge = zonal.size(0);
  const int n_slot = static_cast<int>(radial.size(1));
  const int n_channel = static_cast<int>(radial.size(2));
  if (n_edge == 0 || n_channel == 0) {
    return {g_zonal, g_radial};
  }

  const unsigned blocks = static_cast<unsigned>((n_edge + kWarps - 1) / kWarps);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define DPA4_LAUNCH_ZONAL_BWD(LV)                                              \
  case LV:                                                                     \
    zonal_scatter_bwd_kernel<LV><<<dim3(blocks), dim3(kThreads), 0, stream>>>( \
        grad_out.data_ptr<float>(), zonal.data_ptr<float>(),                   \
        radial.data_ptr<float>(), dst.data_ptr<int64_t>(),                     \
        node_scale.data_ptr<float>(), g_zonal.data_ptr<float>(),               \
        g_radial.data_ptr<float>(), n_edge, n_channel, n_slot);                \
    break;
  switch (lmax) {
    DPA4_ZONAL_FOR_EACH_LMAX(DPA4_LAUNCH_ZONAL_BWD)
    default:
      break;
  }
#undef DPA4_LAUNCH_ZONAL_BWD
  DPA4_CHECK_LAUNCH("dpa4_zonal_scatter_backward");
  return {g_zonal, g_radial};
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "dpa4_zonal_scatter(Tensor zonal, Tensor radial, Tensor dst, "
      "Tensor dst_order, Tensor dst_rowptr, Tensor node_scale, "
      "SymInt node_count) -> Tensor");
  m.impl("dpa4_zonal_scatter", torch::kCUDA, &dpa4_zonal_scatter);
  m.def(
      "dpa4_zonal_scatter_backward(Tensor grad_out, Tensor zonal, "
      "Tensor radial, Tensor dst, Tensor node_scale) -> "
      "(Tensor g_zonal, Tensor g_radial)");
  m.impl("dpa4_zonal_scatter_backward", torch::kCUDA,
         &dpa4_zonal_scatter_backward);
}
