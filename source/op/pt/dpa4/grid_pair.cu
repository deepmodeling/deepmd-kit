// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused grid pair product for SeZM / DPA4 inference.
//
// Every grid operator of the model -- the parameter-free node product, the
// polynomial grid MLP, and the branch mixer at a single branch -- evaluates the
// same core expression on coefficient operands:
//
//   out = from_grid( to_grid(left) * to_grid(right) )
//
// Written as three tensor contractions with ``P`` coefficient slots, ``G`` grid
// points and ``C`` channels,
//
//   lg[n, g, c] = sum_p T[g, p] * left[n, p, c]
//   rg[n, g, c] = sum_p T[g, p] * right[n, p, c]
//   out[n, p, c] = sum_g F[g, p] * lg[n, g, c] * rg[n, g, c]
//
// The grid field is the reason to fuse. Across the model zoo it is 6 to 10
// times larger than the coefficient operand that produces it -- 2.1 GB per call
// for 8000 nodes at the widest shape -- and, because the contraction is
// expressed as an einsum over non-adjacent axes, the compiler surrounds each
// multiply with full-size layout copies as well. Here the grid field never
// leaves registers: a warp walks the grid points holding both coefficient
// operands and the output accumulator, and device traffic drops to the operands
// and the result.
//
// Two resources bound that arrangement, and the zoo spans a wide enough range
// of
// ``P`` (12, 27, 48, 75, 108) that both bind:
//
//   registers  a lane holding whole coefficient vectors needs ``arrays * P`` of
//              them, which is 324 for the forward at ``P = 108``;
//   shared     staging both projectors costs ``2 * G * P`` floats, which is
//              297 KB at ``P = 108``, past any part.
//
// So the warp is split in two dimensions -- ``CPW`` channels by ``GROUP``
// coefficient slices -- and the projectors are staged one ``GB``-row block at a
// time. A lane then holds ``ceil(P / GROUP)`` coefficients and the block fits a
// fixed shared-memory budget, at the cost of one ``log2(GROUP)``-step warp
// reduction per grid point. ``GROUP == 1`` is the unsplit arrangement and needs
// no reduction at all, which is what the narrow shapes get.
//
// Numerics are IEEE fp32. The grid sum runs in the natural order of ``g``, so
// the result is bitwise reproducible.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <tuple>

namespace {

#define DPA4_CHECK_LAUNCH(what)                                           \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

constexpr int kWarp = 32;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / kWarp;

/// Lane split of one warp: ``CPW`` channels by ``GROUP`` coefficient slices.
///
/// ``GROUP`` is the smallest power of two that brings the register arrays of
/// the kernel inside ``kRegisterBudget``; ``arrays`` is how many ``P``-sized
/// arrays the kernel holds (three in the forward, five in the backward).
/// ``GROUP == 1`` leaves the warp unsplit and needs no reduction.
constexpr int kRegisterBudget = 150;

/// Coefficients a lane holds, padded to the 16-byte vector width.
constexpr int slice_len(int p, int group) {
  return ((p + group - 1) / group + 3) & ~3;
}

constexpr int group_for(int p, int arrays) {
  int group = 1;
  while (group < 32 && slice_len(p, group) * arrays > kRegisterBudget) {
    group *= 2;
  }
  return group;
}

/// Grid rows staged per pass, capped by a fixed shared-memory budget.
constexpr int kSharedBudget = 24 * 1024;

constexpr int block_rows(int p, int group) {
  const int row =
      group * slice_len(p, group) * 2 * static_cast<int>(sizeof(float));
  int rows = 64;
  while (rows > 1 && rows * row > kSharedBudget) {
    rows >>= 1;
  }
  return rows;
}

/// Stage one block of both projectors, reordered for the lane split.
///
/// Slice ``pg`` of grid row ``g`` lands contiguously at ``[g][pg][.]``, so a
/// lane reads its own coefficients as 16-byte vectors and the lanes of one
/// slice share the address. Pad slots are zeroed and multiply zero operands.
template <int P, int GROUP>
__device__ __forceinline__ void stage_block(const float* __restrict__ to_grid,
                                            const float* __restrict__ from_grid,
                                            float* sm_t,
                                            float* sm_f,
                                            int g0,
                                            int rows) {
  constexpr int SL = slice_len(P, GROUP);
  const int span = GROUP * SL;
  for (int i = threadIdx.x; i < rows * span; i += kThreads) {
    const int g = i / span;
    const int rest = i - g * span;
    const int pg = rest / SL;
    const int k = rest - pg * SL;
    const int slot = pg + k * GROUP;
    const long src = static_cast<long>(g0 + g) * P + slot;
    sm_t[i] = (slot < P) ? to_grid[src] : 0.f;
    sm_f[i] = (slot < P) ? from_grid[src] : 0.f;
  }
}

/// Dot of one staged projector slice with a register operand.
template <int SL>
__device__ __forceinline__ float slice_dot(const float* row,
                                           const float (&v)[SL]) {
  float acc = 0.f;
#pragma unroll
  for (int q = 0; q < SL / 4; ++q) {
    const float4 t = reinterpret_cast<const float4*>(row)[q];
    acc += t.x * v[q * 4] + t.y * v[q * 4 + 1] + t.z * v[q * 4 + 2] +
           t.w * v[q * 4 + 3];
  }
  return acc;
}

/// Scaled accumulation of one staged projector slice into a register operand.
template <int SL>
__device__ __forceinline__ void slice_axpy(const float* row,
                                           float scale,
                                           float (&v)[SL]) {
#pragma unroll
  for (int q = 0; q < SL / 4; ++q) {
    const float4 t = reinterpret_cast<const float4*>(row)[q];
    v[q * 4] += t.x * scale;
    v[q * 4 + 1] += t.y * scale;
    v[q * 4 + 2] += t.z * scale;
    v[q * 4 + 3] += t.w * scale;
  }
}

/// Sum a value across the coefficient slices of one channel.
///
/// The lanes of one channel sit ``CPW`` apart, so the butterfly starts there.
/// Every lane leaves with the total, which is what the pointwise product and
/// the back projection both need.
template <int GROUP, int CPW>
__device__ __forceinline__ float slice_sum(float v) {
#pragma unroll
  for (int step = CPW; step < GROUP * CPW; step <<= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, step);
  }
  return v;
}

/// One warp per ``(node, channel block)`` pair.
///
/// ``P`` sizes the coefficient vectors and ``GROUP`` the lane split, both
/// compile-time so the register arrays and the reduction unroll.
template <int P>
__global__ __launch_bounds__(kThreads) void grid_pair_fwd_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    const float* __restrict__ to_grid,
    const float* __restrict__ from_grid,
    float* __restrict__ out,
    int n_node,
    int c_wide,
    int n_grid) {
  constexpr int GROUP = group_for(P, 3);
  constexpr int CPW = kWarp / GROUP;
  constexpr int SL = slice_len(P, GROUP);
  constexpr int GB = block_rows(P, GROUP);
  extern __shared__ float sm[];
  float* sm_t = sm;
  float* sm_f = sm + static_cast<long>(GB) * GROUP * SL;

  const int warp = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & (kWarp - 1);
  const int pg = lane / CPW;
  const int ci = lane - pg * CPW;
  // A warp covers ``CPW`` channels, so the channel blocks follow the lane
  // split.
  const int chunks = c_wide / CPW;
  const bool live = warp < n_node * chunks;
  const int node = live ? warp / chunks : 0;
  const int channel = live ? (warp - node * chunks) * CPW + ci : 0;
  const long base = static_cast<long>(node) * P * c_wide + channel;

  // === Step 1. Load this lane's coefficient slice ===
  float lv[SL];
  float rv[SL];
  float acc[SL];
#pragma unroll
  for (int k = 0; k < SL; ++k) {
    const int slot = pg + k * GROUP;
    const bool has = live && slot < P;
    const long at = base + static_cast<long>(slot) * c_wide;
    lv[k] = has ? left[at] : 0.f;
    rv[k] = has ? right[at] : 0.f;
    acc[k] = 0.f;
  }

  // === Step 2. Walk the grid one staged block at a time ===
  for (int g0 = 0; g0 < n_grid; g0 += GB) {
    const int rows = min(GB, n_grid - g0);
    __syncthreads();
    stage_block<P, GROUP>(to_grid, from_grid, sm_t, sm_f, g0, rows);
    __syncthreads();
    const float* my_t = sm_t + pg * SL;
    const float* my_f = sm_f + pg * SL;
#pragma unroll 2
    for (int g = 0; g < rows; ++g) {
      const int off = g * GROUP * SL;
      const float lg = slice_sum<GROUP, CPW>(slice_dot<SL>(my_t + off, lv));
      const float rg = slice_sum<GROUP, CPW>(slice_dot<SL>(my_t + off, rv));
      slice_axpy<SL>(my_f + off, lg * rg, acc);
    }
  }

  // === Step 3. Store this lane's slice of the result ===
  if (!live) {
    return;
  }
#pragma unroll
  for (int k = 0; k < SL; ++k) {
    const int slot = pg + k * GROUP;
    if (slot < P) {
      out[base + static_cast<long>(slot) * c_wide] = acc[k];
    }
  }
}

template <int P>
__global__ __launch_bounds__(kThreads) void grid_pair_bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ left,
    const float* __restrict__ right,
    const float* __restrict__ to_grid,
    const float* __restrict__ from_grid,
    float* __restrict__ g_left,
    float* __restrict__ g_right,
    int n_node,
    int c_wide,
    int n_grid) {
  constexpr int GROUP = group_for(P, 5);
  constexpr int CPW = kWarp / GROUP;
  constexpr int SL = slice_len(P, GROUP);
  constexpr int GB = block_rows(P, GROUP);
  extern __shared__ float sm[];
  float* sm_t = sm;
  float* sm_f = sm + static_cast<long>(GB) * GROUP * SL;

  const int warp = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & (kWarp - 1);
  const int pg = lane / CPW;
  const int ci = lane - pg * CPW;
  // A warp covers ``CPW`` channels, so the channel blocks follow the lane
  // split.
  const int chunks = c_wide / CPW;
  const bool live = warp < n_node * chunks;
  const int node = live ? warp / chunks : 0;
  const int channel = live ? (warp - node * chunks) * CPW + ci : 0;
  const long base = static_cast<long>(node) * P * c_wide + channel;

  // === Step 1. Load this lane's coefficient slice ===
  float lv[SL];
  float rv[SL];
  float go[SL];
  float gl[SL];
  float gr[SL];
#pragma unroll
  for (int k = 0; k < SL; ++k) {
    const int slot = pg + k * GROUP;
    const bool has = live && slot < P;
    const long at = base + static_cast<long>(slot) * c_wide;
    lv[k] = has ? left[at] : 0.f;
    rv[k] = has ? right[at] : 0.f;
    go[k] = has ? grad_out[at] : 0.f;
    gl[k] = 0.f;
    gr[k] = 0.f;
  }

  // === Step 2. Walk the grid one staged block at a time ===
  for (int g0 = 0; g0 < n_grid; g0 += GB) {
    const int rows = min(GB, n_grid - g0);
    __syncthreads();
    stage_block<P, GROUP>(to_grid, from_grid, sm_t, sm_f, g0, rows);
    __syncthreads();
    const float* my_t = sm_t + pg * SL;
    const float* my_f = sm_f + pg * SL;
#pragma unroll 2
    for (int g = 0; g < rows; ++g) {
      const int off = g * GROUP * SL;
      const float lg = slice_sum<GROUP, CPW>(slice_dot<SL>(my_t + off, lv));
      const float rg = slice_sum<GROUP, CPW>(slice_dot<SL>(my_t + off, rv));
      const float gv = slice_sum<GROUP, CPW>(slice_dot<SL>(my_f + off, go));
      slice_axpy<SL>(my_t + off, gv * rg, gl);
      slice_axpy<SL>(my_t + off, gv * lg, gr);
    }
  }

  // === Step 3. Store this lane's slice of both cotangents ===
  if (!live) {
    return;
  }
#pragma unroll
  for (int k = 0; k < SL; ++k) {
    const int slot = pg + k * GROUP;
    if (slot < P) {
      const long at = base + static_cast<long>(slot) * c_wide;
      g_left[at] = gl[k];
      g_right[at] = gr[k];
    }
  }
}

/// Channels one warp covers.
template <int P, int ARRAYS>
constexpr int channels_per_warp() {
  return kWarp / group_for(P, ARRAYS);
}

/// Shared memory one launch needs, in bytes.
template <int P, int ARRAYS>
constexpr int shared_bytes() {
  constexpr int GROUP = group_for(P, ARRAYS);
  return block_rows(P, GROUP) * GROUP * slice_len(P, GROUP) * 2 *
         static_cast<int>(sizeof(float));
}

/// Coefficient-slot counts with an instantiation.
///
/// ``P = coeff_dim * n_frames``, so the SO(3) grids of the zoo give
/// ``3 * (l + 1)^2`` for degrees one to six, and 9 is the matching S2 grid.
#define DPA4_GRID_FOR_EACH_P(macro) \
  macro(9) macro(12) macro(27) macro(48) macro(75) macro(108) macro(147)

bool grid_p_supported(int p) {
#define DPA4_CASE(PV) \
  if (p == PV) {      \
    return true;      \
  }
  DPA4_GRID_FOR_EACH_P(DPA4_CASE)
#undef DPA4_CASE
  return false;
}

#define DPA4_GRID_DISPATCH(p, body) \
  do {                              \
    switch (p) {                    \
      DPA4_GRID_FOR_EACH_P(body)    \
      default:                      \
        break;                      \
    }                               \
  } while (0)

/// Blocks covering every ``(node, channel block)`` pair of one launch.
dim3 warp_grid(int n_node, int c_wide, int channels) {
  const long warps = static_cast<long>(n_node) * (c_wide / channels);
  return dim3(static_cast<unsigned>((warps + kWarps - 1) / kWarps));
}

void check_grid_inputs(const torch::Tensor& left,
                       const torch::Tensor& right,
                       const torch::Tensor& to_grid,
                       const torch::Tensor& from_grid) {
  TORCH_CHECK(left.is_cuda() && left.scalar_type() == torch::kFloat,
              "dpa4_grid_pair: operands must be cuda fp32");
  TORCH_CHECK(left.dim() == 3 && right.sizes() == left.sizes(),
              "dpa4_grid_pair: operands must be (N, P, C) of equal shape");
  TORCH_CHECK(left.size(2) % kWarp == 0,
              "dpa4_grid_pair: channel width must be a multiple of 32");
  TORCH_CHECK(grid_p_supported(static_cast<int>(left.size(1))),
              "dpa4_grid_pair: unsupported coefficient-slot count");
  TORCH_CHECK(to_grid.dim() == 2 && from_grid.dim() == 2 &&
                  to_grid.size(1) == left.size(1) &&
                  from_grid.sizes() == to_grid.sizes(),
              "dpa4_grid_pair: projectors must both be (G, P)");
}

}  // namespace

torch::Tensor dpa4_grid_pair(torch::Tensor left,
                             torch::Tensor right,
                             torch::Tensor to_grid,
                             torch::Tensor from_grid) {
  const at::cuda::OptionalCUDAGuard device_guard(left.device());
  check_grid_inputs(left, right, to_grid, from_grid);
  left = left.contiguous();
  right = right.contiguous();
  to_grid = to_grid.contiguous();
  from_grid = from_grid.contiguous();

  auto out = torch::empty_like(left);
  const int n_node = static_cast<int>(left.size(0));
  const int p_dim = static_cast<int>(left.size(1));
  const int c_wide = static_cast<int>(left.size(2));
  const int n_grid = static_cast<int>(to_grid.size(0));
  if (n_node == 0 || c_wide == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
#define DPA4_LAUNCH_GRID_FWD(PV)                                    \
  case PV:                                                          \
    grid_pair_fwd_kernel<PV>                                        \
        <<<warp_grid(n_node, c_wide, channels_per_warp<PV, 3>()),   \
           dim3(kThreads), shared_bytes<PV, 3>(), stream>>>(        \
            left.data_ptr<float>(), right.data_ptr<float>(),        \
            to_grid.data_ptr<float>(), from_grid.data_ptr<float>(), \
            out.data_ptr<float>(), n_node, c_wide, n_grid);         \
    break;
  DPA4_GRID_DISPATCH(p_dim, DPA4_LAUNCH_GRID_FWD);
#undef DPA4_LAUNCH_GRID_FWD
  DPA4_CHECK_LAUNCH("dpa4_grid_pair");
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> dpa4_grid_pair_backward(
    torch::Tensor grad_out,
    torch::Tensor left,
    torch::Tensor right,
    torch::Tensor to_grid,
    torch::Tensor from_grid) {
  const at::cuda::OptionalCUDAGuard device_guard(left.device());
  check_grid_inputs(left, right, to_grid, from_grid);
  grad_out = grad_out.contiguous();
  left = left.contiguous();
  right = right.contiguous();
  to_grid = to_grid.contiguous();
  from_grid = from_grid.contiguous();

  auto g_left = torch::empty_like(left);
  auto g_right = torch::empty_like(right);
  const int n_node = static_cast<int>(left.size(0));
  const int p_dim = static_cast<int>(left.size(1));
  const int c_wide = static_cast<int>(left.size(2));
  const int n_grid = static_cast<int>(to_grid.size(0));
  if (n_node == 0 || c_wide == 0) {
    return {g_left, g_right};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
#define DPA4_LAUNCH_GRID_BWD(PV)                                   \
  case PV:                                                         \
    grid_pair_bwd_kernel<PV>                                       \
        <<<warp_grid(n_node, c_wide, channels_per_warp<PV, 5>()),  \
           dim3(kThreads), shared_bytes<PV, 5>(), stream>>>(       \
            grad_out.data_ptr<float>(), left.data_ptr<float>(),    \
            right.data_ptr<float>(), to_grid.data_ptr<float>(),    \
            from_grid.data_ptr<float>(), g_left.data_ptr<float>(), \
            g_right.data_ptr<float>(), n_node, c_wide, n_grid);    \
    break;
  DPA4_GRID_DISPATCH(p_dim, DPA4_LAUNCH_GRID_BWD);
#undef DPA4_LAUNCH_GRID_BWD
  DPA4_CHECK_LAUNCH("dpa4_grid_pair_backward");
  return {g_left, g_right};
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "dpa4_grid_pair(Tensor left, Tensor right, Tensor to_grid, "
      "Tensor from_grid) -> Tensor");
  m.impl("dpa4_grid_pair", torch::kCUDA, &dpa4_grid_pair);
  m.def(
      "dpa4_grid_pair_backward(Tensor grad_out, Tensor left, Tensor right, "
      "Tensor to_grid, Tensor from_grid) -> (Tensor g_left, Tensor g_right)");
  m.impl("dpa4_grid_pair_backward", torch::kCUDA, &dpa4_grid_pair_backward);
}
