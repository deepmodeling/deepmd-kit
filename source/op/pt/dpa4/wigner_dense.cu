// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused dense Wigner-D build for SeZM / DPA4 inference.
//
// Every element of the packed block-diagonal Wigner matrix is a homogeneous
// polynomial of degree ``2 l`` in the unit quaternion (the ``l = 0`` block is
// the constant one, the ``l = 1`` block is the quadratic rotation matrix).
// The Python side fits one sparse monomial table per degree against the
// reference calculator and concatenates them element-major:
//
//   D[e, row[j], col[j]] = sum_{k in elem_ptr[j]..elem_ptr[j+1]}
//                            coeff[k] * prod_i q[e, i] ^ exp(mono[k], i)
//
// The module-composition path evaluates the same polynomials as a dense
// monomial basis, a GEMM per degree pair, an ``index_put_`` into a zero
// block-diagonal frame, and a transposed copy -- five full-size passes over
// the ``(E, D, D)`` pair. Here one kernel reads the quaternion, evaluates the
// sparse table in registers against a per-edge power table in shared memory,
// assembles the block in shared memory, and streams ``D_full`` and
// ``Dt_full`` out with coalesced writes. Device traffic drops to the
// quaternion read and the two output writes, which is the lower bound.
//
// The table is shared by every block and is a few hundred kilobytes at most,
// so its reads stay resident in L2. Entries are grouped per element; a warp
// walks edge-major over the ``(edge, element)`` task space, so the 32 lanes
// of a warp read the same entry (one broadcast) while their power-table reads
// spread over edges (no bank conflicts in the edge-major power layout).
//
// The gradient with respect to the quaternion follows by exact exponent
// manipulation inside the same table walk. The polynomial is differentiated
// as written; the radial component of that gradient (the homogeneity
// direction) is projected out upstream by the quaternion normalization, so
// the extension ambiguity off the unit sphere is immaterial.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <tuple>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxEdgesPerBlock = 32;
// Degrees above ten leave the dedicated monomial path of the reference
// calculator as well; the Python gate falls back there.
constexpr int kMaxLmax = 10;

#define DPA4_CHECK_LAUNCH(what)                                           \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

/// Product of the four quaternion powers named by one packed exponent tuple.
///
/// ``pw`` holds the per-edge power table in edge-major layout,
/// ``pw[comp * n_pow * te_cap + k * te_cap + te] = q[e, comp]^k``.
__device__ __forceinline__ float monomial_value(
    const float* __restrict__ pw, int mono, int n_pow, int te_cap, int te) {
  const int a = mono & 0xff;
  const int b = (mono >> 8) & 0xff;
  const int c = (mono >> 16) & 0xff;
  const int d = (mono >> 24) & 0xff;
  return pw[(0 * n_pow + a) * te_cap + te] * pw[(1 * n_pow + b) * te_cap + te] *
         pw[(2 * n_pow + c) * te_cap + te] * pw[(3 * n_pow + d) * te_cap + te];
}

/// Fill the edge-major power table for the edges owned by this block.
__device__ __forceinline__ void build_power_table(
    const float* __restrict__ quat,
    float* __restrict__ pw,
    long edge_base,
    long n_edge,
    int n_pow,
    int te_cap) {
  for (int t = threadIdx.x; t < te_cap * 4; t += blockDim.x) {
    const int te = t & (te_cap - 1);
    const int comp = t / te_cap;
    const long edge = edge_base + te;
    const float q = edge < n_edge ? quat[edge * 4 + comp] : 0.f;
    float p = 1.f;
    for (int k = 0; k < n_pow; ++k) {
      pw[(comp * n_pow + k) * te_cap + te] = p;
      p *= q;
    }
  }
}

/// One block assembles ``TE`` edges' Wigner pairs in shared memory.
///
/// Shared layout: the power table (``4 * n_pow * TE`` floats, edge-major)
/// followed by the block-diagonal staging frame (``TE * D^2`` floats). The
/// frame is zero-filled once, the table walk writes only the block-diagonal
/// elements, and both outputs stream out linearly.
__global__ __launch_bounds__(kThreads) void wigner_dense_fwd_kernel(
    const float* __restrict__ quat,         // (E, 4)
    const int* __restrict__ elem_ptr,       // (NB + 1,)
    const int* __restrict__ elem_pos,       // (NB,) packed r * D + c
    const float* __restrict__ entry_coeff,  // (K,)
    const int* __restrict__ entry_mono,     // (K,) packed exponents
    float* __restrict__ d_out,              // (E, D, D)
    float* __restrict__ dt_out,             // (E, D, D)
    long n_edge,
    int n_elem,
    int dim,
    int n_pow,
    int te_cap) {
  extern __shared__ float smem[];
  float* pw = smem;                          // (4, n_pow, TE)
  float* frame = smem + 4 * n_pow * te_cap;  // (TE, D, D)

  const long edge_base = static_cast<long>(blockIdx.x) * te_cap;
  const int dd = dim * dim;

  build_power_table(quat, pw, edge_base, n_edge, n_pow, te_cap);
  for (int i = threadIdx.x; i < te_cap * dd; i += blockDim.x) {
    frame[i] = 0.f;
  }
  __syncthreads();

  // === Step 1. Evaluate the sparse table over the (edge, element) tasks ===
  const int n_task = te_cap * n_elem;
  for (int task = threadIdx.x; task < n_task; task += blockDim.x) {
    const int te = task % te_cap;
    const int j = task / te_cap;
    const int begin = __ldg(elem_ptr + j);
    const int end = __ldg(elem_ptr + j + 1);
    float acc = 0.f;
    for (int k = begin; k < end; ++k) {
      acc = fmaf(__ldg(entry_coeff + k),
                 monomial_value(pw, __ldg(entry_mono + k), n_pow, te_cap, te),
                 acc);
    }
    frame[te * dd + __ldg(elem_pos + j)] = acc;
  }
  __syncthreads();

  // === Step 2. Stream the pair out with coalesced writes ===
  for (int i = threadIdx.x; i < te_cap * dd; i += blockDim.x) {
    const int te = i / dd;
    const long edge = edge_base + te;
    if (edge >= n_edge) {
      break;
    }
    const int rem = i - te * dd;
    const int r = rem / dim;
    const int c = rem - r * dim;
    d_out[edge * dd + rem] = frame[te * dd + rem];
    dt_out[edge * dd + rem] = frame[te * dd + c * dim + r];
  }
}

/// Quaternion cotangent: the same table walk against the summed output
/// cotangent ``g[r, c] = g_D[r, c] + g_Dt[c, r]``, staged in shared memory by
/// two coalesced passes. Each task differentiates its element's entries by
/// exponent manipulation and accumulates into per-edge shared slots.
__global__ __launch_bounds__(kThreads) void wigner_dense_bwd_kernel(
    const float* __restrict__ g_d,          // (E, D, D)
    const float* __restrict__ g_dt,         // (E, D, D)
    const float* __restrict__ quat,         // (E, 4)
    const int* __restrict__ elem_ptr,       // (NB + 1,)
    const int* __restrict__ elem_pos,       // (NB,)
    const float* __restrict__ entry_coeff,  // (K,)
    const int* __restrict__ entry_mono,     // (K,)
    float* __restrict__ g_quat,             // (E, 4)
    long n_edge,
    int n_elem,
    int dim,
    int n_pow,
    int te_cap) {
  extern __shared__ float smem[];
  float* pw = smem;                         // (4, n_pow, TE)
  float* gsum = smem + 4 * n_pow * te_cap;  // (TE, D, D)
  float* gq = gsum + te_cap * dim * dim;    // (TE, 4)

  const long edge_base = static_cast<long>(blockIdx.x) * te_cap;
  const int dd = dim * dim;

  build_power_table(quat, pw, edge_base, n_edge, n_pow, te_cap);
  for (int i = threadIdx.x; i < te_cap * 4; i += blockDim.x) {
    gq[i] = 0.f;
  }
  // === Step 1. Stage the summed cotangent block ===
  for (int i = threadIdx.x; i < te_cap * dd; i += blockDim.x) {
    const int te = i / dd;
    const long edge = edge_base + te;
    gsum[i] = edge < n_edge ? g_d[edge * dd + (i - te * dd)] : 0.f;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < te_cap * dd; i += blockDim.x) {
    const int te = i / dd;
    const long edge = edge_base + te;
    if (edge < n_edge) {
      const int rem = i - te * dd;
      const int r = rem / dim;
      const int c = rem - r * dim;
      // gsum[te][c][r] += g_dt[edge][r][c]; the (i -> transposed slot) map is
      // a bijection, so no two threads touch the same slot.
      gsum[te * dd + c * dim + r] += g_dt[edge * dd + rem];
    }
  }
  __syncthreads();

  // === Step 2. Contract the differentiated table against the cotangent ===
  const int n_task = te_cap * n_elem;
  for (int task = threadIdx.x; task < n_task; task += blockDim.x) {
    const int te = task % te_cap;
    const int j = task / te_cap;
    const float g = gsum[te * dd + __ldg(elem_pos + j)];
    const int begin = __ldg(elem_ptr + j);
    const int end = __ldg(elem_ptr + j + 1);
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int k = begin; k < end; ++k) {
      const float coeff = __ldg(entry_coeff + k);
      const int mono = __ldg(entry_mono + k);
      const int a = mono & 0xff;
      const int b = (mono >> 8) & 0xff;
      const int c = (mono >> 16) & 0xff;
      const int d = (mono >> 24) & 0xff;
      const float pa = pw[(0 * n_pow + a) * te_cap + te];
      const float pb = pw[(1 * n_pow + b) * te_cap + te];
      const float pc = pw[(2 * n_pow + c) * te_cap + te];
      const float pd = pw[(3 * n_pow + d) * te_cap + te];
      if (a > 0) {
        acc0 = fmaf(coeff * a,
                    pw[(0 * n_pow + a - 1) * te_cap + te] * pb * pc * pd, acc0);
      }
      if (b > 0) {
        acc1 = fmaf(coeff * b,
                    pa * pw[(1 * n_pow + b - 1) * te_cap + te] * pc * pd, acc1);
      }
      if (c > 0) {
        acc2 = fmaf(coeff * c,
                    pa * pb * pw[(2 * n_pow + c - 1) * te_cap + te] * pd, acc2);
      }
      if (d > 0) {
        acc3 = fmaf(coeff * d,
                    pa * pb * pc * pw[(3 * n_pow + d - 1) * te_cap + te], acc3);
      }
    }
    atomicAdd(gq + te * 4 + 0, g * acc0);
    atomicAdd(gq + te * 4 + 1, g * acc1);
    atomicAdd(gq + te * 4 + 2, g * acc2);
    atomicAdd(gq + te * 4 + 3, g * acc3);
  }
  __syncthreads();

  // === Step 3. Write the quaternion cotangent ===
  for (int i = threadIdx.x; i < te_cap * 4; i += blockDim.x) {
    const long edge = edge_base + i / 4;
    if (edge < n_edge) {
      g_quat[edge * 4 + (i & 3)] = gq[i];
    }
  }
}

void check_inputs(const torch::Tensor& quat,
                  const torch::Tensor& elem_ptr,
                  const torch::Tensor& elem_pos,
                  const torch::Tensor& entry_coeff,
                  const torch::Tensor& entry_mono,
                  int64_t lmax) {
  TORCH_CHECK(quat.is_cuda() && quat.scalar_type() == torch::kFloat,
              "dpa4_wigner_dense: the quaternion must be cuda fp32");
  TORCH_CHECK(quat.dim() == 2 && quat.size(1) == 4,
              "dpa4_wigner_dense: the quaternion must have shape (E, 4)");
  TORCH_CHECK(1 <= lmax && lmax <= kMaxLmax,
              "dpa4_wigner_dense: degree out of the supported range");
  const int dim = static_cast<int>((lmax + 1) * (lmax + 1));
  int n_elem = 0;
  for (int l = 0; l <= lmax; ++l) {
    n_elem += (2 * l + 1) * (2 * l + 1);
  }
  TORCH_CHECK(elem_ptr.scalar_type() == torch::kInt &&
                  elem_pos.scalar_type() == torch::kInt &&
                  entry_mono.scalar_type() == torch::kInt &&
                  entry_coeff.scalar_type() == torch::kFloat,
              "dpa4_wigner_dense: table dtypes must be (int32, int32, fp32, "
              "int32)");
  TORCH_CHECK(elem_ptr.numel() == n_elem + 1 && elem_pos.numel() == n_elem,
              "dpa4_wigner_dense: the element table must cover every "
              "block-diagonal element of degree ",
              lmax);
  TORCH_CHECK(dim <= 121, "dpa4_wigner_dense: block dimension overflow");
}

/// Edges staged per block: as many as the shared budget holds, capped at 32.
///
/// ``extra`` counts per-edge floats beyond the power table and the frame.
int edges_per_block(int dim, int n_pow, int extra) {
  const int per_edge = (dim * dim + 4 * n_pow + extra) * 4;
  constexpr int kBudget = 48 * 1024;
  int te = kBudget / per_edge;
  te = te < 1 ? 1 : (te > kMaxEdgesPerBlock ? kMaxEdgesPerBlock : te);
  // A power-of-two count keeps the modulo in the task walk cheap.
  while (te & (te - 1)) {
    te &= te - 1;
  }
  return te;
}

/// Raise the dynamic shared-memory ceiling when one edge exceeds the 48 KiB
/// default (the largest supported degree at a single staged edge).
template <typename Kernel>
void allow_large_smem(Kernel kernel, int smem) {
  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> dpa4_wigner_dense(
    torch::Tensor quat,
    torch::Tensor elem_ptr,
    torch::Tensor elem_pos,
    torch::Tensor entry_coeff,
    torch::Tensor entry_mono,
    int64_t lmax) {
  const at::cuda::OptionalCUDAGuard device_guard(quat.device());
  check_inputs(quat, elem_ptr, elem_pos, entry_coeff, entry_mono, lmax);
  quat = quat.contiguous();

  const long n_edge = quat.size(0);
  const int dim = static_cast<int>((lmax + 1) * (lmax + 1));
  const int n_elem = static_cast<int>(elem_pos.numel());
  const int n_pow = 2 * static_cast<int>(lmax) + 1;
  auto d_out = torch::empty({n_edge, dim, dim}, quat.options());
  auto dt_out = torch::empty({n_edge, dim, dim}, quat.options());
  if (n_edge == 0) {
    return {d_out, dt_out};
  }

  const int te = edges_per_block(dim, n_pow, 0);
  const int smem = (4 * n_pow + dim * dim) * te * 4;
  const unsigned blocks = static_cast<unsigned>((n_edge + te - 1) / te);
  const auto stream = at::cuda::getCurrentCUDAStream();
  allow_large_smem(wigner_dense_fwd_kernel, smem);
  wigner_dense_fwd_kernel<<<dim3(blocks), dim3(kThreads), smem, stream>>>(
      quat.data_ptr<float>(), elem_ptr.data_ptr<int>(),
      elem_pos.data_ptr<int>(), entry_coeff.data_ptr<float>(),
      entry_mono.data_ptr<int>(), d_out.data_ptr<float>(),
      dt_out.data_ptr<float>(), n_edge, n_elem, dim, n_pow, te);
  DPA4_CHECK_LAUNCH("dpa4_wigner_dense");
  return {d_out, dt_out};
}

torch::Tensor dpa4_wigner_dense_backward(torch::Tensor g_d,
                                         torch::Tensor g_dt,
                                         torch::Tensor quat,
                                         torch::Tensor elem_ptr,
                                         torch::Tensor elem_pos,
                                         torch::Tensor entry_coeff,
                                         torch::Tensor entry_mono,
                                         int64_t lmax) {
  const at::cuda::OptionalCUDAGuard device_guard(quat.device());
  check_inputs(quat, elem_ptr, elem_pos, entry_coeff, entry_mono, lmax);
  quat = quat.contiguous();
  g_d = g_d.contiguous();
  g_dt = g_dt.contiguous();

  const long n_edge = quat.size(0);
  const int dim = static_cast<int>((lmax + 1) * (lmax + 1));
  const int n_elem = static_cast<int>(elem_pos.numel());
  const int n_pow = 2 * static_cast<int>(lmax) + 1;
  auto g_quat = torch::empty_like(quat);
  if (n_edge == 0) {
    return g_quat;
  }

  const int te = edges_per_block(dim, n_pow, 4);
  const int smem = (4 * n_pow + dim * dim + 4) * te * 4;
  const unsigned blocks = static_cast<unsigned>((n_edge + te - 1) / te);
  const auto stream = at::cuda::getCurrentCUDAStream();
  allow_large_smem(wigner_dense_bwd_kernel, smem);
  wigner_dense_bwd_kernel<<<dim3(blocks), dim3(kThreads), smem, stream>>>(
      g_d.data_ptr<float>(), g_dt.data_ptr<float>(), quat.data_ptr<float>(),
      elem_ptr.data_ptr<int>(), elem_pos.data_ptr<int>(),
      entry_coeff.data_ptr<float>(), entry_mono.data_ptr<int>(),
      g_quat.data_ptr<float>(), n_edge, n_elem, dim, n_pow, te);
  DPA4_CHECK_LAUNCH("dpa4_wigner_dense_backward");
  return g_quat;
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "dpa4_wigner_dense(Tensor quat, Tensor elem_ptr, Tensor elem_pos, "
      "Tensor entry_coeff, Tensor entry_mono, int lmax) -> (Tensor, Tensor)");
  m.impl("dpa4_wigner_dense", torch::kCUDA, &dpa4_wigner_dense);
  m.def(
      "dpa4_wigner_dense_backward(Tensor g_d, Tensor g_dt, Tensor quat, "
      "Tensor elem_ptr, Tensor elem_pos, Tensor entry_coeff, "
      "Tensor entry_mono, int lmax) -> Tensor");
  m.impl("dpa4_wigner_dense_backward", torch::kCUDA,
         &dpa4_wigner_dense_backward);
}
