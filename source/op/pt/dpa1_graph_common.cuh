// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Shared device / host infrastructure of the DPA1 graph-lower CUDA operators.
//
// The MLP descriptor (dpa1_graph_descriptor.cu) and the tabulated compressed
// descriptor (dpa1_graph_compress.cu) evaluate the same environment matrix,
// CSR run structure, moment layout and G^T G contraction; they differ only in
// how the per-edge embedding ``g`` is produced (three-layer MLP versus quintic
// spline table). This header carries everything outside that difference:
//
// * the per-edge staging (environment-matrix row, type-pair index, smooth
//   switch, center node) and the parallel CSR run scan over a tile;
// * the sort-free CSR edge ordering (histogram + exclusive scan + scatter);
// * the gram contraction kernels (forward and backward);
// * float4 fragment load / store helpers and the launch utilities.
//
// Everything lives in an anonymous namespace: each translation unit
// instantiates its own internal copy, which keeps the operators independently
// compilable (and their heavy template sets compiling in parallel).

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cub/device/device_scan.cuh>
#include <limits>

namespace {

constexpr int kThreads = 256;                // 8 warps per CTA
constexpr int kTileEdges = 128;              // BE: edges per forward tile pass
constexpr int kTileStride = kTileEdges + 4;  // padded [channel][edge] stride

#define DEV_INLINE __device__ __forceinline__

#define DPA1_CHECK_LAUNCH(what)                                           \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

// Quintic smooth switch on [rmin, rmax] and its derivative.
DEV_INLINE float switch_val(float r, float rmin, float rmax) {
  float u = (fminf(fmaxf(r, rmin), rmax) - rmin) / (rmax - rmin);
  float u2 = u * u;
  return u2 * u * (-6.f * u2 + 15.f * u - 10.f) + 1.f;
}

DEV_INLINE float switch_deriv(float r, float rmin, float rmax) {
  if (r <= rmin || r >= rmax) {
    return 0.f;
  }
  float u = (r - rmin) / (rmax - rmin), u2 = u * u;
  return (-30.f * u2 * u2 + 60.f * u2 * u - 30.f * u2) / (rmax - rmin);
}

// Eight-row float4 fragment load / store on a 16-byte-aligned span.
DEV_INLINE void load8(const float* p, float (&a)[8]) {
  const float4 v0 = *reinterpret_cast<const float4*>(p);
  const float4 v1 = *reinterpret_cast<const float4*>(p + 4);
  a[0] = v0.x;
  a[1] = v0.y;
  a[2] = v0.z;
  a[3] = v0.w;
  a[4] = v1.x;
  a[5] = v1.y;
  a[6] = v1.z;
  a[7] = v1.w;
}

DEV_INLINE void store8(float* p, const float (&a)[8]) {
  float4 v0, v1;
  v0.x = a[0];
  v0.y = a[1];
  v0.z = a[2];
  v0.w = a[3];
  v1.x = a[4];
  v1.y = a[5];
  v1.z = a[6];
  v1.w = a[7];
  *reinterpret_cast<float4*>(p) = v0;
  *reinterpret_cast<float4*>(p + 4) = v1;
}

// Streaming (evict-first) variants for the once-per-pass spill tensors; the
// cache hint keeps them from evicting the L1-resident weight lines.
DEV_INLINE void load8_streaming(const float* p, float (&a)[8]) {
  const float4 v0 = __ldcs(reinterpret_cast<const float4*>(p));
  const float4 v1 = __ldcs(reinterpret_cast<const float4*>(p) + 1);
  a[0] = v0.x;
  a[1] = v0.y;
  a[2] = v0.z;
  a[3] = v0.w;
  a[4] = v1.x;
  a[5] = v1.y;
  a[6] = v1.z;
  a[7] = v1.w;
}

DEV_INLINE void store8_streaming(float* p, const float (&a)[8]) {
  float4 v0, v1;
  v0.x = a[0];
  v0.y = a[1];
  v0.z = a[2];
  v0.w = a[3];
  v1.x = a[4];
  v1.y = a[5];
  v1.z = a[6];
  v1.w = a[7];
  __stcs(reinterpret_cast<float4*>(p), v0);
  __stcs(reinterpret_cast<float4*>(p) + 1, v1);
}

// Warp-uniform float4 weight broadcast (all lanes read the same address, so
// the load costs one L1 transaction per warp).
DEV_INLINE float4 weight4(const float* p) {
  return __ldg(reinterpret_cast<const float4*>(p));
}

// Generic N-row float4 fragment load / store (N a multiple of 4). These back
// the edge-fragment loads of the backward kernels, whose per-thread edge count
// (EPT) is a template parameter; the forwards keep the fixed load8 / store8.
template <int N>
DEV_INLINE void loadN(const float* p, float (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    const float4 v = *reinterpret_cast<const float4*>(p + i);
    a[i] = v.x;
    a[i + 1] = v.y;
    a[i + 2] = v.z;
    a[i + 3] = v.w;
  }
}

template <int N>
DEV_INLINE void storeN(float* p, const float (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    *reinterpret_cast<float4*>(p + i) =
        make_float4(a[i], a[i + 1], a[i + 2], a[i + 3]);
  }
}

template <int N>
DEV_INLINE void loadN_streaming(const float* p, float (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    const float4 v = __ldcs(reinterpret_cast<const float4*>(p + i));
    a[i] = v.x;
    a[i + 1] = v.y;
    a[i + 2] = v.z;
    a[i + 3] = v.w;
  }
}

template <int N>
DEV_INLINE void storeN_streaming(float* p, const float (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    __stcs(reinterpret_cast<float4*>(p + i),
           make_float4(a[i], a[i + 1], a[i + 2], a[i + 3]));
  }
}

__host__ __device__ constexpr long ceil_div(long a, long b) {
  return (a + b - 1) / b;
}

// ======================================================================
// Sort-free CSR ordering: histogram + exclusive scan + scatter.
//
// Graph builders emit real edges already sorted by the center (dst) node
// with a masked padding tail, which keeps the identity permutation (masked
// rows contribute exactly zero to the moment, so their position within the
// order is irrelevant). A genuinely unsorted stream falls back to the
// scatter, which makes each node's edges contiguous.
// ======================================================================
__global__ void edge_order_count_kernel(long n_edge,
                                        const long* __restrict__ dst,
                                        const bool* __restrict__ mask,
                                        int* __restrict__ counts,
                                        int* __restrict__ unsorted) {
  const long e = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (e >= n_edge) {
    return;
  }
  const long d = dst[e];
  atomicAdd(&counts[d], 1);
  if (e > 0 && mask[e] && mask[e - 1] && dst[e - 1] > d) {
    atomicOr(unsorted, 1);
  }
}

__global__ void edge_order_scatter_kernel(long n_edge,
                                          const long* __restrict__ dst,
                                          const int* __restrict__ offsets,
                                          const int* __restrict__ unsorted,
                                          int* __restrict__ cursor,
                                          int* __restrict__ order) {
  const long e = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (e >= n_edge) {
    return;
  }
  if (*unsorted == 0) {
    order[e] = (int)e;
    return;
  }
  const long d = dst[e];
  const int pos = atomicAdd(&cursor[d], 1);
  order[offsets[d] + pos] = (int)e;
}

// ======================================================================
// Per-edge staging: environment-matrix row, type-pair index, switch value
// and center node.
// ======================================================================
struct EdgeStage {
  float r0, r1, r2, r3;  // normalized environment-matrix row
  float sw;              // raw smooth-switch value (type-pair gate factor)
  int pair_idx;
  int dst;
  bool valid;
};

DEV_INLINE EdgeStage stage_edge(long e,
                                long n_edge,
                                int ntypes,
                                int one_side,
                                float rcut,
                                float rcut_smth,
                                float protection,
                                const float* __restrict__ edge_vec,
                                const long* __restrict__ edge_index,
                                const bool* __restrict__ edge_mask,
                                const long* __restrict__ atype,
                                const float* __restrict__ davg,
                                const float* __restrict__ inv_dstd) {
  EdgeStage s;
  const long src = edge_index[e];
  const long dst = edge_index[n_edge + e];
  s.dst = (int)dst;
  const int ct = (int)atype[dst];
  const int nt = (int)atype[src];
  s.valid = edge_mask[e];
  const float x = edge_vec[e * 3 + 0];
  const float y = edge_vec[e * 3 + 1];
  const float z = edge_vec[e * 3 + 2];
  float len = sqrtf(x * x + y * y + z * z);
  // Reference guard: padding edges enter the smooth switch at |r| + 1 so the
  // 1/q factors stay finite; their moment weight is zeroed separately.
  len += s.valid ? 0.f : 1.f;
  const float q = len + protection;
  const float sw = switch_val(len, rcut_smth, rcut);
  const float rq = 1.f / q;
  const float t0 = sw * rq, iq2 = sw * rq * rq;
  const float* av = davg + (long)ct * 4;
  const float* isd = inv_dstd + (long)ct * 4;
  s.r0 = (t0 - av[0]) * isd[0];
  s.r1 = (x * iq2 - av[1]) * isd[1];
  s.r2 = (y * iq2 - av[2]) * isd[2];
  s.r3 = (z * iq2 - av[3]) * isd[3];
  s.sw = sw;
  s.pair_idx = one_side ? nt : ct * ntypes + nt;
  return s;
}

// Parallel CSR run scan over one tile: row r is a run head iff
// dst[r] != dst[r - 1]; per-warp head ballots give every row its run index
// as a popcount prefix. All threads must call; barriers inside.
template <int NW>
DEV_INLINE void scan_runs(int tid,
                          int rows,
                          const int* dst,
                          unsigned* head_masks,
                          int* run_node,
                          int* run_begin,
                          int* run_of,
                          int* n_runs) {
  constexpr int TILE = NW * 32;  // edges per tile
  const int r = tid;
  bool head = false;
  if (r < TILE) {
    head = r < rows && (r == 0 || dst[r] != dst[r - 1]);
  }
  if (tid < TILE) {
    const unsigned m = __ballot_sync(0xffffffffu, head);
    if ((tid & 31) == 0) {
      head_masks[tid >> 5] = m;
    }
  }
  __syncthreads();
  if (r < TILE) {
    const int w = r >> 5, lane = r & 31;
    int idx = 0;  // heads strictly before row r
#pragma unroll
    for (int i = 0; i < NW; ++i) {
      if (i < w) {
        idx += __popc(head_masks[i]);
      }
    }
    idx += __popc(head_masks[w] & ((1u << lane) - 1u));
    if (r < rows) {
      // A head row's exclusive count equals its run index; a non-head row's
      // run head lies before it, so the exclusive count overshoots by one.
      run_of[r] = head ? idx : idx - 1;
      if (head) {
        run_node[idx] = dst[r];
        run_begin[idx] = r;
      }
    } else {
      run_of[r] = 0;
    }
    if (r == 0) {
      int total = 0;
#pragma unroll
      for (int i = 0; i < NW; ++i) {
        total += __popc(head_masks[i]);
      }
      *n_runs = total;
      run_begin[total] = rows;
    }
  }
  __syncthreads();
}

// Per-edge tile tables shared by the forward and backward kernels; the tile
// width (edges per tile) is a template parameter so a forward can keep the
// 128-edge tile while its backward uses a narrower tile (fewer edges per
// thread, so the per-thread register footprint drops below the spill wall).
template <int TILE>
struct EdgeTablesT {
  float rr[TILE][4];   // env-mat row premultiplied by mask / nnei
  float radial[TILE];  // rr0 (unmasked MLP / table input)
  float sw[TILE];      // raw switch value (strip gate factor)
  int pair_idx[TILE];
  int dst[TILE];
  int run_node[TILE];
  int run_begin[TILE + 1];
  int run_of[TILE];
  unsigned head_masks[TILE / 32];
  int n_runs;
};

template <int TILE>
DEV_INLINE void stage_tile(int tid,
                           long tile_base,
                           int rows,
                           long n_edge,
                           int ntypes,
                           int one_side,
                           float rcut,
                           float rcut_smth,
                           float protection,
                           float inv_nnei,
                           const float* __restrict__ edge_vec,
                           const long* __restrict__ edge_index,
                           const bool* __restrict__ edge_mask,
                           const long* __restrict__ atype,
                           const float* __restrict__ davg,
                           const float* __restrict__ inv_dstd,
                           const int* __restrict__ order,
                           EdgeTablesT<TILE>& T) {
  if (tid < TILE) {
    if (tid < rows) {
      const int e = order[tile_base + tid];
      const auto s =
          stage_edge(e, n_edge, ntypes, one_side, rcut, rcut_smth, protection,
                     edge_vec, edge_index, edge_mask, atype, davg, inv_dstd);
      const float mm = (s.valid ? 1.f : 0.f) * inv_nnei;
      T.radial[tid] = s.r0;
      T.rr[tid][0] = s.r0 * mm;
      T.rr[tid][1] = s.r1 * mm;
      T.rr[tid][2] = s.r2 * mm;
      T.rr[tid][3] = s.r3 * mm;
      T.sw[tid] = s.sw;
      T.pair_idx[tid] = s.pair_idx;
      T.dst[tid] = s.dst;
    } else {
      // Tail rows of a partial tile: finite MLP input, zero moment weight,
      // sentinel dst excluded from every run.
      T.radial[tid] = 0.f;
      T.rr[tid][0] = 0.f;
      T.rr[tid][1] = 0.f;
      T.rr[tid][2] = 0.f;
      T.rr[tid][3] = 0.f;
      T.sw[tid] = 0.f;
      T.pair_idx[tid] = 0;
      T.dst[tid] = -1;
    }
  }
  __syncthreads();
  scan_runs<TILE / 32>(tid, rows, T.dst, T.head_masks, T.run_node, T.run_begin,
                       T.run_of, &T.n_runs);
}

// Type-pair gate factor of one edge and channel (strip mode):
//   gate_eff = gate_table[pair, c] (* sw when smooth_type_embedding).
// The gathered scalar rides L1/L2 (edges of one run share dst but not the
// neighbor type, so the rows repeat within a few hundred distinct pairs).
DEV_INLINE float gate_factor(const float* __restrict__ gate_table,
                             int ng,
                             int pair_idx,
                             int c,
                             float sw,
                             int smooth) {
  const float gate = __ldg(gate_table + (long)pair_idx * ng + c);
  return smooth ? gate * sw : gate;
}

// ======================================================================
// Gram contraction: one CTA per node.
//   grrg[n, i * axis + j] = sum_k gr[n, k, i] * gr[n, k, j]
//   rot_mat[n, i, :]      = gr[n, 1:4, i]
// plus the appended center type embedding when concat_tebd is set.
// ======================================================================
__global__ void gram_kernel(int n_node,
                            int ng,
                            int axis,
                            int tebd_dim,
                            int concat_tebd,
                            const float* __restrict__ gr,
                            const float* __restrict__ type_embedding,
                            const long* __restrict__ atype,
                            float* __restrict__ grrg,
                            float* __restrict__ rot_mat) {
  const int n = blockIdx.x;
  extern __shared__ float s_gr[];  // (4, ng)
  for (int t = threadIdx.x; t < 4 * ng; t += blockDim.x) {
    s_gr[t] = gr[(long)n * 4 * ng + t];
  }
  __syncthreads();
  const int out_dim = ng * axis + (concat_tebd ? tebd_dim : 0);
  float* out = grrg + (long)n * out_dim;
  for (int i = threadIdx.x; i < ng; i += blockDim.x) {
    const float g0 = s_gr[0 * ng + i], g1 = s_gr[1 * ng + i];
    const float g2 = s_gr[2 * ng + i], g3 = s_gr[3 * ng + i];
    for (int j = 0; j < axis; ++j) {
      out[i * axis + j] = g0 * s_gr[0 * ng + j] + g1 * s_gr[1 * ng + j] +
                          g2 * s_gr[2 * ng + j] + g3 * s_gr[3 * ng + j];
    }
    if (rot_mat != nullptr) {
      rot_mat[((long)n * ng + i) * 3 + 0] = g1;
      rot_mat[((long)n * ng + i) * 3 + 1] = g2;
      rot_mat[((long)n * ng + i) * 3 + 2] = g3;
    }
  }
  if (concat_tebd) {
    const float* te = type_embedding + atype[n] * tebd_dim;
    for (int t = threadIdx.x; t < tebd_dim; t += blockDim.x) {
      out[ng * axis + t] = te[t];
    }
  }
}

// Gram backward: with D = d(grrg) reshaped to (ng, axis) rows and R = d(rot),
//   dgr[n, k, c] = sum_j D[c, j] gr[k, j]
//                + (c < axis) sum_i D[i, c] gr[k, i]
//                + (k >= 1) R[c, k - 1].
// The concat tebd tail of d(grrg) is a constant feature and carries no
// gradient; grrg_stride skips it.
__global__ void gram_backward_kernel(int n_node,
                                     int ng,
                                     int axis,
                                     int grrg_stride,
                                     const float* __restrict__ d_grrg,
                                     const float* __restrict__ d_rot,
                                     const float* __restrict__ gr,
                                     float* __restrict__ dgr) {
  const int n = blockIdx.x;
  extern __shared__ float sm[];  // [0, 4*ng): gr; [4*ng, ...): d_grrg row
  float* s_gr = sm;
  float* s_dg = sm + 4 * ng;
  for (int t = threadIdx.x; t < 4 * ng; t += blockDim.x) {
    s_gr[t] = gr[(long)n * 4 * ng + t];
  }
  for (int t = threadIdx.x; t < ng * axis; t += blockDim.x) {
    s_dg[t] = d_grrg[(long)n * grrg_stride + t];
  }
  __syncthreads();
  for (int c = threadIdx.x; c < ng; c += blockDim.x) {
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int j = 0; j < axis; ++j) {
      const float d = s_dg[c * axis + j];
      acc0 = fmaf(d, s_gr[0 * ng + j], acc0);
      acc1 = fmaf(d, s_gr[1 * ng + j], acc1);
      acc2 = fmaf(d, s_gr[2 * ng + j], acc2);
      acc3 = fmaf(d, s_gr[3 * ng + j], acc3);
    }
    if (c < axis) {
      for (int i = 0; i < ng; ++i) {
        const float d = s_dg[i * axis + c];
        acc0 = fmaf(d, s_gr[0 * ng + i], acc0);
        acc1 = fmaf(d, s_gr[1 * ng + i], acc1);
        acc2 = fmaf(d, s_gr[2 * ng + i], acc2);
        acc3 = fmaf(d, s_gr[3 * ng + i], acc3);
      }
    }
    if (d_rot) {
      acc1 += d_rot[((long)n * ng + c) * 3 + 0];
      acc2 += d_rot[((long)n * ng + c) * 3 + 1];
      acc3 += d_rot[((long)n * ng + c) * 3 + 2];
    }
    dgr[((long)n * 4 + 0) * ng + c] = acc0;
    dgr[((long)n * 4 + 1) * ng + c] = acc1;
    dgr[((long)n * 4 + 2) * ng + c] = acc2;
    dgr[((long)n * 4 + 3) * ng + c] = acc3;
  }
}

// ======================================================================
// Host-side launch utilities
// ======================================================================
const float* optional_ptr(const torch::Tensor& t) {
  return t.defined() && t.numel() ? t.data_ptr<float>() : nullptr;
}

int persistent_grid(long n_edge, int ctas_per_sm, int tile = kTileEdges) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return (int)std::min((long)sms * ctas_per_sm, ceil_div(n_edge, tile));
}

// CSR edge ordering shared by the descriptor forward entry points (the
// backwards receive the order as a saved tensor).
torch::Tensor build_edge_order(const torch::Tensor& edge_index,
                               const torch::Tensor& edge_mask,
                               long n_edge,
                               long n_node,
                               cudaStream_t stream) {
  TORCH_CHECK(
      n_edge <= std::numeric_limits<int>::max(),
      "dpa1_graph_descriptor: edge count exceeds the int32 order limit");
  auto i32 =
      torch::TensorOptions().dtype(torch::kInt32).device(edge_index.device());
  auto counts = torch::zeros({n_node + 1}, i32);
  auto unsorted = torch::zeros({1}, i32);
  const long* dst = edge_index.data_ptr<long>() + n_edge;
  edge_order_count_kernel<<<ceil_div(n_edge, 256), 256, 0, stream>>>(
      n_edge, dst, edge_mask.data_ptr<bool>(), counts.data_ptr<int>(),
      unsorted.data_ptr<int>());
  auto offsets = torch::empty({n_node + 1}, i32);
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts.data_ptr<int>(),
                                  offsets.data_ptr<int>(), n_node + 1, stream);
    auto temp =
        torch::empty({(long)temp_bytes}, torch::TensorOptions()
                                             .dtype(torch::kUInt8)
                                             .device(edge_index.device()));
    cub::DeviceScan::ExclusiveSum(temp.data_ptr(), temp_bytes,
                                  counts.data_ptr<int>(),
                                  offsets.data_ptr<int>(), n_node + 1, stream);
  }
  auto cursor = torch::zeros({n_node}, i32);
  auto order = torch::empty({n_edge}, i32);
  edge_order_scatter_kernel<<<ceil_div(n_edge, 256), 256, 0, stream>>>(
      n_edge, dst, offsets.data_ptr<int>(), unsorted.data_ptr<int>(),
      cursor.data_ptr<int>(), order.data_ptr<int>());
  DPA1_CHECK_LAUNCH("dpa1_graph edge order");
  return order;
}

}  // namespace
