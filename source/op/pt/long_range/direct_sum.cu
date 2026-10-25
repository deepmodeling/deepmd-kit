// SPDX-License-Identifier: LGPL-3.0-or-later
//
// CUDA kernels of the latent-charge long range of non-periodic frames, by the
// direct sum over their pairs (schema and formulas in ops.cc, the entry in
// latent_charge_sum.cu).
//
// Every node reads every node of its frame, so the memory is linear in the
// node count while the work is quadratic. One thread owns one node and walks
// the nodes of its frame in tiles staged in shared memory; the partner of a
// thread at a given step is the same for the whole block, so each tile entry
// is a broadcast read. Each pair is evaluated by both of its nodes, which
// keeps every sum in the registers of its owner: no atomic operation is
// needed, and the result does not depend on scheduling.
//
// Positions are carried relative to the first node of their frame as a
// double-float pair: a float32 head and the float32 tail of the remainder.
// The displacement of two nodes, formed as the difference of the heads plus
// the difference of the tails, keeps the float32 precision of the
// displacement itself wherever the frame lies, while every pair evaluates in
// float32; the far field of the erf kernel forms it from the heads alone (see
// evaluate_kernel). For the erf kernel the positions are scaled by alpha, so
// that the pair loop evaluates erf(x) / x and its slope without a
// coefficient, and the reduction restores the units.
//
// A thread sums one tile in float32 and adds the tile sums with compensated
// float32 arithmetic. The partner axis is split into contiguous chunks, so
// that a launch fills the device several times over; every chunk writes its
// sums, and a second pass adds the chunks in a fixed order in float64. A
// system too small to fill the device with tiles of kWideTile nodes takes
// tiles of one warp, so that its pairs spread over more multiprocessors. Its
// chunks hold kNarrowChunkTiles tiles each, counted from the first partner of
// each block, rather than equal parts of the partner range: the chunks of a
// block then do not depend on the partners past the frames of its nodes.
//
// The charge constraint and the per-frame virial reduce over the nodes of
// every frame, in slices whose float64 sums are added in a fixed order, so
// that every output is independent of scheduling.
//
// The nodes whose sums a call evaluates, its owners, are either every node
// or, across the domains of a decomposed system, the charged nodes of one
// MPI rank, which the entry gathers with those of every rank into one frame;
// each rank evaluates its own nodes against the whole frame, so that the
// quadratic work divides among the ranks. Every rank shifts the constrained
// channel by the same sum of the gathered charges; the mean potential of the
// constraint adds the sums of the ranks in rank order, and the per-frame
// virial is the share of the owners of the rank.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include "charge_sum.h"
#include "frame_sums.cuh"

namespace {

// Threads per block of the per-node kernels.
constexpr int kThreads = 128;
// Nodes of a tile of the pair kernel: a block owns one tile of nodes and
// walks the partners of its chunk in tiles of the same length. Every thread
// sums the pairs of a tile in float32 and adds the tile sum to its
// compensated sums.
constexpr int kWideTile = 128;
constexpr int kNarrowTile = 32;
// Partners evaluated together, which share every coefficient load.
constexpr int kBatch = 4;
constexpr int kMaxGaussians = 32;
// Fills of the resident block slots of the device that one launch aims for.
constexpr int kWaves = 8;
// Tiles per chunk of wide tiles where a frame that spans every partner takes
// the most chunks it may: pair_launch rounds the chunk count up from its tile
// count over this.
constexpr int kWideChunkTiles = 2;
// Tiles of a chunk of narrow tiles: two, so that a chunk that holds a tile
// entirely past the partners of a frame holds at most one tile with partners
// of that frame. The compensated sum of one tile carries no error, so that
// the zero sums of the tiles past the frame leave it unchanged, as the frame
// on its own would.
constexpr int kNarrowChunkTiles = 2;

enum KernelKind : int { kLes = 0, kSog = 1 };

// Sums of one node: the potential of every channel, the force direction and
// the six independent components of the virial.
template <int kChannels>
constexpr int sum_width() {
  return kChannels + 3 + 6;
}

// Running sum with Kahan compensation.
struct CompensatedSum {
  float sum = 0.f;
  float error = 0.f;

  __device__ __forceinline__ void add(float value) {
    const float corrected = value - error;
    const float total = sum + corrected;
    error = (total - sum) - corrected;
    sum = total;
  }
};

// Writes the double-float positions relative to the first node of each
// frame, scaled by alpha for the erf kernel, the frame of every node, and the
// charges of the pairs: the raw charges of the charged nodes, shifted in the
// constrained channel, and zero for every other node. ``alpha`` is a device
// scalar, or null for a kernel read in plain units; ``frame_stat`` is null
// without a constraint.
__global__ void prepare_nodes(const double* __restrict__ coord,
                              const float* __restrict__ charge,
                              const bool* __restrict__ node_mask,
                              const int64_t* __restrict__ frame_ptr,
                              int frame_count,
                              int node_count,
                              int channels,
                              int channel,
                              const double* __restrict__ alpha,
                              const double* __restrict__ frame_stat,
                              float4* __restrict__ head,
                              float4* __restrict__ tail,
                              int* __restrict__ frame,
                              float* __restrict__ pair_charge) {
  const int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= node_count) {
    return;
  }
  const double scale = alpha == nullptr ? 1.0 : *alpha;
  const int f = frame_of(frame_ptr, frame_count, node);
  double3 offset = make_double3(0.0, 0.0, 0.0);
  if (f >= 0) {
    const int64_t first = frame_ptr[f];
    offset.x = scale * (coord[3 * node + 0] - coord[3 * first + 0]);
    offset.y = scale * (coord[3 * node + 1] - coord[3 * first + 1]);
    offset.z = scale * (coord[3 * node + 2] - coord[3 * first + 2]);
  }
  const float hx = __double2float_rn(offset.x);
  const float hy = __double2float_rn(offset.y);
  const float hz = __double2float_rn(offset.z);
  head[node] = make_float4(hx, hy, hz, 0.f);
  tail[node] = make_float4(__double2float_rn(offset.x - hx),
                           __double2float_rn(offset.y - hy),
                           __double2float_rn(offset.z - hz), 0.f);
  frame[node] = f;
  const bool charged = f >= 0 && node_mask[node];
  const double shift = charged && frame_stat != nullptr
                           ? frame_stat[f * kStatWidth + kStatShift]
                           : 0.0;
  for (int c = 0; c < channels; ++c) {
    const int64_t index = static_cast<int64_t>(node) * channels + c;
    const double value = charged ? static_cast<double>(charge[index]) : 0.0;
    pair_charge[index] =
        static_cast<float>(c == channel ? value - shift : value);
  }
}

// K and K'(r) / r of a batch of pairs at the squared distances ``r2``.
//
// The erf kernel reads positions scaled by alpha, x = alpha r, and evaluates
// erf(x) / x and its slope (2 / sqrt(pi) exp(-x^2) - erf(x) / x) / x^2, with
// erf(x) = 1 - t P(t) exp(-x^2), t = 1 / (1 + p x), and the coefficients of
// Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7), so that one
// exponential serves the kernel and its slope. The two terms of the slope
// agree to second order in x, which amplifies the error of the approximation
// by 1 / x^2; below x^2 = kSeriesSquared the kernel and its slope therefore
// come from their Taylor series in x^2,
//
//   erf(x) / x  = 2 / sqrt(pi) sum_n (-1)^n x^(2n) / (n! (2n + 1)),
//   slope       = 2 / sqrt(pi) sum_{n >= 1} (-1)^n 2n x^(2n - 2) /
//                 (n! (2n + 1)),
//
// whose first six terms reach float32 precision there. The relative error of
// the kernel stays below 1e-6 for every x, that of its slope below 7e-6, the
// largest just above the threshold. A warp evaluates the series only for a
// batch with a pair below the threshold: a batch that pairs a node with
// itself, or a contact shorter than 0.4 / alpha, which for an alpha of 0.5 per
// angstrom or more is shorter than every bond but that of H2. Every other
// batch evaluates the closed form without a branch.
//
// Beyond x^2 = kFarSquared the float32 kernel equals 1 / x and its slope
// -1 / x^3 bit for bit: exp(-x^2) and erfc(x) fall below half a unit in the
// last place of the terms they would correct. A warp whose batch lies there
// entirely skips the evaluation of erf, which most pairs of a large system
// do, and forms the displacements of the batch from the heads alone. The
// omitted tails move a displacement by at most 2^-24 (|x_i| + |x_j|), with x
// the alpha-scaled offset of a node from the first node of its frame: the
// rounding of float32 positions over the extent of the frame, against a
// distance of at least sqrt(kFarSquared). For a cluster of 100,000 carbon
// atoms at alpha = 0.5 per angstrom, with offsets of up to 26 units, the
// potential and the force keep the error of the exact displacements.
//
// The sum of Gaussians evaluates sum_k a_k 2^(-c_k r^2) and its slope
// sum_k s_k 2^(-c_k r^2), s_k = -2 ln(2) c_k a_k, from the staged
// coefficients ``{a_k, -c_k, s_k}``.
constexpr float kFarSquared = 20.25f;
constexpr float kSeriesSquared = 0.16f;

// A series in x^2 with the coefficients ``c``, in increasing powers, by
// Horner's rule.
template <int kTerms>
__device__ __forceinline__ float even_series(const float (&c)[kTerms],
                                             float x2) {
  float sum = c[kTerms - 1];
#pragma unroll
  for (int n = kTerms - 2; n >= 0; --n) {
    sum = fmaf(sum, x2, c[n]);
  }
  return sum;
}

// Whether every pair of the batch of every thread of the warp lies in the far
// field of the erf kernel, judged from the head displacements, whose squared
// lengths it writes to ``r2``. The sum of Gaussians has no far field.
template <int kKernel>
__device__ __forceinline__ bool far_batch(const float (&dx)[kBatch],
                                          const float (&dy)[kBatch],
                                          const float (&dz)[kBatch],
                                          float (&r2)[kBatch]) {
  if constexpr (kKernel == kSog) {
    return false;
  } else {
    float nearest = std::numeric_limits<float>::infinity();
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      r2[b] = fmaf(dx[b], dx[b], fmaf(dy[b], dy[b], dz[b] * dz[b]));
      nearest = fminf(nearest, r2[b]);
    }
    return __all_sync(0xffffffffu, nearest >= kFarSquared);
  }
}

// erf(x) / x and its slope by the approximation of Abramowitz and Stegun at
// the squared distances ``r2``.
__device__ __forceinline__ void les_closed_form(const float (&r2)[kBatch],
                                                float (&value)[kBatch],
                                                float (&slope)[kBatch]) {
#pragma unroll
  for (int b = 0; b < kBatch; ++b) {
    const float inverse = rsqrtf(r2[b]);
    const float x = r2[b] * inverse;
    const float t = __fdividef(1.f, fmaf(0.3275911f, x, 1.f));
    const float poly =
        t *
        fmaf(
            t,
            fmaf(t, fmaf(t, fmaf(t, 1.061405429f, -1.453152027f), 1.421413741f),
                 -0.284496736f),
            0.254829592f);
    const float gaussian = exp2f(-1.44269504088896341f * r2[b]);
    value[b] = fmaf(-poly, gaussian, 1.f) * inverse;
    slope[b] =
        fmaf(1.12837916709551257f, gaussian, -value[b]) * (inverse * inverse);
  }
}

template <int kKernel>
__device__ __forceinline__ void evaluate_kernel(const float (&r2)[kBatch],
                                                const float4* gaussians,
                                                int gaussian_count,
                                                float (&value)[kBatch],
                                                float (&slope)[kBatch]) {
  if constexpr (kKernel == kLes) {
    float nearest = r2[0];
#pragma unroll
    for (int b = 1; b < kBatch; ++b) {
      nearest = fminf(nearest, r2[b]);
    }
    if (!__any_sync(0xffffffffu, nearest < kSeriesSquared)) {
      les_closed_form(r2, value, slope);
      return;
    }
    constexpr float kValueSeries[6] = {1.128379167f,    -0.376126389f,
                                       0.1128379167f,   -0.02686617065f,
                                       0.005223977625f, -0.0008548327023f};
    constexpr float kSlopeSeries[6] = {-0.7522527781f,   0.4513516668f,
                                       -0.1611970239f,   0.041791821f,
                                       -0.008548327023f, 0.001446639958f};
    // The closed form reads the threshold where the series is selected.
    float clamped[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      clamped[b] = fmaxf(r2[b], kSeriesSquared);
    }
    les_closed_form(clamped, value, slope);
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      if (r2[b] < kSeriesSquared) {
        value[b] = even_series(kValueSeries, r2[b]);
        slope[b] = even_series(kSlopeSeries, r2[b]);
      }
    }
  } else {
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      value[b] = 0.f;
      slope[b] = 0.f;
    }
    for (int k = 0; k < gaussian_count; ++k) {
      const float4 coefficient = gaussians[k];
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        const float weight = exp2f(coefficient.y * r2[b]);
        value[b] = fmaf(coefficient.x, weight, value[b]);
        slope[b] = fmaf(coefficient.z, weight, slope[b]);
      }
    }
  }
}

// Partners a tile holds besides those of the frame of the node.
enum Exclusion : int {
  // Every partner belongs to the frame and differs from the node.
  kNoExclusion = 0,
  // The node itself is among the partners.
  kExcludeSelf = 1,
  // The partners may also lie in other frames.
  kExcludeOutsideFrame = 2,
};

// The state of the node a thread owns.
template <int kChannels>
struct OwnNode {
  int index;
  int low;   // first node of its frame
  int high;  // past the last node of its frame
  float4 head;
  float4 tail;
  float charge[kChannels];
};

// Adds the unscaled potential, force direction and virial that the partners
// of one staged tile give the node to ``sum``.
template <int kKernel, int kChannels, int kExclusion, int kTile>
__device__ __forceinline__ void accumulate_tile(
    const float4* __restrict__ tile_head,
    const float4* __restrict__ tile_tail,
    const float* __restrict__ tile_charge,
    const float4* __restrict__ gaussians,
    int gaussian_count,
    int start,
    const OwnNode<kChannels>& own,
    float (&sum)[sum_width<kChannels>()]) {
#pragma unroll 4
  for (int base = 0; base < kTile; base += kBatch) {
    float dx[kBatch];
    float dy[kBatch];
    float dz[kBatch];
    float r2[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const float4 other_head = tile_head[base + b];
      dx[b] = own.head.x - other_head.x;
      dy[b] = own.head.y - other_head.y;
      dz[b] = own.head.z - other_head.z;
    }
    float value[kBatch];
    float slope[kBatch];
    if (far_batch<kKernel>(dx, dy, dz, r2)) {
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        const float inverse = rsqrtf(r2[b]);
        value[b] = inverse;
        slope[b] = -inverse * (inverse * inverse);
      }
    } else {
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        const float4 other_tail = tile_tail[base + b];
        dx[b] += own.tail.x - other_tail.x;
        dy[b] += own.tail.y - other_tail.y;
        dz[b] += own.tail.z - other_tail.z;
        // Both kernels are finite at r2 = 0, which a node paired with itself
        // and coincident padding nodes reach with a zero weight.
        r2[b] = fmaf(dx[b], dx[b], fmaf(dy[b], dy[b], dz[b] * dz[b]));
      }
      evaluate_kernel<kKernel>(r2, gaussians, gaussian_count, value, slope);
    }
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int other = start + base + b;
      float weight = 1.f;
      if constexpr (kExclusion == kExcludeSelf) {
        weight = other != own.index ? 1.f : 0.f;
      } else if constexpr (kExclusion == kExcludeOutsideFrame) {
        weight = (other != own.index &&
                  static_cast<unsigned>(other - own.low) <
                      static_cast<unsigned>(own.high - own.low))
                     ? 1.f
                     : 0.f;
      }
      float product = 0.f;
#pragma unroll
      for (int c = 0; c < kChannels; ++c) {
        const float other_charge =
            kExclusion == kNoExclusion
                ? tile_charge[(base + b) * kChannels + c]
                : weight * tile_charge[(base + b) * kChannels + c];
        sum[c] = fmaf(other_charge, value[b], sum[c]);
        product = fmaf(own.charge[c], other_charge, product);
      }
      const float strength = product * slope[b];
      const float sx = strength * dx[b];
      const float sy = strength * dy[b];
      const float sz = strength * dz[b];
      sum[kChannels + 0] += sx;
      sum[kChannels + 1] += sy;
      sum[kChannels + 2] += sz;
      sum[kChannels + 3] = fmaf(sx, dx[b], sum[kChannels + 3]);
      sum[kChannels + 4] = fmaf(sy, dy[b], sum[kChannels + 4]);
      sum[kChannels + 5] = fmaf(sz, dz[b], sum[kChannels + 5]);
      sum[kChannels + 6] = fmaf(sx, dy[b], sum[kChannels + 6]);
      sum[kChannels + 7] = fmaf(sx, dz[b], sum[kChannels + 7]);
      sum[kChannels + 8] = fmaf(sy, dz[b], sum[kChannels + 8]);
    }
  }
}

// Sums, for every owner node of one block and every partner of one chunk, the
// unscaled potential, force direction and virial of the node, and writes them
// to ``partial[chunk][component][node - owner_begin]``. The owners are the
// nodes [owner_begin, owner_begin + owner_count). A chunk is the part
// blockIdx.y of gridDim.y equal parts of the partner range of the block or,
// with ``chunk_tiles`` > 0, its chunk_tiles tiles from tile blockIdx.y *
// chunk_tiles of the range.
template <int kKernel, int kChannels, int kTile>
__global__ void __launch_bounds__(kTile)
    direct_sum_kernel(const float4* __restrict__ head,
                      const float4* __restrict__ tail,
                      const float* __restrict__ charge,
                      const int* __restrict__ frame,
                      const int64_t* __restrict__ frame_ptr,
                      int owner_begin,
                      int owner_count,
                      const double* __restrict__ coefficient,
                      int gaussian_count,
                      int chunk_tiles,
                      float* __restrict__ partial) {
  constexpr int kWidth = sum_width<kChannels>();
  __shared__ float4 tile_head[kTile];
  __shared__ float4 tile_tail[kTile];
  __shared__ float tile_charge[kTile * kChannels];
  __shared__ float4 gaussians[kMaxGaussians];
  __shared__ int range[2];
  __shared__ bool single_frame;

  const int owner_end = owner_begin + owner_count;
  const int first = owner_begin + blockIdx.x * kTile;
  const int node = first + threadIdx.x;
  const int own_frame = node < owner_end ? frame[node] : -1;

  if constexpr (kKernel == kSog) {
    for (int k = threadIdx.x; k < gaussian_count; k += kTile) {
      const float amplitude = static_cast<float>(coefficient[k]);
      const float rate = static_cast<float>(coefficient[gaussian_count + k]);
      gaussians[k] = make_float4(amplitude, -rate,
                                 -1.38629436111989061f * rate * amplitude, 0.f);
    }
  }
  // The partners of the block are the nodes of the frames its nodes belong
  // to: a contiguous range, since frames are contiguous on the node axis and
  // padding nodes trail them.
  if (threadIdx.x == 0) {
    int last = min(first + kTile, owner_end) - 1;
    while (last > first && frame[last] < 0) {
      --last;
    }
    const int low_frame = frame[first];
    const int high_frame = frame[last];
    range[0] = low_frame < 0 ? 0 : static_cast<int>(frame_ptr[low_frame]);
    range[1] = low_frame < 0 ? 0 : static_cast<int>(frame_ptr[high_frame + 1]);
    // A block whose nodes all lie in one frame finds every partner of its
    // range in that frame, which spares the frame test of every pair.
    single_frame = low_frame >= 0 && low_frame == high_frame &&
                   min(first + kTile, owner_end) - 1 == last;
  }
  __syncthreads();

  // The chunk of the partner range this block sums, in whole tiles.
  const int64_t tile_count = (range[1] - range[0] + kTile - 1) / kTile;
  int64_t first_tile = tile_count * blockIdx.y / gridDim.y;
  int64_t end_tile = tile_count * (blockIdx.y + 1) / gridDim.y;
  if (chunk_tiles > 0) {
    first_tile = min(tile_count, int64_t{blockIdx.y} * chunk_tiles);
    end_tile = min(tile_count, first_tile + chunk_tiles);
  }
  const int begin = range[0] + static_cast<int>(first_tile) * kTile;
  const int end = min(range[1], range[0] + static_cast<int>(end_tile) * kTile);

  OwnNode<kChannels> own;
  own.index = node;
  own.low = 0;
  own.high = 0;
  own.head = make_float4(0.f, 0.f, 0.f, 0.f);
  own.tail = own.head;
  if (own_frame >= 0) {
    own.head = head[node];
    own.tail = tail[node];
    own.low = static_cast<int>(frame_ptr[own_frame]);
    own.high = static_cast<int>(frame_ptr[own_frame + 1]);
  }
#pragma unroll
  for (int c = 0; c < kChannels; ++c) {
    own.charge[c] = own_frame >= 0 ? charge[node * kChannels + c] : 0.f;
  }

  CompensatedSum total[kWidth];
  for (int start = begin; start < end; start += kTile) {
    const int partner = start + threadIdx.x;
    const bool staged = partner < end;
    // A slot past the chunk carries a zero charge and contributes nothing.
    tile_head[threadIdx.x] =
        staged ? head[partner] : make_float4(0.f, 0.f, 0.f, 0.f);
    tile_tail[threadIdx.x] =
        staged ? tail[partner] : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int c = 0; c < kChannels; ++c) {
      tile_charge[threadIdx.x * kChannels + c] =
          staged ? charge[partner * kChannels + c] : 0.f;
    }
    __syncthreads();

    float sum[kWidth];
#pragma unroll
    for (int k = 0; k < kWidth; ++k) {
      sum[k] = 0.f;
    }
    // In a single-frame block only the tiles that overlap the block's own
    // nodes hold a node itself, so the tests are confined to them and to the
    // blocks that span frames; the choice is uniform over the block.
    if (!single_frame) {
      accumulate_tile<kKernel, kChannels, kExcludeOutsideFrame, kTile>(
          tile_head, tile_tail, tile_charge, gaussians, gaussian_count, start,
          own, sum);
    } else if (start < first + kTile && first < start + kTile) {
      accumulate_tile<kKernel, kChannels, kExcludeSelf, kTile>(
          tile_head, tile_tail, tile_charge, gaussians, gaussian_count, start,
          own, sum);
    } else {
      accumulate_tile<kKernel, kChannels, kNoExclusion, kTile>(
          tile_head, tile_tail, tile_charge, gaussians, gaussian_count, start,
          own, sum);
    }
#pragma unroll
    for (int k = 0; k < kWidth; ++k) {
      total[k].add(sum[k]);
    }
    __syncthreads();
  }

  if (node < owner_end) {
    float* out = partial +
                 static_cast<int64_t>(blockIdx.y) * kWidth * owner_count +
                 (node - owner_begin);
#pragma unroll
    for (int k = 0; k < kWidth; ++k) {
      out[static_cast<int64_t>(k) * owner_count] = total[k].sum;
    }
  }
}

// Adds the chunk sums of one component of one owner node in chunk order, one
// thread per component and owner, and scales the result into the potential,
// the force or the per-node virial of the owner. With positions scaled by
// alpha, K(r) = alpha f(x), K'(r) / r = alpha^3 f'(x) / x and d = d' / alpha,
// so the potential carries alpha, the force alpha^2 and the virial alpha.
template <int kChannels>
__global__ void reduce_chunks(const float* __restrict__ partial,
                              int owner_count,
                              int chunk_count,
                              double prefactor,
                              const double* __restrict__ alpha,
                              float* __restrict__ potential,
                              float* __restrict__ force,
                              float* __restrict__ atom_virial) {
  constexpr int kWidth = sum_width<kChannels>();
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<int64_t>(kWidth) * owner_count) {
    return;
  }
  const int component = static_cast<int>(index / owner_count);
  const int node = static_cast<int>(index - component * int64_t(owner_count));
  double value = 0.0;
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    value +=
        partial[static_cast<int64_t>(chunk) * kWidth * owner_count + index];
  }
  const double unit = alpha == nullptr ? 1.0 : *alpha;
  if (component < kChannels) {
    potential[node * kChannels + component] =
        static_cast<float>(prefactor * unit * value);
  } else if (component < kChannels + 3) {
    force[node * 3 + component - kChannels] =
        static_cast<float>(-prefactor * unit * unit * value);
  } else {
    // Each pair term is split half to each of its two nodes. The six sums
    // are xx, yy, zz, xy, xz, yz.
    constexpr int kRow[6] = {0, 1, 2, 0, 0, 1};
    constexpr int kColumn[6] = {0, 1, 2, 1, 2, 2};
    const int pair = component - kChannels - 3;
    const float entry = static_cast<float>(-0.5 * prefactor * unit * value);
    float* out = atom_virial + static_cast<int64_t>(node) * 9;
    out[3 * kRow[pair] + kColumn[pair]] = entry;
    out[3 * kColumn[pair] + kRow[pair]] = entry;
  }
}

// Sums, over one slice of one frame, the per-node virial and, with a
// constraint, the potential of the charged nodes in the constrained channel.
template <bool kConstrained>
__global__ void __launch_bounds__(kFrameThreads)
    virial_slices(const float* __restrict__ atom_virial,
                  const float* __restrict__ potential,
                  const bool* __restrict__ node_mask,
                  const int64_t* __restrict__ frame_ptr,
                  int channels,
                  int channel,
                  double* __restrict__ slice_sum) {
  constexpr int kComponents = kConstrained ? 10 : 9;
  const NodeRange slice = slice_of_frame(frame_ptr);
  double value[kComponents] = {};
  for (int64_t node = slice.begin + threadIdx.x; node < slice.end;
       node += kFrameThreads) {
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      value[k] += atom_virial[node * 9 + k];
    }
    if constexpr (kConstrained) {
      if (node_mask[node]) {
        value[9] += potential[node * channels + channel];
      }
    }
  }
  write_slice_sum<kComponents>(value, slice_sum);
}

// Adds the slice sums of virial_slices into the virial of every frame and,
// with a constraint, the potential summed over its charged nodes.
template <bool kConstrained>
__global__ void virial_total(const double* __restrict__ slice_sum,
                             int slice_count,
                             float* __restrict__ virial,
                             double* __restrict__ frame_stat) {
  constexpr int kComponents = kConstrained ? 10 : 9;
  if (threadIdx.x >= kComponents) {
    return;
  }
  double value = 0.0;
  for (int slice = 0; slice < slice_count; ++slice) {
    value +=
        slice_sum[(static_cast<int64_t>(blockIdx.x) * slice_count + slice) *
                      kComponents +
                  threadIdx.x];
  }
  if (threadIdx.x < 9) {
    virial[blockIdx.x * 9 + threadIdx.x] = static_cast<float>(value);
  } else {
    frame_stat[blockIdx.x * kStatWidth + kStatPotential] = value;
  }
}

// Writes the atomic energy q_i . phi_i / 2 of every owner node and the
// derivative of the energy with respect to its raw charges: w_i (phi_i -
// mean_f) in the constrained channel and w_i phi_i in the others, w_i marking
// the charged nodes. ``pair_charge``, ``node_mask`` and ``frame`` start at the
// first owner.
__global__ void project_charges(const float* __restrict__ potential,
                                const float* __restrict__ pair_charge,
                                const bool* __restrict__ node_mask,
                                const int* __restrict__ frame,
                                const double* __restrict__ frame_stat,
                                int owner_count,
                                int channels,
                                int channel,
                                double* __restrict__ atom_energy,
                                float* __restrict__ charge_gradient) {
  const int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= owner_count) {
    return;
  }
  const int f = frame[node];
  const bool charged = f >= 0 && node_mask[node];
  const double mean = charged && frame_stat != nullptr
                          ? frame_stat[f * kStatWidth + kStatPotential] /
                                frame_stat[f * kStatWidth + kStatCount]
                          : 0.0;
  double energy = 0.0;
  for (int c = 0; c < channels; ++c) {
    const int64_t index = static_cast<int64_t>(node) * channels + c;
    const double phi = potential[index];
    energy += static_cast<double>(pair_charge[index]) * phi;
    const double gradient = charged ? phi : 0.0;
    charge_gradient[index] =
        static_cast<float>(c == channel ? gradient - mean : gradient);
  }
  atom_energy[node] = 0.5 * energy;
}

// The staged partners, one row per node, and the sums of the owners.
struct DirectSumBuffers {
  const float4* head;
  const float4* tail;
  const float* charge;
  const int* frame;
  const int64_t* frame_ptr;
  int node_count;
  const double* coefficient;
  int gaussian_count;
  float* potential;
  float* force;
  float* atom_virial;
};

// The launch of the pair kernel: whether it takes narrow tiles, and its
// chunks of the partner axis, ``chunk_count`` of them and, for narrow tiles,
// ``chunk_tiles`` tiles each (direct_sum_kernel).
struct PairLaunch {
  bool narrow;
  int chunk_count;
  int chunk_tiles;
};

// The launch for the given counts of owners and partners. The blocks of one
// launch of wide tiles all carry the same work, so the launch fills the
// resident block slots of the device several times over, which bounds the
// idle share of the last wave. A frame that spans every partner has at most
// one chunk per kWideChunkTiles of its tiles, rounded up, so that in a launch
// with slots to spare no chunk holds more than kWideChunkTiles tiles: the
// longest block sets the duration of such a launch. A system whose wide
// blocks would fill less than a third of the resident block slots leaves too
// few warps on the multiprocessors to hide the latency of the pair loop, and
// takes narrow tiles in chunks of kNarrowChunkTiles, as many as a frame that
// spans every partner holds.
PairLaunch pair_launch(int owner_count,
                       int partner_count,
                       int resident_blocks_per_multiprocessor) {
  const int multiprocessors =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int owner_blocks = (owner_count + kWideTile - 1) / kWideTile;
  const int partner_tiles = (partner_count + kWideTile - 1) / kWideTile;
  const int64_t wanted_blocks = static_cast<int64_t>(kWaves) *
                                resident_blocks_per_multiprocessor *
                                multiprocessors;
  const int wanted =
      static_cast<int>((wanted_blocks + owner_blocks - 1) / owner_blocks);
  const int chunk_limit =
      (partner_tiles + kWideChunkTiles - 1) / kWideChunkTiles;
  const int chunk_count = std::max(1, std::min(wanted, chunk_limit));
  if (3 * int64_t{owner_blocks} * chunk_count >=
      int64_t{resident_blocks_per_multiprocessor} * multiprocessors) {
    return {false, chunk_count, 0};
  }
  const int narrow_tiles = (partner_count + kNarrowTile - 1) / kNarrowTile;
  return {
      true,
      std::max(1, (narrow_tiles + kNarrowChunkTiles - 1) / kNarrowChunkTiles),
      kNarrowChunkTiles};
}

template <int kKernel, int kChannels>
int resident_blocks_per_multiprocessor() {
  // The occupancy is a property of the compiled kernel and of the device; a
  // process drives one device architecture.
  static const int blocks = [] {
    int value = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &value, direct_sum_kernel<kKernel, kChannels, kWideTile>, kWideTile,
        0));
    return std::max(1, value);
  }();
  return blocks;
}

template <int kKernel, int kChannels, int kTile>
void launch_tiles(const DirectSumBuffers& buffers,
                  int owner_begin,
                  int owner_count,
                  const PairLaunch& pair,
                  double prefactor,
                  cudaStream_t stream) {
  constexpr int kWidth = sum_width<kChannels>();
  auto partial = torch::empty(
      {static_cast<int64_t>(pair.chunk_count) * owner_count * kWidth},
      torch::TensorOptions()
          .dtype(torch::kFloat32)
          .device(torch::kCUDA, c10::cuda::current_device()));
  const dim3 grid((owner_count + kTile - 1) / kTile, pair.chunk_count);
  direct_sum_kernel<kKernel, kChannels, kTile><<<grid, kTile, 0, stream>>>(
      buffers.head, buffers.tail, buffers.charge, buffers.frame,
      buffers.frame_ptr, owner_begin, owner_count, buffers.coefficient,
      buffers.gaussian_count, pair.chunk_tiles, partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t reduce_threads = static_cast<int64_t>(kWidth) * owner_count;
  reduce_chunks<kChannels>
      <<<static_cast<unsigned>((reduce_threads + 255) / 256), 256, 0, stream>>>(
          partial.data_ptr<float>(), owner_count, pair.chunk_count, prefactor,
          kKernel == kLes ? buffers.coefficient : nullptr, buffers.potential,
          buffers.force, buffers.atom_virial);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kKernel, int kChannels>
void launch(const DirectSumBuffers& buffers,
            int owner_begin,
            int owner_count,
            double prefactor,
            cudaStream_t stream) {
  const PairLaunch pair =
      pair_launch(owner_count, buffers.node_count,
                  resident_blocks_per_multiprocessor<kKernel, kChannels>());
  if (pair.narrow) {
    launch_tiles<kKernel, kChannels, kNarrowTile>(
        buffers, owner_begin, owner_count, pair, prefactor, stream);
  } else {
    launch_tiles<kKernel, kChannels, kWideTile>(
        buffers, owner_begin, owner_count, pair, prefactor, stream);
  }
}

template <int kKernel>
void dispatch_channels(int channels,
                       const DirectSumBuffers& buffers,
                       int owner_begin,
                       int owner_count,
                       double prefactor,
                       cudaStream_t stream) {
  switch (channels) {
    case 1:
      return launch<kKernel, 1>(buffers, owner_begin, owner_count, prefactor,
                                stream);
    case 2:
      return launch<kKernel, 2>(buffers, owner_begin, owner_count, prefactor,
                                stream);
    case 3:
      return launch<kKernel, 3>(buffers, owner_begin, owner_count, prefactor,
                                stream);
    case 4:
      return launch<kKernel, 4>(buffers, owner_begin, owner_count, prefactor,
                                stream);
    case 5:
      return launch<kKernel, 5>(buffers, owner_begin, owner_count, prefactor,
                                stream);
    case 6:
      return launch<kKernel, 6>(buffers, owner_begin, owner_count, prefactor,
                                stream);
    case 7:
      return launch<kKernel, 7>(buffers, owner_begin, owner_count, prefactor,
                                stream);
    case 8:
      return launch<kKernel, 8>(buffers, owner_begin, owner_count, prefactor,
                                stream);
    default:
      TORCH_CHECK(false, "latent_charge_sum: unsupported channel count ",
                  channels);
  }
}

}  // namespace

namespace deepmd {
namespace long_range {

// Evaluates the owner nodes [owner_begin, owner_begin + owner_count) against
// the nodes of their frames. Returns the outputs of the owners, row r for
// owner owner_begin + r, and the per-frame virial of the owners. Owners other
// than every node lie in a single frame. A non-null ``communicator`` spans
// the domains whose owners share that frame, over which the potential of the
// constraint is summed.
ChargeSumOutputs direct_sum(const torch::Tensor& coord,
                            const torch::Tensor& charge,
                            const torch::Tensor& node_mask,
                            const torch::Tensor& n_node_per_frame,
                            const torch::Tensor& total_charge,
                            const torch::Tensor& coefficient,
                            int64_t kernel,
                            int64_t channel,
                            double prefactor,
                            int owner_begin,
                            int owner_count,
                            const torch::Tensor* communicator) {
  TORCH_CHECK(coord.size(0) < std::numeric_limits<int>::max() / kThreads,
              "latent_charge_sum: too many nodes");
  const int node_count = static_cast<int>(coord.size(0));
  const int frame_count = static_cast<int>(n_node_per_frame.size(0));
  const int64_t channels = charge.size(1);
  const auto float_options = charge.options();
  const auto double_options = coord.options();
  if (node_count == 0 || frame_count == 0) {
    return {torch::zeros({owner_count, 1}, double_options),
            torch::zeros({owner_count, channels}, float_options),
            torch::zeros({owner_count, 3}, float_options),
            torch::zeros({owner_count, 9}, float_options),
            torch::zeros({frame_count, 9}, float_options)};
  }
  const bool every_node = owner_begin == 0 && owner_count == node_count;
  TORCH_CHECK(every_node || frame_count == 1,
              "latent_charge_sum: owners other than every node lie in "
              "one frame");
  const bool constrained = channel >= 0;
  const int constrained_channel = static_cast<int>(channel);
  // The kernels write every entry of the outputs.
  auto atom_energy = torch::empty({owner_count, 1}, double_options);
  auto charge_gradient = torch::empty({owner_count, channels}, float_options);
  auto force = torch::empty({owner_count, 3}, float_options);
  auto atom_virial = torch::empty({owner_count, 9}, float_options);
  auto virial = torch::empty({frame_count, 9}, float_options);

  const auto counts = n_node_per_frame.to(torch::kInt64).contiguous();
  const auto slices_for = [frame_count](int64_t nodes) {
    const int64_t average_nodes = (nodes + frame_count - 1) / frame_count;
    return static_cast<int>(std::min<int64_t>(
        kMaxSlices,
        std::max<int64_t>(1, (average_nodes + kSliceNodes - 1) / kSliceNodes)));
  };
  const int slice_count = slices_for(node_count);
  const int owner_slice_count = slices_for(owner_count);
  const int components = constrained ? 10 : 9;
  Workspace workspace;
  const int64_t frame_ptr_at = workspace.reserve<int64_t>(frame_count + 1);
  const int64_t head_at = workspace.reserve<float4>(node_count);
  const int64_t tail_at = workspace.reserve<float4>(node_count);
  const int64_t frame_at = workspace.reserve<int>(node_count);
  const int64_t pair_charge_at =
      workspace.reserve<float>(int64_t{node_count} * channels);
  const int64_t potential_at =
      workspace.reserve<float>(int64_t{owner_count} * channels);
  const int64_t frame_stat_at =
      workspace.reserve<double>(int64_t{frame_count} * kStatWidth);
  const int64_t slice_charge_at =
      workspace.reserve<double>(int64_t{frame_count} * slice_count * 2);
  const int64_t slice_virial_at = workspace.reserve<double>(
      int64_t{frame_count} * owner_slice_count * components);
  workspace.allocate(coord.device());
  auto* frame_ptr = workspace.at<int64_t>(frame_ptr_at);
  auto* head = workspace.at<float4>(head_at);
  auto* tail = workspace.at<float4>(tail_at);
  auto* frame = workspace.at<int>(frame_at);
  auto* pair_charge = workspace.at<float>(pair_charge_at);
  auto* potential = workspace.at<float>(potential_at);
  auto* frame_stat =
      constrained ? workspace.at<double>(frame_stat_at) : nullptr;
  auto* slice_charge = workspace.at<double>(slice_charge_at);
  auto* slice_virial = workspace.at<double>(slice_virial_at);

  const auto stream = at::cuda::getCurrentCUDAStream();
  frame_offsets<<<1, kOffsetThreads, 0, stream>>>(counts.data_ptr<int64_t>(),
                                                  frame_count, frame_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (constrained) {
    charge_slices<<<dim3(frame_count, slice_count), kFrameThreads, 0, stream>>>(
        charge.data_ptr<float>(), node_mask.data_ptr<bool>(), frame_ptr,
        static_cast<int>(channels), constrained_channel, slice_charge);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    charge_shift<<<(frame_count + kFrameThreads - 1) / kFrameThreads,
                   kFrameThreads, 0, stream>>>(
        slice_charge, slice_count, frame_count, total_charge.data_ptr<double>(),
        frame_stat);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // Alpha stays on the device, so that a call never waits for it.
  const double* alpha =
      kernel == kLes ? coefficient.data_ptr<double>() : nullptr;
  prepare_nodes<<<(node_count + kThreads - 1) / kThreads, kThreads, 0,
                  stream>>>(coord.data_ptr<double>(), charge.data_ptr<float>(),
                            node_mask.data_ptr<bool>(), frame_ptr, frame_count,
                            node_count, static_cast<int>(channels),
                            constrained_channel, alpha, frame_stat, head, tail,
                            frame, pair_charge);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (owner_count > 0) {
    const DirectSumBuffers buffers{head,
                                   tail,
                                   pair_charge,
                                   frame,
                                   frame_ptr,
                                   node_count,
                                   coefficient.data_ptr<double>(),
                                   static_cast<int>(coefficient.numel() / 2),
                                   potential,
                                   force.data_ptr<float>(),
                                   atom_virial.data_ptr<float>()};
    if (kernel == kLes) {
      dispatch_channels<kLes>(static_cast<int>(channels), buffers, owner_begin,
                              owner_count, prefactor, stream.stream());
    } else {
      dispatch_channels<kSog>(static_cast<int>(channels), buffers, owner_begin,
                              owner_count, prefactor, stream.stream());
    }
  }
  // The frames of the owners, on the axis of the owners: the frames
  // themselves when every node is an owner, else the one frame the owners
  // share, written on the device.
  torch::Tensor owner_frame_tensor;
  const int64_t* owner_frame_ptr = frame_ptr;
  if (!every_node) {
    owner_frame_tensor =
        torch::full({2}, int64_t{owner_count}, counts.options());
    owner_frame_tensor.select(0, 0).zero_();
    owner_frame_ptr = owner_frame_tensor.data_ptr<int64_t>();
  }
  const bool* owner_mask = node_mask.data_ptr<bool>() + owner_begin;
  const dim3 owner_slice_grid(frame_count, owner_slice_count);
  if (constrained) {
    virial_slices<true><<<owner_slice_grid, kFrameThreads, 0, stream>>>(
        atom_virial.data_ptr<float>(), potential, owner_mask, owner_frame_ptr,
        static_cast<int>(channels), constrained_channel, slice_virial);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    virial_total<true><<<frame_count, 32, 0, stream>>>(
        slice_virial, owner_slice_count, virial.data_ptr<float>(), frame_stat);
  } else {
    virial_slices<false><<<owner_slice_grid, kFrameThreads, 0, stream>>>(
        atom_virial.data_ptr<float>(), potential, owner_mask, owner_frame_ptr,
        static_cast<int>(channels), constrained_channel, slice_virial);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    virial_total<false><<<frame_count, 32, 0, stream>>>(
        slice_virial, owner_slice_count, virial.data_ptr<float>(), nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (constrained && communicator != nullptr) {
    // The mean potential of the constraint runs over the charged nodes of
    // every domain.
    auto stat =
        torch::from_blob(frame_stat, {frame_count, kStatWidth}, double_options);
    auto potential_sum = stat.select(1, kStatPotential);
    potential_sum.copy_(
        sum_over_domains(*communicator, potential_sum.contiguous()));
  }
  if (owner_count > 0) {
    project_charges<<<(owner_count + kThreads - 1) / kThreads, kThreads, 0,
                      stream>>>(
        potential, pair_charge + int64_t{owner_begin} * channels, owner_mask,
        frame + owner_begin, frame_stat, owner_count,
        static_cast<int>(channels), constrained_channel,
        atom_energy.data_ptr<double>(), charge_gradient.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {atom_energy, charge_gradient, force, atom_virial, virial};
}

}  // namespace long_range
}  // namespace deepmd
