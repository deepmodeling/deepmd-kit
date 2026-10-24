// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Per-frame bookkeeping shared by the CUDA units of the latent-charge long
// range: the workspace of a call, the offsets of the frames on the node axis,
// the sums over the nodes of every frame in a fixed order, and the statistics
// of the charge constraint.
//
// The kernels are defined in an unnamed namespace, so that every unit that
// includes this header compiles its own copy.

#pragma once

#include <cuda_runtime.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>

namespace {

// The intermediate buffers of one call, carved out of a single device
// allocation: ``reserve`` lays out a buffer of ``count`` elements and returns
// its byte offset, and ``at`` its address once ``allocate`` has made the
// allocation. Every buffer starts on a 256-byte boundary.
class Workspace {
 public:
  template <typename T>
  int64_t reserve(int64_t count) {
    bytes_ = (bytes_ + kAlignment - 1) / kAlignment * kAlignment;
    const int64_t offset = bytes_;
    bytes_ += count * static_cast<int64_t>(sizeof(T));
    return offset;
  }

  void allocate(const torch::Device& device) {
    storage_ = torch::empty(
        {std::max<int64_t>(bytes_, 1)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
  }

  template <typename T>
  T* at(int64_t offset) const {
    return reinterpret_cast<T*>(storage_.data_ptr<uint8_t>() + offset);
  }

 private:
  static constexpr int64_t kAlignment = 256;
  int64_t bytes_ = 0;
  torch::Tensor storage_;
};

// Index of the frame holding ``node``, or -1 for a padding node past every
// frame. ``frame_ptr`` holds ``frame_count + 1`` offsets.
__device__ __forceinline__ int frame_of(const int64_t* frame_ptr,
                                        int frame_count,
                                        int64_t node) {
  if (node >= frame_ptr[frame_count]) {
    return -1;
  }
  int low = 0;
  int high = frame_count;
  while (high - low > 1) {
    const int middle = (low + high) / 2;
    if (frame_ptr[middle] <= node) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

// Writes the offsets of the frames on the node axis: frame_ptr[f] counts the
// nodes of the frames before f, for f in [0, frame_count]. One block scans
// the counts, each thread a contiguous run of frames.
constexpr int kOffsetThreads = 256;

__global__ void __launch_bounds__(kOffsetThreads)
    frame_offsets(const int64_t* __restrict__ n_node_per_frame,
                  int frame_count,
                  int64_t* __restrict__ frame_ptr) {
  __shared__ int64_t run_total[kOffsetThreads];
  const int run = (frame_count + kOffsetThreads - 1) / kOffsetThreads;
  const int begin = min(frame_count, static_cast<int>(threadIdx.x) * run);
  const int end = min(frame_count, begin + run);
  int64_t total = 0;
  for (int f = begin; f < end; ++f) {
    total += n_node_per_frame[f];
  }
  run_total[threadIdx.x] = total;
  __syncthreads();
  // Inclusive scan of the run totals.
  for (int stride = 1; stride < kOffsetThreads; stride *= 2) {
    const int64_t addend =
        threadIdx.x >= stride ? run_total[threadIdx.x - stride] : 0;
    __syncthreads();
    run_total[threadIdx.x] += addend;
    __syncthreads();
  }
  int64_t offset = threadIdx.x == 0 ? 0 : run_total[threadIdx.x - 1];
  for (int f = begin; f < end; ++f) {
    frame_ptr[f] = offset;
    offset += n_node_per_frame[f];
  }
  if (threadIdx.x == kOffsetThreads - 1) {
    frame_ptr[frame_count] = run_total[kOffsetThreads - 1];
  }
}

// Per-frame sums. The nodes of a frame are split into ``gridDim.y``
// contiguous slices, one block of kFrameThreads each, and a finalizing kernel
// adds the float64 sums of the slices in slice order.
constexpr int kFrameThreads = 256;
// Nodes per slice of a frame that a slice pass aims for.
constexpr int kSliceNodes = 2048;
constexpr int kMaxSlices = 256;

struct NodeRange {
  int64_t begin;
  int64_t end;
};

// The nodes of the slice of the frame that this block sums.
__device__ __forceinline__ NodeRange
slice_of_frame(const int64_t* __restrict__ frame_ptr) {
  const int64_t frame_begin = frame_ptr[blockIdx.x];
  const int64_t frame_end = frame_ptr[blockIdx.x + 1];
  const int64_t length = (frame_end - frame_begin + gridDim.y - 1) / gridDim.y;
  const int64_t begin = frame_begin + blockIdx.y * length;
  return {begin, min(frame_end, begin + length)};
}

// Adds ``value`` over the threads of the block in a fixed order and writes the
// totals to the entry of this slice in ``slice_sum``.
template <int kComponents>
__device__ __forceinline__ void write_slice_sum(
    const double (&value)[kComponents], double* __restrict__ slice_sum) {
  __shared__ double partial[kComponents][kFrameThreads];
#pragma unroll
  for (int k = 0; k < kComponents; ++k) {
    partial[k][threadIdx.x] = value[k];
  }
  __syncthreads();
  for (int stride = kFrameThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int k = 0; k < kComponents; ++k) {
        partial[k][threadIdx.x] += partial[k][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < kComponents) {
    slice_sum[(static_cast<int64_t>(blockIdx.x) * gridDim.y + blockIdx.y) *
                  kComponents +
              threadIdx.x] = partial[threadIdx.x][0];
  }
}

// Per-frame statistics of the charge constraint: the shift of the
// constrained channel, the number of charged nodes, and the potential of the
// charged nodes in that channel summed over them.
constexpr int kStatShift = 0;
constexpr int kStatCount = 1;
constexpr int kStatPotential = 2;
constexpr int kStatWidth = 3;

// Sums, over one slice of one frame, the number of charged nodes and their raw
// charges in the constrained channel.
__global__ void __launch_bounds__(kFrameThreads)
    charge_slices(const float* __restrict__ charge,
                  const bool* __restrict__ node_mask,
                  const int64_t* __restrict__ frame_ptr,
                  int channels,
                  int channel,
                  double* __restrict__ slice_sum) {
  const NodeRange slice = slice_of_frame(frame_ptr);
  double value[2] = {0.0, 0.0};
  for (int64_t node = slice.begin + threadIdx.x; node < slice.end;
       node += kFrameThreads) {
    if (node_mask[node]) {
      value[0] += 1.0;
      value[1] += charge[node * channels + channel];
    }
  }
  write_slice_sum<2>(value, slice_sum);
}

// The shift s_f = (sum_f c - Q_f) / max(n_f, 1) of the constrained channel of
// every frame, from the slice sums of charge_slices.
__global__ void charge_shift(const double* __restrict__ slice_sum,
                             int slice_count,
                             int frame_count,
                             const double* __restrict__ total_charge,
                             double* __restrict__ frame_stat) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= frame_count) {
    return;
  }
  double count = 0.0;
  double sum = 0.0;
  for (int slice = 0; slice < slice_count; ++slice) {
    const double* entry =
        slice_sum + (static_cast<int64_t>(f) * slice_count + slice) * 2;
    count += entry[0];
    sum += entry[1];
  }
  // A frame without a charged node has nothing to shift.
  count = fmax(count, 1.0);
  frame_stat[f * kStatWidth + kStatShift] = (sum - total_charge[f]) / count;
  frame_stat[f * kStatWidth + kStatCount] = count;
}

}  // namespace
