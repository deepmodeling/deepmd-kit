// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Geometrically compressed DPA1 graph descriptor over destination CSR edges.
//
// A warp owns one center node. Widths from 16 through 64 use two 16-lane
// sub-warps on alternating edges; each lane evaluates one or more spline
// channels. Wider tables retain one edge per warp to bound register pressure.
// The node moment and its Gram contraction remain in the same kernel.
//
// The backward recomputes the inexpensive spline value/derivative, contracts
// the descriptor gradient into the four environment channels, and writes each
// edge gradient exactly once. It is inference-oriented (one backward); the
// registered Python autograd bridge continues to expose the edge-vector
// gradient for the level-1 graph path.
//
// Every specialization has balanced (two CTA/SM launch bound) and occupancy
// (four CTA/SM launch bound) resource variants. The first uncaptured call times
// 128- and 256-thread launches on a bounded node sample and caches the selected
// variant per device and workload class. Device-family defaults remain valid
// when timing is disabled or CUDA Graph capture is active.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cmath>
#include <optional>
#include <tuple>

#include "dpa1_graph_compress_tuning.h"

namespace {

using deepmd::dpa1_compress_tuning::device_properties;
using deepmd::dpa1_compress_tuning::KernelDirection;
using deepmd::dpa1_compress_tuning::LaunchConfig;
using deepmd::dpa1_compress_tuning::ResourcePolicy;
using deepmd::dpa1_compress_tuning::select_launch_config;
using deepmd::dpa1_compress_tuning::TuningKey;
using deepmd::dpa1_compress_tuning::type_count_class;
using deepmd::dpa1_compress_tuning::workload_degree_class;
using deepmd::dpa1_compress_tuning::workload_size_class;

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

#define COMPRESS_CHECK_LAUNCH(name)                                           \
  do {                                                                        \
    const cudaError_t error = cudaGetLastError();                             \
    TORCH_CHECK(error == cudaSuccess, name, ": ", cudaGetErrorString(error)); \
  } while (0)

__device__ __forceinline__ float switch_value(float radius,
                                              float lower,
                                              float upper) {
  const float coordinate =
      __fdividef(fminf(fmaxf(radius, lower), upper) - lower, upper - lower);
  const float square = coordinate * coordinate;
  return square * coordinate * (-6.0f * square + 15.0f * coordinate - 10.0f) +
         1.0f;
}

__device__ __forceinline__ float switch_derivative(float radius,
                                                   float lower,
                                                   float upper) {
  if (radius <= lower || radius >= upper) {
    return 0.0f;
  }
  const float coordinate = __fdividef(radius - lower, upper - lower);
  const float square = coordinate * coordinate;
  return __fdividef(
      -30.0f * square * square + 60.0f * square * coordinate - 30.0f * square,
      upper - lower);
}

struct TableLocation {
  int index;
  float coordinate;
  float extrapolation;
};

__device__ __forceinline__ int high_tail_index(
    float lower, float upper, float table_max, float stride0, float stride1) {
  const float boundary = nextafterf(table_max, lower);
  const int first_stride = static_cast<int>(__fdividef(upper - lower, stride0));
  return first_stride + static_cast<int>(__fdividef(boundary - upper, stride1));
}

__device__ __forceinline__ TableLocation locate_table(float radial,
                                                      float lower,
                                                      float upper,
                                                      float table_max,
                                                      float stride0,
                                                      float stride1) {
  TableLocation location;
  location.coordinate = radial;
  location.extrapolation = 0.0f;
  if (radial < lower) {
    location.index = 0;
    location.coordinate = 0.0f;
    location.extrapolation = radial - lower;
  } else if (radial < upper) {
    location.index = static_cast<int>(__fdividef(radial - lower, stride0));
    location.coordinate -= location.index * stride0 + lower;
  } else if (radial < table_max) {
    const int first_stride =
        static_cast<int>(__fdividef(upper - lower, stride0));
    location.index =
        first_stride + static_cast<int>(__fdividef(radial - upper, stride1));
    location.coordinate -= (location.index - first_stride) * stride1 + upper;
  } else {
    const int first_stride =
        static_cast<int>(__fdividef(upper - lower, stride0));
    location.index = high_tail_index(lower, upper, table_max, stride0, stride1);
    location.coordinate =
        table_max - ((location.index - first_stride) * stride1 + upper);
    location.extrapolation = radial - table_max;
  }
  return location;
}

__device__ __forceinline__ void load_coefficients(const float* table,
                                                  const TableLocation& location,
                                                  int channel,
                                                  int width,
                                                  float2& c01,
                                                  float2& c23,
                                                  float2& c45) {
  const long offset = static_cast<long>(location.index) * width * 6 +
                      static_cast<long>(channel) * 6;
  c01 = __ldg(reinterpret_cast<const float2*>(table + offset));
  c23 = __ldg(reinterpret_cast<const float2*>(table + offset + 2));
  c45 = __ldg(reinterpret_cast<const float2*>(table + offset + 4));
}

__device__ __forceinline__ float evaluate_table_forward(
    const float* table, const TableLocation& location, int channel, int width) {
  float2 c01, c23, c45;
  load_coefficients(table, location, channel, width, c01, c23, c45);
  const float value =
      c01.x + (c01.y + (c23.x + (c23.y + (c45.x + c45.y * location.coordinate) *
                                             location.coordinate) *
                                    location.coordinate) *
                           location.coordinate) *
                  location.coordinate;
  if (location.extrapolation == 0.0f) {
    return value;
  }
  const float derivative =
      c01.y +
      (2.0f * c23.x +
       (3.0f * c23.y + (4.0f * c45.x + 5.0f * c45.y * location.coordinate) *
                           location.coordinate) *
           location.coordinate) *
          location.coordinate;
  return value + derivative * location.extrapolation;
}

__device__ __forceinline__ float2 evaluate_table_backward(
    const float* table, const TableLocation& location, int channel, int width) {
  float2 c01, c23, c45;
  load_coefficients(table, location, channel, width, c01, c23, c45);
  float value = c45.y;
  float derivative = 0.0f;
  derivative = fmaf(derivative, location.coordinate, value);
  value = fmaf(value, location.coordinate, c45.x);
  derivative = fmaf(derivative, location.coordinate, value);
  value = fmaf(value, location.coordinate, c23.y);
  derivative = fmaf(derivative, location.coordinate, value);
  value = fmaf(value, location.coordinate, c23.x);
  derivative = fmaf(derivative, location.coordinate, value);
  value = fmaf(value, location.coordinate, c01.y);
  derivative = fmaf(derivative, location.coordinate, value);
  value = fmaf(value, location.coordinate, c01.x);
  return make_float2(value + derivative * location.extrapolation, derivative);
}

struct EdgeEnvironment {
  float radial;
  float r0;
  float r1;
  float r2;
  float r3;
  float switch_factor;
  float x;
  float y;
  float z;
  float radius;
  int pair_index;
};

template <typename index_t>
__device__ __forceinline__ EdgeEnvironment
load_environment(long edge,
                 int center_type,
                 int ntypes,
                 bool one_side,
                 float rcut,
                 float rcut_smooth,
                 float protection,
                 float inverse_neighbors,
                 const float* edge_vec,
                 const index_t* edge_index,
                 const long* atype,
                 const float* average,
                 const float* inverse_stddev) {
  EdgeEnvironment environment;
  const long source = static_cast<long>(edge_index[edge]);
  const int neighbor_type = static_cast<int>(atype[source]);
  environment.x = edge_vec[edge * 3 + 0];
  environment.y = edge_vec[edge * 3 + 1];
  environment.z = edge_vec[edge * 3 + 2];
  const float square_length = environment.x * environment.x +
                              environment.y * environment.y +
                              environment.z * environment.z;
  environment.radius =
      square_length > 0.0f ? square_length * rsqrtf(square_length) : 0.0f;
  const float denominator = environment.radius + protection;
  environment.switch_factor =
      switch_value(environment.radius, rcut_smooth, rcut);
  const float inverse_radius = __fdividef(1.0f, denominator);
  const float radial_scale =
      environment.switch_factor * inverse_radius * inverse_radius;
  const float* center_average = average + static_cast<long>(center_type) * 4;
  const float* center_inverse_stddev =
      inverse_stddev + static_cast<long>(center_type) * 4;
  environment.radial =
      (environment.switch_factor * inverse_radius - center_average[0]) *
      center_inverse_stddev[0];
  environment.r0 = environment.radial * inverse_neighbors;
  environment.r1 = (environment.x * radial_scale - center_average[1]) *
                   center_inverse_stddev[1] * inverse_neighbors;
  environment.r2 = (environment.y * radial_scale - center_average[2]) *
                   center_inverse_stddev[2] * inverse_neighbors;
  environment.r3 = (environment.z * radial_scale - center_average[3]) *
                   center_inverse_stddev[3] * inverse_neighbors;
  environment.pair_index =
      one_side ? neighbor_type : center_type * ntypes + neighbor_type;
  return environment;
}

__device__ __forceinline__ EdgeEnvironment
broadcast_environment(EdgeEnvironment value, int source_lane, unsigned mask) {
  value.radial = __shfl_sync(mask, value.radial, source_lane);
  value.r0 = __shfl_sync(mask, value.r0, source_lane);
  value.r1 = __shfl_sync(mask, value.r1, source_lane);
  value.r2 = __shfl_sync(mask, value.r2, source_lane);
  value.r3 = __shfl_sync(mask, value.r3, source_lane);
  value.switch_factor = __shfl_sync(mask, value.switch_factor, source_lane);
  value.x = __shfl_sync(mask, value.x, source_lane);
  value.y = __shfl_sync(mask, value.y, source_lane);
  value.z = __shfl_sync(mask, value.z, source_lane);
  value.radius = __shfl_sync(mask, value.radius, source_lane);
  value.pair_index = __shfl_sync(mask, value.pair_index, source_lane);
  return value;
}

__device__ __forceinline__ TableLocation broadcast_location(TableLocation value,
                                                            int source_lane,
                                                            unsigned mask) {
  value.index = __shfl_sync(mask, value.index, source_lane);
  value.coordinate = __shfl_sync(mask, value.coordinate, source_lane);
  value.extrapolation = __shfl_sync(mask, value.extrapolation, source_lane);
  return value;
}

__device__ __forceinline__ void store_edge_gradient(
    long edge,
    const EdgeEnvironment& environment,
    float partial0,
    float partial1,
    float partial2,
    float partial3,
    float partial_radial,
    float partial_switch,
    float inverse_neighbors,
    float inverse_stddev0,
    float inverse_stddev1,
    float inverse_stddev2,
    float inverse_stddev3,
    float rcut,
    float rcut_smooth,
    float protection,
    float* edge_gradient) {
  const float inverse_denominator =
      __fdividef(1.0f, environment.radius + protection);
  const float inverse_length =
      environment.radius > 0.0f ? __fdividef(1.0f, environment.radius) : 0.0f;
  const float switch_gradient =
      switch_derivative(environment.radius, rcut_smooth, rcut);
  const float gradient_radial =
      (partial0 * inverse_neighbors + partial_radial) * inverse_stddev0;
  const float gradient_x = partial1 * inverse_neighbors * inverse_stddev1;
  const float gradient_y = partial2 * inverse_neighbors * inverse_stddev2;
  const float gradient_z = partial3 * inverse_neighbors * inverse_stddev3;
  const float directional = gradient_x * environment.x +
                            gradient_y * environment.y +
                            gradient_z * environment.z;
  const float coefficient =
      (gradient_radial * inverse_denominator *
           (switch_gradient - environment.switch_factor * inverse_denominator) +
       directional * inverse_denominator * inverse_denominator *
           (switch_gradient -
            2.0f * environment.switch_factor * inverse_denominator) +
       partial_switch * switch_gradient) *
      inverse_length;
  const float vector_scale =
      environment.switch_factor * inverse_denominator * inverse_denominator;
  edge_gradient[edge * 3 + 0] =
      coefficient * environment.x + vector_scale * gradient_x;
  edge_gradient[edge * 3 + 1] =
      coefficient * environment.y + vector_scale * gradient_y;
  edge_gradient[edge * 3 + 2] =
      coefficient * environment.z + vector_scale * gradient_z;
}

template <int Width>
struct ChannelPolicy {
  static constexpr bool use_half_warp = Width >= 16 && Width <= 64;
  static constexpr int accumulation_groups =
      use_half_warp ? Width / 16 : (Width + 31) / 32;
  static constexpr int gradient_groups = (Width + 31) / 32;
};

template <bool Canonical, typename index_t>
__device__ __forceinline__ long edge_at_csr_position(
    long position, const index_t* destination_order) {
  if constexpr (Canonical) {
    return position;
  }
  return static_cast<long>(destination_order[position]);
}

__device__ __forceinline__ bool edge_is_active(long edge,
                                               const bool* edge_mask) {
  return edge_mask[edge];
}

template <int Width, bool Canonical, typename index_t, int MinimumBlocks>
__global__
__launch_bounds__(kThreads, MinimumBlocks) void compressed_forward_kernel(
    long node_count,
    int ntypes,
    bool one_side,
    bool smooth,
    int axis,
    bool concatenate_type_embedding,
    bool write_rotation,
    int type_embedding_dim,
    float rcut,
    float rcut_smooth,
    float protection,
    float inverse_neighbors,
    float lower,
    float upper,
    float table_max,
    float stride0,
    float stride1,
    const float* __restrict__ edge_vec,
    const index_t* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const index_t* __restrict__ destination_order,
    const long* __restrict__ destination_row_ptr,
    const long* __restrict__ atype,
    const float* __restrict__ type_embedding,
    const float* __restrict__ average,
    const float* __restrict__ inverse_stddev,
    const float* __restrict__ table,
    const float* __restrict__ gate_table,
    float* __restrict__ descriptor,
    float* __restrict__ rotation,
    float* __restrict__ moment) {
  constexpr unsigned kWarpMask = 0xffffffffu;
  constexpr int kGroups = ChannelPolicy<Width>::accumulation_groups;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x / kWarpSize;
  const long node = static_cast<long>(blockIdx.x) * warps_per_block + warp;
  if (node >= node_count) {
    return;
  }

  float accumulator0[kGroups] = {};
  float accumulator1[kGroups] = {};
  float accumulator2[kGroups] = {};
  float accumulator3[kGroups] = {};
  int center_type = lane == 0 ? static_cast<int>(atype[node]) : 0;
  center_type = __shfl_sync(kWarpMask, center_type, 0);
  const long begin = destination_row_ptr[node];
  const long end = destination_row_ptr[node + 1];

  if constexpr (ChannelPolicy<Width>::use_half_warp) {
    const int half = lane >> 4;
    const int half_lane = lane & 15;
    const int leader = half * 16;
    const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
    for (long position = begin + half; position < end; position += 2) {
      const long edge =
          edge_at_csr_position<Canonical>(position, destination_order);
      if (!edge_is_active(edge, edge_mask)) {
        continue;
      }
      EdgeEnvironment environment{};
      TableLocation location{};
      if (half_lane == 0) {
        environment = load_environment(edge, center_type, ntypes, one_side,
                                       rcut, rcut_smooth, protection,
                                       inverse_neighbors, edge_vec, edge_index,
                                       atype, average, inverse_stddev);
        location = locate_table(environment.radial, lower, upper, table_max,
                                stride0, stride1);
      }
      environment = broadcast_environment(environment, leader, half_mask);
      location = broadcast_location(location, leader, half_mask);
#pragma unroll
      for (int group = 0; group < kGroups; ++group) {
        const int channel = group * 16 + half_lane;
        const float table_value =
            evaluate_table_forward(table, location, channel, Width);
        const float gate =
            __ldg(gate_table +
                  static_cast<long>(environment.pair_index) * Width + channel);
        const float effective_gate =
            smooth ? gate * environment.switch_factor : gate;
        const float embedding = table_value * (1.0f + effective_gate);
        accumulator0[group] =
            fmaf(environment.r0, embedding, accumulator0[group]);
        accumulator1[group] =
            fmaf(environment.r1, embedding, accumulator1[group]);
        accumulator2[group] =
            fmaf(environment.r2, embedding, accumulator2[group]);
        accumulator3[group] =
            fmaf(environment.r3, embedding, accumulator3[group]);
      }
    }
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
      accumulator0[group] +=
          __shfl_xor_sync(kWarpMask, accumulator0[group], 16);
      accumulator1[group] +=
          __shfl_xor_sync(kWarpMask, accumulator1[group], 16);
      accumulator2[group] +=
          __shfl_xor_sync(kWarpMask, accumulator2[group], 16);
      accumulator3[group] +=
          __shfl_xor_sync(kWarpMask, accumulator3[group], 16);
    }
  } else {
    for (long position = begin; position < end; ++position) {
      const long edge =
          edge_at_csr_position<Canonical>(position, destination_order);
      if (!edge_is_active(edge, edge_mask)) {
        continue;
      }
      EdgeEnvironment environment{};
      TableLocation location{};
      if (lane == 0) {
        environment = load_environment(edge, center_type, ntypes, one_side,
                                       rcut, rcut_smooth, protection,
                                       inverse_neighbors, edge_vec, edge_index,
                                       atype, average, inverse_stddev);
        location = locate_table(environment.radial, lower, upper, table_max,
                                stride0, stride1);
      }
      environment = broadcast_environment(environment, 0, kWarpMask);
      location = broadcast_location(location, 0, kWarpMask);
#pragma unroll
      for (int group = 0; group < kGroups; ++group) {
        const int channel = group * 32 + lane;
        if (channel < Width) {
          const float table_value =
              evaluate_table_forward(table, location, channel, Width);
          const float gate = __ldg(
              gate_table + static_cast<long>(environment.pair_index) * Width +
              channel);
          const float effective_gate =
              smooth ? gate * environment.switch_factor : gate;
          const float embedding = table_value * (1.0f + effective_gate);
          accumulator0[group] =
              fmaf(environment.r0, embedding, accumulator0[group]);
          accumulator1[group] =
              fmaf(environment.r1, embedding, accumulator1[group]);
          accumulator2[group] =
              fmaf(environment.r2, embedding, accumulator2[group]);
          accumulator3[group] =
              fmaf(environment.r3, embedding, accumulator3[group]);
        }
      }
    }
  }

  const long moment_base = node * 4 * Width;
  if constexpr (ChannelPolicy<Width>::use_half_warp) {
    if (lane < 16) {
#pragma unroll
      for (int group = 0; group < kGroups; ++group) {
        const int channel = group * 16 + lane;
        moment[moment_base + 0 * Width + channel] = accumulator0[group];
        moment[moment_base + 1 * Width + channel] = accumulator1[group];
        moment[moment_base + 2 * Width + channel] = accumulator2[group];
        moment[moment_base + 3 * Width + channel] = accumulator3[group];
      }
    }
  } else {
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
      const int channel = group * 32 + lane;
      if (channel < Width) {
        moment[moment_base + 0 * Width + channel] = accumulator0[group];
        moment[moment_base + 1 * Width + channel] = accumulator1[group];
        moment[moment_base + 2 * Width + channel] = accumulator2[group];
        moment[moment_base + 3 * Width + channel] = accumulator3[group];
      }
    }
  }

  const int output_dim =
      Width * axis + (concatenate_type_embedding ? type_embedding_dim : 0);
  float* output = descriptor + node * output_dim;
  if constexpr (ChannelPolicy<Width>::use_half_warp) {
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
      const int channel = group * 16 + (lane & 15);
      for (int axis_channel = 0; axis_channel < axis; ++axis_channel) {
        const float axis0 =
            __shfl_sync(kWarpMask, accumulator0[0], axis_channel);
        const float axis1 =
            __shfl_sync(kWarpMask, accumulator1[0], axis_channel);
        const float axis2 =
            __shfl_sync(kWarpMask, accumulator2[0], axis_channel);
        const float axis3 =
            __shfl_sync(kWarpMask, accumulator3[0], axis_channel);
        if (lane < 16) {
          output[channel * axis + axis_channel] =
              accumulator0[group] * axis0 + accumulator1[group] * axis1 +
              accumulator2[group] * axis2 + accumulator3[group] * axis3;
        }
      }
      if (write_rotation && lane < 16) {
        float* rotation_row = rotation + (node * Width + channel) * 3;
        rotation_row[0] = accumulator1[group];
        rotation_row[1] = accumulator2[group];
        rotation_row[2] = accumulator3[group];
      }
    }
  } else {
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
      const int channel = group * 32 + lane;
      for (int axis_channel = 0; axis_channel < axis; ++axis_channel) {
        const float axis0 =
            __shfl_sync(kWarpMask, accumulator0[0], axis_channel);
        const float axis1 =
            __shfl_sync(kWarpMask, accumulator1[0], axis_channel);
        const float axis2 =
            __shfl_sync(kWarpMask, accumulator2[0], axis_channel);
        const float axis3 =
            __shfl_sync(kWarpMask, accumulator3[0], axis_channel);
        if (channel < Width) {
          output[channel * axis + axis_channel] =
              accumulator0[group] * axis0 + accumulator1[group] * axis1 +
              accumulator2[group] * axis2 + accumulator3[group] * axis3;
        }
      }
      if (write_rotation && channel < Width) {
        float* rotation_row = rotation + (node * Width + channel) * 3;
        rotation_row[0] = accumulator1[group];
        rotation_row[1] = accumulator2[group];
        rotation_row[2] = accumulator3[group];
      }
    }
  }
  if (concatenate_type_embedding) {
    for (int channel = lane; channel < type_embedding_dim; channel += 32) {
      output[Width * axis + channel] =
          type_embedding[static_cast<long>(center_type) * type_embedding_dim +
                         channel];
    }
  }
}

template <int Width, bool Canonical, typename index_t, int MinimumBlocks>
__global__
__launch_bounds__(kThreads, MinimumBlocks) void compressed_backward_kernel(
    long node_count,
    long edge_count,
    int ntypes,
    bool one_side,
    bool smooth,
    int axis,
    int descriptor_stride,
    float rcut,
    float rcut_smooth,
    float protection,
    float inverse_neighbors,
    float lower,
    float upper,
    float table_max,
    float stride0,
    float stride1,
    const float* __restrict__ descriptor_gradient,
    const float* __restrict__ rotation_gradient,
    const float* __restrict__ moment,
    const float* __restrict__ edge_vec,
    const index_t* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const index_t* __restrict__ destination_order,
    const long* __restrict__ destination_row_ptr,
    const long* __restrict__ atype,
    const float* __restrict__ average,
    const float* __restrict__ inverse_stddev,
    const float* __restrict__ table,
    const float* __restrict__ gate_table,
    float* __restrict__ edge_gradient) {
  constexpr unsigned kWarpMask = 0xffffffffu;
  constexpr int kGradientGroups = ChannelPolicy<Width>::gradient_groups;
  constexpr int kEdgeGroups = ChannelPolicy<Width>::accumulation_groups;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x / kWarpSize;
  const long node = static_cast<long>(blockIdx.x) * warps_per_block + warp;
  if (node >= node_count) {
    return;
  }

  float gradient0[kGradientGroups] = {};
  float gradient1[kGradientGroups] = {};
  float gradient2[kGradientGroups] = {};
  float gradient3[kGradientGroups] = {};
  float own_moment0[kGradientGroups] = {};
  float own_moment1[kGradientGroups] = {};
  float own_moment2[kGradientGroups] = {};
  float own_moment3[kGradientGroups] = {};
  const long moment_base = node * 4 * Width;
  const float* node_descriptor_gradient =
      descriptor_gradient + node * descriptor_stride;

#pragma unroll
  for (int group = 0; group < kGradientGroups; ++group) {
    const int channel = group * 32 + lane;
    if (channel < Width) {
      own_moment0[group] = __ldg(moment + moment_base + 0 * Width + channel);
      own_moment1[group] = __ldg(moment + moment_base + 1 * Width + channel);
      own_moment2[group] = __ldg(moment + moment_base + 2 * Width + channel);
      own_moment3[group] = __ldg(moment + moment_base + 3 * Width + channel);
    }
    for (int axis_channel = 0; axis_channel < axis; ++axis_channel) {
      const float axis0 = __shfl_sync(kWarpMask, own_moment0[0], axis_channel);
      const float axis1 = __shfl_sync(kWarpMask, own_moment1[0], axis_channel);
      const float axis2 = __shfl_sync(kWarpMask, own_moment2[0], axis_channel);
      const float axis3 = __shfl_sync(kWarpMask, own_moment3[0], axis_channel);
      if (channel < Width) {
        const float value =
            __ldg(node_descriptor_gradient + channel * axis + axis_channel);
        gradient0[group] = fmaf(value, axis0, gradient0[group]);
        gradient1[group] = fmaf(value, axis1, gradient1[group]);
        gradient2[group] = fmaf(value, axis2, gradient2[group]);
        gradient3[group] = fmaf(value, axis3, gradient3[group]);
      }
    }
    if (channel < axis) {
      for (int input = 0; input < Width; ++input) {
        const float value =
            __ldg(node_descriptor_gradient + input * axis + channel);
        gradient0[group] =
            fmaf(value, __ldg(moment + moment_base + 0 * Width + input),
                 gradient0[group]);
        gradient1[group] =
            fmaf(value, __ldg(moment + moment_base + 1 * Width + input),
                 gradient1[group]);
        gradient2[group] =
            fmaf(value, __ldg(moment + moment_base + 2 * Width + input),
                 gradient2[group]);
        gradient3[group] =
            fmaf(value, __ldg(moment + moment_base + 3 * Width + input),
                 gradient3[group]);
      }
    }
    if (rotation_gradient != nullptr && channel < Width) {
      const long rotation_offset = (node * Width + channel) * 3;
      gradient1[group] += __ldg(rotation_gradient + rotation_offset + 0);
      gradient2[group] += __ldg(rotation_gradient + rotation_offset + 1);
      gradient3[group] += __ldg(rotation_gradient + rotation_offset + 2);
    }
  }

  int center_type = lane == 0 ? static_cast<int>(atype[node]) : 0;
  center_type = __shfl_sync(kWarpMask, center_type, 0);
  const float inverse_stddev0 =
      __ldg(inverse_stddev + static_cast<long>(center_type) * 4 + 0);
  const float inverse_stddev1 =
      __ldg(inverse_stddev + static_cast<long>(center_type) * 4 + 1);
  const float inverse_stddev2 =
      __ldg(inverse_stddev + static_cast<long>(center_type) * 4 + 2);
  const float inverse_stddev3 =
      __ldg(inverse_stddev + static_cast<long>(center_type) * 4 + 3);
  const long begin = destination_row_ptr[node];
  const long end = destination_row_ptr[node + 1];

  if constexpr (ChannelPolicy<Width>::use_half_warp) {
    const int half = lane >> 4;
    const int half_lane = lane & 15;
    const int leader = half * 16;
    const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
    float edge_gradient0[kEdgeGroups] = {};
    float edge_gradient1[kEdgeGroups] = {};
    float edge_gradient2[kEdgeGroups] = {};
    float edge_gradient3[kEdgeGroups] = {};
#pragma unroll
    for (int group = 0; group < kEdgeGroups; ++group) {
      const int channel = group * 16 + half_lane;
      const int owner_group = channel / 32;
      const int owner_lane = channel & 31;
      edge_gradient0[group] =
          __shfl_sync(kWarpMask, gradient0[owner_group], owner_lane);
      edge_gradient1[group] =
          __shfl_sync(kWarpMask, gradient1[owner_group], owner_lane);
      edge_gradient2[group] =
          __shfl_sync(kWarpMask, gradient2[owner_group], owner_lane);
      edge_gradient3[group] =
          __shfl_sync(kWarpMask, gradient3[owner_group], owner_lane);
    }

    for (long position = begin + half; position < end; position += 2) {
      const long edge =
          edge_at_csr_position<Canonical>(position, destination_order);
      if (!edge_is_active(edge, edge_mask)) {
        if (half_lane == 0) {
          edge_gradient[edge * 3 + 0] = 0.0f;
          edge_gradient[edge * 3 + 1] = 0.0f;
          edge_gradient[edge * 3 + 2] = 0.0f;
        }
        continue;
      }
      EdgeEnvironment environment{};
      TableLocation location{};
      if (half_lane == 0) {
        environment = load_environment(edge, center_type, ntypes, one_side,
                                       rcut, rcut_smooth, protection,
                                       inverse_neighbors, edge_vec, edge_index,
                                       atype, average, inverse_stddev);
        location = locate_table(environment.radial, lower, upper, table_max,
                                stride0, stride1);
      }
      environment = broadcast_environment(environment, leader, half_mask);
      location = broadcast_location(location, leader, half_mask);
      float partial0 = 0.0f;
      float partial1 = 0.0f;
      float partial2 = 0.0f;
      float partial3 = 0.0f;
      float partial_radial = 0.0f;
      float partial_switch = 0.0f;
#pragma unroll
      for (int group = 0; group < kEdgeGroups; ++group) {
        const int channel = group * 16 + half_lane;
        const float d0 = edge_gradient0[group];
        const float d1 = edge_gradient1[group];
        const float d2 = edge_gradient2[group];
        const float d3 = edge_gradient3[group];
        const float descriptor_product =
            environment.r0 * d0 + environment.r1 * d1 + environment.r2 * d2 +
            environment.r3 * d3;
        const float2 table_value =
            evaluate_table_backward(table, location, channel, Width);
        const float gate =
            __ldg(gate_table +
                  static_cast<long>(environment.pair_index) * Width + channel);
        const float effective_gate =
            smooth ? gate * environment.switch_factor : gate;
        const float embedding = table_value.x * (1.0f + effective_gate);
        if (smooth) {
          partial_switch =
              fmaf(descriptor_product * table_value.x, gate, partial_switch);
        }
        partial_radial = fmaf(descriptor_product * (1.0f + effective_gate),
                              table_value.y, partial_radial);
        partial0 = fmaf(embedding, d0, partial0);
        partial1 = fmaf(embedding, d1, partial1);
        partial2 = fmaf(embedding, d2, partial2);
        partial3 = fmaf(embedding, d3, partial3);
      }
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1) {
        partial0 += __shfl_down_sync(half_mask, partial0, offset, 16);
        partial1 += __shfl_down_sync(half_mask, partial1, offset, 16);
        partial2 += __shfl_down_sync(half_mask, partial2, offset, 16);
        partial3 += __shfl_down_sync(half_mask, partial3, offset, 16);
        partial_radial +=
            __shfl_down_sync(half_mask, partial_radial, offset, 16);
        partial_switch +=
            __shfl_down_sync(half_mask, partial_switch, offset, 16);
      }
      if (half_lane == 0) {
        store_edge_gradient(edge, environment, partial0, partial1, partial2,
                            partial3, partial_radial, partial_switch,
                            inverse_neighbors, inverse_stddev0, inverse_stddev1,
                            inverse_stddev2, inverse_stddev3, rcut, rcut_smooth,
                            protection, edge_gradient);
      }
    }
  } else {
    for (long position = begin; position < end; ++position) {
      const long edge =
          edge_at_csr_position<Canonical>(position, destination_order);
      if (!edge_is_active(edge, edge_mask)) {
        if (lane == 0) {
          edge_gradient[edge * 3 + 0] = 0.0f;
          edge_gradient[edge * 3 + 1] = 0.0f;
          edge_gradient[edge * 3 + 2] = 0.0f;
        }
        continue;
      }
      EdgeEnvironment environment{};
      TableLocation location{};
      if (lane == 0) {
        environment = load_environment(edge, center_type, ntypes, one_side,
                                       rcut, rcut_smooth, protection,
                                       inverse_neighbors, edge_vec, edge_index,
                                       atype, average, inverse_stddev);
        location = locate_table(environment.radial, lower, upper, table_max,
                                stride0, stride1);
      }
      environment = broadcast_environment(environment, 0, kWarpMask);
      location = broadcast_location(location, 0, kWarpMask);
      float partial0 = 0.0f;
      float partial1 = 0.0f;
      float partial2 = 0.0f;
      float partial3 = 0.0f;
      float partial_radial = 0.0f;
      float partial_switch = 0.0f;
#pragma unroll
      for (int group = 0; group < kGradientGroups; ++group) {
        const int channel = group * 32 + lane;
        if (channel < Width) {
          const float d0 = gradient0[group];
          const float d1 = gradient1[group];
          const float d2 = gradient2[group];
          const float d3 = gradient3[group];
          const float descriptor_product =
              environment.r0 * d0 + environment.r1 * d1 + environment.r2 * d2 +
              environment.r3 * d3;
          const float2 table_value =
              evaluate_table_backward(table, location, channel, Width);
          const float gate = __ldg(
              gate_table + static_cast<long>(environment.pair_index) * Width +
              channel);
          const float effective_gate =
              smooth ? gate * environment.switch_factor : gate;
          const float embedding = table_value.x * (1.0f + effective_gate);
          if (smooth) {
            partial_switch =
                fmaf(descriptor_product * table_value.x, gate, partial_switch);
          }
          partial_radial = fmaf(descriptor_product * (1.0f + effective_gate),
                                table_value.y, partial_radial);
          partial0 = fmaf(embedding, d0, partial0);
          partial1 = fmaf(embedding, d1, partial1);
          partial2 = fmaf(embedding, d2, partial2);
          partial3 = fmaf(embedding, d3, partial3);
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial0 += __shfl_down_sync(kWarpMask, partial0, offset);
        partial1 += __shfl_down_sync(kWarpMask, partial1, offset);
        partial2 += __shfl_down_sync(kWarpMask, partial2, offset);
        partial3 += __shfl_down_sync(kWarpMask, partial3, offset);
        partial_radial += __shfl_down_sync(kWarpMask, partial_radial, offset);
        partial_switch += __shfl_down_sync(kWarpMask, partial_switch, offset);
      }
      if (lane == 0) {
        store_edge_gradient(edge, environment, partial0, partial1, partial2,
                            partial3, partial_radial, partial_switch,
                            inverse_neighbors, inverse_stddev0, inverse_stddev1,
                            inverse_stddev2, inverse_stddev3, rcut, rcut_smooth,
                            protection, edge_gradient);
      }
    }
  }
}

template <bool Canonical, typename index_t>
__global__ void zero_padding_kernel(
    long node_count,
    long edge_count,
    const index_t* __restrict__ destination_order,
    const long* __restrict__ destination_row_ptr,
    float* __restrict__ edge_gradient) {
  const long valid_edge_count = destination_row_ptr[node_count];
  for (long position = valid_edge_count + blockIdx.x * blockDim.x + threadIdx.x;
       position < edge_count;
       position += static_cast<long>(blockDim.x) * gridDim.x) {
    const long edge =
        edge_at_csr_position<Canonical>(position, destination_order);
    edge_gradient[edge * 3 + 0] = 0.0f;
    edge_gradient[edge * 3 + 1] = 0.0f;
    edge_gradient[edge * 3 + 2] = 0.0f;
  }
}

template <int Width, typename index_t, bool Canonical, int MinimumBlocks>
void launch_forward_variant(long node_count,
                            int threads,
                            int ntypes,
                            bool one_side,
                            bool smooth,
                            int axis,
                            bool concatenate_type_embedding,
                            bool write_rotation,
                            int type_embedding_dim,
                            float rcut,
                            float rcut_smooth,
                            float protection,
                            float inverse_neighbors,
                            float lower,
                            float upper,
                            float table_max,
                            float stride0,
                            float stride1,
                            const torch::Tensor& edge_vec,
                            const torch::Tensor& edge_index,
                            const torch::Tensor& edge_mask,
                            const torch::Tensor& destination_order,
                            const torch::Tensor& destination_row_ptr,
                            const torch::Tensor& atype,
                            const torch::Tensor& type_embedding,
                            const torch::Tensor& average,
                            const torch::Tensor& inverse_stddev,
                            const torch::Tensor& table,
                            const torch::Tensor& gate_table,
                            torch::Tensor& descriptor,
                            torch::Tensor& rotation,
                            torch::Tensor& moment,
                            cudaStream_t stream) {
  const int warps_per_block = threads / kWarpSize;
  const int blocks =
      static_cast<int>((node_count + warps_per_block - 1) / warps_per_block);
  compressed_forward_kernel<Width, Canonical, index_t, MinimumBlocks>
      <<<blocks, threads, 0, stream>>>(
          node_count, ntypes, one_side, smooth, axis,
          concatenate_type_embedding, write_rotation, type_embedding_dim, rcut,
          rcut_smooth, protection, inverse_neighbors, lower, upper, table_max,
          stride0, stride1, edge_vec.data_ptr<float>(),
          edge_index.data_ptr<index_t>(), edge_mask.data_ptr<bool>(),
          destination_order.data_ptr<index_t>(),
          destination_row_ptr.data_ptr<long>(), atype.data_ptr<long>(),
          type_embedding.data_ptr<float>(), average.data_ptr<float>(),
          inverse_stddev.data_ptr<float>(), table.data_ptr<float>(),
          gate_table.data_ptr<float>(), descriptor.data_ptr<float>(),
          write_rotation ? rotation.data_ptr<float>() : nullptr,
          moment.data_ptr<float>());
}

template <int Width, typename index_t, bool Canonical>
void launch_forward(long node_count,
                    int ntypes,
                    bool one_side,
                    bool smooth,
                    int axis,
                    bool concatenate_type_embedding,
                    bool write_rotation,
                    int type_embedding_dim,
                    float rcut,
                    float rcut_smooth,
                    float protection,
                    float inverse_neighbors,
                    float lower,
                    float upper,
                    float table_max,
                    float stride0,
                    float stride1,
                    const torch::Tensor& edge_vec,
                    const torch::Tensor& edge_index,
                    const torch::Tensor& edge_mask,
                    const torch::Tensor& destination_order,
                    const torch::Tensor& destination_row_ptr,
                    const torch::Tensor& atype,
                    const torch::Tensor& type_embedding,
                    const torch::Tensor& average,
                    const torch::Tensor& inverse_stddev,
                    const torch::Tensor& table,
                    const torch::Tensor& gate_table,
                    torch::Tensor& descriptor,
                    torch::Tensor& rotation,
                    torch::Tensor& moment,
                    cudaStream_t stream) {
  const int device = edge_vec.get_device();
  const cudaDeviceProp& properties = device_properties(device);
  const TuningKey key = {
      device,
      static_cast<int>(KernelDirection::kForward),
      Width,
      axis,
      Canonical ? 1 : 0,
      static_cast<int>(sizeof(index_t)),
      (one_side ? 1 : 0) | (smooth ? 2 : 0) |
          (concatenate_type_embedding ? 4 : 0) | (write_rotation ? 8 : 0),
      concatenate_type_embedding ? type_embedding_dim : 0,
      type_count_class(ntypes),
      workload_size_class(node_count, properties.multiProcessorCount),
      workload_degree_class(node_count, edge_vec.size(0)),
  };
  const auto launch = [&](const LaunchConfig& config, long count) {
    if (config.resource == ResourcePolicy::kOccupancy) {
      launch_forward_variant<Width, index_t, Canonical, 4>(
          count, config.threads, ntypes, one_side, smooth, axis,
          concatenate_type_embedding, write_rotation, type_embedding_dim, rcut,
          rcut_smooth, protection, inverse_neighbors, lower, upper, table_max,
          stride0, stride1, edge_vec, edge_index, edge_mask, destination_order,
          destination_row_ptr, atype, type_embedding, average, inverse_stddev,
          table, gate_table, descriptor, rotation, moment, stream);
    } else {
      launch_forward_variant<Width, index_t, Canonical, 2>(
          count, config.threads, ntypes, one_side, smooth, axis,
          concatenate_type_embedding, write_rotation, type_embedding_dim, rcut,
          rcut_smooth, protection, inverse_neighbors, lower, upper, table_max,
          stride0, stride1, edge_vec, edge_index, edge_mask, destination_order,
          destination_row_ptr, atype, type_embedding, average, inverse_stddev,
          table, gate_table, descriptor, rotation, moment, stream);
    }
  };
  const LaunchConfig config =
      select_launch_config(key, properties, node_count, stream, launch);
  launch(config, node_count);
  COMPRESS_CHECK_LAUNCH("dpa1_graph_compress forward");
}

template <int Width, typename index_t, bool Canonical, int MinimumBlocks>
void launch_backward_variant(long node_count,
                             int threads,
                             long edge_count,
                             int ntypes,
                             bool one_side,
                             bool smooth,
                             int axis,
                             int descriptor_stride,
                             float rcut,
                             float rcut_smooth,
                             float protection,
                             float inverse_neighbors,
                             float lower,
                             float upper,
                             float table_max,
                             float stride0,
                             float stride1,
                             const torch::Tensor& descriptor_gradient,
                             const float* rotation_gradient,
                             const torch::Tensor& moment,
                             const torch::Tensor& edge_vec,
                             const torch::Tensor& edge_index,
                             const torch::Tensor& edge_mask,
                             const torch::Tensor& destination_order,
                             const torch::Tensor& destination_row_ptr,
                             const torch::Tensor& atype,
                             const torch::Tensor& average,
                             const torch::Tensor& inverse_stddev,
                             const torch::Tensor& table,
                             const torch::Tensor& gate_table,
                             torch::Tensor& edge_gradient,
                             cudaStream_t stream) {
  const int warps_per_block = threads / kWarpSize;
  const int blocks =
      static_cast<int>((node_count + warps_per_block - 1) / warps_per_block);
  compressed_backward_kernel<Width, Canonical, index_t, MinimumBlocks>
      <<<blocks, threads, 0, stream>>>(
          node_count, edge_count, ntypes, one_side, smooth, axis,
          descriptor_stride, rcut, rcut_smooth, protection, inverse_neighbors,
          lower, upper, table_max, stride0, stride1,
          descriptor_gradient.data_ptr<float>(), rotation_gradient,
          moment.data_ptr<float>(), edge_vec.data_ptr<float>(),
          edge_index.data_ptr<index_t>(), edge_mask.data_ptr<bool>(),
          destination_order.data_ptr<index_t>(),
          destination_row_ptr.data_ptr<long>(), atype.data_ptr<long>(),
          average.data_ptr<float>(), inverse_stddev.data_ptr<float>(),
          table.data_ptr<float>(), gate_table.data_ptr<float>(),
          edge_gradient.data_ptr<float>());
}

template <int Width, typename index_t, bool Canonical>
void launch_backward(long node_count,
                     long edge_count,
                     int ntypes,
                     bool one_side,
                     bool smooth,
                     int axis,
                     int descriptor_stride,
                     float rcut,
                     float rcut_smooth,
                     float protection,
                     float inverse_neighbors,
                     float lower,
                     float upper,
                     float table_max,
                     float stride0,
                     float stride1,
                     const torch::Tensor& descriptor_gradient,
                     const float* rotation_gradient,
                     const torch::Tensor& moment,
                     const torch::Tensor& edge_vec,
                     const torch::Tensor& edge_index,
                     const torch::Tensor& edge_mask,
                     const torch::Tensor& destination_order,
                     const torch::Tensor& destination_row_ptr,
                     const torch::Tensor& atype,
                     const torch::Tensor& average,
                     const torch::Tensor& inverse_stddev,
                     const torch::Tensor& table,
                     const torch::Tensor& gate_table,
                     torch::Tensor& edge_gradient,
                     cudaStream_t stream) {
  const int device = edge_vec.get_device();
  const cudaDeviceProp& properties = device_properties(device);
  const TuningKey key = {
      device,
      static_cast<int>(KernelDirection::kBackward),
      Width,
      axis,
      Canonical ? 1 : 0,
      static_cast<int>(sizeof(index_t)),
      (one_side ? 1 : 0) | (smooth ? 2 : 0) |
          (rotation_gradient != nullptr ? 8 : 0),
      descriptor_stride,
      type_count_class(ntypes),
      workload_size_class(node_count, properties.multiProcessorCount),
      workload_degree_class(node_count, edge_count),
  };
  const auto launch = [&](const LaunchConfig& config, long count) {
    if (config.resource == ResourcePolicy::kOccupancy) {
      launch_backward_variant<Width, index_t, Canonical, 4>(
          count, config.threads, edge_count, ntypes, one_side, smooth, axis,
          descriptor_stride, rcut, rcut_smooth, protection, inverse_neighbors,
          lower, upper, table_max, stride0, stride1, descriptor_gradient,
          rotation_gradient, moment, edge_vec, edge_index, edge_mask,
          destination_order, destination_row_ptr, atype, average,
          inverse_stddev, table, gate_table, edge_gradient, stream);
    } else {
      launch_backward_variant<Width, index_t, Canonical, 2>(
          count, config.threads, edge_count, ntypes, one_side, smooth, axis,
          descriptor_stride, rcut, rcut_smooth, protection, inverse_neighbors,
          lower, upper, table_max, stride0, stride1, descriptor_gradient,
          rotation_gradient, moment, edge_vec, edge_index, edge_mask,
          destination_order, destination_row_ptr, atype, average,
          inverse_stddev, table, gate_table, edge_gradient, stream);
    }
  };
  const LaunchConfig config =
      select_launch_config(key, properties, node_count, stream, launch);
  launch(config, node_count);
  COMPRESS_CHECK_LAUNCH("dpa1_graph_compress backward");
  zero_padding_kernel<Canonical, index_t><<<1, kThreads, 0, stream>>>(
      node_count, edge_count, destination_order.data_ptr<index_t>(),
      destination_row_ptr.data_ptr<long>(), edge_gradient.data_ptr<float>());
  COMPRESS_CHECK_LAUNCH("dpa1_graph_compress padding");
}

template <typename index_t>
void dispatch_forward(int width,
                      long node_count,
                      int ntypes,
                      bool one_side,
                      bool smooth,
                      int axis,
                      bool canonical,
                      bool concatenate_type_embedding,
                      bool write_rotation,
                      int type_embedding_dim,
                      float rcut,
                      float rcut_smooth,
                      float protection,
                      float inverse_neighbors,
                      float lower,
                      float upper,
                      float table_max,
                      float stride0,
                      float stride1,
                      const torch::Tensor& edge_vec,
                      const torch::Tensor& edge_index,
                      const torch::Tensor& edge_mask,
                      const torch::Tensor& destination_order,
                      const torch::Tensor& destination_row_ptr,
                      const torch::Tensor& atype,
                      const torch::Tensor& type_embedding,
                      const torch::Tensor& average,
                      const torch::Tensor& inverse_stddev,
                      const torch::Tensor& table,
                      const torch::Tensor& gate_table,
                      torch::Tensor& descriptor,
                      torch::Tensor& rotation,
                      torch::Tensor& moment,
                      cudaStream_t stream) {
#define DISPATCH_WIDTH(value)                                               \
  if (width == value) {                                                     \
    if (canonical) {                                                        \
      launch_forward<value, index_t, true>(                                 \
          node_count, ntypes, one_side, smooth, axis,                       \
          concatenate_type_embedding, write_rotation, type_embedding_dim,   \
          rcut, rcut_smooth, protection, inverse_neighbors, lower, upper,   \
          table_max, stride0, stride1, edge_vec, edge_index, edge_mask,     \
          destination_order, destination_row_ptr, atype, type_embedding,    \
          average, inverse_stddev, table, gate_table, descriptor, rotation, \
          moment, stream);                                                  \
    } else {                                                                \
      launch_forward<value, index_t, false>(                                \
          node_count, ntypes, one_side, smooth, axis,                       \
          concatenate_type_embedding, write_rotation, type_embedding_dim,   \
          rcut, rcut_smooth, protection, inverse_neighbors, lower, upper,   \
          table_max, stride0, stride1, edge_vec, edge_index, edge_mask,     \
          destination_order, destination_row_ptr, atype, type_embedding,    \
          average, inverse_stddev, table, gate_table, descriptor, rotation, \
          moment, stream);                                                  \
    }                                                                       \
    return;                                                                 \
  }
  DISPATCH_WIDTH(8)
  DISPATCH_WIDTH(16)
  DISPATCH_WIDTH(32)
  DISPATCH_WIDTH(64)
  DISPATCH_WIDTH(128)
  DISPATCH_WIDTH(256)
#undef DISPATCH_WIDTH
  TORCH_CHECK(false, "dpa1_graph_compress: unsupported width ", width);
}

template <typename index_t>
void dispatch_backward(int width,
                       long node_count,
                       long edge_count,
                       int ntypes,
                       bool one_side,
                       bool smooth,
                       int axis,
                       bool canonical,
                       int descriptor_stride,
                       float rcut,
                       float rcut_smooth,
                       float protection,
                       float inverse_neighbors,
                       float lower,
                       float upper,
                       float table_max,
                       float stride0,
                       float stride1,
                       const torch::Tensor& descriptor_gradient,
                       const float* rotation_gradient,
                       const torch::Tensor& moment,
                       const torch::Tensor& edge_vec,
                       const torch::Tensor& edge_index,
                       const torch::Tensor& edge_mask,
                       const torch::Tensor& destination_order,
                       const torch::Tensor& destination_row_ptr,
                       const torch::Tensor& atype,
                       const torch::Tensor& average,
                       const torch::Tensor& inverse_stddev,
                       const torch::Tensor& table,
                       const torch::Tensor& gate_table,
                       torch::Tensor& edge_gradient,
                       cudaStream_t stream) {
#define DISPATCH_WIDTH(value)                                                  \
  if (width == value) {                                                        \
    if (canonical) {                                                           \
      launch_backward<value, index_t, true>(                                   \
          node_count, edge_count, ntypes, one_side, smooth, axis,              \
          descriptor_stride, rcut, rcut_smooth, protection, inverse_neighbors, \
          lower, upper, table_max, stride0, stride1, descriptor_gradient,      \
          rotation_gradient, moment, edge_vec, edge_index, edge_mask,          \
          destination_order, destination_row_ptr, atype, average,              \
          inverse_stddev, table, gate_table, edge_gradient, stream);           \
    } else {                                                                   \
      launch_backward<value, index_t, false>(                                  \
          node_count, edge_count, ntypes, one_side, smooth, axis,              \
          descriptor_stride, rcut, rcut_smooth, protection, inverse_neighbors, \
          lower, upper, table_max, stride0, stride1, descriptor_gradient,      \
          rotation_gradient, moment, edge_vec, edge_index, edge_mask,          \
          destination_order, destination_row_ptr, atype, average,              \
          inverse_stddev, table, gate_table, edge_gradient, stream);           \
    }                                                                          \
    return;                                                                    \
  }
  DISPATCH_WIDTH(8)
  DISPATCH_WIDTH(16)
  DISPATCH_WIDTH(32)
  DISPATCH_WIDTH(64)
  DISPATCH_WIDTH(128)
  DISPATCH_WIDTH(256)
#undef DISPATCH_WIDTH
  TORCH_CHECK(false, "dpa1_graph_compress_backward: unsupported width ", width);
}

void validate_inputs(const torch::Tensor& edge_vec,
                     const torch::Tensor& edge_index,
                     const torch::Tensor& edge_mask,
                     const torch::Tensor& destination_order,
                     const torch::Tensor& destination_row_ptr,
                     const torch::Tensor& atype,
                     const torch::Tensor& average,
                     const torch::Tensor& inverse_stddev,
                     const torch::Tensor& table,
                     const torch::Tensor& gate_table,
                     int width,
                     int axis) {
  TORCH_CHECK(edge_vec.is_cuda() && edge_index.is_cuda() &&
                  edge_mask.is_cuda() && destination_order.is_cuda() &&
                  destination_row_ptr.is_cuda() && atype.is_cuda() &&
                  average.is_cuda() && inverse_stddev.is_cuda() &&
                  table.is_cuda() && gate_table.is_cuda(),
              "dpa1_graph_compress: inputs must be CUDA tensors");
  TORCH_CHECK(
      edge_vec.is_contiguous() && edge_index.is_contiguous() &&
          edge_mask.is_contiguous() && destination_order.is_contiguous() &&
          destination_row_ptr.is_contiguous() && atype.is_contiguous() &&
          average.is_contiguous() && inverse_stddev.is_contiguous() &&
          table.is_contiguous() && gate_table.is_contiguous(),
      "dpa1_graph_compress: inputs must be contiguous");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt32 ||
                  edge_index.scalar_type() == torch::kInt64,
              "dpa1_graph_compress: edge_index must be int32 or int64");
  TORCH_CHECK(destination_order.scalar_type() == edge_index.scalar_type(),
              "dpa1_graph_compress: destination_order must match the "
              "edge_index dtype");
  TORCH_CHECK(edge_mask.scalar_type() == torch::kBool,
              "dpa1_graph_compress: edge_mask must be bool");
  TORCH_CHECK(atype.scalar_type() == torch::kInt64,
              "dpa1_graph_compress: atype must be int64");
  TORCH_CHECK(destination_row_ptr.scalar_type() == torch::kInt64,
              "dpa1_graph_compress: destination_row_ptr must be int64");
  TORCH_CHECK(average.scalar_type() == torch::kFloat32 &&
                  inverse_stddev.scalar_type() == torch::kFloat32 &&
                  table.scalar_type() == torch::kFloat32 &&
                  gate_table.scalar_type() == torch::kFloat32,
              "dpa1_graph_compress: statistics and tables must be fp32");
  TORCH_CHECK(axis > 0 && axis <= 16 && axis <= width,
              "dpa1_graph_compress: axis must be in [1, min(16, width)]");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dpa1_graph_compress(
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor type_embedding,
    torch::Tensor average,
    torch::Tensor inverse_stddev,
    torch::Tensor table,
    torch::Tensor gate_table,
    int64_t type_one_side,
    int64_t concatenate_type_embedding,
    int64_t write_rotation,
    int64_t smooth,
    int64_t axis,
    bool canonical,
    double lower,
    double upper,
    double table_max,
    double stride0,
    double stride1,
    double rcut,
    double rcut_smooth,
    double protection,
    double neighbors) {
  const long node_count = atype.size(0);
  const int width = static_cast<int>(table.size(1) / 6);
  validate_inputs(edge_vec, edge_index, edge_mask, destination_order,
                  destination_row_ptr, atype, average, inverse_stddev, table,
                  gate_table, width, static_cast<int>(axis));
  TORCH_CHECK(type_embedding.is_cuda() && type_embedding.is_contiguous() &&
                  type_embedding.scalar_type() == torch::kFloat32,
              "dpa1_graph_compress: type_embedding must be contiguous fp32 "
              "on CUDA");
  const int ntypes = static_cast<int>(type_embedding.size(0));
  const int type_embedding_dim = static_cast<int>(type_embedding.size(1));
  const int output_dim = width * static_cast<int>(axis) +
                         (concatenate_type_embedding ? type_embedding_dim : 0);
  auto options = edge_vec.options().dtype(torch::kFloat32);
  auto descriptor = torch::empty({node_count, output_dim}, options);
  auto rotation =
      torch::empty({write_rotation ? node_count : 0, width, 3}, options);
  auto moment = torch::empty({node_count, 4, width}, options);
  if (node_count == 0) {
    return {descriptor, rotation, moment};
  }
  const auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  const auto stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto index_tag) {
    using index_t = decltype(index_tag);
    dispatch_forward<index_t>(
        width, node_count, ntypes, type_one_side != 0, smooth != 0,
        static_cast<int>(axis), canonical, concatenate_type_embedding != 0,
        write_rotation != 0, type_embedding_dim, static_cast<float>(rcut),
        static_cast<float>(rcut_smooth), static_cast<float>(protection),
        static_cast<float>(1.0 / neighbors), static_cast<float>(lower),
        static_cast<float>(upper), static_cast<float>(table_max),
        static_cast<float>(stride0), static_cast<float>(stride1),
        edge_vec_float, edge_index, edge_mask, destination_order,
        destination_row_ptr, atype, type_embedding, average, inverse_stddev,
        table, gate_table, descriptor, rotation, moment, stream);
  };
  if (edge_index.scalar_type() == torch::kInt32) {
    launch(int{});
  } else {
    launch(long{});
  }
  return {descriptor, rotation, moment};
}

torch::Tensor dpa1_graph_compress_backward(
    torch::Tensor descriptor_gradient,
    std::optional<torch::Tensor> rotation_gradient,
    torch::Tensor moment,
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor average,
    torch::Tensor inverse_stddev,
    torch::Tensor table,
    torch::Tensor gate_table,
    int64_t type_one_side,
    int64_t smooth,
    int64_t axis,
    bool canonical,
    double lower,
    double upper,
    double table_max,
    double stride0,
    double stride1,
    double rcut,
    double rcut_smooth,
    double protection,
    double neighbors) {
  const long node_count = atype.size(0);
  const long edge_count = edge_vec.size(0);
  const int width = static_cast<int>(table.size(1) / 6);
  validate_inputs(edge_vec, edge_index, edge_mask, destination_order,
                  destination_row_ptr, atype, average, inverse_stddev, table,
                  gate_table, width, static_cast<int>(axis));
  if (node_count == 0) {
    return torch::zeros_like(edge_vec);
  }
  const int ntypes = type_one_side
                         ? static_cast<int>(gate_table.size(0))
                         : static_cast<int>(llround(
                               sqrt(static_cast<double>(gate_table.size(0)))));
  auto descriptor_gradient_float =
      descriptor_gradient.to(torch::kFloat32).contiguous();
  torch::Tensor rotation_gradient_float;
  const float* rotation_gradient_ptr = nullptr;
  if (rotation_gradient.has_value() && rotation_gradient->defined() &&
      rotation_gradient->numel() > 0) {
    rotation_gradient_float =
        rotation_gradient->to(torch::kFloat32).contiguous();
    rotation_gradient_ptr = rotation_gradient_float.data_ptr<float>();
  }
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  auto edge_gradient = torch::empty_like(edge_vec_float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto index_tag) {
    using index_t = decltype(index_tag);
    dispatch_backward<index_t>(
        width, node_count, edge_count, ntypes, type_one_side != 0, smooth != 0,
        static_cast<int>(axis), canonical,
        static_cast<int>(descriptor_gradient_float.size(1)),
        static_cast<float>(rcut), static_cast<float>(rcut_smooth),
        static_cast<float>(protection), static_cast<float>(1.0 / neighbors),
        static_cast<float>(lower), static_cast<float>(upper),
        static_cast<float>(table_max), static_cast<float>(stride0),
        static_cast<float>(stride1), descriptor_gradient_float,
        rotation_gradient_ptr, moment, edge_vec_float, edge_index, edge_mask,
        destination_order, destination_row_ptr, atype, average, inverse_stddev,
        table, gate_table, edge_gradient, stream);
  };
  if (edge_index.scalar_type() == torch::kInt32) {
    launch(int{});
  } else {
    launch(long{});
  }
  return edge_gradient.to(edge_vec.scalar_type());
}

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "dpa1_graph_compress(Tensor edge_vec, Tensor edge_index, "
      "Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor atype, "
      "Tensor type_embedding, Tensor average, Tensor inverse_stddev, "
      "Tensor table, Tensor gate_table, int type_one_side, "
      "int concatenate_type_embedding, int write_rotation, int smooth, "
      "int axis, bool canonical, float lower, float upper, float table_max, "
      "float stride0, float stride1, "
      "float rcut, float rcut_smooth, float protection, float neighbors) "
      "-> (Tensor descriptor, Tensor rotation, Tensor moment)");
  library.impl("dpa1_graph_compress", torch::kCUDA, &dpa1_graph_compress);
  library.def(
      "dpa1_graph_compress_backward(Tensor descriptor_gradient, "
      "Tensor? rotation_gradient, Tensor moment, Tensor edge_vec, "
      "Tensor edge_index, Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor atype, Tensor average, "
      "Tensor inverse_stddev, Tensor table, "
      "Tensor gate_table, int type_one_side, int smooth, int axis, "
      "bool canonical, float lower, float upper, float table_max, float "
      "stride0, "
      "float stride1, float rcut, float rcut_smooth, float protection, "
      "float neighbors) -> Tensor");
  library.impl("dpa1_graph_compress_backward", torch::kCUDA,
               &dpa1_graph_compress_backward);
}
