// SPDX-License-Identifier: LGPL-3.0-or-later
//
// CUDA kernels of the geometrically compressed DPA1 graph descriptor.
//
// This header is included by exactly one translation unit per channel width,
// which lets the expensive basis, topology, index, and resource-policy
// specializations compile in parallel.
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

#pragma once

#include <cuda_runtime.h>

#include <cmath>

#include "dpa1_graph_compress_launch.h"

#ifndef DEEPMD_ENABLE_DPA1_HIGH_LMAX
#define DEEPMD_ENABLE_DPA1_HIGH_LMAX 0
#endif

#include "dpa1_graph_compress_tuning.h"
#include "dpa1_moment_basis.cuh"

// The lmax=2/3/4 compressed kernels are retained below as an experimental
// implementation, but their template instances are disabled by default.
// Current DPA1 production artifacts use lmax=1; the CMake option can restore
// the higher-degree binary specializations without recovering deleted code.

namespace deepmd_dpa1_compress {
namespace {

using deepmd::dpa1_compress_tuning::DeviceProperties;
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

template <int BasisDim>
struct EdgeEnvironment {
  float radial;
  float basis[BasisDim];
  float switch_factor;
  float x;
  float y;
  float z;
  float radius;
  int pair_index;
};

template <int BasisDim, typename index_t>
__device__ __forceinline__ EdgeEnvironment<BasisDim> load_environment(
    long edge,
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
  EdgeEnvironment<BasisDim> environment;
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
  environment.basis[0] = environment.radial * inverse_neighbors;
  environment.basis[1] = (environment.x * radial_scale - center_average[1]) *
                         center_inverse_stddev[1] * inverse_neighbors;
  environment.basis[2] = (environment.y * radial_scale - center_average[2]) *
                         center_inverse_stddev[2] * inverse_neighbors;
  environment.basis[3] = (environment.z * radial_scale - center_average[3]) *
                         center_inverse_stddev[3] * inverse_neighbors;
  if constexpr (BasisDim > 4) {
    const float inverse_length =
        environment.radius > 0.0f ? __fdividef(1.0f, environment.radius) : 0.0f;
    const float ux = environment.x * inverse_length;
    const float uy = environment.y * inverse_length;
    const float uz = environment.z * inverse_length;
    const float radial = environment.radius > 0.0f
                             ? environment.switch_factor * inverse_radius *
                                   center_inverse_stddev[0] * inverse_neighbors
                             : 0.0f;
    if constexpr (BasisDim == 9) {
      deepmd::dpa1::fill_degree_two_basis<BasisDim>(environment.basis, ux, uy,
                                                    uz, radial);
    } else {
      deepmd::dpa1::fill_angular_basis<BasisDim>(environment.basis, ux, uy, uz,
                                                 radial);
    }
  }
  environment.pair_index =
      one_side ? neighbor_type : center_type * ntypes + neighbor_type;
  return environment;
}

template <int BasisDim>
__device__ __forceinline__ EdgeEnvironment<BasisDim> broadcast_environment(
    EdgeEnvironment<BasisDim> value, int source_lane, unsigned mask) {
  value.radial = __shfl_sync(mask, value.radial, source_lane);
#pragma unroll
  for (int k = 0; k < BasisDim; ++k) {
    value.basis[k] = __shfl_sync(mask, value.basis[k], source_lane);
  }
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

template <int BasisDim>
__device__ __forceinline__ void store_edge_gradient(
    long edge,
    const EdgeEnvironment<BasisDim>& environment,
    const float (&partial_basis)[BasisDim],
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
      (partial_basis[0] * inverse_neighbors + partial_radial) * inverse_stddev0;
  const float gradient_x =
      partial_basis[1] * inverse_neighbors * inverse_stddev1;
  const float gradient_y =
      partial_basis[2] * inverse_neighbors * inverse_stddev2;
  const float gradient_z =
      partial_basis[3] * inverse_neighbors * inverse_stddev3;
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
  float output_x = coefficient * environment.x + vector_scale * gradient_x;
  float output_y = coefficient * environment.y + vector_scale * gradient_y;
  float output_z = coefficient * environment.z + vector_scale * gradient_z;
  if constexpr (BasisDim == 9) {
    deepmd::dpa1::add_degree_two_edge_gradient<BasisDim>(
        partial_basis, inverse_neighbors, environment.x, environment.y,
        environment.z, environment.radius, inverse_denominator, inverse_stddev0,
        environment.switch_factor, switch_gradient, output_x, output_y,
        output_z);
  } else if constexpr (BasisDim > 9) {
    deepmd::dpa1::add_angular_edge_gradient<BasisDim>(
        partial_basis, inverse_neighbors, environment.x, environment.y,
        environment.z, environment.radius, inverse_denominator, inverse_stddev0,
        environment.switch_factor, switch_gradient, output_x, output_y,
        output_z);
  }
  edge_gradient[edge * 3 + 0] = output_x;
  edge_gradient[edge * 3 + 1] = output_y;
  edge_gradient[edge * 3 + 2] = output_z;
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

template <bool Masked>
__device__ __forceinline__ bool edge_is_active(long edge,
                                               const bool* edge_mask) {
  if constexpr (Masked) {
    return edge_mask[edge];
  }
  return true;
}

template <int Width,
          int BasisDim,
          bool Canonical,
          bool Masked,
          typename index_t,
          int MinimumBlocks>
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
    const float* __restrict__ degree_gain_raw,
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

  float accumulator[BasisDim][kGroups] = {};
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
      if (!edge_is_active<Masked>(edge, edge_mask)) {
        continue;
      }
      EdgeEnvironment<BasisDim> environment{};
      TableLocation location{};
      if (half_lane == 0) {
        environment = load_environment<BasisDim>(
            edge, center_type, ntypes, one_side, rcut, rcut_smooth, protection,
            inverse_neighbors, edge_vec, edge_index, atype, average,
            inverse_stddev);
        location = locate_table(environment.radial, lower, upper, table_max,
                                stride0, stride1);
      }
      environment =
          broadcast_environment<BasisDim>(environment, leader, half_mask);
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
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          accumulator[k][group] =
              fmaf(environment.basis[k], embedding, accumulator[k][group]);
        }
      }
    }
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
#pragma unroll
      for (int k = 0; k < BasisDim; ++k) {
        accumulator[k][group] +=
            __shfl_xor_sync(kWarpMask, accumulator[k][group], 16);
      }
    }
  } else {
    for (long position = begin; position < end; ++position) {
      const long edge =
          edge_at_csr_position<Canonical>(position, destination_order);
      if (!edge_is_active<Masked>(edge, edge_mask)) {
        continue;
      }
      EdgeEnvironment<BasisDim> environment{};
      TableLocation location{};
      if (lane == 0) {
        environment = load_environment<BasisDim>(
            edge, center_type, ntypes, one_side, rcut, rcut_smooth, protection,
            inverse_neighbors, edge_vec, edge_index, atype, average,
            inverse_stddev);
        location = locate_table(environment.radial, lower, upper, table_max,
                                stride0, stride1);
      }
      environment = broadcast_environment<BasisDim>(environment, 0, kWarpMask);
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
#pragma unroll
          for (int k = 0; k < BasisDim; ++k) {
            accumulator[k][group] =
                fmaf(environment.basis[k], embedding, accumulator[k][group]);
          }
        }
      }
    }
  }

  const long moment_base = node * BasisDim * Width;
  if constexpr (ChannelPolicy<Width>::use_half_warp) {
    if (lane < 16) {
#pragma unroll
      for (int group = 0; group < kGroups; ++group) {
        const int channel = group * 16 + lane;
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          moment[moment_base + k * Width + channel] = accumulator[k][group];
        }
      }
    }
  } else {
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
      const int channel = group * 32 + lane;
      if (channel < Width) {
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          moment[moment_base + k * Width + channel] = accumulator[k][group];
        }
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
        float value = 0.0f;
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          const float axis_value =
              __shfl_sync(kWarpMask, accumulator[k][0], axis_channel);
          const float weight =
              BasisDim == 4 ? 1.0f
                            : deepmd::dpa1::degree_weight(k, degree_gain_raw);
          value = fmaf(accumulator[k][group] * weight, axis_value, value);
        }
        if (lane < 16) {
          output[channel * axis + axis_channel] = value;
        }
      }
      if (write_rotation && lane < 16) {
        float* rotation_row = rotation + (node * Width + channel) * 3;
        rotation_row[0] = accumulator[1][group];
        rotation_row[1] = accumulator[2][group];
        rotation_row[2] = accumulator[3][group];
      }
    }
  } else {
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
      const int channel = group * 32 + lane;
      for (int axis_channel = 0; axis_channel < axis; ++axis_channel) {
        float value = 0.0f;
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          const float axis_value =
              __shfl_sync(kWarpMask, accumulator[k][0], axis_channel);
          const float weight =
              BasisDim == 4 ? 1.0f
                            : deepmd::dpa1::degree_weight(k, degree_gain_raw);
          value = fmaf(accumulator[k][group] * weight, axis_value, value);
        }
        if (channel < Width) {
          output[channel * axis + axis_channel] = value;
        }
      }
      if (write_rotation && channel < Width) {
        float* rotation_row = rotation + (node * Width + channel) * 3;
        rotation_row[0] = accumulator[1][group];
        rotation_row[1] = accumulator[2][group];
        rotation_row[2] = accumulator[3][group];
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

template <int Width,
          int BasisDim,
          bool Canonical,
          bool Masked,
          typename index_t,
          int MinimumBlocks>
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
    const float* __restrict__ degree_gain_raw,
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

  float gradient[BasisDim][kGradientGroups] = {};
  const long moment_base = node * BasisDim * Width;
  const float* node_descriptor_gradient =
      descriptor_gradient + node * descriptor_stride;
  float axis_moment[BasisDim] = {};
  if (lane < Width) {
#pragma unroll
    for (int k = 0; k < BasisDim; ++k) {
      axis_moment[k] = __ldg(moment + moment_base + k * Width + lane);
    }
  }

#pragma unroll
  for (int group = 0; group < kGradientGroups; ++group) {
    const int channel = group * 32 + lane;
    for (int axis_channel = 0; axis_channel < axis; ++axis_channel) {
      float axis_value[BasisDim];
#pragma unroll
      for (int k = 0; k < BasisDim; ++k) {
        axis_value[k] = __shfl_sync(kWarpMask, axis_moment[k], axis_channel);
      }
      if (channel < Width) {
        const float value =
            __ldg(node_descriptor_gradient + channel * axis + axis_channel);
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          gradient[k][group] = fmaf(value, axis_value[k], gradient[k][group]);
        }
      }
    }
    if (channel < axis) {
      for (int input = 0; input < Width; ++input) {
        const float value =
            __ldg(node_descriptor_gradient + input * axis + channel);
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          gradient[k][group] =
              fmaf(value, __ldg(moment + moment_base + k * Width + input),
                   gradient[k][group]);
        }
      }
    }
    if constexpr (BasisDim > 4) {
#pragma unroll
      for (int k = 0; k < BasisDim; ++k) {
        gradient[k][group] *= deepmd::dpa1::degree_weight(k, degree_gain_raw);
      }
    }
    if (rotation_gradient != nullptr && channel < Width) {
      const long rotation_offset = (node * Width + channel) * 3;
      gradient[1][group] += __ldg(rotation_gradient + rotation_offset + 0);
      gradient[2][group] += __ldg(rotation_gradient + rotation_offset + 1);
      gradient[3][group] += __ldg(rotation_gradient + rotation_offset + 2);
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
    float edge_moment_gradient[BasisDim][kEdgeGroups] = {};
#pragma unroll
    for (int group = 0; group < kEdgeGroups; ++group) {
      const int channel = group * 16 + half_lane;
      const int owner_group = channel / 32;
      const int owner_lane = channel & 31;
#pragma unroll
      for (int k = 0; k < BasisDim; ++k) {
        edge_moment_gradient[k][group] =
            __shfl_sync(kWarpMask, gradient[k][owner_group], owner_lane);
      }
    }

    for (long position = begin + half; position < end; position += 2) {
      const long edge =
          edge_at_csr_position<Canonical>(position, destination_order);
      if (!edge_is_active<Masked>(edge, edge_mask)) {
        if (half_lane == 0) {
          edge_gradient[edge * 3 + 0] = 0.0f;
          edge_gradient[edge * 3 + 1] = 0.0f;
          edge_gradient[edge * 3 + 2] = 0.0f;
        }
        continue;
      }
      EdgeEnvironment<BasisDim> environment{};
      TableLocation location{};
      if (half_lane == 0) {
        environment = load_environment<BasisDim>(
            edge, center_type, ntypes, one_side, rcut, rcut_smooth, protection,
            inverse_neighbors, edge_vec, edge_index, atype, average,
            inverse_stddev);
        location = locate_table(environment.radial, lower, upper, table_max,
                                stride0, stride1);
      }
      environment =
          broadcast_environment<BasisDim>(environment, leader, half_mask);
      location = broadcast_location(location, leader, half_mask);
      float partial_basis[BasisDim] = {};
      float partial_radial = 0.0f;
      float partial_switch = 0.0f;
#pragma unroll
      for (int group = 0; group < kEdgeGroups; ++group) {
        const int channel = group * 16 + half_lane;
        float descriptor_product = 0.0f;
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          descriptor_product =
              fmaf(environment.basis[k], edge_moment_gradient[k][group],
                   descriptor_product);
        }
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
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          partial_basis[k] =
              fmaf(embedding, edge_moment_gradient[k][group], partial_basis[k]);
        }
      }
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1) {
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          partial_basis[k] +=
              __shfl_down_sync(half_mask, partial_basis[k], offset, 16);
        }
        partial_radial +=
            __shfl_down_sync(half_mask, partial_radial, offset, 16);
        partial_switch +=
            __shfl_down_sync(half_mask, partial_switch, offset, 16);
      }
      if (half_lane == 0) {
        store_edge_gradient<BasisDim>(
            edge, environment, partial_basis, partial_radial, partial_switch,
            inverse_neighbors, inverse_stddev0, inverse_stddev1,
            inverse_stddev2, inverse_stddev3, rcut, rcut_smooth, protection,
            edge_gradient);
      }
    }
  } else {
    for (long position = begin; position < end; ++position) {
      const long edge =
          edge_at_csr_position<Canonical>(position, destination_order);
      if (!edge_is_active<Masked>(edge, edge_mask)) {
        if (lane == 0) {
          edge_gradient[edge * 3 + 0] = 0.0f;
          edge_gradient[edge * 3 + 1] = 0.0f;
          edge_gradient[edge * 3 + 2] = 0.0f;
        }
        continue;
      }
      EdgeEnvironment<BasisDim> environment{};
      TableLocation location{};
      if (lane == 0) {
        environment = load_environment<BasisDim>(
            edge, center_type, ntypes, one_side, rcut, rcut_smooth, protection,
            inverse_neighbors, edge_vec, edge_index, atype, average,
            inverse_stddev);
        location = locate_table(environment.radial, lower, upper, table_max,
                                stride0, stride1);
      }
      environment = broadcast_environment<BasisDim>(environment, 0, kWarpMask);
      location = broadcast_location(location, 0, kWarpMask);
      float partial_basis[BasisDim] = {};
      float partial_radial = 0.0f;
      float partial_switch = 0.0f;
#pragma unroll
      for (int group = 0; group < kGradientGroups; ++group) {
        const int channel = group * 32 + lane;
        if (channel < Width) {
          float descriptor_product = 0.0f;
#pragma unroll
          for (int k = 0; k < BasisDim; ++k) {
            descriptor_product = fmaf(environment.basis[k], gradient[k][group],
                                      descriptor_product);
          }
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
#pragma unroll
          for (int k = 0; k < BasisDim; ++k) {
            partial_basis[k] =
                fmaf(embedding, gradient[k][group], partial_basis[k]);
          }
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int k = 0; k < BasisDim; ++k) {
          partial_basis[k] +=
              __shfl_down_sync(kWarpMask, partial_basis[k], offset);
        }
        partial_radial += __shfl_down_sync(kWarpMask, partial_radial, offset);
        partial_switch += __shfl_down_sync(kWarpMask, partial_switch, offset);
      }
      if (lane == 0) {
        store_edge_gradient<BasisDim>(
            edge, environment, partial_basis, partial_radial, partial_switch,
            inverse_neighbors, inverse_stddev0, inverse_stddev1,
            inverse_stddev2, inverse_stddev3, rcut, rcut_smooth, protection,
            edge_gradient);
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

template <int Width,
          int BasisDim,
          typename index_t,
          bool Canonical,
          bool Masked,
          int MinimumBlocks>
void launch_forward_variant(const Arguments& arguments,
                            long node_count,
                            int threads,
                            cudaStream_t stream) {
  const int warps_per_block = threads / kWarpSize;
  const int blocks =
      static_cast<int>((node_count + warps_per_block - 1) / warps_per_block);
  compressed_forward_kernel<Width, BasisDim, Canonical, Masked, index_t,
                            MinimumBlocks><<<blocks, threads, 0, stream>>>(
      node_count, arguments.type_count, arguments.one_side, arguments.smooth,
      arguments.axis, arguments.concatenate_type_embedding,
      arguments.write_rotation, arguments.type_embedding_dim, arguments.rcut,
      arguments.rcut_smooth, arguments.protection, arguments.inverse_neighbors,
      arguments.lower, arguments.upper, arguments.table_max, arguments.stride0,
      arguments.stride1, arguments.edge_vec,
      static_cast<const index_t*>(arguments.edge_index), arguments.edge_mask,
      static_cast<const index_t*>(arguments.destination_order),
      arguments.destination_row_ptr, arguments.atype, arguments.type_embedding,
      arguments.average, arguments.inverse_stddev, arguments.table,
      arguments.gate_table, arguments.degree_gain, arguments.descriptor,
      arguments.rotation, arguments.moment_out);
}

template <int Width,
          int BasisDim,
          typename index_t,
          bool Canonical,
          bool Masked>
cudaError_t launch_forward(const Arguments& arguments, cudaStream_t stream) {
  const DeviceProperties properties = {arguments.device_major,
                                       arguments.multiprocessor_count};
  const TuningKey key = {
      arguments.device,
      static_cast<int>(KernelDirection::kForward),
      Width,
      BasisDim,
      arguments.axis,
      Canonical ? 1 : 0,
      static_cast<int>(sizeof(index_t)),
      (arguments.one_side ? 1 : 0) | (arguments.smooth ? 2 : 0) |
          (arguments.concatenate_type_embedding ? 4 : 0) |
          (arguments.write_rotation ? 8 : 0) | (Masked ? 16 : 0),
      arguments.concatenate_type_embedding ? arguments.type_embedding_dim : 0,
      type_count_class(arguments.type_count),
      workload_size_class(arguments.node_count,
                          properties.multiprocessor_count),
      workload_degree_class(arguments.node_count, arguments.edge_count),
  };
  const auto launch = [&](const LaunchConfig& config, long count) {
    if (config.resource == ResourcePolicy::kOccupancy) {
      launch_forward_variant<Width, BasisDim, index_t, Canonical, Masked, 4>(
          arguments, count, config.threads, stream);
    } else {
      launch_forward_variant<Width, BasisDim, index_t, Canonical, Masked, 2>(
          arguments, count, config.threads, stream);
    }
  };
  const LaunchConfig config = select_launch_config(
      key, properties, arguments.node_count, stream, launch);
  launch(config, arguments.node_count);
  return cudaGetLastError();
}

template <int Width,
          int BasisDim,
          typename index_t,
          bool Canonical,
          bool Masked,
          int MinimumBlocks>
void launch_backward_variant(const Arguments& arguments,
                             long node_count,
                             int threads,
                             cudaStream_t stream) {
  const int warps_per_block = threads / kWarpSize;
  const int blocks =
      static_cast<int>((node_count + warps_per_block - 1) / warps_per_block);
  compressed_backward_kernel<Width, BasisDim, Canonical, Masked, index_t,
                             MinimumBlocks><<<blocks, threads, 0, stream>>>(
      node_count, arguments.edge_count, arguments.type_count,
      arguments.one_side, arguments.smooth, arguments.axis,
      arguments.descriptor_stride, arguments.rcut, arguments.rcut_smooth,
      arguments.protection, arguments.inverse_neighbors, arguments.lower,
      arguments.upper, arguments.table_max, arguments.stride0,
      arguments.stride1, arguments.descriptor_gradient,
      arguments.rotation_gradient, arguments.moment, arguments.edge_vec,
      static_cast<const index_t*>(arguments.edge_index), arguments.edge_mask,
      static_cast<const index_t*>(arguments.destination_order),
      arguments.destination_row_ptr, arguments.atype, arguments.average,
      arguments.inverse_stddev, arguments.table, arguments.gate_table,
      arguments.degree_gain, arguments.edge_gradient);
}

template <int Width,
          int BasisDim,
          typename index_t,
          bool Canonical,
          bool Masked>
cudaError_t launch_backward(const Arguments& arguments, cudaStream_t stream) {
  const DeviceProperties properties = {arguments.device_major,
                                       arguments.multiprocessor_count};
  const TuningKey key = {
      arguments.device,
      static_cast<int>(KernelDirection::kBackward),
      Width,
      BasisDim,
      arguments.axis,
      Canonical ? 1 : 0,
      static_cast<int>(sizeof(index_t)),
      (arguments.one_side ? 1 : 0) | (arguments.smooth ? 2 : 0) |
          (arguments.rotation_gradient != nullptr ? 8 : 0) | (Masked ? 16 : 0),
      arguments.descriptor_stride,
      type_count_class(arguments.type_count),
      workload_size_class(arguments.node_count,
                          properties.multiprocessor_count),
      workload_degree_class(arguments.node_count, arguments.edge_count),
  };
  const auto launch = [&](const LaunchConfig& config, long count) {
    if (config.resource == ResourcePolicy::kOccupancy) {
      launch_backward_variant<Width, BasisDim, index_t, Canonical, Masked, 4>(
          arguments, count, config.threads, stream);
    } else {
      launch_backward_variant<Width, BasisDim, index_t, Canonical, Masked, 2>(
          arguments, count, config.threads, stream);
    }
  };
  const LaunchConfig config = select_launch_config(
      key, properties, arguments.node_count, stream, launch);
  launch(config, arguments.node_count);
  const cudaError_t backward_error = cudaGetLastError();
  if (backward_error != cudaSuccess) {
    return backward_error;
  }
  zero_padding_kernel<Canonical, index_t><<<1, kThreads, 0, stream>>>(
      arguments.node_count, arguments.edge_count,
      static_cast<const index_t*>(arguments.destination_order),
      arguments.destination_row_ptr, arguments.edge_gradient);
  return cudaGetLastError();
}

template <int Width, int BasisDim, typename index_t>
cudaError_t dispatch_forward_topology(const Arguments& arguments,
                                      cudaStream_t stream) {
  if (arguments.canonical && !arguments.masked) {
    return launch_forward<Width, BasisDim, index_t, true, false>(arguments,
                                                                 stream);
  }
  if (arguments.canonical) {
    return launch_forward<Width, BasisDim, index_t, true, true>(arguments,
                                                                stream);
  }
  return launch_forward<Width, BasisDim, index_t, false, true>(arguments,
                                                               stream);
}

template <int Width, int BasisDim, typename index_t>
cudaError_t dispatch_backward_topology(const Arguments& arguments,
                                       cudaStream_t stream) {
  if (arguments.canonical && !arguments.masked) {
    return launch_backward<Width, BasisDim, index_t, true, false>(arguments,
                                                                  stream);
  }
  if (arguments.canonical) {
    return launch_backward<Width, BasisDim, index_t, true, true>(arguments,
                                                                 stream);
  }
  return launch_backward<Width, BasisDim, index_t, false, true>(arguments,
                                                                stream);
}

template <int Width, typename index_t>
cudaError_t dispatch_forward_basis(const Arguments& arguments,
                                   cudaStream_t stream) {
  if (arguments.basis_dim == 4) {
    return dispatch_forward_topology<Width, 4, index_t>(arguments, stream);
  }
#if DEEPMD_ENABLE_DPA1_HIGH_LMAX
  if (arguments.basis_dim == 9) {
    return dispatch_forward_topology<Width, 9, index_t>(arguments, stream);
  }
  if constexpr (Width >= 16 && Width <= 128) {
    if (arguments.basis_dim == 16) {
      return dispatch_forward_topology<Width, 16, index_t>(arguments, stream);
    }
    if (arguments.basis_dim == 25) {
      return dispatch_forward_topology<Width, 25, index_t>(arguments, stream);
    }
  }
#endif
  return cudaErrorInvalidValue;
}

template <int Width, typename index_t>
cudaError_t dispatch_backward_basis(const Arguments& arguments,
                                    cudaStream_t stream) {
  if (arguments.basis_dim == 4) {
    return dispatch_backward_topology<Width, 4, index_t>(arguments, stream);
  }
#if DEEPMD_ENABLE_DPA1_HIGH_LMAX
  if (arguments.basis_dim == 9) {
    return dispatch_backward_topology<Width, 9, index_t>(arguments, stream);
  }
  if constexpr (Width >= 16 && Width <= 128) {
    if (arguments.basis_dim == 16) {
      return dispatch_backward_topology<Width, 16, index_t>(arguments, stream);
    }
    if (arguments.basis_dim == 25) {
      return dispatch_backward_topology<Width, 25, index_t>(arguments, stream);
    }
  }
#endif
  return cudaErrorInvalidValue;
}

template <int Width>
cudaError_t dispatch_forward_index(const Arguments& arguments,
                                   cudaStream_t stream) {
  if (arguments.index_kind == IndexKind::kInt32) {
    return dispatch_forward_basis<Width, int>(arguments, stream);
  }
  return dispatch_forward_basis<Width, long>(arguments, stream);
}

template <int Width>
cudaError_t dispatch_backward_index(const Arguments& arguments,
                                    cudaStream_t stream) {
  if (arguments.index_kind == IndexKind::kInt32) {
    return dispatch_backward_basis<Width, int>(arguments, stream);
  }
  return dispatch_backward_basis<Width, long>(arguments, stream);
}

}  // namespace

#define DPA1_COMPRESS_DEFINE_CHANNEL(width)                        \
  cudaError_t launch_forward_c##width(const Arguments& arguments,  \
                                      cudaStream_t stream) {       \
    return dispatch_forward_index<width>(arguments, stream);       \
  }                                                                \
  cudaError_t launch_backward_c##width(const Arguments& arguments, \
                                       cudaStream_t stream) {      \
    return dispatch_backward_index<width>(arguments, stream);      \
  }

}  // namespace deepmd_dpa1_compress
