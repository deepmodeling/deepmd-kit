// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Kernels of the compressed degree-wise DPA4C descriptor.
//
// This header is included by exactly one translation unit per scalar width,
// which instantiates the angular-degree and topology specializations of that
// width. Keeping the instantiations separated bounds the compile time of any
// single translation unit and lets them proceed in parallel.

#pragma once

#include <stdexcept>

#include "graph_compress.cuh"

namespace deepmd_dpa4c {

// Shared mode profiles are cached per edge group. The stride is a multiple of
// four so that each row is aligned for the vector reads of the mode residual,
// and it keeps the concurrent groups of one warp on distinct banks.
constexpr int kModeStride = kMaxRadialModes + 4;

// === Native spin block accessors ===
//
// Channel zero of each spin block is the node-local on-site channel, which is
// stored outside the reduced region because it carries no neighborhood
// normalizer. Reading both through one accessor keeps the Gram loops free of
// that distinction.

/// Flat moment coordinate of one entry of the joint degree-one spin block.
///
/// The block holds the on-site moment at channel zero, the ``Cs`` isotropic
/// neighbour channels, and the ``Cs`` bond-projected neighbour channels, in
/// that order. The two neighbour families share the grading of the block, so
/// the Gram of the whole block is what emits every admissible degree-one spin
/// invariant; resolving all three regions through one accessor keeps the Gram
/// and its VJP free of the distinction between them.
template <int Channels, int Lmax>
__device__ __forceinline__ int spin_vector_offset(int component, int channel) {
  using P = Profile<Channels, Lmax, true>;
  if (channel == 0) {
    return P::SpinOnsiteVector + component;
  }
  const int neighbor = channel - 1;
  return neighbor < P::Cs
             ? P::SpinVector + component * P::Cs + neighbor
             : P::SpinBond + component * P::Cs + (neighbor - P::Cs);
}

template <int Channels, int Lmax>
__device__ __forceinline__ float spin_vector_value(const float* moments,
                                                   int component,
                                                   int channel) {
  return moments[spin_vector_offset<Channels, Lmax>(component, channel)];
}

template <int Channels, int Lmax>
__device__ __forceinline__ float spin_tensor_value(const float* moments,
                                                   int component,
                                                   int channel) {
  using P = Profile<Channels, Lmax, true>;
  return channel == 0 ? moments[P::SpinOnsiteTensor + component]
                      : moments[P::SpinTensor + component];
}

/// Decode one retained entry of the quadrupole Gram, whose on-site self-term
/// is omitted because ``|B_2(s)|^2 = |s|^4`` makes it a per-type constant
/// times the vector block's own on-site self-term.
__device__ __forceinline__ void decode_spin_tensor_pair(int entry,
                                                        int& row,
                                                        int& column) {
  row = entry;
  column = 1;
}

// === Forward ===

template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          typename index_t>
__global__ __launch_bounds__(
    Profile<Channels, Lmax, HasSpin>::Threads,
    Profile<Channels, Lmax, HasSpin>::
        MinBlocksPerSm) void forward_kernel(Arguments args) {
  using P = Profile<Channels, Lmax, HasSpin>;
  constexpr int EdgeWidth = P::ForwardEdgeWidth;
  constexpr int Groups = kWarpSize / EdgeWidth;
  constexpr int ChannelTiles = Channels / EdgeWidth;
  constexpr int AngularTiles = (P::C1 + EdgeWidth - 1) / EdgeWidth;
  constexpr int TensorTiles = (P::C2 + EdgeWidth - 1) / EdgeWidth;
  constexpr int HighTiles = (P::HighCount + EdgeWidth - 1) / EdgeWidth;
  // The spin families share the neighbour width with degree two, so they
  // inherit its tiling. The single neighbour quadrupole distributes its five
  // components across lanes, exactly as the single-channel high degrees do.
  constexpr int SpinTiles = HasSpin ? TensorTiles : 0;
  constexpr int SpinTensorTiles = HasSpin ? (5 + EdgeWidth - 1) / EdgeWidth : 0;

  const int thread = threadIdx.x;
  const long node = blockIdx.x;
  if (node >= args.node_count) {
    return;
  }
  const int base_channel = thread & (EdgeWidth - 1);
  const int group = thread / EdgeWidth;
  const int leader = group * EdgeWidth;
  const unsigned mask = subwarp_mask<EdgeWidth>(leader);
  const auto* edge_index = static_cast<const index_t*>(args.edge_index);
  const auto* destination_order =
      static_cast<const index_t*>(args.destination_order);
  const int center_type = static_cast<int>(args.atype[args.node_begin + node]);
  const long begin = args.destination_row_ptr[node];
  const long end = args.destination_row_ptr[node + 1];
  const int radial_modes = HasModes ? args.radial_modes : 0;

  float scalar[ChannelTiles] = {};
  float vector[AngularTiles][3] = {};
  float tensor[TensorTiles][5] = {};
  float high[HighTiles > 0 ? HighTiles : 1] = {};
  float scalar_mass = 0.0f;
  float angular_mass = 0.0f;
  float spin_magnitude[SpinTiles > 0 ? SpinTiles : 1] = {};
  float spin_coordination[SpinTiles > 0 ? SpinTiles : 1] = {};
  float spin_vector[SpinTiles > 0 ? SpinTiles : 1][3] = {};
  float spin_bond[SpinTiles > 0 ? SpinTiles : 1][3] = {};
  float spin_tensor[SpinTensorTiles > 0 ? SpinTensorTiles : 1] = {};

  // Explicitly aligned: the mode residual reads these rows as float4, and
  // a preceding shared array of arbitrary length would otherwise leave the
  // block on a four-byte boundary.
  __shared__ __align__(
      16) float mode_cache[HasModes ? Groups * kModeStride : 1];
  float* modes = mode_cache + (HasModes ? group * kModeStride : 0);

  // === Step 1. Reduce the destination row into degree-wise moments ===
  for (long position = begin + group; position < end; position += Groups) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (args.edge_mask != nullptr && !args.edge_mask[edge]) {
      continue;
    }
    // Every lane of the group reloads the shared edge state instead of
    // broadcasting it from a leader. The addresses are identical inside the
    // group, so the memory system serves one transaction either way, whereas
    // a leader-only branch pays the same issue slots and adds ten shuffles.
    const EdgeGeometry geometry = load_geometry<Canonical, index_t>(
        edge, args.rcut, args.eps, args.edge_vec, edge_index, args.atype);
    const TableLocation location =
        locate_table<Canonical>(geometry.radius, args.table_stride,
                                args.table_max, args.interval_count);
    if constexpr (!Canonical) {
      if (center_type >= args.type_count - 1 ||
          geometry.source_type >= args.type_count - 1) {
        continue;
      }
    }
    const TableRow row = table_row(args.table, location, args.table_width);
    const float coordinate = location.coordinate;
    if constexpr (HasModes) {
      __syncwarp(mask);
      for (int mode = base_channel; mode < radial_modes; mode += EdgeWidth) {
        modes[mode] = evaluate_table(row, Channels + mode, coordinate);
      }
      __syncwarp(mask);
    }

    float basis[9];
    fill_angular_basis(geometry, basis);
    const float envelope = geometry.envelope;
    const long pair =
        static_cast<long>(center_type) * args.type_count + geometry.source_type;
    const float2* film_row =
        reinterpret_cast<const float2*>(args.pair_film + pair * Channels * 2) +
        base_channel;
    const float* mixing_row =
        HasModes
            ? args.pair_mixing + (pair * Channels + base_channel) * radial_modes
            : nullptr;

    // The spin payload is hoisted out of the channel loop: every channel
    // multiplies the same conditioned neighbour moment and the same projection
    // of that moment onto the bond, and only the amplitude varies with the
    // channel.
    float neighbor_spin[3] = {0.0f, 0.0f, 0.0f};
    float neighbor_magnitude = 0.0f;
    float neighbor_gate = 0.0f;
    float bond_alignment = 0.0f;
    const float2* spin_row = nullptr;
    if constexpr (HasSpin) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, geometry.source_type);
      load_conditioned_spin(args.spin, weights.scale, geometry.source,
                            neighbor_spin);
      neighbor_magnitude = neighbor_spin[0] * neighbor_spin[0] +
                           neighbor_spin[1] * neighbor_spin[1] +
                           neighbor_spin[2] * neighbor_spin[2];
      neighbor_gate = weights.gate;
      // Component of the neighbour moment along the bond. The bond-projected
      // family is this scalar times the unit direction, which is `basis[1..3]`.
      bond_alignment = neighbor_spin[0] * basis[1] +
                       neighbor_spin[1] * basis[2] +
                       neighbor_spin[2] * basis[3];
      spin_row =
          reinterpret_cast<const float2*>(args.spin_pair + pair * P::Cs * 2) +
          base_channel;
    }

    float angular_zero = 0.0f;
    float spin_zero = 0.0f;
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      const float radial = evaluate_table(row, channel, coordinate);
      const float2 film = __ldg(film_row + tile * EdgeWidth);
      float film_value = fmaf(film.x, radial, film.y);
      if constexpr (HasModes) {
        accumulate_modes(mixing_row + tile * (EdgeWidth * radial_modes), modes,
                         radial_modes, film_value);
      }
      // Degree zero carries one envelope factor; every non-scalar degree
      // carries a second one.
      const float amplitude = film_value * envelope;
      const float angular = amplitude * envelope;
      scalar[tile] += amplitude;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          vector[tile][component] =
              fmaf(angular, basis[1 + component], vector[tile][component]);
        }
      }
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          tensor[tile][component] =
              fmaf(angular, basis[4 + component], tensor[tile][component]);
        }
      }
      if constexpr (HasSpin) {
        if (channel < P::Cs) {
          const float2 spin_film = __ldg(spin_row + tile * EdgeWidth);
          const float weight =
              fmaf(spin_film.x, radial, spin_film.y) * envelope * envelope;
          spin_magnitude[tile] =
              fmaf(weight, neighbor_magnitude, spin_magnitude[tile]);
          spin_coordination[tile] =
              fmaf(weight, neighbor_gate, spin_coordination[tile]);
          const float projected = weight * bond_alignment;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            spin_vector[tile][component] = fmaf(
                weight, neighbor_spin[component], spin_vector[tile][component]);
            spin_bond[tile][component] = fmaf(projected, basis[1 + component],
                                              spin_bond[tile][component]);
          }
          if (channel == 0) {
            spin_zero = weight;
          }
        }
      }
      if constexpr (Lmax >= 3) {
        if (channel == 0) {
          angular_zero = angular;
        }
      }
    }
    if constexpr (HasSpin) {
      // The single neighbour quadrupole reads the leading spin channel only,
      // so its five components are distributed across the lanes of the edge
      // group and each is evaluated on the one lane that owns it.
      const float weight = __shfl_sync(mask, spin_zero, leader);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        if (component % EdgeWidth == base_channel) {
          spin_tensor[component / EdgeWidth] =
              fmaf(weight,
                   spin_quadrupole_component(neighbor_spin[0], neighbor_spin[1],
                                             neighbor_spin[2], component),
                   spin_tensor[component / EdgeWidth]);
        }
      }
    }
    if constexpr (Lmax >= 3) {
      // Degrees three and above read only the leading channel, so their
      // components are distributed across the lanes of the edge group. The
      // component is iterated at compile time and selected with a lane
      // predicate, which folds the harmonic selector and keeps the tile index
      // constant.
      const float amplitude = __shfl_sync(mask, angular_zero, leader);
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          high[component / EdgeWidth] =
              fmaf(amplitude, high_basis_value(geometry, component),
                   high[component / EdgeWidth]);
        }
      }
    }
    if (base_channel == 0) {
      const float squared = envelope * envelope;
      scalar_mass += squared;
      angular_mass = fmaf(squared, squared, angular_mass);
    }
  }

  // === Step 2. Merge the concurrent edge groups and normalize ===
  if constexpr (Groups > 1) {
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      scalar[tile] = reduce_channel_groups<EdgeWidth>(scalar[tile]);
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        vector[tile][component] =
            reduce_channel_groups<EdgeWidth>(vector[tile][component]);
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        tensor[tile][component] =
            reduce_channel_groups<EdgeWidth>(tensor[tile][component]);
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int tile = 0; tile < HighTiles; ++tile) {
        high[tile] = reduce_channel_groups<EdgeWidth>(high[tile]);
      }
    }
    if constexpr (HasSpin) {
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        spin_magnitude[tile] =
            reduce_channel_groups<EdgeWidth>(spin_magnitude[tile]);
        spin_coordination[tile] =
            reduce_channel_groups<EdgeWidth>(spin_coordination[tile]);
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          spin_vector[tile][component] =
              reduce_channel_groups<EdgeWidth>(spin_vector[tile][component]);
          spin_bond[tile][component] =
              reduce_channel_groups<EdgeWidth>(spin_bond[tile][component]);
        }
      }
#pragma unroll
      for (int tile = 0; tile < SpinTensorTiles; ++tile) {
        spin_tensor[tile] = reduce_channel_groups<EdgeWidth>(spin_tensor[tile]);
      }
    }
  }
  __shared__ float normalizer_shared[4];
  {
    const float total_scalar = warp_sum(scalar_mass);
    const float total_angular = warp_sum(angular_mass);
    if (thread == 0) {
      normalizer_shared[0] = rsqrtf(total_scalar + args.degree_floor);
      normalizer_shared[1] = rsqrtf(total_angular + args.degree_floor);
      normalizer_shared[2] = sqrtf(total_scalar + args.degree_floor);
      normalizer_shared[3] = sqrtf(total_angular + args.degree_floor);
    }
  }
  __syncthreads();
  const float scalar_norm = normalizer_shared[0];
  const float angular_norm = normalizer_shared[1];
  if (thread < 2) {
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputDivisor + thread,
                     normalizer_shared[2 + thread]);
  }

  // === Step 3. Publish the normalized moments and the saved state ===
  __shared__ float moments[P::MomentWidth];
  if (thread < EdgeWidth) {
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      moments[P::ScalarOffset + base_channel + tile * EdgeWidth] =
          scalar[tile] * scalar_norm;
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          moments[P::VectorOffset + component * P::C1 + channel] =
              vector[tile][component] * angular_norm;
        }
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          moments[P::TensorOffset + component * P::C2 + channel] =
              tensor[tile][component] * angular_norm;
        }
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          moments[P::HighOffset + component] =
              high[component / EdgeWidth] * angular_norm;
        }
      }
    }
    if constexpr (HasSpin) {
      // The reduced spin families carry the same squared envelope as every
      // non-scalar geometric moment and therefore share its normalizer.
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        const int channel = base_channel + tile * EdgeWidth;
        if (channel < P::Cs) {
          moments[P::SpinMagnitude + channel] =
              spin_magnitude[tile] * angular_norm;
          moments[P::SpinCoordination + channel] =
              spin_coordination[tile] * angular_norm;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            moments[P::SpinVector + component * P::Cs + channel] =
                spin_vector[tile][component] * angular_norm;
            moments[P::SpinBond + component * P::Cs + channel] =
                spin_bond[tile][component] * angular_norm;
          }
        }
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        if (component % EdgeWidth == base_channel) {
          moments[P::SpinTensor + component] =
              spin_tensor[component / EdgeWidth] * angular_norm;
        }
      }
    }
  }
  if constexpr (HasSpin) {
    // The on-site families are node local. They are written after the
    // division so that an invariant pairing an on-site channel with a
    // neighbour channel carries exactly one neighborhood normalizer.
    if (thread == 0) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, center_type);
      float center_spin[3];
      load_conditioned_spin(args.spin, weights.scale, args.node_begin + node,
                            center_spin);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        moments[P::SpinOnsiteVector + component] =
            weights.vector * center_spin[component];
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        moments[P::SpinOnsiteTensor + component] =
            weights.quadrupole *
            spin_quadrupole_component(center_spin[0], center_spin[1],
                                      center_spin[2], component);
      }
    }
  }
  __syncthreads();

  const long state_offset = node * P::StateWidth;
  for (int coordinate = thread; coordinate < P::MomentWidth;
       coordinate += P::Threads) {
    args.state_out[state_offset + coordinate] = moments[coordinate];
  }
  if (thread == 0) {
    args.state_out[state_offset + P::MomentWidth] = scalar_norm;
    args.state_out[state_offset + P::MomentWidth + 1] = angular_norm;
  }

  // === Step 4. Align and project the wide degrees ===
  __shared__ float aligned[P::AlignedWidth];
  if (thread < P::C1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(moments[P::VectorOffset + component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 0,
                                                    input, thread),
                     value);
      }
      aligned[component * P::C1 + thread] = value;
    }
  }
  if (thread < P::C2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(moments[P::TensorOffset + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 2,
                                                    input, thread),
                     value);
      }
      aligned[3 * P::C1 + component * P::C2 + thread] = value;
    }
  }
  __syncthreads();

  __shared__ float probes[P::ProbeWidth];
  if (thread < P::K1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(aligned[component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 4,
                                                    input, thread),
                     value);
      }
      probes[component * P::K1 + thread] = value;
    }
  }
  if (thread < P::K2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(aligned[3 * P::C1 + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 6,
                                                    input, thread),
                     value);
      }
      probes[3 * P::K1 + component * P::K2 + thread] = value;
    }
  }
  __syncthreads();

  // === Step 5. Emit the invariant blocks ===
  for (int channel = thread; channel < Channels; channel += P::Threads) {
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputScalar + channel,
                     moments[P::ScalarOffset + channel]);
    store_descriptor(
        args.descriptor, args.output_mean, args.output_inv_std, node,
        P::OutputWidth, P::OutputType + channel,
        __ldg(args.type_embedding + static_cast<long>(center_type) * Channels +
              channel));
  }

  for (int pair = thread; pair < P::Gram1; pair += P::Threads) {
    int row, column;
    decode_upper_pair(pair, P::C1, row, column);
    float value = 0.0f;
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      value = fmaf(aligned[component * P::C1 + row],
                   aligned[component * P::C1 + column], value);
    }
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputGram1 + pair,
                     (row == column ? 1.0f : kSqrtTwo) * value);
  }
  for (int pair = thread; pair < P::Gram2; pair += P::Threads) {
    int row, column;
    decode_upper_pair(pair, P::C2, row, column);
    float value = 0.0f;
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      value = fmaf(aligned[3 * P::C1 + component * P::C2 + row],
                   aligned[3 * P::C1 + component * P::C2 + column], value);
    }
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputGram2 + pair,
                     (row == column ? 1.0f : kSqrtTwo) * value);
  }
  if constexpr (Lmax >= 3) {
    if (thread < Lmax - 2) {
      const int offset = P::HighOffset + (thread == 0 ? 0 : P::High3);
      const int count = 2 * (3 + thread) + 1;
      float value = 0.0f;
      for (int component = 0; component < count; ++component) {
        const float moment = moments[offset + component];
        value = fmaf(moment, moment, value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputGram3 + thread, value);
    }
  }

  for (int output = thread; output < P::Bis112; output += P::Threads) {
    const int tensor_index = output % P::K2;
    int first, second;
    decode_upper_pair(output / P::K2, P::K1, first, second);
    float packed[5];
    float left[3];
    float right[3];
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      packed[component] = probes[3 * P::K1 + component * P::K2 + tensor_index];
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      left[component] = probes[component * P::K1 + first];
      right[component] = probes[component * P::K1 + second];
    }
    float product[3];
    matrix_vector(packed_to_stf(packed), right, product);
    const float value =
        -kInvSqrtFive *
        (left[0] * product[0] + left[1] * product[1] + left[2] * product[2]);
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputBis112 + output,
                     (first == second ? 1.0f : kSqrtTwo) * value);
  }

  if (thread < P::Bis222) {
    constexpr int entries[4][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    constexpr float scales[4] = {1.0f, kSqrtThree, kSqrtThree, 1.0f};
    Matrix3 matrices[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      float packed[5];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] =
            probes[3 * P::K1 + component * P::K2 + entries[thread][axis]];
      }
      matrices[axis] = packed_to_stf(packed);
    }
    const float value =
        kBis222Scale *
        matrix_trace(matrix_product(matrix_product(matrices[0], matrices[1]),
                                    matrices[2]));
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputBis222 + thread,
                     scales[thread] * value);
  }

  if constexpr (Lmax >= 3) {
    for (int record = 0; record < args.coupling_count; ++record) {
      const int* meta = args.coupling_meta + record * 8;
      const int degree_1 = meta[0];
      const int degree_2 = meta[1];
      const int degree_3 = meta[2];
      const int nonzero_begin = meta[3];
      const int nonzero_count = meta[4];
      const int probe_begin = meta[5];
      const int probe_count = meta[6];
      const int coordinate = meta[7];
      for (int output = thread; output < probe_count; output += P::Threads) {
        const int selection = __ldg(args.coupling_entry + probe_begin + output);
        const int index_1 = selection & 0xFF;
        const int index_2 = (selection >> 8) & 0xFF;
        const int index_3 = (selection >> 16) & 0xFF;
        float value = 0.0f;
        for (int term = 0; term < nonzero_count; ++term) {
          const int components =
              __ldg(args.coupling_entry + nonzero_begin + term);
          const float weight =
              __ldg(args.coupling_value + nonzero_begin + term);
          const float first = probe_value<Channels, Lmax>(
              probes, moments, degree_1, components & 0xFF, index_1);
          const float second = probe_value<Channels, Lmax>(
              probes, moments, degree_2, (components >> 8) & 0xFF, index_2);
          const float third = probe_value<Channels, Lmax>(
              probes, moments, degree_3, (components >> 16) & 0xFF, index_3);
          value = fmaf(weight * first * second, third, value);
        }
        store_descriptor(
            args.descriptor, args.output_mean, args.output_inv_std, node,
            P::OutputWidth, coordinate + output,
            value * __ldg(args.coupling_value + probe_begin + output));
      }
    }
  }

  for (int output = thread; output < P::Quartic; output += P::Threads) {
    const int vector_index = output % P::K1;
    const int tensor_index = output / P::K1;
    float packed[5];
    float value[3];
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      packed[component] = probes[3 * P::K1 + component * P::K2 + tensor_index];
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      value[component] = probes[component * P::K1 + vector_index];
    }
    float product[3];
    matrix_vector(packed_to_stf(packed), value, product);
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputQuartic + output,
                     product[0] * product[0] + product[1] * product[1] +
                         product[2] * product[2]);
  }

  // === Step 6. Emit the spin invariants of even spin order ===
  if constexpr (HasSpin) {
    for (int pair = thread; pair < P::SpinGramVector; pair += P::Threads) {
      int row, column;
      decode_upper_pair(pair, P::SpinVectorWidth, row, column);
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        value =
            fmaf(spin_vector_value<Channels, Lmax>(moments, component, row),
                 spin_vector_value<Channels, Lmax>(moments, component, column),
                 value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinGramVector + pair,
                       (row == column ? 1.0f : kSqrtTwo) * value);
    }
    if (thread < P::SpinGramTensor) {
      int row, column;
      decode_spin_tensor_pair(thread, row, column);
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        value =
            fmaf(spin_tensor_value<Channels, Lmax>(moments, component, row),
                 spin_tensor_value<Channels, Lmax>(moments, component, column),
                 value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinGramTensor + thread,
                       (row == column ? 1.0f : kSqrtTwo) * value);
    }
    // Cross Gram against the unaligned geometric degree-two moments. Both
    // factors have even spin order, and with a unit direction the on-site row
    // evaluates to the single-ion anisotropy sum over neighbours.
    for (int output = thread; output < P::SpinCross; output += P::Threads) {
      const int probe = output / P::C2;
      const int channel = output % P::C2;
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        value =
            fmaf(spin_tensor_value<Channels, Lmax>(moments, component, probe),
                 moments[P::TensorOffset + component * P::C2 + channel], value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinCross + output,
                       value);
    }
    for (int channel = thread; channel < P::Cs; channel += P::Threads) {
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinMagnitude + channel,
                       moments[P::SpinMagnitude + channel]);
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth,
                       P::OutputSpinCoordination + channel,
                       moments[P::SpinCoordination + channel]);
    }
  }
}

// === Node readout backward ===

// The four symmetric 222 outputs are
//   k * {tr(Q0^3), sqrt(3) tr(Q0^2 Q1), sqrt(3) tr(Q0 Q1^2), tr(Q1^3)}.
// Their gradients need only Q0^2, Q1^2, and the symmetrized Q0 Q1 product, so
// evaluating this closed form inside the node backward avoids a second probe
// projection and the associated global gradient checkpoint.
template <int Channels, int Lmax, bool HasSpin>
__device__ __forceinline__ void add_bis222_probe_gradient(
    int lane,
    long node,
    const Arguments& args,
    const float* __restrict__ probes,
    float (&d_tensor)[5]) {
  using P = Profile<Channels, Lmax, HasSpin>;
  float packed_0[5];
  float packed_1[5];
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    packed_0[component] = probes[component * P::K2 + 0];
    packed_1[component] = probes[component * P::K2 + 1];
  }
  const Matrix3 matrix_0 = packed_to_stf(packed_0);
  const Matrix3 matrix_1 = packed_to_stf(packed_1);
  const float gradient_000 =
      kBis222Scale * load_output_gradient(args.descriptor_gradient,
                                          args.output_inv_std, node,
                                          P::OutputWidth, P::OutputBis222 + 0);
  const float gradient_001 =
      kBis222Scale * kSqrtThree *
      load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                           P::OutputWidth, P::OutputBis222 + 1);
  const float gradient_011 =
      kBis222Scale * kSqrtThree *
      load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                           P::OutputWidth, P::OutputBis222 + 2);
  const float gradient_111 =
      kBis222Scale * load_output_gradient(args.descriptor_gradient,
                                          args.output_inv_std, node,
                                          P::OutputWidth, P::OutputBis222 + 3);

  Matrix3 matrix_gradient{};
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      float product_00 = 0.0f;
      float product_11 = 0.0f;
      float product_01 = 0.0f;
      float product_10 = 0.0f;
#pragma unroll
      for (int inner = 0; inner < 3; ++inner) {
        product_00 = fmaf(matrix_0.value[row][inner],
                          matrix_0.value[inner][column], product_00);
        product_11 = fmaf(matrix_1.value[row][inner],
                          matrix_1.value[inner][column], product_11);
        product_01 = fmaf(matrix_0.value[row][inner],
                          matrix_1.value[inner][column], product_01);
        product_10 = fmaf(matrix_1.value[row][inner],
                          matrix_0.value[inner][column], product_10);
      }
      matrix_gradient.value[row][column] =
          lane == 0 ? 3.0f * gradient_000 * product_00 +
                          gradient_001 * (product_01 + product_10) +
                          gradient_011 * product_11
                    : gradient_001 * product_00 +
                          gradient_011 * (product_01 + product_10) +
                          3.0f * gradient_111 * product_11;
    }
  }
  float packed_gradient[5];
  matrix_gradient_to_packed(matrix_gradient, packed_gradient);
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    d_tensor[component] += packed_gradient[component];
  }
}

// Scatter the sparse-coupling VJP of one probe slot into a shared scratch row.
// The scratch is indexed by harmonic component, which a register array cannot
// address without spilling, and it is private to the lane, so the reduction
// stays deterministic.
template <int Channels, int Lmax, bool HasSpin>
__device__ __forceinline__ void accumulate_coupling_gradient(
    long node,
    const Arguments& args,
    const float* __restrict__ probes,
    const float* __restrict__ moments,
    int degree,
    int rank_index,
    float* __restrict__ scratch) {
  using P = Profile<Channels, Lmax, HasSpin>;
  for (int record = 0; record < args.coupling_count; ++record) {
    const int* meta = args.coupling_meta + record * 8;
    const int degrees[3] = {meta[0], meta[1], meta[2]};
    if (degrees[0] != degree && degrees[1] != degree && degrees[2] != degree) {
      continue;
    }
    const int nonzero_begin = meta[3];
    const int nonzero_count = meta[4];
    const int probe_begin = meta[5];
    const int probe_count = meta[6];
    const int coordinate = meta[7];
    for (int output = 0; output < probe_count; ++output) {
      const int selection = __ldg(args.coupling_entry + probe_begin + output);
      const int indices[3] = {selection & 0xFF, (selection >> 8) & 0xFF,
                              (selection >> 16) & 0xFF};
      const bool active[3] = {degrees[0] == degree && indices[0] == rank_index,
                              degrees[1] == degree && indices[1] == rank_index,
                              degrees[2] == degree && indices[2] == rank_index};
      if (!active[0] && !active[1] && !active[2]) {
        continue;
      }
      const float upstream =
          __ldg(args.coupling_value + probe_begin + output) *
          load_output_gradient(args.descriptor_gradient, args.output_inv_std,
                               node, P::OutputWidth, coordinate + output);
      for (int term = 0; term < nonzero_count; ++term) {
        const int components =
            __ldg(args.coupling_entry + nonzero_begin + term);
        const int component[3] = {components & 0xFF, (components >> 8) & 0xFF,
                                  (components >> 16) & 0xFF};
        const float weight =
            upstream * __ldg(args.coupling_value + nonzero_begin + term);
        const float first = probe_value<Channels, Lmax>(
            probes, moments, degrees[0], component[0], indices[0]);
        const float second = probe_value<Channels, Lmax>(
            probes, moments, degrees[1], component[1], indices[1]);
        const float third = probe_value<Channels, Lmax>(
            probes, moments, degrees[2], component[2], indices[2]);
        if (active[0]) {
          scratch[component[0]] += weight * second * third;
        }
        if (active[1]) {
          scratch[component[1]] += weight * first * third;
        }
        if (active[2]) {
          scratch[component[2]] += weight * first * second;
        }
      }
    }
  }
}

// Four independent lane groups share one warp. An incomplete final block
// aliases inactive groups to the last valid node so every thread reaches each
// block-wide barrier; stores from those groups are suppressed.
template <int Channels, int Lmax, bool HasSpin, int NodeLanes>
__global__ __launch_bounds__(
    Profile<Channels, Lmax, HasSpin, NodeLanes>::Threads,
    2) void node_backward_kernel(Arguments args) {
  using P = Profile<Channels, Lmax, HasSpin, NodeLanes>;
  constexpr int MaxComponents = 9;
  const int thread = threadIdx.x;
  const int group = thread / P::NodeWidth;
  const int lane = thread & (P::NodeWidth - 1);
  const long candidate = static_cast<long>(blockIdx.x) * P::NodeGroups + group;
  const bool active = candidate < args.node_count;
  const long node = active ? candidate : args.node_count - 1;
  const long state_offset = node * P::StateWidth;

  __shared__ float moments_storage[P::NodeGroups * P::MomentWidth];
  float* moments = moments_storage + group * P::MomentWidth;
  for (int coordinate = lane; coordinate < P::MomentWidth;
       coordinate += P::NodeWidth) {
    moments[coordinate] = __ldg(args.state + state_offset + coordinate);
  }
  const float scalar_norm = __ldg(args.state + state_offset + P::MomentWidth);
  const float angular_norm =
      __ldg(args.state + state_offset + P::MomentWidth + 1);
  __syncthreads();

  __shared__ float aligned_storage[P::NodeGroups * P::AlignedWidth];
  float* aligned = aligned_storage + group * P::AlignedWidth;
  for (int channel = lane; channel < P::C1; channel += P::NodeWidth) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(moments[P::VectorOffset + component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 0,
                                                    input, channel),
                     value);
      }
      aligned[component * P::C1 + channel] = value;
    }
  }
  if (lane < P::C2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(moments[P::TensorOffset + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 2,
                                                    input, lane),
                     value);
      }
      aligned[3 * P::C1 + component * P::C2 + lane] = value;
    }
  }
  __syncthreads();

  __shared__ float probes_storage[P::NodeGroups * P::ProbeWidth];
  float* probes = probes_storage + group * P::ProbeWidth;
  if (lane < P::K1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(aligned[component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 4,
                                                    input, lane),
                     value);
      }
      probes[component * P::K1 + lane] = value;
    }
  }
  if (lane < P::K2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(aligned[3 * P::C1 + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 6,
                                                    input, lane),
                     value);
      }
      probes[3 * P::K1 + component * P::K2 + lane] = value;
    }
  }
  __syncthreads();

  // Sparse-coupling scratch, one private row per lane.
  __shared__ float
      coupling_scratch[Lmax >= 3 ? P::NodeGroups * P::NodeWidth * MaxComponents
                                 : 1];
  float* scratch = nullptr;
  if constexpr (Lmax >= 3) {
    scratch = coupling_scratch + (group * P::NodeWidth + lane) * MaxComponents;
#pragma unroll
    for (int component = 0; component < MaxComponents; ++component) {
      scratch[component] = 0.0f;
    }
  }

  __shared__ float d_probes_storage[P::NodeGroups * P::ProbeWidth];
  float* d_probes = d_probes_storage + group * P::ProbeWidth;
  if (lane < P::K1) {
    float d_vector[3] = {};
    for (int output = 0; output < P::Bis112; ++output) {
      const int tensor_index = output % P::K2;
      int first, second;
      decode_upper_pair(output / P::K2, P::K1, first, second);
      if (lane != first && lane != second) {
        continue;
      }
      float packed[5];
      float other[3];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] =
            probes[3 * P::K1 + component * P::K2 + tensor_index];
      }
      const int other_index = lane == first ? second : first;
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        other[component] = probes[component * P::K1 + other_index];
      }
      const float gradient =
          -kInvSqrtFive * (first == second ? 2.0f : kSqrtTwo) *
          load_output_gradient(args.descriptor_gradient, args.output_inv_std,
                               node, P::OutputWidth, P::OutputBis112 + output);
      float product[3];
      matrix_vector(packed_to_stf(packed), other, product);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_vector[component] =
            fmaf(gradient, product[component], d_vector[component]);
      }
    }
    for (int tensor_index = 0; tensor_index < P::K2; ++tensor_index) {
      const float gradient = load_output_gradient(
          args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
          P::OutputQuartic + tensor_index * P::K1 + lane);
      float packed[5];
      float value[3];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] =
            probes[3 * P::K1 + component * P::K2 + tensor_index];
      }
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        value[component] = probes[component * P::K1 + lane];
      }
      const Matrix3 matrix = packed_to_stf(packed);
      float first[3];
      float second[3];
      matrix_vector(matrix, value, first);
      matrix_vector(matrix, first, second);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_vector[component] =
            fmaf(2.0f * gradient, second[component], d_vector[component]);
      }
    }
    if constexpr (Lmax >= 3) {
      accumulate_coupling_gradient<Channels, Lmax, HasSpin>(
          node, args, probes, moments, 1, lane, scratch);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_vector[component] += scratch[component];
        scratch[component] = 0.0f;
      }
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      d_probes[component * P::K1 + lane] = d_vector[component];
    }
  }

  if (lane < P::K2) {
    float d_tensor[5] = {};
    add_bis222_probe_gradient<Channels, Lmax, HasSpin>(
        lane, node, args, probes + 3 * P::K1, d_tensor);
    for (int output = lane; output < P::Bis112; output += P::K2) {
      int first, second;
      decode_upper_pair(output / P::K2, P::K1, first, second);
      float left[3];
      float right[3];
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        left[component] = probes[component * P::K1 + first];
        right[component] = probes[component * P::K1 + second];
      }
      const float gradient =
          -kInvSqrtFive * (first == second ? 1.0f : kSqrtTwo) *
          load_output_gradient(args.descriptor_gradient, args.output_inv_std,
                               node, P::OutputWidth, P::OutputBis112 + output);
      Matrix3 matrix{};
#pragma unroll
      for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int column = 0; column < 3; ++column) {
          matrix.value[row][column] = gradient * left[row] * right[column];
        }
      }
      float packed_gradient[5];
      matrix_gradient_to_packed(matrix, packed_gradient);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_tensor[component] += packed_gradient[component];
      }
    }
    for (int vector_index = 0; vector_index < P::K1; ++vector_index) {
      const float gradient = load_output_gradient(
          args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
          P::OutputQuartic + lane * P::K1 + vector_index);
      float packed[5];
      float value[3];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] = probes[3 * P::K1 + component * P::K2 + lane];
      }
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        value[component] = probes[component * P::K1 + vector_index];
      }
      float product[3];
      matrix_vector(packed_to_stf(packed), value, product);
      Matrix3 matrix{};
#pragma unroll
      for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int column = 0; column < 3; ++column) {
          matrix.value[row][column] =
              2.0f * gradient * product[row] * value[column];
        }
      }
      float packed_gradient[5];
      matrix_gradient_to_packed(matrix, packed_gradient);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_tensor[component] += packed_gradient[component];
      }
    }
    if constexpr (Lmax >= 3) {
      accumulate_coupling_gradient<Channels, Lmax, HasSpin>(
          node, args, probes, moments, 2, lane, scratch);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_tensor[component] += scratch[component];
        scratch[component] = 0.0f;
      }
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      d_probes[3 * P::K1 + component * P::K2 + lane] = d_tensor[component];
    }
  }
  __syncthreads();

  __shared__ float d_aligned_storage[P::NodeGroups * P::AlignedWidth];
  float* d_aligned = d_aligned_storage + group * P::AlignedWidth;
  for (int channel = lane; channel < P::C1; channel += P::NodeWidth) {
    float gradient[3] = {};
    for (int other = 0; other < P::C1; ++other) {
      const float upstream =
          (channel == other ? 2.0f : kSqrtTwo) *
          load_output_gradient(
              args.descriptor_gradient, args.output_inv_std, node,
              P::OutputWidth,
              P::OutputGram1 + gram_pair_position(channel, other, P::C1));
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        gradient[component] = fmaf(upstream, aligned[component * P::C1 + other],
                                   gradient[component]);
      }
    }
    for (int probe = 0; probe < P::K1; ++probe) {
      const float weight = readout_weight<Channels, Lmax>(args.readout_matrices,
                                                          5, probe, channel);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        gradient[component] = fmaf(weight, d_probes[component * P::K1 + probe],
                                   gradient[component]);
      }
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      d_aligned[component * P::C1 + channel] = gradient[component];
    }
  }
  if (lane < P::C2) {
    float gradient[5] = {};
    for (int other = 0; other < P::C2; ++other) {
      const float upstream =
          (lane == other ? 2.0f : kSqrtTwo) *
          load_output_gradient(
              args.descriptor_gradient, args.output_inv_std, node,
              P::OutputWidth,
              P::OutputGram2 + gram_pair_position(lane, other, P::C2));
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        gradient[component] =
            fmaf(upstream, aligned[3 * P::C1 + component * P::C2 + other],
                 gradient[component]);
      }
    }
    for (int probe = 0; probe < P::K2; ++probe) {
      const float weight =
          readout_weight<Channels, Lmax>(args.readout_matrices, 7, probe, lane);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        gradient[component] =
            fmaf(weight, d_probes[3 * P::K1 + component * P::K2 + probe],
                 gradient[component]);
      }
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      d_aligned[3 * P::C1 + component * P::C2 + lane] = gradient[component];
    }
  }
  __syncthreads();

  __shared__ float d_moments_storage[P::NodeGroups * P::MomentWidth];
  float* d_moments = d_moments_storage + group * P::MomentWidth;
  for (int channel = lane; channel < Channels; channel += P::NodeWidth) {
    d_moments[P::ScalarOffset + channel] =
        load_output_gradient(args.descriptor_gradient, args.output_inv_std,
                             node, P::OutputWidth, P::OutputScalar + channel);
  }
  for (int channel = lane; channel < P::C1; channel += P::NodeWidth) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int output = 0; output < P::C1; ++output) {
        value = fmaf(d_aligned[component * P::C1 + output],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 1,
                                                    output, channel),
                     value);
      }
      d_moments[P::VectorOffset + component * P::C1 + channel] = value;
    }
  }
  if (lane < P::C2) {
    // The spin cross Gram contracts the spin quadrupoles against the
    // unaligned degree-two moments, so its cotangent joins this channel here
    // rather than passing through the alignment transpose above.
    float cross[HasSpin ? 2 : 1];
    if constexpr (HasSpin) {
#pragma unroll
      for (int probe = 0; probe < 2; ++probe) {
        cross[probe] = load_output_gradient(
            args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
            P::OutputSpinCross + probe * P::C2 + lane);
      }
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int output = 0; output < P::C2; ++output) {
        value = fmaf(d_aligned[3 * P::C1 + component * P::C2 + output],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 3,
                                                    output, lane),
                     value);
      }
      if constexpr (HasSpin) {
#pragma unroll
        for (int probe = 0; probe < 2; ++probe) {
          value =
              fmaf(cross[probe],
                   spin_tensor_value<Channels, Lmax>(moments, component, probe),
                   value);
        }
      }
      d_moments[P::TensorOffset + component * P::C2 + lane] = value;
    }
  }
  if constexpr (Lmax >= 3) {
    if (lane < Lmax - 2) {
      const int degree = 3 + lane;
      const int offset = P::HighOffset + (lane == 0 ? 0 : P::High3);
      const int count = 2 * degree + 1;
      const float gram =
          2.0f * load_output_gradient(args.descriptor_gradient,
                                      args.output_inv_std, node, P::OutputWidth,
                                      P::OutputGram3 + lane);
      accumulate_coupling_gradient<Channels, Lmax, HasSpin>(
          node, args, probes, moments, degree, 0, scratch);
      for (int component = 0; component < count; ++component) {
        d_moments[offset + component] =
            fmaf(gram, moments[offset + component], scratch[component]);
      }
    }
  }

  // === Spin readout VJP and on-site magnetic gradient ===
  if constexpr (HasSpin) {
    float onsite_vector[3] = {0.0f, 0.0f, 0.0f};
    float onsite_tensor[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int channel = lane; channel < P::SpinVectorWidth;
         channel += P::NodeWidth) {
      float gradient[3] = {0.0f, 0.0f, 0.0f};
      for (int other = 0; other < P::SpinVectorWidth; ++other) {
        const float upstream =
            (channel == other ? 2.0f : kSqrtTwo) *
            load_output_gradient(
                args.descriptor_gradient, args.output_inv_std, node,
                P::OutputWidth,
                P::OutputSpinGramVector +
                    gram_pair_position(channel, other, P::SpinVectorWidth));
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          gradient[component] =
              fmaf(upstream,
                   spin_vector_value<Channels, Lmax>(moments, component, other),
                   gradient[component]);
        }
      }
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        if (channel == 0) {
          // The on-site family carries no normalizer, so its cotangent leaves
          // through the magnetic gradient below and must not reach the edge
          // scan or the mass VJP.
          onsite_vector[component] = gradient[component];
          d_moments[P::SpinOnsiteVector + component] = 0.0f;
        } else {
          d_moments[spin_vector_offset<Channels, Lmax>(component, channel)] =
              gradient[component];
        }
      }
    }
    if (lane < 2) {
      float gradient[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int other = 0; other < 2; ++other) {
        // The on-site self-term is not emitted, so it contributes nothing.
        if (lane == 0 && other == 0) {
          continue;
        }
        const float upstream =
            (lane == other ? 2.0f : kSqrtTwo) *
            load_output_gradient(
                args.descriptor_gradient, args.output_inv_std, node,
                P::OutputWidth,
                P::OutputSpinGramTensor + (lane == 1 && other == 1 ? 1 : 0));
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          gradient[component] =
              fmaf(upstream,
                   spin_tensor_value<Channels, Lmax>(moments, component, other),
                   gradient[component]);
        }
      }
      for (int channel = 0; channel < P::C2; ++channel) {
        const float upstream = load_output_gradient(
            args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
            P::OutputSpinCross + lane * P::C2 + channel);
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          gradient[component] = fmaf(
              upstream, moments[P::TensorOffset + component * P::C2 + channel],
              gradient[component]);
        }
      }
      if (lane == 0) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          onsite_tensor[component] = gradient[component];
          d_moments[P::SpinOnsiteTensor + component] = 0.0f;
        }
      } else {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          d_moments[P::SpinTensor + component] = gradient[component];
        }
      }
    }
    for (int channel = lane; channel < P::Cs; channel += P::NodeWidth) {
      d_moments[P::SpinMagnitude + channel] = load_output_gradient(
          args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
          P::OutputSpinMagnitude + channel);
      d_moments[P::SpinCoordination + channel] = load_output_gradient(
          args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
          P::OutputSpinCoordination + channel);
    }
    // Lane zero owns both on-site cotangents, so the magnetic gradient of the
    // centre closes here without a shuffle.
    if (active && lane == 0) {
      const int center_type =
          static_cast<int>(args.atype[args.node_begin + node]);
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, center_type);
      float center_spin[3];
      load_conditioned_spin(args.spin, weights.scale, args.node_begin + node,
                            center_spin);
      float quadrupole[3];
      spin_quadrupole_vjp(onsite_tensor, 1.0f, center_spin, quadrupole);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        args.spin_gradient[node * 3 + component] =
            weights.scale * fmaf(weights.quadrupole, quadrupole[component],
                                 weights.vector * onsite_vector[component]);
      }
    }
  }
  __syncthreads();

  // === Normalizer VJPs ===
  // The scalar and non-scalar blocks carry independent envelope masses, so
  // each contributes its own smooth normalizer cotangent.
  float scalar_dot = 0.0f;
  float angular_dot = 0.0f;
  for (int coordinate = lane; coordinate < P::MomentWidth;
       coordinate += P::NodeWidth) {
    const float product = d_moments[coordinate] * moments[coordinate];
    if (coordinate < Channels) {
      scalar_dot += product;
    } else {
      angular_dot += product;
    }
  }
  const unsigned node_mask = subwarp_mask<P::NodeWidth>(group * P::NodeWidth);
  scalar_dot = subwarp_sum<P::NodeWidth>(scalar_dot, node_mask);
  angular_dot = subwarp_sum<P::NodeWidth>(angular_dot, node_mask);
  // Each mass reaches the output twice: through the moments it normalizes,
  // whose cotangent carries the factor -n^2/2, and through its own divisor
  // sqrt(mass + floor), whose derivative is n/2.
  const float scalar_mass_gradient =
      0.5f * scalar_norm *
      (load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                            P::OutputWidth, P::OutputDivisor + 0) -
       scalar_dot * scalar_norm);
  const float angular_mass_gradient =
      0.5f * angular_norm *
      (load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                            P::OutputWidth, P::OutputDivisor + 1) -
       angular_dot * angular_norm);
  for (int coordinate = lane; coordinate < P::MomentWidth;
       coordinate += P::NodeWidth) {
    const float value = d_moments[coordinate] *
                        (coordinate < Channels ? scalar_norm : angular_norm);
    if (active) {
      args.moment_gradient[node * P::StateWidth + coordinate] = value;
    }
  }
  if (active && lane == 0) {
    args.moment_gradient[node * P::StateWidth + P::MomentWidth] =
        scalar_mass_gradient;
    args.moment_gradient[node * P::StateWidth + P::MomentWidth + 1] =
        angular_mass_gradient;
  }
}

// === Edge recomputation backward ===

template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          typename index_t>
__global__ __launch_bounds__(
    Profile<Channels, Lmax, HasSpin>::Threads,
    Profile<Channels, Lmax, HasSpin>::
        MinBlocksPerSm) void edge_backward_kernel(Arguments args) {
  using P = Profile<Channels, Lmax, HasSpin>;
  constexpr int EdgeWidth = P::BackwardEdgeWidth;
  constexpr int Groups = kWarpSize / EdgeWidth;
  constexpr int ChannelTiles = Channels / EdgeWidth;
  constexpr int AngularTiles = (P::C1 + EdgeWidth - 1) / EdgeWidth;
  constexpr int TensorTiles = (P::C2 + EdgeWidth - 1) / EdgeWidth;
  constexpr int HighTiles = (P::HighCount + EdgeWidth - 1) / EdgeWidth;

  const int thread = threadIdx.x;
  const long node = blockIdx.x;
  if (node >= args.node_count) {
    return;
  }
  const int base_channel = thread & (EdgeWidth - 1);
  const int group = thread / EdgeWidth;
  const int leader = group * EdgeWidth;
  const unsigned mask = subwarp_mask<EdgeWidth>(leader);
  const auto* edge_index = static_cast<const index_t*>(args.edge_index);
  const auto* destination_order =
      static_cast<const index_t*>(args.destination_order);
  const int center_type = static_cast<int>(args.atype[args.node_begin + node]);
  const long begin = args.destination_row_ptr[node];
  const long end = args.destination_row_ptr[node + 1];
  const int radial_modes = HasModes ? args.radial_modes : 0;
  const long gradient_offset = node * P::StateWidth;
  const float scalar_mass_gradient =
      __ldg(args.moment_gradient + gradient_offset + P::MomentWidth);
  const float angular_mass_gradient =
      __ldg(args.moment_gradient + gradient_offset + P::MomentWidth + 1);

  // The scalar cotangent is one vector of width C0 that every edge rereads.
  // Holding it in registers costs one entry per channel tile and spills the
  // widest profiles; shared memory serves it as a conflict-free broadcast
  // because concurrent edge groups address identical channels.
  __shared__ float scalar_gradient[Channels];
  for (int channel = thread; channel < Channels; channel += P::Threads) {
    scalar_gradient[channel] = __ldg(args.moment_gradient + gradient_offset +
                                     P::ScalarOffset + channel);
  }
  // The reduced spin cotangent is a node constant that every edge rereads and
  // that every lane of a group needs in full, so it is staged in shared
  // memory for the same reason as the scalar cotangent above.
  __shared__ float spin_gradient[HasSpin ? P::SpinEdgeWidth : 1];
  if constexpr (HasSpin) {
    for (int coordinate = thread; coordinate < P::SpinEdgeWidth;
         coordinate += P::Threads) {
      spin_gradient[coordinate] = __ldg(args.moment_gradient + gradient_offset +
                                        P::SpinOffset + coordinate);
    }
  }
  float d_vector[AngularTiles][3] = {};
  float d_tensor[TensorTiles][5] = {};
  float d_high[HighTiles > 0 ? HighTiles : 1] = {};
#pragma unroll
  for (int tile = 0; tile < AngularTiles; ++tile) {
    const int channel = base_channel + tile * EdgeWidth;
    if (channel < P::C1) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_vector[tile][component] =
            __ldg(args.moment_gradient + gradient_offset + P::VectorOffset +
                  component * P::C1 + channel);
      }
    }
  }
#pragma unroll
  for (int tile = 0; tile < TensorTiles; ++tile) {
    const int channel = base_channel + tile * EdgeWidth;
    if (channel < P::C2) {
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_tensor[tile][component] =
            __ldg(args.moment_gradient + gradient_offset + P::TensorOffset +
                  component * P::C2 + channel);
      }
    }
  }
  if constexpr (Lmax >= 3) {
#pragma unroll
    for (int component = 0; component < P::HighCount; ++component) {
      if (component % EdgeWidth == base_channel) {
        d_high[component / EdgeWidth] = __ldg(
            args.moment_gradient + gradient_offset + P::HighOffset + component);
      }
    }
  }

  // Explicitly aligned: the mode residual reads these rows as float4, and
  // a preceding shared array of arbitrary length would otherwise leave the
  // block on a four-byte boundary.
  __shared__ __align__(
      16) float mode_cache[HasModes ? Groups * kModeStride : 1];
  __shared__ __align__(
      16) float mode_derivative_cache[HasModes ? Groups * kModeStride : 1];
  const int mode_offset = HasModes ? group * kModeStride : 0;
  float* modes = mode_cache + mode_offset;
  float* mode_derivatives = mode_derivative_cache + mode_offset;
  __syncthreads();

  // Offsets of the five spin families inside the staged cotangent row.
  constexpr int SpinMagnitudeSlot = P::SpinMagnitude - P::SpinOffset;
  constexpr int SpinCoordinationSlot = P::SpinCoordination - P::SpinOffset;
  constexpr int SpinVectorSlot = P::SpinVector - P::SpinOffset;
  constexpr int SpinBondSlot = P::SpinBond - P::SpinOffset;
  constexpr int SpinTensorSlot = P::SpinTensor - P::SpinOffset;

  for (long position = begin + group; position < end; position += Groups) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (args.edge_mask != nullptr && !args.edge_mask[edge]) {
      if (thread == leader) {
        args.edge_gradient[edge * 3 + 0] = 0.0f;
        args.edge_gradient[edge * 3 + 1] = 0.0f;
        args.edge_gradient[edge * 3 + 2] = 0.0f;
        if constexpr (HasSpin) {
          args.edge_spin_gradient[edge * 3 + 0] = 0.0f;
          args.edge_spin_gradient[edge * 3 + 1] = 0.0f;
          args.edge_spin_gradient[edge * 3 + 2] = 0.0f;
        }
      }
      continue;
    }
    const EdgeGeometry geometry = load_geometry<Canonical, index_t>(
        edge, args.rcut, args.eps, args.edge_vec, edge_index, args.atype);
    const TableLocation location =
        locate_table<Canonical>(geometry.radius, args.table_stride,
                                args.table_max, args.interval_count);
    if constexpr (!Canonical) {
      if (center_type >= args.type_count - 1 ||
          geometry.source_type >= args.type_count - 1) {
        if (thread == leader) {
          args.edge_gradient[edge * 3 + 0] = 0.0f;
          args.edge_gradient[edge * 3 + 1] = 0.0f;
          args.edge_gradient[edge * 3 + 2] = 0.0f;
          if constexpr (HasSpin) {
            args.edge_spin_gradient[edge * 3 + 0] = 0.0f;
            args.edge_spin_gradient[edge * 3 + 1] = 0.0f;
            args.edge_spin_gradient[edge * 3 + 2] = 0.0f;
          }
        }
        continue;
      }
    }
    const TableRow row = table_row(args.table, location, args.table_width);
    const float coordinate = location.coordinate;
    const bool clamped = location.clamped;
    if constexpr (HasModes) {
      __syncwarp(mask);
      for (int mode = base_channel; mode < radial_modes; mode += EdgeWidth) {
        const float2 value = evaluate_table_with_derivative(
            row, Channels + mode, coordinate, clamped);
        modes[mode] = value.x;
        mode_derivatives[mode] = value.y;
      }
      __syncwarp(mask);
    }

    float basis[9];
    fill_angular_basis(geometry, basis);
    const float envelope = geometry.envelope;
    const long pair =
        static_cast<long>(center_type) * args.type_count + geometry.source_type;
    const float2* film_row =
        reinterpret_cast<const float2*>(args.pair_film + pair * Channels * 2) +
        base_channel;
    const float* mixing_row =
        HasModes
            ? args.pair_mixing + (pair * Channels + base_channel) * radial_modes
            : nullptr;

    // The high-degree cotangent enters the leading channel only, so its
    // angular contraction is reduced onto the lane that owns that channel.
    float high_angular = 0.0f;
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          high_angular =
              fmaf(d_high[component / EdgeWidth],
                   high_basis_value(geometry, component), high_angular);
        }
      }
      high_angular = subwarp_sum<EdgeWidth>(high_angular, mask);
    }

    // The spin payload of this edge, hoisted like the forward one: every
    // channel multiplies the same conditioned neighbour moment and the same
    // projection of that moment onto the bond.
    float neighbor_spin[3] = {0.0f, 0.0f, 0.0f};
    float neighbor_magnitude = 0.0f;
    float neighbor_gate = 0.0f;
    float neighbor_scale = 0.0f;
    float bond_alignment = 0.0f;
    float spin_cotangent[3] = {0.0f, 0.0f, 0.0f};
    const float2* spin_row = nullptr;
    if constexpr (HasSpin) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, geometry.source_type);
      load_conditioned_spin(args.spin, weights.scale, geometry.source,
                            neighbor_spin);
      neighbor_magnitude = neighbor_spin[0] * neighbor_spin[0] +
                           neighbor_spin[1] * neighbor_spin[1] +
                           neighbor_spin[2] * neighbor_spin[2];
      neighbor_gate = weights.gate;
      neighbor_scale = weights.scale;
      bond_alignment = neighbor_spin[0] * basis[1] +
                       neighbor_spin[1] * basis[2] +
                       neighbor_spin[2] * basis[3];
      spin_row =
          reinterpret_cast<const float2*>(args.spin_pair + pair * P::Cs * 2) +
          base_channel;
    }

    float radial_gradient = 0.0f;
    float envelope_gradient = 0.0f;
    float d_basis[9] = {};
    // Direction cotangent of every family that evaluates its own angular chain
    // rule: the single-channel high degrees, which fill it after the channel
    // loop, and the bond-projected spin channels, which fill it inside. Both
    // reach the coordinate gradient through the one transverse projection in
    // `basis_vjp`, so no family is reduced separately.
    float direction_gradient[3] = {0.0f, 0.0f, 0.0f};
    float angular_zero = 0.0f;
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      const float2 radial =
          evaluate_table_with_derivative(row, channel, coordinate, clamped);
      const float2 film = __ldg(film_row + tile * EdgeWidth);
      // ``film_value`` is the pre-envelope FiLM amplitude phi; the reduced
      // payload is phi * chi for degree zero and phi * chi^2 above it.
      float film_value = fmaf(film.x, radial.x, film.y);
      float film_derivative = film.x * radial.y;
      if constexpr (HasModes) {
        accumulate_modes_with_derivative(
            mixing_row + tile * (EdgeWidth * radial_modes), modes,
            mode_derivatives, radial_modes, film_value, film_derivative);
      }
      float angular = 0.0f;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          angular =
              fmaf(d_vector[tile][component], basis[1 + component], angular);
        }
      }
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          angular =
              fmaf(d_tensor[tile][component], basis[4 + component], angular);
        }
      }
      if constexpr (Lmax >= 3) {
        if (channel == 0) {
          angular += high_angular;
        }
      }
      // d/dphi   = chi * (p + chi * a)
      // d/dchi   = phi * (p + 2 * chi * a)
      const float scaled = envelope * angular;
      const float film_gradient = scalar_gradient[channel] + scaled;
      radial_gradient =
          fmaf(envelope * film_gradient, film_derivative, radial_gradient);
      envelope_gradient =
          fmaf(film_gradient + scaled, film_value, envelope_gradient);
      const float angular_payload = film_value * envelope * envelope;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          d_basis[1 + component] =
              fmaf(d_vector[tile][component], angular_payload,
                   d_basis[1 + component]);
        }
      }
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          d_basis[4 + component] =
              fmaf(d_tensor[tile][component], angular_payload,
                   d_basis[4 + component]);
        }
      }
      if constexpr (HasSpin) {
        if (channel < P::Cs) {
          const float2 spin_film = __ldg(spin_row + tile * EdgeWidth);
          const float profile = fmaf(spin_film.x, radial.x, spin_film.y);
          const float envelope_squared = envelope * envelope;
          const float weight = profile * envelope_squared;
          const float magnitude_gradient =
              spin_gradient[SpinMagnitudeSlot + channel];
          // The two degree-one cotangents are read once and used by the
          // amplitude, the magnetic and the angular accumulations below.
          float vector_gradient[3];
          float bond_gradient[3];
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            vector_gradient[component] =
                spin_gradient[SpinVectorSlot + component * P::Cs + channel];
            bond_gradient[component] =
                spin_gradient[SpinBondSlot + component * P::Cs + channel];
          }
          // Component of the bond cotangent along the bond. The bond-projected
          // family is `P = w (s . u) u`, so this one dot product carries all
          // three of its VJPs: `dP/dw` contracts to `(s . u) (dP . u)`,
          // `dP/ds` to `w (dP . u) u`, and the direction term below reuses it
          // once more.
          float bond_projection = 0.0f;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            bond_projection = fmaf(bond_gradient[component],
                                   basis[1 + component], bond_projection);
          }
          float sigma = fmaf(
              magnitude_gradient, neighbor_magnitude,
              spin_gradient[SpinCoordinationSlot + channel] * neighbor_gate);
          sigma = fmaf(bond_alignment, bond_projection, sigma);
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            sigma = fmaf(vector_gradient[component], neighbor_spin[component],
                         sigma);
          }
          // Magnetic cotangent of the source moment. The coordination family
          // reads the type gate rather than the moment, so it contributes
          // nothing here.
          const float linear = 2.0f * weight * magnitude_gradient;
          const float projected = weight * bond_projection;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            spin_cotangent[component] =
                fmaf(linear, neighbor_spin[component],
                     fmaf(projected, basis[1 + component],
                          fmaf(weight, vector_gradient[component],
                               spin_cotangent[component])));
          }
          // Angular cotangent of the bond-projected family. Differentiating
          // `(s . u) u` with respect to `u` at fixed `s` gives the rank-one
          // sum `s (dP . u) + (s . u) dP`, which the transverse projection in
          // `basis_vjp` turns into the coordinate gradient. Every other spin
          // family is independent of the direction and contributes nothing.
          const float aligned_weight = weight * bond_alignment;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            direction_gradient[component] =
                fmaf(projected, neighbor_spin[component],
                     fmaf(aligned_weight, bond_gradient[component],
                          direction_gradient[component]));
          }
          if (channel == 0) {
            // The single neighbour quadrupole rides the leading spin channel.
            // Both of its contributions are formed here rather than hoisted,
            // so its five cotangent components never stay live across the
            // channel loop.
#pragma unroll
            for (int component = 0; component < 5; ++component) {
              sigma = fmaf(
                  spin_gradient[SpinTensorSlot + component],
                  spin_quadrupole_component(neighbor_spin[0], neighbor_spin[1],
                                            neighbor_spin[2], component),
                  sigma);
            }
            float quadrupole[3];
            spin_quadrupole_vjp(spin_gradient + SpinTensorSlot, weight,
                                neighbor_spin, quadrupole);
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              spin_cotangent[component] += quadrupole[component];
            }
          }
          // w = chi^2 (gamma g + beta): the distance enters through the table
          // and through the envelope, and `sigma` is the cotangent of that
          // shared amplitude over all five families.
          radial_gradient = fmaf(envelope_squared * sigma,
                                 spin_film.x * radial.y, radial_gradient);
          envelope_gradient =
              fmaf(2.0f * envelope * profile, sigma, envelope_gradient);
        }
      }
      if constexpr (Lmax >= 3) {
        if (channel == 0) {
          angular_zero = angular_payload;
        }
      }
    }

    if constexpr (Lmax >= 3) {
      const float amplitude = __shfl_sync(mask, angular_zero, leader);
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          high_basis_gradient(geometry, component,
                              d_high[component / EdgeWidth] * amplitude,
                              direction_gradient);
        }
      }
    }

    // Both envelope masses contribute once per edge, so their cotangent joins
    // the single lane that also owns the leading channel.
    if (thread == leader) {
      const float squared = envelope * envelope;
      envelope_gradient =
          fmaf(4.0f * squared * envelope, angular_mass_gradient,
               fmaf(2.0f * envelope, scalar_mass_gradient, envelope_gradient));
    }
    radial_gradient = fmaf(envelope_gradient,
                           c3_envelope_derivative(geometry.radius, args.rcut),
                           radial_gradient);

    // The basis VJP is linear in the radial and angular cotangents, so
    // applying it per lane reduces three Cartesian components instead of the
    // full set of angular components across the edge group.
    float output[3];
    basis_vjp(geometry, d_basis, direction_gradient, radial_gradient, output);
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      output[component] = subwarp_sum<EdgeWidth>(output[component], mask);
    }
    if (thread == leader) {
      args.edge_gradient[edge * 3 + 0] = output[0];
      args.edge_gradient[edge * 3 + 1] = output[1];
      args.edge_gradient[edge * 3 + 2] = output[2];
    }
    if constexpr (HasSpin) {
      // The magnetic cotangent belongs to the source node, which this
      // destination-major scan does not own. It is emitted per edge and
      // reduced onto sources by the shared edge assembly, which already walks
      // the source CSR for the conservative force.
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        spin_cotangent[component] =
            subwarp_sum<EdgeWidth>(spin_cotangent[component], mask);
      }
      if (thread == leader) {
        // Every contribution above differentiates the conditioned moment, so
        // the store is the single point that applies the remaining chain
        // factor and hands back a gradient of the raw input moment.
        args.edge_spin_gradient[edge * 3 + 0] =
            neighbor_scale * spin_cotangent[0];
        args.edge_spin_gradient[edge * 3 + 1] =
            neighbor_scale * spin_cotangent[1];
        args.edge_spin_gradient[edge * 3 + 2] =
            neighbor_scale * spin_cotangent[2];
      }
    }
  }
}

template <bool Canonical, typename index_t>
__global__ void zero_padding_kernel(long node_count,
                                    long edge_count,
                                    const index_t* destination_order,
                                    const long* destination_row_ptr,
                                    float* edge_gradient,
                                    float* edge_spin_gradient) {
  const long valid_edge_count = destination_row_ptr[node_count];
  for (long position = valid_edge_count + blockIdx.x * blockDim.x + threadIdx.x;
       position < edge_count;
       position += static_cast<long>(blockDim.x) * gridDim.x) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    edge_gradient[edge * 3 + 0] = 0.0f;
    edge_gradient[edge * 3 + 1] = 0.0f;
    edge_gradient[edge * 3 + 2] = 0.0f;
    if (edge_spin_gradient != nullptr) {
      edge_spin_gradient[edge * 3 + 0] = 0.0f;
      edge_spin_gradient[edge * 3 + 1] = 0.0f;
      edge_spin_gradient[edge * 3 + 2] = 0.0f;
    }
  }
}

// === Launch dispatch ===

template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          typename index_t>
struct ForwardLauncher {
  static void run(const Arguments& args, cudaStream_t stream) {
    using P = Profile<Channels, Lmax, HasSpin>;
    forward_kernel<Channels, Lmax, HasModes, HasSpin, Canonical, index_t>
        <<<static_cast<int>(args.node_count), P::Threads, 0, stream>>>(args);
  }
};

template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          typename index_t>
struct BackwardLauncher {
  // Pick the node-group width from what the running device grants the two
  // compiled kernels rather than from its architecture: the six shared arrays
  // of the node backward scale with the group count and the scalar width, so
  // the same source is shared-memory bound on a part with a 100 KB budget per
  // multiprocessor and register bound on one with more than twice that.
  //
  // Widening the group halves that footprint and so buys resident warps, but
  // every lane of a group reloads its node's shared state, so it also doubles
  // the redundant load work. Measured across the scalar widths, the trade pays
  // only where the narrow variant is starved of warps outright: at a quarter of
  // the device's warp capacity the wide variant wins by 2.1 % of the whole
  // backward, at a third the two are within noise, and above that the narrow
  // variant wins by up to 3.6 %. The threshold is therefore stated against the
  // device's own warp capacity, and the query is made once per configuration.
  static bool prefer_wide_nodes() {
    static const bool wide = [] {
      using Narrow = Profile<Channels, Lmax, HasSpin, kNodeLanesNarrow>;
      using Wide = Profile<Channels, Lmax, HasSpin, kNodeLanesWide>;
      int narrow_blocks = 0;
      int wide_blocks = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &narrow_blocks,
          node_backward_kernel<Channels, Lmax, HasSpin, kNodeLanesNarrow>,
          Narrow::Threads, 0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &wide_blocks,
          node_backward_kernel<Channels, Lmax, HasSpin, kNodeLanesWide>,
          Wide::Threads, 0);
      int device = 0;
      cudaGetDevice(&device);
      int threads_per_sm = 0;
      cudaDeviceGetAttribute(&threads_per_sm,
                             cudaDevAttrMaxThreadsPerMultiProcessor, device);
      const int warps_per_sm = threads_per_sm / kWarpSize;
      const int narrow_warps = narrow_blocks * (Narrow::Threads / kWarpSize);
      return wide_blocks > narrow_blocks && narrow_warps * 4 < warps_per_sm;
    }();
    return wide;
  }

  template <int NodeLanes>
  static void launch_node_backward(const Arguments& args, cudaStream_t stream) {
    using NodeProfile = Profile<Channels, Lmax, HasSpin, NodeLanes>;
    const int node_blocks =
        static_cast<int>((args.node_count + NodeProfile::NodeGroups - 1) /
                         NodeProfile::NodeGroups);
    node_backward_kernel<Channels, Lmax, HasSpin, NodeLanes>
        <<<node_blocks, NodeProfile::Threads, 0, stream>>>(args);
  }

  static void run(const Arguments& args, cudaStream_t stream) {
    using P = Profile<Channels, Lmax, HasSpin>;
    if (prefer_wide_nodes()) {
      launch_node_backward<kNodeLanesWide>(args, stream);
    } else {
      launch_node_backward<kNodeLanesNarrow>(args, stream);
    }
    edge_backward_kernel<Channels, Lmax, HasModes, HasSpin, Canonical, index_t>
        <<<static_cast<int>(args.node_count), P::Threads, 0, stream>>>(args);
    // The reserved edge slots beyond the physical count are only known on the
    // device, so the grid is sized from the storage bound and the surplus
    // blocks retire immediately.
    if (args.clear_padding) {
      constexpr int kPaddingThreads = 128;
      constexpr long kPaddingBlockLimit = 1024;
      const long padding_blocks = min(
          kPaddingBlockLimit,
          max(1L, (args.edge_count + kPaddingThreads - 1) / kPaddingThreads));
      zero_padding_kernel<Canonical, index_t>
          <<<static_cast<int>(padding_blocks), kPaddingThreads, 0, stream>>>(
              args.node_count, args.edge_count,
              static_cast<const index_t*>(args.destination_order),
              args.destination_row_ptr, args.edge_gradient,
              HasSpin ? args.edge_spin_gradient : nullptr);
    }
  }
};

// A signed and an unsigned 32-bit index share their representation over the
// non-negative range that node and edge identifiers occupy, so the topology
// collapses to two element widths.
template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          template <int, int, bool, bool, bool, typename> class L>
void dispatch_topology(const Arguments& args, cudaStream_t stream) {
  const bool wide = args.index_kind == IndexKind::Bits64;
  if (args.canonical) {
    if (wide) {
      L<Channels, Lmax, HasModes, HasSpin, true, long>::run(args, stream);
    } else {
      L<Channels, Lmax, HasModes, HasSpin, true, std::uint32_t>::run(args,
                                                                     stream);
    }
  } else {
    if (wide) {
      L<Channels, Lmax, HasModes, HasSpin, false, long>::run(args, stream);
    } else {
      L<Channels, Lmax, HasModes, HasSpin, false, std::uint32_t>::run(args,
                                                                      stream);
    }
  }
}

// Native spin is a compile-time specialization for the same reason as the
// mode residual: a spinless descriptor must not carry the spin accumulators,
// the staged spin cotangent, or the wider moment state of one that has them.
// The neighbour spin width is derived from the degree profile, so presence is
// the whole choice and the instantiation matrix only doubles.
template <int Channels,
          int Lmax,
          bool HasModes,
          template <int, int, bool, bool, bool, typename> class L>
void dispatch_spin(const Arguments& args, cudaStream_t stream) {
  if (args.has_spin) {
    dispatch_topology<Channels, Lmax, HasModes, true, L>(args, stream);
  } else {
    dispatch_topology<Channels, Lmax, HasModes, false, L>(args, stream);
  }
}

// The mode residual is a compile-time specialization for the same reason as
// the angular degree: a descriptor without radial modes must not carry the
// vector temporaries and the shared profile cache of one that has them.
template <int Channels,
          int Lmax,
          template <int, int, bool, bool, bool, typename> class L>
void dispatch_modes(const Arguments& args, cudaStream_t stream) {
  if (args.radial_modes > 0) {
    dispatch_spin<Channels, Lmax, true, L>(args, stream);
  } else {
    dispatch_spin<Channels, Lmax, false, L>(args, stream);
  }
}

// The angular degree is a compile-time specialization because degrees three
// and four add moment accumulators that must not enter the register budget of
// the production ``lmax=2`` path.
//
// The operator entry point validates the degree before it reaches this
// dispatch. The unreachable default is still checked rather than folded into
// the highest degree, so that a degree outside the compiled set can only ever
// fail loudly instead of running a kernel for a different model.
template <int Channels, template <int, int, bool, bool, bool, typename> class L>
void dispatch_degree(const Arguments& args, cudaStream_t stream) {
  switch (args.lmax) {
    case 2:
      dispatch_modes<Channels, 2, L>(args, stream);
      return;
    case 3:
      dispatch_modes<Channels, 3, L>(args, stream);
      return;
    case 4:
      dispatch_modes<Channels, 4, L>(args, stream);
      return;
    default:
      throw std::runtime_error("dpa4c: uncompiled angular degree");
  }
}

#define DPA4C_DEFINE_CHANNEL(width)                              \
  void launch_forward_c##width(const Arguments& arguments,       \
                               cudaStream_t stream) {            \
    dispatch_degree<width, ForwardLauncher>(arguments, stream);  \
  }                                                              \
  void launch_backward_c##width(const Arguments& arguments,      \
                                cudaStream_t stream) {           \
    dispatch_degree<width, BackwardLauncher>(arguments, stream); \
  }

}  // namespace deepmd_dpa4c
