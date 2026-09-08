// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Invariant readout of the compressed DPA4C descriptor, forward and backward.
//
// The readout is node local and its matrices are small -- the widest degree
// carries sixteen channels and the widest probe rank is eight -- so it is
// written as plain scalar arithmetic on stack workspaces. Vectorizing it
// would fight the block shapes for a part of the step that the edge scan
// dominates by two orders of magnitude.
//
// Included by graph_compress_cpu_kernel.h inside the per-instruction-set
// namespace.

namespace {

/// Widest degree channel width and probe rank the profile family reaches.
constexpr int kMaxWidth = 16;
constexpr int kMaxRank = 16;
constexpr int kMaxComponents = 9;

/// Stack workspace of one node's readout.
struct ReadoutWorkspace {
  float aligned[kMaxDegree][kMaxComponents][kMaxWidth];
  float probe[kMaxDegree][kMaxComponents][kMaxRank];
  float aligned_cotangent[kMaxDegree][kMaxComponents][kMaxWidth];
  float probe_cotangent[kMaxDegree][kMaxComponents][kMaxRank];
  float vectors[kMaxRank][3];
  float tensors[kMaxRank][3][3];
  float tensor_vector[kMaxRank][kMaxRank][3];
  float tensor_cotangent[kMaxRank][3][3];
  float tensor_vector_cotangent[kMaxRank][kMaxRank][3];
};

/// Multiply a degree block by one packed readout matrix.
///
/// \param source Degree block with shape `(components, rows)`.
/// \param matrix Packed matrix with row stride `stride`.
/// \param out Result with shape `(components, columns)`.
inline void project(const float* source,
                    int source_stride,
                    const float* matrix,
                    int stride,
                    int rows,
                    int columns,
                    int component_count,
                    float* out,
                    int out_stride) {
  for (int component = 0; component < component_count; ++component) {
    const float* in = source + component * source_stride;
    float* result = out + component * out_stride;
    for (int column = 0; column < columns; ++column) {
      result[column] = 0.0f;
    }
    for (int row = 0; row < rows; ++row) {
      const float weight = in[row];
      const float* line = matrix + row * stride;
      for (int column = 0; column < columns; ++column) {
        result[column] += weight * line[column];
      }
    }
  }
}

/// Convert packed degree-two coefficients to a symmetric traceless matrix.
inline void packed_to_tensor(const float* packed, int stride, float tensor[3][3]) {
  const float inv_sqrt_two = 1.0f / kSqrtTwo;
  const float inv_sqrt_six = 1.0f / kSqrtSix;
  const float q0 = packed[0];
  const float q1 = packed[stride];
  const float q2 = packed[2 * stride];
  const float q3 = packed[3 * stride];
  const float q4 = packed[4 * stride];
  const float xy = q0 * inv_sqrt_two;
  const float yz = q1 * inv_sqrt_two;
  const float xz = q3 * inv_sqrt_two;
  tensor[0][0] = -q2 * inv_sqrt_six + q4 * inv_sqrt_two;
  tensor[1][1] = -q2 * inv_sqrt_six - q4 * inv_sqrt_two;
  tensor[2][2] = 2.0f * q2 * inv_sqrt_six;
  tensor[0][1] = xy;
  tensor[1][0] = xy;
  tensor[1][2] = yz;
  tensor[2][1] = yz;
  tensor[0][2] = xz;
  tensor[2][0] = xz;
}

/// Pull a symmetric-traceless cotangent back to the packed coefficients.
inline void packed_to_tensor_backward(const float cotangent[3][3],
                                      float* packed,
                                      int stride) {
  const float inv_sqrt_two = 1.0f / kSqrtTwo;
  const float inv_sqrt_six = 1.0f / kSqrtSix;
  packed[0] += (cotangent[0][1] + cotangent[1][0]) * inv_sqrt_two;
  packed[stride] += (cotangent[1][2] + cotangent[2][1]) * inv_sqrt_two;
  packed[2 * stride] +=
      (2.0f * cotangent[2][2] - cotangent[0][0] - cotangent[1][1]) * inv_sqrt_six;
  packed[3 * stride] += (cotangent[0][2] + cotangent[2][0]) * inv_sqrt_two;
  packed[4 * stride] += (cotangent[0][0] - cotangent[1][1]) * inv_sqrt_two;
}

/// Isometric scale of one symmetric 222 probe multiset.
inline float multiset_scale(int first, int second, int third) {
  if (first == second && second == third) {
    return 1.0f;
  }
  if (first == second || second == third || first == third) {
    return kSqrtThree;
  }
  return 2.44948974968f;  // sqrt(6)
}

/// Build the aligned and probe blocks of every non-scalar degree.
template <int LMAX>
inline void build_blocks(const Layout& layout,
                         const float* readout,
                         const float* moments,
                         ReadoutWorkspace& work) {
  const int width_one = layout.degree_channels[1];
  const int stride = width_one;
  for (int degree = 1; degree <= LMAX; ++degree) {
    const int width = layout.degree_channels[degree];
    const int rank = layout.ranks[degree - 1];
    const int count = components(degree);
    const float* block = moments + layout.degree_offset(degree);
    float* aligned = &work.aligned[degree - 1][0][0];
    float* probe = &work.probe[degree - 1][0][0];
    if (degree <= 2) {
      project(block, width, readout + (2 * (degree - 1)) * stride * stride, stride,
              width, width, count, aligned, kMaxWidth);
      project(aligned, kMaxWidth, readout + (4 + 2 * (degree - 1)) * stride * stride,
              stride, width, rank, count, probe, kMaxRank);
    } else {
      for (int component = 0; component < count; ++component) {
        for (int channel = 0; channel < width; ++channel) {
          aligned[component * kMaxWidth + channel] = block[component * width + channel];
        }
        for (int index = 0; index < rank; ++index) {
          probe[component * kMaxRank + index] = block[component * width + index];
        }
      }
    }
  }
}

/// Build the probe vectors, tensors and their product.
template <int LMAX>
inline void build_probe_products(const Layout& layout, ReadoutWorkspace& work) {
  const int rank_one = layout.ranks[0];
  const int rank_two = layout.ranks[1];
  for (int index = 0; index < rank_one; ++index) {
    for (int component = 0; component < 3; ++component) {
      work.vectors[index][component] = work.probe[0][component][index];
    }
  }
  for (int index = 0; index < rank_two; ++index) {
    packed_to_tensor(&work.probe[1][0][index], kMaxRank, work.tensors[index]);
  }
  for (int tensor = 0; tensor < rank_two; ++tensor) {
    for (int vector = 0; vector < rank_one; ++vector) {
      for (int row = 0; row < 3; ++row) {
        float accumulator = 0.0f;
        for (int column = 0; column < 3; ++column) {
          accumulator += work.tensors[tensor][row][column] * work.vectors[vector][column];
        }
        work.tensor_vector[tensor][vector][row] = accumulator;
      }
    }
  }
}

/// Trace of the product of three symmetric matrices.
inline float triple_trace(const float first[3][3],
                          const float second[3][3],
                          const float third[3][3]) {
  float trace = 0.0f;
  for (int row = 0; row < 3; ++row) {
    for (int middle = 0; middle < 3; ++middle) {
      float partial = 0.0f;
      for (int column = 0; column < 3; ++column) {
        partial += second[middle][column] * third[column][row];
      }
      trace += first[row][middle] * partial;
    }
  }
  return trace;
}

/// Contract the invariant readout of one node into the descriptor row.
template <int LMAX>
inline void readout_forward(const Arguments& arguments,
                            const Layout& layout,
                            const float* moments,
                            float scalar_divisor,
                            float angular_divisor,
                            int64_t type,
                            ReadoutWorkspace& work,
                            float* row) {
  build_blocks<LMAX>(layout, arguments.readout, moments, work);
  build_probe_products<LMAX>(layout, work);

  for (int channel = 0; channel < layout.channels; ++channel) {
    row[channel] = moments[channel];
  }

  int coordinate = layout.gram_base;
  for (int degree = 1; degree <= LMAX; ++degree) {
    const int width = layout.degree_channels[degree];
    const int count = components(degree);
    const float* aligned = &work.aligned[degree - 1][0][0];
    for (int first = 0; first < width; ++first) {
      for (int second = first; second < width; ++second) {
        float accumulator = 0.0f;
        for (int component = 0; component < count; ++component) {
          accumulator += aligned[component * kMaxWidth + first] *
                         aligned[component * kMaxWidth + second];
        }
        row[coordinate++] =
            first == second ? accumulator : kSqrtTwo * accumulator;
      }
    }
  }

  const int rank_one = layout.ranks[0];
  const int rank_two = layout.ranks[1];
  int index = layout.bispectrum_base;
  for (int first = 0; first < rank_one; ++first) {
    for (int second = first; second < rank_one; ++second) {
      const float scale =
          (first == second ? 1.0f : kSqrtTwo) * -kInvSqrtFive;
      for (int tensor = 0; tensor < rank_two; ++tensor) {
        float accumulator = 0.0f;
        for (int component = 0; component < 3; ++component) {
          accumulator += work.vectors[first][component] *
                         work.tensor_vector[tensor][second][component];
        }
        row[index++] = scale * accumulator;
      }
    }
  }

  index = layout.closed_222_base;
  for (int first = 0; first < rank_two; ++first) {
    for (int second = first; second < rank_two; ++second) {
      for (int third = second; third < rank_two; ++third) {
        row[index++] = -kBis222Scale * multiset_scale(first, second, third) *
                       triple_trace(work.tensors[first], work.tensors[second],
                                    work.tensors[third]);
      }
    }
  }

  for (int record = 0; record < arguments.coupling_count; ++record) {
    const int32_t* meta = arguments.coupling_meta + 8 * record;
    const int degree_one = meta[0];
    const int degree_two = meta[1];
    const int degree_three = meta[2];
    const int nonzero_begin = meta[3];
    const int nonzero_count = meta[4];
    const int probe_begin = meta[5];
    const int probe_count = meta[6];
    const int base = meta[7];
    const float* first_block = &work.probe[degree_one - 1][0][0];
    const float* second_block = &work.probe[degree_two - 1][0][0];
    const float* third_block = &work.probe[degree_three - 1][0][0];
    for (int entry = 0; entry < probe_count; ++entry) {
      const int32_t packed = arguments.coupling_entry[probe_begin + entry];
      const int first = packed & 0xFF;
      const int second = (packed >> 8) & 0xFF;
      const int third = (packed >> 16) & 0xFF;
      float accumulator = 0.0f;
      for (int nonzero = 0; nonzero < nonzero_count; ++nonzero) {
        const int32_t components_packed =
            arguments.coupling_entry[nonzero_begin + nonzero];
        accumulator +=
            arguments.coupling_value[nonzero_begin + nonzero] *
            first_block[(components_packed & 0xFF) * kMaxRank + first] *
            second_block[((components_packed >> 8) & 0xFF) * kMaxRank + second] *
            third_block[((components_packed >> 16) & 0xFF) * kMaxRank + third];
      }
      row[base + entry] = arguments.coupling_value[probe_begin + entry] * accumulator;
    }
  }

  index = layout.quartic_base;
  for (int tensor = 0; tensor < rank_two; ++tensor) {
    for (int vector = 0; vector < rank_one; ++vector) {
      const float* value = work.tensor_vector[tensor][vector];
      row[index++] =
          value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
    }
  }

  row[layout.divisor_base] = scalar_divisor;
  row[layout.divisor_base + 1] = angular_divisor;
  const float* embedding = arguments.type_embedding + type * layout.channels;
  for (int channel = 0; channel < layout.channels; ++channel) {
    row[layout.type_base + channel] = embedding[channel];
  }

  for (int output = 0; output < layout.output_width; ++output) {
    row[output] =
        (row[output] - arguments.output_mean[output]) * arguments.output_inv_std[output];
  }
}

/// Differentiate the invariant readout of one node.
///
/// \param cotangent Descriptor cotangent of this node, already calibrated.
/// \param moments Normalized moments saved by the forward.
/// \param moment_cotangent Receives the cotangent of the normalized moments.
/// \param divisor_cotangent Receives the two divisor cotangents, including the
///        contribution the normalization itself makes.
template <int LMAX>
inline void readout_backward(const Arguments& arguments,
                             const Layout& layout,
                             const float* cotangent,
                             const float* moments,
                             float scalar_divisor,
                             float angular_divisor,
                             ReadoutWorkspace& work,
                             float* moment_cotangent,
                             float* divisor_cotangent) {
  build_blocks<LMAX>(layout, arguments.readout, moments, work);
  build_probe_products<LMAX>(layout, work);

  std::memset(work.aligned_cotangent, 0, sizeof(work.aligned_cotangent));
  std::memset(work.probe_cotangent, 0, sizeof(work.probe_cotangent));
  std::memset(work.tensor_cotangent, 0, sizeof(work.tensor_cotangent));
  std::memset(work.tensor_vector_cotangent, 0, sizeof(work.tensor_vector_cotangent));

  const int rank_one = layout.ranks[0];
  const int rank_two = layout.ranks[1];

  int index = layout.quartic_base;
  for (int tensor = 0; tensor < rank_two; ++tensor) {
    for (int vector = 0; vector < rank_one; ++vector) {
      const float weight = 2.0f * cotangent[index++];
      for (int component = 0; component < 3; ++component) {
        work.tensor_vector_cotangent[tensor][vector][component] +=
            weight * work.tensor_vector[tensor][vector][component];
      }
    }
  }

  index = layout.bispectrum_base;
  for (int first = 0; first < rank_one; ++first) {
    for (int second = first; second < rank_one; ++second) {
      const float scale = (first == second ? 1.0f : kSqrtTwo) * -kInvSqrtFive;
      for (int tensor = 0; tensor < rank_two; ++tensor) {
        const float weight = scale * cotangent[index++];
        for (int component = 0; component < 3; ++component) {
          work.probe_cotangent[0][component][first] +=
              weight * work.tensor_vector[tensor][second][component];
          work.tensor_vector_cotangent[tensor][second][component] +=
              weight * work.vectors[first][component];
        }
      }
    }
  }

  index = layout.closed_222_base;
  for (int first = 0; first < rank_two; ++first) {
    for (int second = first; second < rank_two; ++second) {
      for (int third = second; third < rank_two; ++third) {
        const float weight = -kBis222Scale * multiset_scale(first, second, third) *
                             cotangent[index++];
        const int order[3][3] = {{first, second, third},
                                 {second, third, first},
                                 {third, first, second}};
        // d tr(ABC) / dA = (BC)^T, and the cyclic rotations give the other two.
        for (int slot = 0; slot < 3; ++slot) {
          const float(*left)[3] = work.tensors[order[slot][1]];
          const float(*right)[3] = work.tensors[order[slot][2]];
          float(*target)[3] = work.tensor_cotangent[order[slot][0]];
          for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
              float partial = 0.0f;
              for (int middle = 0; middle < 3; ++middle) {
                partial += left[column][middle] * right[middle][row];
              }
              target[row][column] += weight * partial;
            }
          }
        }
      }
    }
  }

  for (int record = 0; record < arguments.coupling_count; ++record) {
    const int32_t* meta = arguments.coupling_meta + 8 * record;
    const int degree_one = meta[0];
    const int degree_two = meta[1];
    const int degree_three = meta[2];
    const int nonzero_begin = meta[3];
    const int nonzero_count = meta[4];
    const int probe_begin = meta[5];
    const int probe_count = meta[6];
    const int base = meta[7];
    const float* first_block = &work.probe[degree_one - 1][0][0];
    const float* second_block = &work.probe[degree_two - 1][0][0];
    const float* third_block = &work.probe[degree_three - 1][0][0];
    float* first_cotangent = &work.probe_cotangent[degree_one - 1][0][0];
    float* second_cotangent = &work.probe_cotangent[degree_two - 1][0][0];
    float* third_cotangent = &work.probe_cotangent[degree_three - 1][0][0];
    for (int entry = 0; entry < probe_count; ++entry) {
      const int32_t packed = arguments.coupling_entry[probe_begin + entry];
      const int first = packed & 0xFF;
      const int second = (packed >> 8) & 0xFF;
      const int third = (packed >> 16) & 0xFF;
      const float weight =
          arguments.coupling_value[probe_begin + entry] * cotangent[base + entry];
      if (weight == 0.0f) {
        continue;
      }
      for (int nonzero = 0; nonzero < nonzero_count; ++nonzero) {
        const int32_t components_packed =
            arguments.coupling_entry[nonzero_begin + nonzero];
        const int first_index = (components_packed & 0xFF) * kMaxRank + first;
        const int second_index = ((components_packed >> 8) & 0xFF) * kMaxRank + second;
        const int third_index = ((components_packed >> 16) & 0xFF) * kMaxRank + third;
        const float value =
            weight * arguments.coupling_value[nonzero_begin + nonzero];
        first_cotangent[first_index] +=
            value * second_block[second_index] * third_block[third_index];
        second_cotangent[second_index] +=
            value * first_block[first_index] * third_block[third_index];
        third_cotangent[third_index] +=
            value * first_block[first_index] * second_block[second_index];
      }
    }
  }

  for (int tensor = 0; tensor < rank_two; ++tensor) {
    for (int vector = 0; vector < rank_one; ++vector) {
      const float* seed = work.tensor_vector_cotangent[tensor][vector];
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          work.tensor_cotangent[tensor][row][column] +=
              seed[row] * work.vectors[vector][column];
          work.probe_cotangent[0][column][vector] +=
              seed[row] * work.tensors[tensor][row][column];
        }
      }
    }
  }
  for (int tensor = 0; tensor < rank_two; ++tensor) {
    packed_to_tensor_backward(work.tensor_cotangent[tensor],
                              &work.probe_cotangent[1][0][tensor], kMaxRank);
  }

  int coordinate = layout.gram_base;
  for (int degree = 1; degree <= LMAX; ++degree) {
    const int width = layout.degree_channels[degree];
    const int count = components(degree);
    const float* aligned = &work.aligned[degree - 1][0][0];
    float* target = &work.aligned_cotangent[degree - 1][0][0];
    for (int first = 0; first < width; ++first) {
      for (int second = first; second < width; ++second) {
        const float weight =
            (first == second ? 1.0f : kSqrtTwo) * cotangent[coordinate++];
        for (int component = 0; component < count; ++component) {
          const float left = aligned[component * kMaxWidth + first];
          const float right = aligned[component * kMaxWidth + second];
          target[component * kMaxWidth + first] += weight * right;
          target[component * kMaxWidth + second] += weight * left;
        }
      }
    }
  }

  const int stride = layout.degree_channels[1];
  for (int degree = 1; degree <= LMAX; ++degree) {
    const int width = layout.degree_channels[degree];
    const int rank = layout.ranks[degree - 1];
    const int count = components(degree);
    float* aligned_cotangent = &work.aligned_cotangent[degree - 1][0][0];
    const float* probe_cotangent = &work.probe_cotangent[degree - 1][0][0];
    float* target = moment_cotangent + layout.degree_offset(degree);
    if (degree <= 2) {
      // The probe projection and the residual alignment are both stored with
      // their transpose, so the pullback is another packed projection.
      const float* probe_transpose =
          arguments.readout + (5 + 2 * (degree - 1)) * stride * stride;
      for (int component = 0; component < count; ++component) {
        const float* seed = probe_cotangent + component * kMaxRank;
        float* accumulator = aligned_cotangent + component * kMaxWidth;
        for (int index = 0; index < rank; ++index) {
          const float weight = seed[index];
          const float* line = probe_transpose + index * stride;
          for (int channel = 0; channel < width; ++channel) {
            accumulator[channel] += weight * line[channel];
          }
        }
      }
      project(aligned_cotangent, kMaxWidth,
              arguments.readout + (1 + 2 * (degree - 1)) * stride * stride, stride,
              width, width, count, target, width);
    } else {
      for (int component = 0; component < count; ++component) {
        for (int channel = 0; channel < width; ++channel) {
          target[component * width + channel] =
              aligned_cotangent[component * kMaxWidth + channel];
        }
        for (int index = 0; index < rank; ++index) {
          target[component * width + index] +=
              probe_cotangent[component * kMaxRank + index];
        }
      }
    }
  }

  for (int channel = 0; channel < layout.channels; ++channel) {
    moment_cotangent[channel] = cotangent[channel];
  }

  // The normalization is irreversible only in the forward: its cotangent
  // splits into the unnormalized moment and the divisor that produced it.
  float scalar_pullback = cotangent[layout.divisor_base];
  float angular_pullback = cotangent[layout.divisor_base + 1];
  const float inverse_scalar = 1.0f / scalar_divisor;
  const float inverse_angular = 1.0f / angular_divisor;
  for (int channel = 0; channel < layout.channels; ++channel) {
    scalar_pullback -= moment_cotangent[channel] * moments[channel] * inverse_scalar;
    moment_cotangent[channel] *= inverse_scalar;
  }
  for (int entry = layout.channels; entry < layout.moment_width; ++entry) {
    angular_pullback -= moment_cotangent[entry] * moments[entry] * inverse_angular;
    moment_cotangent[entry] *= inverse_angular;
  }
  divisor_cotangent[0] = 0.5f * scalar_pullback * inverse_scalar;
  divisor_cotangent[1] = 0.5f * angular_pullback * inverse_angular;
}

}  // namespace
