// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Destination-local edge scans of the compressed DPA4C descriptor.
//
// One node and every edge that reduces onto it belong to one thread, so the
// moment accumulators live in registers for the whole scan and the operator
// contains no atomic and no cross-thread reduction. The channel axis is the
// vector axis throughout; the angular components are the outer loop, which
// keeps their accumulators resident while the wide scalar block streams.
//
// Included by graph_compress_cpu_kernel.h inside the per-instruction-set
// namespace.

namespace {

/// Flat index of harmonic component `m` of degree `l`.
constexpr int angular_slot(int degree, int component) {
  return degree * degree - 1 + component;
}

/// Independent partial sums carried through a per-edge reduction.
///
/// Both reductions of the backward run over at most a couple of dozen terms,
/// which is short enough that a single accumulator turns the loop into a
/// chain of dependent fused multiply-adds and the kernel waits on their
/// four-cycle latency rather than on the two-per-cycle issue rate. Four
/// partial sums cover that latency on every current x86 core.
constexpr int kWays = 4;

/// Upper bound on the padded component count, for stack partial sums.
constexpr int kMaxSlotPad = 32;

/// Per-thread buffers of one node range.
///
/// All of them are carved out of one cache-line-aligned arena: every buffer
/// is a whole number of vector blocks, so aligned loads and stores are the
/// only accesses the channel loops issue, and a thread's working set stays
/// within a few kilobytes of one another in L1.
class ScanWorkspace {
 public:
  ScanWorkspace(const Layout& layout, int angular_pad, int slot_pad)
      : channels_(layout.padded_channels),
        angular_pad_(angular_pad),
        slot_pad_(slot_pad) {
    const int slots = angular_components(layout.lmax);
    const size_t sizes[] = {
        static_cast<size_t>(channels_),                    // value
        static_cast<size_t>(channels_),                    // slope
        static_cast<size_t>(channels_),                    // amplitude
        static_cast<size_t>(channels_),                    // cotangent
        static_cast<size_t>(channels_),                    // scalar
        static_cast<size_t>(angular_pad_),                 // combined
        static_cast<size_t>(slot_pad_),                    // projection
        static_cast<size_t>(slots) * angular_pad_,         // angular
        static_cast<size_t>(angular_pad_) * slot_pad_,     // transposed
        round_up(static_cast<size_t>(layout.moment_width)),         // moments
        round_up(2 * static_cast<size_t>(std::max(layout.modes, 1)))  // mixed
    };
    size_t total = 0;
    for (size_t size : sizes) {
      total += round_up(size);
    }
    arena_.assign(total + kAlignment / sizeof(float), 0.0f);
    float* cursor = align(arena_.data());
    float** targets[] = {&value,      &slope,      &amplitude, &cotangent,
                         &scalar,     &combined,   &projection, &angular,
                         &transposed, &moments,    &mixed};
    for (size_t index = 0; index < sizeof(sizes) / sizeof(sizes[0]); ++index) {
      *targets[index] = cursor;
      cursor += round_up(sizes[index]);
    }
  }

  float* value;       ///< Spline values, padded channels.
  float* slope;       ///< Spline distance derivatives.
  float* amplitude;   ///< Ordered FiLM amplitude before the envelope.
  float* cotangent;   ///< Amplitude cotangent, backward only.
  float* scalar;      ///< Degree-zero accumulator or cotangent.
  float* combined;    ///< Basis-weighted angular cotangent sum.
  float* projection;  ///< Amplitude-weighted angular cotangent sum.
  float* angular;     ///< Angular accumulators or cotangents, component major.
  float* transposed;  ///< The same, channel major, backward only.
  float* moments;     ///< Packed normalized moments.
  float* mixed;       ///< Mode values and their derivatives.
  ReadoutWorkspace readout;

 private:
  static constexpr size_t kAlignment = 64;

  static size_t round_up(size_t count) {
    const size_t lanes = kAlignment / sizeof(float);
    return (count + lanes - 1) / lanes * lanes;
  }

  static float* align(float* pointer) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return reinterpret_cast<float*>((address + kAlignment - 1) & ~(kAlignment - 1));
  }

  std::vector<float> arena_;
  int channels_;
  int angular_pad_;
  int slot_pad_;
};

/// Resolve the payload slot of one CSR entry.
inline int64_t payload_slot(const Arguments& arguments, int64_t entry) {
  return arguments.destination_order == nullptr ? entry
                                                : arguments.destination_order[entry];
}

/// Evaluate the ordered FiLM amplitude of one edge over the padded channels.
///
/// One mode at a time, each a broadcast-weighted pass over contiguous
/// channels. Folding the mode loop inside the channel loop keeps the
/// amplitude in registers and looks like the better trade, but it makes the
/// inner trip count a run-time value that the vectorizer cannot carry an
/// accumulator across: measured at two and a half times slower on the
/// mode-carrying grades. The amplitude fits L1 several times over, so the
/// repeated passes cost nothing the reload does not already hide.
template <bool HAS_MODES>
inline void film_amplitude(const Arguments& arguments,
                           const Layout& layout,
                           int64_t pair,
                           const float* __restrict value,
                           const float* __restrict modes,
                           float* __restrict amplitude) {
  const int padded = layout.padded_channels;
  const float* __restrict scale = arguments.tables->film.data() + pair * 2 * padded;
  const float* __restrict shift = scale + padded;
  for (int base = 0; base < padded; base += kBlock) {
    for (int lane = 0; lane < kBlock; ++lane) {
      amplitude[base + lane] =
          value[base + lane] * scale[base + lane] + shift[base + lane];
    }
  }
  if (HAS_MODES) {
    const float* __restrict mixing =
        arguments.tables->mixing.data() + pair * layout.modes * padded;
    for (int mode = 0; mode < layout.modes; ++mode) {
      const float weight = modes[mode];
      const float* __restrict line = mixing + mode * padded;
      for (int base = 0; base < padded; base += kBlock) {
        for (int lane = 0; lane < kBlock; ++lane) {
          amplitude[base + lane] += weight * line[base + lane];
        }
      }
    }
  }
}

/// Accumulate every degree-wise moment of one node range.
template <int LMAX, bool HAS_MODES>
void forward_scan(const Arguments& arguments,
                  const Layout& layout,
                  int64_t node_begin,
                  int64_t node_end) {
  const int padded = layout.padded_channels;
  const int angular_pad =
      ((layout.degree_channels[1] + kBlock - 1) / kBlock) * kBlock;
  const int slots = angular_components(LMAX);
  ScanWorkspace work(layout, angular_pad, kBlock);
  float basis[angular_components(kMaxDegree)];

  for (int64_t node = node_begin; node < node_end; ++node) {
    std::fill(work.scalar, work.scalar + padded, 0.0f);
    std::fill(work.angular, work.angular + slots * angular_pad, 0.0f);
    float scalar_mass = 0.0f;
    float angular_mass = 0.0f;
    const int64_t center_type = arguments.atype[node];

    if (center_type < layout.type_count - 1) {
      for (int64_t entry = arguments.row_ptr[node];
           entry < arguments.row_ptr[node + 1]; ++entry) {
        const int64_t edge = payload_slot(arguments, entry);
        if (arguments.edge_mask != nullptr && !arguments.edge_mask[edge]) {
          continue;
        }
        EdgeGeometry geometry;
        if (!resolve_edge(arguments, layout, edge, center_type, &geometry)) {
          continue;
        }
        const float* interval =
            arguments.tables->spline.data() + geometry.interval * layout.spline_stride;
        spline_value(interval, geometry.dx, layout.channel_blocks,
                     work.value);
        if (HAS_MODES) {
          mode_value(interval + layout.channel_blocks * 6 * kBlock, geometry.dx,
                     layout.modes, work.mixed);
        }
        film_amplitude<HAS_MODES>(arguments, layout, geometry.pair,
                                  work.value, work.mixed,
                                  work.amplitude);

        const float chi = geometry.chi;
        const float chi_squared = chi * chi;
        scalar_mass += chi_squared;
        angular_mass += chi_squared * chi_squared;
        float* __restrict scalar = work.scalar;
        const float* __restrict amplitude = work.amplitude;
        for (int base = 0; base < padded; base += kBlock) {
          for (int lane = 0; lane < kBlock; ++lane) {
            scalar[base + lane] += chi * amplitude[base + lane];
          }
        }
        harmonics<LMAX>(geometry.direction[0], geometry.direction[1],
                        geometry.direction[2], basis);
        for (int slot = 0; slot < slots; ++slot) {
          const float weight = chi_squared * basis[slot];
          float* __restrict target = work.angular + slot * angular_pad;
          for (int base = 0; base < angular_pad; base += kBlock) {
            for (int lane = 0; lane < kBlock; ++lane) {
              target[base + lane] += weight * amplitude[base + lane];
            }
          }
        }
      }
    }

    const float scalar_divisor = std::sqrt(scalar_mass + arguments.degree_floor);
    const float angular_divisor = std::sqrt(angular_mass + arguments.degree_floor);
    const float inverse_scalar = 1.0f / scalar_divisor;
    const float inverse_angular = 1.0f / angular_divisor;
    float* moments = work.moments;
    for (int channel = 0; channel < layout.channels; ++channel) {
      moments[channel] = work.scalar[channel] * inverse_scalar;
    }
    for (int degree = 1; degree <= LMAX; ++degree) {
      const int width = layout.degree_channels[degree];
      float* target = moments + layout.degree_offset(degree);
      for (int component = 0; component < components(degree); ++component) {
        const float* source =
            work.angular + angular_slot(degree, component) * angular_pad;
        for (int channel = 0; channel < width; ++channel) {
          target[component * width + channel] = source[channel] * inverse_angular;
        }
      }
    }

    float* state = arguments.state + node * (layout.moment_width + 2);
    state[0] = scalar_divisor;
    state[1] = angular_divisor;
    std::memcpy(state + 2, moments, sizeof(float) * layout.moment_width);
    readout_forward<LMAX>(arguments, layout, moments, scalar_divisor,
                          angular_divisor, center_type, work.readout,
                          arguments.descriptor + node * layout.output_width);
  }
}

/// Emit the edge cotangent of one node range.
template <int LMAX, bool HAS_MODES>
void backward_scan(const Arguments& arguments,
                   const Layout& layout,
                   int64_t node_begin,
                   int64_t node_end) {
  const int padded = layout.padded_channels;
  const int angular_pad =
      ((layout.degree_channels[1] + kBlock - 1) / kBlock) * kBlock;
  const int slots = angular_components(LMAX);
  const int slot_pad = ((slots + kBlock - 1) / kBlock) * kBlock;
  ScanWorkspace work(layout, angular_pad, slot_pad);
  std::vector<float> calibrated(layout.output_width, 0.0f);
  float basis[angular_components(kMaxDegree)];
  float basis_cotangent[angular_components(kMaxDegree)];
  float divisor_cotangent[2];

  for (int64_t node = node_begin; node < node_end; ++node) {
    const float* state = arguments.state + node * (layout.moment_width + 2);
    const float scalar_divisor = state[0];
    const float angular_divisor = state[1];
    const float* gradient = arguments.descriptor_gradient + node * layout.output_width;
    for (int output = 0; output < layout.output_width; ++output) {
      calibrated[output] = gradient[output] * arguments.output_inv_std[output];
    }
    readout_backward<LMAX>(arguments, layout, calibrated.data(), state + 2,
                           scalar_divisor, angular_divisor, work.readout,
                           work.moments, divisor_cotangent);

    std::fill(work.scalar, work.scalar + padded, 0.0f);
    std::fill(work.angular, work.angular + slots * angular_pad, 0.0f);
    std::fill(work.transposed, work.transposed + angular_pad * slot_pad, 0.0f);
    std::memcpy(work.scalar, work.moments,
                sizeof(float) * layout.channels);
    for (int degree = 1; degree <= LMAX; ++degree) {
      const int width = layout.degree_channels[degree];
      const float* source = work.moments + layout.degree_offset(degree);
      for (int component = 0; component < components(degree); ++component) {
        const int slot = angular_slot(degree, component);
        float* target = work.angular + slot * angular_pad;
        for (int channel = 0; channel < width; ++channel) {
          const float value = source[component * width + channel];
          target[channel] = value;
          work.transposed[channel * slot_pad + slot] = value;
        }
      }
    }

    const int64_t center_type = arguments.atype[node];
    const bool real_center = center_type < layout.type_count - 1;
    for (int64_t entry = arguments.row_ptr[node];
         entry < arguments.row_ptr[node + 1]; ++entry) {
      const int64_t edge = payload_slot(arguments, entry);
      float* target = arguments.edge_gradient + 3 * edge;
      EdgeGeometry geometry;
      const bool active =
          real_center &&
          (arguments.edge_mask == nullptr || arguments.edge_mask[edge]) &&
          resolve_edge(arguments, layout, edge, center_type, &geometry);
      if (!active) {
        target[0] = 0.0f;
        target[1] = 0.0f;
        target[2] = 0.0f;
        continue;
      }
      const float* interval =
          arguments.tables->spline.data() + geometry.interval * layout.spline_stride;
      spline_value_and_derivative(interval, geometry.dx, layout.channel_blocks,
                                  work.value, work.slope);
      if (HAS_MODES) {
        mode_value_and_derivative(interval + layout.channel_blocks * 6 * kBlock,
                                  geometry.dx, layout.modes, work.mixed,
                                  work.mixed + layout.modes);
      }
      film_amplitude<HAS_MODES>(arguments, layout, geometry.pair, work.value,
                                work.mixed, work.amplitude);

      const float chi = geometry.chi;
      const float chi_squared = chi * chi;
      harmonics<LMAX>(geometry.direction[0], geometry.direction[1],
                      geometry.direction[2], basis);

      // The angular cotangents reach the amplitude through a basis-weighted
      // sum over components, and the basis through an amplitude-weighted sum
      // over channels. The second is a reduction along the channel axis, so
      // it runs against the component-major copy, where the component index
      // is the vector axis and no horizontal reduction is issued at all.
      const float* __restrict amplitude = work.amplitude;
      float* __restrict combined = work.combined;
      float* __restrict projection = work.projection;
      float partial[kWays][kMaxSlotPad];

      for (int way = 0; way < kWays; ++way) {
        for (int base = 0; base < angular_pad; base += kBlock) {
          for (int lane = 0; lane < kBlock; ++lane) {
            partial[way][base + lane] = 0.0f;
          }
        }
      }
      int slot = 0;
      for (; slot + kWays <= slots; slot += kWays) {
        for (int way = 0; way < kWays; ++way) {
          const float* __restrict source =
              work.angular + (slot + way) * angular_pad;
          const float weight = basis[slot + way];
          for (int base = 0; base < angular_pad; base += kBlock) {
            for (int lane = 0; lane < kBlock; ++lane) {
              partial[way][base + lane] += weight * source[base + lane];
            }
          }
        }
      }
      for (int base = 0; base < angular_pad; base += kBlock) {
        for (int lane = 0; lane < kBlock; ++lane) {
          combined[base + lane] =
              (partial[0][base + lane] + partial[1][base + lane]) +
              (partial[2][base + lane] + partial[3][base + lane]);
        }
      }
      for (; slot < slots; ++slot) {
        const float* __restrict source = work.angular + slot * angular_pad;
        const float weight = basis[slot];
        for (int base = 0; base < angular_pad; base += kBlock) {
          for (int lane = 0; lane < kBlock; ++lane) {
            combined[base + lane] += weight * source[base + lane];
          }
        }
      }

      for (int way = 0; way < kWays; ++way) {
        for (int base = 0; base < slot_pad; base += kBlock) {
          for (int lane = 0; lane < kBlock; ++lane) {
            partial[way][base + lane] = 0.0f;
          }
        }
      }
      for (int channel = 0; channel < angular_pad; channel += kWays) {
        for (int way = 0; way < kWays; ++way) {
          const float weight = amplitude[channel + way];
          const float* __restrict line =
              work.transposed + (channel + way) * slot_pad;
          for (int base = 0; base < slot_pad; base += kBlock) {
            for (int lane = 0; lane < kBlock; ++lane) {
              partial[way][base + lane] += weight * line[base + lane];
            }
          }
        }
      }
      for (int base = 0; base < slot_pad; base += kBlock) {
        for (int lane = 0; lane < kBlock; ++lane) {
          projection[base + lane] =
              (partial[0][base + lane] + partial[1][base + lane]) +
              (partial[2][base + lane] + partial[3][base + lane]);
        }
      }
      // The amplitude and the angular cotangent meet in the same contraction
      // the basis cotangent already formed, so the envelope pullback reads it
      // off the components instead of reducing the channel axis again.
      float amplitude_dot = 0.0f;
      for (int slot = 0; slot < slots; ++slot) {
        basis_cotangent[slot] = chi_squared * projection[slot];
        amplitude_dot += basis[slot] * projection[slot];
      }

      const float* __restrict scalar = work.scalar;
      const float* __restrict film =
          arguments.tables->film.data() + geometry.pair * 2 * padded;
      const float* __restrict slope = work.slope;
      float* __restrict amplitude_cotangent = work.cotangent;
      float envelope_lanes[kBlock];
      float radial_lanes[kBlock];
      for (int lane = 0; lane < kBlock; ++lane) {
        envelope_lanes[lane] = 0.0f;
        radial_lanes[lane] = 0.0f;
      }
      // The amplitude cotangent, the envelope pullback and the radial
      // pullback all sweep the channel axis once, so they share it. Only the
      // leading blocks carry an angular contribution, which keeps both halves
      // free of a per-channel predicate.
      //
      // Both pullbacks accumulate one partial sum per lane and are folded
      // once, after the sweep. A scalar accumulator would forbid vectorizing
      // the sweep at all: reassociating a floating-point sum is not a
      // transformation a compiler may make unbidden, so the whole loop would
      // fall back to scalar arithmetic over every channel.
      for (int base = 0; base < padded; base += kBlock) {
        const bool angular_block = base < angular_pad;
        for (int lane = 0; lane < kBlock; ++lane) {
          const int channel = base + lane;
          const float value =
              (angular_block ? chi_squared * combined[channel] : 0.0f) +
              chi * scalar[channel];
          amplitude_cotangent[channel] = value;
          envelope_lanes[lane] += scalar[channel] * amplitude[channel];
          radial_lanes[lane] += value * film[channel] * slope[channel];
        }
      }
      float envelope_accumulator = 0.0f;
      float radial = 0.0f;
      for (int lane = 0; lane < kBlock; ++lane) {
        envelope_accumulator += envelope_lanes[lane];
        radial += radial_lanes[lane];
      }
      if (HAS_MODES) {
        const float* __restrict mixing =
            arguments.tables->mixing.data() + geometry.pair * layout.modes * padded;
        const float* __restrict mode_slope = work.mixed + layout.modes;
        for (int mode = 0; mode < layout.modes; ++mode) {
          const float* __restrict line = mixing + mode * padded;
          float mode_lanes[kBlock];
          for (int lane = 0; lane < kBlock; ++lane) {
            mode_lanes[lane] = 0.0f;
          }
          for (int base = 0; base < padded; base += kBlock) {
            for (int lane = 0; lane < kBlock; ++lane) {
              mode_lanes[lane] +=
                  amplitude_cotangent[base + lane] * line[base + lane];
            }
          }
          float mode_projection = 0.0f;
          for (int lane = 0; lane < kBlock; ++lane) {
            mode_projection += mode_lanes[lane];
          }
          radial += mode_projection * mode_slope[mode];
        }
      }
      radial += (envelope_accumulator + 2.0f * chi * divisor_cotangent[0] +
                 4.0f * chi * chi_squared * divisor_cotangent[1] +
                 2.0f * chi * amplitude_dot) *
                geometry.chi_slope;

      float direction_cotangent[3];
      harmonics_backward<LMAX>(geometry.direction[0], geometry.direction[1],
                               geometry.direction[2], basis_cotangent,
                               direction_cotangent);
      const float radial_part = direction_cotangent[0] * geometry.direction[0] +
                                direction_cotangent[1] * geometry.direction[1] +
                                direction_cotangent[2] * geometry.direction[2];
      const float inverse_radius = 1.0f / geometry.radius;
      for (int component = 0; component < 3; ++component) {
        target[component] =
            radial * geometry.direction[component] +
            inverse_radius * (direction_cotangent[component] -
                              radial_part * geometry.direction[component]);
      }
    }
  }
}

}  // namespace
