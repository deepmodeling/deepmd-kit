// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Compressed DPA4C descriptor kernels, compiled once per instruction set.
//
// The file is included once per level with `DPA4C_CPU_ISA` naming the target
// namespace, and each including translation unit is compiled with that
// level's flags. It therefore carries no include guard on purpose.
//
// Structure of one evaluation
// ---------------------------
// A thread owns a contiguous range of destination nodes together with every
// edge that reduces onto them. For each node it
//
//   1. scans its edges, evaluating the radial spline, the ordered FiLM
//      amplitude and the Cartesian harmonics, and accumulating the two
//      envelope masses and every degree-wise moment in registers;
//   2. normalizes the moments and writes the saved state;
//   3. contracts the invariant readout and writes the calibrated descriptor.
//
// The backward reverses the same three steps: it differentiates the readout
// from the saved state, then rescans the edges, recomputing the spline value
// and derivative from one table row, and emits one edge cotangent.
//
// Numerics
// --------
// Everything is IEEE float32, matching the CUDA path. The harmonics are
// evaluated with the squared norm of the regularized direction substituted by
// one: the two polynomials agree on the unit sphere, and their gradients
// differ by a purely radial term that the tangential projection closing the
// coordinate backward annihilates exactly.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "graph_compress_cpu.h"

namespace deepmd_dpa4c_cpu {
namespace DPA4C_CPU_ISA {

namespace {

/// Vector width of this level, in float lanes.
///
/// Compile time on purpose: every channel loop of the kernel is a whole
/// number of blocks, so a constant lane count turns each one into a straight
/// run of vector instructions with no peel and no tail.
constexpr int kBlock = DPA4C_CPU_BLOCK;

constexpr float kSqrtTwo = 1.41421356237309504880f;
constexpr float kSqrtThree = 1.73205080756887729353f;
constexpr float kSqrtFive = 2.23606797749978969641f;
constexpr float kSqrtSix = 2.44948974968f;
constexpr float kSqrtFifteen = 3.87298334620741688518f;
constexpr float kInvSqrtFive = 0.44721359549995793928f;
// Unit-Frobenius Cartesian normalization of the symmetric 222 coupling.
constexpr float kBis222Scale = 0.58554004376911709f;  // sqrt(12 / 35)

/// Number of harmonic components of degree `l`.
constexpr int components(int degree) { return 2 * degree + 1; }

/// Number of harmonic components of degrees one through `lmax`.
constexpr int angular_components(int lmax) {
  return lmax * (lmax + 2);  // sum_{l=1..lmax} (2l + 1)
}

// === Cutoff envelope ===

/// Evaluate the exponent-five C³ envelope and its derivative.
///
/// \param radius Regularized distance in Å.
/// \param rcut Outer cutoff in Å.
/// \param derivative Receives d(envelope)/d(radius).
/// \return The envelope value, exactly zero at and beyond the cutoff.
inline float envelope(float radius, float rcut, float* derivative) {
  if (radius >= rcut) {
    *derivative = 0.0f;
    return 0.0f;
  }
  const float inv_rcut = 1.0f / rcut;
  const float x = radius * inv_rcut;
  const float u = 1.0f - x;
  const float series =
      1.0f + x * (4.0f + x * (10.0f + x * (20.0f + 35.0f * x)));
  const float series_derivative = 4.0f + x * (20.0f + x * (60.0f + 140.0f * x));
  const float u2 = u * u;
  const float u3 = u2 * u;
  const float u4 = u3 * u;
  *derivative = inv_rcut * (u4 * series_derivative - 4.0f * u3 * series);
  return u4 * series;
}

// === Cartesian harmonics ===

/// Evaluate the real Cartesian harmonics of degrees one through `LMAX`.
///
/// The scalar degree is omitted because it is the constant one and never
/// enters an angular moment. Output component `l * l + m - 1` holds
/// \f$B^{(l)}_m\f$.
template <int LMAX>
inline void harmonics(float x, float y, float z, float* basis) {
  basis[0] = x;
  basis[1] = y;
  basis[2] = z;
  if (LMAX >= 2) {
    basis[3] = kSqrtThree * x * y;
    basis[4] = kSqrtThree * y * z;
    basis[5] = 0.5f * (3.0f * z * z - 1.0f);
    basis[6] = kSqrtThree * x * z;
    basis[7] = 0.5f * kSqrtThree * (x * x - y * y);
  }
  if (LMAX >= 3) {
    const float z2 = z * z;
    basis[8] = 0.79056941504209483f * y * (3.0f * x * x - y * y);
    basis[9] = kSqrtFifteen * x * y * z;
    basis[10] = 0.61237243569579452f * y * (5.0f * z2 - 1.0f);
    basis[11] = 0.5f * z * (5.0f * z2 - 3.0f);
    basis[12] = 0.61237243569579452f * x * (5.0f * z2 - 1.0f);
    basis[13] = 0.5f * kSqrtFifteen * z * (x * x - y * y);
    basis[14] = 0.79056941504209483f * x * (x * x - 3.0f * y * y);
  }
  if (LMAX >= 4) {
    const float z2 = z * z;
    const float x2 = x * x;
    const float y2 = y * y;
    const float difference = x2 - y2;
    basis[15] = 2.95803989154980802f * x * y * difference;
    basis[16] = 2.09165006633518887f * y * z * (3.0f * x2 - y2);
    basis[17] = 1.11803398874989484f * x * y * (7.0f * z2 - 1.0f);
    basis[18] = 0.79056941504209483f * y * z * (7.0f * z2 - 3.0f);
    basis[19] = 0.125f * (35.0f * z2 * z2 - 30.0f * z2 + 3.0f);
    basis[20] = 0.79056941504209483f * x * z * (7.0f * z2 - 3.0f);
    basis[21] = 0.55901699437494742f * difference * (7.0f * z2 - 1.0f);
    basis[22] = 2.09165006633518887f * x * z * (x2 - 3.0f * y2);
    basis[23] = 0.73950997288745200f * (x2 * x2 - 6.0f * x2 * y2 + y2 * y2);
  }
}

/// Accumulate the direction cotangent of the harmonics.
///
/// \param x,y,z Unit direction components.
/// \param cotangent Harmonic cotangents in the layout of :func:`harmonics`.
/// \param direction Receives \f$\sum_m \bar B_m \partial B_m/\partial u\f$.
template <int LMAX>
inline void harmonics_backward(
    float x, float y, float z, const float* cotangent, float* direction) {
  float dx = cotangent[0];
  float dy = cotangent[1];
  float dz = cotangent[2];
  if (LMAX >= 2) {
    dx += kSqrtThree * (cotangent[3] * y + cotangent[6] * z + cotangent[7] * x);
    dy += kSqrtThree * (cotangent[3] * x + cotangent[4] * z - cotangent[7] * y);
    dz += kSqrtThree * (cotangent[4] * y + cotangent[6] * x) +
          3.0f * cotangent[5] * z;
  }
  if (LMAX >= 3) {
    const float z2 = z * z;
    const float five_z2_minus_one = 5.0f * z2 - 1.0f;
    dx += 0.79056941504209483f * cotangent[8] * 6.0f * x * y +
          kSqrtFifteen * cotangent[9] * y * z +
          0.61237243569579452f * cotangent[12] * five_z2_minus_one +
          kSqrtFifteen * cotangent[13] * z * x +
          0.79056941504209483f * cotangent[14] * 3.0f * (x * x - y * y);
    dy += 0.79056941504209483f * cotangent[8] * 3.0f * (x * x - y * y) +
          kSqrtFifteen * cotangent[9] * x * z +
          0.61237243569579452f * cotangent[10] * five_z2_minus_one -
          kSqrtFifteen * cotangent[13] * z * y -
          0.79056941504209483f * cotangent[14] * 6.0f * x * y;
    dz += kSqrtFifteen * cotangent[9] * x * y +
          0.61237243569579452f * cotangent[10] * 10.0f * y * z +
          0.5f * cotangent[11] * (15.0f * z2 - 3.0f) +
          0.61237243569579452f * cotangent[12] * 10.0f * x * z +
          0.5f * kSqrtFifteen * cotangent[13] * (x * x - y * y);
  }
  if (LMAX >= 4) {
    const float z2 = z * z;
    const float x2 = x * x;
    const float y2 = y * y;
    const float seven_z2_minus_one = 7.0f * z2 - 1.0f;
    const float seven_z2_minus_three = 7.0f * z2 - 3.0f;
    dx += 2.95803989154980802f * cotangent[15] * y * (3.0f * x2 - y2) +
          2.09165006633518887f * cotangent[16] * 6.0f * x * y * z +
          1.11803398874989484f * cotangent[17] * y * seven_z2_minus_one +
          0.79056941504209483f * cotangent[20] * z * seven_z2_minus_three +
          0.55901699437494742f * cotangent[21] * 2.0f * x * seven_z2_minus_one +
          2.09165006633518887f * cotangent[22] * 3.0f * z * (x2 - y2) +
          0.73950997288745200f * cotangent[23] * 4.0f * x * (x2 - 3.0f * y2);
    dy += 2.95803989154980802f * cotangent[15] * x * (x2 - 3.0f * y2) +
          2.09165006633518887f * cotangent[16] * 3.0f * z * (x2 - y2) +
          1.11803398874989484f * cotangent[17] * x * seven_z2_minus_one +
          0.79056941504209483f * cotangent[18] * z * seven_z2_minus_three -
          0.55901699437494742f * cotangent[21] * 2.0f * y * seven_z2_minus_one -
          2.09165006633518887f * cotangent[22] * 6.0f * x * y * z +
          0.73950997288745200f * cotangent[23] * 4.0f * y * (y2 - 3.0f * x2);
    dz += 2.09165006633518887f * cotangent[16] * y * (3.0f * x2 - y2) +
          1.11803398874989484f * cotangent[17] * x * y * 14.0f * z +
          0.79056941504209483f * cotangent[18] * y * (21.0f * z2 - 3.0f) +
          0.125f * cotangent[19] * (140.0f * z2 * z - 60.0f * z) +
          0.79056941504209483f * cotangent[20] * x * (21.0f * z2 - 3.0f) +
          0.55901699437494742f * cotangent[21] * (x2 - y2) * 14.0f * z +
          2.09165006633518887f * cotangent[22] * x * (x2 - 3.0f * y2);
  }
  direction[0] = dx;
  direction[1] = dy;
  direction[2] = dz;
}

// === Radial spline ===

/// Evaluate the prepared spline over every channel block.
///
/// The interval stores six coefficient vectors per block, so the evaluation
/// is one Horner chain of contiguous fused multiply-adds.
inline void spline_value(const float* __restrict interval,
                         float dx,
                         int blocks,
                         float* __restrict value) {
  for (int index = 0; index < blocks; ++index) {
    const float* __restrict coefficients = interval + index * 6 * kBlock;
    float* __restrict out = value + index * kBlock;
    for (int lane = 0; lane < kBlock; ++lane) {
      float accumulator = coefficients[5 * kBlock + lane];
      accumulator = accumulator * dx + coefficients[4 * kBlock + lane];
      accumulator = accumulator * dx + coefficients[3 * kBlock + lane];
      accumulator = accumulator * dx + coefficients[2 * kBlock + lane];
      accumulator = accumulator * dx + coefficients[1 * kBlock + lane];
      out[lane] = accumulator * dx + coefficients[lane];
    }
  }
}

/// Evaluate the prepared spline and its distance derivative.
///
/// Value and slope are two independent Horner chains over the same six
/// coefficient vectors, so the loads are shared and the two dependency
/// chains interleave.
inline void spline_value_and_derivative(const float* __restrict interval,
                                        float dx,
                                        int blocks,
                                        float* __restrict value,
                                        float* __restrict derivative) {
  for (int index = 0; index < blocks; ++index) {
    const float* __restrict coefficients = interval + index * 6 * kBlock;
    float* __restrict out_value = value + index * kBlock;
    float* __restrict out_derivative = derivative + index * kBlock;
    for (int lane = 0; lane < kBlock; ++lane) {
      const float c5 = coefficients[5 * kBlock + lane];
      const float c4 = coefficients[4 * kBlock + lane];
      const float c3 = coefficients[3 * kBlock + lane];
      const float c2 = coefficients[2 * kBlock + lane];
      const float c1 = coefficients[1 * kBlock + lane];
      const float c0 = coefficients[lane];
      float accumulator = c5;
      accumulator = accumulator * dx + c4;
      accumulator = accumulator * dx + c3;
      accumulator = accumulator * dx + c2;
      accumulator = accumulator * dx + c1;
      out_value[lane] = accumulator * dx + c0;
      float slope = 5.0f * c5;
      slope = slope * dx + 4.0f * c4;
      slope = slope * dx + 3.0f * c3;
      slope = slope * dx + 2.0f * c2;
      out_derivative[lane] = slope * dx + c1;
    }
  }
}

/// Evaluate the `R` shared mode profiles of one interval.
inline void mode_value(const float* __restrict modes,
                       float dx,
                       int count,
                       float* __restrict value) {
  for (int mode = 0; mode < count; ++mode) {
    const float* __restrict coefficients = modes + mode * 6;
    float accumulator = coefficients[5];
    for (int order = 4; order >= 0; --order) {
      accumulator = accumulator * dx + coefficients[order];
    }
    value[mode] = accumulator;
  }
}

/// Evaluate the mode profiles and their distance derivatives.
inline void mode_value_and_derivative(const float* __restrict modes,
                                      float dx,
                                      int count,
                                      float* __restrict value,
                                      float* __restrict derivative) {
  for (int mode = 0; mode < count; ++mode) {
    const float* __restrict coefficients = modes + mode * 6;
    float accumulator = coefficients[5];
    float slope = 5.0f * coefficients[5];
    for (int order = 4; order >= 1; --order) {
      accumulator = accumulator * dx + coefficients[order];
      slope = slope * dx + static_cast<float>(order) * coefficients[order];
    }
    value[mode] = accumulator * dx + coefficients[0];
    derivative[mode] = slope;
  }
}

// === Per-edge geometry ===

/// Everything one edge contributes, resolved once per direction.
struct EdgeGeometry {
  float direction[3];
  float radius;
  float chi;
  float chi_slope;
  int64_t interval;
  float dx;
  int64_t pair;
};

/// Resolve one edge, returning false when it contributes nothing.
inline bool resolve_edge(const Arguments& arguments,
                         const Layout& layout,
                         int64_t edge,
                         int64_t center_type,
                         EdgeGeometry* geometry) {
  const int64_t neighbor = arguments.source[edge];
  const int64_t neighbor_type = arguments.atype[neighbor];
  if (neighbor_type >= layout.type_count - 1) {
    return false;
  }
  const float* vector = arguments.edge_vec + 3 * edge;
  const float squared =
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
  const float radius = std::sqrt(squared + arguments.eps * arguments.eps);
  float slope = 0.0f;
  const float chi = envelope(radius, arguments.rcut, &slope);
  if (chi == 0.0f) {
    return false;
  }
  const float inverse = 1.0f / radius;
  geometry->direction[0] = vector[0] * inverse;
  geometry->direction[1] = vector[1] * inverse;
  geometry->direction[2] = vector[2] * inverse;
  geometry->radius = radius;
  geometry->chi = chi;
  geometry->chi_slope = slope;
  const float coordinate = std::min(radius, arguments.table_max);
  int64_t interval = static_cast<int64_t>(coordinate / arguments.table_stride);
  interval = std::min<int64_t>(interval, layout.spline_count - 1);
  geometry->interval = interval;
  geometry->dx =
      coordinate - static_cast<float>(interval) * arguments.table_stride;
  geometry->pair = center_type * layout.type_count + neighbor_type;
  return true;
}

}  // namespace

#include "graph_compress_cpu_readout.inc"
#include "graph_compress_cpu_scan.inc"

/// Return the entry points of this instruction-set level.
Kernels kernels(int lmax, bool has_modes) {
  switch (lmax) {
    case 2:
      return has_modes
                 ? Kernels{forward_scan<2, true>, backward_scan<2, true>}
                 : Kernels{forward_scan<2, false>, backward_scan<2, false>};
    case 3:
      return has_modes
                 ? Kernels{forward_scan<3, true>, backward_scan<3, true>}
                 : Kernels{forward_scan<3, false>, backward_scan<3, false>};
    default:
      return has_modes
                 ? Kernels{forward_scan<4, true>, backward_scan<4, true>}
                 : Kernels{forward_scan<4, false>, backward_scan<4, false>};
  }
}

}  // namespace DPA4C_CPU_ISA
}  // namespace deepmd_dpa4c_cpu
