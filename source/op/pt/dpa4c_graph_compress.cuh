// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Device templates for the compressed degree-wise DPA4C descriptor.
//
// One warp owns one destination node. The forward scan evaluates the
// tabulated radial branch and its shared mode profiles, applies the ordered
// PairFiLM amplitude, and reduces both envelope masses together with every
// degree-wise moment in a single pass over the destination CSR row. The
// analytical backward runs one node-readout VJP followed by one edge
// recomputation scan that reproduces the amplitude instead of storing it.
//
// Degrees one and two carry the wide channel blocks and are contracted in
// closed form. Degrees three and four carry a single channel each, so their
// couplings run through a compact sparse Cartesian Gaunt table supplied as a
// compression artifact.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "dpa4c_graph_compress_launch.h"

namespace deepmd_dpa4c {

constexpr unsigned kWarpMask = 0xffffffffu;
constexpr int kMaxRadialModes = 8;
constexpr float kSqrtTwo = 1.4142135623730950488f;
constexpr float kSqrtThree = 1.7320508075688772935f;
constexpr float kInvSqrtTwo = 0.7071067811865475244f;
constexpr float kInvSqrtFive = 0.44721359549995793928f;
constexpr float kInvSqrtSix = 0.4082482904638630164f;

// Unit-Frobenius normalization of the Cartesian 222 Gaunt tensor. The triple
// product needs only one operator ordering: the three factors are symmetric,
// so `tr(ABC) = tr((ABC)^T) = tr(CBA) = tr(ACB)` and the two orderings of the
// contraction are identical.
constexpr float kBis222Scale = -0.58554004376911988f;

// === Warp primitives ===

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kWarpMask, value, offset);
  }
  return __shfl_sync(kWarpMask, value, 0);
}

template <int Width>
__device__ __forceinline__ unsigned subwarp_mask(int leader) {
  if constexpr (Width == kWarpSize) {
    return kWarpMask;
  } else {
    return ((1u << Width) - 1u) << leader;
  }
}

template <int Width>
__device__ __forceinline__ float reduce_channel_groups(float value) {
  if constexpr (Width < kWarpSize) {
#pragma unroll
    for (int offset = Width; offset < kWarpSize; offset <<= 1) {
      value += __shfl_xor_sync(kWarpMask, value, offset);
    }
  }
  return value;
}

template <int Width>
__device__ __forceinline__ float subwarp_sum(float value, unsigned mask) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, Width);
  }
  return value;
}

template <bool Canonical, typename index_t>
__device__ __forceinline__ long edge_at_position(
    long position, const index_t* destination_order) {
  if constexpr (Canonical) {
    return position;
  }
  return static_cast<long>(destination_order[position]);
}

// === Radial table ===

struct TableLocation {
  int index;
  float coordinate;
  bool clamped;
};

template <bool Canonical>
__device__ __forceinline__ TableLocation
locate_table(float radius, float stride, float table_max, int interval_count) {
  const float coordinate =
      Canonical ? radius : fminf(fmaxf(radius, 0.0f), table_max);
  int index = static_cast<int>(__fdividef(coordinate, stride));
  index = min(index, interval_count - 1);
  return {index, coordinate - static_cast<float>(index) * stride,
          !Canonical && radius >= table_max};
}

/// Address of the two coefficient blocks of one channel.
///
/// The interval row splits into a quartet block and a pair block, so the six
/// spline coefficients arrive in one 128-bit and one 64-bit load rather than
/// three 64-bit loads, at identical traffic. Both bases are resolved once per
/// edge, which turns every channel offset into a compile-time immediate.
struct TableRow {
  const float4* quartet;
  const float2* pair;
};

__device__ __forceinline__ TableRow table_row(const float* table,
                                              const TableLocation& location,
                                              int width) {
  const float* row = table + static_cast<long>(location.index) * width * 6;
  return {reinterpret_cast<const float4*>(row),
          reinterpret_cast<const float2*>(row + 4 * width)};
}

__device__ __forceinline__ float evaluate_table(const TableRow& row,
                                                int channel,
                                                float x) {
  const float4 low = __ldg(row.quartet + channel);
  const float2 high = __ldg(row.pair + channel);
  return low.x +
         (low.y + (low.z + (low.w + (high.x + high.y * x) * x) * x) * x) * x;
}

/// Evaluate the spline and its derivative as two independent Horner chains.
///
/// A simultaneous sweep that carries the derivative alongside the value issues
/// fewer instructions but serializes them into one chain of ten dependent
/// products; the two shorter chains below overlap and measure faster at the
/// widest channel profile.
__device__ __forceinline__ float2 evaluate_table_with_derivative(
    const TableRow& row, int channel, float x, bool clamped) {
  const float4 low = __ldg(row.quartet + channel);
  const float2 high = __ldg(row.pair + channel);
  const float value =
      low.x + (low.y + (low.z + (low.w + (high.x + high.y * x) * x) * x) * x) * x;
  const float derivative =
      low.y + (2.0f * low.z +
               (3.0f * low.w + (4.0f * high.x + 5.0f * high.y * x) * x) * x) *
                  x;
  return make_float2(value, clamped ? 0.0f : derivative);
}

/// Accumulate the pair-conditioned radial mode residual of one channel.
///
/// The mode axis is innermost in the ordered mixing cache, so a supported rank
/// of two, four, or eight is covered by at most one 128-bit and one 64-bit
/// global load, and the shared profile in one matching vector read. Splitting
/// the shared read into pairs or single elements lowers register pressure but
/// measures slower at the widest profiles, where the shared-memory
/// instruction count dominates.
__device__ __forceinline__ void accumulate_modes(const float* mixing,
                                                 const float* profile,
                                                 int rank,
                                                 float& value) {
  int mode = 0;
  for (; mode + 4 <= rank; mode += 4) {
    const float4 weight = __ldg(reinterpret_cast<const float4*>(mixing + mode));
    const float4 shape = *reinterpret_cast<const float4*>(profile + mode);
    value = fmaf(weight.x, shape.x, value);
    value = fmaf(weight.y, shape.y, value);
    value = fmaf(weight.z, shape.z, value);
    value = fmaf(weight.w, shape.w, value);
  }
  if (mode < rank) {
    const float2 weight = __ldg(reinterpret_cast<const float2*>(mixing + mode));
    const float2 shape = *reinterpret_cast<const float2*>(profile + mode);
    value = fmaf(weight.x, shape.x, value);
    value = fmaf(weight.y, shape.y, value);
  }
}

/// Accumulate the mode residual and its distance derivative together.
__device__ __forceinline__ void accumulate_modes_with_derivative(
    const float* mixing,
    const float* profile,
    const float* slope,
    int rank,
    float& value,
    float& derivative) {
  int mode = 0;
  for (; mode + 4 <= rank; mode += 4) {
    const float4 weight = __ldg(reinterpret_cast<const float4*>(mixing + mode));
    const float4 shape = *reinterpret_cast<const float4*>(profile + mode);
    const float4 rate = *reinterpret_cast<const float4*>(slope + mode);
    value = fmaf(weight.x, shape.x, value);
    value = fmaf(weight.y, shape.y, value);
    value = fmaf(weight.z, shape.z, value);
    value = fmaf(weight.w, shape.w, value);
    derivative = fmaf(weight.x, rate.x, derivative);
    derivative = fmaf(weight.y, rate.y, derivative);
    derivative = fmaf(weight.z, rate.z, derivative);
    derivative = fmaf(weight.w, rate.w, derivative);
  }
  if (mode < rank) {
    const float2 weight = __ldg(reinterpret_cast<const float2*>(mixing + mode));
    const float2 shape = *reinterpret_cast<const float2*>(profile + mode);
    const float2 rate = *reinterpret_cast<const float2*>(slope + mode);
    value = fmaf(weight.x, shape.x, value);
    value = fmaf(weight.y, shape.y, value);
    derivative = fmaf(weight.x, rate.x, derivative);
    derivative = fmaf(weight.y, rate.y, derivative);
  }
}

__device__ __forceinline__ float c3_envelope(float radius, float rcut) {
  const float u = fminf(fmaxf(__fdividef(rcut - radius, rcut), 0.0f), 1.0f);
  const float x = 1.0f - u;
  const float series =
      1.0f + x * (4.0f + x * (10.0f + x * (20.0f + 35.0f * x)));
  const float u2 = u * u;
  return u2 * u2 * series;
}

__device__ __forceinline__ float c3_envelope_derivative(float radius,
                                                        float rcut) {
  if (radius <= 0.0f || radius >= rcut) {
    return 0.0f;
  }
  const float x = __fdividef(radius, rcut);
  const float u = 1.0f - x;
  const float x2 = x * x;
  const float x3 = x2 * x;
  const float series =
      1.0f + 4.0f * x + 10.0f * x2 + 20.0f * x3 + 35.0f * x2 * x2;
  const float derivative = 4.0f + 20.0f * x + 60.0f * x2 + 140.0f * x3;
  const float u2 = u * u;
  return __fdividef(-4.0f * u2 * u * series + u2 * u2 * derivative, rcut);
}

// === Edge geometry and angular basis ===

struct EdgeGeometry {
  float ux;
  float uy;
  float uz;
  float radius;
  float inverse_radius;
  float envelope;
  int source_type;
};

template <bool Canonical, typename index_t>
__device__ __forceinline__ EdgeGeometry load_geometry(long edge,
                                                      float rcut,
                                                      float eps,
                                                      const float* edge_vec,
                                                      const index_t* edge_index,
                                                      const long* atype) {
  EdgeGeometry geometry;
  const long source = static_cast<long>(edge_index[edge]);
  geometry.source_type = static_cast<int>(atype[source]);
  const float x = edge_vec[edge * 3 + 0];
  const float y = edge_vec[edge * 3 + 1];
  const float z = edge_vec[edge * 3 + 2];
  const float square = x * x + y * y + z * z + eps * eps;
  geometry.inverse_radius = rsqrtf(square);
  geometry.radius = square * geometry.inverse_radius;
  geometry.ux = x * geometry.inverse_radius;
  geometry.uy = y * geometry.inverse_radius;
  geometry.uz = z * geometry.inverse_radius;
  geometry.envelope = c3_envelope(geometry.radius, rcut);
  return geometry;
}

// The Cartesian harmonics are evaluated on the unit direction, so the squared
// norm that makes each degree traceless is exactly one. Two polynomials that
// agree on the unit sphere differ by a multiple of `|u|^2 - 1`, whose gradient
// at a unit vector is purely radial and is therefore annihilated by the
// tangential projection that closes the coordinate VJP. Substituting the
// constant is thus exact on the unit sphere for both the value and the
// projected gradient. The regularized direction departs from unit norm by a
// relative `eps^2 / rho^2`, far below single precision at any physical
// separation.

// Degrees zero through two, which every profile evaluates.
__device__ __forceinline__ void fill_angular_basis(const EdgeGeometry& geometry,
                                                   float (&basis)[9]) {
  basis[0] = 1.0f;
  basis[1] = geometry.ux;
  basis[2] = geometry.uy;
  basis[3] = geometry.uz;
  basis[4] = kSqrtThree * geometry.ux * geometry.uy;
  basis[5] = kSqrtThree * geometry.uy * geometry.uz;
  basis[6] = 0.5f * (3.0f * geometry.uz * geometry.uz - 1.0f);
  basis[7] = kSqrtThree * geometry.ux * geometry.uz;
  basis[8] = 0.5f * kSqrtThree *
             (geometry.ux * geometry.ux - geometry.uy * geometry.uy);
}

// Real Cartesian harmonics of degrees three and four, addressed by the flat
// index `m` for degree three and `7 + m` for degree four. These degrees carry
// one channel, so their components are distributed across the lanes of an edge
// group. The caller therefore iterates the component at compile time and
// selects with a lane predicate, which folds the selector below into the one
// case that lane needs.
__device__ __forceinline__ float high_basis_value(const EdgeGeometry& geometry,
                                                  int index) {
  const float x = geometry.ux;
  const float y = geometry.uy;
  const float z = geometry.uz;
  const float z2 = z * z;
  const float difference = x * x - y * y;
  switch (index) {
    case 0:
      return 0.79056941504209483f * y * (3.0f * x * x - y * y);
    case 1:
      return 3.87298334620741689f * x * y * z;
    case 2:
      return 0.61237243569579452f * y * (5.0f * z2 - 1.0f);
    case 3:
      return 0.5f * z * (5.0f * z2 - 3.0f);
    case 4:
      return 0.61237243569579452f * x * (5.0f * z2 - 1.0f);
    case 5:
      return 1.93649167310370844f * z * difference;
    case 6:
      return 0.79056941504209483f * x * (x * x - 3.0f * y * y);
    case 7:
      return 2.95803989154980802f * x * y * difference;
    case 8:
      return 2.09165006633518887f * y * z * (3.0f * x * x - y * y);
    case 9:
      return 1.11803398874989485f * x * y * (7.0f * z2 - 1.0f);
    case 10:
      return 0.79056941504209483f * y * z * (7.0f * z2 - 3.0f);
    case 11:
      return 0.125f * (35.0f * z2 * z2 - 30.0f * z2 + 3.0f);
    case 12:
      return 0.79056941504209483f * x * z * (7.0f * z2 - 3.0f);
    case 13:
      return 0.55901699437494742f * difference * (7.0f * z2 - 1.0f);
    case 14:
      return 2.09165006633518887f * x * z * (x * x - 3.0f * y * y);
    default:
      return 0.73950997288745200f *
             (x * x * x * x - 6.0f * x * x * y * y + y * y * y * y);
  }
}

// Accumulate `weight * grad B_index` into the Cartesian derivative triple.
__device__ __forceinline__ void high_basis_gradient(
    const EdgeGeometry& geometry, int index, float weight, float (&du)[3]) {
  const float x = geometry.ux;
  const float y = geometry.uy;
  const float z = geometry.uz;
  const float z2 = z * z;
  const float difference = x * x - y * y;
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  switch (index) {
    case 0: {
      constexpr float k = 0.79056941504209483f;
      dx = 6.0f * k * x * y;
      dy = 3.0f * k * difference;
      break;
    }
    case 1: {
      constexpr float k = 3.87298334620741689f;
      dx = k * y * z;
      dy = k * x * z;
      dz = k * x * y;
      break;
    }
    case 2: {
      constexpr float k = 0.61237243569579452f;
      dy = k * (5.0f * z2 - 1.0f);
      dz = 10.0f * k * y * z;
      break;
    }
    case 3: {
      dz = 0.5f * (15.0f * z2 - 3.0f);
      break;
    }
    case 4: {
      constexpr float k = 0.61237243569579452f;
      dx = k * (5.0f * z2 - 1.0f);
      dz = 10.0f * k * x * z;
      break;
    }
    case 5: {
      constexpr float k = 1.93649167310370844f;
      dx = 2.0f * k * x * z;
      dy = -2.0f * k * y * z;
      dz = k * difference;
      break;
    }
    case 6: {
      constexpr float k = 0.79056941504209483f;
      dx = 3.0f * k * difference;
      dy = -6.0f * k * x * y;
      break;
    }
    case 7: {
      constexpr float k = 2.95803989154980802f;
      dx = k * y * (3.0f * x * x - y * y);
      dy = k * x * (x * x - 3.0f * y * y);
      break;
    }
    case 8: {
      constexpr float k = 2.09165006633518887f;
      dx = 6.0f * k * x * y * z;
      dy = 3.0f * k * z * difference;
      dz = k * y * (3.0f * x * x - y * y);
      break;
    }
    case 9: {
      constexpr float k = 1.11803398874989485f;
      dx = k * y * (7.0f * z2 - 1.0f);
      dy = k * x * (7.0f * z2 - 1.0f);
      dz = 14.0f * k * x * y * z;
      break;
    }
    case 10: {
      constexpr float k = 0.79056941504209483f;
      dy = k * z * (7.0f * z2 - 3.0f);
      dz = k * y * (21.0f * z2 - 3.0f);
      break;
    }
    case 11: {
      dz = 17.5f * z2 * z - 7.5f * z;
      break;
    }
    case 12: {
      constexpr float k = 0.79056941504209483f;
      dx = k * z * (7.0f * z2 - 3.0f);
      dz = k * x * (21.0f * z2 - 3.0f);
      break;
    }
    case 13: {
      constexpr float k = 0.55901699437494742f;
      dx = 2.0f * k * x * (7.0f * z2 - 1.0f);
      dy = -2.0f * k * y * (7.0f * z2 - 1.0f);
      dz = 14.0f * k * difference * z;
      break;
    }
    case 14: {
      constexpr float k = 2.09165006633518887f;
      dx = 3.0f * k * z * difference;
      dy = -6.0f * k * x * y * z;
      dz = k * x * (x * x - 3.0f * y * y);
      break;
    }
    default: {
      constexpr float k = 0.73950997288745200f;
      dx = 4.0f * k * x * (x * x - 3.0f * y * y);
      dy = -4.0f * k * y * (3.0f * x * x - y * y);
      break;
    }
  }
  du[0] = fmaf(weight, dx, du[0]);
  du[1] = fmaf(weight, dy, du[1]);
  du[2] = fmaf(weight, dz, du[2]);
}

// === Symmetric traceless degree-two algebra ===

struct Matrix3 {
  float value[3][3];
};

__device__ __forceinline__ Matrix3 packed_to_stf(const float (&packed)[5]) {
  Matrix3 matrix;
  matrix.value[0][0] = -packed[2] * kInvSqrtSix + packed[4] * kInvSqrtTwo;
  matrix.value[1][1] = -packed[2] * kInvSqrtSix - packed[4] * kInvSqrtTwo;
  matrix.value[2][2] = 2.0f * packed[2] * kInvSqrtSix;
  matrix.value[0][1] = matrix.value[1][0] = packed[0] * kInvSqrtTwo;
  matrix.value[1][2] = matrix.value[2][1] = packed[1] * kInvSqrtTwo;
  matrix.value[0][2] = matrix.value[2][0] = packed[3] * kInvSqrtTwo;
  return matrix;
}

__device__ __forceinline__ void matrix_vector(const Matrix3& matrix,
                                              const float (&vector)[3],
                                              float (&output)[3]) {
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    output[row] = 0.0f;
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      output[row] =
          fmaf(matrix.value[row][column], vector[column], output[row]);
    }
  }
}

__device__ __forceinline__ Matrix3 matrix_product(const Matrix3& left,
                                                  const Matrix3& right) {
  Matrix3 output{};
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
#pragma unroll
      for (int inner = 0; inner < 3; ++inner) {
        output.value[row][column] =
            fmaf(left.value[row][inner], right.value[inner][column],
                 output.value[row][column]);
      }
    }
  }
  return output;
}

__device__ __forceinline__ float matrix_trace(const Matrix3& matrix) {
  return matrix.value[0][0] + matrix.value[1][1] + matrix.value[2][2];
}

__device__ __forceinline__ void matrix_gradient_to_packed(const Matrix3& matrix,
                                                          float (&packed)[5]) {
  packed[0] = (matrix.value[0][1] + matrix.value[1][0]) * kInvSqrtTwo;
  packed[1] = (matrix.value[1][2] + matrix.value[2][1]) * kInvSqrtTwo;
  packed[2] =
      (-matrix.value[0][0] - matrix.value[1][1] + 2.0f * matrix.value[2][2]) *
      kInvSqrtSix;
  packed[3] = (matrix.value[0][2] + matrix.value[2][0]) * kInvSqrtTwo;
  packed[4] = (matrix.value[0][0] - matrix.value[1][1]) * kInvSqrtTwo;
}

// === Descriptor and readout helpers ===

__device__ __forceinline__ long gram_pair_position(int first,
                                                   int second,
                                                   int width) {
  const int row = min(first, second);
  const int column = max(first, second);
  return static_cast<long>(row) * width -
         static_cast<long>(row) * (row - 1) / 2 + column - row;
}

__device__ __forceinline__ void decode_upper_pair(int pair,
                                                  int width,
                                                  int& row,
                                                  int& column) {
  row = 0;
  while (pair >= width - row) {
    pair -= width - row;
    ++row;
  }
  column = row + pair;
}

__device__ __forceinline__ void store_descriptor(float* descriptor,
                                                 const float* mean,
                                                 const float* inverse_stddev,
                                                 long node,
                                                 int output_width,
                                                 int coordinate,
                                                 float value) {
  const long index = node * output_width + coordinate;
  descriptor[index] =
      (value - __ldg(mean + coordinate)) * __ldg(inverse_stddev + coordinate);
}

__device__ __forceinline__ float load_output_gradient(
    const float* gradient,
    const float* inverse_stddev,
    long node,
    int output_width,
    int coordinate) {
  return __ldg(gradient + node * output_width + coordinate) *
         __ldg(inverse_stddev + coordinate);
}

template <int Channels, int Lmax>
__device__ __forceinline__ float readout_weight(const float* matrices,
                                                int matrix,
                                                int row,
                                                int column) {
  using P = Profile<Channels, Lmax>;
  return __ldg(matrices + (static_cast<long>(matrix) * P::C1 + row) * P::C1 +
               column);
}

// Probe coordinate of one degree. Degrees three and above carry a single
// channel whose alignment and probe projections are both the identity, so
// their probes are the stored moments.
template <int Channels, int Lmax>
__device__ __forceinline__ float probe_value(const float* probes,
                                             const float* moments,
                                             int degree,
                                             int component,
                                             int rank_index) {
  using P = Profile<Channels, Lmax>;
  if (degree == 1) {
    return probes[component * P::K1 + rank_index];
  }
  if (degree == 2) {
    return probes[3 * P::K1 + component * P::K2 + rank_index];
  }
  return moments[P::HighOffset + (degree == 3 ? 0 : P::High3) + component];
}

// The Cartesian basis VJP maps angular cotangents to a coordinate gradient.
// Applying it per lane reduces three Cartesian components instead of the full
// set of angular components across the edge group.
__device__ __forceinline__ void basis_vjp(const EdgeGeometry& geometry,
                                          const float (&d_basis)[9],
                                          const float (&high_du)[3],
                                          float radial_gradient,
                                          float (&output)[3]) {
  const float dux =
      high_du[0] + d_basis[1] +
      kSqrtThree * (d_basis[4] * geometry.uy + d_basis[7] * geometry.uz +
                    d_basis[8] * geometry.ux);
  const float duy =
      high_du[1] + d_basis[2] +
      kSqrtThree * (d_basis[4] * geometry.ux + d_basis[5] * geometry.uz -
                    d_basis[8] * geometry.uy);
  const float duz =
      high_du[2] + d_basis[3] +
      kSqrtThree * (d_basis[5] * geometry.uy + d_basis[7] * geometry.ux) +
      3.0f * d_basis[6] * geometry.uz;
  const float dot = geometry.ux * dux + geometry.uy * duy + geometry.uz * duz;
  output[0] = (dux - geometry.ux * dot) * geometry.inverse_radius +
              radial_gradient * geometry.ux;
  output[1] = (duy - geometry.uy * dot) * geometry.inverse_radius +
              radial_gradient * geometry.uy;
  output[2] = (duz - geometry.uz * dot) * geometry.inverse_radius +
              radial_gradient * geometry.uz;
}

}  // namespace deepmd_dpa4c
