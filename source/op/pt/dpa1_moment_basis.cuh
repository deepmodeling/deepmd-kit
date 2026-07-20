// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Compact Cartesian real spherical harmonics for DPA1 moment aggregation.
//
// Rows [l^2, (l+1)^2) contain degree l in m=-l,...,l order. Degrees 2-4
// use norm normalization, so their inner product equals P_l(u dot v).

#pragma once

#include <cuda_runtime.h>

namespace deepmd::dpa1 {

#define DPA1_MOMENT_INLINE __device__ __forceinline__

struct Jet3 {
  float value;
  float dx;
  float dy;
  float dz;
};

DPA1_MOMENT_INLINE float degree_weight(
    int row, const float* __restrict__ degree_gain_raw) {
  if (row < 4) {
    return 1.0f;
  }
  const int degree_index = row < 9 ? 0 : (row < 16 ? 1 : 2);
  const float gain = __ldg(degree_gain_raw + degree_index);
  return gain * gain;
}

DPA1_MOMENT_INLINE Jet3 operator+(const Jet3& a, const Jet3& b) {
  return {a.value + b.value, a.dx + b.dx, a.dy + b.dy, a.dz + b.dz};
}

DPA1_MOMENT_INLINE Jet3 operator-(const Jet3& a, const Jet3& b) {
  return {a.value - b.value, a.dx - b.dx, a.dy - b.dy, a.dz - b.dz};
}

DPA1_MOMENT_INLINE Jet3 operator*(const Jet3& a, const Jet3& b) {
  return {a.value * b.value, a.dx * b.value + a.value * b.dx,
          a.dy * b.value + a.value * b.dy, a.dz * b.value + a.value * b.dz};
}

DPA1_MOMENT_INLINE Jet3 operator*(float scale, const Jet3& value) {
  return {scale * value.value, scale * value.dx, scale * value.dy,
          scale * value.dz};
}

DPA1_MOMENT_INLINE Jet3 operator*(const Jet3& value, float scale) {
  return scale * value;
}

template <typename T, int BasisDim>
DPA1_MOMENT_INLINE void evaluate_angular_basis(const T& x,
                                               const T& y,
                                               const T& z,
                                               T* output) {
  static_assert(BasisDim == 4 || BasisDim == 9 || BasisDim == 16 ||
                BasisDim == 25);
  if constexpr (BasisDim == 4) {
    return;
  }

  constexpr float sqrt3 = 1.7320508075688772935f;
  const T x2 = x * x;
  const T y2 = y * y;
  const T z2 = z * z;
  const T q = x2 + y2 + z2;
  const T x2_minus_y2 = x2 - y2;
  output[0] = sqrt3 * x * y;
  output[1] = sqrt3 * y * z;
  output[2] = 0.5f * (3.0f * z2 - q);
  output[3] = sqrt3 * x * z;
  output[4] = (0.5f * sqrt3) * x2_minus_y2;

  if constexpr (BasisDim >= 16) {
    constexpr float sqrt5_over8 = 0.79056941504209483299f;
    constexpr float sqrt15 = 3.8729833462074168852f;
    constexpr float sqrt3_over8 = 0.61237243569579452455f;
    output[5] = sqrt5_over8 * y * (3.0f * x2 - y2);
    output[6] = sqrt15 * x * y * z;
    output[7] = sqrt3_over8 * y * (5.0f * z2 - q);
    output[8] = 0.5f * z * (5.0f * z2 - 3.0f * q);
    output[9] = sqrt3_over8 * x * (5.0f * z2 - q);
    output[10] = (0.5f * sqrt15) * z * x2_minus_y2;
    output[11] = sqrt5_over8 * x * (x2 - 3.0f * y2);
  }

  if constexpr (BasisDim == 25) {
    constexpr float sqrt35 = 5.9160797830996160426f;
    constexpr float sqrt70 = 8.3666002653407554798f;
    constexpr float sqrt5 = 2.2360679774997896964f;
    constexpr float sqrt10 = 3.1622776601683793320f;
    output[12] = (0.5f * sqrt35) * x * y * x2_minus_y2;
    output[13] = (0.25f * sqrt70) * y * z * (3.0f * x2 - y2);
    output[14] = (0.5f * sqrt5) * x * y * (7.0f * z2 - q);
    output[15] = (0.25f * sqrt10) * y * z * (7.0f * z2 - 3.0f * q);
    output[16] = 0.125f * (35.0f * z2 * z2 - 30.0f * z2 * q + 3.0f * q * q);
    output[17] = (0.25f * sqrt10) * x * z * (7.0f * z2 - 3.0f * q);
    output[18] = (0.25f * sqrt5) * x2_minus_y2 * (7.0f * z2 - q);
    output[19] = (0.25f * sqrt70) * x * z * (x2 - 3.0f * y2);
    output[20] = (0.125f * sqrt35) * (x2 * x2 - 6.0f * x2 * y2 + y2 * y2);
  }
}

template <int BasisDim>
DPA1_MOMENT_INLINE void fill_degree_two_basis(
    float* basis, float ux, float uy, float uz, float radial) {
  static_assert(BasisDim == 9);
  constexpr float sqrt3 = 1.7320508075688772935f;
  basis[4] = radial * sqrt3 * ux * uy;
  basis[5] = radial * sqrt3 * uy * uz;
  basis[6] = radial * 0.5f * (3.0f * uz * uz - 1.0f);
  basis[7] = radial * sqrt3 * ux * uz;
  basis[8] = radial * 0.5f * sqrt3 * (ux * ux - uy * uy);
}

template <int BasisDim>
DPA1_MOMENT_INLINE void fill_angular_basis(
    float* basis, float ux, float uy, float uz, float radial) {
  if constexpr (BasisDim > 4) {
    float angular[BasisDim - 4];
    evaluate_angular_basis<float, BasisDim>(ux, uy, uz, angular);
#pragma unroll
    for (int row = 0; row < BasisDim - 4; ++row) {
      basis[4 + row] = radial * angular[row];
    }
  }
}

template <int BasisDim>
DPA1_MOMENT_INLINE void add_degree_two_edge_gradient(
    const float (&d_basis)[BasisDim],
    float inverse_neighbors,
    float x,
    float y,
    float z,
    float radius,
    float inverse_denominator,
    float inverse_stddev0,
    float switch_value,
    float switch_gradient,
    float& output_x,
    float& output_y,
    float& output_z) {
  static_assert(BasisDim == 9);
  if (radius <= 0.0f) {
    return;
  }
  const float inverse_radius = 1.0f / radius;
  const float ux = x * inverse_radius;
  const float uy = y * inverse_radius;
  const float uz = z * inverse_radius;
  constexpr float sqrt3 = 1.7320508075688772935f;
  const float y2[5] = {
      sqrt3 * ux * uy,
      sqrt3 * uy * uz,
      0.5f * (3.0f * uz * uz - 1.0f),
      sqrt3 * ux * uz,
      0.5f * sqrt3 * (ux * ux - uy * uy),
  };
  float radial_partial = 0.0f;
#pragma unroll
  for (int row = 0; row < 5; ++row) {
    radial_partial =
        fmaf(d_basis[4 + row] * inverse_neighbors, y2[row], radial_partial);
  }
  const float d4 = d_basis[4] * inverse_neighbors;
  const float d5 = d_basis[5] * inverse_neighbors;
  const float d6 = d_basis[6] * inverse_neighbors;
  const float d7 = d_basis[7] * inverse_neighbors;
  const float d8 = d_basis[8] * inverse_neighbors;
  const float grad_ux = sqrt3 * (d4 * uy + d7 * uz + d8 * ux);
  const float grad_uy = sqrt3 * (d4 * ux + d5 * uz - d8 * uy);
  const float grad_uz = sqrt3 * (d5 * uy + d7 * ux) + 3.0f * d6 * uz;
  const float unit_dot = grad_ux * ux + grad_uy * uy + grad_uz * uz;
  const float amplitude = switch_value * inverse_denominator * inverse_stddev0;
  const float amplitude_gradient =
      inverse_stddev0 * inverse_denominator *
      (switch_gradient - switch_value * inverse_denominator);
  output_x += radial_partial * amplitude_gradient * ux +
              amplitude * inverse_radius * (grad_ux - unit_dot * ux);
  output_y += radial_partial * amplitude_gradient * uy +
              amplitude * inverse_radius * (grad_uy - unit_dot * uy);
  output_z += radial_partial * amplitude_gradient * uz +
              amplitude * inverse_radius * (grad_uz - unit_dot * uz);
}

template <int BasisDim>
DPA1_MOMENT_INLINE void angular_basis_vjp(const float (&d_basis)[BasisDim],
                                          float inverse_neighbors,
                                          float ux,
                                          float uy,
                                          float uz,
                                          float& radial_partial,
                                          float& grad_ux,
                                          float& grad_uy,
                                          float& grad_uz) {
  if constexpr (BasisDim > 4) {
    const Jet3 x{ux, 1.0f, 0.0f, 0.0f};
    const Jet3 y{uy, 0.0f, 1.0f, 0.0f};
    const Jet3 z{uz, 0.0f, 0.0f, 1.0f};
    Jet3 angular[BasisDim - 4];
    evaluate_angular_basis<Jet3, BasisDim>(x, y, z, angular);
#pragma unroll
    for (int row = 0; row < BasisDim - 4; ++row) {
      const float gradient = d_basis[4 + row] * inverse_neighbors;
      radial_partial = fmaf(gradient, angular[row].value, radial_partial);
      grad_ux = fmaf(gradient, angular[row].dx, grad_ux);
      grad_uy = fmaf(gradient, angular[row].dy, grad_uy);
      grad_uz = fmaf(gradient, angular[row].dz, grad_uz);
    }
  }
}

template <int BasisDim>
DPA1_MOMENT_INLINE void add_angular_edge_gradient(
    const float (&d_basis)[BasisDim],
    float inverse_neighbors,
    float x,
    float y,
    float z,
    float radius,
    float inverse_denominator,
    float inverse_stddev0,
    float switch_value,
    float switch_gradient,
    float& output_x,
    float& output_y,
    float& output_z) {
  if constexpr (BasisDim > 4) {
    if (radius <= 0.0f) {
      return;
    }
    const float inverse_radius = 1.0f / radius;
    const float ux = x * inverse_radius;
    const float uy = y * inverse_radius;
    const float uz = z * inverse_radius;
    float radial_partial = 0.0f;
    float grad_ux = 0.0f;
    float grad_uy = 0.0f;
    float grad_uz = 0.0f;
    angular_basis_vjp<BasisDim>(d_basis, inverse_neighbors, ux, uy, uz,
                                radial_partial, grad_ux, grad_uy, grad_uz);
    const float unit_dot = grad_ux * ux + grad_uy * uy + grad_uz * uz;
    const float amplitude =
        switch_value * inverse_denominator * inverse_stddev0;
    const float amplitude_gradient =
        inverse_stddev0 * inverse_denominator *
        (switch_gradient - switch_value * inverse_denominator);
    output_x += radial_partial * amplitude_gradient * ux +
                amplitude * inverse_radius * (grad_ux - unit_dot * ux);
    output_y += radial_partial * amplitude_gradient * uy +
                amplitude * inverse_radius * (grad_uy - unit_dot * uy);
    output_z += radial_partial * amplitude_gradient * uz +
                amplitude * inverse_radius * (grad_uz - unit_dot * uz);
  }
}

#undef DPA1_MOMENT_INLINE

}  // namespace deepmd::dpa1
