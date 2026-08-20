// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Compact Cartesian real spherical harmonics for DPA1 moment aggregation.
//
// Rows [l^2, (l+1)^2) contain degree l in m=-l,...,l order. Degrees 2-4
// use norm-normalized regular solid harmonics. On unit vectors, their inner
// product equals P_l(u dot v).

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
DPA1_MOMENT_INLINE void fill_angular_basis(
    float* basis, float x, float y, float z, float radial) {
  if constexpr (BasisDim > 4) {
    float angular[BasisDim - 4];
    evaluate_angular_basis<float, BasisDim>(x, y, z, angular);
#pragma unroll
    for (int row = 0; row < BasisDim - 4; ++row) {
      basis[4 + row] = radial * angular[row];
    }
  }
}

template <int BasisDim>
DPA1_MOMENT_INLINE void angular_basis_vjp(const float (&d_basis)[BasisDim],
                                          float inverse_neighbors,
                                          float x,
                                          float y,
                                          float z,
                                          float& radial_partial,
                                          float& grad_x,
                                          float& grad_y,
                                          float& grad_z) {
  if constexpr (BasisDim > 4) {
    const Jet3 x_jet{x, 1.0f, 0.0f, 0.0f};
    const Jet3 y_jet{y, 0.0f, 1.0f, 0.0f};
    const Jet3 z_jet{z, 0.0f, 0.0f, 1.0f};
    Jet3 angular[BasisDim - 4];
    evaluate_angular_basis<Jet3, BasisDim>(x_jet, y_jet, z_jet, angular);
#pragma unroll
    for (int row = 0; row < BasisDim - 4; ++row) {
      const float gradient = d_basis[4 + row] * inverse_neighbors;
      radial_partial = fmaf(gradient, angular[row].value, radial_partial);
      grad_x = fmaf(gradient, angular[row].dx, grad_x);
      grad_y = fmaf(gradient, angular[row].dy, grad_y);
      grad_z = fmaf(gradient, angular[row].dz, grad_z);
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
    const float nx = x * inverse_radius;
    const float ny = y * inverse_radius;
    const float nz = z * inverse_radius;
    const float vx = x * inverse_denominator;
    const float vy = y * inverse_denominator;
    const float vz = z * inverse_denominator;
    float radial_partial = 0.0f;
    float grad_vx = 0.0f;
    float grad_vy = 0.0f;
    float grad_vz = 0.0f;
    angular_basis_vjp<BasisDim>(d_basis, inverse_neighbors, vx, vy, vz,
                                radial_partial, grad_vx, grad_vy, grad_vz);
    const float protected_dot = grad_vx * vx + grad_vy * vy + grad_vz * vz;
    const float amplitude =
        switch_value * inverse_denominator * inverse_stddev0;
    const float amplitude_gradient =
        inverse_stddev0 * inverse_denominator *
        (switch_gradient - switch_value * inverse_denominator);
    output_x +=
        radial_partial * amplitude_gradient * nx +
        amplitude * inverse_denominator * (grad_vx - protected_dot * nx);
    output_y +=
        radial_partial * amplitude_gradient * ny +
        amplitude * inverse_denominator * (grad_vy - protected_dot * ny);
    output_z +=
        radial_partial * amplitude_gradient * nz +
        amplitude * inverse_denominator * (grad_vz - protected_dot * nz);
  }
}

#undef DPA1_MOMENT_INLINE

}  // namespace deepmd::dpa1
