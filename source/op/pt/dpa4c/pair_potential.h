// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cmath>

namespace deepmd_dpa4c {

/// Aligned row: four amplitudes A, four rates c, cutoff R, -V'''(R)/6,
/// two padding floats and four endpoint amplitudes B = A exp(-cR).
constexpr int kPairRow = 16;

struct PairTerm {
  float energy;
  float slope;
};

#ifdef __CUDACC__
#define DPA4C_PAIR_INLINE __host__ __device__ __forceinline__
#else
#define DPA4C_PAIR_INLINE inline
#endif

DPA4C_PAIR_INLINE float screened_remainder(
    float amplitude, float rate, float endpoint, float radius, float y) {
  if (y <= 1.0f) {
    // Sum y^4/4! through y^12/12! in Horner form. The relative tail
    // is below single-precision roundoff on this interval.
    float tail = 1.0f;
    for (int k = 12; k > 4; --k) {
      tail = 1.0f + y * (tail / static_cast<float>(k));
    }
    const float y2 = y * y;
    return endpoint * (y2 * y2) * (tail / 24.0f);
  }
  const float cubic = 1.0f + y * (1.0f + y * (0.5f + y / 6.0f));
  return amplitude * expf(-rate * radius) - endpoint * cubic;
}

/// Evaluate a subset of the four screening terms and its radial derivative.
/// CPU scans take all four; cooperating GPU lanes take interleaved terms.
template <int Width>
DPA4C_PAIR_INLINE PairTerm
pair_term(const float* row, int lane, float radius, float inverse) {
  const float h = row[8] - radius;
  if (h <= 0.0f) {
    return {0.0f, 0.0f};
  }
  // r*phi = -V'''(R)*h^4/6 + sum B*(exp(ch)-T3(ch)).
  const float cubic = lane == 0 ? row[9] * h * h * h : 0.0f;
  float numerator = cubic * h;
  float derivative = 4.0f * cubic;
  for (int k = lane; k < 4; k += Width) {
    const float rate = row[4 + k];
    const float endpoint = row[12 + k];
    const float y = rate * h;
    const float value = screened_remainder(row[k], rate, endpoint, radius, y);
    numerator += value;
    // d/dh B*(exp(y)-T3(y)) = c*B*(exp(y)-T2(y)).
    derivative += rate * (value + endpoint * (y * y * y / 6.0f));
  }
  const float energy = numerator * inverse;
  return {energy, -(derivative + energy) * inverse};
}

#undef DPA4C_PAIR_INLINE

}  // namespace deepmd_dpa4c
