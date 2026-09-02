// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace deepmd {

// Vectorizable float32 transcendentals for the fitting epilogues.
//
// ``std::tanh`` and ``std::exp`` are opaque library calls, so a loop that
// contains one cannot be vectorized at all and pays the scalar latency of a
// libm evaluation per element. The fitting network of a released DPA4C grade
// evaluates its activation several million times per step -- three hidden
// layers of a few hundred channels over every atom, twice, because the
// backward re-derives the derivative from the pre-activation -- which made the
// activation the largest single term of the fitting network.
//
// The replacements are the standard Cephes minimax forms, written as plain
// expressions so that the compiler vectorizes the loop around them. Both are
// accurate to one unit in the last place across the float32 range and are
// smooth wherever the function they approximate is, which a potential-energy
// surface requires.

namespace detail {

/// Reinterpret an integer bit pattern as a float.
inline float bits_to_float(std::int32_t bits) {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace detail

/**
 * @brief Natural exponential, float32 minimax approximation.
 *
 * The argument is split as ``x = m * ln 2 + r`` with integer ``m`` and
 * ``|r| <= ln2 / 2``; a degree-5 polynomial covers ``exp(r)`` and the power of
 * two is applied by constructing its exponent field directly. The Cody-Waite
 * two-term split of ``ln 2`` keeps the reduction exact enough that the
 * polynomial's error dominates.
 *
 * @param x Argument.
 * @return exp(x), to one unit in the last place; zero or infinity outside the
 *   float32 range.
 */
inline float fast_exp(float x) {
  //: Largest magnitude whose exponential is a finite float32.
  constexpr float kRange = 88.3762626647950F;
  constexpr float kLog2E = 1.44269504088896341F;
  constexpr float kLn2High = 0.693145751953125F;
  constexpr float kLn2Low = 1.428606765330187e-06F;

  // ``fmin``/``fmax`` rather than ``std::min``/``std::max``: the latter are
  // conditional expressions, and a loop containing one next to this much
  // arithmetic exceeds what the compiler will if-convert, which costs the
  // vectorization of the whole loop. These two lower to a single instruction
  // each and are defined for a quiet NaN, so the integer conversion below
  // cannot see one.
  const float clamped = std::fmin(std::fmax(x, -kRange), kRange);
  const float scaled = std::floor(clamped * kLog2E + 0.5F);
  const float remainder = clamped - scaled * kLn2High - scaled * kLn2Low;
  const float square = remainder * remainder;
  float series = 1.9875691500e-4F;
  series = series * remainder + 1.3981999507e-3F;
  series = series * remainder + 8.3334519073e-3F;
  series = series * remainder + 4.1665795894e-2F;
  series = series * remainder + 1.6666665459e-1F;
  series = series * remainder + 5.0000001201e-1F;
  const float polynomial = 1.0F + remainder + square * series;
  const auto exponent = static_cast<std::int32_t>(scaled);
  return polynomial * detail::bits_to_float((exponent + 127) << 23);
}

/// Logistic function built on ``fast_exp``.
inline float fast_sigmoid(float x) { return 1.0F / (1.0F + fast_exp(-x)); }

/**
 * @brief Hyperbolic tangent, float32, monotone by construction.
 *
 * Cephes single-precision form: an odd degree-11 polynomial near the origin,
 * and ``1 - 2 / (exp(2|x|) + 1)`` beyond, sign-applied. The outer branch is
 * monotone because the exponential is, which a minimax rational approximation
 * over the whole line is not: a rational form accurate to 3 units in the last
 * place still reverses direction wherever its error curve turns, and a
 * potential-energy surface cannot carry a non-monotone activation.
 *
 * Both branches are evaluated and selected, so the loop around this function
 * stays free of control flow and vectorizes.
 *
 * @param x Argument.
 * @return tanh(x), to one unit in the last place.
 */
inline float fast_tanh(float x) {
  //: Below this magnitude the polynomial branch is the accurate one.
  constexpr float kCrossover = 0.625F;
  constexpr float kP0 = -5.70498872745e-03F;
  constexpr float kP1 = 2.06390887954e-02F;
  constexpr float kP2 = -5.37397155531e-02F;
  constexpr float kP3 = 1.33314422036e-01F;
  constexpr float kP4 = -3.33332819422e-01F;

  const float magnitude = std::abs(x);
  const float scaled = fast_exp(magnitude + magnitude);
  const float saturating = 1.0F - 2.0F / (scaled + 1.0F);
  // Bound the unselected polynomial branch so its degree-11 term stays finite;
  // multiplying an overflowing branch by a zero selector would produce NaN.
  const float bounded = std::copysign(std::fmin(magnitude, kCrossover), x);
  const float square = bounded * bounded;
  float series = kP0;
  series = series * square + kP1;
  series = series * square + kP2;
  series = series * square + kP3;
  series = series * square + kP4;
  const float central = series * square * bounded + bounded;
  // Select arithmetically. Any form that reaches the compiler as a
  // conditional -- a ternary, or a bool cast to float -- is control flow next
  // to this much inlined arithmetic, and costs the vectorization of the whole
  // loop. The sign of ``magnitude - kCrossover`` carries the same predicate
  // through ``copysign``, which is one branchless instruction; at exactly the
  // crossover the sign is positive, matching the closed inequality.
  const float pick =
      0.5F * (1.0F + std::copysign(1.0F, magnitude - kCrossover));
  return pick * std::copysign(saturating, x) + (1.0F - pick) * central;
}

}  // namespace deepmd
