// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace deepmd {

// Multiply non-negative tensor dimensions without invoking signed overflow.
inline bool tabulate_checked_product(const int64_t lhs,
                                     const int64_t rhs,
                                     int64_t& product) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return false;
  }
  product = lhs * rhs;
  return true;
}

// Validate the five table metadata values consumed by the native tabulation
// kernels and reproduce the spline-row count used by the table generator.
// Keeping this calculation shared prevents the TensorFlow and PyTorch wrappers
// from accepting different raw-buffer contracts.
template <typename FPTYPE>
bool tabulate_required_table_rows(const FPTYPE* table_info,
                                  const bool symmetric_range,
                                  int64_t& required_rows,
                                  std::string& error) {
  const FPTYPE lower_value = table_info[0];
  const FPTYPE upper_value = table_info[1];
  const FPTYPE max_value = table_info[2];
  const FPTYPE stride0_value = table_info[3];
  const FPTYPE stride1_value = table_info[4];
  const double lower = static_cast<double>(lower_value);
  const double upper = static_cast<double>(upper_value);
  const double max = static_cast<double>(max_value);
  const double stride0 = static_cast<double>(stride0_value);
  const double stride1 = static_cast<double>(stride1_value);
  if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(max) ||
      !std::isfinite(stride0) || !std::isfinite(stride1)) {
    error = "table_info values must be finite";
    return false;
  }
  if (stride0 <= 0.0 || stride1 <= 0.0) {
    error = "table_info strides must be positive";
    return false;
  }

  const double min = symmetric_range ? -max : lower;
  if (min > lower || lower > upper || upper > max) {
    error = symmetric_range
                ? "table_info must satisfy -max <= lower <= upper <= max"
                : "table_info must satisfy lower <= upper <= max";
    return false;
  }

  const double total_intervals =
      (symmetric_range ? (lower - min) / stride1 : 0.0) +
      (upper - lower) / stride0 + (max - upper) / stride1;
  if (!std::isfinite(total_intervals) || total_intervals <= 0.0) {
    error = "table_info must describe at least one spline interval";
    return false;
  }

  // Native locators truncate each region in FPTYPE precision and clamp the
  // high-tail branch with nextafter. Reproduce those operations exactly and
  // size for the largest reachable table index, rather than flooring the sum
  // of fractional interval counts once.
  const FPTYPE min_value = symmetric_range ? -max_value : lower_value;
  const double native_int_max =
      static_cast<double>(std::numeric_limits<int>::max());
  auto truncate_native_ratio = [&](const FPTYPE numerator,
                                   const FPTYPE denominator,
                                   int64_t& result) -> bool {
    const FPTYPE ratio_value = numerator / denominator;
    const double ratio = static_cast<double>(ratio_value);
    if (!std::isfinite(ratio) || ratio < 0.0 || ratio > native_int_max) {
      error = "table_info describes an invalid spline index";
      return false;
    }
    result = static_cast<int>(ratio_value);
    return true;
  };

  int64_t lower_count = 0;
  int64_t middle_count = 0;
  if ((symmetric_range &&
       !truncate_native_ratio(lower_value - min_value, stride1_value,
                              lower_count)) ||
      !truncate_native_ratio(upper_value - lower_value, stride0_value,
                             middle_count)) {
    return false;
  }
  const int64_t first_upper = lower_count + middle_count;
  if (first_upper > std::numeric_limits<int>::max()) {
    error = "table_info describes too many spline intervals";
    return false;
  }

  int64_t max_reachable_index = 0;
  auto include_region_end = [&](const FPTYPE end,
                                const FPTYPE start,
                                const FPTYPE stride,
                                const int64_t base) -> bool {
    if (!(end > start)) {
      return true;
    }
    int64_t offset = 0;
    if (!truncate_native_ratio(std::nextafter(end, start) - start, stride,
                               offset)) {
      return false;
    }
    const int64_t candidate = base + offset;
    if (candidate > std::numeric_limits<int>::max()) {
      error = "table_info describes too many spline intervals";
      return false;
    }
    max_reachable_index = std::max(max_reachable_index, candidate);
    return true;
  };

  if ((symmetric_range &&
       !include_region_end(lower_value, min_value, stride1_value, 0)) ||
      !include_region_end(upper_value, lower_value, stride0_value,
                          lower_count) ||
      !include_region_end(max_value, upper_value, stride1_value,
                          first_upper)) {
    return false;
  }

  // Inputs at or above max use this exact high-tail index. When max == upper,
  // it deliberately selects the row after an aligned middle region.
  int64_t high_tail_offset = 0;
  const FPTYPE high_tail_boundary = std::nextafter(max_value, min_value);
  const FPTYPE high_tail_delta = high_tail_boundary - upper_value;
  const FPTYPE high_tail_ratio = high_tail_delta / stride1_value;
  const double high_tail_ratio_double = static_cast<double>(high_tail_ratio);
  if (!std::isfinite(high_tail_ratio_double) ||
      high_tail_ratio_double <
          static_cast<double>(std::numeric_limits<int>::min()) ||
      high_tail_ratio_double > native_int_max) {
    error = "table_info describes an invalid spline index";
    return false;
  }
  high_tail_offset = static_cast<int>(high_tail_ratio);
  const int64_t high_tail_index = first_upper + high_tail_offset;
  if (high_tail_index < 0 ||
      high_tail_index > std::numeric_limits<int>::max()) {
    error = "table_info describes an invalid spline index";
    return false;
  }
  max_reachable_index = std::max(max_reachable_index, high_tail_index);
  required_rows = max_reachable_index + 1;
  return true;
}

// Convert the validated row count into the flattened coefficient count while
// guarding the multiplication used by both framework wrappers.
inline bool tabulate_required_table_elements(const int64_t required_rows,
                                             const int64_t last_layer_size,
                                             int64_t& required_elements,
                                             std::string& error) {
  constexpr int64_t coefficients_per_feature = 6;
  if (required_rows <= 0 || last_layer_size <= 0) {
    error = "table dimensions must be positive";
    return false;
  }
  int64_t feature_elements = 0;
  if (!tabulate_checked_product(last_layer_size, coefficients_per_feature,
                                feature_elements) ||
      !tabulate_checked_product(required_rows, feature_elements,
                                required_elements)) {
    error = "required table size exceeds the supported integer range";
    return false;
  }
  return true;
}

}  // namespace deepmd
