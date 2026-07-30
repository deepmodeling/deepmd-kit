// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

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
  const double lower = static_cast<double>(table_info[0]);
  const double upper = static_cast<double>(table_info[1]);
  const double max = static_cast<double>(table_info[2]);
  const double stride0 = static_cast<double>(table_info[3]);
  const double stride1 = static_cast<double>(table_info[4]);
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

  const double lower_tail = symmetric_range ? (lower - min) / stride1 : 0.0;
  const double middle = (upper - lower) / stride0;
  const double upper_tail = (max - upper) / stride1;
  const double total_intervals = lower_tail + middle + upper_tail;
  const double max_segment =
      static_cast<double>(std::numeric_limits<int>::max());
  if (!std::isfinite(lower_tail) || !std::isfinite(middle) ||
      !std::isfinite(upper_tail) || !std::isfinite(total_intervals) ||
      total_intervals > max_segment) {
    error = "table_info describes too many spline intervals";
    return false;
  }

  // The Python table builder converts the sum to an integer once, which is
  // observably different from truncating each range separately for SE-T.
  required_rows = static_cast<int64_t>(total_intervals);
  if (required_rows <= 0) {
    error = "table_info must describe at least one spline interval";
    return false;
  }
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
