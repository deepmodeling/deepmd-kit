// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <stdint.h>

/** Width of node indices in bytes; edge counts and CSR positions are int64. */
typedef enum DP_CanonicalIndexWidth {
  DP_CANONICAL_INDEX32 = 4,
  DP_CANONICAL_INDEX64 = 8
} DP_CanonicalIndexWidth;

#ifdef __cplusplus
namespace deepmd {
/** Select node-index width, preserving signed words at native host boundaries.
 */
inline DP_CanonicalIndexWidth canonical_index_width(int64_t node_count) {
  return node_count <= (int64_t{1} << 31) ? DP_CANONICAL_INDEX32
                                          : DP_CANONICAL_INDEX64;
}
}  // namespace deepmd
#endif
