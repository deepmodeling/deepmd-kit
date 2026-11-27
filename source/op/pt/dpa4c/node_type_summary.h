// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

namespace deepmd::dpa4c {

// Padding has no element. Every real node, including a ghost, contributes
// to the bitmap and to the single-type dispatch decision.
template <int Words>
#ifdef __CUDACC__
__host__ __device__
#endif
    inline bool scan_node_type(long type,
                               long first,
                               unsigned (&present)[Words]) {
  if (type < 0) {
    return true;
  }
  for (int word = 0; word < Words; ++word) {
    if ((type >> 5) == word) {
      present[word] |= 1u << (type & 31);
    }
  }
  return type == first;
}

}  // namespace deepmd::dpa4c
