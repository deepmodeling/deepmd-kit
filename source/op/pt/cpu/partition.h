// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Work partitioning for CSR-grouped edge reductions.
//
// Every CPU kernel of the graph lower reduces a contiguous run of edges onto
// the node that owns them, so a thread must receive whole nodes. Splitting
// the node axis evenly is wrong whenever the degree distribution is not: a
// slab surface, a molecular box, or a padded frame leaves some threads with
// several times the work of others, and the reduction is a barrier. The
// partition below equalizes the edge count instead, which is what the kernel
// cost is proportional to.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace deepmd_cpu {

/// Contiguous half-open node range assigned to one thread.
struct NodeRange {
  int64_t begin;
  int64_t end;
};

/// Split the node axis into ranges of roughly equal edge count.
///
/// The row pointers are non-decreasing, so the node that owns edge `k` is
/// found by one binary search. Ranges are emitted in node order and cover
/// `[0, node_count)` exactly; a range may be empty when the requested part
/// count exceeds the node count.
///
/// \param row_ptr CSR offsets with `node_count + 1` entries.
/// \param node_count Number of nodes.
/// \param parts Requested number of ranges, clamped to at least one.
/// \return Node ranges in ascending order.
inline std::vector<NodeRange> balanced_ranges(const int64_t* row_ptr,
                                              int64_t node_count,
                                              int parts) {
  parts = std::max(parts, 1);
  std::vector<NodeRange> ranges;
  ranges.reserve(static_cast<size_t>(parts));
  const int64_t edge_count = node_count > 0 ? row_ptr[node_count] : 0;
  int64_t begin = 0;
  for (int part = 0; part < parts; ++part) {
    int64_t end = node_count;
    if (part + 1 < parts) {
      const int64_t target = edge_count * (part + 1) / parts;
      end =
          std::lower_bound(row_ptr, row_ptr + node_count + 1, target) - row_ptr;
      end = std::min(std::max(end, begin), node_count);
    }
    ranges.push_back({begin, end});
    begin = end;
  }
  return ranges;
}

}  // namespace deepmd_cpu
