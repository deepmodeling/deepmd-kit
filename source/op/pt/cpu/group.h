// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace deepmd {

/**
 * @brief Group an edge axis by a node-valued key with a threaded counting sort.
 *
 * The keys are bounded node indices, so sorting them comparison-wise costs a
 * factor of log(E) that a histogram and a prefix sum do not. On a
 * production-sized neighbor list that factor is the dominant cost of building
 * the source-major view of a graph.
 *
 * Each chunk owns a contiguous histogram of its own, indexed by node. That is
 * the layout the two hot passes want: they scatter into random nodes of one
 * chunk, so a chunk-contiguous histogram keeps the working set at one node
 * column -- tens of kilobytes, resident in L2 -- while the node-major
 * alternative spreads each chunk's counters across the whole table and turns
 * every increment into a last-level access. The two prefix passes read across
 * chunks instead, which is strided, but they touch the table once each against
 * the edge axis twice.
 *
 * The chunking is derived from the thread count rather than taken from the
 * scheduler, which keeps the permutation independent of how the work happens
 * to be distributed. The chunk count is capped so the histogram stays a
 * bounded fraction of the payload it describes.
 *
 * @param key Node index of each edge, length ``edge_count``, values in
 *   ``[0, node_count)``.
 * @param edge_count Number of edges to group.
 * @param node_count Number of nodes.
 * @param row_ptr Receives the CSR offsets, length ``node_count + 1``.
 * @param order Receives the grouped permutation, length ``edge_count``.
 */
inline void group_by_node(const std::int64_t* key,
                          const std::int64_t edge_count,
                          const std::int64_t node_count,
                          std::int64_t* row_ptr,
                          std::int64_t* order) {
  //: A chunk below this many edges does not pay for its histogram column.
  constexpr std::int64_t kMinChunkEdges = 1 << 15;
  //: Upper bound on histogram entries, keeping it near the payload's size.
  constexpr std::int64_t kMaxHistogram = 1 << 23;
  const std::int64_t by_threads = std::max<std::int64_t>(
      std::min<std::int64_t>(at::get_num_threads(),
                             edge_count / kMinChunkEdges),
      1);
  const std::int64_t chunks = std::max<std::int64_t>(
      std::min(by_threads, kMaxHistogram / (node_count + 1)), 1);
  const std::int64_t span = (edge_count + chunks - 1) / chunks;
  const std::int64_t stride = node_count + 1;
  std::vector<std::int64_t> histogram(
      static_cast<size_t>(stride) * static_cast<size_t>(chunks), 0);

  // Counting and filling walk the same chunk boundaries; the cursor a chunk
  // reads in the second pass is the running offset the prefix left behind.
  const auto walk = [&](std::int64_t begin, std::int64_t end, const bool fill) {
    for (std::int64_t chunk = begin; chunk < end; ++chunk) {
      std::int64_t* column = &histogram[static_cast<size_t>(chunk) * stride];
      const std::int64_t first = chunk * span;
      const std::int64_t last = std::min(first + span, edge_count);
      for (std::int64_t edge = first; edge < last; ++edge) {
        std::int64_t& slot = column[key[edge]];
        if (fill) {
          order[slot++] = edge;
        } else {
          ++slot;
        }
      }
    }
  };
  at::parallel_for(0, chunks, 1, [&](std::int64_t begin, std::int64_t end) {
    walk(begin, end, /*fill=*/false);
  });
  at::parallel_for(0, stride, 64, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t node = begin; node < end; ++node) {
      std::int64_t total = 0;
      for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
        total += histogram[static_cast<size_t>(chunk) * stride + node];
      }
      row_ptr[node] = total;
    }
  });
  std::int64_t running = 0;
  for (std::int64_t node = 0; node <= node_count; ++node) {
    const std::int64_t total = row_ptr[node];
    row_ptr[node] = running;
    running += total;
  }
  at::parallel_for(0, stride, 64, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t node = begin; node < end; ++node) {
      std::int64_t cursor = row_ptr[node];
      for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
        std::int64_t& slot =
            histogram[static_cast<size_t>(chunk) * stride + node];
        const std::int64_t total = slot;
        slot = cursor;
        cursor += total;
      }
    }
  });
  at::parallel_for(0, chunks, 1, [&](std::int64_t begin, std::int64_t end) {
    walk(begin, end, /*fill=*/true);
  });
}

}  // namespace deepmd
