// SPDX-License-Identifier: LGPL-3.0-or-later
#include "local_node_extent.h"

#include <algorithm>

namespace deepmd {
thread_local LocalNodeExtent::Bound LocalNodeExtent::current_{};
LocalNodeExtent::LocalNodeExtent(const void* counts, std::int64_t rows)
    : previous_(current_) {
  current_ = {counts, rows};
}
LocalNodeExtent::~LocalNodeExtent() { current_ = previous_; }
std::int64_t LocalNodeExtent::lookup(const void* counts,
                                     std::int64_t fallback) {
  return current_.counts && current_.counts == counts
             ? std::min(current_.rows, fallback)
             : fallback;
}
}  // namespace deepmd
