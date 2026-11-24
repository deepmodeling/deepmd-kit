// SPDX-License-Identifier: LGPL-3.0-or-later
#include "local_node_extent.h"

#include <algorithm>

namespace deepmd {
thread_local const LocalNodeExtent* LocalNodeExtent::current_ = nullptr;
LocalNodeExtent::LocalNodeExtent(const void* counts, std::int64_t rows)
    : counts_(counts), rows_(rows), previous_(current_) {
  current_ = this;
}
LocalNodeExtent::~LocalNodeExtent() { current_ = previous_; }
std::int64_t LocalNodeExtent::lookup(const void* counts,
                                     std::int64_t fallback) {
  return current_ && current_->counts_ == counts
             ? std::min(current_->rows_, fallback)
             : fallback;
}
}  // namespace deepmd
