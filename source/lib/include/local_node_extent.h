// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>

namespace deepmd {
// A frozen AOT model passes the device count tensor through to the fused
// operator. Associate a host launch bound with that exact input for this call.
class LocalNodeExtent {
 public:
  LocalNodeExtent(const void* counts, std::int64_t rows);
  ~LocalNodeExtent();
  LocalNodeExtent(const LocalNodeExtent&) = delete;
  LocalNodeExtent& operator=(const LocalNodeExtent&) = delete;

  static std::int64_t lookup(const void* counts, std::int64_t fallback);

 private:
  const void* counts_;
  std::int64_t rows_;
  const LocalNodeExtent* previous_;
  static thread_local const LocalNodeExtent* current_;
};
}  // namespace deepmd
