// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <stdexcept>
#include <string>

namespace deepmd {
/**
 * @brief General DeePMD-kit exception. Throw if anything doesn't work.
 **/
struct deepmd_exception : public std::runtime_error {
 public:
  deepmd_exception() : runtime_error("DeePMD-kit Error!") {};
  deepmd_exception(const std::string& msg)
      : runtime_error(std::string("DeePMD-kit Error: ") + msg) {};
};

struct deepmd_exception_oom : public deepmd_exception {
 public:
  deepmd_exception_oom() : deepmd_exception("DeePMD-kit OOM!") {};
  deepmd_exception_oom(const std::string& msg)
      : deepmd_exception(std::string("DeePMD-kit OOM: ") + msg) {};
};

/**
 * @brief Invalid external neighbor-list row capacity.
 *
 * Backends may translate this input-specific failure to their native invalid
 * argument status without reclassifying unrelated GPU runtime exceptions.
 */
struct deepmd_exception_nlist_capacity : public deepmd_exception {
 public:
  using deepmd_exception::deepmd_exception;
};
};  // namespace deepmd
