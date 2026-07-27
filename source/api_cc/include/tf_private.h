// SPDX-License-Identifier: LGPL-3.0-or-later
/**
 * @file tf_private.h
 * @brief This file includes TensorFlow headers used for compilation.
 *
 */

#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"

namespace deepmd {
#if TF_MAJOR_VERSION >= 2 && TF_MINOR_VERSION >= 2
typedef tensorflow::tstring STRINGTYPE;
#else
typedef std::string STRINGTYPE;
#endif

/**
 * @brief Close and release an exclusively owned TensorFlow session.
 *
 * TensorFlow permits concurrent Run calls, but requires them all to finish
 * before the sole Close call.  Owners must therefore stop concurrent use
 * before destruction; this helper only centralizes the non-throwing cleanup
 * needed by destructors and partially constructed API objects.
 *
 * @param[in,out] session Exclusively owned session, reset to nullptr.
 */
inline void close_and_delete_session(tensorflow::Session*& session) noexcept {
  if (session == nullptr) {
    return;
  }
  session->Close().IgnoreError();
  delete session;
  session = nullptr;
}

/**
 * @brief Roll back a session assigned during a potentially throwing init call.
 *
 * Call release() only after initialization has completed successfully.
 */
class SessionCleanupGuard {
 public:
  explicit SessionCleanupGuard(tensorflow::Session*& session) noexcept
      : session_(session), active_(true) {}

  ~SessionCleanupGuard() {
    if (active_) {
      close_and_delete_session(session_);
    }
  }

  SessionCleanupGuard(const SessionCleanupGuard&) = delete;
  SessionCleanupGuard& operator=(const SessionCleanupGuard&) = delete;

  void release() noexcept { active_ = false; }

 private:
  tensorflow::Session*& session_;
  bool active_;
};
}  // namespace deepmd
