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
}  // namespace deepmd
