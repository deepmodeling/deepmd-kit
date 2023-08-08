// SPDX-License-Identifier: LGPL-3.0-or-later
/**
 * @file tf_public.h
 * @brief This file declares incompleted TensorFlow class used for public
 * headers.
 *
 */

// skip if TF headers have been included
#ifndef TF_MAJOR_VERSION
namespace tensorflow {
class Session;
class Tensor;
class GraphDef;
class Status;
}  // namespace tensorflow
#endif
