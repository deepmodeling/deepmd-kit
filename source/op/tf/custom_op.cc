// SPDX-License-Identifier: LGPL-3.0-or-later
#include "custom_op.h"

#include "errors.h"

namespace deepmd {
void safe_compute(OpKernelContext* context,
                  std::function<void(OpKernelContext*)> ff) {
  try {
    ff(context);
  } catch (deepmd::deepmd_exception_oom& e) {
    OP_REQUIRES_OK(context, errors::ResourceExhausted(
                                "Operation received an exception: ", e.what(),
                                ", in file ", __FILE__, ":", __LINE__));
  } catch (deepmd::deepmd_exception& e) {
    OP_REQUIRES_OK(
        context, errors::Internal("Operation received an exception: ", e.what(),
                                  ", in file ", __FILE__, ":", __LINE__));
  }
}
};  // namespace deepmd
