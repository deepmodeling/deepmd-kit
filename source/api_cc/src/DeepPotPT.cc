// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PYTORCH
#include <torch/script.h>

void test_function_please_remove_after_torch_is_actually_used() {
  torch::Tensor tensor = torch::rand({2, 3});
}
#endif
