// SPDX-License-Identifier: LGPL-3.0-or-later
#include <iostream>

#include "tensorflow/core/public/version.h"
int main(int argc, char* argv[]) {
#if (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 9) || TF_MAJOR_VERSION > 2
#error "TF>=2.9 should not execute this file..."
#else
  std::cout << tf_cxx11_abi_flag();
#endif
  return 0;
}
