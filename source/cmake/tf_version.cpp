// SPDX-License-Identifier: LGPL-3.0-or-later
#include <iostream>

#include "tensorflow/c/c_api.h"

int main(int argc, char* argv[]) {
  // See
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h
  // TF_VERSION_STRING has been available since TensorFlow v0.6
  // Aug 2025: since TF 2.20, TF_VERSION_STRING is no more available;
  // try to use the C API TF_Version
  std::cout << TF_Version();
  return 0;
}
