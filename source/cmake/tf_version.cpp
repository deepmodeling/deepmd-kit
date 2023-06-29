// SPDX-License-Identifier: LGPL-3.0-or-later
#include <iostream>

#include "tensorflow/core/public/version.h"

int main(int argc, char* argv[]) {
  // See
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h
  // TF_VERSION_STRING has been avaiable since TensorFlow v0.6
  std::cout << TF_VERSION_STRING;
  return 0;
}
