// SPDX-License-Identifier: LGPL-3.0-or-later
#include <string>

#include "tensorflow/core/framework/shape_inference.h"
int main() {
  auto ignore = tensorflow::strings::StrCat("a", "b");
  return 0;
}
