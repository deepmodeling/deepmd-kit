// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cmath>
#include <limits>

#include "RandomGenerator.h"

class Gaussian {
 public:
  void set_seed(unsigned long seed);
  void gen(double* vec, const int numb_gen);
};
