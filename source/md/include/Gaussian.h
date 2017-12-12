#pragma once

#include <cmath>
#include <limits>

#include "RandomGenerator.h"

using namespace std;

class Gaussian 
{
public:
  void set_seed (unsigned long seed);
  void gen (double * vec, const int numb_gen);
};


