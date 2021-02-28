#pragma once
#include <vector>
#include <climits>
#include <stdio.h>
#include <iostream>

#define TPB 256
#define SQRT_2_PI 0.7978845608028654 
typedef unsigned long long int_64;

#if GOOGLE_CUDA
#include "gpu_cuda.h"
#endif