// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <stdio.h>

#include <climits>
#include <iostream>
#include <vector>

#define TPB 256
#define SQRT_2_PI 0.7978845608028654
typedef long long int_64;
typedef unsigned long long uint_64;

#if GOOGLE_CUDA
#include "gpu_cuda.h"
#endif

#if TENSORFLOW_USE_ROCM
#include "gpu_rocm.h"
#endif
