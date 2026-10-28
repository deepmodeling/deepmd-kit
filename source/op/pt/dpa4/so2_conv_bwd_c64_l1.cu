// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused DPA4 / SeZM SO(2) convolution backward instantiated for degree 1 on a
// 64-channel focus stream.

#define DPA4_CONV_L 1
#define DPA4_CONV_CF 64
#define DPA4_CONV_BACKWARD 1

#include "so2_conv_instantiate.cuh"
