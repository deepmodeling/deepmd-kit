// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused DPA4 / SeZM SO(2) convolution backward instantiated for degree 2 on a
// 32-channel focus stream.

#define DPA4_CONV_L 2
#define DPA4_CONV_CF 32
#define DPA4_CONV_BACKWARD 1

#include "so2_conv_instantiate.cuh"
