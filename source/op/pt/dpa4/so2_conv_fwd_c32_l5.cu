// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused DPA4 / SeZM SO(2) convolution forward instantiated for degree 5 on a
// 32-channel focus stream.

#define DPA4_CONV_L 5
#define DPA4_CONV_CF 32
#define DPA4_CONV_BACKWARD 0

#include "so2_conv_instantiate.cuh"
