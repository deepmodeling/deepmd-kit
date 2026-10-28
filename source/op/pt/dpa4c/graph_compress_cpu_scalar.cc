// SPDX-License-Identifier: LGPL-3.0-or-later
//
// scalar instantiation of the compressed DPA4C CPU kernels.
//
// The build compiles this translation unit with that level's instruction-set
// flags, and the vector width enters the shared body as a compile-time
// constant: with a run-time lane count GCC emits a generic vectorized loop
// with a peel and a tail, which on the short channel blocks of this kernel
// costs more than the block itself.

#define DPA4C_CPU_ISA scalar
#define DPA4C_CPU_BLOCK 4
#include "graph_compress_cpu_kernel.h"
