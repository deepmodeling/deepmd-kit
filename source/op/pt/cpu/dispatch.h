// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Instruction-set selection for the CPU graph-lower kernels.
//
// A kernel body is compiled once per instruction set into its own namespace
// and the running CPU picks one on first use. The alternative, building the
// whole library for the host with `-march=native`, would make the artifact
// unusable on any older machine, which matters because the operator library
// ships inside a wheel and inside the LAMMPS deployment tree.
//
// Only two levels exist. AVX-512 doubles the vector width over AVX2 and is
// the level every Skylake-SP-or-newer server part provides; AVX2 with FMA is
// the floor for x86-64-v3 and covers everything else. Sub-AVX2 hosts fall
// back to the reference path rather than to a third compiled level.

#pragma once

#include <cstdint>

namespace deepmd_cpu {

/// Compiled instruction-set levels, in increasing capability order.
enum class Isa : int {
  kScalar = 0,
  kAvx2 = 1,
  kAvx512 = 2,
};

/// Return the highest level the running CPU supports.
///
/// The result is resolved once per process. `__builtin_cpu_supports` reads
/// CPUID rather than a compiler-visible architecture name, so a hypervisor
/// that masks a feature is respected.
inline Isa host_isa() {
  static const Isa resolved = [] {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl")) {
      return Isa::kAvx512;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
      return Isa::kAvx2;
    }
#endif
    return Isa::kScalar;
  }();
  return resolved;
}

/// Vector width in `float` lanes of one compiled level.
constexpr int lanes_of(Isa isa) {
  return isa == Isa::kAvx512 ? 16 : (isa == Isa::kAvx2 ? 8 : 1);
}

}  // namespace deepmd_cpu
