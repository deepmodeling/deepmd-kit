// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Retain large heap blocks across inference steps.
//
// glibc services an allocation above its dynamic mmap threshold -- capped at
// 32 MiB on 64-bit -- with mmap, and returns it with munmap on free. Every
// buffer of a graph-lower step that scales with the edge count crosses that
// cap on a production system, so each step re-faults its whole working set:
// the kernel hands back zero pages, and the first touch of each one is a
// minor fault. The cost is proportional to the working set rather than to the
// arithmetic, so it appears as a throughput cliff exactly where the step
// outgrows the cap. Measured on a 125,000-atom diamond supercell with the
// compressed DPA4C neo grade, raising the thresholds takes one step from 205
// ms to 90 ms and makes throughput flat in system size.
//
// Retaining the blocks trades that back for resident memory. The knee is
// broad: on the same 125,000-atom system a 128 MiB threshold reaches 91% of
// the throughput at 3.1 GiB resident, 256 MiB reaches all of it at 4.9 GiB,
// and 1 GiB adds nothing over 256 MiB while reaching 7.2 GiB, against 2.1
// GiB and less than half the throughput for the glibc default.
// ``DP_CPU_MALLOC_RETAIN=0`` restores that default for a host that would
// rather return the memory.
//
// The policy is set from a library initializer because both consumers of the
// fused CPU path -- the Python package and the LAMMPS deployment tree -- load
// this library and neither shares an earlier entry point.

#include <cstdlib>
#include <cstring>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {

/// Blocks up to this size are served from the heap rather than by mmap.
constexpr int kRetainBytes = 256 << 20;

/// Apply the retention policy once, before the first large allocation.
struct AllocatorPolicy {
  AllocatorPolicy() {
#if defined(__GLIBC__)
    const char* opt_out = std::getenv("DP_CPU_MALLOC_RETAIN");
    if (opt_out != nullptr && std::strcmp(opt_out, "0") == 0) {
      return;
    }
    // Setting the threshold explicitly also disables glibc's dynamic
    // adjustment, which would otherwise creep back up to the 32 MiB cap.
    mallopt(M_MMAP_THRESHOLD, kRetainBytes);
    mallopt(M_TRIM_THRESHOLD, kRetainBytes);
#endif
  }
};

const AllocatorPolicy policy;

}  // namespace
