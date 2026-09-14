// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal Kokkos Tools observer: count existing production profiling labels.
// The callback ABI does not require linking a second Kokkos runtime.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

namespace {
std::map<std::string, std::uint64_t> calls;
}

extern "C" void kokkosp_push_profile_region(const char* name) { ++calls[name]; }

extern "C" void kokkosp_pop_profile_region() {}

extern "C" void kokkosp_begin_parallel_for(const char* name,
                                           const std::uint32_t,
                                           std::uint64_t* id) {
  *id = ++calls[name];
}

extern "C" void kokkosp_end_parallel_for(const std::uint64_t) {}

extern "C" void kokkosp_finalize_library() {
  const char* path = std::getenv("DEEPMD_KOKKOS_PROBE_OUTPUT");
  if (path == nullptr) {
    return;
  }
  std::ofstream output(path);
  for (const auto& entry : calls) {
    output << entry.first << '\t' << entry.second << '\n';
  }
}
