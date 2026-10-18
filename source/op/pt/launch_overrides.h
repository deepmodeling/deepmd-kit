// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Overrides of the launch choices that the CUDA operators derive from the
// device they run on and from the system they scan.
//
// Every choice below follows from device properties, or from the node types
// of the system, by a rule measured on the devices the code was tuned for. On
// another device the alternatives can be timed without a rebuild by setting
// the variable of the choice, which is read once per process. An unset or
// empty variable, or "auto", keeps the derived choice; any other value outside
// the listed ones fails every call that reads it.
//
//   DP_DPA4C_ROW_SPLIT      1, 2, 4 or 8: warps that share one destination
//                           row of a canonical edge scan with 32-bit indices
//                           that runs blocks of independent rows. Changes the
//                           summation order of the row.
//   DP_DPA4C_COSCHEDULE     0: no co-scheduled lockstep blocks; the wide
//                           profiles scan blocks of independent rows as
//                           well. Changes the summation order of the row.
//   DP_DPA4C_BOND_PREFETCH  0 or 1: the L2 prefetch of the bond vectors of a
//                           row, and of its source indices in a slotted scan,
//                           off or on. Leaves every output unchanged.
//   DP_DPA4C_NODE_LANES     8 or 16: lanes of a node group of the node
//                           backward. Changes the summation order of the
//                           per-node sums of the node backward: the
//                           normalizer dot products and, from degree three,
//                           the coupling cotangents.
//   DP_DPA4C_TYPE_SLOTS     0: no slotted edge scans; a canonical system of
//                           at most eight node types, two or more for a
//                           spin-free model, scans the mode mixing and the
//                           FiLM coefficients of every edge pairwise and
//                           reads the radial table. Changes the summation
//                           order of the moments and the rounding of the
//                           channels that a low-rank table expands.
//   DP_FITTING_FUSED        0: no fused fitting network; the per-layer
//                           kernels serve every run. Leaves every output
//                           unchanged.

#pragma once

#include <c10/util/Exception.h>

#include <cstdlib>
#include <initializer_list>
#include <string>

/// Value of the override `name`, or -1 to keep the derived choice. The value
/// must be one of `allowed`.
inline long launch_override(const char* name,
                            std::initializer_list<long> allowed) {
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0' || std::string(text) == "auto") {
    return -1;
  }
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  bool listed = false;
  std::string accepted;
  for (const long option : allowed) {
    listed = listed || value == option;
    accepted += std::to_string(option) + ", ";
  }
  TORCH_CHECK(end != text && *end == '\0' && listed, name, "=", text,
              " is not one of the values the override accepts (", accepted,
              "or auto); unset it or set it to auto to keep the derived "
              "choice");
  return value;
}
