// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LMP_DEEPMD_MODIFY_LOOKUP_H
#define LMP_DEEPMD_MODIFY_LOOKUP_H

#include <string>

#include "modify.h"

namespace LAMMPS_NS {

class Compute;
class Fix;

// The compute or the fix with the given ID, or nullptr if there is none.
// LAMMPS 7Jan2022 introduced the pointer lookups of Modify; later versions
// remove the index lookups and the public lists of fixes and computes.
inline Compute* deepmd_compute_by_id(Modify* modify, const std::string& id) {
#if LAMMPS_VERSION_NUMBER >= 20220107
  return modify->get_compute_by_id(id);
#else
  const int index = modify->find_compute(id);
  return index < 0 ? nullptr : modify->compute[index];
#endif
}

inline Fix* deepmd_fix_by_id(Modify* modify, const std::string& id) {
#if LAMMPS_VERSION_NUMBER >= 20220107
  return modify->get_fix_by_id(id);
#else
  const int index = modify->find_fix(id);
  return index < 0 ? nullptr : modify->fix[index];
#endif
}

}  // namespace LAMMPS_NS

#endif
