// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef DEEPMD_LAMMPS_MODIFY_COMPAT_H
#define DEEPMD_LAMMPS_MODIFY_COMPAT_H

#include <string>

#include "modify.h"

namespace deepmd_lammps {

// LAMMPS 13 Nov 2026 removed the public fix and compute arrays.
inline LAMMPS_NS::Compute* get_compute(LAMMPS_NS::Modify* modify,
                                       const std::string& id) {
#if LAMMPS_VERSION_NUMBER >= 20261113
  return modify->get_compute_by_id(id);
#else
  const int index = modify->find_compute(id);
  return index < 0 ? nullptr : modify->compute[index];
#endif
}

inline LAMMPS_NS::Fix* get_fix(LAMMPS_NS::Modify* modify,
                               const std::string& id) {
#if LAMMPS_VERSION_NUMBER >= 20261113
  return modify->get_fix_by_id(id);
#else
  const int index = modify->find_fix(id);
  return index < 0 ? nullptr : modify->fix[index];
#endif
}

}  // namespace deepmd_lammps

#endif
