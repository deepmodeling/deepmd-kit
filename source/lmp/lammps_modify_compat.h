// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef DEEPMD_LAMMPS_MODIFY_COMPAT_H
#define DEEPMD_LAMMPS_MODIFY_COMPAT_H

#include <string>

#include "modify.h"

namespace deepmd_lammps {
namespace detail {

// Prefer the by-ID accessor when present, including on development snapshots
// whose release date predates the removal of the legacy arrays.
template <typename Modify>
auto get_compute(Modify* modify, const std::string& id, int)
    -> decltype(modify->get_compute_by_id(id)) {
  return modify->get_compute_by_id(id);
}

template <typename Modify>
LAMMPS_NS::Compute* get_compute(Modify* modify, const std::string& id, long) {
  const int index = modify->find_compute(id);
  return index < 0 ? nullptr : modify->compute[index];
}

template <typename Modify>
auto get_fix(Modify* modify, const std::string& id, int)
    -> decltype(modify->get_fix_by_id(id)) {
  return modify->get_fix_by_id(id);
}

template <typename Modify>
LAMMPS_NS::Fix* get_fix(Modify* modify, const std::string& id, long) {
  const int index = modify->find_fix(id);
  return index < 0 ? nullptr : modify->fix[index];
}

}  // namespace detail

inline LAMMPS_NS::Compute* get_compute(LAMMPS_NS::Modify* modify,
                                       const std::string& id) {
  return detail::get_compute(modify, id, 0);
}

inline LAMMPS_NS::Fix* get_fix(LAMMPS_NS::Modify* modify,
                               const std::string& id) {
  return detail::get_fix(modify, id, 0);
}

}  // namespace deepmd_lammps
#endif
