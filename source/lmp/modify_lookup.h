// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LMP_DEEPMD_MODIFY_LOOKUP_H
#define LMP_DEEPMD_MODIFY_LOOKUP_H

#include <string>

#include "modify.h"

namespace LAMMPS_NS {

class Compute;
class Fix;

namespace deepmd_modify_detail {

// Prefer the by-ID accessor when present, including on development snapshots
// whose release date predates the removal of the legacy arrays.
template <typename Modify>
auto get_compute(Modify* modify, const std::string& id, int)
    -> decltype(modify->get_compute_by_id(id)) {
  return modify->get_compute_by_id(id);
}

template <typename Modify>
Compute* get_compute(Modify* modify, const std::string& id, long) {
  const int index = modify->find_compute(id);
  return index < 0 ? nullptr : modify->compute[index];
}

template <typename Modify>
auto get_fix(Modify* modify, const std::string& id, int)
    -> decltype(modify->get_fix_by_id(id)) {
  return modify->get_fix_by_id(id);
}

template <typename Modify>
Fix* get_fix(Modify* modify, const std::string& id, long) {
  const int index = modify->find_fix(id);
  return index < 0 ? nullptr : modify->fix[index];
}

}  // namespace deepmd_modify_detail

// The compute or the fix with the given ID, or nullptr if there is none.
inline Compute* deepmd_compute_by_id(Modify* modify, const std::string& id) {
  return deepmd_modify_detail::get_compute(modify, id, 0);
}

inline Fix* deepmd_fix_by_id(Modify* modify, const std::string& id) {
  return deepmd_modify_detail::get_fix(modify, id, 0);
}

}  // namespace LAMMPS_NS
#endif
