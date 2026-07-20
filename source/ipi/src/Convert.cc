// SPDX-License-Identifier: LGPL-3.0-or-later
#include "Convert.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

template <typename VALUETYPE>
Convert<VALUETYPE>::Convert(const std::vector<std::string>& atomname,
                            std::map<std::string, int>& name_type_map) {
  int natoms = atomname.size();
  atype.resize(natoms);
  for (unsigned ii = 0; ii < atype.size(); ++ii) {
    const auto type_iter = name_type_map.find(atomname[ii]);
    if (type_iter == name_type_map.end()) {
      // Do not use operator[] for this lookup. A missing key would mutate the
      // map and value-initialize its integer to 0, silently evaluating an
      // unknown atom as the valid DeePMD type 0.
      throw std::invalid_argument("Unknown atom name '" + atomname[ii] +
                                  "' in coordinate file: no matching entry "
                                  "in atom_type.");
    }
    atype[ii] = type_iter->second;
  }
  std::vector<std::pair<int, int> > sorting(natoms);
  for (unsigned ii = 0; ii < sorting.size(); ++ii) {
    sorting[ii] = std::pair<int, int>(atype[ii], ii);
  }
  // sort (sorting.begin(), sorting.end());
  idx_map.resize(natoms);
  for (unsigned ii = 0; ii < idx_map.size(); ++ii) {
    idx_map[ii] = sorting[ii].second;
    atype[ii] = sorting[ii].first;
  }
}

template <typename VALUETYPE>
void Convert<VALUETYPE>::forward(std::vector<VALUETYPE>& out,
                                 const std::vector<double>& in,
                                 const int stride) const {
  assert(in.size() == stride * idx_map.size());
  int natoms = idx_map.size();
  out.resize(static_cast<size_t>(stride) * natoms);
  for (int ii = 0; ii < natoms; ++ii) {
    int gro_i = idx_map[ii];
    for (int dd = 0; dd < stride; ++dd) {
      out[ii * stride + dd] = in[gro_i * stride + dd];
    }
  }
}

template <typename VALUETYPE>
void Convert<VALUETYPE>::backward(std::vector<VALUETYPE>& out,
                                  const std::vector<double>& in,
                                  const int stride) const {
  int natoms = idx_map.size();
  assert(in.size() == stride * idx_map.size());
  out.resize(static_cast<size_t>(stride) * natoms);
  for (int ii = 0; ii < natoms; ++ii) {
    int gro_i = idx_map[ii];
    for (int dd = 0; dd < stride; ++dd) {
      out[gro_i * stride + dd] = in[ii * stride + dd];
    }
  }
}

template class Convert<float>;
template class Convert<double>;
