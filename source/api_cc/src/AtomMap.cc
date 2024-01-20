// SPDX-License-Identifier: LGPL-3.0-or-later
#include "AtomMap.h"

#include <algorithm>
#include <cassert>

using namespace deepmd;

AtomMap::AtomMap() {}

AtomMap::AtomMap(const std::vector<int>::const_iterator in_begin,
                 const std::vector<int>::const_iterator in_end) {
  int natoms = in_end - in_begin;
  atype.resize(natoms);
  std::vector<std::pair<int, int> > sorting(natoms);
  std::vector<int>::const_iterator iter = in_begin;
  for (unsigned ii = 0; ii < sorting.size(); ++ii) {
    sorting[ii] = std::pair<int, int>(*(iter++), ii);
  }
  sort(sorting.begin(), sorting.end());
  idx_map.resize(natoms);
  fwd_idx_map.resize(natoms);
  for (unsigned ii = 0; ii < idx_map.size(); ++ii) {
    idx_map[ii] = sorting[ii].second;
    fwd_idx_map[sorting[ii].second] = ii;
    atype[ii] = sorting[ii].first;
  }
}

template <typename VALUETYPE>
void AtomMap::forward(typename std::vector<VALUETYPE>::iterator out,
                      const typename std::vector<VALUETYPE>::const_iterator in,
                      const int stride,
                      const int nframes,
                      const int nall) const {
  int natoms = idx_map.size();
  for (int kk = 0; kk < nframes; ++kk) {
    for (int ii = 0; ii < natoms; ++ii) {
      int gro_i = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd) {
        // out[ii*stride+dd] = in[gro_i*stride+dd];
        *(out + static_cast<std::ptrdiff_t>(kk) * nall * stride +
          static_cast<std::ptrdiff_t>(ii) * stride + dd) =
            *(in + static_cast<std::ptrdiff_t>(kk) * nall * stride +
              static_cast<std::ptrdiff_t>(gro_i) * stride + dd);
      }
    }
  }
}

template <typename VALUETYPE>
void AtomMap::backward(typename std::vector<VALUETYPE>::iterator out,
                       const typename std::vector<VALUETYPE>::const_iterator in,
                       const int stride,
                       const int nframes,
                       const int nall) const {
  int natoms = idx_map.size();
  for (int kk = 0; kk < nframes; ++kk) {
    for (int ii = 0; ii < natoms; ++ii) {
      int gro_i = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd) {
        // out[gro_i*stride+dd] = in[ii*stride+dd];
        *(out + static_cast<std::ptrdiff_t>(kk) * nall * stride +
          static_cast<std::ptrdiff_t>(gro_i) * stride + dd) =
            *(in + static_cast<std::ptrdiff_t>(kk) * nall * stride +
              static_cast<std::ptrdiff_t>(ii) * stride + dd);
      }
    }
  }
}

template void AtomMap::forward<double>(
    typename std::vector<double>::iterator out,
    const typename std::vector<double>::const_iterator in,
    const int stride,
    const int nframes,
    const int nall) const;

template void AtomMap::forward<float>(
    typename std::vector<float>::iterator out,
    const typename std::vector<float>::const_iterator in,
    const int stride,
    const int nframes,
    const int nall) const;

template void AtomMap::forward<int>(
    typename std::vector<int>::iterator out,
    const typename std::vector<int>::const_iterator in,
    const int stride,
    const int nframes,
    const int nall) const;

template void AtomMap::backward<double>(
    typename std::vector<double>::iterator out,
    const typename std::vector<double>::const_iterator in,
    const int stride,
    const int nframes,
    const int nall) const;

template void AtomMap::backward<float>(
    typename std::vector<float>::iterator out,
    const typename std::vector<float>::const_iterator in,
    const int stride,
    const int nframes,
    const int nall) const;
