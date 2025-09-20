// SPDX-License-Identifier: LGPL-3.0-or-later
#include "pairwise.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "errors.h"

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
  // https://stackoverflow.com/a/12399290/9567349
  // by Lukasz Wiklendt under CC BY-SA 4.0
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

void deepmd::group_atoms_cpu(std::vector<std::vector<int>> &fragments,
                             const std::vector<int> &idxs) {
  int natoms = idxs.size();
  // sort idxs
  std::vector<size_t> idxs_idx = sort_indexes(idxs);
  // now idxs_idx is sorted index, so we can easily group atoms in only one loop
  int last_frag = -1;
  for (size_t ii = 0; ii < idxs.size(); ii++) {
    int frag = idxs[idxs_idx[ii]];
    if (frag == -1) {
      // -1 is the place holder
      continue;
    }
    if (frag != last_frag) {
      last_frag = frag;
      fragments.emplace_back();
    }
    // push to the last fragment
    fragments.back().push_back(idxs_idx[ii]);
  }
}

void deepmd::dprc_pairwise_map_cpu(
    std::vector<int> &forward_qm_map,
    std::vector<int> &backward_qm_map,
    std::vector<int> &forward_qmmm_map,
    std::vector<int> &backward_qmmm_map,
    int &nloc_qm,
    int &nloc_qmmm,
    int &nall_qm,
    int &nall_qmmm,
    const std::vector<std::vector<int>> &fragments,
    const int nloc,
    const int nall) {
  int nfragments = fragments.size();
  if (nfragments == 0) {
    throw deepmd::deepmd_exception("fragments is empty");
  }
  int nqm = fragments[0].size();
  // assume fragments = ((3,4,10), (0,1,2,11), (5,6,7), (8,9))
  // 10, 11 is ghost atoms
  // (3, 4, 10)
  forward_qm_map = fragments[0];
  // (-1, -1, -1, 0, 1, -1, -1, -1, -1, -1, 2, -1)
  backward_qm_map.resize(nall);
  std::fill(backward_qm_map.begin(), backward_qm_map.end(), -1);
  for (int ii = 0; ii < forward_qm_map.size(); ++ii) {
    backward_qm_map[forward_qm_map[ii]] = ii;
  }

  // get max size of fragments
  int max_fragment_real_size = 0;
  int max_fragment_ghost_size = 0;
  for (int ii = 1; ii < nfragments; ++ii) {
    int fragment_real_size = 0;
    int fragment_ghost_size = 0;
    for (int jj = 0; jj < fragments[ii].size(); ++jj) {
      if (fragments[ii][jj] >= nloc) {
        fragment_ghost_size++;
      } else {
        fragment_real_size++;
      }
    }
    if (fragment_real_size > max_fragment_real_size) {
      max_fragment_real_size = fragment_real_size;
    }
    if (fragment_ghost_size > max_fragment_ghost_size) {
      max_fragment_ghost_size = fragment_ghost_size;
    }
  }
  int max_fragment_size = max_fragment_real_size + max_fragment_ghost_size;
  int map_size = nqm + max_fragment_real_size + max_fragment_ghost_size;
  // (3, 4, 0, 1, 2, 10, 11),
  // (3, 4, 5, 6, 7, 10, -1),
  // (3, 4, 8, 9, -1, 10, -1)
  forward_qmmm_map.resize(static_cast<size_t>(nfragments - 1) * map_size);
  std::fill(forward_qmmm_map.begin(), forward_qmmm_map.end(), -1);
  int nqm_real = nloc;  // init for nfragments = 1
  for (int ii = 0; ii < nfragments - 1; ++ii) {
    // real
    for (int jj = 0, kk = 0; jj < nqm; ++jj) {
      if (fragments[0][jj] < nloc) {
        forward_qmmm_map[ii * map_size + kk] = fragments[0][jj];
        kk++;
      }
      if (jj == nqm - 1) {
        nqm_real = kk;
      }
    }
    for (int jj = 0, kk = 0; jj < fragments[ii + 1].size(); ++jj) {
      if (fragments[ii + 1][jj] < nloc) {
        forward_qmmm_map[ii * map_size + nqm_real + kk] = fragments[ii + 1][jj];
        kk++;
      }
    }
    // ghost
    for (int jj = 0, kk = 0; jj < nqm; ++jj) {
      if (fragments[0][jj] >= nloc) {
        forward_qmmm_map[ii * map_size + nqm_real + max_fragment_real_size +
                         kk] = fragments[0][jj];
        kk++;
      }
    }
    for (int jj = 0, kk = 0; jj < fragments[ii + 1].size(); ++jj) {
      if (fragments[ii + 1][jj] >= nloc) {
        forward_qmmm_map[ii * map_size + nqm + max_fragment_real_size + kk] =
            fragments[ii + 1][jj];
        kk++;
      }
    }
  }

  // (2, 3, 4, 0, 1, -1, -1, -1, -1, -1, 5, 6)
  // (-1, -1, -1, 0, 1, 2, 3, 4, -1, -1, 5, -1)
  // (-1, -1, -1, 0, 1, -1, -1, -1, 2, 3, 5, -1)
  backward_qmmm_map.resize(static_cast<size_t>(nfragments - 1) * nall);
  std::fill(backward_qmmm_map.begin(), backward_qmmm_map.end(), -1);
  for (int ii = 0; ii < nfragments - 1; ++ii) {
    for (int jj = 0; jj < map_size; ++jj) {
      if (forward_qmmm_map[ii * map_size + jj] != -1) {
        backward_qmmm_map[ii * nall + forward_qmmm_map[ii * map_size + jj]] =
            jj;
      }
    }
  }
  // natoms
  nloc_qm = nqm_real;
  nloc_qmmm = nqm_real + max_fragment_real_size;
  nall_qm = nqm;
  nall_qmmm = nqm + max_fragment_size;
}
