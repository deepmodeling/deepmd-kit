// SPDX-License-Identifier: LGPL-3.0-or-later
/**
 * Infer water using a neighbor list
 */

#ifdef USE_NATIVE_CXX_API
#include "deepmd/DeepPot.h"
using deepmd::convert_nlist;
using deepmd::DeepPot;
using deepmd::InputNlist;
#else
#include "deepmd/deepmd.hpp"
using deepmd::hpp::convert_nlist;
using deepmd::hpp::DeepPot;
using deepmd::hpp::InputNlist;
#endif

int main() {
  DeepPot dp("graph.pb");
  std::vector<double> coord = {1., 0., 0., 0., 0., 1.5, 1., 0., 3.};
  std::vector<double> cell = {10., 0., 0., 0., 10., 0., 0., 0., 10.};
  std::vector<int> atype = {1, 0, 1};
  // neighbor list
  std::vector<std::vector<int>> nlist_vec = {{1, 2}, {0, 2}, {0, 1}};
  double e;
  std::vector<double> f, v;
  std::vector<int> ilist(3), numneigh(3);
  std::vector<int*> firstneigh(3);
  InputNlist nlist(3, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(nlist, nlist_vec);
  dp.compute(e, f, v, coord, atype, cell, 0, nlist, 0);
  // print results
  printf("energy: %f\n", e);
  for (int ii = 0; ii < 9; ++ii) {
    printf("force[%d]: %f\n", ii, f[ii]);
  }
  for (int ii = 0; ii < 9; ++ii) {
    printf("force[%d]: %f\n", ii, v[ii]);
  }
}
