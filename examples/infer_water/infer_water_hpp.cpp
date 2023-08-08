// SPDX-License-Identifier: LGPL-3.0-or-later
/* header only C++ library
 */
#include "deepmd/deepmd.hpp"

int main() {
  deepmd::hpp::DeepPot dp("graph.pb");
  std::vector<double> coord = {1., 0., 0., 0., 0., 1.5, 1., 0., 3.};
  std::vector<double> cell = {10., 0., 0., 0., 10., 0., 0., 0., 10.};
  std::vector<int> atype = {1, 0, 1};
  double e;
  std::vector<double> f, v;
  dp.compute(e, f, v, coord, atype, cell);
  // print results
  printf("energy: %f\n", e);
  for (int ii = 0; ii < 9; ++ii) {
    printf("force[%d]: %f\n", ii, f[ii]);
  }
  for (int ii = 0; ii < 9; ++ii) {
    printf("force[%d]: %f\n", ii, v[ii]);
  }
}
