// SPDX-License-Identifier: LGPL-3.0-or-later
#include "deepmd/DeepPot.h"

int main() {
  deepmd::DeepPot dp("graph.pb");
  std::vector<double> coord = {1., 0., 0., 0., 0., 1.5, 1., 0., 3.};
  std::vector<double> cell = {10., 0., 0., 0., 10., 0., 0., 0., 10.};
  std::vector<int> atype = {1, 0, 1};
  double e;
  std::vector<double> f, v;
  dp.compute(e, f, v, coord, atype, cell);
}
