// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "common.h"

TEST(SelectRealAtomsCoord, PreservesWideAtomicParameters) {
  constexpr int nframes = 2;
  constexpr int nall = 3;
  constexpr int nghost = 1;
  constexpr int ntypes = 2;
  constexpr int daparam = 2;

  std::vector<double> coord(nframes * nall * 3, 0.0);
  std::vector<int> atype = {0, ntypes, 1};
  std::vector<double> input_aparam(nframes * nall * daparam);
  std::iota(input_aparam.begin(), input_aparam.end(), 0.0);

  std::vector<double> selected_coord;
  std::vector<int> selected_atype;
  std::vector<double> selected_aparam;
  std::vector<int> forward_map;
  std::vector<int> backward_map;
  int selected_nghost = 0;
  int selected_nall = 0;
  int selected_nloc = 0;

  deepmd::select_real_atoms_coord(
      selected_coord, selected_atype, selected_aparam, selected_nghost,
      forward_map, backward_map, selected_nall, selected_nloc, coord, atype,
      input_aparam, nghost, ntypes, nframes, daparam, nall, true);

  EXPECT_EQ(selected_nall, 2);
  EXPECT_EQ(selected_nloc, 1);
  EXPECT_EQ(selected_nghost, 1);
  EXPECT_EQ(selected_aparam,
            (std::vector<double>{0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0}));
}
