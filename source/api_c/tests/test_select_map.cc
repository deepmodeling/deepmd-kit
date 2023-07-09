// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "deepmd.hpp"
class TestSelectMap : public ::testing::Test {
 protected:
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};

  std::vector<int> fwd_map_0 = {0, -1, -1, 1, -1, -1};
  std::vector<int> fwd_map_1 = {-1, 0, 1, -1, 2, 3};

  std::vector<int> expected_atype_out_0 = {0, 0};
  std::vector<int> expected_atype_out_1 = {1, 1, 1, 1};

  std::vector<int> atype_out_0;
  std::vector<int> atype_out_1;

  int natoms;

  void SetUp() override {
    natoms = atype.size();
    EXPECT_EQ(natoms, fwd_map_0.size());
    EXPECT_EQ(2, expected_atype_out_0.size());
    EXPECT_EQ(natoms, fwd_map_1.size());
    EXPECT_EQ(4, expected_atype_out_1.size());

    for (int ii = 0; ii < 2; ii++) {
      atype_out_0.push_back(0);
    }
    for (int ii = 0; ii < 4; ii++) {
      atype_out_1.push_back(0);
    }
  }
};

TEST_F(TestSelectMap, selectmap_type0) {
  deepmd::hpp::select_map(this->atype_out_0, this->atype, this->fwd_map_0, 1);
  EXPECT_EQ(2, this->atype_out_0.size());
  for (int ii = 0; ii < 2; ++ii) {
    EXPECT_EQ(this->expected_atype_out_0[ii], this->atype_out_0[ii]);
  }
}

TEST_F(TestSelectMap, selectmap_type1) {
  deepmd::hpp::select_map(this->atype_out_1, this->atype, this->fwd_map_1, 1);
  EXPECT_EQ(4, this->atype_out_1.size());
  for (int ii = 0; ii < 4; ++ii) {
    EXPECT_EQ(this->expected_atype_out_1[ii], this->atype_out_1[ii]);
  }
}
