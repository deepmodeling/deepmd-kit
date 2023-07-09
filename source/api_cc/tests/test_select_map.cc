// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "common.h"
template <class VALUETYPE>
class TestSelectMap : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};

  std::vector<int> fwd_map_0 = {0, -1, -1, 1, -1, -1};
  std::vector<int> fwd_map_1 = {-1, 0, 1, -1, 2, 3};

  std::vector<VALUETYPE> expected_coord_out_0 = {12.83, 2.56, 2.18,
                                                 3.36,  3.00, 1.81};
  std::vector<VALUETYPE> expected_coord_out_1 = {
      12.09, 2.87, 2.74, 00.25, 3.32, 1.68, 3.51, 2.51, 2.60, 4.27, 3.22, 1.56};

  std::vector<VALUETYPE> coord_out_0;
  std::vector<VALUETYPE> coord_out_1;

  std::vector<int> expected_atype_out_0 = {0, 0};
  std::vector<int> expected_atype_out_1 = {1, 1, 1, 1};

  std::vector<int> atype_out_0;
  std::vector<int> atype_out_1;

  int natoms;

  void SetUp() override {
    natoms = atype.size();
    EXPECT_EQ(natoms, fwd_map_0.size());
    EXPECT_EQ(2, expected_atype_out_0.size());
    EXPECT_EQ(6, expected_coord_out_0.size());
    EXPECT_EQ(natoms, fwd_map_1.size());
    EXPECT_EQ(4, expected_atype_out_1.size());
    EXPECT_EQ(12, expected_coord_out_1.size());

    for (int ii = 0; ii < 6; ++ii) {
      coord_out_0.push_back(0.0);
    }
    for (int ii = 0; ii < 12; ii++) {
      coord_out_1.push_back(0.0);
    }
    for (int ii = 0; ii < 2; ii++) {
      atype_out_0.push_back(0);
    }
    for (int ii = 0; ii < 4; ii++) {
      atype_out_1.push_back(0);
    }
  }
};

TYPED_TEST_SUITE(TestSelectMap, ValueTypes);

TYPED_TEST(TestSelectMap, selectmap_coord0) {
  deepmd::select_map(this->coord_out_0, this->coord, this->fwd_map_0, 3);
  EXPECT_EQ(6, this->coord_out_0.size());
  for (int ii = 0; ii < 6; ++ii) {
    EXPECT_EQ(this->expected_coord_out_0[ii], this->coord_out_0[ii]);
  }
}

TYPED_TEST(TestSelectMap, selectmap_coord1) {
  deepmd::select_map(this->coord_out_1, this->coord, this->fwd_map_1, 3);
  EXPECT_EQ(12, this->coord_out_1.size());
  for (int ii = 0; ii < 12; ++ii) {
    EXPECT_EQ(this->expected_coord_out_1[ii], this->coord_out_1[ii]);
  }
}

TYPED_TEST(TestSelectMap, selectmap_type0) {
  deepmd::select_map(this->atype_out_0, this->atype, this->fwd_map_0, 1);
  EXPECT_EQ(2, this->atype_out_0.size());
  for (int ii = 0; ii < 2; ++ii) {
    EXPECT_EQ(this->expected_atype_out_0[ii], this->atype_out_0[ii]);
  }
}

TYPED_TEST(TestSelectMap, selectmap_type1) {
  deepmd::select_map(this->atype_out_1, this->atype, this->fwd_map_1, 1);
  EXPECT_EQ(4, this->atype_out_1.size());
  for (int ii = 0; ii < 4; ++ii) {
    EXPECT_EQ(this->expected_atype_out_1[ii], this->atype_out_1[ii]);
  }
}
