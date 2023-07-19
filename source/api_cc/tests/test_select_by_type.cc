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
class TestSelectByType : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  int nghost = 10;

  std::vector<int> sel_type_0 = {0};
  std::vector<int> expected_fwd_map_0 = {0, -1, -1, 1, -1, -1};
  std::vector<int> expected_bkw_map_0 = {0, 3};
  int expected_nghost_real_0 = 2;
  std::vector<int> fwd_map_0;
  std::vector<int> bkw_map_0;
  int nghost_real_0;

  std::vector<int> sel_type_1 = {1};
  std::vector<int> expected_fwd_map_1 = {-1, 0, 1, -1, 2, 3};
  std::vector<int> expected_bkw_map_1 = {1, 2, 4, 5};
  int expected_nghost_real_1 = 4;
  std::vector<int> fwd_map_1;
  std::vector<int> bkw_map_1;
  int nghost_real_1;

  int natoms;

  void SetUp() override {
    natoms = atype.size();
    EXPECT_EQ(natoms, expected_fwd_map_0.size());
    EXPECT_EQ(2, expected_bkw_map_0.size());
    EXPECT_EQ(natoms, expected_fwd_map_1.size());
    EXPECT_EQ(4, expected_bkw_map_1.size());
  }
};

TYPED_TEST_SUITE(TestSelectByType, ValueTypes);

TYPED_TEST(TestSelectByType, selectbytype0) {
  deepmd::select_by_type(this->fwd_map_0, this->bkw_map_0, this->nghost_real_0,
                         this->coord, this->atype, this->nghost,
                         this->sel_type_0);
  EXPECT_EQ(this->natoms, this->fwd_map_0.size());
  EXPECT_EQ(2, this->bkw_map_0.size());
  EXPECT_EQ(this->expected_nghost_real_0, this->nghost_real_0);
  for (int ii = 0; ii < this->natoms; ++ii) {
    EXPECT_EQ(this->expected_fwd_map_0[ii], this->fwd_map_0[ii]);
  }
  for (int ii = 0; ii < 2; ++ii) {
    EXPECT_EQ(this->expected_bkw_map_0[ii], this->bkw_map_0[ii]);
  }
}

TYPED_TEST(TestSelectByType, selectbytype1) {
  deepmd::select_by_type(this->fwd_map_1, this->bkw_map_1, this->nghost_real_1,
                         this->coord, this->atype, this->nghost,
                         this->sel_type_1);
  EXPECT_EQ(this->natoms, this->fwd_map_1.size());
  EXPECT_EQ(4, this->bkw_map_1.size());
  EXPECT_EQ(this->expected_nghost_real_1, this->nghost_real_1);
  for (int ii = 0; ii < this->natoms; ++ii) {
    EXPECT_EQ(this->expected_fwd_map_1[ii], this->fwd_map_1[ii]);
  }
  for (int ii = 0; ii < 4; ++ii) {
    EXPECT_EQ(this->expected_bkw_map_1[ii], this->bkw_map_1[ii]);
  }
}
