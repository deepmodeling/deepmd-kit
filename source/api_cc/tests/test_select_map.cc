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

TYPED_TEST(TestSelectMap, select_real_atoms_coord_aparam_local) {
  constexpr int nframes = 2;
  constexpr int daparam = 2;
  constexpr int nall = 5;
  constexpr int nghost = 2;
  constexpr int ntypes = 2;

  // Type 2 represents a virtual atom.  The input contains virtual atoms in
  // both the local [0, 3) and ghost [3, 5) portions of the neighbor list.
  const std::vector<int> atype = {0, 2, 1, 0, 2};
  std::vector<TypeParam> coord(nframes * nall * 3);
  const std::vector<TypeParam> aparam_in = {
      10, 11, 20, 21, 30, 31,  // frame 0: three local atoms
      40, 41, 50, 51, 60, 61,  // frame 1: three local atoms
  };
  const std::vector<TypeParam> expected_aparam = {
      10, 11, 30, 31,  // frame 0: local virtual atom removed
      40, 41, 60, 61,  // frame 1: local virtual atom removed
  };

  std::vector<TypeParam> coord_out;
  std::vector<int> atype_out;
  // Seed the output with the expected logical size.  The helper must preserve
  // this size contract while remapping every daparam component below.
  std::vector<TypeParam> aparam_out(expected_aparam.size());
  int nghost_real = 0;
  std::vector<int> fwd_map;
  std::vector<int> bkw_map;
  int nall_real = 0;
  int nloc_real = 0;

  deepmd::select_real_atoms_coord(coord_out, atype_out, aparam_out, nghost_real,
                                  fwd_map, bkw_map, nall_real, nloc_real, coord,
                                  atype, aparam_in, nghost, ntypes, nframes,
                                  daparam, nall, false);

  ASSERT_EQ(aparam_out.size(), expected_aparam.size());
  EXPECT_EQ(aparam_out, expected_aparam);
  EXPECT_EQ(nloc_real, 2);
  EXPECT_EQ(nghost_real, 1);
}

TYPED_TEST(TestSelectMap, select_real_atoms_coord_aparam_all) {
  constexpr int nframes = 2;
  constexpr int daparam = 2;
  constexpr int nall = 5;
  constexpr int nghost = 2;
  constexpr int ntypes = 2;

  const std::vector<int> atype = {0, 2, 1, 0, 2};
  std::vector<TypeParam> coord(nframes * nall * 3);
  const std::vector<TypeParam> aparam_in = {
      10, 11, 20, 21, 30, 31, 40, 41, 50,  51,   // frame 0: all atoms
      60, 61, 70, 71, 80, 81, 90, 91, 100, 101,  // frame 1: all atoms
  };
  const std::vector<TypeParam> expected_aparam = {
      10, 11, 30, 31, 40, 41,  // frame 0: both virtual atoms removed
      60, 61, 80, 81, 90, 91,  // frame 1: both virtual atoms removed
  };

  std::vector<TypeParam> coord_out;
  std::vector<int> atype_out;
  // As above, assert the full scalar-count contract rather than only checking
  // the remapped prefix of the output buffer.
  std::vector<TypeParam> aparam_out(expected_aparam.size());
  int nghost_real = 0;
  std::vector<int> fwd_map;
  std::vector<int> bkw_map;
  int nall_real = 0;
  int nloc_real = 0;

  deepmd::select_real_atoms_coord(coord_out, atype_out, aparam_out, nghost_real,
                                  fwd_map, bkw_map, nall_real, nloc_real, coord,
                                  atype, aparam_in, nghost, ntypes, nframes,
                                  daparam, nall, true);

  ASSERT_EQ(aparam_out.size(), expected_aparam.size());
  EXPECT_EQ(aparam_out, expected_aparam);
  EXPECT_EQ(nloc_real, 2);
  EXPECT_EQ(nghost_real, 1);
}
