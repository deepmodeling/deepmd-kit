// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "DeepPot.h"
#include "neighbor_list.h"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferMixedType : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {
      12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56,
      12.09, 2.87, 2.74, 00.25, 3.32, 1.68, 12.83, 2.56, 2.18,
      3.51,  2.51, 2.60, 4.27,  3.22, 1.56, 3.36,  3.00, 1.81};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.,
                                13., 0., 0., 0., 13., 0., 0., 0., 13.};
  int natoms;
  int nframes = 2;

  deepmd::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/virtual_type.pbtxt";
    deepmd::convert_pbtxt_to_pb("../../tests/infer/virtual_type.pbtxt",
                                "virtual_type.pb");

    dp.init("virtual_type.pb");

    natoms = atype.size() / nframes;
  };

  void TearDown() override { remove("deeppot.pb"); };
};

TYPED_TEST_SUITE(TestInferMixedType, ValueTypes);

TYPED_TEST(TestInferMixedType, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  int& nframes = this->nframes;
  deepmd::DeepPot& dp = this->dp;
  std::vector<double> ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute_mixed_type(ener, force, virial, nframes, coord, atype, box);

  EXPECT_EQ(ener.size(), nframes);
  EXPECT_EQ(force.size(), nframes * natoms * 3);
  EXPECT_EQ(virial.size(), nframes * 9);

  EXPECT_LT(fabs(ener[0] - ener[1]), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    if (ii / 3 == 0 || ii / 3 == 3) {
      EXPECT_LT(fabs(force[ii] - force[natoms * 3 + ii + 6]), EPSILON);
    } else {
      EXPECT_LT(fabs(force[ii] - force[natoms * 3 + ii - 3]), EPSILON);
    }
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - virial[9 + ii]), EPSILON);
  }
}

TYPED_TEST(TestInferMixedType, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  int& nframes = this->nframes;
  deepmd::DeepPot& dp = this->dp;
  std::vector<double> ener;
  std::vector<VALUETYPE> force, virial, atom_ener, atom_vir;
  dp.compute_mixed_type(ener, force, virial, atom_ener, atom_vir, nframes,
                        coord, atype, box);

  EXPECT_EQ(ener.size(), nframes);
  EXPECT_EQ(force.size(), nframes * natoms * 3);
  EXPECT_EQ(virial.size(), nframes * 9);
  EXPECT_EQ(atom_ener.size(), nframes * natoms);
  EXPECT_EQ(atom_vir.size(), nframes * natoms * 9);

  for (int ii = 0; ii < natoms * 3; ++ii) {
    if (ii / 3 == 0 || ii / 3 == 3) {
      EXPECT_LT(fabs(force[ii] - force[natoms * 3 + ii + 6]), EPSILON);
    } else {
      EXPECT_LT(fabs(force[ii] - force[natoms * 3 + ii - 3]), EPSILON);
    }
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - virial[9 + ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    if (ii == 0 || ii == 3) {
      EXPECT_LT(fabs(atom_ener[ii] - atom_ener[natoms + ii + 2]), EPSILON);
    } else {
      EXPECT_LT(fabs(atom_ener[ii] - atom_ener[natoms + ii - 1]), EPSILON);
    }
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    if (ii / 9 == 0 || ii / 9 == 3) {
      EXPECT_LT(fabs(atom_vir[ii] - atom_vir[natoms * 9 + ii + 18]), EPSILON);
    } else {
      EXPECT_LT(fabs(atom_vir[ii] - atom_vir[natoms * 9 + ii - 9]), EPSILON);
    }
  }
}

template <class VALUETYPE>
class TestInferVirtualType : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {
      12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68, 3.36,  3.00,
      1.81,  3.51, 2.51, 2.60,  4.27, 3.22, 1.56,  0.00, 0.00, 0.00,  0.00,
      0.00,  0.00, 0.00, 0.00,  0.00, 0.00, 0.00,  0.00, 0.00, 0.00,  0.00,
      0.00,  0.00, 0.00, 12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32,
      1.68,  3.36, 3.00, 1.81,  3.51, 2.51, 2.60,  4.27, 3.22, 1.56};
  std::vector<int> atype = {0,  1,  1,  0, 1, 1, -1, -1, -1,
                            -1, -1, -1, 0, 1, 1, 0,  1,  1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.,
                                13., 0., 0., 0., 13., 0., 0., 0., 13.};
  int natoms;
  int nframes = 2;

  deepmd::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/virtual_type.pbtxt";
    deepmd::convert_pbtxt_to_pb("../../tests/infer/virtual_type.pbtxt",
                                "virtual_type.pb");

    dp.init("virtual_type.pb");

    natoms = atype.size() / nframes;
  };

  void TearDown() override { remove("deeppot.pb"); };
};

TYPED_TEST_SUITE(TestInferVirtualType, ValueTypes);

TYPED_TEST(TestInferVirtualType, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  int& nframes = this->nframes;
  deepmd::DeepPot& dp = this->dp;
  std::vector<double> ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute_mixed_type(ener, force, virial, nframes, coord, atype, box);

  EXPECT_EQ(ener.size(), nframes);
  EXPECT_EQ(force.size(), nframes * natoms * 3);
  EXPECT_EQ(virial.size(), nframes * 9);

  EXPECT_LT(fabs(ener[0] - ener[1]), EPSILON);
  for (int ii = 0; ii < (natoms - 3) * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - force[natoms * 3 + ii + 9]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - virial[9 + ii]), EPSILON);
  }
}

TYPED_TEST(TestInferVirtualType, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  int& nframes = this->nframes;
  deepmd::DeepPot& dp = this->dp;
  std::vector<double> ener;
  std::vector<VALUETYPE> force, virial, atom_ener, atom_vir;
  dp.compute_mixed_type(ener, force, virial, atom_ener, atom_vir, nframes,
                        coord, atype, box);

  EXPECT_EQ(ener.size(), nframes);
  EXPECT_EQ(force.size(), nframes * natoms * 3);
  EXPECT_EQ(virial.size(), nframes * 9);
  EXPECT_EQ(atom_ener.size(), nframes * natoms);
  EXPECT_EQ(atom_vir.size(), nframes * natoms * 9);

  for (int ii = 0; ii < (natoms - 3) * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - force[natoms * 3 + ii + 9]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - virial[9 + ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms - 3; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - atom_ener[natoms + ii + 3]), EPSILON);
  }
  for (int ii = 0; ii < (natoms - 3) * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - atom_vir[natoms * 9 + ii + 27]), EPSILON);
  }
}
