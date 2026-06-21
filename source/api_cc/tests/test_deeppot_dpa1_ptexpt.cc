// SPDX-License-Identifier: LGPL-3.0-or-later
// Test C++ inference for pt_expt (.pt2) backend with DPA1 (mixed-type) model.
// Uses the same model weights as test_deeppot_dpa1_pt.cc (generated from
// dpmodel with type_one_side=True via gen_dpa1.py).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "DeepPot.h"
#include "DeepPotPTExpt.h"
#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-10 : 1e-4)

namespace {
constexpr const char* kRefPath = "../../tests/infer/deeppot_dpa1.expected";
constexpr const char* kModelPath = "../../tests/infer/deeppot_dpa1.pt2";
}  // namespace

template <class VALUETYPE>
class TestInferDeepPotDpa1PtExpt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_v;
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  static deepmd::DeepPot dp;

  static void SetUpTestSuite() {
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
    dp.init(kModelPath);
#endif
  }

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("pbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("pbc", "expected_f");
    expected_v = ref.get<VALUETYPE>("pbc", "expected_v");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.assign(9, 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    for (int ii = 0; ii < natoms; ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        expected_tot_v[dd] += expected_v[ii * 9 + dd];
      }
    }
  };

  void TearDown() override {};

  static void TearDownTestSuite() { dp = deepmd::DeepPot(); }
};

template <class VALUETYPE>
deepmd::DeepPot TestInferDeepPotDpa1PtExpt<VALUETYPE>::dp;

TYPED_TEST_SUITE(TestInferDeepPotDpa1PtExpt, ValueTypes);

TYPED_TEST(TestInferDeepPotDpa1PtExpt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute(ener, force, virial, coord, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepPotDpa1PtExpt, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial, atom_ener, atom_vir;
  dp.compute(ener, force, virial, atom_ener, atom_vir, coord, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  }
}

template <class VALUETYPE>
class TestInferDeepPotDpa1PtExptNoPbc : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {};
  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_v;
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  static deepmd::DeepPot dp;

  static void SetUpTestSuite() {
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
    dp.init(kModelPath);
#endif
  }

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("nopbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("nopbc", "expected_f");
    expected_v = ref.get<VALUETYPE>("nopbc", "expected_v");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.assign(9, 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    for (int ii = 0; ii < natoms; ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        expected_tot_v[dd] += expected_v[ii * 9 + dd];
      }
    }
  };

  void TearDown() override {};

  static void TearDownTestSuite() { dp = deepmd::DeepPot(); }
};

template <class VALUETYPE>
deepmd::DeepPot TestInferDeepPotDpa1PtExptNoPbc<VALUETYPE>::dp;

TYPED_TEST_SUITE(TestInferDeepPotDpa1PtExptNoPbc, ValueTypes);

TYPED_TEST(TestInferDeepPotDpa1PtExptNoPbc, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute(ener, force, virial, coord, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepPotDpa1PtExptNoPbc, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;

  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, virial, coord, atype, box, 0, inlist, 0);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}
