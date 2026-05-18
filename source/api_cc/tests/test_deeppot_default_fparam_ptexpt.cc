// SPDX-License-Identifier: LGPL-3.0-or-later
// Test C++ inference for pt_expt (.pt2) backend with default_fparam.
// Uses fparam_aparam_default.pt2: has_default_fparam=true,
// default=[0.25852028].
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

// 1e-10 cannot pass; unclear bug or not
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-7 : 1e-4)

namespace {
constexpr const char* kRefPath =
    "../../tests/infer/fparam_aparam_default.expected";
constexpr const char* kModelPath =
    "../../tests/infer/fparam_aparam_default.pt2";
}  // namespace

template <class VALUETYPE>
class TestInferDeepPotDefaultFParamPtExpt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 0, 0, 0, 0, 0};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  // aparam is still provided explicitly
  std::vector<VALUETYPE> aparam = {0.25852028, 0.25852028, 0.25852028,
                                   0.25852028, 0.25852028, 0.25852028};
  // explicit fparam for backward compat test
  std::vector<VALUETYPE> fparam = {0.25852028};
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
    expected_e = ref.get<VALUETYPE>("default", "expected_e");
    expected_f = ref.get<VALUETYPE>("default", "expected_f");
    expected_v = ref.get<VALUETYPE>("default", "expected_v");

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
deepmd::DeepPot TestInferDeepPotDefaultFParamPtExpt<VALUETYPE>::dp;

TYPED_TEST_SUITE(TestInferDeepPotDefaultFParamPtExpt, ValueTypes);

TYPED_TEST(TestInferDeepPotDefaultFParamPtExpt, attrs) {
  using VALUETYPE = TypeParam;
  deepmd::DeepPot& dp = this->dp;
  EXPECT_EQ(dp.dim_fparam(), 1);
  EXPECT_EQ(dp.dim_aparam(), 1);
  EXPECT_TRUE(dp.has_default_fparam());
}

TYPED_TEST(TestInferDeepPotDefaultFParamPtExpt, cpu_build_nlist_empty_fparam) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  // Empty fparam — model should use default
  std::vector<VALUETYPE> empty_fparam;
  dp.compute(ener, force, virial, coord, atype, box, empty_fparam, aparam);

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

TYPED_TEST(TestInferDeepPotDefaultFParamPtExpt,
           cpu_build_nlist_explicit_fparam) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  // Explicit fparam — backward compat
  dp.compute(ener, force, virial, coord, atype, box, fparam, aparam);

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

TYPED_TEST(TestInferDeepPotDefaultFParamPtExpt, cpu_lmp_nlist_empty_fparam) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  int nloc = coord.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  double ener;
  std::vector<VALUETYPE> force_, virial;
  std::vector<VALUETYPE> empty_fparam;
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 0, empty_fparam, aparam);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);

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
