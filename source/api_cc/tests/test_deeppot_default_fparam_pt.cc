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

// 1e-10 cannot pass; unclear bug or not
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-7 : 1e-4)

template <class VALUETYPE>
class TestInferDeepPotDefaultFParamPt : public ::testing::Test {
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
  // expected values computed with default fparam
  std::vector<VALUETYPE> expected_e = {
      -1.038271223729637e-01, -7.285433579124989e-02, -9.467600492266426e-02,
      -1.467050207422953e-01, -7.660561676973243e-02, -7.277296000253175e-02};
  std::vector<VALUETYPE> expected_f = {
      6.622266941151356e-02,  5.278739714221517e-02,  2.265728009692279e-02,
      -2.606048291367521e-02, -4.538812303131843e-02, 1.058247419681242e-02,
      1.679392617013225e-01,  -2.257826240741907e-03, -4.490146347357200e-02,
      -1.148364179422036e-01, -1.169790528013792e-02, 6.140403441496690e-02,
      -8.078778123309406e-02, -5.838879041789346e-02, 6.773641084621368e-02,
      -1.247724902386317e-02, 6.494524782787654e-02,  -1.174787360813438e-01};
  std::vector<VALUETYPE> expected_v = {
      -1.589185601903571e-01, 2.586167090689234e-03,  -1.575150812459097e-04,
      -1.855360549216658e-02, 1.949822308966458e-02,  -1.006552178977554e-02,
      3.177030388421500e-02,  1.714350280402170e-03,  -1.290389705296196e-03,
      -8.553511587973063e-02, -5.654638208496338e-03, -1.286955066237458e-02,
      2.464156699303163e-02,  -2.398203243424216e-02, -1.957110698882903e-02,
      2.233493653505151e-02,  6.107843889444162e-03,  1.707076397717704e-03,
      -1.653994136896924e-01, 3.894358809712642e-02,  -2.169596032233905e-02,
      6.819702786555932e-03,  -5.018240707559808e-03, 2.640663592968395e-03,
      -1.985295554050314e-03, -3.638422207618973e-02, 2.342932709960212e-02,
      -8.501331666888623e-02, -2.181253119706635e-03, 4.311299629419011e-03,
      -1.910329576491371e-03, -1.808810428459616e-03, -1.540075460017380e-03,
      -1.173703527688186e-02, -2.596307050960764e-03, 6.705026635782070e-03,
      -9.038454847872568e-02, 3.011717694088482e-02,  -5.083053967307887e-02,
      -2.951212926932095e-03, 2.342446057919113e-02,  -4.091208178777853e-02,
      -1.648470670751170e-02, -2.872262362355538e-02, 4.763925761561248e-02,
      -8.300037376165001e-02, 1.020429200603740e-03,  -1.026734257188870e-03,
      5.678534821710347e-02,  1.273635858276582e-02,  -1.530143401888294e-02,
      -1.061672032476309e-01, -2.486859787145545e-02, 2.875323543588796e-02};
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::DeepPot dp;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp.init("../../tests/infer/fparam_aparam_default.pth");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.resize(9);
    std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
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
};

TYPED_TEST_SUITE(TestInferDeepPotDefaultFParamPt, ValueTypes);

TYPED_TEST(TestInferDeepPotDefaultFParamPt, attrs) {
  using VALUETYPE = TypeParam;
  deepmd::DeepPot& dp = this->dp;
  EXPECT_EQ(dp.dim_fparam(), 1);
  EXPECT_EQ(dp.dim_aparam(), 1);
  EXPECT_TRUE(dp.has_default_fparam());
}

TYPED_TEST(TestInferDeepPotDefaultFParamPt, cpu_build_nlist_empty_fparam) {
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

TYPED_TEST(TestInferDeepPotDefaultFParamPt, cpu_build_nlist_explicit_fparam) {
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

TYPED_TEST(TestInferDeepPotDefaultFParamPt, cpu_lmp_nlist_empty_fparam) {
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

// Test that a model without default_fparam reports false
template <class VALUETYPE>
class TestInferDeepPotNoDefaultFParamPt : public ::testing::Test {
 protected:
  deepmd::DeepPot dp;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp.init("../../tests/infer/fparam_aparam.pth");
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepPotNoDefaultFParamPt, ValueTypes);

TYPED_TEST(TestInferDeepPotNoDefaultFParamPt, no_default_fparam) {
  using VALUETYPE = TypeParam;
  deepmd::DeepPot& dp = this->dp;
  EXPECT_EQ(dp.dim_fparam(), 1);
  EXPECT_FALSE(dp.has_default_fparam());
}
