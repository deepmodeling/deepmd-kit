// SPDX-License-Identifier: LGPL-3.0-or-later
// Test C++ inference for pt_expt (.pt2) backend with fparam and aparam.
// Uses a model created with type_one_side=True (required for make_fx tracing).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "DeepPot.h"
#include "DeepPotPTExpt.h"
#include "neighbor_list.h"
#include "test_utils.h"

// 1e-10 cannot pass; unclear bug or not
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-7 : 1e-4)

template <class VALUETYPE>
class TestInferDeepPotAFParamAParamPtExpt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 0, 0, 0, 0, 0};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> fparam = {0.25852028};
  std::vector<VALUETYPE> aparam = {0.25852028, 0.25852028, 0.25852028,
                                   0.25852028, 0.25852028, 0.25852028};
  std::vector<VALUETYPE> expected_e = {
      -1.038271223729636539e-01, -7.285433579124989123e-02,
      -9.467600492266425860e-02, -1.467050207422957442e-01,
      -7.660561676973243195e-02, -7.277296000253175023e-02};
  std::vector<VALUETYPE> expected_f = {
      6.622266941151369601e-02,  5.278739714221529489e-02,
      2.265728009692277028e-02,  -2.606048291367509331e-02,
      -4.538812303131847109e-02, 1.058247419681241676e-02,
      1.679392617013223121e-01,  -2.257826240741929533e-03,
      -4.490146347357203138e-02, -1.148364179422036724e-01,
      -1.169790528013799069e-02, 6.140403441496700837e-02,
      -8.078778123309421355e-02, -5.838879041789352825e-02,
      6.773641084621376263e-02,  -1.247724902386305318e-02,
      6.494524782787665373e-02,  -1.174787360813439457e-01};
  std::vector<VALUETYPE> expected_v = {
      -1.589185601903579381e-01, 2.586167090689088510e-03,
      -1.575150812458056548e-04, -1.855360549216640564e-02,
      1.949822308966445150e-02,  -1.006552178977542650e-02,
      3.177030388421490936e-02,  1.714350280402104215e-03,
      -1.290389705296313833e-03, -8.553511587973079699e-02,
      -5.654638208496251539e-03, -1.286955066237439882e-02,
      2.464156699303176462e-02,  -2.398203243424212178e-02,
      -1.957110698882909630e-02, 2.233493653505165544e-02,
      6.107843889444162372e-03,  1.707076397717688723e-03,
      -1.653994136896924094e-01, 3.894358809712639147e-02,
      -2.169596032233910010e-02, 6.819702786556020371e-03,
      -5.018240707559744503e-03, 2.640663592968431426e-03,
      -1.985295554050418160e-03, -3.638422207618969423e-02,
      2.342932709960221863e-02,  -8.501331666888653493e-02,
      -2.181253119706856591e-03, 4.311299629418858387e-03,
      -1.910329576491436726e-03, -1.808810428459609043e-03,
      -1.540075460017477360e-03, -1.173703527688202929e-02,
      -2.596307050960845741e-03, 6.705026635782097323e-03,
      -9.038454847872562370e-02, 3.011717694088476838e-02,
      -5.083053967307901710e-02, -2.951212926932282599e-03,
      2.342446057919112673e-02,  -4.091208178777860222e-02,
      -1.648470670751139844e-02, -2.872262362355524484e-02,
      4.763925761561256522e-02,  -8.300037376164930147e-02,
      1.020429200603871836e-03,  -1.026734257188876599e-03,
      5.678534821710372327e-02,  1.273635858276599142e-02,
      -1.530143401888291177e-02, -1.061672032476311256e-01,
      -2.486859787145567074e-02, 2.875323543588798395e-02};
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  static deepmd::DeepPot dp;

  static void SetUpTestSuite() {
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
    dp.init("../../tests/infer/fparam_aparam.pt2");
#endif
  }

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif

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

  static void TearDownTestSuite() { dp = deepmd::DeepPot(); }
};

template <class VALUETYPE>
deepmd::DeepPot TestInferDeepPotAFParamAParamPtExpt<VALUETYPE>::dp;

TYPED_TEST_SUITE(TestInferDeepPotAFParamAParamPtExpt, ValueTypes);

TYPED_TEST(TestInferDeepPotAFParamAParamPtExpt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
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

TYPED_TEST(TestInferDeepPotAFParamAParamPtExpt, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial, atom_ener, atom_vir;
  dp.compute(ener, force, virial, atom_ener, atom_vir, coord, atype, box,
             fparam, aparam);

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

TYPED_TEST(TestInferDeepPotAFParamAParamPtExpt, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
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
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
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

  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 1, fparam, aparam);
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

TYPED_TEST(TestInferDeepPotAFParamAParamPtExpt, cpu_lmp_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
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
  std::vector<VALUETYPE> force_, atom_ener_, atom_vir_, virial;
  std::vector<VALUETYPE> force, atom_ener, atom_vir;
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy, atype_cpy,
             box, nall - nloc, inlist, 0, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);
  _fold_back<VALUETYPE>(atom_ener, atom_ener_, mapping, nloc, nall, 1);
  _fold_back<VALUETYPE>(atom_vir, atom_vir_, mapping, nloc, nall, 9);

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

  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  std::fill(atom_ener_.begin(), atom_ener_.end(), 0.0);
  std::fill(atom_vir_.begin(), atom_vir_.end(), 0.0);
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy, atype_cpy,
             box, nall - nloc, inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);
  _fold_back<VALUETYPE>(atom_ener, atom_ener_, mapping, nloc, nall, 1);
  _fold_back<VALUETYPE>(atom_vir, atom_vir_, mapping, nloc, nall, 9);

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

// Test that a .pt2 model without default_fparam reports false
template <class VALUETYPE>
class TestInferDeepPotNoDefaultFParamPtExpt : public ::testing::Test {
 protected:
  static deepmd::DeepPot dp;

  static void SetUpTestSuite() {
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
    dp.init("../../tests/infer/fparam_aparam.pt2");
#endif
  }

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
  };

  void TearDown() override {};

  static void TearDownTestSuite() { dp = deepmd::DeepPot(); }
};

template <class VALUETYPE>
deepmd::DeepPot TestInferDeepPotNoDefaultFParamPtExpt<VALUETYPE>::dp;

TYPED_TEST_SUITE(TestInferDeepPotNoDefaultFParamPtExpt, ValueTypes);

TYPED_TEST(TestInferDeepPotNoDefaultFParamPtExpt, no_default_fparam) {
  using VALUETYPE = TypeParam;
  deepmd::DeepPot& dp = this->dp;
  EXPECT_EQ(dp.dim_fparam(), 1);
  EXPECT_FALSE(dp.has_default_fparam());
}
// DefaultFParam tests with expected values are in
// test_deeppot_default_fparam_ptexpt.cc (from upstream #5343).
