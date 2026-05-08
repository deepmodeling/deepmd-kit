// SPDX-License-Identifier: LGPL-3.0-or-later
// Test C API inference for pt_expt (.pt2) backend with fparam and aparam.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "c_api.h"

class TestInferDeepPotAFParamAParamPtExptC : public ::testing::Test {
 protected:
  double coord[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  int atype[6] = {0, 0, 0, 0, 0, 0};
  double box[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  float boxf[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  double fparam[1] = {0.25852028};
  float fparamf[1] = {0.25852028};
  double aparam[6] = {0.25852028, 0.25852028, 0.25852028,
                      0.25852028, 0.25852028, 0.25852028};
  float aparamf[6] = {0.25852028, 0.25852028, 0.25852028,
                      0.25852028, 0.25852028, 0.25852028};
  // Same reference values as test_deeppot_a_fparam_aparam_ptexpt.cc (C++ API)
  // Generated from pre-committed fparam_aparam_default.pth
  std::vector<double> expected_e = {
      -1.038271223729636539e-01, -7.285433579124989123e-02,
      -9.467600492266425860e-02, -1.467050207422957442e-01,
      -7.660561676973243195e-02, -7.277296000253175023e-02};
  std::vector<double> expected_f = {
      6.622266941151369601e-02,  5.278739714221529489e-02,
      2.265728009692277028e-02,  -2.606048291367509331e-02,
      -4.538812303131847109e-02, 1.058247419681241676e-02,
      1.679392617013223121e-01,  -2.257826240741929533e-03,
      -4.490146347357203138e-02, -1.148364179422036724e-01,
      -1.169790528013799069e-02, 6.140403441496700837e-02,
      -8.078778123309421355e-02, -5.838879041789352825e-02,
      6.773641084621376263e-02,  -1.247724902386305318e-02,
      6.494524782787665373e-02,  -1.174787360813439457e-01};
  std::vector<double> expected_v = {
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
  std::vector<double> expected_tot_v;

  DP_DeepPot* dp = nullptr;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    const char* model_file = "../../tests/infer/fparam_aparam.pt2";
    dp = DP_NewDeepPot(model_file);

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

  void TearDown() override { DP_DeleteDeepPot(dp); };
};

TEST_F(TestInferDeepPotAFParamAParamPtExptC, double_infer) {
  double* ener_ = new double;
  double* force_ = new double[natoms * 3];
  double* virial_ = new double[9];
  double* atomic_ener_ = new double[natoms];
  double* atomic_virial_ = new double[natoms * 9];

  DP_DeepPotCompute2(dp, 1, natoms, coord, atype, box, fparam, aparam, ener_,
                     force_, virial_, atomic_ener_, atomic_virial_);

  double ener = *ener_;
  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_[ii] - expected_f[ii]), 1e-10);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial_[ii] - expected_tot_v[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener_[ii] - expected_e[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial_[ii] - expected_v[ii]), 1e-10);
  }

  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepPotAFParamAParamPtExptC, float_infer) {
  double* ener_ = new double;
  float* force_ = new float[natoms * 3];
  float* virial_ = new float[9];
  float* atomic_ener_ = new float[natoms];
  float* atomic_virial_ = new float[natoms * 9];

  DP_DeepPotComputef2(dp, 1, natoms, coordf, atype, boxf, fparamf, aparamf,
                      ener_, force_, virial_, atomic_ener_, atomic_virial_);

  double ener = *ener_;
  EXPECT_LT(fabs(ener - expected_tot_e), 1e-6);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_[ii] - expected_f[ii]), 1e-4);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial_[ii] - expected_tot_v[ii]), 1e-4);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener_[ii] - expected_e[ii]), 1e-5);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial_[ii] - expected_v[ii]), 1e-4);
  }

  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}
