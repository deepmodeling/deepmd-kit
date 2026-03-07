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
  std::vector<double> expected_e = {
      1.596836265688982293e-01, 1.596933624175455035e-01,
      1.596859462832928844e-01, 1.596779837732069107e-01,
      1.596776702807257142e-01, 1.596869048501883825e-01};
  std::vector<double> expected_f = {
      1.112134318320094352e-05,  1.085230789880272100e-04,
      9.298442641358874074e-07,  1.491517597320257530e-04,
      -1.250419527572717225e-05, -9.768265174690741028e-05,
      -5.021052645725073397e-05, -9.741678916887848341e-05,
      9.375317764392495572e-05,  8.664103999852425394e-05,
      -4.538513400016661465e-05, 8.561605116672722879e-05,
      -2.454811055475981441e-05, 1.079491454988312104e-04,
      -1.656974003674590440e-04, -1.721555059017403750e-04,
      -6.116610604208616705e-05, 8.308097903957835525e-05};
  std::vector<double> expected_v = {
      -1.264062189119726106e-04, -1.636544077308309524e-05,
      4.453224130911556590e-05,  -7.947403699519458518e-06,
      -4.603504987332719071e-05, 9.491045850088816000e-06,
      4.131028921467394082e-05,  9.691472468201876704e-06,
      -3.323572704427471520e-05, -1.024556912293147473e-04,
      5.530809120954559223e-06,  5.211030391191971272e-05,
      -3.851138686809524851e-06, 2.101414374153427902e-07,
      3.247573516972862344e-06,  4.561253716254950400e-05,
      -3.865680092083280590e-06, -3.262252150841829144e-05,
      -1.166788692566841262e-04, -1.814499890570940753e-05,
      2.155064011880968559e-05,  -1.629918981392229854e-05,
      -3.245631268444005592e-05, 2.968538417601228937e-05,
      2.463149007223104176e-05,  3.660689861518736949e-05,
      -3.586518711234946879e-05, -1.424206401855391917e-04,
      -1.017840928263488617e-05, 1.421307534994552908e-05,
      -8.618294024757048886e-06, -2.192409332705383732e-05,
      3.461715847634955364e-05,  1.277625693457703254e-05,
      3.486479415793123331e-05,  -5.604161168847292539e-05,
      -8.612844407008925294e-05, 2.508361660152536905e-06,
      -1.633895954532816838e-07, 1.903591783622687189e-06,
      -3.028341203071209831e-05, 4.685511271783763848e-05,
      2.876824509984768127e-06,  4.576515617130315025e-05,
      -7.108738780331672730e-05, -1.062354815105974551e-04,
      -2.954644717832223544e-05, 4.075640001084848793e-05,
      -3.138369091725678469e-05, -1.316088004849699161e-05,
      1.786389692843508954e-05,  4.579187321116929540e-05,
      1.869753034515593833e-05,  -2.550749273395911822e-05};
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
