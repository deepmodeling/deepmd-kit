// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "c_api.h"

class TestInferDeepSpinA : public ::testing::Test {
 protected:
  double coord[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  double spin[18] = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                     0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  float spinf[18] = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                     0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  int atype[6] = {0, 1, 1, 0, 1, 1};
  double box[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  float boxf[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<double> expected_e = {
      -4.736511666341149118e-02, 2.663324583045018068e-01,
      2.662707880956619588e-01,  -4.025182706753409334e-02,
      2.661383816099721633e-01,  2.663717506235512844e-01,
  };
  std::vector<double> expected_f = {
      1.309983290673980352e-02,  6.799786488006153221e-03,
      6.649738725468718609e-03,  -1.144456277614634265e-03,
      -3.728471118137270928e-03, -6.491178701583714961e-03,
      -5.361730092721230315e-03, -2.772299524882485336e-03,
      -4.492382686597841141e-04, 6.983026947998328959e-03,
      5.505263649468144886e-03,  1.166735707547454995e-02,
      -5.356940629013495075e-03, -1.082410608691008200e-03,
      -8.305081927004348899e-03, -8.219732855388919845e-03,
      -4.721868885763546436e-03, -3.071596903695449424e-03,
  };
  std::vector<double> expected_fm = {
      -2.702014342752852571e-01, -1.027058879186482226e-01,
      -7.359539608958531876e-02, 0.000000000000000000e+00,
      0.000000000000000000e+00,  0.000000000000000000e+00,
      0.000000000000000000e+00,  0.000000000000000000e+00,
      0.000000000000000000e+00,  4.837763296762351284e-02,
      7.279460222389372293e-02,  5.692337419156127953e-02,
      0.000000000000000000e+00,  0.000000000000000000e+00,
      0.000000000000000000e+00,  0.000000000000000000e+00,
      0.000000000000000000e+00,  0.000000000000000000e+00,
  };
  int natoms;
  double expected_tot_e;
  // std::vector<double> expected_tot_v;

  DP_DeepSpin* dp = nullptr;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp = DP_NewDeepSpin("../../tests/infer/deeppot_dpa_spin.pth");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    // EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    // expected_tot_v.resize(9);
    // std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    // for (int ii = 0; ii < natoms; ++ii) {
    //   for (int dd = 0; dd < 9; ++dd) {
    //     expected_tot_v[dd] += expected_v[ii * 9 + dd];
    //   }
    // }
  };

  void TearDown() override { DP_DeleteDeepSpin(dp); };
};

TEST_F(TestInferDeepSpinA, double_infer) {
  double* ener_ = new double;
  double* force_ = new double[natoms * 3];
  double* force_mag_ = new double[natoms * 3];
  double* virial_ = new double[9];
  double* atomic_ener_ = new double[natoms];
  double* atomic_virial_ = new double[natoms * 9];

  DP_DeepSpinCompute2(dp, 1, natoms, coord, spin, atype, box, nullptr, nullptr,
                      ener_, force_, force_mag_, virial_, atomic_ener_,
                      atomic_virial_);

  double ener = *ener_;
  std::vector<double> force(force_, force_ + natoms * 3);
  std::vector<double> force_mag(force_mag_, force_mag_ + natoms * 3);
  // std::vector<double> virial(virial_, virial_ + 9);
  std::vector<double> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  // std::vector<double> atomic_virial(atomic_virial_,
  //                                   atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), 1e-10);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  // }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-10);
  }
  // for (int ii = 0; ii < natoms * 9; ++ii) {
  //   EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-10);
  // }

  delete ener_;
  delete[] force_;
  delete[] force_mag_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepSpinA, float_infer) {
  double* ener_ = new double;
  float* force_ = new float[natoms * 3];
  float* force_mag_ = new float[natoms * 3];
  float* virial_ = new float[9];
  float* atomic_ener_ = new float[natoms];
  float* atomic_virial_ = new float[natoms * 9];

  DP_DeepSpinComputef2(dp, 1, natoms, coordf, spinf, atype, boxf, nullptr,
                       nullptr, ener_, force_, force_mag_, virial_,
                       atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<float> force(force_, force_ + natoms * 3);
  std::vector<float> force_mag(force_mag_, force_mag_ + natoms * 3);
  // std::vector<float> virial(virial_, virial_ + 9);
  std::vector<float> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  // std::vector<float> atomic_virial(atomic_virial_,
  //                                   atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-6);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-6);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), 1e-6);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-6);
  // }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-5);
  }
  // for (int ii = 0; ii < natoms * 9; ++ii) {
  //   EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-6);
  // }

  delete ener_;
  delete[] force_;
  delete[] force_mag_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepSpinA, cutoff) {
  double cutoff = DP_DeepSpinGetCutoff(dp);
  EXPECT_EQ(cutoff, 4.0);
}

TEST_F(TestInferDeepSpinA, numb_types) {
  int numb_types = DP_DeepSpinGetNumbTypes(dp);
  EXPECT_EQ(numb_types, 3);
}

TEST_F(TestInferDeepSpinA, numb_types_spin) {
  int numb_types_spin = DP_DeepSpinGetNumbTypesSpin(dp);
  EXPECT_EQ(numb_types_spin, 0);
}

TEST_F(TestInferDeepSpinA, type_map) {
  const char* type_map = DP_DeepSpinGetTypeMap(dp);
  char expected_type_map[] = "Ni O H";
  EXPECT_EQ(strcmp(type_map, expected_type_map), 0);
  DP_DeleteChar(type_map);
}

class TestInferDeepSpinANoPBC : public ::testing::Test {
 protected:
  double coord[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  double spin[18] = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                     0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  float spinf[18] = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                     0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  int atype[6] = {0, 1, 1, 0, 1, 1};
  std::vector<double> expected_e = {
      -4.263824715539775434e-02, 2.659531961465150807e-01,
      2.679077652998275716e-01,  -4.042215423551187570e-02,
      2.666009600722629158e-01,  2.666060991401024149e-01,
  };
  std::vector<double> expected_f = {
      9.664606789041574331e-04,  4.875258490407333167e-03,
      8.372291856602237514e-03,  -9.664606789041574331e-04,
      -4.875258490407331433e-03, -8.372291856602209759e-03,
      3.462063530624336170e-03,  -3.516414654756409722e-04,
      1.540231672543395411e-04,  9.145994016724501297e-03,
      6.127039584892107760e-03,  1.114133934233926815e-02,
      -5.161415144547300271e-03, -1.229667699599260294e-03,
      -8.053323256964733598e-03, -7.446642402801547170e-03,
      -4.545730419817223950e-03, -3.242039252628891276e-03,
  };
  std::vector<double> expected_fm = {
      2.426746573908480920e-02,  -2.632265214292328626e-02,
      -4.622069216845703377e-02, 0.000000000000000000e+00,
      0.000000000000000000e+00,  0.000000000000000000e+00,
      0.000000000000000000e+00,  0.000000000000000000e+00,
      0.000000000000000000e+00,  5.019524717501219757e-02,
      7.318340727755086317e-02,  5.710550212233043987e-02,
      0.000000000000000000e+00,  0.000000000000000000e+00,
      0.000000000000000000e+00,  0.000000000000000000e+00,
      0.000000000000000000e+00,  0.000000000000000000e+00,
  };
  int natoms;
  double expected_tot_e;
  // std::vector<double> expected_tot_v;

  DP_DeepSpin* dp = nullptr;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp = DP_NewDeepSpin("../../tests/infer/deeppot_dpa_spin.pth");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    // EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    // expected_tot_v.resize(9);
    // std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    // for (int ii = 0; ii < natoms; ++ii) {
    //   for (int dd = 0; dd < 9; ++dd) {
    //     expected_tot_v[dd] += expected_v[ii * 9 + dd];
    //   }
    // }
  };

  void TearDown() override { DP_DeleteDeepSpin(dp); };
};

TEST_F(TestInferDeepSpinANoPBC, double_infer) {
  double* ener_ = new double;
  double* force_ = new double[natoms * 3];
  double* force_mag_ = new double[natoms * 3];
  double* virial_ = new double[9];
  double* atomic_ener_ = new double[natoms];
  double* atomic_virial_ = new double[natoms * 9];

  DP_DeepSpinCompute2(dp, 1, natoms, coord, spin, atype, nullptr, nullptr,
                      nullptr, ener_, force_, force_mag_, virial_, atomic_ener_,
                      atomic_virial_);

  double ener = *ener_;
  std::vector<double> force(force_, force_ + natoms * 3);
  std::vector<double> force_mag(force_mag_, force_mag_ + natoms * 3);
  // std::vector<double> virial(virial_, virial_ + 9);
  std::vector<double> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  // std::vector<double> atomic_virial(atomic_virial_,
  //                                   atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), 1e-10);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  // }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-10);
  }
  // for (int ii = 0; ii < natoms * 9; ++ii) {
  //   EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-10);
  // }

  delete ener_;
  delete[] force_;
  delete[] force_mag_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepSpinANoPBC, float_infer) {
  double* ener_ = new double;
  float* force_ = new float[natoms * 3];
  float* force_mag_ = new float[natoms * 3];
  float* virial_ = new float[9];
  float* atomic_ener_ = new float[natoms];
  float* atomic_virial_ = new float[natoms * 9];

  DP_DeepSpinComputef2(dp, 1, natoms, coordf, spinf, atype, nullptr, nullptr,
                       nullptr, ener_, force_, force_mag_, virial_,
                       atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<float> force(force_, force_ + natoms * 3);
  std::vector<float> force_mag(force_mag_, force_mag_ + natoms * 3);
  // std::vector<float> virial(virial_, virial_ + 9);
  std::vector<float> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  // std::vector<float> atomic_virial(atomic_virial_,
  //                                   atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-6);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-6);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), 1e-6);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-6);
  // }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-5);
  }
  // for (int ii = 0; ii < natoms * 9; ++ii) {
  //   EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-6);
  // }

  delete ener_;
  delete[] force_;
  delete[] force_mag_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}
