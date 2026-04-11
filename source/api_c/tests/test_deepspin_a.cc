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
      7.020322773655288e-03, 1.099636038493644e-01, 1.093176595258250e-01,
      4.865300228001564e-02, 1.096547558413134e-01, 1.099754340356070e-01,
  };
  std::vector<double> expected_f = {
      2.980086586841411e-02,  2.670602118823960e-03,  -6.205408022135627e-03,
      -7.946653268248605e-03, 4.217792180550986e-03,  1.822080579891798e-03,
      -3.416928812442276e-03, -6.992749479424899e-03, 4.728288289346775e-03,
      5.049869641953204e-03,  1.550913149717830e-02,  1.801899070929784e-02,
      -1.411871008097311e-02, -8.283139367982638e-03, -7.058623315726573e-03,
      -9.368443348703582e-03, -7.121636949145627e-03, -1.130532824067422e-02,
  };
  std::vector<double> expected_fm = {
      -1.112646578617150e+00, -2.239176906831133e-01, -2.513101985142691e-01,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      -9.763058480695873e-02, 1.564710428447471e-02,  -3.735332673990924e-02,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
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
  EXPECT_EQ(cutoff, 6.0);
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
      1.298915294144196e-02, 1.095576145701290e-01, 1.083914166945241e-01,
      4.932338375417146e-02, 1.099860785812512e-01, 1.100478936528533e-01,
  };
  std::vector<double> expected_f = {
      1.300765817095240e-02,  -1.593967210478553e-03, -3.196759265340465e-03,
      -1.300765817095220e-02, 1.593967210478477e-03,  3.196759265340509e-03,
      9.196695370628910e-03,  -9.044559760114149e-04, 5.658266727670325e-04,
      1.012744085443978e-02,  1.680427054831429e-02,  1.807036969424208e-02,
      -1.133453822298158e-02, -8.941333904804914e-03, -6.627672717506913e-03,
      -7.989598002087154e-03, -6.958480667497971e-03, -1.200852364950221e-02,
  };
  std::vector<double> expected_fm = {
      -9.651705644713781e-01, -1.704326891282164e-01, -2.605677204117113e-01,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      -9.168034653189444e-02, 1.736913887115685e-02,  -3.908906640474424e-02,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
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
