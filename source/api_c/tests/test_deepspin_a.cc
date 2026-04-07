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
      -1.8626545229251095e+00, -2.3502165071948093e+00, -2.3500944968573521e+00,
      -2.0688274735854710e+00, -2.3485113271625320e+00, -2.3489022338537353e+00,
  };
  std::vector<double> expected_f = {
      3.7989110974834261e-02,  -6.8203560994098300e-02, 3.1554995279414300e-02,
      -6.0769407958790114e-02, 5.6658432967656878e-03,  2.1814741358389407e-02,
      1.5027739412753049e-02,  6.2090755323245192e-02,  -5.3346442187326704e-02,
      -5.2134406995188787e-02, 4.0990812807417676e-02,  -1.6987454510304811e-02,
      -6.7153786204261134e-03, -5.3801784772022326e-02, 5.6707773168242034e-02,
      6.6602343186817375e-02,  1.3257934338691726e-02,  -3.9743613108414025e-02,
  };
  std::vector<double> expected_fm = {
      4.8385521455777196e+00, 5.3158441514550137e-01, 1.0855626815019124e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      1.2140862110260138e+00, 9.6823434985033552e-01, 1.0689000529371890e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
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
      -1.9136796509970209e+00, -2.3532121417832528e+00,
      -2.3589759416772553e+00, -2.0689533840218703e+00,
      -2.3485273598793084e+00, -2.3489022338537353e+00};
  std::vector<double> expected_f = {
      5.2440246818294511e-02,  -8.2643189092284075e-03, -1.6057110078610215e-02,
      -5.2440246818295698e-02, 8.2643189092281334e-03,  1.6057110078610277e-02,
      -1.6724663644564395e-03, 7.9346065821642349e-05,  -2.5251632397208987e-04,
      -5.6934098675373246e-02, 4.0398593044712161e-02,  -1.6520316500527876e-02,
      -7.9878577602028808e-03, -5.3736758888210570e-02, 5.6516778947603999e-02,
      6.6594422800032166e-02,  1.3258819777676990e-02,  -3.9743946123104140e-02,
  };
  std::vector<double> expected_fm = {
      4.5904360179010135e+00, 6.2821415259365443e-01, 9.2483695213043082e-01,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      1.2125967529512662e+00, 9.6807902483755459e-01, 1.0691011858092361e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00};
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
