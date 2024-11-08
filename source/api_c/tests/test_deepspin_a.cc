// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "c_api.h"

class TestInferDeepSpinA : public ::testing::Test {
 protected:
  double coord[12] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                      3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[12] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                      3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  double spin[12] = {0., 0., 1.2737, 0., 0., 1.2737, 0., 0., 0., 0., 0., 0.};
  float spinf[12] = {0., 0., 1.2737, 0., 0., 1.2737, 0., 0., 0., 0., 0., 0.};
  int atype[4] = {0, 0, 1, 1};
  double box[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  float boxf[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<double> expected_e = {-7.314365618560289, -7.313531316181837,
                                    -2.8980532245013997, -2.897373810282277};
  std::vector<double> expected_f = {
      0.0275132293555514,  -0.0112057401883111, -0.0212278132621243,
      -0.0229926640905535, 0.0114378553363334,  0.019670014885563,
      0.0086502856137601,  0.0088926283192558,  -0.0127014507822769,
      -0.013170850878758,  -0.009124743467278,  0.0142592491588383};
  std::vector<double> expected_fm = {
      0.0066245455049449,  -0.0023055088004378, 0.0294608578045521,
      -0.0041979452385972, 0.0025775020220167,  0.0316295420619988,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000};
  int natoms;
  double expected_tot_e;
  // std::vector<double> expected_tot_v;

  DP_DeepSpin* dp;

  void SetUp() override {
    const char* file_name = "../../tests/infer/deepspin_nlist.pbtxt";
    const char* model_file = "deepspin_nlist.pb";
    DP_ConvertPbtxtToPb(file_name, model_file);

    dp = DP_NewDeepSpin(model_file);

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

  void TearDown() override {
    remove("deepspin_nlist.pb");
    DP_DeleteDeepSpin(dp);
  };
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
  EXPECT_EQ(numb_types, 2);
}

TEST_F(TestInferDeepSpinA, numb_types_spin) {
  int numb_types_spin = DP_DeepSpinGetNumbTypesSpin(dp);
  EXPECT_EQ(numb_types_spin, 1);
}

TEST_F(TestInferDeepSpinA, type_map) {
  const char* type_map = DP_DeepSpinGetTypeMap(dp);
  char expected_type_map[] = "O H";
  EXPECT_EQ(strcmp(type_map, expected_type_map), 0);
  DP_DeleteChar(type_map);
}

class TestInferDeepSpinANoPBC : public ::testing::Test {
 protected:
  double coord[12] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                      3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[12] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                      3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  double spin[12] = {0., 0., 1.2737, 0., 0., 1.2737, 0., 0., 0., 0., 0., 0.};
  float spinf[12] = {0., 0., 1.2737, 0., 0., 1.2737, 0., 0., 0., 0., 0., 0.};
  int atype[4] = {0, 0, 1, 1};
  std::vector<double> expected_e = {-7.313160384523243, -7.312173646552338,
                                    -2.8984477845267067, -2.8984477845267067};
  std::vector<double> expected_f = {
      0.0277100137316238,  -0.0116082489956803, -0.0211484273275705,
      -0.0277100137316238, 0.0116082489956803,  0.0211484273275705,
      0.0097588349924651,  0.0091168063745397,  -0.0133541952528469,
      -0.0097588349924651, -0.0091168063745397, 0.0133541952528469};
  std::vector<double> expected_fm = {
      0.0058990325687816,  -0.0024712163463815, 0.0296682261295907,
      -0.0060028470719556, 0.0025147062058193,  0.0321884178873188,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000};
  int natoms;
  double expected_tot_e;
  // std::vector<double> expected_tot_v;

  DP_DeepSpin* dp;

  void SetUp() override {
    const char* file_name = "../../tests/infer/deepspin_nlist.pbtxt";
    const char* model_file = "deepspin_nlist.pb";
    DP_ConvertPbtxtToPb(file_name, model_file);

    dp = DP_NewDeepSpin(model_file);

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

  void TearDown() override {
    remove("deepspin_nlist.pb");
    DP_DeleteDeepSpin(dp);
  };
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
