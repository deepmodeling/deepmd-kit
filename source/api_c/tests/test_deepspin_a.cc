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
  std::vector<double> expected_e = {-5.835211567762678, -5.071189078159807,
                                    -5.044361601406714, -5.582324154346981,
                                    -5.059906899269188, -5.074135576182056};
  std::vector<double> expected_f = {
      -0.0619881702551019, 0.0646720543680939,  0.2137632336140025,
      0.037800173877136,   -0.096327623008356,  -0.1531911892384847,
      -0.112204927558682,  0.0299145670766557,  -0.0589474826303666,
      0.2278904556868233,  0.0382061907026398,  0.0888060647788163,
      -0.0078898845686437, 0.0019385598635839,  -0.0791616129664364,
      -0.083607647181527,  -0.0384037490026167, -0.0112690135575317};
  std::vector<double> expected_fm = {
      -3.0778301386623275,
      -1.3135930534661662,
      -0.8332043979367366,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      -0.5452347545527696,
      -0.2051506559632127,
      -0.4908015055951312,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
  };
  int natoms;
  double expected_tot_e;
  // std::vector<double> expected_tot_v;

  DP_DeepSpin* dp;

  void SetUp() override {
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
  EXPECT_EQ(numb_types, 2);
}

TEST_F(TestInferDeepSpinA, numb_types_spin) {
  int numb_types_spin = DP_DeepSpinGetNumbTypesSpin(dp);
  EXPECT_EQ(numb_types_spin, 0);
}

TEST_F(TestInferDeepSpinA, type_map) {
  const char* type_map = DP_DeepSpinGetTypeMap(dp);
  char expected_type_map[] = "Ni O";
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
  std::vector<double> expected_e = {-5.921669893870771, -5.1676693791758685,
                                    -5.205933794558385, -5.58688965168251,
                                    -5.080322972018686, -5.08213772482076};
  std::vector<double> expected_f = {
      -0.2929142244191496, 0.0801070990501456,  0.148216178514704,
      0.2929142244191503,  -0.0801070990501454, -0.1482161785147037,
      -0.2094984819251435, 0.0241594118950041,  -0.0215199116994508,
      0.3068843038300324,  -0.001620530344866,  0.1508093841389746,
      -0.0122719879278721, 0.0186341247897136,  -0.1137104245023705,
      -0.0851138339770169, -0.0411730063398516, -0.0155790479371533};
  std::vector<double> expected_fm = {-1.5298530476860008,
                                     0.0071315024546899,
                                     0.0650492472558729,
                                     0.,
                                     0.,
                                     0.,
                                     0.,
                                     0.,
                                     0.,
                                     -0.6212052813442365,
                                     -0.2290265978320395,
                                     -0.5101405083352206,
                                     0.,
                                     0.,
                                     0.,
                                     0.,
                                     0.,
                                     0.};
  int natoms;
  double expected_tot_e;
  // std::vector<double> expected_tot_v;

  DP_DeepSpin* dp;

  void SetUp() override {
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
