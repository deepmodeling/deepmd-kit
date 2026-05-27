// SPDX-License-Identifier: LGPL-3.0-or-later
// Test C API inference for pt_expt (.pt2) backend.
// Uses the same model (converted from deeppot_sea.pth) and reference values
// as test_deeppot_ptexpt.cc (C++ API) to verify C API works with .pt2.
#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "c_api.h"

class TestInferDeepPotAPtExptC : public ::testing::Test {
 protected:
  double coord[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  int atype[6] = {0, 1, 1, 0, 1, 1};
  double box[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  float boxf[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  // Same reference values as test_deeppot_ptexpt.cc
  std::vector<double> expected_e = {-93.016873944029,  -185.923296645958,
                                    -185.927096544970, -93.019371018039,
                                    -185.926179995548, -185.924351901852};
  std::vector<double> expected_f = {
      0.006277522211,  -0.001117962774, 0.000618580445,  0.009928999655,
      0.003026035654,  -0.006941982227, 0.000667853212,  -0.002449963843,
      0.006506463508,  -0.007284129115, 0.000530662205,  -0.000028806821,
      0.000068097781,  0.006121331983,  -0.009019754602, -0.009658343745,
      -0.006110103225, 0.008865499697};
  std::vector<double> expected_v = {
      -0.000155238009, 0.000116605516,  -0.007869862476, 0.000465578340,
      0.008182547185,  -0.002398713212, -0.008112887338, -0.002423738425,
      0.007210716605,  -0.019203504012, 0.001724938709,  0.009909211091,
      0.001153857542,  -0.001600015103, -0.000560024090, 0.010727836276,
      -0.001034836404, -0.007973454377, -0.021517399106, -0.004064359664,
      0.004866398692,  -0.003360038617, -0.007241406162, 0.005920941051,
      0.004899151657,  0.006290788591,  -0.006478820311, 0.001921504710,
      0.001313470921,  -0.000304091236, 0.001684345981,  0.004124109256,
      -0.006396084465, -0.000701095618, -0.006356507032, 0.009818550859,
      -0.015230664587, -0.000110244376, 0.000690319396,  0.000045953023,
      -0.005726548770, 0.008769818495,  -0.000572380210, 0.008860603423,
      -0.013819348050, -0.021227082558, -0.004977781343, 0.006646239696,
      -0.005987066507, -0.002767831232, 0.003746502525,  0.007697590397,
      0.003746130152,  -0.005172634748};
  int natoms;
  double expected_tot_e;
  std::vector<double> expected_tot_v;

  DP_DeepPot* dp = nullptr;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    const char* model_file = "../../tests/infer/deeppot_sea.pt2";
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

TEST_F(TestInferDeepPotAPtExptC, double_infer) {
  double* ener_ = new double;
  double* force_ = new double[natoms * 3];
  double* virial_ = new double[9];
  double* atomic_ener_ = new double[natoms];
  double* atomic_virial_ = new double[natoms * 9];

  DP_DeepPotCompute(dp, natoms, coord, atype, box, ener_, force_, virial_,
                    atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<double> force(force_, force_ + natoms * 3);
  std::vector<double> virial(virial_, virial_ + 9);
  std::vector<double> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  std::vector<double> atomic_virial(atomic_virial_,
                                    atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-10);
  }

  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepPotAPtExptC, float_infer) {
  double* ener_ = new double;
  float* force_ = new float[natoms * 3];
  float* virial_ = new float[9];
  float* atomic_ener_ = new float[natoms];
  float* atomic_virial_ = new float[natoms * 9];

  DP_DeepPotComputef(dp, natoms, coordf, atype, boxf, ener_, force_, virial_,
                     atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<float> force(force_, force_ + natoms * 3);
  std::vector<float> virial(virial_, virial_ + 9);
  std::vector<float> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  std::vector<float> atomic_virial(atomic_virial_, atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-6);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-4);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-4);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-5);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-4);
  }

  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepPotAPtExptC, cutoff) {
  double cutoff = DP_DeepPotGetCutoff(dp);
  EXPECT_EQ(cutoff, 6.0);
}

TEST_F(TestInferDeepPotAPtExptC, numb_types) {
  int numb_types = DP_DeepPotGetNumbTypes(dp);
  EXPECT_EQ(numb_types, 2);
}

TEST_F(TestInferDeepPotAPtExptC, numb_types_spin) {
  int numb_types_spin = DP_DeepPotGetNumbTypesSpin(dp);
  EXPECT_EQ(numb_types_spin, 0);
}

TEST_F(TestInferDeepPotAPtExptC, type_map) {
  const char* type_map = DP_DeepPotGetTypeMap(dp);
  std::string type_map_str(type_map);
  std::istringstream iss(type_map_str);
  std::vector<std::string> types;
  std::string token;
  while (iss >> token) {
    types.push_back(token);
  }
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0], "O");
  EXPECT_EQ(types[1], "H");
  DP_DeleteChar(type_map);
}

class TestInferDeepPotAPtExptCNoPbc : public ::testing::Test {
 protected:
  double coord[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  int atype[6] = {0, 1, 1, 0, 1, 1};
  // Same NoPbc reference values as test_deeppot_ptexpt.cc
  std::vector<double> expected_e = {-93.003304908874,  -185.915806542480,
                                    -185.928116717624, -93.017934934346,
                                    -185.924393412278, -185.923906740801};
  std::vector<double> expected_f = {
      0.000868182637,  -0.000363698132, -0.000657003077, -0.000868182637,
      0.000363698132,  0.000657003077,  0.007932614680,  -0.001003609844,
      0.000737731722,  -0.003883788858, 0.000686896282,  -0.000578400682,
      0.004064895086,  0.006115547962,  -0.008747097814, -0.008113720908,
      -0.005798834400, 0.008587766774};
  std::vector<double> expected_v = {
      0.007762485364,  -0.003251851977, -0.005874313248, -0.003251851977,
      0.001362262315,  0.002460860955,  -0.005874313248, 0.002460860955,
      0.004445426242,  -0.007120030212, 0.002982715359,  0.005388130971,
      0.002982715359,  -0.001249515894, -0.002257190002, 0.005388130971,
      -0.002257190002, -0.004077504519, -0.015805863589, 0.001952684835,
      -0.001522876482, 0.001796574704,  -0.000358803950, 0.000369710813,
      -0.001108943040, 0.000332585300,  -0.000395481309, 0.008873525623,
      0.001919112114,  -0.001486235522, 0.002002929532,  0.004222469272,
      -0.006517211126, -0.001656192522, -0.006501210045, 0.010118622295,
      -0.006548889778, -0.000465126991, 0.001002876603,  0.000240398734,
      -0.005794489784, 0.008940685179,  -0.000121727685, 0.008931999051,
      -0.013852797563, -0.017962955675, -0.004645050453, 0.006214692837,
      -0.005278283465, -0.002662692758, 0.003618275905,  0.007095320684,
      0.003648086464,  -0.005023397513};
  int natoms;
  double expected_tot_e;
  std::vector<double> expected_tot_v;

  DP_DeepPot* dp = nullptr;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    const char* model_file = "../../tests/infer/deeppot_sea.pt2";
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

TEST_F(TestInferDeepPotAPtExptCNoPbc, double_infer) {
  double* ener_ = new double;
  double* force_ = new double[natoms * 3];
  double* virial_ = new double[9];
  double* atomic_ener_ = new double[natoms];
  double* atomic_virial_ = new double[natoms * 9];

  DP_DeepPotCompute(dp, natoms, coord, atype, nullptr, ener_, force_, virial_,
                    atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<double> force(force_, force_ + natoms * 3);
  std::vector<double> virial(virial_, virial_ + 9);
  std::vector<double> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  std::vector<double> atomic_virial(atomic_virial_,
                                    atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-10);
  }

  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepPotAPtExptCNoPbc, float_infer) {
  double* ener_ = new double;
  float* force_ = new float[natoms * 3];
  float* virial_ = new float[9];
  float* atomic_ener_ = new float[natoms];
  float* atomic_virial_ = new float[natoms * 9];

  DP_DeepPotComputef(dp, natoms, coordf, atype, nullptr, ener_, force_, virial_,
                     atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<float> force(force_, force_ + natoms * 3);
  std::vector<float> virial(virial_, virial_ + 9);
  std::vector<float> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  std::vector<float> atomic_virial(atomic_virial_, atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-6);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-4);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-4);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-5);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-4);
  }

  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}
