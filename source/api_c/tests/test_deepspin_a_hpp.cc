// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "deepmd.hpp"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepSpinAHPP : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                                 0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> expected_e = {
      7.020322773655288e-03, 1.099636038493644e-01, 1.093176595258250e-01,
      4.865300228001564e-02, 1.096547558413134e-01, 1.099754340356070e-01,
  };
  std::vector<VALUETYPE> expected_f = {
      2.980086586841411e-02,  2.670602118823960e-03,  -6.205408022135627e-03,
      -7.946653268248605e-03, 4.217792180550986e-03,  1.822080579891798e-03,
      -3.416928812442276e-03, -6.992749479424899e-03, 4.728288289346775e-03,
      5.049869641953204e-03,  1.550913149717830e-02,  1.801899070929784e-02,
      -1.411871008097311e-02, -8.283139367982638e-03, -7.058623315726573e-03,
      -9.368443348703582e-03, -7.121636949145627e-03, -1.130532824067422e-02,
  };
  std::vector<VALUETYPE> expected_fm = {
      -1.112646578617150e+00, -2.239176906831133e-01, -2.513101985142691e-01,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      -9.763058480695873e-02, 1.564710428447471e-02,  -3.735332673990924e-02,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
  };
  unsigned int natoms;
  double expected_tot_e;
  // std::vector<VALUETYPE> expected_tot_v;

  deepmd::hpp::DeepSpin dp;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp.init("../../tests/infer/deeppot_dpa_spin.pth");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    // EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    // expected_tot_v.resize(9);
    // std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (unsigned int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    // for (unsigned int ii = 0; ii < natoms; ++ii) {
    //   for (int dd = 0; dd < 9; ++dd) {
    //     expected_tot_v[dd] += expected_v[ii * 9 + dd];
    //   }
    // }
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepSpinAHPP, ValueTypes);

TYPED_TEST(TestInferDeepSpinAHPP, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;

  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
}

TYPED_TEST(TestInferDeepSpinAHPP, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  // EXPECT_EQ(atom_vir.size(), natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  // for (int ii = 0; ii < natoms * 9; ++ii) {
  //   EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  // }
}

TYPED_TEST(TestInferDeepSpinAHPP, print_summary) {
  deepmd::hpp::DeepSpin& dp = this->dp;
  dp.print_summary("");
}

template <class VALUETYPE>
class TestInferDeepSpinANoPbcHPP : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                                 0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {};
  std::vector<VALUETYPE> expected_e = {
      1.298915294144196e-02, 1.095576145701290e-01, 1.083914166945241e-01,
      4.932338375417146e-02, 1.099860785812512e-01, 1.100478936528533e-01,
  };
  std::vector<VALUETYPE> expected_f = {
      1.300765817095240e-02,  -1.593967210478553e-03, -3.196759265340465e-03,
      -1.300765817095220e-02, 1.593967210478477e-03,  3.196759265340509e-03,
      9.196695370628910e-03,  -9.044559760114149e-04, 5.658266727670325e-04,
      1.012744085443978e-02,  1.680427054831429e-02,  1.807036969424208e-02,
      -1.133453822298158e-02, -8.941333904804914e-03, -6.627672717506913e-03,
      -7.989598002087154e-03, -6.958480667497971e-03, -1.200852364950221e-02,
  };
  std::vector<VALUETYPE> expected_fm = {
      -9.651705644713781e-01, -1.704326891282164e-01, -2.605677204117113e-01,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      -9.168034653189444e-02, 1.736913887115685e-02,  -3.908906640474424e-02,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
      0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
  };
  unsigned int natoms;
  double expected_tot_e;
  // std::vector<VALUETYPE> expected_tot_v;

  deepmd::hpp::DeepSpin dp;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp.init("../../tests/infer/deeppot_dpa_spin.pth");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    // EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    // expected_tot_v.resize(9);
    // std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (unsigned int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    // for (unsigned int ii = 0; ii < natoms; ++ii) {
    //   for (int dd = 0; dd < 9; ++dd) {
    //     expected_tot_v[dd] += expected_v[ii * 9 + dd];
    //   }
    // }
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepSpinANoPbcHPP, ValueTypes);

TYPED_TEST(TestInferDeepSpinANoPbcHPP, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (unsigned int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (unsigned int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (unsigned int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
}

TYPED_TEST(TestInferDeepSpinANoPbcHPP, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::hpp::InputNlist inlist(natoms, &ilist[0], &numneigh[0],
                                 &firstneigh[0]);
  deepmd::hpp::convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box, 0, inlist,
             0);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
}

TYPED_TEST(TestInferDeepSpinANoPbcHPP, cpu_lmp_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  // std::vector<VALUETYPE>& expected_v = this->expected_v;
  unsigned int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  // std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::hpp::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::hpp::InputNlist inlist(natoms, &ilist[0], &numneigh[0],
                                 &firstneigh[0]);
  deepmd::hpp::convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box, 0, inlist, 0);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  // EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  // for (int ii = 0; ii < 3 * 3; ++ii) {
  //   EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  // }
}
