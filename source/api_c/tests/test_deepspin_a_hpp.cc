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
      -4.736511666341149118e-02, 2.663324583045018068e-01,
      2.662707880956619588e-01,  -4.025182706753409334e-02,
      2.661383816099721633e-01,  2.663717506235512844e-01,
  };
  std::vector<VALUETYPE> expected_f = {
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
  std::vector<VALUETYPE> expected_fm = {
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
      -4.263824715539775434e-02, 2.659531961465150807e-01,
      2.679077652998275716e-01,  -4.042215423551187570e-02,
      2.666009600722629158e-01,  2.666060991401024149e-01,
  };
  std::vector<VALUETYPE> expected_f = {
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
  std::vector<VALUETYPE> expected_fm = {
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
