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
      -1.8626545229251095e+00, -2.3502165071948093e+00, -2.3500944968573521e+00,
      -2.0688274735854710e+00, -2.3485113271625320e+00, -2.3489022338537353e+00,
  };
  std::vector<VALUETYPE> expected_f = {
      3.7989110974834261e-02,  -6.8203560994098300e-02, 3.1554995279414300e-02,
      -6.0769407958790114e-02, 5.6658432967656878e-03,  2.1814741358389407e-02,
      1.5027739412753049e-02,  6.2090755323245192e-02,  -5.3346442187326704e-02,
      -5.2134406995188787e-02, 4.0990812807417676e-02,  -1.6987454510304811e-02,
      -6.7153786204261134e-03, -5.3801784772022326e-02, 5.6707773168242034e-02,
      6.6602343186817375e-02,  1.3257934338691726e-02,  -3.9743613108414025e-02,
  };
  std::vector<VALUETYPE> expected_fm = {
      4.8385521455777196e+00, 5.3158441514550137e-01, 1.0855626815019124e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      1.2140862110260138e+00, 9.6823434985033552e-01, 1.0689000529371890e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
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
      -1.9136796509970209e+00, -2.3532121417832528e+00,
      -2.3589759416772553e+00, -2.0689533840218703e+00,
      -2.3485273598793084e+00, -2.3489022338537353e+00};
  std::vector<VALUETYPE> expected_f = {
      5.2440246818294511e-02,  -8.2643189092284075e-03, -1.6057110078610215e-02,
      -5.2440246818295698e-02, 8.2643189092281334e-03,  1.6057110078610277e-02,
      -1.6724663644564395e-03, 7.9346065821642349e-05,  -2.5251632397208987e-04,
      -5.6934098675373246e-02, 4.0398593044712161e-02,  -1.6520316500527876e-02,
      -7.9878577602028808e-03, -5.3736758888210570e-02, 5.6516778947603999e-02,
      6.6594422800032166e-02,  1.3258819777676990e-02,  -3.9743946123104140e-02,
  };
  std::vector<VALUETYPE> expected_fm = {
      4.5904360179010135e+00, 6.2821415259365443e-01, 9.2483695213043082e-01,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      1.2125967529512662e+00, 9.6807902483755459e-01, 1.0691011858092361e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
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
