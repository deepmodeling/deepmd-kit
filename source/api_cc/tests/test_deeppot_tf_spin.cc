// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "DeepSpin.h"
#include "neighbor_list.h"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepSpin : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0., 0., 1.2737, 0., 0., 1.2737,
                                 0., 0., 0.,     0., 0., 0.};
  std::vector<int> atype = {0, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> expected_e = {-7.314365618560289, -7.313531316181837,
                                       -2.8980532245013997, -2.897373810282277};
  std::vector<VALUETYPE> expected_f = {
      0.0275132293555514,  -0.0112057401883111, -0.0212278132621243,
      -0.0229926640905535, 0.0114378553363334,  0.019670014885563,
      0.0086502856137601,  0.0088926283192558,  -0.0127014507822769,
      -0.013170850878758,  -0.009124743467278,  0.0142592491588383};
  std::vector<VALUETYPE> expected_fm = {
      0.0066245455049449,  -0.0023055088004378, 0.0294608578045521,
      -0.0041979452385972, 0.0025775020220167,  0.0316295420619988,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000};
  std::vector<VALUETYPE> expected_v = {
      0.0070639867264982,  -0.0005923577001662, -0.0015491268442953,
      -0.0005741900039506, 0.0004072991754844,  0.0005919446476345,
      -0.0013659665914274, 0.0005245686552392,  0.0011288634277803,
      0.0074611996305919,  -0.0015158254500315, -0.0030704181444311,
      -0.0015503527871207, 0.0006417155838534,  0.0010901024672963,
      -0.0032762727340245, 0.0011481000769186,  0.0022122852076016,
      -0.0049637269273085, -0.0033079530214069, 0.0048850199723435,
      -0.0032277537906931, -0.0030526361938397, 0.0044721003136312,
      0.0053457625015160,  0.0044600355962439,  -0.0065441506206723,
      -0.0044231868209291, -0.0033953486551904, 0.0050014995082810,
      -0.0035584060948890, -0.0032308004485022, 0.0047399657455500,
      0.0056902937417672,  0.0047696802946761,  -0.0070004831270587,
      0.0034978220789713,  -0.0044217265408896, -0.0075771507215158,
      -0.0043265981217727, 0.0016344211766637,  0.0031438764476946,
      -0.0069613658908443, 0.0032277030414985,  0.0055466693735168,
      -0.0182670501038624, -0.0030197903610554, 0.0012333318415169,
      -0.0030157009303137, 0.0006787737562374,  0.0017594542103399,
      0.0025814653441594,  0.0020137939338955,  0.0014966802677115};
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::DeepSpin dp;

  void SetUp() override {
#ifndef BUILD_TENSORFLOW
    GTEST_SKIP() << "Skip because TensorFlow support is not enabled.";
#endif
    std::string file_name = "../../tests/infer/deepspin_nlist.pbtxt";
    deepmd::convert_pbtxt_to_pb("../../tests/infer/deepspin_nlist.pbtxt",
                                "deepspin_nlist.pb");

    dp.init("deepspin_nlist.pb");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    EXPECT_EQ((natoms + 2) * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.resize(9);
    std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    for (int ii = 0; ii < (natoms + 2); ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        expected_tot_v[dd] += expected_v[ii * 9 + dd];
      }
    }
  };

  void TearDown() override { remove("deepspin_nlist.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepSpin, ValueTypes);

TYPED_TEST(TestInferDeepSpin, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);
  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepSpin, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box);
  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);
  // EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), (natoms + 2) * 9);
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < (natoms + 2) * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  }
}

template <class VALUETYPE>
class TestInferDeepSpinNopbc : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0., 0., 1.2737, 0., 0., 1.2737,
                                 0., 0., 0.,     0., 0., 0.};
  std::vector<int> atype = {0, 0, 1, 1};
  std::vector<VALUETYPE> box = {100., 0., 0., 0., 100., 0., 0., 0., 100.};
  std::vector<VALUETYPE> expected_e = {-7.313160384523243, -7.312173646552338,
                                       -2.8984477845267067,
                                       -2.8984477845267067};
  std::vector<VALUETYPE> expected_f = {
      0.0277100137316238,  -0.0116082489956803, -0.0211484273275705,
      -0.0277100137316238, 0.0116082489956803,  0.0211484273275705,
      0.0097588349924651,  0.0091168063745397,  -0.0133541952528469,
      -0.0097588349924651, -0.0091168063745397, 0.0133541952528469};
  std::vector<VALUETYPE> expected_fm = {
      0.0058990325687816,  -0.0024712163463815, 0.0296682261295907,
      -0.0060028470719556, 0.0025147062058193,  0.0321884178873188,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000,
      0.0000000000000000,  0.00000000000000000, 0.00000000000000000};
  std::vector<VALUETYPE> expected_v = {
      0.0021380771762615,  -0.0008956809792447, -0.0016180043496033,
      -0.0008956809792447, 0.0003752177075214,  0.0006778126329419,
      -0.0014520530654550, 0.0006082925003933,  0.0010988509684524,
      0.0034592108484302,  -0.0014491288689370, -0.0026177811825959,
      -0.0014491288689370, 0.0006070674991493,  0.0010966380629793,
      -0.0027640824464858, 0.0011579264302846,  0.0020917380676109,
      -0.0037083572971367, -0.0034643864223251, 0.0050745941960818,
      -0.0034643864223251, -0.0032364662629616, 0.0047407393147607,
      0.0050745941960818,  0.0047407393147607,  -0.0069441815314804,
      -0.0037083572971367, -0.0034643864223251, 0.0050745941960818,
      -0.0034643864223251, -0.0032364662629616, 0.0047407393147607,
      0.0050745941960818,  0.0047407393147607,  -0.0069441815314804,
      0.0103691205704445,  -0.0043438207795105, -0.0078469020533093,
      -0.0043438207795105, 0.0018197087049301,  0.0032872157250350,
      -0.0076002352547860, 0.0031838823364644,  0.0057515293820002,
      0.0045390015662654,  -0.0019014736291112, -0.0034349201042009,
      -0.0019014736291112, 0.0007965632770601,  0.0014389530166247,
      -0.0038334654556754, 0.0016059112044046,  0.0029010008853761};
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::DeepSpin dp;

  void SetUp() override {
#ifndef BUILD_TENSORFLOW
    GTEST_SKIP() << "Skip because TensorFlow support is not enabled.";
#endif
    std::string file_name = "../../tests/infer/deepspin_nlist.pbtxt";
    deepmd::convert_pbtxt_to_pb("../../tests/infer/deepspin_nlist.pbtxt",
                                "deepspin_nlist.pb");

    dp.init("deepspin_nlist.pb");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    EXPECT_EQ((natoms + 2) * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.resize(9);
    std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    for (int ii = 0; ii < natoms + 2; ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        expected_tot_v[dd] += expected_v[ii * 9 + dd];
      }
    }
  };

  void TearDown() override { remove("deepspin_nlist.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepSpinNopbc, ValueTypes);

TYPED_TEST(TestInferDeepSpinNopbc, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepSpinNopbc, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);
  // EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  EXPECT_EQ(atom_vir.size(), (natoms + 2) * 9);

  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < (natoms + 2) * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepSpinNopbc, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;

  std::vector<std::vector<int> > nlist_data = {{1}, {0}, {3}, {2}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box, 0, inlist,
             0);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepSpinNopbc, cpu_lmp_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;

  std::vector<std::vector<int> > nlist_data = {{1}, {0}, {3}, {2}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box, 0, inlist, 0);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), (natoms + 2) * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < (natoms + 2) * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  }
}
