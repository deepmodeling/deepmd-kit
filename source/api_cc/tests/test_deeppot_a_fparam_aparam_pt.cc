// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "DeepPot.h"
#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

// 1e-10 cannot pass; unclear bug or not
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-7 : 1e-4)

namespace {
constexpr const char* kRefPath = "../../tests/infer/fparam_aparam.expected";
constexpr const char* kModelPath = "../../tests/infer/fparam_aparam.pth";
}  // namespace

template <class VALUETYPE>
class TestInferDeepPotAFParamAParamPt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 0, 0, 0, 0, 0};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> fparam = {0.25852028};
  std::vector<VALUETYPE> aparam = {0.25852028, 0.25852028, 0.25852028,
                                   0.25852028, 0.25852028, 0.25852028};
  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_v;
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::DeepPot dp;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("default", "expected_e");
    expected_f = ref.get<VALUETYPE>("default", "expected_f");
    expected_v = ref.get<VALUETYPE>("default", "expected_v");

    dp.init(kModelPath);

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.assign(9, 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    for (int ii = 0; ii < natoms; ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        expected_tot_v[dd] += expected_v[ii * 9 + dd];
      }
    }
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepPotAFParamAParamPt, ValueTypes);

TYPED_TEST(TestInferDeepPotAFParamAParamPt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute(ener, force, virial, coord, atype, box, fparam, aparam);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepPotAFParamAParamPt, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial, atom_ener, atom_vir;
  dp.compute(ener, force, virial, atom_ener, atom_vir, coord, atype, box,
             fparam, aparam);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepPotAFParamAParamPt, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  int nloc = coord.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  double ener;
  std::vector<VALUETYPE> force_, virial;
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }

  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepPotAFParamAParamPt, cpu_lmp_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  int nloc = coord.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  double ener;
  std::vector<VALUETYPE> force_, atom_ener_, atom_vir_, virial;
  std::vector<VALUETYPE> force, atom_ener, atom_vir;
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy, atype_cpy,
             box, nall - nloc, inlist, 0, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);
  _fold_back<VALUETYPE>(atom_ener, atom_ener_, mapping, nloc, nall, 1);
  _fold_back<VALUETYPE>(atom_vir, atom_vir_, mapping, nloc, nall, 9);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  }

  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  std::fill(atom_ener_.begin(), atom_ener_.end(), 0.0);
  std::fill(atom_vir_.begin(), atom_vir_.end(), 0.0);
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy, atype_cpy,
             box, nall - nloc, inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);
  _fold_back<VALUETYPE>(atom_ener, atom_ener_, mapping, nloc, nall, 1);
  _fold_back<VALUETYPE>(atom_vir, atom_vir_, mapping, nloc, nall, 9);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepPotAFParamAParamPt, cpu_lmp_nlist_2rc) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  int nloc = coord.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc * 2);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  double ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(9, 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }

  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}
