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
#include "neighbor_list.h"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepPotAPt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> expected_e = {-53.7147059, -49.94367351,
                                       -49.9434903, -53.71488615,
                                       -49.9438445, -49.94379288};
  std::vector<VALUETYPE> expected_f = {
      -0.00120548, 0.00809608,  -0.00064437, 0.00641376, -0.00109731,
      -0.00529648, -0.00551271, -0.0070066,  0.00597296, 0.00397807,
      -0.00078519, 0.00165795,  0.0003451,   0.00228841, -0.0036,
      -0.00401874, -0.00149539, 0.00190994};
  std::vector<VALUETYPE> expected_v = {
      -3.72974819e-03, -9.98227220e-04, 3.37664388e-03,  -1.10154761e-03,
      -4.68713907e-03, 1.84604369e-03,  3.41017698e-03,  1.81164563e-03,
      -3.22243690e-03, -1.99493894e-03, 5.97534768e-05,  1.80584573e-03,
      -2.62579356e-04, -1.12440024e-04, 2.42097709e-04,  1.89528468e-03,
      -3.24777812e-05, -1.60401783e-03, -8.97106345e-04, -1.15497829e-03,
      1.17916578e-03,  -7.55372233e-04, -8.56277743e-04, 7.92455483e-04,
      1.07793900e-03,  1.10623290e-03,  -1.12596261e-03, -2.90655831e-03,
      -3.70173918e-04, 2.98300648e-04,  -3.95329633e-04, -7.64980354e-04,
      1.16308129e-03,  3.15914358e-04,  1.15922497e-03,  -1.79216186e-03,
      3.24153403e-04,  -3.17404215e-04, 4.23089883e-04,  -1.83222341e-04,
      -4.67327552e-04, 7.21883299e-04,  2.39115354e-04,  7.09732333e-04,
      -1.09491651e-03, -3.87972886e-04, -3.39632835e-04, 4.29685964e-04,
      -4.22611827e-04, -2.23877230e-04, 3.02037131e-04,  5.74301510e-04,
      3.13240553e-04,  -4.22617216e-04};
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deeppot_sea.pth";

    dp.init(file_name);

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

  void TearDown() override { remove("deeppot.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepPotAPt, ValueTypes);

TYPED_TEST(TestInferDeepPotAPt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute(ener, force, virial, coord, atype, box);

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

TYPED_TEST(TestInferDeepPotAPt, cpu_build_nlist_numfv) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  class MyModel : public EnergyModelTest<VALUETYPE> {
    deepmd::DeepPot& mydp;
    const std::vector<int> atype;

   public:
    MyModel(deepmd::DeepPot& dp_, const std::vector<int>& atype_)
        : mydp(dp_), atype(atype_){};
    virtual void compute(double& ener,
                         std::vector<VALUETYPE>& force,
                         std::vector<VALUETYPE>& virial,
                         const std::vector<VALUETYPE>& coord,
                         const std::vector<VALUETYPE>& box) {
      mydp.compute(ener, force, virial, coord, atype, box);
    }
  };
  MyModel model(dp, atype);
  model.test_f(coord, box);
  model.test_v(coord, box);
  std::vector<VALUETYPE> box_(box);
  box_[1] -= 0.4;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[2] += 0.5;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[4] += 0.2;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[3] -= 0.3;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[6] -= 0.7;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[7] += 0.6;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
}

TYPED_TEST(TestInferDeepPotAPt, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial, atom_ener, atom_vir;
  dp.compute(ener, force, virial, atom_ener, atom_vir, coord, atype, box);

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

TYPED_TEST(TestInferDeepPotAPt, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
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
             inlist, 0);
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
             inlist, 1);
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

TYPED_TEST(TestInferDeepPotAPt, cpu_lmp_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
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
             box, nall - nloc, inlist, 0);
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
             box, nall - nloc, inlist, 1);
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

TYPED_TEST(TestInferDeepPotAPt, cpu_lmp_nlist_2rc) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
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
             inlist, 0);
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
             inlist, 1);
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

TYPED_TEST(TestInferDeepPotAPt, cpu_lmp_nlist_type_sel) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  float rc = dp.cutoff();

  // add vir atoms
  int nvir = 2;
  std::vector<VALUETYPE> coord_vir(nvir * 3);
  std::vector<int> atype_vir(nvir, 2);
  for (int ii = 0; ii < nvir; ++ii) {
    coord_vir[ii] = coord[ii];
  }
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  atype.insert(atype.begin(), atype_vir.begin(), atype_vir.end());
  natoms += nvir;
  std::vector<VALUETYPE> expected_f_vir(nvir * 3, 0.0);
  expected_f.insert(expected_f.begin(), expected_f_vir.begin(),
                    expected_f_vir.end());

  // build nlist
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

  // dp compute
  double ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(9, 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 0);
  // fold back
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
}

TYPED_TEST(TestInferDeepPotAPt, cpu_lmp_nlist_type_sel_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  float rc = dp.cutoff();

  // add vir atoms
  int nvir = 2;
  std::vector<VALUETYPE> coord_vir(nvir * 3);
  std::vector<int> atype_vir(nvir, 2);
  for (int ii = 0; ii < nvir; ++ii) {
    coord_vir[ii] = coord[ii];
  }
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  atype.insert(atype.begin(), atype_vir.begin(), atype_vir.end());
  natoms += nvir;
  std::vector<VALUETYPE> expected_f_vir(nvir * 3, 0.0);
  expected_f.insert(expected_f.begin(), expected_f_vir.begin(),
                    expected_f_vir.end());

  // build nlist
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

  // dp compute
  double ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(9, 0.0), atomic_energy,
      atomic_virial;
  dp.compute(ener, force_, virial, atomic_energy, atomic_virial, coord_cpy,
             atype_cpy, box, nall - nloc, inlist, 0);
  // fold back
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
}

TYPED_TEST(TestInferDeepPotAPt, print_summary) {
  deepmd::DeepPot& dp = this->dp;
  dp.print_summary("");
}

template <class VALUETYPE>
class TestInferDeepPotAPtNoPbc : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {};
  std::vector<VALUETYPE> expected_e = {-53.70576165, -49.9427077,  -49.68506898,
                                       -53.71490191, -49.94386836, -49.9438074};
  std::vector<VALUETYPE> expected_f = {
      -0.05793241, 0.02426898,  0.04384074,  0.05793241,  -0.02426898,
      -0.04384074, 0.0061137,   -0.00121445, -0.00052675, -0.04546273,
      0.00444365,  -0.00071007, 0.01261648,  -0.00107514, 0.00036128,
      0.02673255,  -0.00215406, 0.00087553};
  std::vector<VALUETYPE> expected_v = {
      -9.02965689e-03, 3.78269410e-03,  6.83325386e-03,  3.78269410e-03,
      -1.58464212e-03, -2.86257932e-03, 6.83325386e-03,  -2.86257932e-03,
      -5.17111103e-03, -3.38403230e-02, 1.41763515e-02,  2.56088931e-02,
      1.41763515e-02,  -5.93874186e-03, -1.07280498e-02, 2.56088931e-02,
      -1.07280498e-02, -1.93797029e-02, 2.16816985e-04,  -1.26051843e-05,
      4.08880185e-06,  -8.85403363e-06, 4.69563180e-08,  4.27221711e-07,
      -5.68134313e-06, 1.25803648e-06,  -1.28751543e-06, -1.55377477e-01,
      1.53205867e-02,  -6.03170230e-03, 1.53638777e-02,  -2.28187375e-03,
      1.66104531e-03,  -6.12092196e-03, 1.66503657e-03,  -1.81384836e-03,
      4.03275109e-02,  -1.02905681e-02, 1.17738547e-02,  -1.01342635e-02,
      2.05898924e-03,  -2.16858200e-03, 1.15576484e-02,  -2.17555574e-03,
      2.21173836e-03,  1.22038623e-01,  -3.36194524e-03, -3.25713729e-03,
      -3.56529198e-03, -1.12860297e-04, 3.54141978e-04,  -2.94194119e-03,
      3.56293646e-04,  -2.61599053e-04};
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deeppot_sea.pth";
    dp.init(file_name);

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

  void TearDown() override { remove("deeppot.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepPotAPtNoPbc, ValueTypes);

TYPED_TEST(TestInferDeepPotAPtNoPbc, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute(ener, force, virial, coord, atype, box);

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
