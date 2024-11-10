// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "deepmd.hpp"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepSpinModeDevi : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                                 0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  int natoms;

  deepmd::hpp::DeepSpin dp0;
  deepmd::hpp::DeepSpin dp1;
  deepmd::hpp::DeepSpinModelDevi dp_md;

  void SetUp() override {
    {
      dp0.init("../../tests/infer/deeppot_dpa_spin.pth");
    }
    {
      dp1.init("../../tests/infer/deeppot_dpa_spin.pth");
    }
    dp_md.init(
        std::vector<std::string>({"../../tests/infer/deeppot_dpa_spin.pth",
                                  "../../tests/infer/deeppot_dpa_spin.pth"}));
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepSpinModeDevi, ValueTypes);

TYPED_TEST(TestInferDeepSpinModeDevi, attrs) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::hpp::DeepSpin& dp0 = this->dp0;
  deepmd::hpp::DeepSpin& dp1 = this->dp1;
  deepmd::hpp::DeepSpinModelDevi& dp_md = this->dp_md;
  EXPECT_EQ(dp0.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp0.numb_types(), dp_md.numb_types());
  // EXPECT_EQ(dp0.dim_fparam(), dp_md.dim_fparam());
  // EXPECT_EQ(dp0.dim_aparam(), dp_md.dim_aparam());
  EXPECT_EQ(dp1.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp1.numb_types(), dp_md.numb_types());
  // EXPECT_EQ(dp1.dim_fparam(), dp_md.dim_fparam());
  // EXPECT_EQ(dp1.dim_aparam(), dp_md.dim_aparam());
}

TYPED_TEST(TestInferDeepSpinModeDevi, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::hpp::DeepSpin& dp0 = this->dp0;
  deepmd::hpp::DeepSpin& dp1 = this->dp1;
  deepmd::hpp::DeepSpinModelDevi& dp_md = this->dp_md;
  float rc = dp_md.cutoff();
  int nloc = coord.size() / 3;

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir(nmodel), fmagdir(nmodel),
      vdir(nmodel), fmd(nmodel), fmmagd(nmodel), vmd;
  dp0.compute(edir[0], fdir[0], fmagdir[0], vdir[0], coord, spin, atype, box);
  dp1.compute(edir[1], fdir[1], fmagdir[1], vdir[1], coord, spin, atype, box);
  dp_md.compute(emd, fmd, fmmagd, vmd, coord, spin, atype, box);

  EXPECT_EQ(edir.size(), emd.size());
  EXPECT_EQ(fdir.size(), fmd.size());
  EXPECT_EQ(fmagdir.size(), fmmagd.size());
  // EXPECT_EQ(vdir.size(), vmd.size());
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_EQ(fdir[kk].size(), fmd[kk].size());
    EXPECT_EQ(fmagdir[kk].size(), fmmagd[kk].size());
    // EXPECT_EQ(vdir[kk].size(), vmd[kk].size());
  }
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_LT(fabs(edir[kk] - emd[kk]), EPSILON);
    for (int ii = 0; ii < fdir[0].size(); ++ii) {
      EXPECT_LT(fabs(fdir[kk][ii] - fmd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < fmagdir[0].size(); ++ii) {
      EXPECT_LT(fabs(fmagdir[kk][ii] - fmmagd[kk][ii]), EPSILON);
    }
    // for (int ii = 0; ii < vdir[0].size(); ++ii) {
    //   EXPECT_LT(fabs(vdir[kk][ii] - vmd[kk][ii]), EPSILON);
    // }
  }
}

TYPED_TEST(TestInferDeepSpinModeDevi, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::hpp::DeepSpin& dp0 = this->dp0;
  deepmd::hpp::DeepSpin& dp1 = this->dp1;
  deepmd::hpp::DeepSpinModelDevi& dp_md = this->dp_md;

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir(nmodel), fmagdir(nmodel),
      vdir(nmodel), fmd(nmodel), fmmagd(nmodel), vmd, aedir(nmodel), aemd,
      avdir(nmodel), avmd(nmodel);
  dp0.compute(edir[0], fdir[0], fmagdir[0], vdir[0], aedir[0], avdir[0], coord,
              spin, atype, box);
  dp1.compute(edir[1], fdir[1], fmagdir[1], vdir[1], aedir[1], avdir[1], coord,
              spin, atype, box);
  dp_md.compute(emd, fmd, fmmagd, vmd, aemd, avmd, coord, spin, atype, box);

  EXPECT_EQ(edir.size(), emd.size());
  EXPECT_EQ(fdir.size(), fmd.size());
  EXPECT_EQ(fmagdir.size(), fmmagd.size());
  // EXPECT_EQ(vdir.size(), vmd.size());
  EXPECT_EQ(aedir.size(), aemd.size());
  // EXPECT_EQ(avdir.size(), avmd.size());
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_EQ(fdir[kk].size(), fmd[kk].size());
    EXPECT_EQ(fmagdir[kk].size(), fmmagd[kk].size());
    // EXPECT_EQ(vdir[kk].size(), vmd[kk].size());
    EXPECT_EQ(aedir[kk].size(), aemd[kk].size());
    // EXPECT_EQ(avdir[kk].size(), avmd[kk].size());
  }
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_LT(fabs(edir[kk] - emd[kk]), EPSILON);
    for (int ii = 0; ii < fdir[0].size(); ++ii) {
      EXPECT_LT(fabs(fdir[kk][ii] - fmd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < fmagdir[0].size(); ++ii) {
      EXPECT_LT(fabs(fmagdir[kk][ii] - fmmagd[kk][ii]), EPSILON);
    }
    // for (int ii = 0; ii < vdir[0].size(); ++ii) {
    //   EXPECT_LT(fabs(vdir[kk][ii] - vmd[kk][ii]), EPSILON);
    // }
    for (int ii = 0; ii < aedir[0].size(); ++ii) {
      EXPECT_LT(fabs(aedir[kk][ii] - aemd[kk][ii]), EPSILON);
    }
    // for (int ii = 0; ii < avdir[0].size(); ++ii) {
    //   EXPECT_LT(fabs(avdir[kk][ii] - avmd[kk][ii]), EPSILON);
    // }
  }
}
