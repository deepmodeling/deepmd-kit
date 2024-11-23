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
class TestInferDeepPotModeDevi : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  int natoms;

  deepmd::DeepPot dp0;
  deepmd::DeepPot dp1;
  deepmd::DeepPotModelDevi dp_md;

  void SetUp() override {
    {
      std::string file_name = "../../tests/infer/deeppot.pbtxt";
      deepmd::convert_pbtxt_to_pb("../../tests/infer/deeppot.pbtxt",
                                  "deeppot.pb");
      dp0.init("deeppot.pb");
    }
    {
      std::string file_name = "../../tests/infer/deeppot-1.pbtxt";
      deepmd::convert_pbtxt_to_pb("../../tests/infer/deeppot-1.pbtxt",
                                  "deeppot-1.pb");
      dp1.init("deeppot-1.pb");
    }
    dp_md.init(std::vector<std::string>({"deeppot.pb", "deeppot-1.pb"}));
  };

  void TearDown() override {
    remove("deeppot.pb");
    remove("deeppot-1.pb");
  };
};

TYPED_TEST_SUITE(TestInferDeepPotModeDevi, ValueTypes);

template <class VALUETYPE>
class TestInferDeepPotModeDeviPython : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {
      4.170220047025740423e-02, 7.203244934421580703e-02,
      1.000114374817344942e-01, 4.053881673400336005e+00,
      4.191945144032948461e-02, 6.852195003967595510e-02,
      1.130233257263184132e+00, 1.467558908171130543e-02,
      1.092338594768797883e-01, 1.862602113776709242e-02,
      1.134556072704304919e+00, 1.396767474230670159e-01,
      5.120445224973151355e+00, 8.781174363909455272e-02,
      2.738759319792616331e-03, 4.067046751017840300e+00,
      1.141730480236712753e+00, 5.586898284457517128e-02,
  };
  std::vector<int> atype = {0, 0, 1, 1, 1, 1};
  std::vector<VALUETYPE> box = {20., 0., 0., 0., 20., 0., 0., 0., 20.};
  int natoms;
  std::vector<VALUETYPE> expected_md_f = {0.509504727653, 0.458424067748,
                                          0.481978258466};  // max min avg
  std::vector<VALUETYPE> expected_md_v = {0.167004837423, 0.00041822790564,
                                          0.0804864867641};  // max min avg

  deepmd::DeepPot dp0;
  deepmd::DeepPot dp1;
  deepmd::DeepPotModelDevi dp_md;

  void SetUp() override {
    {
      std::string file_name = "../../tests/infer/deeppot.pbtxt";
      deepmd::convert_pbtxt_to_pb("../../tests/infer/deeppot.pbtxt",
                                  "deeppot.pb");
      dp0.init("deeppot.pb");
    }
    {
      std::string file_name = "../../tests/infer/deeppot-1.pbtxt";
      deepmd::convert_pbtxt_to_pb("../../tests/infer/deeppot-1.pbtxt",
                                  "deeppot-1.pb");
      dp1.init("deeppot-1.pb");
    }
    dp_md.init(std::vector<std::string>({"deeppot.pb", "deeppot-1.pb"}));
  };

  void TearDown() override {
    remove("deeppot.pb");
    remove("deeppot-1.pb");
  };
};

TYPED_TEST_SUITE(TestInferDeepPotModeDeviPython, ValueTypes);

TYPED_TEST(TestInferDeepPotModeDevi, attrs) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::DeepPot& dp0 = this->dp0;
  deepmd::DeepPot& dp1 = this->dp1;
  deepmd::DeepPotModelDevi& dp_md = this->dp_md;
  EXPECT_EQ(dp0.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp0.numb_types(), dp_md.numb_types());
  EXPECT_EQ(dp0.dim_fparam(), dp_md.dim_fparam());
  EXPECT_EQ(dp0.dim_aparam(), dp_md.dim_aparam());
  EXPECT_EQ(dp1.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp1.numb_types(), dp_md.numb_types());
  EXPECT_EQ(dp1.dim_fparam(), dp_md.dim_fparam());
  EXPECT_EQ(dp1.dim_aparam(), dp_md.dim_aparam());
}

TYPED_TEST(TestInferDeepPotModeDevi, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::DeepPot& dp0 = this->dp0;
  deepmd::DeepPot& dp1 = this->dp1;
  deepmd::DeepPotModelDevi& dp_md = this->dp_md;
  float rc = dp_md.cutoff();
  int nloc = coord.size() / 3;

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir(nmodel), vdir(nmodel), fmd(nmodel),
      vmd;
  dp0.compute(edir[0], fdir[0], vdir[0], coord, atype, box);
  dp1.compute(edir[1], fdir[1], vdir[1], coord, atype, box);
  dp_md.compute(emd, fmd, vmd, coord, atype, box);

  EXPECT_EQ(edir.size(), emd.size());
  EXPECT_EQ(fdir.size(), fmd.size());
  EXPECT_EQ(vdir.size(), vmd.size());
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_EQ(fdir[kk].size(), fmd[kk].size());
    EXPECT_EQ(vdir[kk].size(), vmd[kk].size());
  }
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_LT(fabs(edir[kk] - emd[kk]), EPSILON);
    for (int ii = 0; ii < fdir[0].size(); ++ii) {
      EXPECT_LT(fabs(fdir[kk][ii] - fmd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < vdir[0].size(); ++ii) {
      EXPECT_LT(fabs(vdir[kk][ii] - vmd[kk][ii]), EPSILON);
    }
  }
}

TYPED_TEST(TestInferDeepPotModeDevi, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::DeepPot& dp0 = this->dp0;
  deepmd::DeepPot& dp1 = this->dp1;
  deepmd::DeepPotModelDevi& dp_md = this->dp_md;

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir(nmodel), vdir(nmodel), fmd(nmodel),
      vmd, aedir(nmodel), aemd, avdir(nmodel), avmd(nmodel);
  dp0.compute(edir[0], fdir[0], vdir[0], aedir[0], avdir[0], coord, atype, box);
  dp1.compute(edir[1], fdir[1], vdir[1], aedir[1], avdir[1], coord, atype, box);
  dp_md.compute(emd, fmd, vmd, aemd, avmd, coord, atype, box);

  EXPECT_EQ(edir.size(), emd.size());
  EXPECT_EQ(fdir.size(), fmd.size());
  EXPECT_EQ(vdir.size(), vmd.size());
  EXPECT_EQ(aedir.size(), aemd.size());
  EXPECT_EQ(avdir.size(), avmd.size());
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_EQ(fdir[kk].size(), fmd[kk].size());
    EXPECT_EQ(vdir[kk].size(), vmd[kk].size());
    EXPECT_EQ(aedir[kk].size(), aemd[kk].size());
    EXPECT_EQ(avdir[kk].size(), avmd[kk].size());
  }
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_LT(fabs(edir[kk] - emd[kk]), EPSILON);
    for (int ii = 0; ii < fdir[0].size(); ++ii) {
      EXPECT_LT(fabs(fdir[kk][ii] - fmd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < vdir[0].size(); ++ii) {
      EXPECT_LT(fabs(vdir[kk][ii] - vmd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < aedir[0].size(); ++ii) {
      EXPECT_LT(fabs(aedir[kk][ii] - aemd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < avdir[0].size(); ++ii) {
      EXPECT_LT(fabs(avdir[kk][ii] - avmd[kk][ii]), EPSILON);
    }
  }
}

TYPED_TEST(TestInferDeepPotModeDevi, cpu_lmp_list) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::DeepPot& dp0 = this->dp0;
  deepmd::DeepPot& dp1 = this->dp1;
  deepmd::DeepPotModelDevi& dp_md = this->dp_md;
  float rc = dp_md.cutoff();
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

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir_(nmodel), fdir(nmodel),
      vdir(nmodel), fmd_, fmd(nmodel), vmd;
  dp0.compute(edir[0], fdir_[0], vdir[0], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0);
  dp1.compute(edir[1], fdir_[1], vdir[1], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0);
  dp_md.compute(emd, fmd_, vmd, coord_cpy, atype_cpy, box, nall - nloc, inlist,
                0);
  for (int kk = 0; kk < nmodel; ++kk) {
    _fold_back<VALUETYPE>(fdir[kk], fdir_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(fmd[kk], fmd_[kk], mapping, nloc, nall, 3);
  }

  EXPECT_EQ(edir.size(), emd.size());
  EXPECT_EQ(fdir.size(), fmd.size());
  EXPECT_EQ(vdir.size(), vmd.size());
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_EQ(fdir[kk].size(), fmd[kk].size());
    EXPECT_EQ(vdir[kk].size(), vmd[kk].size());
  }
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_LT(fabs(edir[kk] - emd[kk]), EPSILON);
    for (int ii = 0; ii < fdir[0].size(); ++ii) {
      EXPECT_LT(fabs(fdir[kk][ii] - fmd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < vdir[0].size(); ++ii) {
      EXPECT_LT(fabs(vdir[kk][ii] - vmd[kk][ii]), EPSILON);
    }
  }
}

TYPED_TEST(TestInferDeepPotModeDevi, cpu_lmp_list_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::DeepPot& dp0 = this->dp0;
  deepmd::DeepPot& dp1 = this->dp1;
  deepmd::DeepPotModelDevi& dp_md = this->dp_md;
  float rc = dp_md.cutoff();
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

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir_(nmodel), fdir(nmodel),
      vdir(nmodel), fmd_, fmd(nmodel), vmd, aedir(nmodel), aemd, avdir(nmodel),
      avdir_(nmodel), avmd(nmodel), avmd_;
  dp0.compute(edir[0], fdir_[0], vdir[0], aedir[0], avdir_[0], coord_cpy,
              atype_cpy, box, nall - nloc, inlist, 0);
  dp1.compute(edir[1], fdir_[1], vdir[1], aedir[1], avdir_[1], coord_cpy,
              atype_cpy, box, nall - nloc, inlist, 0);
  dp_md.compute(emd, fmd_, vmd, aemd, avmd_, coord_cpy, atype_cpy, box,
                nall - nloc, inlist, 0);
  for (int kk = 0; kk < nmodel; ++kk) {
    _fold_back<VALUETYPE>(fdir[kk], fdir_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(fmd[kk], fmd_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(avdir[kk], avdir_[kk], mapping, nloc, nall, 9);
    _fold_back<VALUETYPE>(avmd[kk], avmd_[kk], mapping, nloc, nall, 9);
  }

  EXPECT_EQ(edir.size(), emd.size());
  EXPECT_EQ(fdir.size(), fmd.size());
  EXPECT_EQ(vdir.size(), vmd.size());
  EXPECT_EQ(aedir.size(), aemd.size());
  EXPECT_EQ(avdir.size(), avmd.size());
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_EQ(fdir[kk].size(), fmd[kk].size());
    EXPECT_EQ(vdir[kk].size(), vmd[kk].size());
    EXPECT_EQ(aedir[kk].size(), aemd[kk].size());
    EXPECT_EQ(avdir[kk].size(), avmd[kk].size());
  }
  for (int kk = 0; kk < nmodel; ++kk) {
    EXPECT_LT(fabs(edir[kk] - emd[kk]), EPSILON);
    for (int ii = 0; ii < fdir[0].size(); ++ii) {
      EXPECT_LT(fabs(fdir[kk][ii] - fmd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < vdir[0].size(); ++ii) {
      EXPECT_LT(fabs(vdir[kk][ii] - vmd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < aedir[0].size(); ++ii) {
      EXPECT_LT(fabs(aedir[kk][ii] - aemd[kk][ii]), EPSILON);
    }
    for (int ii = 0; ii < avdir[0].size(); ++ii) {
      EXPECT_LT(fabs(avdir[kk][ii] - avmd[kk][ii]), EPSILON);
    }
  }
}

TYPED_TEST(TestInferDeepPotModeDevi, cpu_lmp_list_std) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::DeepPot& dp0 = this->dp0;
  deepmd::DeepPot& dp1 = this->dp1;
  deepmd::DeepPotModelDevi& dp_md = this->dp_md;
  float rc = dp_md.cutoff();
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

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir_(nmodel), fdir(nmodel),
      vdir(nmodel), fmd_, fmd(nmodel), vmd;
  std::vector<std::vector<VALUETYPE> > aemd(nmodel), aemd_, avmd(nmodel), avmd_;
  dp0.compute(edir[0], fdir_[0], vdir[0], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0);
  dp1.compute(edir[1], fdir_[1], vdir[1], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0);
  dp_md.compute(emd, fmd_, vmd, aemd_, avmd_, coord_cpy, atype_cpy, box,
                nall - nloc, inlist, 0);
  for (int kk = 0; kk < nmodel; ++kk) {
    _fold_back<VALUETYPE>(fdir[kk], fdir_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(fmd[kk], fmd_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(avmd[kk], avmd_[kk], mapping, nloc, nall, 9);
    aemd[kk].resize(nloc);
    for (int ii = 0; ii < nloc; ++ii) {
      aemd[kk][ii] = aemd_[kk][ii];
    }
  }

  // dp compute std e
  std::vector<VALUETYPE> avg_e, std_e;
  dp_md.compute_avg(avg_e, aemd);
  dp_md.compute_std_e(std_e, avg_e, aemd);

  // manual compute std e
  std::vector<double> manual_avg_e(nloc);
  std::vector<double> manual_std_e(nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    double avg_e(0.0);
    for (int kk = 0; kk < nmodel; ++kk) {
      avg_e += aemd[kk][ii];
    }
    avg_e /= nmodel;
    manual_avg_e[ii] = avg_e;
    double std = 0;
    for (int kk = 0; kk < nmodel; ++kk) {
      std += (aemd[kk][ii] - avg_e) * (aemd[kk][ii] - avg_e);
    }
    std = sqrt(std / nmodel);
    manual_std_e[ii] = std;
  }
  EXPECT_EQ(manual_std_e.size(), std_e.size());
  for (int ii = 0; ii < std_e.size(); ++ii) {
    EXPECT_LT(fabs(manual_avg_e[ii] - avg_e[ii]), EPSILON);
    EXPECT_LT(fabs(manual_std_e[ii] - std_e[ii]), EPSILON);
  }

  // dp compute std f
  std::vector<VALUETYPE> avg_f, std_f;
  dp_md.compute_avg(avg_f, fmd);
  dp_md.compute_std_f(std_f, avg_f, fmd);

  // manual compute std f
  std::vector<VALUETYPE> manual_std_f(nloc);
  std::vector<VALUETYPE> manual_rel_std_f(nloc);
  VALUETYPE eps = 0.2;
  EXPECT_EQ(fmd[0].size(), nloc * 3);
  for (int ii = 0; ii < nloc; ++ii) {
    std::vector<VALUETYPE> avg_f(3, 0.0);
    for (int dd = 0; dd < 3; ++dd) {
      for (int kk = 0; kk < nmodel; ++kk) {
        avg_f[dd] += fmd[kk][ii * 3 + dd];
      }
      avg_f[dd] /= (nmodel) * 1.0;
    }
    VALUETYPE std = 0.;
    for (int kk = 0; kk < nmodel; ++kk) {
      for (int dd = 0; dd < 3; ++dd) {
        VALUETYPE tmp = fmd[kk][ii * 3 + dd] - avg_f[dd];
        std += tmp * tmp;
      }
    }
    VALUETYPE f_norm = 0;
    for (int dd = 0; dd < 3; ++dd) {
      f_norm += avg_f[dd] * avg_f[dd];
    }
    f_norm = sqrt(f_norm);
    std /= nmodel * 1.0;
    manual_std_f[ii] = sqrt(std);
    manual_rel_std_f[ii] = manual_std_f[ii] / (f_norm + eps);
  }

  EXPECT_EQ(manual_std_f.size(), std_f.size());
  for (int ii = 0; ii < std_f.size(); ++ii) {
    EXPECT_LT(fabs(manual_std_f[ii] - std_f[ii]), EPSILON);
  }
  dp_md.compute_relative_std_f(std_f, avg_f, eps);
  EXPECT_EQ(manual_std_f.size(), std_f.size());
  for (int ii = 0; ii < std_f.size(); ++ii) {
    EXPECT_LT(fabs(manual_rel_std_f[ii] - std_f[ii]), EPSILON);
  }
}

template <class VALUETYPE>
inline VALUETYPE mymax(const std::vector<VALUETYPE>& xx) {
  VALUETYPE ret = 0;
  for (int ii = 0; ii < xx.size(); ++ii) {
    if (xx[ii] > ret) {
      ret = xx[ii];
    }
  }
  return ret;
};
template <class VALUETYPE>
inline VALUETYPE mymin(const std::vector<VALUETYPE>& xx) {
  VALUETYPE ret = 1e10;
  for (int ii = 0; ii < xx.size(); ++ii) {
    if (xx[ii] < ret) {
      ret = xx[ii];
    }
  }
  return ret;
};
template <class VALUETYPE>
inline VALUETYPE myavg(const std::vector<VALUETYPE>& xx) {
  VALUETYPE ret = 0;
  for (int ii = 0; ii < xx.size(); ++ii) {
    ret += xx[ii];
  }
  return (ret / xx.size());
};
template <class VALUETYPE>
inline VALUETYPE mystd(const std::vector<VALUETYPE>& xx) {
  VALUETYPE ret = 0;
  for (int ii = 0; ii < xx.size(); ++ii) {
    ret += xx[ii] * xx[ii];
  }
  return sqrt(ret / xx.size());
};

TYPED_TEST(TestInferDeepPotModeDeviPython, cpu_lmp_list_std) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;
  deepmd::DeepPot& dp0 = this->dp0;
  deepmd::DeepPot& dp1 = this->dp1;
  deepmd::DeepPotModelDevi& dp_md = this->dp_md;
  std::vector<VALUETYPE>& expected_md_f = this->expected_md_f;
  std::vector<VALUETYPE>& expected_md_v = this->expected_md_v;
  float rc = dp_md.cutoff();
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

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir_(nmodel), fdir(nmodel),
      vdir(nmodel), fmd_, fmd(nmodel), vmd;
  std::vector<std::vector<VALUETYPE> > aemd(nmodel), aemd_, avmd(nmodel), avmd_;
  dp0.compute(edir[0], fdir_[0], vdir[0], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0);
  dp1.compute(edir[1], fdir_[1], vdir[1], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0);
  dp_md.compute(emd, fmd_, vmd, aemd_, avmd_, coord_cpy, atype_cpy, box,
                nall - nloc, inlist, 0);
  for (int kk = 0; kk < nmodel; ++kk) {
    _fold_back<VALUETYPE>(fdir[kk], fdir_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(fmd[kk], fmd_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(avmd[kk], avmd_[kk], mapping, nloc, nall, 9);
    aemd[kk].resize(nloc);
    for (int ii = 0; ii < nloc; ++ii) {
      aemd[kk][ii] = aemd_[kk][ii];
    }
  }

  // dp compute std e
  std::vector<VALUETYPE> avg_e, std_e;
  dp_md.compute_avg(avg_e, aemd);
  dp_md.compute_std_e(std_e, avg_e, aemd);

  // dp compute std f
  std::vector<VALUETYPE> avg_f, std_f;
  dp_md.compute_avg(avg_f, fmd);
  dp_md.compute_std_f(std_f, avg_f, fmd);
  EXPECT_LT(fabs(mymax(std_f) - expected_md_f[0]), EPSILON);
  EXPECT_LT(fabs(mymin(std_f) - expected_md_f[1]), EPSILON);
  EXPECT_LT(fabs(myavg(std_f) - expected_md_f[2]), EPSILON);

  // dp compute std v
  // we normalize v by number of atoms
  for (int ii = 0; ii < vmd.size(); ++ii) {
    for (int jj = 0; jj < vmd[ii].size(); ++jj) {
      vmd[ii][jj] /= VALUETYPE(atype.size());
    }
  }
  std::vector<VALUETYPE> avg_v, std_v;
  dp_md.compute_avg(avg_v, vmd);
  dp_md.compute_std(std_v, avg_v, vmd, 1);
  EXPECT_LT(fabs(mymax(std_v) - expected_md_v[0]), EPSILON);
  EXPECT_LT(fabs(mymin(std_v) - expected_md_v[1]), EPSILON);
  EXPECT_LT(fabs(mystd(std_v) - expected_md_v[2]), EPSILON);
}
