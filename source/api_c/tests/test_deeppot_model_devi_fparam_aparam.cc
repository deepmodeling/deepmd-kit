// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "deepmd.hpp"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepPotModeDeviFparamAparam : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 0, 0, 0, 0, 0};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> fparam = {0.25852028};
  std::vector<VALUETYPE> aparam = {0.25852028, 0.25852028, 0.25852028,
                                   0.25852028, 0.25852028, 0.25852028};
  int natoms;

  deepmd::hpp::DeepPot dp0;
  deepmd::hpp::DeepPot dp1;
  deepmd::hpp::DeepPotModelDevi dp_md;

  void SetUp() override {
    {
      std::string file_name = "../../tests/infer/fparam_aparam.pbtxt";
      deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/fparam_aparam.pbtxt",
                                       "fparam_aparam.pb");
      dp0.init("fparam_aparam.pb");
    }
    {
      // std::string file_name = "../../tests/infer/fparam_aparam.pbtxt";
      // deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/fparam_aparam.pbtxt",
      //                             "fparam_aparam.pb");
      dp1.init("fparam_aparam.pb");
    }
    dp_md.init(
        std::vector<std::string>({"fparam_aparam.pb", "fparam_aparam.pb"}));
  };

  void TearDown() override {
    remove("fparam_aparam.pb");
    // remove("fparam_aparam.pb");
  };
};

TYPED_TEST_SUITE(TestInferDeepPotModeDeviFparamAparam, ValueTypes);

TYPED_TEST(TestInferDeepPotModeDeviFparamAparam, attrs) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  int& natoms = this->natoms;
  deepmd::hpp::DeepPot& dp0 = this->dp0;
  deepmd::hpp::DeepPot& dp1 = this->dp1;
  deepmd::hpp::DeepPotModelDevi& dp_md = this->dp_md;
  EXPECT_EQ(dp0.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp0.numb_types(), dp_md.numb_types());
  EXPECT_EQ(dp0.dim_fparam(), dp_md.dim_fparam());
  EXPECT_EQ(dp0.dim_aparam(), dp_md.dim_aparam());
  EXPECT_EQ(dp1.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp1.numb_types(), dp_md.numb_types());
  EXPECT_EQ(dp1.dim_fparam(), dp_md.dim_fparam());
  EXPECT_EQ(dp1.dim_aparam(), dp_md.dim_aparam());
}

TYPED_TEST(TestInferDeepPotModeDeviFparamAparam, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  int& natoms = this->natoms;
  deepmd::hpp::DeepPot& dp0 = this->dp0;
  deepmd::hpp::DeepPot& dp1 = this->dp1;
  deepmd::hpp::DeepPotModelDevi& dp_md = this->dp_md;
  float rc = dp_md.cutoff();
  int nloc = coord.size() / 3;

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir(nmodel), vdir(nmodel), fmd(nmodel),
      vmd;
  dp0.compute(edir[0], fdir[0], vdir[0], coord, atype, box, fparam, aparam);
  dp1.compute(edir[1], fdir[1], vdir[1], coord, atype, box, fparam, aparam);
  dp_md.compute(emd, fmd, vmd, coord, atype, box, fparam, aparam);

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

TYPED_TEST(TestInferDeepPotModeDeviFparamAparam, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  int& natoms = this->natoms;
  deepmd::hpp::DeepPot& dp0 = this->dp0;
  deepmd::hpp::DeepPot& dp1 = this->dp1;
  deepmd::hpp::DeepPotModelDevi& dp_md = this->dp_md;

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir(nmodel), vdir(nmodel), fmd(nmodel),
      vmd, aedir(nmodel), aemd, avdir(nmodel), avmd(nmodel);
  dp0.compute(edir[0], fdir[0], vdir[0], aedir[0], avdir[0], coord, atype, box,
              fparam, aparam);
  dp1.compute(edir[1], fdir[1], vdir[1], aedir[1], avdir[1], coord, atype, box,
              fparam, aparam);
  dp_md.compute(emd, fmd, vmd, aemd, avmd, coord, atype, box, fparam, aparam);

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

TYPED_TEST(TestInferDeepPotModeDeviFparamAparam, cpu_lmp_list) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  int& natoms = this->natoms;
  deepmd::hpp::DeepPot& dp0 = this->dp0;
  deepmd::hpp::DeepPot& dp1 = this->dp1;
  deepmd::hpp::DeepPotModelDevi& dp_md = this->dp_md;
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
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir_(nmodel), fdir(nmodel),
      vdir(nmodel), fmd_, fmd(nmodel), vmd;
  dp0.compute(edir[0], fdir_[0], vdir[0], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0, fparam, aparam);
  dp1.compute(edir[1], fdir_[1], vdir[1], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0, fparam, aparam);
  dp_md.compute(emd, fmd_, vmd, coord_cpy, atype_cpy, box, nall - nloc, inlist,
                0, fparam, aparam);
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

TYPED_TEST(TestInferDeepPotModeDeviFparamAparam, cpu_lmp_list_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  int& natoms = this->natoms;
  deepmd::hpp::DeepPot& dp0 = this->dp0;
  deepmd::hpp::DeepPot& dp1 = this->dp1;
  deepmd::hpp::DeepPotModelDevi& dp_md = this->dp_md;
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
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir_(nmodel), fdir(nmodel),
      vdir(nmodel), fmd_, fmd(nmodel), vmd, aedir(nmodel), aemd, avdir(nmodel),
      avdir_(nmodel), avmd(nmodel), avmd_;
  dp0.compute(edir[0], fdir_[0], vdir[0], aedir[0], avdir_[0], coord_cpy,
              atype_cpy, box, nall - nloc, inlist, 0, fparam, aparam);
  dp1.compute(edir[1], fdir_[1], vdir[1], aedir[1], avdir_[1], coord_cpy,
              atype_cpy, box, nall - nloc, inlist, 0, fparam, aparam);
  dp_md.compute(emd, fmd_, vmd, aemd, avmd_, coord_cpy, atype_cpy, box,
                nall - nloc, inlist, 0, fparam, aparam);
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

TYPED_TEST(TestInferDeepPotModeDeviFparamAparam, cpu_lmp_list_std) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  int& natoms = this->natoms;
  deepmd::hpp::DeepPot& dp0 = this->dp0;
  deepmd::hpp::DeepPot& dp1 = this->dp1;
  deepmd::hpp::DeepPotModelDevi& dp_md = this->dp_md;
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
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  int nmodel = 2;
  std::vector<double> edir(nmodel), emd;
  std::vector<std::vector<VALUETYPE> > fdir_(nmodel), fdir(nmodel),
      vdir(nmodel), fmd_, fmd(nmodel), vmd;
  std::vector<std::vector<VALUETYPE> > aemd(nmodel), aemd_, avmd(nmodel), avmd_;
  dp0.compute(edir[0], fdir_[0], vdir[0], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0, fparam, aparam);
  dp1.compute(edir[1], fdir_[1], vdir[1], coord_cpy, atype_cpy, box,
              nall - nloc, inlist, 0, fparam, aparam);
  dp_md.compute(emd, fmd_, vmd, aemd_, avmd_, coord_cpy, atype_cpy, box,
                nall - nloc, inlist, 0, fparam, aparam);
  for (int kk = 0; kk < nmodel; ++kk) {
    _fold_back<VALUETYPE>(fdir[kk], fdir_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(fmd[kk], fmd_[kk], mapping, nloc, nall, 3);
    _fold_back<VALUETYPE>(avmd[kk], avmd_[kk], mapping, nloc, nall, 9);
    aemd[kk].resize(nloc);
    for (int ii = 0; ii < nloc; ++ii) {
      aemd[kk][ii] = aemd_[kk][ii];
    }
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
