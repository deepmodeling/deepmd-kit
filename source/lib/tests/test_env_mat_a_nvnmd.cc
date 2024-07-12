// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <iostream>

#include "device.h"
#include "env_mat_nvnmd.h"
#include "fmt_nlist.h"
#include "neighbor_list.h"
#include "prod_env_mat_nvnmd.h"

class TestEnvMatANvnmd : public ::testing::Test {
 protected:
  std::vector<double> posi = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                              00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                              3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<double> posi_cpy;
  std::vector<int> atype_cpy;
  int nloc, nall;
  double rc = 6;
  double rc_smth = 0.8;
  SimulationRegion<double> region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 10, 20};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a, nlist_r;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  int ntypes = sec_a.size() - 1;
  int nnei = sec_a.back();
  int ndescrpt = nnei * 4;
  /* r_ij^2, x_ij, y_ij, z_ij */
  std::vector<double> expected_env = {
      12.791382, 3.529999,  0.440000,  -0.370000, 0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.957299,  -0.740000,
      0.310000,  0.560000,  1.003999,  0.420000,  0.760000,  -0.500000,
      13.721283, 3.679998,  -0.050000, 0.420000,  20.533585, 4.439999,
      0.660000,  -0.620000, 0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.957299,  0.740000,  -0.310000, -0.560000,
      19.114655, 4.269997,  0.130000,  -0.930000, 0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      2.671698,  1.160000,  0.450000,  -1.059999, 19.685577, 4.419998,
      -0.360000, -0.140000, 28.347244, 5.179996,  0.350000,  -1.179999,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  1.003999,  -0.420000,
      -0.760000, 0.500000,  9.791389,  3.109999,  -0.320000, 0.130000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  2.671698,  -1.160000, -0.450000, 1.059999,
      12.130081, 3.259998,  -0.810000, 0.920000,  16.184769, 4.019997,
      -0.100000, -0.120000, 0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      12.791382, -3.529999, -0.440000, 0.370000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.886699,  0.150000,
      -0.490000, 0.790000,  0.938999,  0.910000,  0.220000,  -0.250000,
      9.791389,  -3.109999, 0.320000,  -0.130000, 19.114655, -4.269997,
      -0.130000, 0.930000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.886699,  -0.150000, 0.490000,  -0.790000,
      13.721283, -3.679998, 0.050000,  -0.420000, 0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      2.163296,  0.760000,  0.710000,  -1.040000, 12.130081, -3.259998,
      0.810000,  -0.920000, 19.685577, -4.419998, 0.360000,  0.140000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.938999,  -0.910000,
      -0.220000, 0.250000,  20.533585, -4.439999, -0.660000, 0.620000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  2.163296,  -0.760000, -0.710000, 1.040000,
      16.184769, -4.019997, 0.100000,  0.120000,  28.347244, -5.179996,
      -0.350000, 1.179999,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000};

  void SetUp() override {
    double box[] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
    region.reinitBox(box);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc,
               region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nat_stt.resize(3);
    ext_stt.resize(3);
    ext_end.resize(3);
    for (int dd = 0; dd < 3; ++dd) {
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    build_nlist(nlist_a, nlist_r, posi, rc, rc, ncell, region);
    build_nlist(nlist_a_cpy, nlist_r_cpy, posi_cpy, nloc, rc, rc, nat_stt,
                ncell, ext_stt, ext_end, region, ncell);
  }
  void TearDown() override {}
};

class TestEnvMatANvnmdShortSel : public ::testing::Test {
 protected:
  std::vector<double> posi = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                              00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                              3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<double> posi_cpy;
  std::vector<int> atype_cpy;
  int nloc, nall;
  double rc = 6;
  double rc_smth = 0.8;
  SimulationRegion<double> region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 2, 4};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a, nlist_r;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  int ntypes = sec_a.size() - 1;
  int nnei = sec_a.back();
  int ndescrpt = nnei * 4;
  std::vector<double> expected_env = {
      12.791382, 3.529999,  0.440000,  -0.370000, 0.000000,  0.000000,
      0.000000,  0.000000,  0.957299,  -0.740000, 0.310000,  0.560000,
      1.003999,  0.420000,  0.760000,  -0.500000, 0.957299,  0.740000,
      -0.310000, -0.560000, 19.114655, 4.269997,  0.130000,  -0.930000,
      2.671698,  1.160000,  0.450000,  -1.059999, 19.685577, 4.419998,
      -0.360000, -0.140000, 1.003999,  -0.420000, -0.760000, 0.500000,
      9.791389,  3.109999,  -0.320000, 0.130000,  2.671698,  -1.160000,
      -0.450000, 1.059999,  12.130081, 3.259998,  -0.810000, 0.920000,
      12.791382, -3.529999, -0.440000, 0.370000,  0.000000,  0.000000,
      0.000000,  0.000000,  0.886699,  0.150000,  -0.490000, 0.790000,
      0.938999,  0.910000,  0.220000,  -0.250000, 0.886699,  -0.150000,
      0.490000,  -0.790000, 13.721283, -3.679998, 0.050000,  -0.420000,
      2.163296,  0.760000,  0.710000,  -1.040000, 12.130081, -3.259998,
      0.810000,  -0.920000, 0.938999,  -0.910000, -0.220000, 0.250000,
      20.533585, -4.439999, -0.660000, 0.620000,  2.163296,  -0.760000,
      -0.710000, 1.040000,  16.184769, -4.019997, 0.100000,  0.120000};

  void SetUp() override {
    double box[] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
    region.reinitBox(box);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc,
               region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nat_stt.resize(3);
    ext_stt.resize(3);
    ext_end.resize(3);
    for (int dd = 0; dd < 3; ++dd) {
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    build_nlist(nlist_a, nlist_r, posi, rc, rc, ncell, region);
    build_nlist(nlist_a_cpy, nlist_r_cpy, posi_cpy, nloc, rc, rc, nat_stt,
                ncell, ext_stt, ext_end, region, ncell);
  }
  void TearDown() override {}
};

/*  env_mat_a_nvnmd_quantize_cpu is not same as env_mat_a.
remove some tests:
TEST_F(TestEnvMatANvnmd, orig_cpy)
TEST_F(TestEnvMatANvnmd, orig_pbc)
TEST_F(TestEnvMatANvnmd, orig_cpy_equal_pbc)
TEST_F(TestEnvMatANvnmd, orig_cpy_num_deriv)
*/

TEST_F(TestEnvMatANvnmd, cpu) {
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = false;
  for (int ii = 0; ii < nloc; ++ii) {
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii,
                                         nlist_a_cpy[ii], rc, sec_a);
    EXPECT_EQ(ret, -1);
    deepmd::env_mat_a_nvnmd_quantize_cpu<double>(
        env, env_deriv, rij_a, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a,
        rc_smth, rc);
    EXPECT_EQ(env.size(), sec_a[2] * 4);
    EXPECT_EQ(env.size(), env_deriv.size() / 3);
    EXPECT_EQ(rij_a.size(), sec_a[2] * 3);
    for (int jj = 0; jj < sec_a[2]; ++jj) {
      for (int dd = 0; dd < 4; ++dd) {
        EXPECT_LT(fabs(env[jj * 4 + dd] -
                       expected_env[ii * sec_a[2] * 4 + jj * 4 + dd]),
                  1e-5);
      }
    }
  }
}

/*  env_mat_a_nvnmd_quantize_cpu is not same as env_mat_a.
remove some tests:
TEST_F(TestEnvMatANvnmd, cpu_equal_orig_cpy)
TEST_F(TestEnvMatANvnmd, cpu_num_deriv)
TEST_F(TestEnvMatANvnmdShortSel, orig_cpy)
TEST_F(TestEnvMatANvnmdShortSel, orig_pbc)
*/

TEST_F(TestEnvMatANvnmdShortSel, cpu) {
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = false;
  for (int ii = 0; ii < nloc; ++ii) {
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii,
                                         nlist_a_cpy[ii], rc, sec_a);
    EXPECT_EQ(ret, 1);
    deepmd::env_mat_a_nvnmd_quantize_cpu<double>(
        env, env_deriv, rij_a, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a,
        rc_smth, rc);
    EXPECT_EQ(env.size(), sec_a[2] * 4);
    EXPECT_EQ(env.size(), env_deriv.size() / 3);
    EXPECT_EQ(rij_a.size(), sec_a[2] * 3);
    for (int jj = 0; jj < sec_a[2]; ++jj) {
      for (int dd = 0; dd < 4; ++dd) {
        EXPECT_LT(fabs(env[jj * 4 + dd] -
                       expected_env[ii * sec_a[2] * 4 + jj * 4 + dd]),
                  1e-5);
      }
    }
  }
}

TEST_F(TestEnvMatANvnmd, prod_cpu) {
  EXPECT_EQ(nlist_r_cpy.size(), nloc);
  int tot_nnei = 0;
  int max_nbor_size = 0;
  for (int ii = 0; ii < nlist_a_cpy.size(); ++ii) {
    tot_nnei += nlist_a_cpy[ii].size();
    if (nlist_a_cpy[ii].size() > max_nbor_size) {
      max_nbor_size = nlist_a_cpy[ii].size();
    }
  }
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  deepmd::convert_nlist(inlist, nlist_a_cpy);

  std::vector<double> em(nloc * ndescrpt), em_deriv(nloc * ndescrpt * 3),
      rij(static_cast<size_t>(nloc) * nnei * 3);
  std::vector<int> nlist(nloc * nnei);
  std::vector<double> avg(ntypes * ndescrpt, 0);
  std::vector<double> std(ntypes * ndescrpt, 1);
  deepmd::prod_env_mat_a_nvnmd_quantize_cpu(
      &em[0], &em_deriv[0], &rij[0], &nlist[0], &posi_cpy[0], &atype_cpy[0],
      inlist, max_nbor_size, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);

  for (int ii = 0; ii < nloc; ++ii) {
    for (int jj = 0; jj < nnei; ++jj) {
      for (int dd = 0; dd < 4; ++dd) {
        EXPECT_LT(fabs(em[ii * nnei * 4 + jj * 4 + dd] -
                       expected_env[ii * nnei * 4 + jj * 4 + dd]),
                  1e-5);
      }
    }
  }
}

TEST_F(TestEnvMatANvnmd, prod_cpu_equal_cpu) {
  EXPECT_EQ(nlist_r_cpy.size(), nloc);
  int tot_nnei = 0;
  int max_nbor_size = 0;
  for (int ii = 0; ii < nlist_a_cpy.size(); ++ii) {
    tot_nnei += nlist_a_cpy[ii].size();
    if (nlist_a_cpy[ii].size() > max_nbor_size) {
      max_nbor_size = nlist_a_cpy[ii].size();
    }
  }
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_a_cpy);
  std::vector<double> em(nloc * ndescrpt), em_deriv(nloc * ndescrpt * 3),
      rij(static_cast<size_t>(nloc) * nnei * 3);
  std::vector<int> nlist(nloc * nnei);
  std::vector<double> avg(ntypes * ndescrpt, 0);
  std::vector<double> std(ntypes * ndescrpt, 1);
  deepmd::prod_env_mat_a_nvnmd_quantize_cpu(
      &em[0], &em_deriv[0], &rij[0], &nlist[0], &posi_cpy[0], &atype_cpy[0],
      inlist, max_nbor_size, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);

  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;
  std::vector<double> env_1, env_deriv_1, rij_a_1;
  for (int ii = 0; ii < nloc; ++ii) {
    int ret_1 = format_nlist_i_cpu<double>(fmt_nlist_a_1, posi_cpy, atype_cpy,
                                           ii, nlist_a_cpy[ii], rc, sec_a);
    EXPECT_EQ(ret_1, -1);
    deepmd::env_mat_a_nvnmd_quantize_cpu<double>(
        env_1, env_deriv_1, rij_a_1, posi_cpy, atype_cpy, ii, fmt_nlist_a_1,
        sec_a, rc_smth, rc);
    EXPECT_EQ(env_1.size(), nnei * 4);
    EXPECT_EQ(env_deriv_1.size(), nnei * 4 * 3);
    EXPECT_EQ(rij_a_1.size(), nnei * 3);
    EXPECT_EQ(fmt_nlist_a_1.size(), nnei);
    EXPECT_EQ(env_1.size() * nloc, em.size());
    EXPECT_EQ(env_deriv_1.size() * nloc, em_deriv.size());
    EXPECT_EQ(rij_a_1.size() * nloc, rij.size());
    EXPECT_EQ(fmt_nlist_a_1.size() * nloc, nlist.size());
    for (unsigned jj = 0; jj < env_1.size(); ++jj) {
      EXPECT_LT(fabs(em[ii * nnei * 4 + jj] - env_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < env_deriv_1.size(); ++jj) {
      EXPECT_LT(fabs(em_deriv[ii * nnei * 4 * 3 + jj] - env_deriv_1[jj]),
                1e-10);
    }
    for (unsigned jj = 0; jj < rij_a_1.size(); ++jj) {
      EXPECT_LT(fabs(rij[ii * nnei * 3 + jj] - rij_a_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < fmt_nlist_a_1.size(); ++jj) {
      EXPECT_EQ(nlist[ii * nnei + jj], fmt_nlist_a_1[jj]);
    }
  }

  for (int ii = 0; ii < nloc; ++ii) {
    for (int jj = 0; jj < nnei; ++jj) {
      for (int dd = 0; dd < 4; ++dd) {
        EXPECT_LT(fabs(em[ii * nnei * 4 + jj * 4 + dd] -
                       expected_env[ii * nnei * 4 + jj * 4 + dd]),
                  1e-5);
      }
    }
  }
}
