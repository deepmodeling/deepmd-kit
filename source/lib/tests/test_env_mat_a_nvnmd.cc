#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat_nvnmd.h"
#include "prod_env_mat_nvnmd.h"
#include "neighbor_list.h"
#include "device.h"


class TestEnvMatANvnmd : public ::testing::Test
{
protected:
  std::vector<double > posi = {12.83, 2.56, 2.18, 
			       12.09, 2.87, 2.74,
			       00.25, 3.32, 1.68,
			       3.36, 3.00, 1.81,
			       3.51, 2.51, 2.60,
			       4.27, 3.22, 1.56
  };
  std::vector<int > atype = {0, 1, 1, 0, 1, 1};
  std::vector<double > posi_cpy;
  std::vector<int > atype_cpy;
  int nloc, nall;
  double rc = 6;
  double rc_smth = 0.8;
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 10, 20};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a, nlist_r;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  int ntypes = sec_a.size()-1;
  int nnei = sec_a.back();
  int ndescrpt = nnei * 4;
  /* r_ij^2, x_ij, y_ij, z_ij */
  std::vector<double > expected_env = {
    12.79150,  3.53003,  0.43994, -0.37000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.95728, -0.73999,  0.31006,  0.56006,  1.00403,  0.42004,  0.76001, -0.50000, 13.72168,  3.68005, -0.05005,  0.42004, 20.53308,  4.43994,  0.66003, -0.62000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, 
    0.95728,  0.73999, -0.31006, -0.56006, 19.11487,  4.27002,  0.13000, -0.93005,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  2.67175,  1.16003,  0.44995, -1.06006, 19.68591,  4.42004, -0.35999, -0.14001, 28.34790,  5.18005,  0.34998, -1.18005,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, 
    1.00403, -0.42004, -0.76001,  0.50000,  9.79126,  3.10999, -0.31995,  0.13000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  2.67175, -1.16003, -0.44995,  1.06006, 12.13025,  3.26001, -0.81006,  0.92004, 16.18494,  4.02002, -0.09998, -0.12000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, 
    12.79150, -3.53003, -0.43994,  0.37000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.88672,  0.15002, -0.48999,  0.79004,  0.93896,  0.91003,  0.21997, -0.25000,  9.79126, -3.10999,  0.31995, -0.13000, 19.11487, -4.27002, -0.13000,  0.93005,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, 
    0.88672, -0.15002,  0.48999, -0.79004, 13.72168, -3.68005,  0.05005, -0.42004,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  2.16333,  0.76001,  0.70996, -1.04004, 12.13025, -3.26001,  0.81006, -0.92004, 19.68591, -4.42004,  0.35999,  0.14001,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, 
    0.93896, -0.91003, -0.21997,  0.25000, 20.53308, -4.43994, -0.66003,  0.62000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  2.16333, -0.76001, -0.70996,  1.04004, 16.18494, -4.02002,  0.09998,  0.12000, 28.34790, -5.18005, -0.34998,  1.18005,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000
  };
  
  void SetUp() override {
    double box[] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
    region.reinitBox(box);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc, region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nat_stt.resize(3);
    ext_stt.resize(3);
    ext_end.resize(3);
    for (int dd = 0; dd < 3; ++dd){
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    build_nlist(nlist_a, nlist_r, posi, rc, rc, ncell, region);
    build_nlist(nlist_a_cpy, nlist_r_cpy, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
  }
  void TearDown() override {
  }
};


class TestEnvMatANvnmdShortSel : public ::testing::Test
{
protected:
  std::vector<double > posi = {12.83, 2.56, 2.18, 
			       12.09, 2.87, 2.74,
			       00.25, 3.32, 1.68,
			       3.36, 3.00, 1.81,
			       3.51, 2.51, 2.60,
			       4.27, 3.22, 1.56
  };
  std::vector<int > atype = {0, 1, 1, 0, 1, 1};
  std::vector<double > posi_cpy;
  std::vector<int > atype_cpy;
  int nloc, nall;
  double rc = 6;
  double rc_smth = 0.8;
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 2, 4};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a, nlist_r;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  int ntypes = sec_a.size()-1;
  int nnei = sec_a.back();
  int ndescrpt = nnei * 4;
  std::vector<double > expected_env = {
    12.79150,  3.53003,  0.43994, -0.37000,  0.00000,  0.00000,  0.00000,  0.00000,  0.95728, -0.73999,  0.31006,  0.56006,  1.00403,  0.42004,  0.76001, -0.50000, 
    0.95728,  0.73999, -0.31006, -0.56006, 19.11487,  4.27002,  0.13000, -0.93005,  2.67175,  1.16003,  0.44995, -1.06006, 19.68591,  4.42004, -0.35999, -0.14001, 
    1.00403, -0.42004, -0.76001,  0.50000,  9.79126,  3.10999, -0.31995,  0.13000,  2.67175, -1.16003, -0.44995,  1.06006, 12.13025,  3.26001, -0.81006,  0.92004, 
    12.79150, -3.53003, -0.43994,  0.37000,  0.00000,  0.00000,  0.00000,  0.00000,  0.88672,  0.15002, -0.48999,  0.79004,  0.93896,  0.91003,  0.21997, -0.25000, 
    0.88672, -0.15002,  0.48999, -0.79004, 13.72168, -3.68005,  0.05005, -0.42004,  2.16333,  0.76001,  0.70996, -1.04004, 12.13025, -3.26001,  0.81006, -0.92004, 
    0.93896, -0.91003, -0.21997,  0.25000, 20.53308, -4.43994, -0.66003,  0.62000,  2.16333, -0.76001, -0.70996,  1.04004, 16.18494, -4.02002,  0.09998,  0.12000
  };  
  
  void SetUp() override {
    double box[] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
    region.reinitBox(box);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc, region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nat_stt.resize(3);
    ext_stt.resize(3);
    ext_end.resize(3);
    for (int dd = 0; dd < 3; ++dd){
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    build_nlist(nlist_a, nlist_r, posi, rc, rc, ncell, region);
    build_nlist(nlist_a_cpy, nlist_r_cpy, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
  }
  void TearDown() override {
  }
};


/*  env_mat_a_nvnmd_quantize_cpu is not same as env_mat_a.
remove some tests:
TEST_F(TestEnvMatANvnmd, orig_cpy)
TEST_F(TestEnvMatANvnmd, orig_pbc)
TEST_F(TestEnvMatANvnmd, orig_cpy_equal_pbc)
TEST_F(TestEnvMatANvnmd, orig_cpy_num_deriv)
*/

TEST_F(TestEnvMatANvnmd, cpu)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = false;
  double precs[3] = {8192, 1024, 16}; // NBIT_DATA_FL, NBIT_FEA_X, NBIT_FEA_X_FL
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);    
    EXPECT_EQ(ret, -1);
    deepmd::env_mat_a_nvnmd_quantize_cpu<double>(env, env_deriv, rij_a, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc, precs);    
    EXPECT_EQ(env.size(), sec_a[2]*4);
    EXPECT_EQ(env.size(), env_deriv.size()/3);
    EXPECT_EQ(rij_a.size(), sec_a[2]*3);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      for (int dd = 0; dd < 4; ++dd){
    	  EXPECT_LT(fabs(env[jj*4+dd] - expected_env[ii*sec_a[2]*4 + jj*4 + dd]) , 1e-5);
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


TEST_F(TestEnvMatANvnmdShortSel, cpu)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = false;
  double precs[3] = {8192, 1024, 16}; // NBIT_DATA_FL, NBIT_FEA_X, NBIT_FEA_X_FL
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);    
    EXPECT_EQ(ret, 1);
    deepmd::env_mat_a_nvnmd_quantize_cpu<double>(env, env_deriv, rij_a, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc, precs);    
    EXPECT_EQ(env.size(), sec_a[2]*4);
    EXPECT_EQ(env.size(), env_deriv.size()/3);
    EXPECT_EQ(rij_a.size(), sec_a[2]*3);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      for (int dd = 0; dd < 4; ++dd){
    	EXPECT_LT(fabs(env[jj*4+dd] - expected_env[ii*sec_a[2]*4 + jj*4 + dd]) , 1e-5);
      }
    }
  }
}


TEST_F(TestEnvMatANvnmd, prod_cpu)
{
  EXPECT_EQ(nlist_r_cpy.size(), nloc);
  int tot_nnei = 0;
  int max_nbor_size = 0;
  double precs[3] = {8192, 1024, 16}; // NBIT_DATA_FL, NBIT_FEA_X, NBIT_FEA_X_FL
  for(int ii = 0; ii < nlist_a_cpy.size(); ++ii){
    tot_nnei += nlist_a_cpy[ii].size();
    if (nlist_a_cpy[ii].size() > max_nbor_size){
      max_nbor_size = nlist_a_cpy[ii].size();
    }
  }
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  deepmd::convert_nlist(inlist, nlist_a_cpy);
  
  std::vector<double > em(nloc * ndescrpt), em_deriv(nloc * ndescrpt * 3), rij(nloc * nnei * 3);
  std::vector<int> nlist(nloc * nnei);
  std::vector<double > avg(ntypes * ndescrpt, 0);
  std::vector<double > std(ntypes * ndescrpt, 1);
  deepmd::prod_env_mat_a_nvnmd_quantize_cpu(
      &em[0],
      &em_deriv[0],
      &rij[0],
      &nlist[0],
      &posi_cpy[0],
      &atype_cpy[0],
      inlist,
      max_nbor_size,
      &avg[0],
      &std[0],
      nloc,
      nall,
      rc, 
      rc_smth,
      sec_a,
      precs);

  for(int ii = 0; ii < nloc; ++ii){
    for (int jj = 0; jj < nnei; ++jj){
      for (int dd = 0; dd < 4; ++dd){
    	EXPECT_LT(fabs(em[ii*nnei*4 + jj*4 + dd] - 
		       expected_env[ii*nnei*4 + jj*4 + dd]) , 
		  1e-5);
      }
    }    
  }
}


TEST_F(TestEnvMatANvnmd, prod_cpu_equal_cpu)
{
  EXPECT_EQ(nlist_r_cpy.size(), nloc);
  int tot_nnei = 0;
  int max_nbor_size = 0;
  double precs[3] = {8192, 1024, 16}; // NBIT_DATA_FL, NBIT_FEA_X, NBIT_FEA_X_FL
  for(int ii = 0; ii < nlist_a_cpy.size(); ++ii){
    tot_nnei += nlist_a_cpy[ii].size();
    if (nlist_a_cpy[ii].size() > max_nbor_size){
      max_nbor_size = nlist_a_cpy[ii].size();
    }
  }
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_a_cpy);
  std::vector<double > em(nloc * ndescrpt), em_deriv(nloc * ndescrpt * 3), rij(nloc * nnei * 3);
  std::vector<int> nlist(nloc * nnei);
  std::vector<double > avg(ntypes * ndescrpt, 0);
  std::vector<double > std(ntypes * ndescrpt, 1);
  deepmd::prod_env_mat_a_nvnmd_quantize_cpu(
      &em[0],
      &em_deriv[0],
      &rij[0],
      &nlist[0],
      &posi_cpy[0],
      &atype_cpy[0],
      inlist,
      max_nbor_size,
      &avg[0],
      &std[0],
      nloc,
      nall,
      rc, 
      rc_smth,
      sec_a,
      precs);

  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;
  std::vector<double> env_1, env_deriv_1, rij_a_1;
  for(int ii = 0; ii < nloc; ++ii){
    int ret_1 = format_nlist_i_cpu<double>(fmt_nlist_a_1, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);  
    EXPECT_EQ(ret_1, -1);
    deepmd::env_mat_a_nvnmd_quantize_cpu<double>(env_1, env_deriv_1, rij_a_1, posi_cpy, atype_cpy, ii, fmt_nlist_a_1, sec_a, rc_smth, rc, precs);
    EXPECT_EQ(env_1.size(), nnei * 4);
    EXPECT_EQ(env_deriv_1.size(), nnei * 4 * 3);
    EXPECT_EQ(rij_a_1.size(), nnei * 3);
    EXPECT_EQ(fmt_nlist_a_1.size(), nnei);
    EXPECT_EQ(env_1.size() * nloc, em.size());
    EXPECT_EQ(env_deriv_1.size() * nloc, em_deriv.size());
    EXPECT_EQ(rij_a_1.size() * nloc, rij.size());
    EXPECT_EQ(fmt_nlist_a_1.size() * nloc, nlist.size());
    for (unsigned jj = 0; jj < env_1.size(); ++jj){
      EXPECT_LT(fabs(em[ii*nnei*4+jj] - env_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < env_deriv_1.size(); ++jj){
      EXPECT_LT(fabs(em_deriv[ii*nnei*4*3+jj] - env_deriv_1[jj]), 1e-10);      
    }    
    for (unsigned jj = 0; jj < rij_a_1.size(); ++jj){
      EXPECT_LT(fabs(rij[ii*nnei*3+jj] - rij_a_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < fmt_nlist_a_1.size(); ++jj){
      EXPECT_EQ(nlist[ii*nnei+jj], fmt_nlist_a_1[jj]);
    }
  }

  for(int ii = 0; ii < nloc; ++ii){
    for (int jj = 0; jj < nnei; ++jj){
      for (int dd = 0; dd < 4; ++dd){
    	EXPECT_LT(fabs(em[ii*nnei*4 + jj*4 + dd] - 
  		       expected_env[ii*nnei*4 + jj*4 + dd]) , 
  		  1e-5);
      }
    }
  }
}

