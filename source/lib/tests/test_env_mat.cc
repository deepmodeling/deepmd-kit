#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "NeighborList.h"

class TestEnvMat : public ::testing::Test
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
  int ntypes = 2;  
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
  std::vector<double > expected_env = {
    0.12206, 0.12047, 0.01502, -0.01263, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.02167, -0.77271, 0.32370, 0.58475, 0.99745, 0.41810, 0.75655, -0.49773, 0.10564, 0.10495, -0.00143, 0.01198, 0.03103, 0.03041, 0.00452, -0.00425, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    1.02167, 0.77271, -0.32370, -0.58475, 0.04135, 0.04039, 0.00123, -0.00880, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.59220, 0.42028, 0.16304, -0.38405, 0.03694, 0.03680, -0.00300, -0.00117, 0.00336, 0.00327, 0.00022, -0.00074, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    0.99745, -0.41810, -0.75655, 0.49773, 0.19078, 0.18961, -0.01951, 0.00793, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.59220, -0.42028, -0.16304, 0.38405, 0.13499, 0.12636, -0.03140, 0.03566, 0.07054, 0.07049, -0.00175, -0.00210, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    0.12206, -0.12047, -0.01502, 0.01263, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.06176, 0.16913, -0.55250, 0.89077, 1.03163, 0.96880, 0.23422, -0.26615, 0.19078, -0.18961, 0.01951, -0.00793, 0.04135, -0.04039, -0.00123, 0.00880, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    1.06176, -0.16913, 0.55250, -0.89077, 0.10564, -0.10495, 0.00143, -0.01198, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.66798, 0.34516, 0.32245, -0.47232, 0.13499, -0.12636, 0.03140, -0.03566, 0.03694, -0.03680, 0.00300, 0.00117, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    1.03163, -0.96880, -0.23422, 0.26615, 0.03103, -0.03041, -0.00452, 0.00425, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.66798, -0.34516, -0.32245, 0.47232, 0.07054, -0.07049, 0.00175, 0.00210, 0.00336, -0.00327, -0.00022, 0.00074, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
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

TEST_F(TestEnvMat, orig_cpy)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = false;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_fill_a(fmt_nlist_a, fmt_nlist_r, posi_cpy, ntypes, atype_cpy, region, pbc, ii, nlist_a_cpy[ii], nlist_r_cpy[ii], rc, sec_a, sec_r);    
    EXPECT_EQ(ret, -1);
    env_mat_a(env, env_deriv, rij_a, posi_cpy, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
    for (int jj = 0; jj < sec_a[2]; ++jj){
      for (int dd = 0; dd < 4; ++dd){
    	EXPECT_LT(fabs(env[jj*4+dd] - expected_env[ii*sec_a[2]*4 + jj*4 + dd]) , 1e-5);
      }
    }    
    // for (int jj = 0; jj < sec_a[2]; ++jj){
    //   printf("%7.5f, %7.5f, %7.5f, %7.5f, ", env[jj*4+0], env[jj*4+1], env[jj*4+2], env[jj*4+3]);
    // }
    // printf("\n");
  }
}

TEST_F(TestEnvMat, orig_pbc)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = true;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_fill_a(fmt_nlist_a, fmt_nlist_r, posi, ntypes, atype, region, pbc, ii, nlist_a[ii], nlist_r[ii], rc, sec_a, sec_r);    
    EXPECT_EQ(ret, -1);
    env_mat_a(env, env_deriv, rij_a, posi, ntypes, atype, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
    for (int jj = 0; jj < sec_a[2]; ++jj){
      for (int dd = 0; dd < 4; ++dd){
    	EXPECT_LT(fabs(env[jj*4+dd] - expected_env[ii*sec_a[2]*4 + jj*4 + dd]) , 1e-5);
      }
    }    
  }
}


TEST_F(TestEnvMat, orig_cpy_equal_pbc)
{
  std::vector<int> fmt_nlist_a_0, fmt_nlist_r_0;
  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;
  std::vector<double> env_0, env_deriv_0, rij_a_0;
  std::vector<double> env_1, env_deriv_1, rij_a_1;
  for(int ii = 0; ii < nloc; ++ii){
    int ret_0 = format_nlist_fill_a(fmt_nlist_a_0, fmt_nlist_r_0, posi_cpy, ntypes, atype_cpy, region, false, ii, nlist_a_cpy[ii], nlist_r_cpy[ii], rc, sec_a, sec_r);
    EXPECT_EQ(ret_0, -1);
    env_mat_a(env_0, env_deriv_0, rij_a_0, posi_cpy, ntypes, atype_cpy, region, false, ii, fmt_nlist_a_0, sec_a, rc_smth, rc);
    int ret_1 = format_nlist_fill_a(fmt_nlist_a_1, fmt_nlist_r_1, posi, ntypes, atype, region, true, ii, nlist_a[ii], nlist_r[ii], rc, sec_a, sec_r);    
    EXPECT_EQ(ret_1, -1);
    env_mat_a(env_1, env_deriv_1, rij_a_1, posi, ntypes, atype, region, true, ii, fmt_nlist_a_1, sec_a, rc_smth, rc);    
    EXPECT_EQ(env_0.size(), env_1.size());
    EXPECT_EQ(env_deriv_0.size(), env_deriv_1.size());
    EXPECT_EQ(rij_a_0.size(), rij_a_1.size());
    for (unsigned jj = 0; jj < env_0.size(); ++jj){
      EXPECT_LT(fabs(env_0[jj] - env_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < env_deriv_0.size(); ++jj){
      EXPECT_LT(fabs(env_deriv_0[jj] - env_deriv_1[jj]), 1e-10);      
    }    
    for (unsigned jj = 0; jj < rij_a_0.size(); ++jj){
      EXPECT_LT(fabs(rij_a_0[jj] - rij_a_1[jj]), 1e-10);
    }
  }
}


TEST_F(TestEnvMat, orig_cpy_num_deriv)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_0, env_1, env_deriv, env_deriv_tmp, rij_a;
  bool pbc = false;
  double hh = 1e-5;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_fill_a(fmt_nlist_a, fmt_nlist_r, posi_cpy, ntypes, atype_cpy, region, pbc, ii, nlist_a_cpy[ii], nlist_r_cpy[ii], rc, sec_a, sec_r);    
    EXPECT_EQ(ret, -1);
    env_mat_a(env, env_deriv, rij_a, posi_cpy, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);    

    for (int jj = 0; jj < sec_a[2]; ++jj){
      int j_idx = fmt_nlist_a[jj];
      if (j_idx < 0) continue;
      for (int kk = 0; kk < 4; ++kk){
	for (int dd = 0; dd < 3; ++dd){
	  std::vector<double> posi_0 = posi_cpy;
	  std::vector<double> posi_1 = posi_cpy;
	  posi_0[j_idx*3+dd] -= hh;
	  posi_1[j_idx*3+dd] += hh;
	  env_mat_a(env_0, env_deriv_tmp, rij_a, posi_0, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	  env_mat_a(env_1, env_deriv_tmp, rij_a, posi_1, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	  double num_deriv = (env_1[jj*4+kk] - env_0[jj*4+kk])/(2.*hh);
	  double ana_deriv = -env_deriv[jj*12+kk*3+dd];
	  EXPECT_LT(fabs(num_deriv - ana_deriv), 1e-5);
	}
      }
    }
    // for (int jj = 0; jj < sec_a[2]; ++jj){
    //   printf("%7.5f, %7.5f, %7.5f, %7.5f, ", env[jj*4+0], env[jj*4+1], env[jj*4+2], env[jj*4+3]);
    // }
    // printf("\n");
  }
}


TEST_F(TestEnvMat, cpu)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = false;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_fill_a(fmt_nlist_a, fmt_nlist_r, posi_cpy, ntypes, atype_cpy, region, pbc, ii, nlist_a_cpy[ii], nlist_r_cpy[ii], rc, sec_a, sec_r);    
    EXPECT_EQ(ret, -1);
    env_mat_a_cpu<double>(env, env_deriv, rij_a, posi_cpy, ntypes, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
    for (int jj = 0; jj < sec_a[2]; ++jj){
      for (int dd = 0; dd < 4; ++dd){
    	EXPECT_LT(fabs(env[jj*4+dd] - expected_env[ii*sec_a[2]*4 + jj*4 + dd]) , 1e-5);
      }
    }    
  }
}

TEST_F(TestEnvMat, cpu_equal_orig_cpy)
{
  std::vector<int> fmt_nlist_a_0, fmt_nlist_r_0;
  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;
  std::vector<double> env_0, env_deriv_0, rij_a_0;
  std::vector<double> env_1, env_deriv_1, rij_a_1;
  for(int ii = 0; ii < nloc; ++ii){
    int ret_0 = format_nlist_fill_a(fmt_nlist_a_0, fmt_nlist_r_0, posi_cpy, ntypes, atype_cpy, region, false, ii, nlist_a_cpy[ii], nlist_r_cpy[ii], rc, sec_a, sec_r);
    EXPECT_EQ(ret_0, -1);
    env_mat_a(env_0, env_deriv_0, rij_a_0, posi_cpy, ntypes, atype_cpy, region, false, ii, fmt_nlist_a_0, sec_a, rc_smth, rc);

    int ret_1 = format_nlist_fill_a(fmt_nlist_a_1, fmt_nlist_r_1, posi_cpy, ntypes, atype_cpy, region, false, ii, nlist_a_cpy[ii], nlist_r_cpy[ii], rc, sec_a, sec_r);  
    EXPECT_EQ(ret_1, -1);
    env_mat_a_cpu<double>(env_1, env_deriv_1, rij_a_1, posi_cpy, ntypes, atype_cpy, ii, fmt_nlist_a_1, sec_a, rc_smth, rc);

    EXPECT_EQ(env_0.size(), env_1.size());
    EXPECT_EQ(env_deriv_0.size(), env_deriv_1.size());
    EXPECT_EQ(rij_a_0.size(), rij_a_1.size());
    for (unsigned jj = 0; jj < env_0.size(); ++jj){
      EXPECT_LT(fabs(env_0[jj] - env_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < env_deriv_0.size(); ++jj){
      EXPECT_LT(fabs(env_deriv_0[jj] - env_deriv_1[jj]), 1e-10);      
    }    
    for (unsigned jj = 0; jj < rij_a_0.size(); ++jj){
      EXPECT_LT(fabs(rij_a_0[jj] - rij_a_1[jj]), 1e-10);
    }
  }
}

TEST_F(TestEnvMat, cpu_num_deriv)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_0, env_1, env_deriv, env_deriv_tmp, rij_a;
  bool pbc = false;
  double hh = 1e-5;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_fill_a(fmt_nlist_a, fmt_nlist_r, posi_cpy, ntypes, atype_cpy, region, pbc, ii, nlist_a_cpy[ii], nlist_r_cpy[ii], rc, sec_a, sec_r);    
    EXPECT_EQ(ret, -1);
    env_mat_a_cpu<double>(env, env_deriv, rij_a, posi_cpy, ntypes, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);    

    for (int jj = 0; jj < sec_a[2]; ++jj){
      int j_idx = fmt_nlist_a[jj];
      if (j_idx < 0) continue;
      for (int kk = 0; kk < 4; ++kk){
	for (int dd = 0; dd < 3; ++dd){
	  std::vector<double> posi_0 = posi_cpy;
	  std::vector<double> posi_1 = posi_cpy;
	  posi_0[j_idx*3+dd] -= hh;
	  posi_1[j_idx*3+dd] += hh;
	  env_mat_a(env_0, env_deriv_tmp, rij_a, posi_0, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	  env_mat_a(env_1, env_deriv_tmp, rij_a, posi_1, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	  double num_deriv = (env_1[jj*4+kk] - env_0[jj*4+kk])/(2.*hh);
	  double ana_deriv = -env_deriv[jj*12+kk*3+dd];
	  EXPECT_LT(fabs(num_deriv - ana_deriv), 1e-5);
	}
      }
    }
    // for (int jj = 0; jj < sec_a[2]; ++jj){
    //   printf("%7.5f, %7.5f, %7.5f, %7.5f, ", env[jj*4+0], env[jj*4+1], env[jj*4+2], env[jj*4+3]);
    // }
    // printf("\n");
  }
}
