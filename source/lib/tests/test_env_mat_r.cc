#include <iostream>
#include <gtest/gtest.h>
#include "env_mat.h"
#include "fmt_nlist.h"
#include "prod_env_mat.h"
#include "neighbor_list.h"

class TestEnvMatR : public ::testing::Test
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
  int nnei = sec_a.back();
  int ndescrpt = nnei * 1;
  std::vector<double > expected_env = {
    0.12206, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.02167, 0.99745, 0.10564, 0.03103, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    1.02167, 0.04135, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.59220, 0.03694, 0.00336, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    0.99745, 0.19078, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.59220, 0.13499, 0.07054, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    0.12206, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.06176, 1.03163, 0.19078, 0.04135, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    1.06176, 0.10564, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.66798, 0.13499, 0.03694, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
    1.03163, 0.03103, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.66798, 0.07054, 0.00336, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
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

TEST_F(TestEnvMatR, orig_cpy)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = false;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
    EXPECT_EQ(ret, -1);
    env_mat_r(env, env_deriv, rij_a, posi_cpy, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
    EXPECT_EQ(env.size(), sec_a[2]);
    EXPECT_EQ(env.size(), env_deriv.size()/3);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      EXPECT_LT(fabs(env[jj] - expected_env[ii*sec_a[2] + jj]) , 1e-5);
    }    
    // for (int jj = 0; jj < sec_a[2]; ++jj){
    //   printf("%7.5f, ", env[jj]);
    // }
    // printf("\n");
  }
}

TEST_F(TestEnvMatR, orig_pbc)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = true;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_fill_a(fmt_nlist_a, fmt_nlist_r, posi, ntypes, atype, region, pbc, ii, nlist_a[ii], nlist_r[ii], rc, sec_a, sec_r);    
    EXPECT_EQ(ret, -1);
    env_mat_r(env, env_deriv, rij_a, posi, ntypes, atype, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
    for (int jj = 0; jj < sec_a[2]; ++jj){
      EXPECT_LT(fabs(env[jj] - expected_env[ii*sec_a[2] + jj]) , 1e-5);
    }    
  }
}


TEST_F(TestEnvMatR, orig_cpy_equal_pbc)
{
  std::vector<int> fmt_nlist_a_0, fmt_nlist_r_0;
  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;
  std::vector<double> env_0, env_deriv_0, rij_a_0;
  std::vector<double> env_1, env_deriv_1, rij_a_1;
  for(int ii = 0; ii < nloc; ++ii){
    int ret_0 = format_nlist_i_cpu<double>(fmt_nlist_a_0, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
    EXPECT_EQ(ret_0, -1);
    env_mat_r(env_0, env_deriv_0, rij_a_0, posi_cpy, ntypes, atype_cpy, region, false, ii, fmt_nlist_a_0, sec_a, rc_smth, rc);
    int ret_1 = format_nlist_i_fill_a(fmt_nlist_a_1, fmt_nlist_r_1, posi, ntypes, atype, region, true, ii, nlist_a[ii], nlist_r[ii], rc, sec_a, sec_r);    
    EXPECT_EQ(ret_1, -1);
    env_mat_r(env_1, env_deriv_1, rij_a_1, posi, ntypes, atype, region, true, ii, fmt_nlist_a_1, sec_a, rc_smth, rc);    
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


TEST_F(TestEnvMatR, orig_cpy_num_deriv)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_0, env_1, env_deriv, env_deriv_tmp, rij_a;
  bool pbc = false;
  double hh = 1e-5;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
    EXPECT_EQ(ret, -1);
    env_mat_r(env, env_deriv, rij_a, posi_cpy, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);    

    for (int jj = 0; jj < sec_a[2]; ++jj){
      int j_idx = fmt_nlist_a[jj];
      if (j_idx < 0) continue;
      for (int dd = 0; dd < 3; ++dd){
	std::vector<double> posi_0 = posi_cpy;
	std::vector<double> posi_1 = posi_cpy;
	posi_0[j_idx*3+dd] -= hh;
	posi_1[j_idx*3+dd] += hh;
	env_mat_r(env_0, env_deriv_tmp, rij_a, posi_0, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	env_mat_r(env_1, env_deriv_tmp, rij_a, posi_1, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	double num_deriv = (env_1[jj] - env_0[jj])/(2.*hh);
	double ana_deriv = -env_deriv[jj*3+dd];
	EXPECT_LT(fabs(num_deriv - ana_deriv), 1e-5);
      }
    }
    // for (int jj = 0; jj < sec_a[2]; ++jj){
    //   printf("%7.5f, %7.5f, %7.5f, %7.5f, ", env[jj*4+0], env[jj*4+1], env[jj*4+2], env[jj*4+3]);
    // }
    // printf("\n");
  }
}


TEST_F(TestEnvMatR, cpu)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_deriv, rij_a;
  bool pbc = false;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);    
    EXPECT_EQ(ret, -1);
    deepmd::env_mat_r_cpu<double>(env, env_deriv, rij_a, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
    for (int jj = 0; jj < sec_a[2]; ++jj){
      EXPECT_LT(fabs(env[jj] - expected_env[ii*sec_a[2] + jj]) , 1e-5);
    }    
  }
}

TEST_F(TestEnvMatR, cpu_equal_orig_cpy)
{
  std::vector<int> fmt_nlist_a_0, fmt_nlist_r_0;
  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;
  std::vector<double> env_0, env_deriv_0, rij_a_0;
  std::vector<double> env_1, env_deriv_1, rij_a_1;
  for(int ii = 0; ii < nloc; ++ii){
    int ret_0 = format_nlist_i_cpu<double>(fmt_nlist_a_0, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
    EXPECT_EQ(ret_0, -1);
    env_mat_r(env_0, env_deriv_0, rij_a_0, posi_cpy, ntypes, atype_cpy, region, false, ii, fmt_nlist_a_0, sec_a, rc_smth, rc);

    int ret_1 = format_nlist_i_cpu<double>(fmt_nlist_a_1, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);  
    EXPECT_EQ(ret_1, -1);
    deepmd::env_mat_r_cpu<double>(env_1, env_deriv_1, rij_a_1, posi_cpy, atype_cpy, ii, fmt_nlist_a_1, sec_a, rc_smth, rc);

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

TEST_F(TestEnvMatR, cpu_num_deriv)
{
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  std::vector<double> env, env_0, env_1, env_deriv, env_deriv_tmp, rij_a;
  bool pbc = false;
  double hh = 1e-5;
  for(int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);    
    EXPECT_EQ(ret, -1);
    deepmd::env_mat_r_cpu<double>(env, env_deriv, rij_a, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);    

    for (int jj = 0; jj < sec_a[2]; ++jj){
      int j_idx = fmt_nlist_a[jj];
      if (j_idx < 0) continue;
	for (int dd = 0; dd < 3; ++dd){
	  std::vector<double> posi_0 = posi_cpy;
	  std::vector<double> posi_1 = posi_cpy;
	  posi_0[j_idx*3+dd] -= hh;
	  posi_1[j_idx*3+dd] += hh;
	  env_mat_r(env_0, env_deriv_tmp, rij_a, posi_0, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	  env_mat_r(env_1, env_deriv_tmp, rij_a, posi_1, ntypes, atype_cpy, region, pbc, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	  double num_deriv = (env_1[jj] - env_0[jj])/(2.*hh);
	  double ana_deriv = -env_deriv[jj*3+dd];
	  EXPECT_LT(fabs(num_deriv - ana_deriv), 1e-5);
	}
    }
    // for (int jj = 0; jj < sec_a[2]; ++jj){
    //   printf("%7.5f, %7.5f, %7.5f, %7.5f, ", env[jj*4+0], env[jj*4+1], env[jj*4+2], env[jj*4+3]);
    // }
    // printf("\n");
  }
}

TEST_F(TestEnvMatR, prod_cpu)
{
  EXPECT_EQ(nlist_r_cpy.size(), nloc);
  int tot_nnei = 0;
  int max_nbor_size = 0;
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
  deepmd::prod_env_mat_r_cpu(
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
      sec_a);

  for(int ii = 0; ii < nloc; ++ii){
    for (int jj = 0; jj < nnei; ++jj){
      for (int dd = 0; dd < 1; ++dd){
    	EXPECT_LT(fabs(em[ii*nnei*1 + jj*1 + dd] - 
		       expected_env[ii*nnei*1 + jj*1 + dd]) , 
		  1e-5);
      }
    }    
  }
}


TEST_F(TestEnvMatR, prod_cpu_equal_cpu)
{
  EXPECT_EQ(nlist_r_cpy.size(), nloc);
  int tot_nnei = 0;
  int max_nbor_size = 0;
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
  deepmd::prod_env_mat_r_cpu(
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
      sec_a);

  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;
  std::vector<double> env_1, env_deriv_1, rij_a_1;
  for(int ii = 0; ii < nloc; ++ii){
    int ret_1 = format_nlist_i_cpu<double>(fmt_nlist_a_1, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
    EXPECT_EQ(ret_1, -1);
    deepmd::env_mat_r_cpu<double>(env_1, env_deriv_1, rij_a_1, posi_cpy, atype_cpy, ii, fmt_nlist_a_1, sec_a, rc_smth, rc);
    EXPECT_EQ(env_1.size(), nnei * 1);
    EXPECT_EQ(env_deriv_1.size(), nnei * 1 * 3);
    EXPECT_EQ(rij_a_1.size(), nnei * 3);
    EXPECT_EQ(fmt_nlist_a_1.size(), nnei);
    EXPECT_EQ(env_1.size() * nloc, em.size());
    EXPECT_EQ(env_deriv_1.size() * nloc, em_deriv.size());
    EXPECT_EQ(rij_a_1.size() * nloc, rij.size());
    EXPECT_EQ(fmt_nlist_a_1.size() * nloc, nlist.size());
    for (unsigned jj = 0; jj < env_1.size(); ++jj){
      EXPECT_LT(fabs(em[ii*nnei*1+jj] - env_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < env_deriv_1.size(); ++jj){
      EXPECT_LT(fabs(em_deriv[ii*nnei*1*3+jj] - env_deriv_1[jj]), 1e-10);      
    }    
    for (unsigned jj = 0; jj < rij_a_1.size(); ++jj){
      EXPECT_LT(fabs(rij[ii*nnei*3+jj] - rij_a_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < fmt_nlist_a_1.size(); ++jj){
      EXPECT_EQ(nlist[ii*nnei+jj], fmt_nlist_a_1[jj]);
    }
  }
}


#if GOOGLE_CUDA
TEST_F(TestEnvMatR, prod_gpu_cuda)
{
  EXPECT_EQ(nlist_r_cpy.size(), nloc);
  int tot_nnei = 0;
  int max_nbor_size = 0;
  for(int ii = 0; ii < nlist_a_cpy.size(); ++ii){
    tot_nnei += nlist_a_cpy[ii].size();
    if (nlist_a_cpy[ii].size() > max_nbor_size){
      max_nbor_size = nlist_a_cpy[ii].size();
    }
  }
  assert(max_nbor_size <= GPU_MAX_NBOR_SIZE);
  if (max_nbor_size <= 1024) {
    max_nbor_size = 1024;
  }
  else if (max_nbor_size <= 2048) {
    max_nbor_size = 2048;
  }
  else {
    max_nbor_size = 4096;
  }
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]), gpu_inlist;
  convert_nlist(inlist, nlist_a_cpy);
  std::vector<double > em(nloc * ndescrpt, 0.0), em_deriv(nloc * ndescrpt * 3, 0.0), rij(nloc * nnei * 3, 0.0);
  std::vector<int> nlist(nloc * nnei, 0);
  std::vector<double > avg(ntypes * ndescrpt, 0);
  std::vector<double > std(ntypes * ndescrpt, 1);

  double * em_dev = NULL, * em_deriv_dev = NULL, * rij_dev = NULL;
  double * posi_cpy_dev = NULL, * avg_dev = NULL, * std_dev = NULL;
  int * atype_cpy_dev = NULL, * nlist_dev = NULL, * array_int_dev = NULL, * memory_dev = NULL;
  int_64 * array_longlong_dev = NULL;
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::malloc_device_memory_sync(em_deriv_dev, em_deriv);
  deepmd::malloc_device_memory_sync(rij_dev, rij);
  deepmd::malloc_device_memory_sync(posi_cpy_dev, posi_cpy);
  deepmd::malloc_device_memory_sync(avg_dev, avg);
  deepmd::malloc_device_memory_sync(std_dev, std);

  deepmd::malloc_device_memory_sync(atype_cpy_dev, atype_cpy);
  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory(array_int_dev, sec_a.size() + nloc * sec_a.size() + nloc);
  deepmd::malloc_device_memory(array_longlong_dev, nloc * GPU_MAX_NBOR_SIZE * 2);
  deepmd::malloc_device_memory(memory_dev, nloc * max_nbor_size);
  deepmd::convert_nlist_gpu_cuda(gpu_inlist, inlist, memory_dev, max_nbor_size);

  deepmd::prod_env_mat_r_gpu_cuda(    
      em_dev, 
      em_deriv_dev, 
      rij_dev, 
      nlist_dev, 
      posi_cpy_dev, 
      atype_cpy_dev, 
      gpu_inlist,
      array_int_dev, 
      array_longlong_dev,
      max_nbor_size,
      avg_dev, 
      std_dev, 
      nloc,
      nall,
      rc, 
      rc_smth,
      sec_a);
  deepmd::memcpy_device_to_host(em_dev, em);
  deepmd::delete_device_memory(em_dev);
  deepmd::delete_device_memory(em_deriv_dev);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(posi_cpy_dev);
  deepmd::delete_device_memory(atype_cpy_dev);
  deepmd::delete_device_memory(array_int_dev);
  deepmd::delete_device_memory(array_longlong_dev);
  deepmd::delete_device_memory(avg_dev);
  deepmd::delete_device_memory(std_dev);
  deepmd::delete_device_memory(memory_dev);
  deepmd::free_nlist_gpu_cuda(gpu_inlist);

  for(int ii = 0; ii < nloc; ++ii){
    for (int jj = 0; jj < nnei; ++jj){
      for (int dd = 0; dd < 1; ++dd){
    	EXPECT_LT(fabs(em[ii*nnei*1 + jj*1 + dd] - 
		       expected_env[ii*nnei*1 + jj*1 + dd]) , 
		  1e-5);
      }
    }    
  }
}

TEST_F(TestEnvMatR, prod_gpu_cuda_equal_cpu)
{
  EXPECT_EQ(nlist_r_cpy.size(), nloc);
  int tot_nnei = 0;
  int max_nbor_size = 0;
  for(int ii = 0; ii < nlist_a_cpy.size(); ++ii){
    tot_nnei += nlist_a_cpy[ii].size();
    if (nlist_a_cpy[ii].size() > max_nbor_size){
      max_nbor_size = nlist_a_cpy[ii].size();
    }
  }
  assert(max_nbor_size <= GPU_MAX_NBOR_SIZE);
  if (max_nbor_size <= 1024) {
    max_nbor_size = 1024;
  }
  else if (max_nbor_size <= 2048) {
    max_nbor_size = 2048;
  }
  else {
    max_nbor_size = 4096;
  }
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]), gpu_inlist;
  convert_nlist(inlist, nlist_a_cpy);
  std::vector<double > em(nloc * ndescrpt, 0.0), em_deriv(nloc * ndescrpt * 3, 0.0), rij(nloc * nnei * 3, 0.0);
  std::vector<int> nlist(nloc * nnei, 0);
  std::vector<double > avg(ntypes * ndescrpt, 0);
  std::vector<double > std(ntypes * ndescrpt, 1);

  double * em_dev = NULL, * em_deriv_dev = NULL, * rij_dev = NULL;
  double * posi_cpy_dev = NULL, * avg_dev = NULL, * std_dev = NULL;
  int * atype_cpy_dev = NULL, * nlist_dev = NULL, * array_int_dev = NULL, * memory_dev = NULL;
  int_64 * array_longlong_dev = NULL;
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::malloc_device_memory_sync(em_deriv_dev, em_deriv);
  deepmd::malloc_device_memory_sync(rij_dev, rij);
  deepmd::malloc_device_memory_sync(posi_cpy_dev, posi_cpy);
  deepmd::malloc_device_memory_sync(avg_dev, avg);
  deepmd::malloc_device_memory_sync(std_dev, std);

  deepmd::malloc_device_memory_sync(atype_cpy_dev, atype_cpy);
  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory(array_int_dev, sec_a.size() + nloc * sec_a.size() + nloc);
  deepmd::malloc_device_memory(array_longlong_dev, nloc * GPU_MAX_NBOR_SIZE * 2);
  deepmd::malloc_device_memory(memory_dev, nloc * max_nbor_size);
  deepmd::convert_nlist_gpu_cuda(gpu_inlist, inlist, memory_dev, max_nbor_size);

  deepmd::prod_env_mat_r_gpu_cuda(    
      em_dev, 
      em_deriv_dev, 
      rij_dev, 
      nlist_dev, 
      posi_cpy_dev, 
      atype_cpy_dev, 
      gpu_inlist,
      array_int_dev, 
      array_longlong_dev,
      max_nbor_size,
      avg_dev, 
      std_dev, 
      nloc,
      nall,
      rc, 
      rc_smth,
      sec_a);
  deepmd::memcpy_device_to_host(em_dev, em);
  deepmd::memcpy_device_to_host(em_deriv_dev, em_deriv);
  deepmd::memcpy_device_to_host(rij_dev, rij);
  deepmd::memcpy_device_to_host(nlist_dev, nlist);
  deepmd::delete_device_memory(em_dev);
  deepmd::delete_device_memory(em_deriv_dev);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(posi_cpy_dev);
  deepmd::delete_device_memory(atype_cpy_dev);
  deepmd::delete_device_memory(array_int_dev);
  deepmd::delete_device_memory(array_longlong_dev);
  deepmd::delete_device_memory(avg_dev);
  deepmd::delete_device_memory(std_dev);
  deepmd::delete_device_memory(memory_dev);
  deepmd::free_nlist_gpu_cuda(gpu_inlist);

  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;
  std::vector<double> env_1, env_deriv_1, rij_a_1;
  for(int ii = 0; ii < nloc; ++ii){
    int ret_1 = format_nlist_i_cpu<double>(fmt_nlist_a_1, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);  
    EXPECT_EQ(ret_1, -1);
    deepmd::env_mat_r_cpu<double>(env_1, env_deriv_1, rij_a_1, posi_cpy, atype_cpy, ii, fmt_nlist_a_1, sec_a, rc_smth, rc);
    EXPECT_EQ(env_1.size(), nnei * 1);
    EXPECT_EQ(env_deriv_1.size(), nnei * 1 * 3);
    EXPECT_EQ(rij_a_1.size(), nnei * 3);
    EXPECT_EQ(fmt_nlist_a_1.size(), nnei);
    EXPECT_EQ(env_1.size() * nloc, em.size());
    EXPECT_EQ(env_deriv_1.size() * nloc, em_deriv.size());
    EXPECT_EQ(rij_a_1.size() * nloc, rij.size());
    EXPECT_EQ(fmt_nlist_a_1.size() * nloc, nlist.size());
    for (unsigned jj = 0; jj < env_1.size(); ++jj){
      EXPECT_LT(fabs(em[ii*nnei*1+jj] - env_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < env_deriv_1.size(); ++jj){
      EXPECT_LT(fabs(em_deriv[ii*nnei*1*3+jj] - env_deriv_1[jj]), 1e-10);      
    }    
    for (unsigned jj = 0; jj < rij_a_1.size(); ++jj){
      EXPECT_LT(fabs(rij[ii*nnei*3+jj] - rij_a_1[jj]), 1e-10);
    }
    for (unsigned jj = 0; jj < fmt_nlist_a_1.size(); ++jj){
      EXPECT_EQ(nlist[ii*nnei+jj], fmt_nlist_a_1[jj]);
    }
  }
}
#endif //GOOGLE_CUDA
