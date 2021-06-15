#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "neighbor_list.h"
#include "prod_force.h"
#include "device.h"

class TestProdForceA : public ::testing::Test
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
  int nloc, nall, nnei, ndescrpt;
  double rc = 6;
  double rc_smth = 0.8;
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 5, 10};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  std::vector<double> net_deriv, in_deriv;
  std::vector<double> env, env_deriv, rij_a;
  std::vector<int> nlist;
  std::vector<int> fmt_nlist_a;
  std::vector<double > expected_force = {
    9.44498, -13.86254, 10.52884, -19.42688,  8.09273, 19.64478,  4.81771, 11.39255, 12.38830, -16.65832,  6.65153, -10.15585,  1.16660, -14.43259, 22.97076, 22.86479,  7.42726, -11.41943, -7.67893, -7.23287, -11.33442, -4.51184, -3.80588, -2.44935,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  1.16217,  6.16192, -28.79094,  3.81076, -0.01986, -1.01629,  3.65869, -0.49195, -0.07437,  1.35028,  0.11969, -0.29201,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
  };  
  
  void SetUp() override {
    double box[] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
    region.reinitBox(box);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc, region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nnei = sec_a.back();
    ndescrpt = nnei * 4;
    nat_stt.resize(3);
    ext_stt.resize(3);
    ext_end.resize(3);
    for (int dd = 0; dd < 3; ++dd){
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    build_nlist(nlist_a_cpy, nlist_r_cpy, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
    nlist.resize(nloc * nnei);
    env.resize(nloc * ndescrpt);
    env_deriv.resize(nloc * ndescrpt * 3);
    rij_a.resize(nloc * nnei * 3);
    for(int ii = 0; ii < nloc; ++ii){      
      // format nlist and record
      format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
      for (int jj = 0; jj < nnei; ++jj){
	nlist[ii*nnei + jj] = fmt_nlist_a[jj];
      }
      std::vector<double > t_env, t_env_deriv, t_rij_a;
      // compute env_mat and its deriv, record
      deepmd::env_mat_a_cpu<double>(t_env, t_env_deriv, t_rij_a, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
      for (int jj = 0; jj < ndescrpt; ++jj){
	env[ii*ndescrpt+jj] = t_env[jj];
	for (int dd = 0; dd < 3; ++dd){
	  env_deriv[ii*ndescrpt*3+jj*3+dd] = t_env_deriv[jj*3+dd];
	}
      }
    }
    net_deriv.resize(nloc * ndescrpt);
    for (int ii = 0; ii < nloc * ndescrpt; ++ii){
      net_deriv[ii] = 10 - ii * 0.01;
    }
  }
  void TearDown() override {
  }
};

TEST_F(TestProdForceA, cpu)
{
  std::vector<double> force(nall * 3);
  int n_a_sel = nnei;
  deepmd::prod_force_a_cpu<double> (&force[0], &net_deriv[0], &env_deriv[0], &nlist[0], nloc, nall, nnei);
  EXPECT_EQ(force.size(), nall * 3);
  EXPECT_EQ(force.size(), expected_force.size());
  for (int jj = 0; jj < force.size(); ++jj){
    EXPECT_LT(fabs(force[jj] - expected_force[jj]) , 1e-5);
  }  
  // for (int jj = 0; jj < nall * 3; ++jj){
  //   printf("%8.5f, ", force[jj]);
  // }
  // printf("\n");
}

#if GOOGLE_CUDA
TEST_F(TestProdForceA, gpu_cuda)
{
  std::vector<double> force(nall * 3, 0.0);
  int n_a_sel = nnei;

  int * nlist_dev = NULL;
  double * force_dev = NULL, * net_deriv_dev = NULL, * env_deriv_dev = NULL;

  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory_sync(force_dev, force);
  deepmd::malloc_device_memory_sync(net_deriv_dev, net_deriv);
  deepmd::malloc_device_memory_sync(env_deriv_dev, env_deriv);

  deepmd::prod_force_a_gpu_cuda<double> (force_dev, net_deriv_dev, env_deriv_dev, nlist_dev, nloc, nall, nnei);
  
  deepmd::memcpy_device_to_host(force_dev, force);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(force_dev);
  deepmd::delete_device_memory(net_deriv_dev);
  deepmd::delete_device_memory(env_deriv_dev);

  EXPECT_EQ(force.size(), nall * 3);
  EXPECT_EQ(force.size(), expected_force.size());
  for (int jj = 0; jj < force.size(); ++jj){
    EXPECT_LT(fabs(force[jj] - expected_force[jj]) , 1e-5);
  }
}
#endif // GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
TEST_F(TestProdForceA, gpu_rocm)
{
  std::vector<double> force(nall * 3, 0.0);
  int n_a_sel = nnei;

  int * nlist_dev = NULL;
  double * force_dev = NULL, * net_deriv_dev = NULL, * env_deriv_dev = NULL;

  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory_sync(force_dev, force);
  deepmd::malloc_device_memory_sync(net_deriv_dev, net_deriv);
  deepmd::malloc_device_memory_sync(env_deriv_dev, env_deriv);

  deepmd::prod_force_a_gpu_rocm<double> (force_dev, net_deriv_dev, env_deriv_dev, nlist_dev, nloc, nall, nnei);
  
  deepmd::memcpy_device_to_host(force_dev, force);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(force_dev);
  deepmd::delete_device_memory(net_deriv_dev);
  deepmd::delete_device_memory(env_deriv_dev);

  EXPECT_EQ(force.size(), nall * 3);
  EXPECT_EQ(force.size(), expected_force.size());
  for (int jj = 0; jj < force.size(); ++jj){
    EXPECT_LT(fabs(force[jj] - expected_force[jj]) , 1e-5);
  }
}
#endif // TENSORFLOW_USE_ROCM
