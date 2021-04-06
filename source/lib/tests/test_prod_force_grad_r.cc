#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "neighbor_list.h"
#include "prod_force_grad.h"
#include "device.h"

class TestProdForceGradR : public ::testing::Test
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
  std::vector<double> grad;
  std::vector<double> env, env_deriv, rij_a;
  std::vector<int> nlist;
  std::vector<int> fmt_nlist_a;
  std::vector<double > expected_grad_net = {
    -0.12141, -1.33062,  0.12948,  0.50970,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.04188,  0.75283,  9.26593,  8.54987, -0.54546, -0.01575,  0.00681,  0.02755,  0.00000,  0.00000, -0.40861,  0.12805, -0.45057, 15.54539, -1.52411,  0.04701,  0.05002,  0.04668,  0.00000,  0.00000, -0.12141, -1.21099,  0.11750,  0.46323,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.16227, -1.14667, 14.61804,  1.94488,  1.18285, -0.04089, -0.01845, -0.00874,  0.00000,  0.00000, -0.58330,  1.13439,  5.18696, 13.10426,  0.49773,  0.01712,  0.00239, -0.01893,  0.00000,  0.00000,
  };  
  
  void SetUp() override {
    double box[] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
    region.reinitBox(box);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc, region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nnei = sec_a.back();
    ndescrpt = nnei * 1;
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
    grad.resize(nloc * 3);
    for (int ii = 0; ii < nloc * 3; ++ii){
      grad[ii] = 10 - ii * 0.1;
    }
  }
  void TearDown() override {
  }
};

TEST_F(TestProdForceGradR, cpu)
{
  std::vector<double> grad_net(nloc * ndescrpt);
  deepmd::prod_force_grad_r_cpu<double>(&grad_net[0], &grad[0], &env_deriv[0], &nlist[0], nloc, nnei);
  EXPECT_EQ(grad_net.size(), nloc * ndescrpt);
  EXPECT_EQ(grad_net.size(), expected_grad_net.size());
  for (int jj = 0; jj < grad_net.size(); ++jj){
    EXPECT_LT(fabs(grad_net[jj] - expected_grad_net[jj]) , 1e-5);
  }  
  // for (int jj = 0; jj < nloc * ndescrpt; ++jj){
  //   printf("%8.5f, ", grad_net[jj]);
  // }
  // printf("\n");
}

#if GOOGLE_CUDA
TEST_F(TestProdForceGradR, gpu)
{
  std::vector<double> grad_net(nloc * ndescrpt);
  int * nlist_dev = NULL;
  double * grad_net_dev = NULL, * grad_dev = NULL, * env_deriv_dev = NULL;

  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory_sync(grad_dev, grad);
  deepmd::malloc_device_memory_sync(env_deriv_dev, env_deriv);
  deepmd::malloc_device_memory(grad_net_dev, nloc * ndescrpt);
  deepmd::prod_force_grad_r_gpu_cuda<double>(grad_net_dev, grad_dev, env_deriv_dev, nlist_dev, nloc, nnei);
  deepmd::memcpy_device_to_host(grad_net_dev, grad_net);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(grad_dev);
  deepmd::delete_device_memory(env_deriv_dev);
  deepmd::delete_device_memory(grad_net_dev);

  EXPECT_EQ(grad_net.size(), nloc * ndescrpt);
  EXPECT_EQ(grad_net.size(), expected_grad_net.size());
  for (int jj = 0; jj < grad_net.size(); ++jj){
    EXPECT_LT(fabs(grad_net[jj] - expected_grad_net[jj]) , 1e-5);
  }  
  // for (int jj = 0; jj < nloc * ndescrpt; ++jj){
  //   printf("%8.5f, ", grad_net[jj]);
  // }
  // printf("\n");
}
#endif // GOOGLE_CUDA
