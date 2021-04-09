#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "neighbor_list.h"
#include "prod_force_grad.h"
#include "device.h"

class TestProdForceGradA : public ::testing::Test
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
    -0.12141, -0.11963,  0.01198,  0.04647,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.04188,  0.37642,  0.28680,  0.26547, -0.40861,  0.25610, -0.02009,  1.00344, -0.16166, -0.16355,  0.03691,  0.01165, -0.08770, -0.08561, -0.00398,  0.02366,  0.00000,  0.00000,  0.00000,  0.00000, -0.04188, -0.37642, -0.28680, -0.26547, -0.03357, -0.03151,  0.00454,  0.01377,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.04304,  0.05219,  0.08677,  0.16032, -0.05232, -0.05123,  0.01227,  0.00935, -0.01420, -0.01366, -0.00022,  0.00404,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.40861, -0.25610,  0.02009, -1.00344, -0.04863, -0.04701,  0.02501,  0.01556,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.04304, -0.05219, -0.08677, -0.16032, -0.08249, -0.07502,  0.04767, -0.00448, -0.08260, -0.08165,  0.01821,  0.01869,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.12141,  0.11963, -0.01198, -0.04647,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.16227,  0.28667,  0.50683,  0.06651, -0.58330, -0.45376,  0.37464,  0.93891, -0.04863,  0.04701, -0.02501, -0.01556, -0.03357,  0.03151, -0.00454, -0.01377,  0.00000,  0.00000,  0.00000,  0.00000, -0.16227, -0.28667, -0.50683, -0.06651, -0.16166,  0.16355, -0.03691, -0.01165,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.04418,  0.09284,  0.09569,  0.19565, -0.08249,  0.07502, -0.04767,  0.00448, -0.05232,  0.05123, -0.01227, -0.00935,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.58330,  0.45376, -0.37464, -0.93891, -0.08770,  0.08561,  0.00398, -0.02366,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -0.04418, -0.09284, -0.09569, -0.19565, -0.08260,  0.08165, -0.01821, -0.01869, -0.01420,  0.01366,  0.00022, -0.00404,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
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
    grad.resize(nloc * 3);
    for (int ii = 0; ii < nloc * 3; ++ii){
      grad[ii] = 10 - ii * 0.1;
    }
  }
  void TearDown() override {
  }
};

TEST_F(TestProdForceGradA, cpu)
{
  std::vector<double> grad_net(nloc * ndescrpt);
  deepmd::prod_force_grad_a_cpu<double>(&grad_net[0], &grad[0], &env_deriv[0], &nlist[0], nloc, nnei);
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
TEST_F(TestProdForceGradA, gpu)
{
  std::vector<double> grad_net(nloc * ndescrpt);
  int * nlist_dev = NULL;
  double * grad_net_dev = NULL, * grad_dev = NULL, * env_deriv_dev = NULL;

  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory_sync(grad_dev, grad);
  deepmd::malloc_device_memory_sync(env_deriv_dev, env_deriv);
  deepmd::malloc_device_memory(grad_net_dev, nloc * ndescrpt);
  deepmd::prod_force_grad_a_gpu_cuda<double>(grad_net_dev, grad_dev, env_deriv_dev, nlist_dev, nloc, nnei);
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
