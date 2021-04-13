#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "neighbor_list.h"
#include "prod_virial.h"
#include "device.h"

class TestProdVirialR : public ::testing::Test
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
  std::vector<double> env, env_deriv, rij;
  std::vector<int> nlist;
  std::vector<int> fmt_nlist_a;
  std::vector<double > expected_virial = {
    105.83531,  8.37873, -26.31645,  8.37873, 25.29640, -22.08303, -26.31645, -22.08303, 41.52565,
  };  
  std::vector<double > expected_atom_virial = {
    5.82162, -2.43879, -4.40555, -2.43879,  1.02165,  1.84557, -4.40555,  1.84557,  3.33393,  5.85102, -2.45110, -4.42780, -2.45110,  1.02681,  1.85489, -4.42780,  1.85489,  3.35077, 12.99134, -1.65136,  1.27337, -1.65136,  0.31236, -0.30952,  1.27337, -0.30952,  0.34172, 14.20717,  0.71207, -0.80046,  0.71207,  3.33417, -5.06665, -0.80046, -5.06665,  7.86673,  6.35288, -0.15554,  0.00838, -0.15554,  4.67701, -7.17573,  0.00838, -7.17573, 11.07561, 14.50559,  3.80226, -5.12103,  3.80226,  2.16638, -2.99774, -5.12103, -2.99774,  4.20621, 13.02204,  4.00163, -2.38372,  4.00163,  5.79404, -3.84611, -2.38372, -3.84611,  2.60729,  9.69976,  1.23534, -3.98748,  1.23534,  0.53911, -1.23540, -3.98748, -1.23540,  3.03034,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  5.21346,  4.51882, -5.24989,  4.51882,  6.27021, -5.00845, -5.24989, -5.00845,  5.37572,  7.57664,  0.67053, -1.12262,  0.67053,  0.07524, -0.08028, -1.12262, -0.08028,  0.18921,  7.32402, -0.29298,  0.42021, -0.29298,  0.01974,  0.00042,  0.42021,  0.00042,  0.06112,  3.26976,  0.42787, -0.51985,  0.42787,  0.05967, -0.06402, -0.51985, -0.06402,  0.08700,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
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
    rij.resize(nloc * nnei * 3);
    for(int ii = 0; ii < nloc; ++ii){      
      // format nlist and record
      format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
      for (int jj = 0; jj < nnei; ++jj){
	nlist[ii*nnei + jj] = fmt_nlist_a[jj];
      }
      std::vector<double > t_env, t_env_deriv, t_rij;
      // compute env_mat and its deriv, record
      deepmd::env_mat_r_cpu<double>(t_env, t_env_deriv, t_rij, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
      for (int jj = 0; jj < ndescrpt; ++jj){
	env[ii*ndescrpt+jj] = t_env[jj];
	for (int dd = 0; dd < 3; ++dd){
	  env_deriv[ii*ndescrpt*3+jj*3+dd] = t_env_deriv[jj*3+dd];
	}
      }
      for (int jj = 0; jj < nnei * 3; ++jj){
	rij[ii*nnei*3 + jj] = t_rij[jj];
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

TEST_F(TestProdVirialR, cpu)
{
  std::vector<double> virial(9);
  std::vector<double> atom_virial(nall * 9);
  int n_a_sel = nnei;
  deepmd::prod_virial_r_cpu<double> (&virial[0], &atom_virial[0], &net_deriv[0], &env_deriv[0], &rij[0], &nlist[0], nloc, nall, nnei);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_virial.size(), nall * 9);  
  EXPECT_EQ(virial.size(), expected_virial.size());
  EXPECT_EQ(atom_virial.size(), expected_atom_virial.size());
  for (int jj = 0; jj < virial.size(); ++jj){
    EXPECT_LT(fabs(virial[jj] - expected_virial[jj]) , 1e-5);
  }  
  for (int jj = 0; jj < atom_virial.size(); ++jj){
    EXPECT_LT(fabs(atom_virial[jj] - expected_atom_virial[jj]) , 1e-5);
  }  
  // for (int jj = 0; jj < 9; ++jj){
  //   printf("%8.5f, ", virial[jj]);
  // }
  // for (int jj = 0; jj < nall * 9; ++jj){
  //   printf("%8.5f, ", atom_virial[jj]);
  // }
  // printf("\n");
}

#if GOOGLE_CUDA
TEST_F(TestProdVirialR, gpu_cuda)
{
  std::vector<double> virial(9, 0.0);
  std::vector<double> atom_virial(nall * 9, 0.0);
  int n_a_sel = nnei;

  int * nlist_dev = NULL;
  double * virial_dev = NULL, *atom_virial_dev = NULL, * net_deriv_dev = NULL, * env_deriv_dev = NULL, * rij_dev = NULL;

  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory_sync(virial_dev, virial);
  deepmd::malloc_device_memory_sync(atom_virial_dev, atom_virial);
  deepmd::malloc_device_memory_sync(net_deriv_dev, net_deriv);  
  deepmd::malloc_device_memory_sync(env_deriv_dev, env_deriv);  
  deepmd::malloc_device_memory_sync(rij_dev, rij);  

  deepmd::prod_virial_r_gpu_cuda<double> (virial_dev, atom_virial_dev, net_deriv_dev, env_deriv_dev, rij_dev, nlist_dev, nloc, nall, nnei);
  
  deepmd::memcpy_device_to_host(virial_dev, virial);
  deepmd::memcpy_device_to_host(atom_virial_dev, atom_virial);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(virial_dev);
  deepmd::delete_device_memory(atom_virial_dev);
  deepmd::delete_device_memory(net_deriv_dev);
  deepmd::delete_device_memory(env_deriv_dev);
  deepmd::delete_device_memory(rij_dev);
  // virial are not calculated in gpu currently;
  // for (int ii = 0; ii < 9; ii++) {
  //   virial[ii] = 0;
  // }
  // for (int ii = 0; ii < nall * 9; ii++) {
  //   virial[ii % 9] += atom_virial[ii];
  // }
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(virial.size(), expected_virial.size());
  EXPECT_EQ(atom_virial.size(), nall * 9);  
  EXPECT_EQ(atom_virial.size(), expected_atom_virial.size());  
  for (int jj = 0; jj < virial.size(); ++jj){
    EXPECT_LT(fabs(virial[jj] - expected_virial[jj]) , 1e-5);
  }  
  for (int jj = 0; jj < atom_virial.size(); ++jj){
    EXPECT_LT(fabs(atom_virial[jj] - expected_atom_virial[jj]) , 1e-5);
  }
}
#endif // GOOGLE_CUDA
