#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "neighbor_list.h"
#include "prod_virial.h"
#include "device.h"

class TestProdVirialA : public ::testing::Test
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
    100.14628,  7.21146, -24.62874,  6.19651, 23.31547, -19.77773, -26.79150, -20.92554, 38.84203,
  };  
  std::vector<double > expected_atom_virial = {
    -3.24191,  1.35810,  2.45333, -9.14879,  3.83260,  6.92341, -10.54930,  4.41930,  7.98326, 14.83563, -6.21493, -11.22697,  4.51124, -1.88984, -3.41391,  2.04717, -0.85760, -1.54921,  0.84708, -0.10308,  0.07324,  3.51825, -0.49788,  0.40314,  2.91345, -0.37264,  0.27386, 12.62246, -5.19874,  7.42677,  4.80217, -2.69029,  5.41896,  9.55811, -2.42899,  5.14893,  9.90295,  4.54279, -7.75115, -2.89155, 13.50055, -20.91993,  4.00314, -1.76293,  2.92724, 20.15105,  2.86856, -3.55868, -4.22796, -1.12700,  1.46999, -21.43180, -9.30194, 12.54538,  2.86811,  5.92934, -3.94618,  4.83313,  5.21197, -3.36488,  6.67852,  8.34225, -5.44992,  5.97941,  1.92669, -4.70211,  4.91215,  1.63145, -3.96250,  3.27415,  1.02612, -2.52585,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  1.38833,  0.50613, -1.26233,  1.39901,  5.18116, -2.18118, -17.72748, -19.52039, 18.66001, 14.31034,  1.31715, -2.05955, -0.10872,  0.00743,  0.03656, -3.85572, -0.33481,  0.57900, 14.31190, -0.53814,  0.89498, -1.94166,  0.07960, -0.10726, -0.35985,  0.03981,  0.03397,  6.17091,  0.81760, -0.97011,  0.53923,  0.07572, -0.08012, -1.34189, -0.17373,  0.21536,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, 
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
    rij.resize(nloc * nnei * 3);
    for(int ii = 0; ii < nloc; ++ii){      
      // format nlist and record
      format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
      for (int jj = 0; jj < nnei; ++jj){
	nlist[ii*nnei + jj] = fmt_nlist_a[jj];
      }
      std::vector<double > t_env, t_env_deriv, t_rij;
      // compute env_mat and its deriv, record
      deepmd::env_mat_a_cpu<double>(t_env, t_env_deriv, t_rij, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
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

TEST_F(TestProdVirialA, cpu)
{
  std::vector<double> virial(9);
  std::vector<double> atom_virial(nall * 9);
  int n_a_sel = nnei;
  deepmd::prod_virial_a_cpu<double> (&virial[0], &atom_virial[0], &net_deriv[0], &env_deriv[0], &rij[0], &nlist[0], nloc, nall, nnei);
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
  // for (int jj = 0; jj < 9; ++jj){
  //   printf("%8.5f, ", virial[jj]);
  // }
  // for (int jj = 0; jj < nall * 9; ++jj){
  //   printf("%8.5f, ", atom_virial[jj]);
  // }
  // printf("\n");
}

#if GOOGLE_CUDA
TEST_F(TestProdVirialA, gpu_cuda)
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

  deepmd::prod_virial_a_gpu_cuda<double> (virial_dev, atom_virial_dev, net_deriv_dev, env_deriv_dev, rij_dev, nlist_dev, nloc, nall, nnei);
  
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
