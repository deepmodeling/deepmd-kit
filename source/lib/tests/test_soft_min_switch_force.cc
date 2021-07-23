#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "neighbor_list.h"
#include "soft_min_switch.h"
#include "soft_min_switch_force.h"

class TestSoftMinSwitchForce : public ::testing::Test
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
  double alpha = .5;
  double rmin = 0.8;
  double rmax = 1.5;
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 5, 10};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  std::vector<double> sw_value, sw_deriv, du;
  std::vector<double> rij;
  std::vector<int> nlist;
  std::vector<int> fmt_nlist_a;
  std::vector<double > expected_force = {
    2.24044, -1.75363, -1.50088, -2.54065,  1.08035,  1.93630,  1.12909,  1.64972, -1.10112, -2.07854,  0.69062, -1.29217,  0.14032, -1.16008,  1.86286,  1.71311,  0.49339, -0.59049, -0.88441, -1.66176,  1.09480, -0.01957, -0.01188,  0.02612,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.36743,  0.67600, -0.43959, -0.03250, -0.00298,  0.00469, -0.02791,  0.00109, -0.00167, -0.00681, -0.00083,  0.00115,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, 
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
      for (int jj = 0; jj < nnei * 3; ++jj){
	rij[ii*nnei*3 + jj] = t_rij[jj];
      }      
    }
    sw_value.resize(nloc);
    sw_deriv.resize(nloc * nnei * 3);
    deepmd::soft_min_switch_cpu<double> (&sw_value[0], &sw_deriv[0], &rij[0], &nlist[0], nloc, 
				 nnei, alpha, rmin, rmax);
    du.resize(nloc);
    for (int ii = 0; ii < nloc; ++ii){
      du[ii] = 1.0 - ii * 0.1;
    }
  }
  void TearDown() override {
  }
};

TEST_F(TestSoftMinSwitchForce, cpu)
{
  std::vector<double> force(nall * 3);
  deepmd::soft_min_switch_force_cpu(
      &force[0],
      &du[0],
      &sw_deriv[0],
      &nlist[0],
      nloc,
      nall,
      nnei);
  EXPECT_EQ(force.size(), expected_force.size());
  for (int jj = 0; jj < force.size(); ++jj){
    EXPECT_LT(fabs(force[jj] - expected_force[jj]) , 1e-5);
  }  
  // for (int ii = 0; ii < nall * 3; ++ii){
  //   printf("%8.5f, ", force[ii]);
  // }  
  // printf("\n");
}

