#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "neighbor_list.h"
#include "soft_min_switch.h"

class TestSoftMinSwitch : public ::testing::Test
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
  double alpha = 0.5;
  double rmin = 0.8;
  double rmax = 1.5;
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 5, 10};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  std::vector<double> net_deriv, in_deriv;
  std::vector<double> rij;
  std::vector<int> nlist;
  std::vector<int> fmt_nlist_a;
  std::vector<double > expected_value = {
     0.84693,  0.57040,  0.41834,  0.89258,  0.63482,  0.60391, 
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
  }
  void TearDown() override {
  }
};

TEST_F(TestSoftMinSwitch, cpu)
{
  std::vector<double> sw_value(nloc);
  std::vector<double> sw_deriv(nloc * nnei * 3);
  deepmd::soft_min_switch_cpu<double> (&sw_value[0], &sw_deriv[0], &rij[0], &nlist[0], nloc, nnei, alpha, rmin, rmax);
  EXPECT_EQ(sw_value.size(), nloc);
  EXPECT_EQ(sw_value.size(), expected_value.size());
  EXPECT_EQ(sw_deriv.size(), nloc * nnei * 3);
  for (int jj = 0; jj < nloc; ++jj){
    EXPECT_LT(fabs(sw_value[jj] - expected_value[jj]) , 1e-5);
  }  
  // for (int jj = 0; jj < nloc; ++jj){
  //   printf("%8.5f, ", sw_value[jj]);
  // }
  // printf("\n");
}

TEST_F(TestSoftMinSwitch, cpu_num_deriv)
{
  std::vector<double> sw_value(nloc);
  std::vector<double> sw_deriv(nloc * nnei * 3);
  std::vector<double> sw_value_0(nloc);
  std::vector<double> sw_deriv_0(nloc * nnei * 3);
  std::vector<double> sw_value_1(nloc);
  std::vector<double> sw_deriv_1(nloc * nnei * 3);
  std::vector<double > env, env_deriv;
  std::vector<double> t_rij_0, t_rij_1;
  std::vector<double> rij_0, rij_1;
  std::vector<int> fmt_nlist_a;
  double hh = 1e-5;
  
  deepmd::soft_min_switch_cpu<double> (&sw_value[0], &sw_deriv[0], &rij[0], &nlist[0], nloc, nnei, alpha, rmin, rmax);
  EXPECT_EQ(sw_value.size(), nloc);
  EXPECT_EQ(sw_deriv.size(), nloc * nnei * 3);

  for (int ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);    
    EXPECT_EQ(ret, -1);
    
    int i_idx = ii;
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[ii*nnei + jj];
      if (j_idx < 0) continue;
      for (int dd = 0; dd < 3; ++dd){
	std::vector<double> posi_0 = posi_cpy;
	std::vector<double> posi_1 = posi_cpy;
	posi_0[j_idx*3+dd] -= hh;
	posi_1[j_idx*3+dd] += hh;
	deepmd::env_mat_a_cpu<double>(env, env_deriv, t_rij_0, posi_0, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	deepmd::env_mat_a_cpu<double>(env, env_deriv, t_rij_1, posi_1, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);
	EXPECT_EQ(t_rij_0.size(), nnei * 3);
	EXPECT_EQ(t_rij_1.size(), nnei * 3);
	rij_0 = rij;
	rij_1 = rij;
	for (int dd1 = 0; dd1 < 3; ++dd1){
	  rij_0[ii*nnei*3 + jj*3 + dd] = t_rij_0[jj*3 + dd];
	  rij_1[ii*nnei*3 + jj*3 + dd] = t_rij_1[jj*3 + dd];
	}      
	deepmd::soft_min_switch_cpu<double> (&sw_value_0[0], &sw_deriv_0[0], &rij_0[0], &nlist[0], nloc, nnei, alpha, rmin, rmax);
	deepmd::soft_min_switch_cpu<double> (&sw_value_1[0], &sw_deriv_1[0], &rij_1[0], &nlist[0], nloc, nnei, alpha, rmin, rmax);
	double ana_deriv = sw_deriv[ii*nnei*3 + jj*3 + dd];
	double num_deriv = (sw_value_1[ii] - sw_value_0[ii]) / (2. * hh);
	EXPECT_LT(fabs(num_deriv - ana_deriv), 1e-5);
      }
    }
  }
}
