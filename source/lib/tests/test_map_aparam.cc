#include <iostream>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "neighbor_list.h"
#include "map_aparam.h"

class TestMapAparam : public ::testing::Test
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
  int numb_aparam = 2;
  double rc = 6;
  double rc_smth = 0.8;
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 5, 10};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  std::vector<int> nlist;
  std::vector<int> fmt_nlist_a;
  std::vector<double> aparam;
  std::vector<double > expected_output = {
    3.40000,  3.30000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  9.80000,  9.70000,  3.60000,  3.50000,  3.20000,  3.10000,  3.00000,  2.90000,  0.00000,  0.00000, 10.00000,  9.90000,  3.40000,  3.30000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  3.60000,  3.50000,  3.20000,  3.10000,  3.00000,  2.90000,  0.00000,  0.00000,  0.00000,  0.00000,  8.80000,  8.70000,  9.40000,  9.30000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  8.60000,  8.50000,  9.20000,  9.10000,  9.00000,  8.90000,  0.00000,  0.00000,  0.00000,  0.00000,  8.80000,  8.70000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  9.20000,  9.10000,  9.00000,  8.90000,  9.60000,  9.50000,  8.60000,  8.50000,  0.00000,  0.00000,  9.40000,  9.30000,  8.80000,  8.70000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  9.00000,  8.90000,  9.60000,  9.50000,  8.60000,  8.50000,  0.00000,  0.00000,  0.00000,  0.00000,  9.40000,  9.30000,  8.80000,  8.70000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  9.20000,  9.10000,  9.60000,  9.50000,  8.60000,  8.50000,  0.00000,  0.00000,  0.00000,  0.00000, 
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
    for(int ii = 0; ii < nloc; ++ii){      
      // format nlist and record
      format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
      for (int jj = 0; jj < nnei; ++jj){
	nlist[ii*nnei + jj] = fmt_nlist_a[jj];
      }
    }
    aparam.resize(nall * numb_aparam);
    for(int ii = 0; ii < nall * numb_aparam; ++ii){
      aparam[ii] = 10 - 0.1 * ii;
    }
  }
  void TearDown() override {
  }
};

TEST_F(TestMapAparam, cpu)
{
  std::vector<double> output(nloc * nnei * numb_aparam);
  deepmd::map_aparam_cpu(
      &output[0],
      &aparam[0],
      &nlist[0],
      nloc,
      nnei,
      numb_aparam);  
  for (int jj = 0; jj < nloc * nnei * numb_aparam; ++jj){
    EXPECT_LT(fabs(output[jj] - expected_output[jj]), 1e-10);
  }
  // for (int jj = 0; jj < nloc * nnei * numb_aparam; ++jj){
  //   printf("%8.5f, ", output[jj]);
  // }
  // printf("\n");
}
