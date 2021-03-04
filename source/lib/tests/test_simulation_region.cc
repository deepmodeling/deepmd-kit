#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "region.h"
#include "SimulationRegion.h"

class TestRegion : public ::testing::Test
{
protected:
  std::vector<double > ref_boxt = {
    3.27785716,  0.09190842,  0.14751448,  0.02331264,  4.36482777, -0.2999871 , -0.47510999, -0.38123489,  5.33561809
  };
  // rec_boxt = boxt^{-T}
  std::vector<double > ref_rec_boxt = {
    3.0385229041853185e-01,  2.3783430948044884e-04, 2.7073513689027690e-02, -7.1670232142159460e-03, 2.3022911797728179e-01,  1.5811897837543720e-02, -8.8035961973365381e-03,  1.2937710358702505e-02, 1.8756020637229892e-01
  };
  std::vector<double > ref_rp = {
    1.5, 2.5, 3.5
  };
  std::vector<double > ref_ri = {
    0.5511303193130958, 0.6201639025532836, 0.6755996039037975, 
  };
};

TEST_F(TestRegion, orig)
{
  SimulationRegion<double> region;
  region.reinitBox(&ref_boxt[0]);
  const double * rec_boxt = region.getRecBoxTensor();
  for(int ii = 0; ii < 9; ++ii){
    EXPECT_LT(fabs(rec_boxt[ii] - ref_rec_boxt[ii]), 1e-10);
  }
  double ri[3];
  region.phys2Inter(ri, &ref_rp[0]);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(ri[ii] - ref_ri[ii]), 1e-10);
  }
}

TEST_F(TestRegion, cpu)
{
  // check rec_box
  Region<double> region;
  init_region_cpu(region, &ref_boxt[0]);
  for(int ii = 0; ii < 9; ++ii){
    EXPECT_LT(fabs(region.rec_boxt[ii] - ref_rec_boxt[ii]), 1e-10);
  }
  // check conversion between phys and inter coords.
  double ri[3];
  convert_to_inter_cpu(ri, region, &ref_rp[0]);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(ri[ii] - ref_ri[ii]), 1e-10);
  }
  double rp2[3];
  convert_to_phys_cpu(rp2, region, ri);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(rp2[ii] - ref_rp[ii]), 1e-10);
  }
  double rp[3];
  convert_to_phys_cpu(rp, region, &ref_ri[0]);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(rp[ii] - ref_rp[ii]), 1e-10);
  }
  double ri2[3];
  convert_to_inter_cpu(ri2, region, rp);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(ri2[ii] - ref_ri[ii]), 1e-10);
  }
}
    

// double square_root (const double xx)
// {
//   return sqrt(xx);
// }

// TEST (SquareRootTest, PositiveNos) { 
//     EXPECT_EQ (18.0, square_root (324.0));
//     EXPECT_EQ (25.4, square_root (645.16));
//     EXPECT_EQ (50.332, square_root (2533.310224));
// }


