#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "region.h"
#include "SimulationRegion.h"
#include "device.h"

class TestRegion : public ::testing::Test
{
protected:
  std::vector<double > ref_boxt = {
    3.27785716,  0.09190842,  0.14751448,  0.02331264,  4.36482777, -0.2999871 , -0.47510999, -0.38123489,  5.33561809
  };
  double expected_vol = 76.26958621360133;
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
  deepmd::Region<double> region;
  init_region_cpu(region, &ref_boxt[0]);
  for(int ii = 0; ii < 9; ++ii){
    EXPECT_LT(fabs(region.rec_boxt[ii] - ref_rec_boxt[ii]), 1e-10);
  }
  // check volume
  double vol = volume_cpu(region);
  EXPECT_LT(fabs(vol - expected_vol), 1e-10);
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
#if GOOGLE_CUDA
TEST_F(TestRegion, gpu)
{
  // check rec_box
  deepmd::Region<double> region;
  deepmd::Region<double> region_dev;
  double * new_boxt = region_dev.boxt;
  double * new_rec_boxt = region_dev.rec_boxt;
  double * boxt_dev = NULL, * rec_boxt_dev = NULL;
  double * ref_rp_dev = NULL, * ref_ri_dev = NULL;
  init_region_cpu(region, &ref_boxt[0]);
  for(int ii = 0; ii < 9; ++ii){
    EXPECT_LT(fabs(region.rec_boxt[ii] - ref_rec_boxt[ii]), 1e-10);
  }
  deepmd::malloc_device_memory_sync(boxt_dev, region.boxt, 9);
  deepmd::malloc_device_memory_sync(rec_boxt_dev, region.rec_boxt, 9);
  deepmd::malloc_device_memory_sync(ref_rp_dev, ref_rp);
  deepmd::malloc_device_memory_sync(ref_ri_dev, ref_ri);
  region_dev.boxt = boxt_dev;
  region_dev.rec_boxt = rec_boxt_dev;
  // check volume
  double vol[1];
  double * vol_dev = NULL;
  deepmd::malloc_device_memory(vol_dev, 1);
  deepmd::volume_gpu(vol_dev, region_dev);
  deepmd::memcpy_device_to_host(vol_dev, vol, 1);
  EXPECT_LT(fabs(vol[0] - expected_vol), 1e-10);
  // check conversion between phys and inter coords.
  double ri[3];
  double * ri_dev = NULL;
  deepmd::malloc_device_memory(ri_dev, 3);
  deepmd::convert_to_inter_gpu(ri_dev, region_dev, ref_rp_dev);
  deepmd::memcpy_device_to_host(ri_dev, ri, 3);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(ri[ii] - ref_ri[ii]), 1e-10);
  }
  double rp2[3];
  double * rp2_dev = NULL;
  deepmd::malloc_device_memory(rp2_dev, 3);
  deepmd::convert_to_phys_gpu(rp2_dev, region_dev, ri_dev);
  deepmd::memcpy_device_to_host(rp2_dev, rp2, 3);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(rp2[ii] - ref_rp[ii]), 1e-10);
  }
  double rp[3];
  double * rp_dev = NULL;
  deepmd::malloc_device_memory(rp_dev, 3);
  deepmd::convert_to_phys_gpu(rp_dev, region_dev, ref_ri_dev);
  deepmd::memcpy_device_to_host(rp_dev, rp, 3);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(rp[ii] - ref_rp[ii]), 1e-10);
  }
  double ri2[3];
  double * ri2_dev = NULL;
  deepmd::malloc_device_memory(ri2_dev, 3);
  deepmd::convert_to_inter_gpu(ri2_dev, region_dev, rp_dev);
  deepmd::memcpy_device_to_host(ri2_dev, ri2, 3);
  for(int ii = 0; ii < 3; ++ii){
    EXPECT_LT(fabs(ri2[ii] - ref_ri[ii]), 1e-10);
  }
  deepmd::delete_device_memory(boxt_dev);
  deepmd::delete_device_memory(rec_boxt_dev);
  deepmd::delete_device_memory(vol_dev);
  deepmd::delete_device_memory(ref_rp_dev);
  deepmd::delete_device_memory(ref_ri_dev);
  deepmd::delete_device_memory(ri_dev);
  deepmd::delete_device_memory(rp2_dev);
  deepmd::delete_device_memory(rp_dev);
  deepmd::delete_device_memory(ri2_dev);
  region_dev.boxt = new_boxt;
  region_dev.rec_boxt = new_rec_boxt;
}
#endif // GOOGLE_CUDA
    

// double square_root (const double xx)
// {
//   return sqrt(xx);
// }

// TEST (SquareRootTest, PositiveNos) { 
//     EXPECT_EQ (18.0, square_root (324.0));
//     EXPECT_EQ (25.4, square_root (645.16));
//     EXPECT_EQ (50.332, square_root (2533.310224));
// }


