#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include "ewald.h"

class TestEwald : public ::testing::Test
{
protected:
  std::vector<double> coord = {
    12.83, 2.56, 2.18,
    12.09, 2.87, 2.74,
    00.25, 3.32, 1.68,
    3.36, 3.00, 1.81,
    3.51, 2.51, 2.60,
    4.27, 3.22, 1.56 
  };
  std::vector<double > charge = {
    -2, 1, 1, -2, 1, 1,
  };
  std::vector<double > boxt = {
    13., 0., 0., 0., 13., 0., 0., 0., 13.
  };
  deepmd::EwaldParameters<double> eparam;
  double expected_e = 4.7215808340392229e+00;
  std::vector<double> expected_f = {
    -5.4937025715874448e+00,5.6659817006308417e+00,3.8059426028301313e-01,2.5210962791915938e+00,-2.6383552457553545e+00,-4.8998411247787405e-01,2.7390037416771147e+00,-3.2890571945143514e+00,3.8057620258450320e-01,6.7561832843578351e+00,-1.3707287681111919e+00,2.7733203842981604e+00,-3.3297964389679557e+00,1.0404967238120841e+00,-1.8035649784287722e+00,-3.1927842946711418e+00,5.9166278393797123e-01,-1.2409417562590299e+00,
  };
  std::vector<double> expected_v = {
    6.5088081157418898e-01,1.9076542856278367e+00,-9.8010077026955389e-01,1.9076542856278367e+00,1.3101841366497322e+00,1.9794445391572657e-01,-9.8010077026955389e-01,1.9794445391572657e-01,1.9232614011636004e+00
  };
  
  void SetUp() override {    
  };
};


TEST_F(TestEwald, cpu)
{
  double ener;
  std::vector<double > force, virial;
  deepmd::Region<double> region;
  init_region_cpu(region, &boxt[0]);
  ewald_recp(ener, force, virial, coord, charge, region, eparam);
  EXPECT_LT(fabs(ener - expected_e), 1e-10);
  for(int ii = 0; ii < force.size(); ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);
  }
  for(int ii = 0; ii < virial.size(); ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_v[ii]), 1e-10);
  }
}

