#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include "deepmd.hpp"

typedef double VALUETYPE;
#define EPSILON 1e-10

class TestInferDeepPotModeDevi : public ::testing::Test
{  
protected:  
  std::vector<VALUETYPE> coord = {
    12.83, 2.56, 2.18,
    12.09, 2.87, 2.74,
    00.25, 3.32, 1.68,
    3.36, 3.00, 1.81,
    3.51, 2.51, 2.60,
    4.27, 3.22, 1.56 
  };
  std::vector<int> atype = {
    0, 1, 1, 0, 1, 1
  };
  std::vector<VALUETYPE> box = {
    13., 0., 0., 0., 13., 0., 0., 0., 13.
  };
  int natoms;

  deepmd::hpp::DeepPot dp0;
  deepmd::hpp::DeepPot dp1;
  deepmd::hpp::DeepPotModelDevi dp_md;

  void SetUp() override {
    {
      std::string file_name = "../../tests/infer/deeppot.pbtxt";
      deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/deeppot.pbtxt", "deeppot.pb");
      dp0.init("deeppot.pb");
    }
    {
      std::string file_name = "../../tests/infer/deeppot-1.pbtxt";
      deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/deeppot-1.pbtxt", "deeppot-1.pb");
      dp1.init("deeppot-1.pb");
    }
    dp_md.init(std::vector<std::string>({"deeppot.pb", "deeppot-1.pb"}));
  };

  void TearDown() override {
    remove( "deeppot.pb" ) ;
    remove( "deeppot-1.pb" ) ;
  };
};


class TestInferDeepPotModeDeviPython : public ::testing::Test
{  
protected:  
  std::vector<VALUETYPE> coord = {
    4.170220047025740423e-02,7.203244934421580703e-02,1.000114374817344942e-01,
    4.053881673400336005e+00,4.191945144032948461e-02,6.852195003967595510e-02,
    1.130233257263184132e+00,1.467558908171130543e-02,1.092338594768797883e-01,
    1.862602113776709242e-02,1.134556072704304919e+00,1.396767474230670159e-01,
    5.120445224973151355e+00,8.781174363909455272e-02,2.738759319792616331e-03,
    4.067046751017840300e+00,1.141730480236712753e+00,5.586898284457517128e-02,
  };
  std::vector<int> atype = {
    0, 0, 1, 1, 1, 1
  };
  std::vector<VALUETYPE> box = {
    20., 0., 0., 0., 20., 0., 0., 0., 20.
  };
  int natoms;
  std::vector<VALUETYPE> expected_md_f = {
    0.509504727653, 0.458424067748, 0.481978258466
  }; // max min avg
  std::vector<VALUETYPE> expected_md_v = {
    0.167004837423,0.00041822790564,0.0804864867641
  }; // max min avg

  deepmd::hpp::DeepPot dp0;
  deepmd::hpp::DeepPot dp1;
  deepmd::hpp::DeepPotModelDevi dp_md;

  void SetUp() override {
    {
      std::string file_name = "../../tests/infer/deeppot.pbtxt";
      deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/deeppot.pbtxt", "deeppot.pb");
      dp0.init("deeppot.pb");
    }
    {
      std::string file_name = "../../tests/infer/deeppot-1.pbtxt";
      deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/deeppot-1.pbtxt", "deeppot-1.pb");
      dp1.init("deeppot-1.pb");
    }
    dp_md.init(std::vector<std::string>({"deeppot.pb", "deeppot-1.pb"}));
  };

  void TearDown() override {
    remove( "deeppot.pb" ) ;
    remove( "deeppot-1.pb" ) ;
  };
};


TEST_F(TestInferDeepPotModeDevi, attrs)
{
  EXPECT_EQ(dp0.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp0.numb_types(), dp_md.numb_types());
  //EXPECT_EQ(dp0.dim_fparam(), dp_md.dim_fparam());
  //EXPECT_EQ(dp0.dim_aparam(), dp_md.dim_aparam());
  EXPECT_EQ(dp1.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp1.numb_types(), dp_md.numb_types());
  //EXPECT_EQ(dp1.dim_fparam(), dp_md.dim_fparam());
  //EXPECT_EQ(dp1.dim_aparam(), dp_md.dim_aparam());
}

inline VALUETYPE mymax(const std::vector<VALUETYPE > & xx)
{
  VALUETYPE ret = 0;
  for (int ii = 0; ii < xx.size(); ++ii){
    if (xx[ii] > ret) {
      ret = xx[ii];
    }
  }
  return ret;
};  
inline VALUETYPE mymin(const std::vector<VALUETYPE > & xx)
{
  VALUETYPE ret = 1e10;
  for (int ii = 0; ii < xx.size(); ++ii){
    if (xx[ii] < ret) {
      ret = xx[ii];
    }
  }
  return ret;
};
inline VALUETYPE myavg(const std::vector<VALUETYPE > & xx)
{
  VALUETYPE ret = 0;
  for (int ii = 0; ii < xx.size(); ++ii){
    ret += xx[ii];
  }
  return (ret / xx.size());
};
inline VALUETYPE mystd(const std::vector<VALUETYPE > & xx)
{
  VALUETYPE ret = 0;
  for (int ii = 0; ii < xx.size(); ++ii){
    ret += xx[ii] * xx[ii];
  }
  return sqrt(ret / xx.size());
};
