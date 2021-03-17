#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include "neighbor_list.h"
#include "test_utils.h"
#include "ewald.h"

class TestInferEwald : public ::testing::Test
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
  std::vector<double> charge = {
    -2, 1, 1, -2, 1, 1
  };
  std::vector<double> box = {
    13., 0., 0., 0., 13., 0., 0., 0., 13.
  };
  void SetUp() override {
  };
  void TearDown() override {
  };
};

TEST_F(TestInferEwald, cpu_numfv)
{
  class MyModel : public EnergyModelTest<double>
  {
    const std::vector<double > & charge;
    deepmd::EwaldParameters<double> eparam;    
public:
    MyModel(
	const std::vector<double> & charge_
	) : charge(charge_) {
      eparam.beta = 0.4;
    };
    virtual void compute (
	double & ener,
	std::vector<double> &	force,
	std::vector<double> &	virial,
	const std::vector<double> & coord,
	const std::vector<double> & box) {
      deepmd::Region<double> region;
      init_region_cpu(region, &box[0]);
      ewald_recp(ener, force, virial, coord, charge, region, eparam);
    }
  };
  MyModel model(charge);
  model.test_f(coord, box);
  model.test_v(coord, box);
  std::vector<double> box_(box);
  box_[1] -= 0.2;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[2] += 0.5;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[4] += 0.2;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[3] -= 0.3;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[6] -= 0.7;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[7] += 0.6;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
}
