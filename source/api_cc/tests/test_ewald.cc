// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "ewald.h"
#include "neighbor_list.h"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferEwald : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> charge = {-2, 1, 1, -2, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  void SetUp() override {};
  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferEwald, ValueTypes);

TYPED_TEST(TestInferEwald, cpu_numfv) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<VALUETYPE>& charge = this->charge;
  std::vector<VALUETYPE>& box = this->box;
  class MyModel : public EnergyModelTest<VALUETYPE> {
    const std::vector<VALUETYPE>& charge;
    deepmd::EwaldParameters<VALUETYPE> eparam;

   public:
    MyModel(const std::vector<VALUETYPE>& charge_) : charge(charge_) {
      eparam.beta = 0.4;
    };
    virtual void compute(double& ener,
                         std::vector<VALUETYPE>& force,
                         std::vector<VALUETYPE>& virial,
                         const std::vector<VALUETYPE>& coord,
                         const std::vector<VALUETYPE>& box) {
      deepmd::Region<VALUETYPE> region;
      init_region_cpu(region, &box[0]);
      VALUETYPE ener_;
      ewald_recp(ener_, force, virial, coord, charge, region, eparam);
      ener = ener_;
    }
  };
  MyModel model(charge);
  model.test_f(coord, box);
  model.test_v(coord, box);
  std::vector<VALUETYPE> box_(box);
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
