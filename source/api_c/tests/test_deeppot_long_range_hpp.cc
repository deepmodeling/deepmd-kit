// SPDX-License-Identifier: LGPL-3.0-or-later
// Test the C API with a pt_expt (.pt2) DPA4C model that carries a
// latent-charge long-range module (source/tests/infer/gen_dpa4c_long_range.py).
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "../../api_cc/tests/expected_ref.h"
#include "DeepPotPTExpt.h"
#include "deepmd.hpp"
#include "test_utils.h"

namespace {
constexpr const char* kModelPath =
    "../../tests/infer/deeppot_dpa4c_long_range.pt2";
constexpr const char* kRefPath =
    "../../tests/infer/deeppot_dpa4c_long_range.expected";
constexpr const char* kShortRangeModelPath =
    "../../tests/infer/deeppot_sea.pth";
}  // namespace

template <class VALUETYPE>
class TestInferDeepPotLongRangeHPP : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  0.25,  3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> expected_f;
  double expected_tot_e = 0.;
  deepmd::hpp::DeepPot dp;

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp.init(kModelPath);
    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_f = ref.get<VALUETYPE>("default", "expected_f");
    for (const VALUETYPE value : ref.get<VALUETYPE>("default", "expected_e")) {
      expected_tot_e += value;
    }
  }
};

TYPED_TEST_SUITE(TestInferDeepPotLongRangeHPP, ValueTypes);

TYPED_TEST(TestInferDeepPotLongRangeHPP, has_long_range) {
  EXPECT_TRUE(this->dp.has_long_range());
}

TEST(TestInferDeepPotModelDeviLongRangeHPP, has_long_range) {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
  GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
  // A long-range model anywhere in the ensemble confines all of them.
  deepmd::hpp::DeepPotModelDevi mixed;
  mixed.init(std::vector<std::string>{kShortRangeModelPath, kModelPath});
  EXPECT_TRUE(mixed.has_long_range());
  deepmd::hpp::DeepPotModelDevi short_range;
  short_range.init(
      std::vector<std::string>{kShortRangeModelPath, kShortRangeModelPath});
  EXPECT_FALSE(short_range.has_long_range());
}

TYPED_TEST(TestInferDeepPotLongRangeHPP, nopbc) {
  using VALUETYPE = TypeParam;
  const double epsilon = std::is_same<VALUETYPE, double>::value ? 1e-8 : 1e-4;
  double ener;
  std::vector<VALUETYPE> force, virial;
  this->dp.compute(ener, force, virial, this->coord, this->atype,
                   std::vector<VALUETYPE>());
  ASSERT_EQ(force.size(), this->expected_f.size());
  EXPECT_LT(fabs(ener - this->expected_tot_e), epsilon);
  for (std::size_t ii = 0; ii < force.size(); ++ii) {
    EXPECT_LT(fabs(force[ii] - this->expected_f[ii]), epsilon);
  }
}
