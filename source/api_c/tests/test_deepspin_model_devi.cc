// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "c_api.h"
#include "test_utils.h"

namespace {

void compute_atomic_virial(DP_DeepSpin* dp,
                           const int natoms,
                           const double* coord,
                           const double* spin,
                           const int* atype,
                           const double* box,
                           double* atomic_virial) {
  DP_DeepSpinCompute2(dp, 1, natoms, coord, spin, atype, box, nullptr, nullptr,
                      nullptr, nullptr, nullptr, nullptr, nullptr,
                      atomic_virial);
}

void compute_atomic_virial(DP_DeepSpin* dp,
                           const int natoms,
                           const float* coord,
                           const float* spin,
                           const int* atype,
                           const float* box,
                           float* atomic_virial) {
  DP_DeepSpinComputef2(dp, 1, natoms, coord, spin, atype, box, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr,
                       atomic_virial);
}

void compute_model_devi_atomic_virial(DP_DeepSpinModelDevi* dp,
                                      const int natoms,
                                      const double* coord,
                                      const double* spin,
                                      const int* atype,
                                      const double* box,
                                      double* atomic_virial) {
  DP_DeepSpinModelDeviCompute2(dp, 1, natoms, coord, spin, atype, box, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, atomic_virial);
}

void compute_model_devi_atomic_virial(DP_DeepSpinModelDevi* dp,
                                      const int natoms,
                                      const float* coord,
                                      const float* spin,
                                      const int* atype,
                                      const float* box,
                                      float* atomic_virial) {
  DP_DeepSpinModelDeviComputef2(dp, 1, natoms, coord, spin, atype, box, nullptr,
                                nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, atomic_virial);
}

template <class VALUETYPE>
class TestInferDeepSpinModelDeviC : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                                 0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};

  DP_DeepSpin* dp = nullptr;
  DP_DeepSpinModelDevi* dp_md = nullptr;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    const char* model = "../../tests/infer/deeppot_dpa_spin.pth";
    const char* models[] = {model, model};
    dp = DP_NewDeepSpin(model);
    dp_md = DP_NewDeepSpinModelDevi(models, 2);

    const char* error = DP_DeepSpinCheckOK(dp);
    const std::string error_message(error);
    DP_DeleteChar(error);
    ASSERT_TRUE(error_message.empty()) << error_message;

    error = DP_DeepSpinModelDeviCheckOK(dp_md);
    const std::string model_devi_error_message(error);
    DP_DeleteChar(error);
    ASSERT_TRUE(model_devi_error_message.empty()) << model_devi_error_message;
  }

  void TearDown() override {
    DP_DeleteDeepSpin(dp);
    DP_DeleteDeepSpinModelDevi(dp_md);
  }
};

TYPED_TEST_SUITE(TestInferDeepSpinModelDeviC, ValueTypes);

TYPED_TEST(TestInferDeepSpinModelDeviC, copies_atomic_virial_for_every_model) {
  using VALUETYPE = TypeParam;
  const int natoms = this->atype.size();
  const int nmodels = 2;
  const size_t per_model_size = static_cast<size_t>(natoms) * 9;
  const VALUETYPE sentinel = std::numeric_limits<VALUETYPE>::quiet_NaN();

  std::vector<VALUETYPE> expected(per_model_size, sentinel);
  compute_atomic_virial(this->dp, natoms, this->coord.data(), this->spin.data(),
                        this->atype.data(), this->box.data(), expected.data());

  // A NaN sentinel makes an omitted copy-out deterministic even when a valid
  // model happens to produce zero for some atomic-virial components.
  std::vector<VALUETYPE> actual(nmodels * per_model_size, sentinel);
  compute_model_devi_atomic_virial(this->dp_md, natoms, this->coord.data(),
                                   this->spin.data(), this->atype.data(),
                                   this->box.data(), actual.data());

  for (size_t ii = 0; ii < per_model_size; ++ii) {
    ASSERT_FALSE(std::isnan(expected[ii]))
        << "Single-model reference did not populate element " << ii;
  }
  for (int model = 0; model < nmodels; ++model) {
    for (size_t ii = 0; ii < per_model_size; ++ii) {
      const size_t output_index = model * per_model_size + ii;
      EXPECT_FALSE(std::isnan(actual[output_index]))
          << "Model-deviation output did not populate model " << model
          << ", element " << ii;
      EXPECT_LT(std::fabs(actual[output_index] - expected[ii]), EPSILON)
          << "Unexpected model-major atomic virial at model " << model
          << ", element " << ii;
    }
  }
}

}  // namespace
