// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "deepmd.hpp"

TEST(TestDeepmdException, deepmdexception) {
  std::string expected_error_message = "DeePMD-kit C API Error: unittest";
  try {
    throw deepmd::hpp::deepmd_exception("unittest");
  } catch (deepmd::hpp::deepmd_exception& ex) {
    EXPECT_STREQ(expected_error_message.c_str(), ex.what());
  }
}

TEST(TestDeepmdException, deepmdexception_nofile) {
  ASSERT_THROW(deepmd::hpp::DeepPot("_no_such_file.pb"),
               deepmd::hpp::deepmd_exception);
}

TEST(TestChargeSpinValidation, exact_frame_values_use_input_storage) {
  std::vector<double> charge_spin = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> tiled;

  const double* result =
      deepmd::hpp::validate_charge_spin(charge_spin, 2, 2, tiled);

  EXPECT_EQ(result, charge_spin.data());
  EXPECT_TRUE(tiled.empty());
}

TEST(TestChargeSpinValidation, single_frame_values_are_tiled) {
  std::vector<double> charge_spin = {1.0, 2.0};
  std::vector<double> tiled;

  const double* result =
      deepmd::hpp::validate_charge_spin(charge_spin, 2, 3, tiled);

  EXPECT_EQ(result, tiled.data());
  EXPECT_EQ(tiled, (std::vector<double>{1.0, 2.0, 1.0, 2.0, 1.0, 2.0}));
}

TEST(TestChargeSpinValidation, empty_values_use_model_default_path) {
  std::vector<double> charge_spin;
  std::vector<double> tiled;

  const double* result =
      deepmd::hpp::validate_charge_spin(charge_spin, 2, 3, tiled);

  EXPECT_EQ(result, nullptr);
  EXPECT_TRUE(tiled.empty());
}

TEST(TestChargeSpinValidation, ignores_values_for_unsupported_model) {
  std::vector<double> charge_spin = {1.0, 2.0};
  std::vector<double> tiled;

  const double* result =
      deepmd::hpp::validate_charge_spin(charge_spin, 0, 1, tiled);

  EXPECT_EQ(result, nullptr);
  EXPECT_TRUE(tiled.empty());
}

TEST(TestChargeSpinValidation, rejects_invalid_size) {
  std::vector<double> charge_spin = {1.0, 2.0, 3.0};
  std::vector<double> tiled;

  EXPECT_THROW(deepmd::hpp::validate_charge_spin(charge_spin, 2, 2, tiled),
               deepmd::hpp::deepmd_exception);
}
