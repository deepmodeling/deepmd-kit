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

#include "c_api.h"
#include "c_api_internal.h"
#include "deepmd.hpp"

namespace {

constexpr char model_devi_nframes_error[] =
    "DeePMD-kit Error: Model-deviation C APIs support exactly one frame.";

template <typename MODEL, typename INVOKE, typename CHECK_OK>
void expect_model_devi_frame_error(const INVOKE& invoke,
                                   const CHECK_OK& check_ok) {
  // Use a fresh error carrier for every public entry point. Reusing a model
  // would let a previous failure remain visible through CheckOK and could hide
  // a wrapper that forgot to validate its own frame count.
  MODEL model;
  ASSERT_NO_THROW(invoke(&model));

  const char* error = check_ok(&model);
  ASSERT_NE(error, nullptr);
  EXPECT_STREQ(error, model_devi_nframes_error);
  DP_DeleteChar(error);
}

}  // namespace

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

TEST(TestModelDeviCAPIExceptionBoundary,
     rejects_two_frames_across_all_multiframe_public_variants) {
  constexpr int nframes = 2;
  constexpr int natoms = 1;
  const int atype[natoms] = {0};
  const double coord[natoms * 3] = {0.0, 0.0, 0.0};
  const float coordf[natoms * 3] = {0.0F, 0.0F, 0.0F};
  const double spin[natoms * 3] = {0.0, 0.0, 0.0};
  const float spinf[natoms * 3] = {0.0F, 0.0F, 0.0F};
  const double charge_spin[1] = {0.0};
  const float charge_spinf[1] = {0.0F};
  DP_Nlist nlist;

  const auto check_pot = [](DP_DeepPotModelDevi* model) {
    return DP_DeepPotModelDeviCheckOK(model);
  };
  const auto check_spin = [](DP_DeepSpinModelDevi* model) {
    return DP_DeepSpinModelDeviCheckOK(model);
  };

  // The four legacy model-deviation entry points have no nframes argument and
  // always delegate with one frame. The twelve calls below cover every public
  // entry point through which a caller can supply an unsupported frame count.
  // Output and optional-parameter pointers may be null by contract. A default
  // model and neighbor list deliberately keep this regression free of
  // backend/model fixtures while verifying validation precedes model access.
  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        DP_DeepPotModelDeviCompute2(model, nframes, natoms, coord, atype,
                                    nullptr, nullptr, nullptr, nullptr, nullptr,
                                    nullptr, nullptr, nullptr);
      },
      check_pot);
  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        DP_DeepPotModelDeviComputef2(model, nframes, natoms, coordf, atype,
                                     nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, nullptr);
      },
      check_pot);
  expect_model_devi_frame_error<DP_DeepSpinModelDevi>(
      [&](DP_DeepSpinModelDevi* model) {
        DP_DeepSpinModelDeviCompute2(
            model, nframes, natoms, coord, spin, atype, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      check_spin);
  expect_model_devi_frame_error<DP_DeepSpinModelDevi>(
      [&](DP_DeepSpinModelDevi* model) {
        DP_DeepSpinModelDeviComputef2(
            model, nframes, natoms, coordf, spinf, atype, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      check_spin);

  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        DP_DeepPotModelDeviComputeNList2(
            model, nframes, natoms, coord, atype, nullptr, 0, &nlist, 0,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      check_pot);
  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        DP_DeepPotModelDeviComputeNListf2(
            model, nframes, natoms, coordf, atype, nullptr, 0, &nlist, 0,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      check_pot);
  expect_model_devi_frame_error<DP_DeepSpinModelDevi>(
      [&](DP_DeepSpinModelDevi* model) {
        DP_DeepSpinModelDeviComputeNList2(model, nframes, natoms, coord, spin,
                                          atype, nullptr, 0, &nlist, 0, nullptr,
                                          nullptr, nullptr, nullptr, nullptr,
                                          nullptr, nullptr, nullptr);
      },
      check_spin);
  expect_model_devi_frame_error<DP_DeepSpinModelDevi>(
      [&](DP_DeepSpinModelDevi* model) {
        DP_DeepSpinModelDeviComputeNListf2(model, nframes, natoms, coordf,
                                           spinf, atype, nullptr, 0, &nlist, 0,
                                           nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr);
      },
      check_spin);

  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        DP_DeepPotModelDeviCompute3(
            model, nframes, natoms, coord, atype, nullptr, nullptr, nullptr,
            charge_spin, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      check_pot);
  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        DP_DeepPotModelDeviComputef3(
            model, nframes, natoms, coordf, atype, nullptr, nullptr, nullptr,
            charge_spinf, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      check_pot);
  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        DP_DeepPotModelDeviComputeNList3(model, nframes, natoms, coord, atype,
                                         nullptr, 0, &nlist, 0, nullptr,
                                         nullptr, charge_spin, nullptr, nullptr,
                                         nullptr, nullptr, nullptr);
      },
      check_pot);
  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        DP_DeepPotModelDeviComputeNListf3(model, nframes, natoms, coordf, atype,
                                          nullptr, 0, &nlist, 0, nullptr,
                                          nullptr, charge_spinf, nullptr,
                                          nullptr, nullptr, nullptr, nullptr);
      },
      check_pot);
}

class TestModelDeviInvalidFrameCount : public ::testing::TestWithParam<int> {};

TEST_P(TestModelDeviInvalidFrameCount, rejects_every_count_except_one) {
  constexpr int natoms = 1;

  expect_model_devi_frame_error<DP_DeepPotModelDevi>(
      [&](DP_DeepPotModelDevi* model) {
        // Required array pointers are deliberately unusable. The unsupported
        // frame count must be rejected before pointer-range construction;
        // otherwise zero/negative nframes could reach undefined pointer math.
        DP_DeepPotModelDeviCompute2(model, GetParam(), natoms, nullptr, nullptr,
                                    nullptr, nullptr, nullptr, nullptr, nullptr,
                                    nullptr, nullptr, nullptr);
      },
      [](DP_DeepPotModelDevi* model) {
        return DP_DeepPotModelDeviCheckOK(model);
      });
}

INSTANTIATE_TEST_SUITE_P(UnsupportedFrameCounts,
                         TestModelDeviInvalidFrameCount,
                         ::testing::Values(-1, 0, 2));

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
