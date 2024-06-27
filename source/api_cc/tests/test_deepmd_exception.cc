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

#include "DataModifier.h"
#include "DeepPot.h"
#include "DeepTensor.h"
#include "errors.h"
TEST(TestDeepmdException, deepmdexception) {
  std::string expected_error_message = "DeePMD-kit Error: unittest";
  try {
    throw deepmd::deepmd_exception("unittest");
  } catch (deepmd::deepmd_exception &ex) {
    EXPECT_STREQ(expected_error_message.c_str(), ex.what());
  }
}

TEST(TestDeepmdException, deepmdexception_nofile_deeppot) {
  ASSERT_THROW(deepmd::DeepPot("_no_such_file.pb"), deepmd::deepmd_exception);
}

TEST(TestDeepmdException, deepmdexception_nofile_deeppot_pt) {
  ASSERT_THROW(deepmd::DeepPot("_no_such_file.pth"), deepmd::deepmd_exception);
}

TEST(TestDeepmdException, deepmdexception_nofile_deeppotmodeldevi) {
  ASSERT_THROW(
      deepmd::DeepPotModelDevi({"_no_such_file.pb", "_no_such_file.pb"}),
      deepmd::deepmd_exception);
}

TEST(TestDeepmdException, deepmdexception_nofile_deeptensor) {
  ASSERT_THROW(deepmd::DeepTensor("_no_such_file.pb"),
               deepmd::deepmd_exception);
}

TEST(TestDeepmdException, deepmdexception_nofile_dipolechargemodifier) {
  ASSERT_THROW(deepmd::DipoleChargeModifier("_no_such_file.pb"),
               deepmd::deepmd_exception);
}
