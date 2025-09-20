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
  } catch (deepmd::hpp::deepmd_exception &ex) {
    EXPECT_STREQ(expected_error_message.c_str(), ex.what());
  }
}

TEST(TestDeepmdException, deepmdexception_nofile) {
  ASSERT_THROW(deepmd::hpp::DeepPot("_no_such_file.pb"),
               deepmd::hpp::deepmd_exception);
}
