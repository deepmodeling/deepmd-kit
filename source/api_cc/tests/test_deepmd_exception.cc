#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "DeepPot.h"
#include "errors.h"
TEST(TestDeepmdException, deepmdexception) {
  std::string expected_error_message = "DeePMD-kit Error: unittest";
  try {
    throw deepmd::deepmd_exception("unittest");
  } catch (deepmd::deepmd_exception &ex) {
    EXPECT_STREQ(expected_error_message.c_str(), ex.what());
  }
}

TEST(TestDeepmdException, deepmdexception_nofile) {
  ASSERT_THROW(deepmd::DeepPot("_no_such_file.pb"), deepmd::deepmd_exception);
}
