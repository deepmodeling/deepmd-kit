// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "deepmd.hpp"
TEST(TestReadFileToString, readfiletostring) {
  std::string file_content;
  deepmd::hpp::read_file_to_string("../../tests/infer/deeppot.txt",
                                   file_content);

  std::string file_name_2 = "../../tests/infer/deeppot.txt";
  std::stringstream buffer;
  std::ifstream file_txt(file_name_2);
  buffer << file_txt.rdbuf();
  std::string expected_out_string = buffer.str();
  EXPECT_STREQ(expected_out_string.c_str(), file_content.c_str());
}

TEST(TestReadFileToString, readfiletostringerr) {
  std::string file_content;
  EXPECT_THROW(
      {
        deepmd::hpp::read_file_to_string(
            "12345_no_such_file_do_not_create_this_file", file_content);
      },
      deepmd::hpp::deepmd_exception);
}
