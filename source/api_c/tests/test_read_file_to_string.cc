// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "c_api.h"
#include "deepmd.hpp"
TEST(TestReadFileToString, readfiletostring) {
#ifndef BUILD_TENSORFLOW
  GTEST_SKIP() << "Skip because TensorFlow support is not enabled.";
#endif
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
#ifndef BUILD_TENSORFLOW
  GTEST_SKIP() << "Skip because TensorFlow support is not enabled.";
#endif
  std::string file_content;
  EXPECT_THROW(
      {
        deepmd::hpp::read_file_to_string(
            "12345_no_such_file_do_not_create_this_file", file_content);
      },
      deepmd::hpp::deepmd_exception);
}

TEST(TestReadFileToString, c_api_preserves_exact_bytes) {
  const std::string file_name = "test_read_file_to_char_exact_bytes.bin";
  const std::string expected("hello\0world \n", 13);
  {
    std::ofstream output(file_name, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(expected.data(), expected.size());
  }

  int size = 0;
  std::unique_ptr<const char, decltype(&DP_DeleteChar)> content(
      DP_ReadFileToChar2(file_name.c_str(), &size), DP_DeleteChar);

  ASSERT_NE(content, nullptr);
  ASSERT_GE(size, 0);
  EXPECT_EQ(size, expected.size());
  EXPECT_EQ(std::string(content.get(), size), expected);

  EXPECT_EQ(std::remove(file_name.c_str()), 0);
}

TEST(TestReadFileToString, legacy_c_api_preserves_trailing_whitespace) {
  const std::string file_name = "test_read_file_to_char_whitespace.txt";
  const std::string expected = "hello world \n";
  {
    std::ofstream output(file_name, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(expected.data(), expected.size());
  }

  std::unique_ptr<const char, decltype(&DP_DeleteChar)> content(
      DP_ReadFileToChar(file_name.c_str()), DP_DeleteChar);

  ASSERT_NE(content, nullptr);
  EXPECT_EQ(std::string(content.get()), expected);

  EXPECT_EQ(std::remove(file_name.c_str()), 0);
}
