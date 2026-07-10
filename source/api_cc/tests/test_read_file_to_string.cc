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

#include "common.h"
TEST(TestReadFileToString, readfiletostring) {
#ifndef BUILD_TENSORFLOW
  GTEST_SKIP() << "Skip because TensorFlow support is not enabled.";
#endif
  std::string file_content;
  deepmd::read_file_to_string("../../tests/infer/deeppot.txt", file_content);

  std::string file_name_2 = "../../tests/infer/deeppot.txt";
  std::stringstream buffer;
  std::ifstream file_txt(file_name_2);
  buffer << file_txt.rdbuf();
  std::string expected_out_string = buffer.str();
  EXPECT_STREQ(expected_out_string.c_str(), file_content.c_str());
}

// Regression test for issue #5620: DP_ReadFileToChar2 must preserve
// exact file bytes including trailing whitespace, and the reported size
// must match the allocated buffer.
TEST(TestReadFileToString, readfiletostring_exact_bytes) {
  // Write a temporary file whose content ends with " \n" (trailing
  // whitespace).  The bug was that string_to_char trimmed this
  // whitespace but the size still reported the original length,
  // causing an over-read when the C++ wrapper reconstructed a string
  // with the reported size.
  std::string tmp_file = "test_readfile_exact_bytes.txt";
  std::string content = "hello world \n";  // ends with space + newline
  {
    std::ofstream ofs(tmp_file);
    ofs << content;
    ofs.close();
  }

  // Read through the C++ wrapper which calls DP_ReadFileToChar2
  std::string file_content;
  deepmd::read_file_to_string(tmp_file, file_content);

  // The full content must be preserved byte-for-byte.
  EXPECT_EQ(content.size(), file_content.size())
      << "Size mismatch: the file has " << content.size()
      << " bytes but read_file_to_string returned " << file_content.size();
  EXPECT_EQ(content, file_content)
      << "Content mismatch: trailing whitespace was not preserved";

  // Clean up the temporary file.
  std::remove(tmp_file.c_str());
}
