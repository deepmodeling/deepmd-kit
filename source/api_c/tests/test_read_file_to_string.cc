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
