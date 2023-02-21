#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "common.h"
TEST(TestReadFileToString, readfiletostring) {
  std::string file_name = "../../tests/infer/deeppot.pbtxt";
  deepmd::convert_pbtxt_to_pb(file_name, "deeppot.pb");
  std::string file_content;
  for(int ii=0; ii<200000; ii++) file_content.append("0");
  deepmd::read_file_to_string("deeppot.pb", file_content);
  remove("deeppot.pb");

  std::string file_name_2 = "../../tests/infer/deeppot.txt";
  std::stringstream buffer;
  std::ifstream file_txt(file_name_2);
  buffer << file_txt.rdbuf();
  std::string expected_out_string = buffer.str();
  EXPECT_STREQ(expected_out_string.c_str(), file_content.c_str());
}