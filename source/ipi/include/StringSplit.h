// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef __StringSplit_h_wanghan__
#define __StringSplit_h_wanghan__

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace StringOperation {
void split(const std::string& in, std::vector<std::string>& out);
}

void StringOperation::split(const std::string& in,
                            std::vector<std::string>& out) {
  std::istringstream iss(in);
  out.clear();

  do {
    std::string sub;
    iss >> sub;
    out.push_back(sub);
    // std::vector<std::string > tokens;
    // tokens.push_back (" ");
    // tokens.push_back ("\t");
    // std::copy(std::istream_iterator<std::string>(iss),
    // 	    std::istream_iterator<std::string>(),
    // 	    std::back_inserter<std::vector<std::string> >(tokens));
  } while (iss);

  out.pop_back();
}

#endif
