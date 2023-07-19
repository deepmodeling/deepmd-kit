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
void split(const std::string& in,
           const std::string& delimiter,
           std::vector<std::string>& out);
}  // namespace StringOperation

#endif
