// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace deepmd_test {

// Loader for sidecar reference files written by
// `gen_common.write_expected_ref`.
//
// File format:
//   # auto-generated -- do not edit
//   [case_name_1]
//   array_name_1 N
//   v0
//   v1
//   ...
//   array_name_2 M
//   ...
//
//   [case_name_2]
//   ...
//
// Lines beginning with '#' or empty lines are ignored.
class ExpectedRef {
 public:
  // Parse `path`. Throws std::runtime_error on malformed input.
  void load(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("ExpectedRef: cannot open " + path);
    }
    sections_.clear();
    std::string line;
    std::string current_section;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      if (line.front() == '[' && line.back() == ']') {
        current_section = line.substr(1, line.size() - 2);
        continue;
      }
      // "<key> <count>" header — followed by `count` numeric lines.
      if (current_section.empty()) {
        throw std::runtime_error("ExpectedRef: array '" + line +
                                 "' before any [section]");
      }
      std::istringstream iss(line);
      std::string key;
      std::size_t n = 0;
      if (!(iss >> key >> n)) {
        throw std::runtime_error("ExpectedRef: bad header line: " + line);
      }
      std::vector<double> values;
      values.reserve(n);
      for (std::size_t i = 0; i < n; ++i) {
        if (!std::getline(in, line)) {
          throw std::runtime_error("ExpectedRef: unexpected EOF in '" + key +
                                   "'");
        }
        values.push_back(std::stod(line));
      }
      sections_[current_section][key] = std::move(values);
    }
  }

  // Get array of `key` from `case_name`. Throws if missing.
  template <typename T = double>
  std::vector<T> get(const std::string& case_name,
                     const std::string& key) const {
    auto sit = sections_.find(case_name);
    if (sit == sections_.end()) {
      throw std::runtime_error("ExpectedRef: missing case '" + case_name + "'");
    }
    auto kit = sit->second.find(key);
    if (kit == sit->second.end()) {
      throw std::runtime_error("ExpectedRef: missing array '" + key +
                               "' in case '" + case_name + "'");
    }
    return std::vector<T>(kit->second.begin(), kit->second.end());
  }

 private:
  std::map<std::string, std::map<std::string, std::vector<double>>> sections_;
};

}  // namespace deepmd_test
