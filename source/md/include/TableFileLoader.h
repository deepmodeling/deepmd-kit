// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef __TableFileLoader_h_wanghan__
#define __TableFileLoader_h_wanghan__

#include <fstream>
#include <vector>

class TableFileLoader {
 public:
  unsigned getNumbColumns();

 public:
  TableFileLoader(const char* file);
  void reinit(const char* file);
  void setColumns(const std::vector<unsigned>& cols);
  void setEvery(const unsigned every);

 public:
  void loadAll(std::vector<std::vector<double> >& data);
  bool loadLine(std::vector<double>& data);

 private:
  std::ifstream data;
  std::string file;
  unsigned count_read;
  unsigned every;
  std::vector<unsigned> inter_cols;
};

#endif
