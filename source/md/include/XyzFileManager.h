// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef __XyzFileManager_h_wanghan__
#define __XyzFileManager_h_wanghan__

#include <vector>

namespace XyzFileManager {

void read(const std::string& file,
          std::vector<std::string>& atom_name,
          std::vector<std::vector<double> >& posi,
          std::vector<std::vector<double> >& velo,
          std::vector<std::vector<double> >& forc,
          std::vector<double>& boxsize);

void getBoxSize(const std::string& name, std::vector<double>& boxsize);

};  // namespace XyzFileManager

#endif
