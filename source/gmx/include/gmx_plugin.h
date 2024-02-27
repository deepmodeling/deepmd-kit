// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef _GMX_PLUGIN_H_
#define _GMX_PLUGIN_H_
#ifdef DP_USE_CXX_API
#include "DeepPot.h"
namespace deepmd_compat = deepmd;
#else
#ifdef DP_GMX_PLUGIN_INTERNAL
#include "deepmd.hpp"
#else
#include "deepmd/deepmd.hpp"
#endif
namespace deepmd_compat = deepmd::hpp;
#endif

namespace deepmd {

class DeepmdPlugin {
 public:
  DeepmdPlugin();
  DeepmdPlugin(char*);
  ~DeepmdPlugin();
  void init_from_json(char*);
  deepmd_compat::DeepPot* nnp;
  std::vector<int> dtype;
  std::vector<int> dindex;
  bool pbc;
  float lmd;
  int natom;
};

}  // namespace deepmd

const float c_dp2gmx = 0.1;
const float e_dp2gmx = 96.48533132;
const float f_dp2gmx = 964.8533132;
#endif
