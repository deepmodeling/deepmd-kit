// SPDX-License-Identifier: LGPL-3.0-or-later
#include <string>

#include "DataModifier.h"
#include "DeepPot.h"
#include "DeepTensor.h"
#include "neighbor_list.h"

// catch deepmd::deepmd_exception and store it in dp->exception
// return nothing
#define DP_REQUIRES_OK(dp, xx)              \
  try {                                     \
    xx;                                     \
  } catch (deepmd::deepmd_exception & ex) { \
    dp->exception = std::string(ex.what()); \
    return;                                 \
  }

#define DP_NEW_OK(dpcls, xx)                     \
  try {                                          \
    xx;                                          \
  } catch (deepmd::deepmd_exception & ex) {      \
    dpcls* _new_dp = new dpcls;                  \
    _new_dp->exception = std::string(ex.what()); \
    return _new_dp;                              \
  }

struct DP_Nlist {
  DP_Nlist();
  DP_Nlist(deepmd::InputNlist& nl);

  deepmd::InputNlist nl;
  std::string exception;
};

struct DP_DeepPot {
  DP_DeepPot();
  DP_DeepPot(deepmd::DeepPot& dp);

  deepmd::DeepPot dp;
  std::string exception;
  int dfparam;
  int daparam;
  bool aparam_nall;
};

struct DP_DeepPotModelDevi {
  DP_DeepPotModelDevi();
  DP_DeepPotModelDevi(deepmd::DeepPotModelDevi& dp);

  deepmd::DeepPotModelDevi dp;
  std::string exception;
  int dfparam;
  int daparam;
  bool aparam_nall;
};

struct DP_DeepTensor {
  DP_DeepTensor();
  DP_DeepTensor(deepmd::DeepTensor& dt);

  deepmd::DeepTensor dt;
  std::string exception;
};

struct DP_DipoleChargeModifier {
  DP_DipoleChargeModifier();
  DP_DipoleChargeModifier(deepmd::DipoleChargeModifier& dcm);

  deepmd::DipoleChargeModifier dcm;
  std::string exception;
};
