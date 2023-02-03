#include "DataModifier.h"
#include "DeepPot.h"
#include "DeepTensor.h"
#include "neighbor_list.h"

struct DP_Nlist {
  DP_Nlist(deepmd::InputNlist& nl);

  deepmd::InputNlist nl;
};

struct DP_DeepPot {
  DP_DeepPot(deepmd::DeepPot& dp);

  deepmd::DeepPot dp;
};

struct DP_DeepPotModelDevi {
  DP_DeepPotModelDevi(deepmd::DeepPotModelDevi& dp);

  deepmd::DeepPotModelDevi dp;
};

struct DP_DeepTensor {
  DP_DeepTensor(deepmd::DeepTensor& dt);

  deepmd::DeepTensor dt;
};

struct DP_DipoleChargeModifier {
  DP_DipoleChargeModifier(deepmd::DipoleChargeModifier& dcm);

  deepmd::DipoleChargeModifier dcm;
};
