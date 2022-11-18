#include "neighbor_list.h"
#include "DeepPot.h"

struct DP_Nlist {
  DP_Nlist(deepmd::InputNlist& nl);

  deepmd::InputNlist nl;
};

struct DP_DeepPot {
  DP_DeepPot(deepmd::DeepPot& dp);

  deepmd::DeepPot dp;
};