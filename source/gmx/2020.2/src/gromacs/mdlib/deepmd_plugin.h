#include "deepmd/DeepPot.h"

typedef struct deepmd_plugin {
    deepmd::DeepPot       nnp;
    std::vector<int >     dtype;
    std::vector<int >     dindex;
    float                 lmd;
    // bool                  useDeepmd;
    bool                  pbc;
    int                   natom;
} deepmd_plugin;

extern deepmd_plugin* deepmdPlugin;
extern bool           useDeepmd;
void init_deepmd();

const float c_dp2gmx = 0.1;
const float e_dp2gmx = 96.48533132;
const float f_dp2gmx = 964.8533132;