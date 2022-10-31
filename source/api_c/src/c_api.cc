#include "c_api.h"

#include <vector>
#include "c_api_internal.h"
#include "DeepPot.h"

extern "C" {

DP_DeepPot::DP_DeepPot(deepmd::DeepPot& dp)
    : dp(dp) {}

DP_DeepPot* DP_NewDeepPot(const char* c_model) {
    std::string model(c_model);
    deepmd::DeepPot dp(model);
    DP_DeepPot* new_dp = new DP_DeepPot(dp);
    return new_dp;
}

void DP_DeepPotCompute (
    DP_DeepPot* dp,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    const ENERGYTYPE* energy,
    const VALUETYPE* force,
    const VALUETYPE* virial
    ) {
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natom);
    std::vector<VALUETYPE> cell_(cell, cell+9);
    energy = new double;
    force = new std::vector<VALUETYPE>;
    virial = new std::vector<VALUETYPE>;

    dp->dp.compute(e, f, v, coord_, atype_, cell_);
}

}