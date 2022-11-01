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
    ENERGYTYPE* energy,
    VALUETYPE* force,
    VALUETYPE* virial
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_(cell, cell+9);
    ENERGYTYPE e;
    std::vector<VALUETYPE> f, v;

    dp->dp.compute(e, f, v, coord_, atype_, cell_);
    // copy from C++ vectors to C arrays
    *energy = e;
    std::copy(f.begin(), f.end(), force);
    std::copy(v.begin(), v.end(), virial);
}

}