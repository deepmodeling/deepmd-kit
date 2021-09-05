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

DP_ComputeResult DP_DeepPotCompute (DP_DeepPot* dp,
                        const int natom,
                        const VALUETYPE* coord,
                        const int* atype,
                        const VALUETYPE* cell) {
    std::vector<VALUETYPE> coord_(coord, coord+natom*3);
    std::vector<int> atype_(atype, atype+natom);
    std::vector<VALUETYPE> cell_(cell, cell+9);
    double e;
    std::vector<VALUETYPE> f, v;

    dp->dp.compute(e, f, v, coord_, atype_, cell_);

    DP_ComputeResult result = {
        e, &f[0], &v[0]
    };
    return result;
}

}