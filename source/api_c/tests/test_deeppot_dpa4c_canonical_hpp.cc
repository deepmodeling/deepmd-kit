// SPDX-License-Identifier: LGPL-3.0-or-later
// The device-resident entry points of the compressed DPA4C archives frozen
// with the compact canonical lower, through deepmd::hpp::DeepPot and
// deepmd::hpp::DeepSpin and so through DP_DPA4ComputeCanonicalGraphGPU and
// DP_DPA4SpinComputeCanonicalGraphGPU of the C API (see
// source/api_cc/tests/dpa4c_canonical_test_common.h for the cases).
#include <gtest/gtest.h>

#include "deepmd.hpp"
#include "test_utils.h"
// The shared cases follow the helpers of test_utils.h, which they call.
#include "../../api_cc/tests/dpa4c_canonical_test_common.h"

namespace deepmd_test {
namespace dpa4c_canonical {

using Models = ::testing::Types<Potential<deepmd::hpp::DeepPot>,
                                Spin<deepmd::hpp::DeepSpin>>;
INSTANTIATE_TYPED_TEST_SUITE_P(C,
                               TestInferDpa4cCanonicalPtExpt,
                               Models,
                               CaseNames);

}  // namespace dpa4c_canonical
}  // namespace deepmd_test
