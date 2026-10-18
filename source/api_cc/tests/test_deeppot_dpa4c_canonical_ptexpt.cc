// SPDX-License-Identifier: LGPL-3.0-or-later
// The device-resident entry points of the compressed DPA4C archives frozen
// with the compact canonical lower, through deepmd::DeepPot and
// deepmd::DeepSpin (see dpa4c_canonical_test_common.h for the cases).
#include <gtest/gtest.h>

#include "DeepPot.h"
#include "DeepSpin.h"
#include "test_utils.h"
// The shared cases follow the helpers of test_utils.h, which they call.
#include "dpa4c_canonical_test_common.h"

namespace deepmd_test {
namespace dpa4c_canonical {

using Models =
    ::testing::Types<Potential<deepmd::DeepPot>, Spin<deepmd::DeepSpin>>;
INSTANTIATE_TYPED_TEST_SUITE_P(CXX,
                               TestInferDpa4cCanonicalPtExpt,
                               Models,
                               CaseNames);

}  // namespace dpa4c_canonical
}  // namespace deepmd_test
