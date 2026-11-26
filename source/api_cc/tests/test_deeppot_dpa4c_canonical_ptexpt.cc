// SPDX-License-Identifier: LGPL-3.0-or-later
// The device-resident entry points of the archives frozen with the compact
// canonical lower, through deepmd::DeepPot and deepmd::DeepSpin (see
// dpa4c_canonical_test_common.h for the cases).
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "DeepPot.h"
#include "DeepSpin.h"
#include "canonical_graph.h"
#include "test_utils.h"
// The shared cases follow the helpers of test_utils.h, which they call.
#include "dpa4c_canonical_test_common.h"

namespace deepmd_test {
namespace dpa4c_canonical {

TEST(CanonicalNodeBucket, reuses_logical_extent_after_large_shrink) {
  deepmd::CanonicalNodeBucket bucket;
  for (const auto& item : std::vector<std::pair<int, int>>{
           {1536, 1536}, {513, 768}, {511, 768}, {513, 768}, {511, 768}}) {
    EXPECT_EQ(bucket.update(item.first, 4096), item.second);
  }
  EXPECT_EQ(bucket.update(511, 600), 512);
  EXPECT_EQ(bucket.update(0, 600), 0);
}

using Models = ::testing::Types<Potential<deepmd::DeepPot>,
                                Spin<deepmd::DeepSpin>,
                                Dpa4Potential<deepmd::DeepPot>,
                                Dpa4Spin<deepmd::DeepSpin>>;
INSTANTIATE_TYPED_TEST_SUITE_P(CXX,
                               TestInferDpa4cCanonicalPtExpt,
                               Models,
                               CaseNames);

using HaloModels = ::testing::Types<MultiblockPotential<deepmd::DeepPot>,
                                    MultiblockSpin<deepmd::DeepSpin>>;
INSTANTIATE_TYPED_TEST_SUITE_P(CXX,
                               TestInferDpa4cCanonicalHalo,
                               HaloModels,
                               CaseNames);

}  // namespace dpa4c_canonical
}  // namespace deepmd_test
