// SPDX-License-Identifier: LGPL-3.0-or-later
// The device-resident entry points of the archives frozen with the compact
// canonical lower, through deepmd::DeepPot and deepmd::DeepSpin (see
// dpa4c_canonical_test_common.h for the cases).
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "../../op/pt/dpa4c/node_type_summary.h"
#include "DeepPot.h"
#include "DeepSpin.h"
#include "canonical_graph.h"
#include "test_utils.h"
// The shared cases follow the helpers of test_utils.h, which they call.
#include "dpa4c_canonical_test_common.h"

namespace deepmd_test {
namespace dpa4c_canonical {

TEST(CanonicalNodeTypes, padding_preserves_dispatch_and_ghosts_contribute) {
  const auto dispatch = [](std::vector<long> types) {
    unsigned bitmap[1] = {};
    bool same = true;
    for (long type : types) {
      same &= deepmd::dpa4c::scan_node_type(type, types.front(), bitmap);
    }
    return std::make_pair(same, bitmap[0]);
  };
  EXPECT_EQ(dispatch({1, 1}), std::make_pair(true, 2u));
  EXPECT_EQ(dispatch({1, 1, -1}), dispatch({1, 1}));
  EXPECT_EQ(dispatch({1, 0, -1}), std::make_pair(false, 3u));
}

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
