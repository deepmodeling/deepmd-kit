#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "pairwise.h"

TEST(TestGroupAtoms, group_atoms) {
  std::vector<int> idxs = {1, 1, 1, 0, 0, 2, 2, 2, 3, 3, 0, 1};
  // ((3,4,10), (0,1,2,11), (5,6,7), (8,9))
  std::vector<std::vector<int>> fragments;
  deepmd::group_atoms_cpu(fragments, idxs);
  EXPECT_EQ(fragments.size(), 4);
  ASSERT_THAT(fragments[0], testing::ElementsAre(3, 4, 10));
  ASSERT_THAT(fragments[1], testing::ElementsAre(0, 1, 2, 11));
  ASSERT_THAT(fragments[2], testing::ElementsAre(5, 6, 7));
  ASSERT_THAT(fragments[3], testing::ElementsAre(8, 9));
}
