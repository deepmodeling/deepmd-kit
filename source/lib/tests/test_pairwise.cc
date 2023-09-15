// SPDX-License-Identifier: LGPL-3.0-or-later
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

TEST(TestPairwiseMap, pairwise_map) {
  std::vector<int> idxs = {1, 1, 1, 0, 0, 2, 2, 2, 3, 3, 0, 1};
  std::vector<std::vector<int>> fragments;
  deepmd::group_atoms_cpu(fragments, idxs);
  std::vector<int> forward_qm_map, backward_qm_map, forward_qmmm_map,
      backward_qmmm_map;
  int nloc_qm, nloc_qmmm, nall_qm, nall_qmmm;
  deepmd::dprc_pairwise_map_cpu(
      forward_qm_map, backward_qm_map, forward_qmmm_map, backward_qmmm_map,
      nloc_qm, nloc_qmmm, nall_qm, nall_qmmm, fragments, 10, 12);
  ASSERT_THAT(forward_qm_map, testing::ElementsAre(3, 4, 10));
  ASSERT_THAT(backward_qm_map, testing::ElementsAre(-1, -1, -1, 0, 1, -1, -1,
                                                    -1, -1, -1, 2, -1));
  ASSERT_THAT(forward_qmmm_map,
              testing::ElementsAre(3, 4, 0, 1, 2, 10, 11, 3, 4, 5, 6, 7, 10, -1,
                                   3, 4, 8, 9, -1, 10, -1));
  ASSERT_THAT(backward_qmmm_map,
              testing::ElementsAre(2, 3, 4, 0, 1, -1, -1, -1, -1, -1, 5, 6, -1,
                                   -1, -1, 0, 1, 2, 3, 4, -1, -1, 5, -1, -1, -1,
                                   -1, 0, 1, -1, -1, -1, 2, 3, 5, -1));
  EXPECT_EQ(nloc_qm, 2);
  EXPECT_EQ(nloc_qmmm, 5);
  EXPECT_EQ(nall_qm, 3);
  EXPECT_EQ(nall_qmmm, 7);
}
