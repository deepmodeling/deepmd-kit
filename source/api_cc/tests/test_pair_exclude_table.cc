// SPDX-License-Identifier: LGPL-3.0-or-later
// Torch-free unit tests for the shared pair-exclusion helpers in common.h:
//   - buildPairExcludeTable  (canonical keep table, also used by the libtorch
//     applyPairExclusion* helpers in commonPT.h)
//   - applyPairExcludeNlistVec  (plain-vector nlist filter used by the TF-C-API
//     DeepPotJAX ingestion seam, which cannot depend on libtorch)
// These run in every build (no TF/PT needed) and pin the exclusion convention
// (keep index center_type*(ntypes+1)+neighbour_type; empty table == identity)
// that the compiled backends rely on.
#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "common.h"

namespace {

TEST(PairExcludeTable, empty_exclude_is_identity_table) {
  // No excluded pairs -> empty table -> callers treat as "no exclusion".
  const auto table = deepmd::buildPairExcludeTable(2, {});
  EXPECT_TRUE(table.empty());
}

TEST(PairExcludeTable, symmetric_and_correct_for_one_pair) {
  // ntypes=2, exclude the (0,1) pair. Table is flat (ntypes+1)^2 = 9.
  const int ntypes = 2;
  const int n1 = ntypes + 1;
  const auto table =
      deepmd::buildPairExcludeTable(ntypes, {std::make_pair(0, 1)});
  ASSERT_EQ(table.size(), static_cast<size_t>(n1) * n1);
  // Excluded (both directions): center 0 <-> neighbour 1.
  EXPECT_EQ(table[0 * n1 + 1], 0);  // center type 0, neighbour type 1
  EXPECT_EQ(table[1 * n1 + 0], 0);  // center type 1, neighbour type 0
  // Kept: same-type and virtual-type (ntypes) rows/cols.
  EXPECT_EQ(table[0 * n1 + 0], 1);
  EXPECT_EQ(table[1 * n1 + 1], 1);
  EXPECT_EQ(table[0 * n1 + 2], 1);  // virtual (empty) neighbour type
  EXPECT_EQ(table[2 * n1 + 1], 1);
}

TEST(PairExcludeNlistVec, erases_excluded_neighbours_both_directions) {
  const int ntypes = 2;
  const auto table =
      deepmd::buildPairExcludeTable(ntypes, {std::make_pair(0, 1)});
  // 4 extended atoms; 2 are local centres. Types O,H,H,O.
  const std::vector<int> atype = {0, 1, 1, 0};
  const int nloc = 2;
  const int max_size = 3;
  // centre 0 (type 0): neighbours 1(H),2(H),3(O) -> drop 1,2 keep 3
  // centre 1 (type 1): neighbours 0(O),2(H),3(O) -> drop 0,3 keep 2
  std::vector<std::int64_t> nlist = {1, 2, 3, 0, 2, 3};
  deepmd::applyPairExcludeNlistVec(nlist, atype, table, ntypes, nloc, max_size);
  const std::vector<std::int64_t> expected = {-1, -1, 3, -1, 2, -1};
  EXPECT_EQ(nlist, expected);
}

TEST(PairExcludeNlistVec, empty_table_is_identity) {
  const std::vector<int> atype = {0, 1, 1, 0};
  std::vector<std::int64_t> nlist = {1, 2, 3, 0, 2, 3};
  const std::vector<std::int64_t> before = nlist;
  deepmd::applyPairExcludeNlistVec(nlist, atype, /*type_mask_table=*/{}, 2,
                                   /*nloc=*/2, /*max_size=*/3);
  EXPECT_EQ(nlist, before);  // no-op
}

TEST(PairExcludeNlistVec, keeps_empty_slots_and_nonexcluded_pairs) {
  const int ntypes = 2;
  const auto table =
      deepmd::buildPairExcludeTable(ntypes, {std::make_pair(0, 1)});
  // centre 0 (type 0): neighbours 3(O), -1(empty), 3(O) -> all kept
  const std::vector<int> atype = {0, 1, 1, 0};
  std::vector<std::int64_t> nlist = {3, -1, 3};
  const std::vector<std::int64_t> before = nlist;
  deepmd::applyPairExcludeNlistVec(nlist, atype, table, ntypes, /*nloc=*/1,
                                   /*max_size=*/3);
  EXPECT_EQ(nlist, before);  // O-O kept, -1 stays -1
}

}  // namespace
