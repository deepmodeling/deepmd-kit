// SPDX-License-Identifier: LGPL-3.0-or-later
// Backend-agnostic unit tests for deepmd::NeighborListData and the
// deepmd::convert_nlist helper.  Exercises edge cases (empty rows, empty
// nlist) that surfaced as latent libstdc++-debug-mode UB assertions in
// production code paths.
#include <gtest/gtest.h>

#include <vector>

#include "common.h"
#ifdef BUILD_PYTORCH
#include "commonPT.h"
#endif
#include "neighbor_list.h"

namespace deepmd {

// Build a NeighborListData with @p nloc local atoms and zero neighbors per
// atom.  Realistic under aggressive spatial partitioning where a subdomain's
// every local atom has no neighbors within cutoff.
TEST(TestNeighborListData, MakeInlistEmptyRows) {
  NeighborListData data;
  const int nloc = 4;
  data.ilist.resize(nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    data.ilist[ii] = ii;
  }
  data.jlist.resize(nloc);  // every row default-constructed to empty

  // Must not trigger UB ('&vec[0]' on empty vector) under libstdc++ debug.
  InputNlist inlist;
  ASSERT_NO_THROW(data.make_inlist(inlist));

  EXPECT_EQ(inlist.inum, nloc);
  ASSERT_NE(inlist.numneigh, nullptr);
  ASSERT_NE(inlist.firstneigh, nullptr);
  for (int ii = 0; ii < nloc; ++ii) {
    EXPECT_EQ(inlist.numneigh[ii], 0);
  }
}

TEST(TestNeighborListData, MakeInlistMixedEmptyAndNonemptyRows) {
  NeighborListData data;
  data.ilist = {0, 1, 2};
  data.jlist.resize(3);
  // row 0: empty (legitimate edge case)
  data.jlist[1] = {7, 8};
  // row 2: empty
  InputNlist inlist;
  ASSERT_NO_THROW(data.make_inlist(inlist));
  EXPECT_EQ(inlist.numneigh[0], 0);
  EXPECT_EQ(inlist.numneigh[1], 2);
  EXPECT_EQ(inlist.numneigh[2], 0);
  // Only the populated row's firstneigh should be dereferenced.
  EXPECT_EQ(inlist.firstneigh[1][0], 7);
  EXPECT_EQ(inlist.firstneigh[1][1], 8);
}

// convert_nlist(jagged) must not dereference an empty row when populating
// firstneigh.
TEST(TestNeighborListData, ConvertNlistEmptyRows) {
  std::vector<std::vector<int>> input = {{}, {}, {}};  // all rows empty
  std::vector<int> ilist(input.size()), numneigh(input.size());
  std::vector<int*> firstneigh(input.size());
  InputNlist out(static_cast<int>(input.size()), ilist.data(), numneigh.data(),
                 firstneigh.data());
  ASSERT_NO_THROW(convert_nlist(out, input));
  EXPECT_EQ(out.inum, 3);
  for (int ii = 0; ii < 3; ++ii) {
    EXPECT_EQ(out.numneigh[ii], 0);
    // firstneigh[ii] may be vector::data()'s sentinel or nullptr — must not
    // be dereferenced when numneigh[ii] == 0.
  }
}

// copy_from_nlist must not dereference an empty source row even when
// memcpy size is 0.  Regression test for the same UB pattern.
TEST(TestNeighborListData, CopyFromNlistEmptyRows) {
  // Build an InputNlist with all empty rows.
  const int nloc = 4;
  std::vector<int> src_ilist(nloc), src_numneigh(nloc, 0);
  std::vector<int*> src_firstneigh(nloc, nullptr);
  for (int ii = 0; ii < nloc; ++ii) {
    src_ilist[ii] = ii;
  }
  InputNlist src(nloc, src_ilist.data(), src_numneigh.data(),
                 src_firstneigh.data());
  src.mask = ~0;  // identity mask, must not be applied to absent neighbors

  NeighborListData data;
  ASSERT_NO_THROW(data.copy_from_nlist(src));
  EXPECT_EQ(static_cast<int>(data.ilist.size()), nloc);
  EXPECT_EQ(static_cast<int>(data.jlist.size()), nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    EXPECT_TRUE(data.jlist[ii].empty());
  }
}

// copy_from_nlist with an empty source list (inum == 0) must not
// dereference '&ilist[0]' on the empty target ilist.
TEST(TestNeighborListData, CopyFromNlistInumZero) {
  InputNlist src;
  src.inum = 0;
  src.ilist = nullptr;
  src.numneigh = nullptr;
  src.firstneigh = nullptr;
  src.mask = ~0;

  NeighborListData data;
  ASSERT_NO_THROW(data.copy_from_nlist(src));
  EXPECT_TRUE(data.ilist.empty());
  EXPECT_TRUE(data.jlist.empty());
}

// Round-trip: convert_nlist(jagged) → copy_from_nlist → make_inlist
// must preserve both empty and non-empty rows without UB.
TEST(TestNeighborListData, RoundTripWithEmptyRows) {
  std::vector<std::vector<int>> input = {{}, {3, 4}, {}, {5}};
  std::vector<int> ilist(input.size()), numneigh(input.size());
  std::vector<int*> firstneigh(input.size());
  InputNlist src(static_cast<int>(input.size()), ilist.data(), numneigh.data(),
                 firstneigh.data());
  ASSERT_NO_THROW(convert_nlist(src, input));

  NeighborListData data;
  ASSERT_NO_THROW(data.copy_from_nlist(src));
  EXPECT_EQ(static_cast<int>(data.jlist.size()), 4);
  EXPECT_TRUE(data.jlist[0].empty());
  EXPECT_EQ(data.jlist[1], (std::vector<int>{3, 4}));
  EXPECT_TRUE(data.jlist[2].empty());
  EXPECT_EQ(data.jlist[3], (std::vector<int>{5}));

  InputNlist out;
  ASSERT_NO_THROW(data.make_inlist(out));
  EXPECT_EQ(out.numneigh[0], 0);
  EXPECT_EQ(out.numneigh[1], 2);
  EXPECT_EQ(out.numneigh[2], 0);
  EXPECT_EQ(out.numneigh[3], 1);
}

#ifdef BUILD_PYTORCH
TEST(TestEdgeTensorPack, CreateEdgeTensorsUsesRowCenters) {
  const torch::Device device(torch::kCPU);
  const std::vector<std::vector<int>> nlist = {{0}, {1}};
  const std::vector<int> centers = {2, 0};
  const std::vector<double> coord = {
      0.0, 0.0, 0.0,  // atom 0
      1.0, 0.0, 0.0,  // atom 1
      2.0, 0.0, 0.0,  // atom 2
  };
  const std::vector<std::int64_t> mapping = {0, 1, 2};

  const auto pack = createEdgeTensors(nlist, coord, mapping, 3, 3, device,
                                      /*with_geometry=*/true, &centers);

  ASSERT_EQ(pack.edge_index.size(1), 4);
  EXPECT_EQ(pack.edge_index.select(0, 0).select(0, 0).item<int64_t>(), 0);
  EXPECT_EQ(pack.edge_index.select(0, 1).select(0, 0).item<int64_t>(), 2);
  EXPECT_EQ(pack.edge_index_ext.select(0, 1).select(0, 0).item<int64_t>(), 2);
  EXPECT_DOUBLE_EQ(pack.edge_vec.select(0, 0).select(0, 0).item<double>(),
                   -2.0);
  EXPECT_EQ(pack.edge_index.select(0, 0).select(0, 1).item<int64_t>(), 1);
  EXPECT_EQ(pack.edge_index.select(0, 1).select(0, 1).item<int64_t>(), 0);
  EXPECT_DOUBLE_EQ(pack.edge_vec.select(0, 1).select(0, 0).item<double>(), 1.0);

  GraphTensorPack graph;
  graph.edge_index = pack.edge_index;
  graph.edge_vec = pack.edge_vec;
  graph.edge_mask = pack.edge_mask;
  buildGraphCSR(graph, 3);
  EXPECT_TRUE(torch::equal(graph.destination_order,
                           torch::tensor({1, 0, 2, 3}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(graph.destination_row_ptr,
                           torch::tensor({0, 1, 1, 2}, torch::kInt64)));

  canonicalizeGraphPayload(graph, 3);
  EXPECT_TRUE(torch::equal(graph.destination_order,
                           torch::tensor({0, 1, 2, 3}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(graph.edge_index.select(0, 1),
                           torch::tensor({0, 2, 0, 0}, torch::kInt64)));
}

TEST(TestEdgeTensorPack, CompactFiltersSkinTopologyAndAppendsDummies) {
  const torch::Device device(torch::kCPU);
  const std::vector<std::vector<int>> nlist = {{1, 2}, {0}};
  const std::vector<double> coord = {
      0.0, 0.0, 0.0,  // atom 0
      0.5, 0.0, 0.0,  // atom 1, inside cutoff
      2.0, 0.0, 0.0,  // atom 2, skin-only ghost
  };
  const std::vector<std::int64_t> mapping = {0, 1, 0};

  const auto skin_topology =
      createEdgeTensors(nlist, coord, mapping, 2, 3, device,
                        /*with_geometry=*/false);
  ASSERT_EQ(skin_topology.edge_index.size(1), 3);
  ASSERT_EQ(skin_topology.edge_index_ext.size(1), 3);
  EXPECT_FALSE(skin_topology.edge_vec.defined());
  EXPECT_FALSE(skin_topology.edge_mask.defined());

  const auto coord_tensor =
      torch::tensor(coord, torch::TensorOptions().dtype(torch::kFloat64))
          .view({1, 3, 3});
  const auto compact =
      compactEdgeTensors(skin_topology.edge_index, skin_topology.edge_index_ext,
                         coord_tensor, /*rcut=*/1.0);

  ASSERT_EQ(compact.edge_index.size(1), 4);
  ASSERT_EQ(compact.edge_index_ext.size(1), 4);
  ASSERT_EQ(compact.edge_vec.size(0), 4);
  ASSERT_EQ(compact.edge_mask.size(0), 4);
  EXPECT_EQ(compact.edge_mask.sum().item<int64_t>(), 2);
  EXPECT_FALSE(compact.edge_mask.select(0, 2).item<bool>());
  EXPECT_FALSE(compact.edge_mask.select(0, 3).item<bool>());

  GraphTensorPack graph;
  graph.edge_index = compact.edge_index;
  graph.edge_mask = compact.edge_mask;
  buildGraphCSR(graph, 2, /*destination_sorted=*/true);
  EXPECT_TRUE(torch::equal(graph.destination_order,
                           torch::tensor({0, 1, 2, 3}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(graph.destination_row_ptr,
                           torch::tensor({0, 1, 2}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(graph.source_order,
                           torch::tensor({1, 0, 2, 3}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(graph.source_row_ptr,
                           torch::tensor({0, 1, 2}, torch::kInt64)));
}

TEST(TestEdgeTensorPack, CanonicalizeGraphPayloadIsStableWithinDestination) {
  GraphTensorPack graph;
  graph.edge_index = torch::tensor({{2, 1, 0, 0}, {1, 0, 0, 0}},
                                   torch::TensorOptions().dtype(torch::kInt64));
  graph.edge_vec =
      torch::arange(12, torch::TensorOptions().dtype(torch::kFloat64))
          .reshape({4, 3});
  graph.edge_mask = torch::tensor({true, true, true, false});

  canonicalizeGraphPayload(graph, 3);

  EXPECT_TRUE(torch::equal(graph.edge_index.select(0, 0),
                           torch::tensor({1, 0, 2, 0}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(graph.destination_order,
                           torch::tensor({0, 1, 2, 3}, torch::kInt64)));
}

#endif

}  // namespace deepmd
