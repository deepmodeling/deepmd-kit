// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <vector>

#include "common.h"
#include "neighbor_list.h"

/**
 * @brief Test fixture for remap_comm_sendlist.
 *
 * Simulates 10 atoms (6 local + 4 ghost) with 3 virtual (NULL-type) atoms
 * at indices 4, 7, 9.
 *
 *   index:    0  1  2  3   4  5  6   7  8   9
 *   fwd_map:  0  1  2  3  -1  4  5  -1  6  -1
 *
 * Two MPI swaps with asymmetric sendnum/recvnum:
 *   Swap 0: send {2,4,5}, firstrecv=6, recvnum=2  (atoms 6,7)
 *   Swap 1: send {6,7,8,9}, firstrecv=7, recvnum=3 (atoms 7,8,9)
 */
class TestRemapCommSendlist : public ::testing::Test {
 protected:
  std::vector<int> fwd_map = {0, 1, 2, 3, -1, 4, 5, -1, 6, -1};

  std::vector<int> sendlist_swap0 = {2, 4, 5};
  std::vector<int> sendlist_swap1 = {6, 7, 8, 9};
  int* sendlist_ptrs[2];

  int sendnum_arr[2] = {3, 4};
  int recvnum_arr[2] = {2, 3};
  int firstrecv_arr[2] = {6, 7};
  int sendproc_arr[2] = {1, 2};
  int recvproc_arr[2] = {1, 2};

  deepmd::InputNlist lmp_list;

  void SetUp() override {
    sendlist_ptrs[0] = sendlist_swap0.data();
    sendlist_ptrs[1] = sendlist_swap1.data();

    lmp_list.nswap = 2;
    lmp_list.sendnum = sendnum_arr;
    lmp_list.recvnum = recvnum_arr;
    lmp_list.firstrecv = firstrecv_arr;
    lmp_list.sendlist = sendlist_ptrs;
    lmp_list.sendproc = sendproc_arr;
    lmp_list.recvproc = recvproc_arr;
    lmp_list.world = nullptr;
  }
};

// Verify that virtual-atom indices are filtered out from sendlist and that
// remapped indices use the real-atom numbering from fwd_map.
TEST_F(TestRemapCommSendlist, basic) {
  std::vector<std::vector<int>> new_sendlist;
  std::vector<int> new_sendnum;
  std::vector<int> new_recvnum;

  deepmd::remap_comm_sendlist(new_sendlist, new_sendnum, new_recvnum, lmp_list,
                              fwd_map);

  ASSERT_EQ(new_sendlist.size(), 2);
  ASSERT_EQ(new_sendnum.size(), 2);
  ASSERT_EQ(new_recvnum.size(), 2);

  // Swap 0: send {2,4,5} → fwd_map → {2,-1,4} → keep {2,4}
  EXPECT_EQ(new_sendnum[0], 2);
  ASSERT_EQ(new_sendlist[0].size(), 2);
  EXPECT_EQ(new_sendlist[0][0], 2);  // fwd_map[2] = 2
  EXPECT_EQ(new_sendlist[0][1], 4);  // fwd_map[5] = 4

  // Swap 1: send {6,7,8,9} → fwd_map → {5,-1,6,-1} → keep {5,6}
  EXPECT_EQ(new_sendnum[1], 2);
  ASSERT_EQ(new_sendlist[1].size(), 2);
  EXPECT_EQ(new_sendlist[1][0], 5);  // fwd_map[6] = 5
  EXPECT_EQ(new_sendlist[1][1], 6);  // fwd_map[8] = 6
}

// Verify that recvnum is independently computed from the firstrecv range,
// NOT copied from sendnum. This is the key correctness property for MPI.
TEST_F(TestRemapCommSendlist, recvnum_independent_of_sendnum) {
  std::vector<std::vector<int>> new_sendlist;
  std::vector<int> new_sendnum;
  std::vector<int> new_recvnum;

  deepmd::remap_comm_sendlist(new_sendlist, new_sendnum, new_recvnum, lmp_list,
                              fwd_map);

  // Swap 0: firstrecv=6, recvnum=2 → atoms {6,7} → fwd_map {5,-1} → 1 real
  EXPECT_EQ(new_recvnum[0], 1);
  // Swap 1: firstrecv=7, recvnum=3 → atoms {7,8,9} → fwd_map {-1,6,-1} → 1
  EXPECT_EQ(new_recvnum[1], 1);

  // Crucially, new_recvnum != new_sendnum in general:
  // new_sendnum = {2, 2}, new_recvnum = {1, 1}
  EXPECT_NE(new_sendnum[0], new_recvnum[0]);
  EXPECT_NE(new_sendnum[1], new_recvnum[1]);
}

// When fwd_map is an identity mapping (no virtual atoms), sendlist indices
// and counts should be preserved unchanged.
TEST_F(TestRemapCommSendlist, no_virtual_atoms) {
  // Override fwd_map to identity: all atoms are real
  std::vector<int> identity_map = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::vector<std::vector<int>> new_sendlist;
  std::vector<int> new_sendnum;
  std::vector<int> new_recvnum;

  deepmd::remap_comm_sendlist(new_sendlist, new_sendnum, new_recvnum, lmp_list,
                              identity_map);

  // sendnum unchanged
  EXPECT_EQ(new_sendnum[0], 3);
  EXPECT_EQ(new_sendnum[1], 4);
  // recvnum unchanged
  EXPECT_EQ(new_recvnum[0], 2);
  EXPECT_EQ(new_recvnum[1], 3);
  // sendlist unchanged
  ASSERT_EQ(new_sendlist[0].size(), 3);
  EXPECT_EQ(new_sendlist[0][0], 2);
  EXPECT_EQ(new_sendlist[0][1], 4);
  EXPECT_EQ(new_sendlist[0][2], 5);
  ASSERT_EQ(new_sendlist[1].size(), 4);
  EXPECT_EQ(new_sendlist[1][0], 6);
  EXPECT_EQ(new_sendlist[1][1], 7);
  EXPECT_EQ(new_sendlist[1][2], 8);
  EXPECT_EQ(new_sendlist[1][3], 9);
}

// When all atoms in a swap's sendlist are virtual, the resulting sendnum
// should be 0 and the sendlist empty.
TEST_F(TestRemapCommSendlist, all_virtual_in_swap) {
  // Make all atoms virtual
  std::vector<int> all_virtual_map(10, -1);

  std::vector<std::vector<int>> new_sendlist;
  std::vector<int> new_sendnum;
  std::vector<int> new_recvnum;

  deepmd::remap_comm_sendlist(new_sendlist, new_sendnum, new_recvnum, lmp_list,
                              all_virtual_map);

  EXPECT_EQ(new_sendnum[0], 0);
  EXPECT_EQ(new_sendnum[1], 0);
  EXPECT_TRUE(new_sendlist[0].empty());
  EXPECT_TRUE(new_sendlist[1].empty());
  EXPECT_EQ(new_recvnum[0], 0);
  EXPECT_EQ(new_recvnum[1], 0);
}
