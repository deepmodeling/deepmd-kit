// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <vector>

#include "common.h"
#include "neighbor_list.h"

using namespace deepmd;

class TestSelectRealAtomsSendlist : public ::testing::Test {
 protected:
  std::vector<std::vector<int>> new_sendlist;
  std::vector<int> new_sendnum;
  std::vector<int> new_recvnum;
};

// Normal case: no virtual atoms, all fwd_map >= 0
TEST_F(TestSelectRealAtomsSendlist, AllReal) {
  // 2 swaps, 6 atoms total, no virtual atoms
  int nswap = 2;
  int sendnum_arr[] = {3, 2};
  int recvnum_arr[] = {3, 2};
  int sendlist0[] = {0, 2, 4};
  int sendlist1[] = {1, 3};
  int* sendlist_arr[] = {sendlist0, sendlist1};

  InputNlist inlist;
  inlist.nswap = nswap;
  inlist.sendnum = sendnum_arr;
  inlist.recvnum = recvnum_arr;
  inlist.sendlist = sendlist_arr;

  // Identity mapping: all atoms are real
  std::vector<int> fwd_map = {0, 1, 2, 3, 4, 5};

  deepmd::select_real_atoms_sendlist(new_sendlist, new_sendnum, new_recvnum,
                                     inlist, fwd_map);

  EXPECT_EQ(new_sendlist.size(), 2);
  // swap 0: all indices mapped identically
  EXPECT_EQ(new_sendnum[0], 3);
  EXPECT_EQ(new_sendlist[0], (std::vector<int>{0, 2, 4}));
  // swap 1
  EXPECT_EQ(new_sendnum[1], 2);
  EXPECT_EQ(new_sendlist[1], (std::vector<int>{1, 3}));
  // recvnum should equal sendnum
  EXPECT_EQ(new_recvnum[0], 3);
  EXPECT_EQ(new_recvnum[1], 2);
}

// With virtual atoms: some fwd_map entries are -1
TEST_F(TestSelectRealAtomsSendlist, WithVirtualAtoms) {
  // 7 atoms: indices 0-5 are real (O,H,H,O,H,H), index 6 is virtual (NULL)
  // fwd_map: virtual atom 6 -> -1, others remapped
  int nswap = 2;
  int sendnum_arr[] = {4, 3};
  int recvnum_arr[] = {4, 3};
  int sendlist0[] = {0, 2, 4, 6};  // includes virtual atom 6
  int sendlist1[] = {1, 3, 6};     // includes virtual atom 6
  int* sendlist_arr[] = {sendlist0, sendlist1};

  InputNlist inlist;
  inlist.nswap = nswap;
  inlist.sendnum = sendnum_arr;
  inlist.recvnum = recvnum_arr;
  inlist.sendlist = sendlist_arr;

  // fwd_map: atom 6 is virtual (-1), others remapped to 0-5
  std::vector<int> fwd_map = {0, 1, 2, 3, 4, 5, -1};

  deepmd::select_real_atoms_sendlist(new_sendlist, new_sendnum, new_recvnum,
                                     inlist, fwd_map);

  EXPECT_EQ(new_sendlist.size(), 2);
  // swap 0: atom 6 filtered out
  EXPECT_EQ(new_sendnum[0], 3);
  EXPECT_EQ(new_sendlist[0], (std::vector<int>{0, 2, 4}));
  // swap 1: atom 6 filtered out
  EXPECT_EQ(new_sendnum[1], 2);
  EXPECT_EQ(new_sendlist[1], (std::vector<int>{1, 3}));
  EXPECT_EQ(new_recvnum[0], 3);
  EXPECT_EQ(new_recvnum[1], 2);
}

// Multiple virtual atoms with index remapping
TEST_F(TestSelectRealAtomsSendlist, MultipleVirtualWithRemap) {
  // 8 atoms: types [O, NULL, H, O, NULL, H, NULL, H]
  // Real atoms: 0, 2, 3, 5, 7 -> remapped to 0, 1, 2, 3, 4
  // Virtual atoms: 1, 4, 6 -> -1
  int nswap = 1;
  int sendnum_arr[] = {5};
  int recvnum_arr[] = {5};
  int sendlist0[] = {0, 1, 3, 4, 7};  // includes virtual atoms 1, 4
  int* sendlist_arr[] = {sendlist0};

  InputNlist inlist;
  inlist.nswap = nswap;
  inlist.sendnum = sendnum_arr;
  inlist.recvnum = recvnum_arr;
  inlist.sendlist = sendlist_arr;

  std::vector<int> fwd_map = {0, -1, 1, 2, -1, 3, -1, 4};

  deepmd::select_real_atoms_sendlist(new_sendlist, new_sendnum, new_recvnum,
                                     inlist, fwd_map);

  EXPECT_EQ(new_sendlist.size(), 1);
  // atoms 1, 4 filtered; 0->0, 3->2, 7->4
  EXPECT_EQ(new_sendnum[0], 3);
  EXPECT_EQ(new_sendlist[0], (std::vector<int>{0, 2, 4}));
  EXPECT_EQ(new_recvnum[0], 3);
}

// Empty sendlist
TEST_F(TestSelectRealAtomsSendlist, EmptySendlist) {
  int nswap = 1;
  int sendnum_arr[] = {0};
  int recvnum_arr[] = {0};
  int* sendlist0 = nullptr;
  int* sendlist_arr[] = {sendlist0};

  InputNlist inlist;
  inlist.nswap = nswap;
  inlist.sendnum = sendnum_arr;
  inlist.recvnum = recvnum_arr;
  inlist.sendlist = sendlist_arr;

  std::vector<int> fwd_map = {0, 1, 2};

  deepmd::select_real_atoms_sendlist(new_sendlist, new_sendnum, new_recvnum,
                                     inlist, fwd_map);

  EXPECT_EQ(new_sendlist.size(), 1);
  EXPECT_EQ(new_sendnum[0], 0);
  EXPECT_EQ(new_sendlist[0].size(), 0);
  EXPECT_EQ(new_recvnum[0], 0);
}

// All virtual atoms in a swap
TEST_F(TestSelectRealAtomsSendlist, AllVirtualInSwap) {
  int nswap = 2;
  int sendnum_arr[] = {2, 1};
  int recvnum_arr[] = {2, 1};
  int sendlist0[] = {1, 3};  // both virtual
  int sendlist1[] = {0};     // real
  int* sendlist_arr[] = {sendlist0, sendlist1};

  InputNlist inlist;
  inlist.nswap = nswap;
  inlist.sendnum = sendnum_arr;
  inlist.recvnum = recvnum_arr;
  inlist.sendlist = sendlist_arr;

  // atoms 1, 3 are virtual
  std::vector<int> fwd_map = {0, -1, 1, -1};

  deepmd::select_real_atoms_sendlist(new_sendlist, new_sendnum, new_recvnum,
                                     inlist, fwd_map);

  EXPECT_EQ(new_sendlist.size(), 2);
  // swap 0: all filtered
  EXPECT_EQ(new_sendnum[0], 0);
  EXPECT_EQ(new_sendlist[0].size(), 0);
  // swap 1: atom 0 -> 0
  EXPECT_EQ(new_sendnum[1], 1);
  EXPECT_EQ(new_sendlist[1], (std::vector<int>{0}));
}
