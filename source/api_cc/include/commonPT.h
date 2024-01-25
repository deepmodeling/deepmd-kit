// SPDX-License-Identifier: LGPL-3.0-or-later
#include <torch/script.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "neighbor_list.h"
namespace deepmd {
struct NeighborListDataPT {
  /// Array stores the core region atom's index
  std::vector<int> ilist;
  /// Array stores the core region atom's neighbor index
  // std::vector<std::vector<int>> jlist;
  int* jlist;
  /// Array stores the number of neighbors of core region atoms
  std::vector<int> numneigh;
  /// Array stores the the location of the first neighbor of core region atoms
  std::vector<int*> firstneigh;

 public:
  void copy_from_nlist(const InputNlist& inlist,
                       int& max_num_neighbors,
                       int nnei);
  // void make_inlist(InputNlist& inlist);
};
}  // namespace deepmd
