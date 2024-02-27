// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PYTORCH
#include "commonPT.h"
using namespace deepmd;
void NeighborListDataPT::copy_from_nlist(const InputNlist& inlist,
                                         unsigned long int& max_num_neighbors) {
  int inum = inlist.inum;
  ilist.resize(inum);
  numneigh.resize(inum);
  memcpy(&ilist[0], inlist.ilist, inum * sizeof(int));
  unsigned long int* max_element =
      std::max_element(inlist.numneigh, inlist.numneigh + inum);
  max_num_neighbors = *max_element;
  jlist.resize(inum * max_num_neighbors);
  memset(&jlist[0], -1, inum * max_num_neighbors * sizeof(int));
  for (int ii = 0; ii < inum; ++ii) {
    int jnum = inlist.numneigh[ii];
    numneigh[ii] = inlist.numneigh[ii];
    memcpy(&jlist[ii * max_num_neighbors], inlist.firstneigh[ii],
           jnum * sizeof(int));
  }
}
#endif
