#include "commonPT.h"

void NeighborListData::copy_from_nlist(const InputNlist& inlist, int& max_num_neighbors, int nnei) 
{
  int inum = inlist.inum;
  ilist.resize(inum);
  numneigh.resize(inum);
  memcpy(&ilist[0], inlist.ilist, inum * sizeof(int));
  int* max_element = std::max_element(inlist.numneigh, inlist.numneigh + inum);
  max_num_neighbors = *max_element;
  if (max_num_neighbors < nnei)
    max_num_neighbors = nnei;
  jlist = (int*)malloc(inum * max_num_neighbors * sizeof(int));
  memset(jlist, -1 , inum * max_num_neighbors * sizeof(int));
  for (int ii = 0; ii < inum; ++ii) {
    int jnum = inlist.numneigh[ii];
    numneigh[ii] = inlist.numneigh[ii];
    memcpy(&jlist[ii * max_num_neighbors], inlist.firstneigh[ii], jnum * sizeof(int));
  }
}