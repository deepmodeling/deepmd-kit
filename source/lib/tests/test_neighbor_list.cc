#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "neighbor_list.h"

class TestNeighborList : public ::testing::Test
{
protected:
  std::vector<double > posi = {
    12.83, 2.56, 2.18, 
    12.09, 2.87, 2.74,
    00.25, 3.32, 1.68,
    3.36, 3.00, 1.81,
    3.51, 2.51, 2.60,
    4.27, 3.22, 1.56
  };
  std::vector<int > atype = {0, 1, 1, 0, 1, 1};
  std::vector<double > posi_cpy;
  std::vector<int > atype_cpy;
  int ntypes = 2;  
  int nloc, nall;
  double rc = 6;
  std::vector<double> boxt = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<int> mapping, ncell, ngcell;
  std::vector<std::vector<int>> expect_nlist_cpy = {
    std::vector<int>({33, 1 , 32, 34, 35,}), 
    std::vector<int>({0 , 33, 32, 34, 35,}),
    std::vector<int>({6 , 3 , 7 , 4 , 5 ,}),
    std::vector<int>({6 , 4 , 5 , 2 , 7 ,}),
    std::vector<int>({3 , 6 , 5 , 2 , 7 ,}),
    std::vector<int>({3 , 6 , 4 , 2 , 7 ,}),
  };

  void SetUp() override {
    SimulationRegion<double> region;
    region.reinitBox(&boxt[0]);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc, region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    EXPECT_EQ(expect_nlist_cpy.size(), nloc);
    for(int ii = 0; ii < nloc; ++ii){
      std::sort(expect_nlist_cpy[ii].begin(), expect_nlist_cpy[ii].end());
    }
  }  
};


TEST_F(TestNeighborList, cpu)
{
  int mem_size = 10;
  int * ilist = new int[nloc];
  int * numneigh = new int[nloc];  
  int ** firstneigh = new int*[nloc];  
  for(int ii = 0; ii < nloc; ++ii){
    firstneigh[ii] = new int[mem_size];
  }

  deepmd::InputNlist nlist(nloc, ilist, numneigh, firstneigh);
  int max_list_size;
  int ret = build_nlist_cpu(
      nlist,
      &max_list_size,
      &posi_cpy[0],
      nloc,
      nall,
      mem_size,
      rc);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(nlist.inum, nloc);
  EXPECT_EQ(max_list_size, 5);
  for(int ii = 0; ii < nloc; ++ii){
    EXPECT_EQ(nlist.ilist[ii], ii);
    EXPECT_EQ(nlist.numneigh[ii], expect_nlist_cpy[ii].size());
    std::sort(nlist.firstneigh[ii], nlist.firstneigh[ii] + nlist.numneigh[ii]);
    for(int jj = 0; jj < nlist.numneigh[ii]; ++jj){
      EXPECT_EQ(nlist.firstneigh[ii][jj], expect_nlist_cpy[ii][jj]);
    }
  }  
  
  delete[] ilist;
  delete[] numneigh;
  for(int ii = 0; ii < nloc; ++ii){
    delete[] firstneigh[ii];
  }
  delete[] firstneigh;
}

TEST_F(TestNeighborList, cpu_lessmem)
{
  int mem_size = 2;
  int * ilist = new int[nloc];
  int * numneigh = new int[nloc];  
  int ** firstneigh = new int*[nloc];  
  for(int ii = 0; ii < nloc; ++ii){
    firstneigh[ii] = new int[mem_size];
  }

  deepmd::InputNlist nlist(nloc, ilist, numneigh, firstneigh);
  int max_list_size;
  int ret = build_nlist_cpu(
      nlist,
      &max_list_size,
      &posi_cpy[0],
      nloc,
      nall,
      mem_size,
      rc);
  EXPECT_EQ(ret, 1);
  EXPECT_EQ(nlist.inum, nloc);
  EXPECT_EQ(max_list_size, 5);
  
  delete[] ilist;
  delete[] numneigh;
  for(int ii = 0; ii < nloc; ++ii){
    delete[] firstneigh[ii];
  }
  delete[] firstneigh;
}

