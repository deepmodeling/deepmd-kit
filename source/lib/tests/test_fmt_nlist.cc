#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "neighbor_list.h"

class TestFormatNlist : public ::testing::Test
{
protected:
  std::vector<double > posi = {12.83, 2.56, 2.18, 
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
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 10, 20};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<int> expect_nlist_cpy = {
    33, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1 , 32, 34, 35, -1, -1, -1, -1, -1, -1, 
    0 , 33, -1, -1, -1, -1, -1, -1, -1, -1, 32, 34, 35, -1, -1, -1, -1, -1, -1, -1, 
    6 , 3 , -1, -1, -1, -1, -1, -1, -1, -1, 7 , 4 , 5 , -1, -1, -1, -1, -1, -1, -1, 
    6 , -1, -1, -1, -1, -1, -1, -1, -1, -1, 4 , 5 , 2 , 7 , -1, -1, -1, -1, -1, -1, 
    3 , 6 , -1, -1, -1, -1, -1, -1, -1, -1, 5 , 2 , 7 , -1, -1, -1, -1, -1, -1, -1, 
    3 , 6 , -1, -1, -1, -1, -1, -1, -1, -1, 4 , 2 , 7 , -1, -1, -1, -1, -1, -1, -1
  };      
  std::vector<int> expect_nlist;

  void SetUp() override {
    double box[] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
    region.reinitBox(box);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc, region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nat_stt.resize(3);
    ext_stt.resize(3);
    ext_end.resize(3);
    for (int dd = 0; dd < 3; ++dd){
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    for (unsigned ii = 0; ii < expect_nlist_cpy.size(); ++ii){
      if (expect_nlist_cpy[ii] >= 0){
	expect_nlist.push_back(mapping[expect_nlist_cpy[ii]]);
      }
      else{
	expect_nlist.push_back(-1);
      }
    }    
  }
  void TearDown() override {
  }
};


class TestFormatNlistShortSel : public ::testing::Test
{
protected:
  std::vector<double > posi = {12.83, 2.56, 2.18, 
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
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sec_a = {0, 2, 4};
  std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<int> expect_nlist_cpy = {
    33, -1,  1, 32, 
    0, 33, 32, 34, 
    6,  3,  7,  4, 
    6, -1,  4,  5, 
    3,  6,  5,  2, 
    3,  6,  4,  2, 
  };      
  std::vector<int> expect_nlist;

  void SetUp() override {
    double box[] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
    region.reinitBox(box);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc, region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nat_stt.resize(3);
    ext_stt.resize(3);
    ext_end.resize(3);
    for (int dd = 0; dd < 3; ++dd){
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    for (unsigned ii = 0; ii < expect_nlist_cpy.size(); ++ii){
      if (expect_nlist_cpy[ii] >= 0){
	expect_nlist.push_back(mapping[expect_nlist_cpy[ii]]);
      }
      else{
	expect_nlist.push_back(-1);
      }
    }    
  }
  void TearDown() override {
  }
};


// orginal implementation. copy ghost
TEST_F(TestFormatNlist, orig_cpy)
{
  std::vector<std::vector<int>> nlist_a, nlist_r;
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  build_nlist(nlist_a, nlist_r, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);

  bool pbc = false;
  int ii = 0;
  for (ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_fill_a(fmt_nlist_a, fmt_nlist_r, posi_cpy, ntypes, atype_cpy, region, pbc, ii, nlist_a[ii], nlist_r[ii], rc, sec_a, sec_r);
  
    EXPECT_EQ(ret, -1);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      EXPECT_EQ(fmt_nlist_a[jj], expect_nlist_cpy[ii*sec_a[2]+jj]);
    }
  }
}

// orginal implementation. copy ghost should be equal to pbc
TEST_F(TestFormatNlist, orig_pbc)
{
  std::vector<std::vector<int>> nlist_a_1, nlist_r_1;
  build_nlist(nlist_a_1, nlist_r_1, posi, rc, rc, ncell, region);
  
  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;

  for (int ii = 0; ii < nloc; ++ii){
    int ret_1 = format_nlist_i_fill_a(fmt_nlist_a_1, fmt_nlist_r_1, posi, ntypes, atype, region, true, ii, nlist_a_1[ii], nlist_r_1[ii], rc, sec_a, sec_r);

    EXPECT_EQ(ret_1, -1);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      EXPECT_EQ(fmt_nlist_a_1[jj], expect_nlist[ii*sec_a[2]+jj]);
    }
  }  
}

// orginal implementation. copy ghost should be equal to pbc
TEST_F(TestFormatNlist, orig_cpy_equal_pbc)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
  std::vector<std::vector<int>> nlist_a_1, nlist_r_1;
  build_nlist(nlist_a_1, nlist_r_1, posi, rc, rc, ncell, region);
  
  std::vector<int> fmt_nlist_a_0, fmt_nlist_r_0;
  std::vector<int> fmt_nlist_a_1, fmt_nlist_r_1;

  for (int ii = 0; ii < nloc; ++ii){
    int ret_0 = format_nlist_i_fill_a(fmt_nlist_a_0, fmt_nlist_r_0, posi_cpy, ntypes, atype_cpy, region, false, ii, nlist_a_0[ii], nlist_r_0[ii], rc, sec_a, sec_r);
    int ret_1 = format_nlist_i_fill_a(fmt_nlist_a_1, fmt_nlist_r_1, posi, ntypes, atype, region, true, ii, nlist_a_1[ii], nlist_r_1[ii], rc, sec_a, sec_r);

    EXPECT_EQ(ret_0, -1);
    EXPECT_EQ(ret_1, -1);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      if (fmt_nlist_a_0[jj] == -1){
	// null record
	EXPECT_EQ(fmt_nlist_a_1[jj], -1);
      }
      else{
	EXPECT_EQ(fmt_nlist_a_1[jj], mapping[fmt_nlist_a_0[jj]]);
      }
    }
  }  
}

TEST_F(TestFormatNlist, cpu_i_equal_orig)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);

  std::vector<int> fmt_nlist_a_0, fmt_nlist_r_0;
  std::vector<int> fmt_nlist_a_1;
  
  for (int ii = 0; ii < nloc; ++ii){
    int ret_0 = format_nlist_i_fill_a(fmt_nlist_a_0, fmt_nlist_r_0, posi_cpy, ntypes, atype_cpy, region, false, ii, nlist_a_0[ii], nlist_r_0[ii], rc, sec_a, sec_r);
    int ret_1 = format_nlist_i_cpu<double>(fmt_nlist_a_1, posi_cpy, atype_cpy, ii, nlist_a_0[ii], rc, sec_a);
    EXPECT_EQ(ret_0, -1);
    EXPECT_EQ(ret_1, -1);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      EXPECT_EQ(fmt_nlist_a_1[jj], fmt_nlist_a_0[jj]);
    }
  }
}

TEST_F(TestFormatNlist, cpu)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);  
  // make a input nlist
  int inum = nlist_a_0.size();
  std::vector<int > ilist(inum);
  std::vector<int > numneigh(inum);
  std::vector<int* > firstneigh(inum);
  deepmd::InputNlist in_nlist(inum, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(in_nlist, nlist_a_0);
  // allocate the mem for the result
  std::vector<int> nlist(inum * sec_a.back());
  EXPECT_EQ(nlist.size(), expect_nlist_cpy.size());
  // format nlist
  format_nlist_cpu(
      &nlist[0], 
      in_nlist,
      &posi_cpy[0],
      &atype_cpy[0],
      nloc,
      nall,
      rc,
      sec_a);
  // validate
  for(int ii = 0; ii < nlist.size(); ++ii){
    EXPECT_EQ(nlist[ii], expect_nlist_cpy[ii]);
  }
}


// orginal implementation. copy ghost
TEST_F(TestFormatNlistShortSel, orig_cpy)
{
  std::vector<std::vector<int>> nlist_a, nlist_r;
  std::vector<int> fmt_nlist_a, fmt_nlist_r;
  build_nlist(nlist_a, nlist_r, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);

  bool pbc = false;
  int ii = 0;
  for (ii = 0; ii < nloc; ++ii){
    int ret = format_nlist_i_fill_a(fmt_nlist_a, fmt_nlist_r, posi_cpy, ntypes, atype_cpy, region, pbc, ii, nlist_a[ii], nlist_r[ii], rc, sec_a, sec_r);  
    EXPECT_EQ(ret, 1);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      EXPECT_EQ(fmt_nlist_a[jj], expect_nlist_cpy[ii*sec_a[2]+jj]);
      // printf("%2d ", fmt_nlist_a[jj]);
    }
    // printf("\n");
  }
}


TEST_F(TestFormatNlistShortSel, cpu_equal_orig)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);

  std::vector<int> fmt_nlist_a_1;
  
  for (int ii = 0; ii < nloc; ++ii){
    int ret_1 = format_nlist_i_cpu<double>(fmt_nlist_a_1, posi_cpy, atype_cpy, ii, nlist_a_0[ii], rc, sec_a);
    EXPECT_EQ(ret_1, 1);
    for (int jj = 0; jj < sec_a[2]; ++jj){
      EXPECT_EQ(fmt_nlist_a_1[jj], expect_nlist_cpy[ii*sec_a[2]+jj]);
    }
  }
}

TEST_F(TestFormatNlistShortSel, cpu)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);  
  // make a input nlist
  int inum = nlist_a_0.size();
  std::vector<int > ilist(inum);
  std::vector<int > numneigh(inum);
  std::vector<int* > firstneigh(inum);
  deepmd::InputNlist in_nlist(inum, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(in_nlist, nlist_a_0);  
  // mem
  std::vector<int> nlist(inum * sec_a.back());
  EXPECT_EQ(nlist.size(), expect_nlist_cpy.size());
  // format nlist
  format_nlist_cpu(
      &nlist[0], 
      in_nlist,
      &posi_cpy[0],
      &atype_cpy[0],
      nloc,
      nall,
      rc,
      sec_a);
  // validate
  for(int ii = 0; ii < nlist.size(); ++ii){
    EXPECT_EQ(nlist[ii], expect_nlist_cpy[ii]);
  }
}

