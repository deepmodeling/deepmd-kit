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
  int max_nbor_size;

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
    max_nbor_size = 0;   
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
  int max_nbor_size;

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
    max_nbor_size = 0;      
  }
  void TearDown() override {
  }
};

class TestEncodingDecodingNborInfo : public ::testing::Test
{
protected:
  std::vector<int > valid_type = {
    0, 1, 127, 77, 47, 9, 11
  };
  std::vector<double > valid_dist = {
    23.3333, 0.001234, 1.456, 127.7, 2.021, 0.409, 11.2
  };
  std::vector<int > valid_index = {
    0, 16777215, 1000000, 10000000, 202149, 478910, 5006
  };
  std::vector<uint_64 > expect_key = {
    26270960290103296UL, 144116577447444479UL, 18304268195882549824UL, 11240646899941283456UL, 6775689283274741157UL, 1297497185738772158UL, 1597877147777635214UL
  };

  std::vector<int > invalid_type = {
    0, 256, 128, 77, 47, 126, 1100
  };
  std::vector<double > invalid_dist = {
    128.0, 0.001234, 1.456, 130.7, 2.021, 0.409, 11.2
  };
  std::vector<int > invalid_index = {
    0, 16777215, 1 << 24, 10000000, 20210409, 478910, 5006
  };
  std::vector<bool> expect_cuda_error_check = {
    false, false, false, false, false, true, false
  };

  std::vector<int > expect_type = valid_type;
  std::vector<int > expect_index = valid_index;
  int size_of_array = valid_type.size();

  void SetUp() override {
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

#if GOOGLE_CUDA
TEST_F(TestFormatNlist, gpu_cuda)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);  
  // make a input nlist
  int inum = nlist_a_0.size();
  std::vector<int > ilist(inum);
  std::vector<int > numneigh(inum);
  std::vector<int* > firstneigh(inum);
  deepmd::InputNlist in_nlist(inum, &ilist[0], &numneigh[0], &firstneigh[0]), gpu_inlist;
  convert_nlist(in_nlist, nlist_a_0);
  // allocate the mem for the result
  std::vector<int> nlist(inum * sec_a.back());
  EXPECT_EQ(nlist.size(), expect_nlist_cpy.size());

  double * posi_cpy_dev = NULL;
  int * atype_cpy_dev = NULL, * nlist_dev = NULL, * array_int_dev = NULL, * memory_dev = NULL;
  uint_64 * array_longlong_dev = NULL;
  for (int ii = 0; ii < inum; ii++) {
    max_nbor_size = max_nbor_size >= numneigh[ii] ? max_nbor_size : numneigh[ii];
  }
  assert(max_nbor_size <= GPU_MAX_NBOR_SIZE);
  if (max_nbor_size <= 1024) {
    max_nbor_size = 1024;
  }
  else if (max_nbor_size <= 2048) {
    max_nbor_size = 2048;
  }
  else {
    max_nbor_size = 4096;
  }
  deepmd::malloc_device_memory_sync(posi_cpy_dev, posi_cpy);
  deepmd::malloc_device_memory_sync(atype_cpy_dev, atype_cpy);
  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory(array_int_dev, sec_a.size() + nloc * sec_a.size() + nloc);
  deepmd::malloc_device_memory(array_longlong_dev, nloc * GPU_MAX_NBOR_SIZE * 2);
  deepmd::malloc_device_memory(memory_dev, nloc * max_nbor_size);
  deepmd::convert_nlist_gpu_cuda(gpu_inlist, in_nlist, memory_dev, max_nbor_size);
  // format nlist
  format_nbor_list_gpu_cuda(
      nlist_dev, 
      posi_cpy_dev, atype_cpy_dev, gpu_inlist, array_int_dev, array_longlong_dev, max_nbor_size, nloc, nall, rc, sec_a);
  deepmd::memcpy_device_to_host(nlist_dev, nlist);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(posi_cpy_dev);
  deepmd::delete_device_memory(atype_cpy_dev);
  deepmd::delete_device_memory(array_int_dev);
  deepmd::delete_device_memory(array_longlong_dev);
  deepmd::delete_device_memory(memory_dev);
  deepmd::free_nlist_gpu_cuda(gpu_inlist);
  // validate
  for(int ii = 0; ii < nlist.size(); ++ii){
    EXPECT_EQ(nlist[ii], expect_nlist_cpy[ii]);
  }
}

TEST_F(TestFormatNlistShortSel, gpu_cuda)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);  
  // make a input nlist
  int inum = nlist_a_0.size();
  std::vector<int > ilist(inum);
  std::vector<int > numneigh(inum);
  std::vector<int* > firstneigh(inum);
  deepmd::InputNlist in_nlist(inum, &ilist[0], &numneigh[0], &firstneigh[0]), gpu_inlist;
  convert_nlist(in_nlist, nlist_a_0);  
  // mem
  std::vector<int> nlist(inum * sec_a.back());
  EXPECT_EQ(nlist.size(), expect_nlist_cpy.size());
  // format nlist
  double * posi_cpy_dev = NULL;
  int * atype_cpy_dev = NULL, * nlist_dev = NULL, * array_int_dev = NULL, * memory_dev = NULL;
  uint_64 * array_longlong_dev = NULL;
  for (int ii = 0; ii < inum; ii++) {
    max_nbor_size = max_nbor_size >= numneigh[ii] ? max_nbor_size : numneigh[ii];
  }
  assert(max_nbor_size <= GPU_MAX_NBOR_SIZE);
  if (max_nbor_size <= 1024) {
    max_nbor_size = 1024;
  }
  else if (max_nbor_size <= 2048) {
    max_nbor_size = 2048;
  }
  else {
    max_nbor_size = 4096;
  }
  deepmd::malloc_device_memory_sync(posi_cpy_dev, posi_cpy);
  deepmd::malloc_device_memory_sync(atype_cpy_dev, atype_cpy);
  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory(array_int_dev, sec_a.size() + nloc * sec_a.size() + nloc);
  deepmd::malloc_device_memory(array_longlong_dev, nloc * GPU_MAX_NBOR_SIZE * 2);
  deepmd::malloc_device_memory(memory_dev, nloc * max_nbor_size);
  deepmd::convert_nlist_gpu_cuda(gpu_inlist, in_nlist, memory_dev, max_nbor_size);
  // format nlist
  format_nbor_list_gpu_cuda(
      nlist_dev, 
      posi_cpy_dev, atype_cpy_dev, gpu_inlist, array_int_dev, array_longlong_dev, max_nbor_size, nloc, nall, rc, sec_a);
  deepmd::memcpy_device_to_host(nlist_dev, nlist);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(posi_cpy_dev);
  deepmd::delete_device_memory(atype_cpy_dev);
  deepmd::delete_device_memory(array_int_dev);
  deepmd::delete_device_memory(array_longlong_dev);
  deepmd::delete_device_memory(memory_dev);
  deepmd::free_nlist_gpu_cuda(gpu_inlist);
  // validate
  for(int ii = 0; ii < nlist.size(); ++ii){
    EXPECT_EQ(nlist[ii], expect_nlist_cpy[ii]);
  }
}

TEST_F(TestEncodingDecodingNborInfo, valid_nbor_info_gpu_cuda) 
{
  int * valid_type_dev = NULL, * valid_index_dev = NULL, * out_type_dev = NULL, * out_index_dev = NULL;
  double * valid_dist_dev = NULL;
  uint_64 * key_dev = NULL;
  std::vector<int> out_type(size_of_array, 0);
  std::vector<int> out_index(size_of_array, 0);
  std::vector<uint_64> key(size_of_array, 0);
  deepmd::malloc_device_memory_sync(valid_type_dev, valid_type);
  deepmd::malloc_device_memory_sync(valid_dist_dev, valid_dist);
  deepmd::malloc_device_memory_sync(valid_index_dev, valid_index);
  deepmd::malloc_device_memory_sync(out_type_dev, out_type);
  deepmd::malloc_device_memory_sync(out_index_dev, out_index);
  deepmd::malloc_device_memory_sync(key_dev, key);

  deepmd::test_encoding_decoding_nbor_info_gpu_cuda(
      key_dev, out_type_dev, out_index_dev,
      valid_type_dev, valid_dist_dev, valid_index_dev, size_of_array
  );

  deepmd::memcpy_device_to_host(key_dev, key);
  deepmd::memcpy_device_to_host(out_type_dev, out_type);
  deepmd::memcpy_device_to_host(out_index_dev, out_index);
  deepmd::delete_device_memory(valid_type_dev);
  deepmd::delete_device_memory(valid_dist_dev);
  deepmd::delete_device_memory(valid_index_dev);
  deepmd::delete_device_memory(out_type_dev);
  deepmd::delete_device_memory(out_index_dev);
  deepmd::delete_device_memory(key_dev);
  // validate
  for(int ii = 0; ii < size_of_array; ii++) {
    EXPECT_EQ(key[ii], expect_key[ii]);
    EXPECT_EQ(out_type[ii], expect_type[ii]);
    EXPECT_EQ(out_index[ii], expect_index[ii]);
  }
}

// TEST_F(TestEncodingDecodingNborInfo, invalid_nbor_info_gpu_cuda) 
// {
//   int * invalid_type_dev = NULL, * invalid_index_dev = NULL, * out_type_dev = NULL, * out_index_dev = NULL;
//   double * invalid_dist_dev = NULL;
//   uint_64 * key_dev = NULL;
//   std::vector<int> out_type(size_of_array, 0);
//   std::vector<int> out_index(size_of_array, 0);
//   std::vector<uint_64> key(size_of_array, 0);
//   deepmd::malloc_device_memory_sync(invalid_type_dev, invalid_type);
//   deepmd::malloc_device_memory_sync(invalid_dist_dev, invalid_dist);
//   deepmd::malloc_device_memory_sync(invalid_index_dev, invalid_index);
//   deepmd::malloc_device_memory_sync(out_type_dev, out_type);
//   deepmd::malloc_device_memory_sync(out_index_dev, out_index);
//   deepmd::malloc_device_memory_sync(key_dev, key);
  
//   EXPECT_EQ(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, true);
//   deepmd::test_encoding_decoding_nbor_info_gpu_cuda(
//       key_dev, out_type_dev, out_index_dev,
//       invalid_type_dev, invalid_dist_dev, invalid_index_dev, size_of_array
//   );
//   EXPECT_EQ(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, false);
//   cudaErrcheck(cudaDeviceReset());
//   deepmd::memcpy_device_to_host(key_dev, key);
//   deepmd::memcpy_device_to_host(out_type_dev, out_type);
//   deepmd::memcpy_device_to_host(out_index_dev, out_index);
//   deepmd::delete_device_memory(invalid_type_dev);
//   deepmd::delete_device_memory(invalid_dist_dev);
//   deepmd::delete_device_memory(invalid_index_dev);
//   deepmd::delete_device_memory(out_type_dev);
//   deepmd::delete_device_memory(out_index_dev);
//   deepmd::delete_device_memory(key_dev);
// }
#endif // GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
TEST_F(TestFormatNlist, gpu_rocm)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);  
  // make a input nlist
  int inum = nlist_a_0.size();
  std::vector<int > ilist(inum);
  std::vector<int > numneigh(inum);
  std::vector<int* > firstneigh(inum);
  deepmd::InputNlist in_nlist(inum, &ilist[0], &numneigh[0], &firstneigh[0]), gpu_inlist;
  convert_nlist(in_nlist, nlist_a_0);
  // allocate the mem for the result
  std::vector<int> nlist(inum * sec_a.back());
  EXPECT_EQ(nlist.size(), expect_nlist_cpy.size());

  double * posi_cpy_dev = NULL;
  int * atype_cpy_dev = NULL, * nlist_dev = NULL, * array_int_dev = NULL, * memory_dev = NULL;
  uint_64 * array_longlong_dev = NULL;
  for (int ii = 0; ii < inum; ii++) {
    max_nbor_size = max_nbor_size >= numneigh[ii] ? max_nbor_size : numneigh[ii];
  }
  assert(max_nbor_size <= GPU_MAX_NBOR_SIZE);
  if (max_nbor_size <= 1024) {
    max_nbor_size = 1024;
  }
  else if (max_nbor_size <= 2048) {
    max_nbor_size = 2048;
  }
  else {
    max_nbor_size = 4096;
  }
  deepmd::malloc_device_memory_sync(posi_cpy_dev, posi_cpy);
  deepmd::malloc_device_memory_sync(atype_cpy_dev, atype_cpy);
  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory(array_int_dev, sec_a.size() + nloc * sec_a.size() + nloc);
  deepmd::malloc_device_memory(array_longlong_dev, nloc * GPU_MAX_NBOR_SIZE * 2);
  deepmd::malloc_device_memory(memory_dev, nloc * max_nbor_size);
  deepmd::convert_nlist_gpu_rocm(gpu_inlist, in_nlist, memory_dev, max_nbor_size);
  // format nlist
  format_nbor_list_gpu_rocm(
      nlist_dev, 
      posi_cpy_dev, atype_cpy_dev, gpu_inlist, array_int_dev, array_longlong_dev, max_nbor_size, nloc, nall, rc, sec_a);
  deepmd::memcpy_device_to_host(nlist_dev, nlist);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(posi_cpy_dev);
  deepmd::delete_device_memory(atype_cpy_dev);
  deepmd::delete_device_memory(array_int_dev);
  deepmd::delete_device_memory(array_longlong_dev);
  deepmd::delete_device_memory(memory_dev);
  deepmd::free_nlist_gpu_rocm(gpu_inlist);
  // validate
  for(int ii = 0; ii < nlist.size(); ++ii){
    EXPECT_EQ(nlist[ii], expect_nlist_cpy[ii]);
  }
}

TEST_F(TestFormatNlistShortSel, gpu_rocm)
{
  std::vector<std::vector<int>> nlist_a_0, nlist_r_0;
  build_nlist(nlist_a_0, nlist_r_0, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);  
  // make a input nlist
  int inum = nlist_a_0.size();
  std::vector<int > ilist(inum);
  std::vector<int > numneigh(inum);
  std::vector<int* > firstneigh(inum);
  deepmd::InputNlist in_nlist(inum, &ilist[0], &numneigh[0], &firstneigh[0]), gpu_inlist;
  convert_nlist(in_nlist, nlist_a_0);  
  // mem
  std::vector<int> nlist(inum * sec_a.back());
  EXPECT_EQ(nlist.size(), expect_nlist_cpy.size());
  // format nlist
  double * posi_cpy_dev = NULL;
  int * atype_cpy_dev = NULL, * nlist_dev = NULL, * array_int_dev = NULL, * memory_dev = NULL;
  uint_64 * array_longlong_dev = NULL;
  for (int ii = 0; ii < inum; ii++) {
    max_nbor_size = max_nbor_size >= numneigh[ii] ? max_nbor_size : numneigh[ii];
  }
  assert(max_nbor_size <= GPU_MAX_NBOR_SIZE);
  if (max_nbor_size <= 1024) {
    max_nbor_size = 1024;
  }
  else if (max_nbor_size <= 2048) {
    max_nbor_size = 2048;
  }
  else {
    max_nbor_size = 4096;
  }
  deepmd::malloc_device_memory_sync(posi_cpy_dev, posi_cpy);
  deepmd::malloc_device_memory_sync(atype_cpy_dev, atype_cpy);
  deepmd::malloc_device_memory_sync(nlist_dev, nlist);
  deepmd::malloc_device_memory(array_int_dev, sec_a.size() + nloc * sec_a.size() + nloc);
  deepmd::malloc_device_memory(array_longlong_dev, nloc * GPU_MAX_NBOR_SIZE * 2);
  deepmd::malloc_device_memory(memory_dev, nloc * max_nbor_size);
  deepmd::convert_nlist_gpu_rocm(gpu_inlist, in_nlist, memory_dev, max_nbor_size);
  // format nlist
  format_nbor_list_gpu_rocm(
      nlist_dev, 
      posi_cpy_dev, atype_cpy_dev, gpu_inlist, array_int_dev, array_longlong_dev, max_nbor_size, nloc, nall, rc, sec_a);
  deepmd::memcpy_device_to_host(nlist_dev, nlist);
  deepmd::delete_device_memory(nlist_dev);
  deepmd::delete_device_memory(posi_cpy_dev);
  deepmd::delete_device_memory(atype_cpy_dev);
  deepmd::delete_device_memory(array_int_dev);
  deepmd::delete_device_memory(array_longlong_dev);
  deepmd::delete_device_memory(memory_dev);
  deepmd::free_nlist_gpu_rocm(gpu_inlist);
  // validate
  for(int ii = 0; ii < nlist.size(); ++ii){
    EXPECT_EQ(nlist[ii], expect_nlist_cpy[ii]);
  }
}

TEST_F(TestEncodingDecodingNborInfo, valid_nbor_info_gpu_rocm) 
{
  int * valid_type_dev = NULL, * valid_index_dev = NULL, * out_type_dev = NULL, * out_index_dev = NULL;
  double * valid_dist_dev = NULL;
  uint_64 * key_dev = NULL;
  std::vector<int> out_type(size_of_array, 0);
  std::vector<int> out_index(size_of_array, 0);
  std::vector<uint_64> key(size_of_array, 0);
  deepmd::malloc_device_memory_sync(valid_type_dev, valid_type);
  deepmd::malloc_device_memory_sync(valid_dist_dev, valid_dist);
  deepmd::malloc_device_memory_sync(valid_index_dev, valid_index);
  deepmd::malloc_device_memory_sync(out_type_dev, out_type);
  deepmd::malloc_device_memory_sync(out_index_dev, out_index);
  deepmd::malloc_device_memory_sync(key_dev, key);

  deepmd::test_encoding_decoding_nbor_info_gpu_rocm(
      key_dev, out_type_dev, out_index_dev,
      valid_type_dev, valid_dist_dev, valid_index_dev, size_of_array
  );

  deepmd::memcpy_device_to_host(key_dev, key);
  deepmd::memcpy_device_to_host(out_type_dev, out_type);
  deepmd::memcpy_device_to_host(out_index_dev, out_index);
  deepmd::delete_device_memory(valid_type_dev);
  deepmd::delete_device_memory(valid_dist_dev);
  deepmd::delete_device_memory(valid_index_dev);
  deepmd::delete_device_memory(out_type_dev);
  deepmd::delete_device_memory(out_index_dev);
  deepmd::delete_device_memory(key_dev);
  // validate
  for(int ii = 0; ii < size_of_array; ii++) {
    EXPECT_EQ(key[ii], expect_key[ii]);
    EXPECT_EQ(out_type[ii], expect_type[ii]);
    EXPECT_EQ(out_index[ii], expect_index[ii]);
  }
}


#endif // TENSORFLOW_USE_ROCM