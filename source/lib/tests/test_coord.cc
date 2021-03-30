#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include "coord.h"
#include "device.h"

class TestNormCoord : public ::testing::Test
{
protected:
  std::vector<double > posi = {
    1.83, 1.56, 1.18, 
    1.09, 1.87, 1.74,
  };
  std::vector<double > boxt = {
    3.27785716,  0.09190842,  0.14751448,  0.02331264,  3.36482777, -0.2999871 , -0.47510999, -0.38123489,  3.33561809
  };
  // 10, 11, 12
  std::vector<double > r0 ={
    29.16369076, 34.91737099, 39.38270378,
    28.42369076, 35.22737099, 39.94270378
  };
  // -10, 11, -12
  std::vector<double > r1 ={
    -24.990812680000005,  42.22883995, -43.622419980000004,
    -25.730812680000003,  42.538839949999996, -43.06241998       
  };
  // 10, -11, 12
  std::vector<double > r2 ={
    28.65081268, -39.10883995,  45.98241998,
    27.91081268, -38.79883995,  46.54241998
  };
  int natoms;
  void SetUp() override {
    natoms = posi.size()/3;
  };
};


TEST_F(TestNormCoord, cpu_case0)
{
  deepmd::Region<double> region;
  init_region_cpu(region, &boxt[0]);
  std::vector<double > out_c(r0);
  normalize_coord_cpu(&out_c[0], natoms, region);
  for(int ii = 0; ii < posi.size(); ++ii){
    EXPECT_LT(fabs(out_c[ii] - posi[ii]), 1e-12);
  }
}

TEST_F(TestNormCoord, cpu_case1)
{
  deepmd::Region<double> region;
  init_region_cpu(region, &boxt[0]);
  std::vector<double > out_c(r1);
  normalize_coord_cpu(&out_c[0], natoms, region);
  for(int ii = 0; ii < posi.size(); ++ii){
    EXPECT_LT(fabs(out_c[ii] - posi[ii]), 1e-12);
  }
}

TEST_F(TestNormCoord, cpu_case2)
{
  deepmd::Region<double> region;
  init_region_cpu(region, &boxt[0]);
  std::vector<double > out_c(r2);
  normalize_coord_cpu(&out_c[0], natoms, region);
  for(int ii = 0; ii < posi.size(); ++ii){
    EXPECT_LT(fabs(out_c[ii] - posi[ii]), 1e-12);
  }
}

#if GOOGLE_CUDA
TEST_F(TestNormCoord, gpu_case0)
{
  deepmd::Region<double> region;
  deepmd::Region<double> region_dev;
  double * new_boxt = region_dev.boxt;
  double * new_rec_boxt = region_dev.rec_boxt;
  init_region_cpu(region, &boxt[0]);
  std::vector<double> box_info;
  box_info.resize(18);
  memcpy(&box_info[0], &boxt[0], sizeof(double)*9);
  memcpy(&box_info[9], region.rec_boxt, sizeof(double)*9);
  double * box_info_dev=NULL;
  double * out_c_dev=NULL;
  std::vector<double > out_c(r0);
  deepmd::malloc_device_memory_sync(box_info_dev, box_info);
  deepmd::malloc_device_memory_sync(out_c_dev, out_c);
  region_dev.boxt = box_info_dev;
  region_dev.rec_boxt = box_info_dev + 9;
  deepmd::normalize_coord_gpu(out_c_dev, natoms, region_dev);
  region_dev.boxt = new_boxt;
  region_dev.rec_boxt = new_rec_boxt;
  deepmd::memcpy_device_to_host(out_c_dev, out_c);
  deepmd::delete_device_memory(box_info_dev);
  deepmd::delete_device_memory(out_c_dev);
  for(int ii = 0; ii < posi.size(); ++ii){
    EXPECT_LT(fabs(out_c[ii] - posi[ii]), 1e-12);
  }
}

TEST_F(TestNormCoord, gpu_case1)
{
  deepmd::Region<double> region;
  deepmd::Region<double> region_dev;
  double * new_boxt = region_dev.boxt;
  double * new_rec_boxt = region_dev.rec_boxt;
  init_region_cpu(region, &boxt[0]);
  std::vector<double> box_info;
  box_info.resize(18);
  memcpy(&box_info[0], &boxt[0], sizeof(double)*9);
  memcpy(&box_info[9], region.rec_boxt, sizeof(double)*9);
  double * box_info_dev=NULL;
  double * out_c_dev=NULL;
  std::vector<double > out_c(r1);
  deepmd::malloc_device_memory_sync(box_info_dev, box_info);
  deepmd::malloc_device_memory_sync(out_c_dev, out_c);
  region_dev.boxt = box_info_dev;
  region_dev.rec_boxt = box_info_dev + 9;
  deepmd::normalize_coord_gpu(out_c_dev, natoms, region_dev);
  region_dev.boxt = new_boxt;
  region_dev.rec_boxt = new_rec_boxt;
  deepmd::memcpy_device_to_host(out_c_dev, out_c);
  deepmd::delete_device_memory(box_info_dev);
  deepmd::delete_device_memory(out_c_dev);
  for(int ii = 0; ii < posi.size(); ++ii){
    EXPECT_LT(fabs(out_c[ii] - posi[ii]), 1e-12);
  }
}

TEST_F(TestNormCoord, gpu_case2)
{
  deepmd::Region<double> region;
  deepmd::Region<double> region_dev;
  double * new_boxt = region_dev.boxt;
  double * new_rec_boxt = region_dev.rec_boxt;
  init_region_cpu(region, &boxt[0]);
  std::vector<double> box_info;
  box_info.resize(18);
  memcpy(&box_info[0], &boxt[0], sizeof(double)*9);
  memcpy(&box_info[9], region.rec_boxt, sizeof(double)*9);
  double * box_info_dev=NULL;
  double * out_c_dev=NULL;
  std::vector<double > out_c(r2);
  deepmd::malloc_device_memory_sync(box_info_dev, box_info);
  deepmd::malloc_device_memory_sync(out_c_dev, out_c);
  region_dev.boxt = box_info_dev;
  region_dev.rec_boxt = box_info_dev + 9;
  deepmd::normalize_coord_gpu(out_c_dev, natoms, region_dev);
  region_dev.boxt = new_boxt;
  region_dev.rec_boxt = new_rec_boxt;
  deepmd::memcpy_device_to_host(out_c_dev, out_c);
  deepmd::delete_device_memory(box_info_dev);
  deepmd::delete_device_memory(out_c_dev);
  for(int ii = 0; ii < posi.size(); ++ii){
    EXPECT_LT(fabs(out_c[ii] - posi[ii]), 1e-12);
  }
}

#endif //GOOGLE_CUDA

typedef std::pair<std::vector<double>,std::vector<int>> atom;

static void
sort_atoms(
    std::vector<double > & coord,
    std::vector<int > & atype,
    std::vector<int > & mapping,
    const std::vector<double > & icoord,
    const std::vector<int > & iatype,
    const std::vector<int > & imapping,
    const int start,
    const int end
    )
{
  int natoms = end - start;
  std::vector<atom> atoms(natoms);
  for(int ii = start; ii < end; ++ii){
    atom tmp_atom;
    tmp_atom.first.resize(3);
    for(int dd = 0; dd < 3; ++dd){
      tmp_atom.first[dd] = icoord[ii*3+dd];
    }
    tmp_atom.second.resize(2);
    tmp_atom.second[0] = iatype[ii];
    tmp_atom.second[1] = imapping[ii];
    atoms[ii-start] = tmp_atom;
  }
  std::sort(atoms.begin(), atoms.end());
  coord = icoord;
  atype = iatype;
  mapping = imapping;
  for(int ii = start; ii < end; ++ii){
    for(int dd = 0; dd < 3; ++dd){
      coord[ii*3+dd] = atoms[ii-start].first[dd];
    }
    atype[ii] = atoms[ii-start].second[0];
    mapping[ii] = atoms[ii-start].second[1];
  }
}

class TestCopyCoord : public ::testing::Test
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
  std::vector<double > _expected_posi_cpy = {
    12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 0.25, 3.32, 1.68, 3.36, 3.00, 1.81, 3.51, 2.51, 2.60, 4.27, 3.22, 1.56, -0.17, 2.56, 2.18, -0.91, 2.87, 2.74, -0.17, 2.56, 15.18, -0.91, 2.87, 15.74, -0.17, 15.56, 2.18, -0.91, 15.87, 2.74, -0.17, 15.56, 15.18, -0.91, 15.87, 15.74, 0.25, 3.32, 14.68, 3.36, 3.00, 14.81, 3.51, 2.51, 15.60, 4.27, 3.22, 14.56, 0.25, 16.32, 1.68, 3.36, 16.00, 1.81, 3.51, 15.51, 2.60, 4.27, 16.22, 1.56, 0.25, 16.32, 14.68, 3.36, 16.00, 14.81, 3.51, 15.51, 15.60, 4.27, 16.22, 14.56, 12.83, 2.56, 15.18, 12.09, 2.87, 15.74, 12.83, 15.56, 2.18, 12.09, 15.87, 2.74, 12.83, 15.56, 15.18, 12.09, 15.87, 15.74, 13.25, 3.32, 1.68, 16.36, 3.00, 1.81, 16.51, 2.51, 2.60, 17.27, 3.22, 1.56, 13.25, 3.32, 14.68, 16.36, 3.00, 14.81, 16.51, 2.51, 15.60, 17.27, 3.22, 14.56, 13.25, 16.32, 1.68, 16.36, 16.00, 1.81, 16.51, 15.51, 2.60, 17.27, 16.22, 1.56, 13.25, 16.32, 14.68, 16.36, 16.00, 14.81, 16.51, 15.51, 15.60, 17.27, 16.22, 14.56, 
  };
  std::vector<double > expected_posi_cpy;
  std::vector<int > _expected_atype_cpy = {
    0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 
  };
  std::vector<int > expected_atype_cpy;
  std::vector<int > _expected_mapping = {
    0, 1, 2, 3, 4, 5, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 
  };
  std::vector<int > expected_mapping;
  int ntypes = 2;  
  int nloc, expected_nall;
  double rc = 6;
  std::vector<double> boxt = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<int> ncell, ngcell;

  void SetUp() override {
    nloc = posi.size() / 3;
    expected_nall = _expected_posi_cpy.size() / 3;
    EXPECT_EQ(expected_nall, _expected_atype_cpy.size());
    EXPECT_EQ(expected_nall, _expected_mapping.size());
    // sort the atoms between nloc and nall, to remove the uncertainty of the ordering
    sort_atoms(
	expected_posi_cpy,
	expected_atype_cpy,
	expected_mapping,
	_expected_posi_cpy,
	_expected_atype_cpy,
	_expected_mapping,
	nloc,
	expected_nall);
  }  
};




TEST_F(TestCopyCoord, cpu)
{
  int mem_size = 1000;
  std::vector<double > out_c(mem_size * 3);
  std::vector<int > out_t(mem_size);
  std::vector<int > mapping(mem_size);
  int nall;
  deepmd::Region<double> region;
  init_region_cpu(region, &boxt[0]);
  
  int ret = copy_coord_cpu(
      &out_c[0],
      &out_t[0],
      &mapping[0],
      &nall,
      &posi[0],
      &atype[0],
      nloc,
      mem_size,
      rc,
      region);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(nall, expected_nall);
  // std::cout << "---------------------" 
  // 	    << nloc << " " 
  // 	    << nall << std::endl;
  
  out_c.resize(nall*3);
  out_t.resize(nall);
  mapping.resize(nall);
  
  std::vector<double > out_c_1(mem_size * 3);
  std::vector<int > out_t_1(mem_size);
  std::vector<int > mapping_1(mem_size);
  sort_atoms(out_c_1, out_t_1, mapping_1, out_c, out_t, mapping, nloc, nall);
  for(int ii = 0; ii < expected_nall; ++ii){
    for(int dd = 0; dd < 3; ++dd){
      EXPECT_LT(fabs(out_c_1[ii*3+dd] - expected_posi_cpy[ii*3+dd]), 1e-12);
    }
    EXPECT_EQ(out_t_1[ii], expected_atype_cpy[ii]);
    EXPECT_EQ(mapping_1[ii], expected_mapping[ii]);
  }  
}

TEST_F(TestCopyCoord, cpu_lessmem)
{
  int mem_size = 40;
  std::vector<double > out_c(mem_size * 3);
  std::vector<int > out_t(mem_size);
  std::vector<int > mapping(mem_size);
  int nall;
  deepmd::Region<double> region;
  init_region_cpu(region, &boxt[0]);
  
  int ret = copy_coord_cpu(
      &out_c[0],
      &out_t[0],
      &mapping[0],
      &nall,
      &posi[0],
      &atype[0],
      nloc,
      mem_size,
      rc,
      region);
  EXPECT_EQ(ret, 1);
  // EXPECT_EQ(nall, expected_nall);
  // std::cout << "---------------------" 
  // 	    << nloc << " " 
  // 	    << nall << std::endl;
}

#if GOOGLE_CUDA
TEST_F(TestCopyCoord, gpu)
{
  int mem_size = 1000;
  std::vector<double > out_c(mem_size * 3);
  std::vector<int > out_t(mem_size);
  std::vector<int > mapping(mem_size);
  int nall;
  std::vector<int> cell_info;
  cell_info.resize(23);
  deepmd::Region<double> region;
  deepmd::Region<double> region_dev;
  double * new_boxt = region_dev.boxt;
  double * new_rec_boxt = region_dev.rec_boxt;
  init_region_cpu(region, &boxt[0]);
  deepmd::compute_cell_info(&cell_info[0], rc, region);
  std::vector<double> box_info;
  box_info.resize(18);
  memcpy(&box_info[0], &boxt[0], sizeof(double)*9);
  memcpy(&box_info[9], region.rec_boxt, sizeof(double)*9);
  const int loc_cellnum=cell_info[21];
  const int total_cellnum=cell_info[22];
  int * cell_info_dev=NULL;
  double * box_info_dev=NULL;
  double * out_c_dev=NULL, * in_c_dev=NULL;
  int * out_t_dev=NULL, * in_t_dev=NULL, * mapping_dev=NULL, * int_data_dev=NULL;
  deepmd::malloc_device_memory_sync(cell_info_dev, cell_info);
  deepmd::malloc_device_memory_sync(box_info_dev, box_info);
  deepmd::malloc_device_memory_sync(in_c_dev, posi);
  deepmd::malloc_device_memory_sync(in_t_dev, atype);
  deepmd::malloc_device_memory(out_c_dev, mem_size * 3);
  deepmd::malloc_device_memory(out_t_dev, mem_size);
  deepmd::malloc_device_memory(mapping_dev, mem_size);
  deepmd::malloc_device_memory(int_data_dev, nloc*3+loc_cellnum+total_cellnum*3+total_cellnum*3+loc_cellnum+1+total_cellnum+1+nloc);
  region_dev.boxt = box_info_dev;
  region_dev.rec_boxt = box_info_dev + 9;
  int ret = deepmd::copy_coord_gpu(
      out_c_dev, 
      out_t_dev, 
      mapping_dev, 
      &nall,
      int_data_dev,
      in_c_dev,
      in_t_dev,
      nloc,
      mem_size,
      loc_cellnum,
      total_cellnum,
      cell_info_dev,
      region_dev);
  region_dev.boxt = new_boxt;
  region_dev.rec_boxt = new_rec_boxt;
  deepmd::memcpy_device_to_host(out_c_dev, out_c);
  deepmd::memcpy_device_to_host(out_t_dev, out_t);
  deepmd::memcpy_device_to_host(mapping_dev, mapping);
  deepmd::delete_device_memory(cell_info_dev);
  deepmd::delete_device_memory(box_info_dev);
  deepmd::delete_device_memory(in_c_dev);
  deepmd::delete_device_memory(in_t_dev);
  deepmd::delete_device_memory(out_c_dev);
  deepmd::delete_device_memory(out_t_dev);
  deepmd::delete_device_memory(mapping_dev);
  deepmd::delete_device_memory(int_data_dev);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(nall, expected_nall);
  out_c.resize(nall*3);
  out_t.resize(nall);
  mapping.resize(nall);

  std::vector<double > out_c_1(mem_size * 3);
  std::vector<int > out_t_1(mem_size);
  std::vector<int > mapping_1(mem_size);
  sort_atoms(out_c_1, out_t_1, mapping_1, out_c, out_t, mapping, nloc, nall);
  for(int ii = 0; ii < expected_nall; ++ii){
    for(int dd = 0; dd < 3; ++dd){
      EXPECT_LT(fabs(out_c_1[ii*3+dd] - expected_posi_cpy[ii*3+dd]), 1e-12);
    }
    EXPECT_EQ(out_t_1[ii], expected_atype_cpy[ii]);
    EXPECT_EQ(mapping_1[ii], expected_mapping[ii]);
  }  
}

TEST_F(TestCopyCoord, gpu_lessmem)
{
  int mem_size = 40;
  std::vector<double > out_c(mem_size * 3);
  std::vector<int > out_t(mem_size);
  std::vector<int > mapping(mem_size);
  int nall;
  std::vector<int> cell_info;
  cell_info.resize(23);
  deepmd::Region<double> region;
  deepmd::Region<double> region_dev;
  double * new_boxt = region_dev.boxt;
  double * new_rec_boxt = region_dev.rec_boxt;
  init_region_cpu(region, &boxt[0]);
  deepmd::compute_cell_info(&cell_info[0], rc, region);
  std::vector<double> box_info;
  box_info.resize(18);
  memcpy(&box_info[0], &boxt[0], sizeof(double)*9);
  memcpy(&box_info[9], region.rec_boxt, sizeof(double)*9);
  const int loc_cellnum=cell_info[21];
  const int total_cellnum=cell_info[22];
  int * cell_info_dev=NULL;
  double * box_info_dev=NULL;
  double * out_c_dev=NULL, * in_c_dev=NULL;
  int * out_t_dev=NULL, * in_t_dev=NULL, * mapping_dev=NULL, * int_data_dev=NULL;
  deepmd::malloc_device_memory_sync(cell_info_dev, cell_info);
  deepmd::malloc_device_memory_sync(box_info_dev, box_info);
  deepmd::malloc_device_memory_sync(in_c_dev, posi);
  deepmd::malloc_device_memory_sync(in_t_dev, atype);
  deepmd::malloc_device_memory(out_c_dev, mem_size * 3);
  deepmd::malloc_device_memory(out_t_dev, mem_size);
  deepmd::malloc_device_memory(mapping_dev, mem_size);
  deepmd::malloc_device_memory(int_data_dev, nloc*3+loc_cellnum+total_cellnum*3+total_cellnum*3+loc_cellnum+1+total_cellnum+1+nloc);
  region_dev.boxt = box_info_dev;
  region_dev.rec_boxt = box_info_dev + 9;
  int ret = deepmd::copy_coord_gpu(
      out_c_dev, 
      out_t_dev, 
      mapping_dev, 
      &nall,
      int_data_dev,
      in_c_dev,
      in_t_dev,
      nloc,
      mem_size,
      loc_cellnum,
      total_cellnum,
      cell_info_dev,
      region_dev);
  region_dev.boxt = new_boxt;
  region_dev.rec_boxt = new_rec_boxt;
  deepmd::memcpy_device_to_host(out_c_dev, out_c);
  deepmd::memcpy_device_to_host(out_t_dev, out_t);
  deepmd::memcpy_device_to_host(mapping_dev, mapping);
  deepmd::delete_device_memory(cell_info_dev);
  deepmd::delete_device_memory(box_info_dev);
  deepmd::delete_device_memory(in_c_dev);
  deepmd::delete_device_memory(in_t_dev);
  deepmd::delete_device_memory(out_c_dev);
  deepmd::delete_device_memory(out_t_dev);
  deepmd::delete_device_memory(mapping_dev);
  deepmd::delete_device_memory(int_data_dev);
  EXPECT_EQ(ret, 1);
  // EXPECT_EQ(nall, expected_nall);
  // std::cout << "---------------------" 
  // 	    << nloc << " " 
  // 	    << nall << std::endl;
}
#endif //GOOGLE_CUDA
