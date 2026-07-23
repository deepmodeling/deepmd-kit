// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "device.h"
#include "errors.h"
#include "fmt_nlist.h"
#include "neighbor_list.h"

template <typename T>
static std::vector<T> repeat_vector(const std::vector<T>& values,
                                    const int repeats) {
  std::vector<T> result;
  result.reserve(static_cast<size_t>(repeats) * values.size());
  for (int ii = 0; ii < repeats; ++ii) {
    result.insert(result.end(), values.begin(), values.end());
  }
  return result;
}

class TestNeighborList : public ::testing::Test {
 protected:
  std::vector<double> posi = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                              00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                              3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<double> posi_cpy;
  std::vector<int> atype_cpy;
  int ntypes = 2;
  int nloc, nall;
  double rc = 6;
  std::vector<double> boxt = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<int> mapping, ncell, ngcell;
  std::vector<std::vector<int>> expect_nlist_cpy = {
      std::vector<int>({
          33,
          1,
          32,
          34,
          35,
      }),
      std::vector<int>({
          0,
          33,
          32,
          34,
          35,
      }),
      std::vector<int>({
          6,
          3,
          7,
          4,
          5,
      }),
      std::vector<int>({
          6,
          4,
          5,
          2,
          7,
      }),
      std::vector<int>({
          3,
          6,
          5,
          2,
          7,
      }),
      std::vector<int>({
          3,
          6,
          4,
          2,
          7,
      }),
  };

  void SetUp() override {
    SimulationRegion<double> region;
    region.reinitBox(&boxt[0]);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc,
               region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    EXPECT_EQ(expect_nlist_cpy.size(), nloc);
    for (int ii = 0; ii < nloc; ++ii) {
      std::sort(expect_nlist_cpy[ii].begin(), expect_nlist_cpy[ii].end());
    }
  }
};

TEST_F(TestNeighborList, cpu) {
  int mem_size = 10;
  int* ilist = new int[nloc];
  int* numneigh = new int[nloc];
  int** firstneigh = new int*[nloc];
  for (int ii = 0; ii < nloc; ++ii) {
    firstneigh[ii] = new int[mem_size];
  }

  deepmd::InputNlist nlist(nloc, ilist, numneigh, firstneigh);
  int max_list_size;
  int ret = build_nlist_cpu(nlist, &max_list_size, &posi_cpy[0], nloc, nall,
                            mem_size, rc);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(nlist.inum, nloc);
  EXPECT_EQ(max_list_size, 5);
  for (int ii = 0; ii < nloc; ++ii) {
    EXPECT_EQ(nlist.ilist[ii], ii);
    EXPECT_EQ(nlist.numneigh[ii], expect_nlist_cpy[ii].size());
    std::sort(nlist.firstneigh[ii], nlist.firstneigh[ii] + nlist.numneigh[ii]);
    for (int jj = 0; jj < nlist.numneigh[ii]; ++jj) {
      EXPECT_EQ(nlist.firstneigh[ii][jj], expect_nlist_cpy[ii][jj]);
    }
  }

  delete[] ilist;
  delete[] numneigh;
  for (int ii = 0; ii < nloc; ++ii) {
    delete[] firstneigh[ii];
  }
  delete[] firstneigh;
}

TEST_F(TestNeighborList, cpu_multiple_frames) {
  constexpr int nframes = 2;
  const int nrows = nframes * nloc;
  int mem_size = 10;
  std::vector<int> ilist(nrows);
  std::vector<int> numneigh(nrows);
  std::vector<int*> firstneigh(nrows);
  std::vector<int> jlist(static_cast<size_t>(nrows) * mem_size);
  std::vector<double> posi_multi = repeat_vector(posi_cpy, nframes);
  for (int ii = 0; ii < nrows; ++ii) {
    firstneigh[ii] = jlist.data() + ii * mem_size;
  }

  deepmd::InputNlist nlist(nrows, ilist.data(), numneigh.data(),
                           firstneigh.data());
  int max_list_size;
  int ret = build_nlist_cpu(nlist, &max_list_size, posi_multi.data(), nloc,
                            nall, mem_size, rc, nframes);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(nlist.inum, nrows);
  EXPECT_EQ(max_list_size, 5);
  for (int ff = 0; ff < nframes; ++ff) {
    for (int ii = 0; ii < nloc; ++ii) {
      const int row = ff * nloc + ii;
      EXPECT_EQ(nlist.ilist[row], ii);
      EXPECT_EQ(nlist.numneigh[row], expect_nlist_cpy[ii].size());
      std::sort(nlist.firstneigh[row],
                nlist.firstneigh[row] + nlist.numneigh[row]);
      for (int jj = 0; jj < nlist.numneigh[row]; ++jj) {
        EXPECT_EQ(nlist.firstneigh[row][jj], expect_nlist_cpy[ii][jj]);
      }
    }
  }
}

TEST_F(TestNeighborList, cpu_lessmem) {
  int mem_size = 2;
  int* ilist = new int[nloc];
  int* numneigh = new int[nloc];
  int** firstneigh = new int*[nloc];
  for (int ii = 0; ii < nloc; ++ii) {
    firstneigh[ii] = new int[mem_size];
  }

  deepmd::InputNlist nlist(nloc, ilist, numneigh, firstneigh);
  int max_list_size;
  int ret = build_nlist_cpu(nlist, &max_list_size, &posi_cpy[0], nloc, nall,
                            mem_size, rc);
  EXPECT_EQ(ret, 1);
  EXPECT_EQ(nlist.inum, nloc);
  EXPECT_EQ(max_list_size, 5);

  delete[] ilist;
  delete[] numneigh;
  for (int ii = 0; ii < nloc; ++ii) {
    delete[] firstneigh[ii];
  }
  delete[] firstneigh;
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TEST(TestNeighborListGpuConversion, rejects_row_larger_than_capacity) {
  std::vector<int> ilist = {0};
  std::vector<int> numneigh = {GPU_MAX_NBOR_SIZE + 1};
  std::vector<int> neighbors(numneigh[0], 0);
  std::vector<int*> firstneigh = {neighbors.data()};
  deepmd::InputNlist cpu_nlist(1, ilist.data(), numneigh.data(),
                               firstneigh.data());
  deepmd::InputNlist gpu_nlist;
  int* gpu_memory = nullptr;

  // This is deliberately checked before any GPU allocation or copy, so the
  // regression test is deterministic and cannot itself trigger corruption.
  try {
    deepmd::convert_nlist_gpu_device(gpu_nlist, cpu_nlist, gpu_memory,
                                     GPU_MAX_NBOR_SIZE);
    FAIL() << "Expected an oversized neighbor-list row to be rejected";
  } catch (const deepmd::deepmd_exception_nlist_capacity& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("4097 neighbors"), std::string::npos);
    EXPECT_NE(message.find("capacity 4096"), std::string::npos);
  }

  EXPECT_EQ(gpu_nlist.inum, 0);
  EXPECT_EQ(gpu_nlist.ilist, nullptr);
  EXPECT_EQ(gpu_nlist.numneigh, nullptr);
  EXPECT_EQ(gpu_nlist.firstneigh, nullptr);
}

TEST_F(TestNeighborList, gpu) {
  int mem_size = 48;

  int *nlist_data_dev = NULL, *jlist_dev = NULL, *ilist_dev = NULL,
      *numneigh_dev = NULL;
  int** firstneigh_dev = NULL;
  std::vector<int*> temp_firstneigh(nloc);
  double* c_cpy_dev = NULL;

  deepmd::malloc_device_memory(nlist_data_dev, 2 * nloc * mem_size);
  deepmd::malloc_device_memory(jlist_dev, nloc * mem_size);
  deepmd::malloc_device_memory(ilist_dev, nloc);
  deepmd::malloc_device_memory(numneigh_dev, nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    temp_firstneigh[ii] = jlist_dev + ii * mem_size;
  }
  deepmd::malloc_device_memory_sync(firstneigh_dev, temp_firstneigh);
  deepmd::malloc_device_memory_sync(c_cpy_dev, posi_cpy);
  deepmd::InputNlist nlist_dev(nloc, ilist_dev, numneigh_dev, firstneigh_dev);

  int max_list_size;
  int ret = deepmd::build_nlist_gpu(nlist_dev, &max_list_size, nlist_data_dev,
                                    c_cpy_dev, nloc, nall, mem_size, rc);

  EXPECT_EQ(ret, 0);
  int* ilist = new int[nloc];
  int* numneigh = new int[nloc];
  int** firstneigh = new int*[nloc];
  int* jlist = new int[nloc * mem_size];
  deepmd::memcpy_device_to_host(jlist_dev, jlist, nloc * mem_size);
  deepmd::memcpy_device_to_host(ilist_dev, ilist, nloc);
  deepmd::memcpy_device_to_host(numneigh_dev, numneigh, nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    firstneigh[ii] = jlist + ii * mem_size;
  }

  deepmd::InputNlist nlist(nlist_dev.inum, ilist, numneigh, firstneigh);
  EXPECT_EQ(nlist.inum, nloc);
  EXPECT_EQ(max_list_size, 5);
  for (int ii = 0; ii < nloc; ++ii) {
    EXPECT_EQ(nlist.ilist[ii], ii);
    EXPECT_EQ(nlist.numneigh[ii], expect_nlist_cpy[ii].size());
    std::sort(nlist.firstneigh[ii], nlist.firstneigh[ii] + nlist.numneigh[ii]);
    for (int jj = 0; jj < nlist.numneigh[ii]; ++jj) {
      EXPECT_EQ(nlist.firstneigh[ii][jj], expect_nlist_cpy[ii][jj]);
    }
  }

  delete[] ilist;
  delete[] numneigh;
  delete[] jlist;
  delete[] firstneigh;
  deepmd::delete_device_memory(nlist_data_dev);
  deepmd::delete_device_memory(jlist_dev);
  deepmd::delete_device_memory(ilist_dev);
  deepmd::delete_device_memory(numneigh_dev);
  deepmd::delete_device_memory(firstneigh_dev);
  deepmd::delete_device_memory(c_cpy_dev);
}

TEST_F(TestNeighborList, gpu_multiple_frames) {
  constexpr int nframes = 2;
  const int nrows = nframes * nloc;
  int mem_size = 48;

  int *nlist_data_dev = NULL, *jlist_dev = NULL, *ilist_dev = NULL,
      *numneigh_dev = NULL;
  int** firstneigh_dev = NULL;
  std::vector<int*> temp_firstneigh(nrows);
  std::vector<double> posi_multi = repeat_vector(posi_cpy, nframes);
  double* c_cpy_dev = NULL;

  deepmd::malloc_device_memory(nlist_data_dev, 2 * nrows * mem_size);
  deepmd::malloc_device_memory(jlist_dev, nrows * mem_size);
  deepmd::malloc_device_memory(ilist_dev, nrows);
  deepmd::malloc_device_memory(numneigh_dev, nrows);
  for (int ii = 0; ii < nrows; ++ii) {
    temp_firstneigh[ii] = jlist_dev + ii * mem_size;
  }
  deepmd::malloc_device_memory_sync(firstneigh_dev, temp_firstneigh);
  deepmd::malloc_device_memory_sync(c_cpy_dev, posi_multi);
  deepmd::InputNlist nlist_dev(nrows, ilist_dev, numneigh_dev, firstneigh_dev);

  int max_list_size;
  int ret =
      deepmd::build_nlist_gpu(nlist_dev, &max_list_size, nlist_data_dev,
                              c_cpy_dev, nloc, nall, mem_size, rc, nframes);

  EXPECT_EQ(ret, 0);
  std::vector<int> ilist(nrows);
  std::vector<int> numneigh(nrows);
  std::vector<int*> firstneigh(nrows);
  std::vector<int> jlist(nrows * mem_size);
  deepmd::memcpy_device_to_host(jlist_dev, jlist.data(), nrows * mem_size);
  deepmd::memcpy_device_to_host(ilist_dev, ilist.data(), nrows);
  deepmd::memcpy_device_to_host(numneigh_dev, numneigh.data(), nrows);
  for (int ii = 0; ii < nrows; ++ii) {
    firstneigh[ii] = jlist.data() + ii * mem_size;
  }

  deepmd::InputNlist nlist(nlist_dev.inum, ilist.data(), numneigh.data(),
                           firstneigh.data());
  EXPECT_EQ(nlist.inum, nrows);
  EXPECT_EQ(max_list_size, 5);
  for (int ff = 0; ff < nframes; ++ff) {
    for (int ii = 0; ii < nloc; ++ii) {
      const int row = ff * nloc + ii;
      EXPECT_EQ(nlist.ilist[row], ii);
      EXPECT_EQ(nlist.numneigh[row], expect_nlist_cpy[ii].size());
      std::sort(nlist.firstneigh[row],
                nlist.firstneigh[row] + nlist.numneigh[row]);
      for (int jj = 0; jj < nlist.numneigh[row]; ++jj) {
        EXPECT_EQ(nlist.firstneigh[row][jj], expect_nlist_cpy[ii][jj]);
      }
    }
  }

  deepmd::delete_device_memory(nlist_data_dev);
  deepmd::delete_device_memory(jlist_dev);
  deepmd::delete_device_memory(ilist_dev);
  deepmd::delete_device_memory(numneigh_dev);
  deepmd::delete_device_memory(firstneigh_dev);
  deepmd::delete_device_memory(c_cpy_dev);
}

TEST_F(TestNeighborList, gpu_lessmem) {
  int mem_size = 47;

  int *nlist_data_dev = NULL, *jlist_dev = NULL, *ilist_dev = NULL,
      *numneigh_dev = NULL;
  int** firstneigh_dev = NULL;
  std::vector<int*> temp_firstneigh(nloc);
  double* c_cpy_dev = NULL;

  deepmd::malloc_device_memory(nlist_data_dev, 2 * nloc * mem_size);
  deepmd::malloc_device_memory(jlist_dev, nloc * mem_size);
  deepmd::malloc_device_memory(ilist_dev, nloc);
  deepmd::malloc_device_memory(numneigh_dev, nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    temp_firstneigh[ii] = jlist_dev + ii * mem_size;
  }
  deepmd::malloc_device_memory_sync(firstneigh_dev, temp_firstneigh);
  deepmd::malloc_device_memory_sync(c_cpy_dev, posi_cpy);
  deepmd::InputNlist nlist_dev(nloc, ilist_dev, numneigh_dev, firstneigh_dev);

  int max_list_size;
  int ret = deepmd::build_nlist_gpu(nlist_dev, &max_list_size, nlist_data_dev,
                                    c_cpy_dev, nloc, nall, mem_size, rc);

  EXPECT_EQ(ret, 1);
  deepmd::delete_device_memory(nlist_data_dev);
  deepmd::delete_device_memory(jlist_dev);
  deepmd::delete_device_memory(ilist_dev);
  deepmd::delete_device_memory(numneigh_dev);
  deepmd::delete_device_memory(firstneigh_dev);
  deepmd::delete_device_memory(c_cpy_dev);
}

TEST(TestNeighborListStandalone, gpu_tail_segment_prefix_scan) {
  const int nloc = 2;
  const int nall = TPB + 68;
  const int mem_size = nall;
  const double rc = 1.0;

  std::vector<double> coord(nall * 3, 0.0);
  for (int ii = 0; ii < nall; ++ii) {
    coord[ii * 3] = ii * 10.0;
  }

  int *nlist_data_dev = NULL, *jlist_dev = NULL, *ilist_dev = NULL,
      *numneigh_dev = NULL;
  int** firstneigh_dev = NULL;
  std::vector<int*> temp_firstneigh(nloc);
  double* coord_dev = NULL;

  deepmd::malloc_device_memory(nlist_data_dev, 2 * nloc * mem_size);
  deepmd::malloc_device_memory(jlist_dev, nloc * mem_size);
  deepmd::malloc_device_memory(ilist_dev, nloc);
  deepmd::malloc_device_memory(numneigh_dev, nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    temp_firstneigh[ii] = jlist_dev + ii * mem_size;
  }
  deepmd::malloc_device_memory_sync(firstneigh_dev, temp_firstneigh);
  deepmd::malloc_device_memory_sync(coord_dev, coord);
  deepmd::InputNlist nlist_dev(nloc, ilist_dev, numneigh_dev, firstneigh_dev);

  int max_list_size = -1;
  int ret = deepmd::build_nlist_gpu(nlist_dev, &max_list_size, nlist_data_dev,
                                    coord_dev, nloc, nall, mem_size, rc);

  EXPECT_EQ(ret, 0);
  EXPECT_EQ(max_list_size, 0);
  std::vector<int> numneigh(nloc, -1);
  deepmd::memcpy_device_to_host(numneigh_dev, numneigh.data(), nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    EXPECT_EQ(numneigh[ii], 0);
  }

  deepmd::delete_device_memory(nlist_data_dev);
  deepmd::delete_device_memory(jlist_dev);
  deepmd::delete_device_memory(ilist_dev);
  deepmd::delete_device_memory(numneigh_dev);
  deepmd::delete_device_memory(firstneigh_dev);
  deepmd::delete_device_memory(coord_dev);
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
