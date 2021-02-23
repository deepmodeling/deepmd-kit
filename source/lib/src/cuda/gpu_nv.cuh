#pragma once
#include "gpu_nv.h"
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

// common part of prod_env_mat
template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void BlockSortKernel(
    Key * d_in,
    Key * d_out)                // Tile of output
{   
  enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
  // Specialize BlockRadixSort type for our thread block
  typedef cub::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
  // Shared memory
  __shared__ union TempStorage
  {
    typename BlockLoadT::TempStorage        load;
    typename BlockRadixSortT::TempStorage   sort;
  } temp_storage;
  // Per-thread tile items
  Key items[ITEMS_PER_THREAD];
  // Our current block's offset
  int block_offset = blockIdx.x * TILE_SIZE;
  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
  // Barrier for smem reuse
  __syncthreads();
  // Sort keys
  BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(items);
  // Store output in striped fashion
  cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items);
}

template<typename FPTYPE>
__device__ inline FPTYPE dev_dot(
    FPTYPE * arr1, 
    FPTYPE * arr2) 
{
  return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

template<typename FPTYPE>
__device__ inline void spline5_switch(
    FPTYPE & vv,
    FPTYPE & dd,
    FPTYPE & xx, 
    const float & rmin, 
    const float & rmax) 
{
  if (xx < rmin) {
    dd = 0;
    vv = 1;
  }
  else if (xx < rmax) {
    FPTYPE uu = (xx - rmin) / (rmax - rmin) ;
    FPTYPE du = 1. / (rmax - rmin) ;
    vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1;
    dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
  }
  else {
    dd = 0;
    vv = 0;
  }
}

template<typename FPTYPE>
__global__ void get_i_idx(
    FPTYPE * i_idx,
    const int nloc,
    const FPTYPE * ilist)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= nloc) {
    return;
  }
  i_idx[ilist[idx]] = idx;
}

template<typename FPTYPE>
__global__ void format_nlist_fill_a(
    int_64 * key,
    const FPTYPE * coord,
    const int * type,
    const int  * jrange,
    const int  * jlist,
    const float rcut,
    int * i_idx,
    const int MAX_NBOR_SIZE)
{   
  // <<<nloc, MAX_NBOR_SIZE>>>
  const unsigned int idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  
  const int nsize = jrange[i_idx[idx] + 1] - jrange[i_idx[idx]];
  if (idy >= nsize) {
    return;
  }
  const int * nei_idx = jlist + jrange[i_idx[idx]];
  // dev_copy(nei_idx, &jlist[jrange[i_idx]], nsize);
  int_64 * key_in = key + idx * MAX_NBOR_SIZE;
  FPTYPE diff[3];
  const int & j_idx = nei_idx[idy];
  for (int dd = 0; dd < 3; dd++) {
    diff[dd] = coord[j_idx * 3 + dd] - coord[idx * 3 + dd];
  }
  FPTYPE rr = sqrt(dev_dot(diff, diff)); 
  if (rr <= rcut) {
    key_in[idy] = type[j_idx] * 1E15+ (int_64)(rr * 1.0E13) / 100000 * 100000 + j_idx;
  }
}

template<typename FPTYPE>
__global__ void format_nlist_fill_b(
    int * nlist,
    const int nlist_size,
    const int nloc,
    const int * jrange,
    const int * jlist,
    FPTYPE * key,
    const int * sec,
    const int sec_size,
    int * nei_iter_dev,
    const int max_nbor_size)
{ 
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= nloc) {
    return;
  }
  
  int * row_nlist = nlist + idx * nlist_size;
  int * nei_iter = nei_iter_dev + idx * sec_size;
  FPTYPE * key_out = key + nloc * max_nbor_size + idx * max_nbor_size;
  for (int ii = 0; ii < sec_size; ii++) {
    nei_iter[ii] = sec[ii];
  }
  
  for (unsigned int kk = 0; key_out[kk] != key_out[max_nbor_size - 1]; kk++) {
    const int & nei_type = key_out[kk] / 1E15;
    if (nei_iter[nei_type] < sec[nei_type + 1]) {
      row_nlist[nei_iter[nei_type]++] = key_out[kk] % 100000;
    }
  }
}

template<typename FPTYPE>
void format_nbor_list_1024 (
    int_64 * key,
    const FPTYPE* coord,
    const int* type,
    const int* jrange,
    const int* jlist,
    const int& nloc,       
    const float& rcut, 
    int * i_idx) 
{   
  const int LEN = 256;
  const int MAX_NBOR_SIZE = 1024;
  const int nblock = (MAX_NBOR_SIZE + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, LEN);
  format_nlist_fill_a<<<block_grid, thread_grid>>> (
      key,
      coord, type, jrange, jlist, rcut, i_idx, MAX_NBOR_SIZE);
  const int ITEMS_PER_THREAD = 8;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (
      key, 
      key + nloc * MAX_NBOR_SIZE);
}

template<typename FPTYPE>
void format_nbor_list_2048 (
    int_64 * key,
    const FPTYPE* coord,
    const int* type,
    const int* jrange,
    const int* jlist,
    const int& nloc,       
    const float& rcut, 
    int * i_idx) 
{   
  const int LEN = 256;
  const int MAX_NBOR_SIZE = 2048;
  const int nblock = (MAX_NBOR_SIZE + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, LEN);
  format_nlist_fill_a<<<block_grid, thread_grid>>> (
      key,
      coord, type, jrange, jlist, rcut, i_idx, MAX_NBOR_SIZE);
  const int ITEMS_PER_THREAD = 8;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (
      key, 
      key + nloc * MAX_NBOR_SIZE);
}

template<typename FPTYPE>
void format_nbor_list_4096 (
    int_64 * key,
    const FPTYPE* coord,
    const int* type,
    const int* jrange,
    const int* jlist,
    const int& nloc,       
    const float& rcut, 
    int * i_idx)
{   
  const int LEN = 256;
  const int MAX_NBOR_SIZE = 4096;
  const int nblock = (MAX_NBOR_SIZE + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, LEN);
  format_nlist_fill_a<<<block_grid, thread_grid>>> (
      key,
      coord, type, jrange, jlist, rcut, i_idx, MAX_NBOR_SIZE);
  const int ITEMS_PER_THREAD = 16;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (
      key, 
      key + nloc * MAX_NBOR_SIZE);
}

template <typename FPTYPE>
void prod_env_mat_common(    
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    int * array_int, 
    int_64 * array_longlong,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec)
{
  const int LEN = 256;
  const int nnei = sec.back();
  const int ndescrpt = nnei * 4;
  int nblock = (nloc + LEN -1) / LEN;
  int * sec_dev = array_int;
  int * nei_iter = array_int + sec.size(); // = new int[sec_size];
  int * i_idx = array_int + sec.size() + nloc * sec.size();
  int_64 * key = array_longlong;
  assert(max_nbor_size == 1024 || max_nbor_size == 2048 || max_nbor_size == 4096);
  cudaErrcheck(cudaMemcpy(sec_dev, &sec[0], sizeof(int) * sec.size(), cudaMemcpyHostToDevice));   
  cudaErrcheck(cudaMemset(key, 0xffffffff, sizeof(int_64) * nloc * max_nbor_size));
  cudaErrcheck(cudaMemset(nlist, -1, sizeof(int) * nloc * nnei));
  cudaErrcheck(cudaMemset(em, 0.0, sizeof(FPTYPE) * nloc * ndescrpt));
  cudaErrcheck(cudaMemset(em_deriv, 0.0, sizeof(FPTYPE) * nloc * ndescrpt * 3));

  get_i_idx<<<nblock, LEN>>>(
      i_idx,
      nloc, ilist);

  if (max_nbor_size == 1024) {
    format_nbor_list_1024 (
        key,
        coord, type, jrange, jlist, nloc, rcut, i_idx); 
  } 
  else if (max_nbor_size == 2048) {
    format_nbor_list_2048 (
        key,
        coord, type, jrange, jlist, nloc, rcut, i_idx); 
  } 
  else if (max_nbor_size == 4096) {
    format_nbor_list_4096 (
        key,
        coord, type, jrange, jlist, nloc, rcut, i_idx); 
  }

  format_nlist_fill_b<<<nblock, LEN>>> (
      nlist,
      nnei, nloc, jrange, jlist, key, sec_dev, sec.size(), nei_iter, max_nbor_size);
}