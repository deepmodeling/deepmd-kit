#include "gpu_cuda.h"
#include "prod_env_mat.h"
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
    const int * numneigh,
    int ** firstneigh,
    const float rcut,
    int * i_idx,
    const int MAX_NBOR_SIZE)
{   
  // <<<nloc, MAX_NBOR_SIZE>>>
  const unsigned int idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  
  const int nsize = numneigh[i_idx[idx]];
  if (idy >= nsize) {
    return;
  }

  const int * nei_idx = firstneigh[i_idx[idx]];
  // dev_copy(nei_idx, &jlist[jrange[i_idx]], nsize);
  int_64 * key_in = key + idx * MAX_NBOR_SIZE;
  FPTYPE diff[3];
  const int & j_idx = nei_idx[idy];
  for (int dd = 0; dd < 3; dd++) {
    diff[dd] = coord[j_idx * 3 + dd] - coord[idx * 3 + dd];
  }
  FPTYPE rr = sqrt(dev_dot(diff, diff)); 
  if (rr <= rcut) {
    key_in[idy] = type[j_idx] * 1E14+ (int_64)(rr * 1.0E12) / 10000000 * 10000000 + j_idx;
  }
}

template<typename FPTYPE>
__global__ void format_nlist_fill_b(
    int * nlist,
    const int nlist_size,
    const int nloc,
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
    const int & nei_type = key_out[kk] / 1E14;
    if (nei_iter[nei_type] < sec[nei_type + 1]) {
      row_nlist[nei_iter[nei_type]++] = key_out[kk] % 10000000;
    }
  }
}

template<typename FPTYPE>
void format_nbor_list_1024 (
    int_64 * key,
    const FPTYPE* coord,
    const int* type,
    const deepmd::InputNlist & gpu_inlist,
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
      coord, type, gpu_inlist.numneigh, gpu_inlist.firstneigh, rcut, i_idx, MAX_NBOR_SIZE);
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
    const deepmd::InputNlist & gpu_inlist,
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
      coord, type, gpu_inlist.numneigh, gpu_inlist.firstneigh, rcut, i_idx, MAX_NBOR_SIZE);
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
    const deepmd::InputNlist & gpu_inlist,
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
      coord, type, gpu_inlist.numneigh, gpu_inlist.firstneigh, rcut, i_idx, MAX_NBOR_SIZE);
  const int ITEMS_PER_THREAD = 16;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (
      key, 
      key + nloc * MAX_NBOR_SIZE);
}

template <typename FPTYPE>
void format_nbor_list(    
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const deepmd::InputNlist & gpu_inlist,
    int * array_int,
    int_64 * array_longlong,
    const int max_nbor_size,
    const int nloc, 
    const int nall, 
    const float rcut, 
    const std::vector<int> sec)
{
  const int LEN = 256;
  const int nnei = sec.back();
  const int nblock = (nloc + LEN -1) / LEN;
  int * sec_dev = array_int;
  int * nei_iter = array_int + sec.size(); // = new int[sec_size];
  int * i_idx = array_int + sec.size() + nloc * sec.size();
  int_64 * key = array_longlong;
  assert(max_nbor_size == 1024 || max_nbor_size == 2048 || max_nbor_size == 4096);
  cudaErrcheck(cudaMemset(nlist, -1, sizeof(int) * nloc * nnei));
  cudaErrcheck(cudaMemset(key, 0xffffffff, sizeof(int_64) * nloc * max_nbor_size));
  cudaErrcheck(cudaMemcpy(sec_dev, &sec[0], sizeof(int) * sec.size(), cudaMemcpyHostToDevice));   

  get_i_idx<<<nblock, LEN>>>(
      i_idx,
      nloc, gpu_inlist.ilist);

  if (max_nbor_size == 1024) {
    format_nbor_list_1024 (
        key,
        coord, type, gpu_inlist, nloc, rcut, i_idx); 
  } 
  else if (max_nbor_size == 2048) {
    format_nbor_list_2048 (
        key,
        coord, type, gpu_inlist, nloc, rcut, i_idx); 
  } 
  else if (max_nbor_size == 4096) {
    format_nbor_list_4096 (
        key,
        coord, type, gpu_inlist, nloc, rcut, i_idx); 
  }

  format_nlist_fill_b<<<nblock, LEN>>> (
      nlist,
      nnei, nloc, key, sec_dev, sec.size(), nei_iter, max_nbor_size);
}

template<
    typename FPTYPE,
    int      THREADS_PER_BLOCK>
__global__ void compute_env_mat_a(
    FPTYPE* em,
    FPTYPE* em_deriv,
    FPTYPE* rij,
    const FPTYPE* coord,
    const FPTYPE* avg,
    const FPTYPE* std,
    const int* type,
    const int* nlist,
    const int nnei,
    const float rmin,
    const float rmax)
{   
  // <<<nloc, TPB>>>
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (tid >= nnei) {
    return;
  }
  const int ndescrpt = nnei * 4;
  const int * row_nlist = nlist + bid * nnei;
  FPTYPE * row_rij = rij + bid * nnei * 3;
  FPTYPE * row_descript = em + bid * nnei * 4;
  FPTYPE * row_descript_deriv = em_deriv + bid * nnei * 12;
  for (int ii = tid; ii < nnei; ii += THREADS_PER_BLOCK) {
    const int idx_value = ii * 4;	  // 4 components
    const int idx_deriv = ii * 12;	// 4 components time 3 directions
    if (row_nlist[ii] >= 0) {
      FPTYPE rr[3]  = {0};
      FPTYPE dd[4]  = {0};
      FPTYPE vv[12] = {0};
      const int j_idx = row_nlist[ii];
      for (int kk = 0; kk < 3; kk++) {
        rr[kk] = coord[j_idx * 3 + kk] - coord[bid * 3 + kk];
        row_rij[ii * 3 + kk] = rr[kk];
      }
      // const FPTYPE * rr = &row_rij[ii * 3];
      FPTYPE nr2 = dev_dot(rr, rr);
      FPTYPE inr = 1./sqrt(nr2);
      FPTYPE nr = nr2 * inr;
      FPTYPE inr2 = inr * inr;
      FPTYPE inr4 = inr2 * inr2;
      FPTYPE inr3 = inr4 * nr;
      FPTYPE sw, dsw;
      spline5_switch(sw, dsw, nr, rmin, rmax);
      dd[0] = (1./nr)       ;//* sw;
      dd[1] = (rr[0] / nr2) ;//* sw;
      dd[2] = (rr[1] / nr2) ;//* sw;
      dd[3] = (rr[2] / nr2) ;//* sw;
      vv[0] = (rr[0] * inr3 * sw - dd[0] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 0) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 0) % (ndescrpt * 3)) / 3];
      vv[1] = (rr[1] * inr3 * sw - dd[0] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 1) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 1) % (ndescrpt * 3)) / 3];
      vv[2] = (rr[2] * inr3 * sw - dd[0] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 2) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 2) % (ndescrpt * 3)) / 3];
      // ****deriv of component x/r2
      vv[3] = ((2. * rr[0] * rr[0] * inr4 - inr2) * sw - dd[1] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 3) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 3) % (ndescrpt * 3)) / 3];
      vv[4] = ((2. * rr[0] * rr[1] * inr4	) * sw - dd[1] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 4) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 4) % (ndescrpt * 3)) / 3];
      vv[5] = ((2. * rr[0] * rr[2] * inr4	) * sw - dd[1] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 5) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 5) % (ndescrpt * 3)) / 3];
      // ***deriv of component y/r2
      vv[6] = ((2. * rr[1] * rr[0] * inr4	) * sw - dd[2] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 6) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 6) % (ndescrpt * 3)) / 3];
      vv[7] = ((2. * rr[1] * rr[1] * inr4 - inr2) * sw - dd[2] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 7) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 7) % (ndescrpt * 3)) / 3];
      vv[8] = ((2. * rr[1] * rr[2] * inr4	) * sw - dd[2] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 8) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 8) % (ndescrpt * 3)) / 3];
      // ***deriv of component z/r2 
      vv[9] = ((2. * rr[2] * rr[0] * inr4	) * sw - dd[3] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 9) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 9) % (ndescrpt * 3)) / 3];
      vv[10]= ((2. * rr[2] * rr[1] * inr4	) * sw - dd[3] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 10) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 10) % (ndescrpt * 3)) / 3];
      vv[11]= ((2. * rr[2] * rr[2] * inr4 - inr2) * sw - dd[3] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 11) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 11) % (ndescrpt * 3)) / 3];
      // 4 value components
      dd[0] *= sw; // * em[idx * ndescrpt + idx_value + 0]);// - avg[type[idx] * ndescrpt + idx_value + 0]) / std[type[idx] * ndescrpt + idx_value + 0];
      dd[1] *= sw; // * em[idx * ndescrpt + idx_value + 1]);// - avg[type[idx] * ndescrpt + idx_value + 1]) / std[type[idx] * ndescrpt + idx_value + 1];
      dd[2] *= sw; // * em[idx * ndescrpt + idx_value + 2]);// - avg[type[idx] * ndescrpt + idx_value + 2]) / std[type[idx] * ndescrpt + idx_value + 2];
      dd[3] *= sw; // * em[idx * ndescrpt + idx_value + 3]);// - avg[type[idx] * ndescrpt + idx_value + 3]) / std[type[idx] * ndescrpt + idx_value + 3];
      for (int ii = 0; ii < 12; ii++) {
        row_descript_deriv[idx_deriv + ii] = vv[ii] / std[type[bid] * ndescrpt + idx_value + ii / 3];
      }
      for (int ii = 0; ii < 4; ii++) {  
        row_descript[idx_value + ii] = (dd[ii] - avg[type[bid] * ndescrpt + idx_value + ii]) / std[type[bid] * ndescrpt + idx_value + ii];
      }
    }
    else {
      // TODO: move it to the memset.
      row_descript[idx_value] -= avg[type[bid] * ndescrpt + idx_value] / std[type[bid] * ndescrpt + idx_value];
    }
  }
}

template<
    typename FPTYPE,
    int      THREADS_PER_BLOCK>
__global__ void compute_env_mat_r(
    FPTYPE* em,
    FPTYPE* em_deriv,
    FPTYPE* rij,
    const FPTYPE* coord,
    const FPTYPE* avg,
    const FPTYPE* std,
    const int* type,
    const int* nlist,
    const int nnei,
    const float rmin,
    const float rmax)
{
  // <<<nloc, TPB>>>
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (tid >= nnei) {
    return;
  }
  const int ndescrpt = nnei;
  const int * row_nlist = nlist + bid * nnei;
  FPTYPE * row_rij = rij + bid * nnei * 3;
  FPTYPE * row_em = em + bid * nnei;
  FPTYPE * row_em_deriv = em_deriv + bid * nnei * 3;
  for (int ii = tid; ii < nnei; ii += THREADS_PER_BLOCK) {
    const int idx_value = ii;	  // 4 components
    const int idx_deriv = ii * 3;	// 4 components time 3 directions
    if (row_nlist[ii] >= 0) {
      FPTYPE rr[3]  = {0};
      FPTYPE vv[3]  = {0};
      FPTYPE dd     = 0;
      const int & j_idx = row_nlist[ii];
      for (int kk = 0; kk < 3; kk++) {
        rr[kk] = coord[j_idx * 3 + kk] - coord[bid * 3 + kk];
        row_rij[ii * 3 + kk] = rr[kk];
      }
      // const FPTYPE * rr = &row_rij[ii * 3];
      FPTYPE nr2 = dev_dot(rr, rr);
      FPTYPE inr = 1./sqrt(nr2);
      FPTYPE nr = nr2 * inr;
      FPTYPE inr2 = inr * inr;
      FPTYPE inr4 = inr2 * inr2;
      FPTYPE inr3 = inr4 * nr;
      FPTYPE sw, dsw;
      spline5_switch(sw, dsw, nr, rmin, rmax);
      dd = (1./nr)       ;//* sw;
      vv[0] = (rr[0] * inr3 * sw - dd * dsw * rr[0] * inr); // avg[type[(idx_deriv + 0) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 0) % (ndescrpt * 3)) / 3];
      vv[1] = (rr[1] * inr3 * sw - dd * dsw * rr[1] * inr); // avg[type[(idx_deriv + 1) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 1) % (ndescrpt * 3)) / 3];
      vv[2] = (rr[2] * inr3 * sw - dd * dsw * rr[2] * inr); // avg[type[(idx_deriv + 2) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 2) % (ndescrpt * 3)) / 3];
      
      // 4 value components
      dd *= sw; // * em[idx * ndescrpt + idx_value + 0]);// - avg[type[idx] * ndescrpt + idx_value + 0]) / std[type[idx] * ndescrpt + idx_value + 0];
      for (int ii = 0; ii < 3; ii++) {
        row_em_deriv[idx_deriv + ii] = vv[ii] / std[type[bid] * ndescrpt + idx_value + ii / 3];
      }
      row_em[idx_value] = (dd - avg[type[bid] * ndescrpt + idx_value]) / std[type[bid] * ndescrpt + idx_value];
    }
    else {
      // TODO: move it to the memset.
      row_em[idx_value] -= avg[type[bid] * ndescrpt + idx_value] / std[type[bid] * ndescrpt + idx_value];
    }
  }
}

namespace deepmd {
template <typename FPTYPE>
void prod_env_mat_a_gpu_cuda(    
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const InputNlist & gpu_inlist,
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
  const int nnei = sec.back();
  const int ndescrpt = nnei * 4;
  cudaErrcheck(cudaMemset(em, 0.0, sizeof(FPTYPE) * nloc * ndescrpt));
  cudaErrcheck(cudaMemset(em_deriv, 0.0, sizeof(FPTYPE) * nloc * ndescrpt * 3));

  format_nbor_list(
      nlist, 
      coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, nloc, nall, rcut, sec);

  compute_env_mat_a<FPTYPE, TPB> <<<nloc, TPB>>> (
      em, em_deriv, rij, 
      coord, avg, std, type, nlist, nnei, rcut_smth, rcut);
}

template <typename FPTYPE>
void prod_env_mat_r_gpu_cuda(    
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const InputNlist & gpu_inlist,
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
  const int nnei = sec.back();
  const int ndescrpt = nnei * 1;
  cudaErrcheck(cudaMemset(em, 0.0, sizeof(FPTYPE) * nloc * ndescrpt));
  cudaErrcheck(cudaMemset(em_deriv, 0.0, sizeof(FPTYPE) * nloc * ndescrpt * 3));

  format_nbor_list(
      nlist, 
      coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, nloc, nall, rcut, sec);

  compute_env_mat_r<FPTYPE, TPB> <<<nloc, TPB>>> (
      em, em_deriv, rij, 
      coord, avg, std, type, nlist, nnei, rcut_smth, rcut);
}

template void prod_env_mat_a_gpu_cuda<float>(float * em, float * em_deriv, float * rij, int * nlist, const float * coord, const int * type, const InputNlist & gpu_inlist, int * array_int, unsigned long long * array_longlong, const int max_nbor_size, const float * avg, const float * std, const int nloc, const int nall, const float rcut, const float rcut_smth, const std::vector<int> sec);
template void prod_env_mat_a_gpu_cuda<double>(double * em, double * em_deriv, double * rij, int * nlist, const double * coord, const int * type, const InputNlist & gpu_inlist, int * array_int, unsigned long long * array_longlong, const int max_nbor_size, const double * avg, const double * std, const int nloc, const int nall, const float rcut, const float rcut_smth, const std::vector<int> sec);
template void prod_env_mat_r_gpu_cuda<float>(float * em, float * em_deriv, float * rij, int * nlist, const float * coord, const int * type, const InputNlist & gpu_inlist, int * array_int, unsigned long long * array_longlong, const int max_nbor_size, const float * avg, const float * std, const int nloc, const int nall, const float rcut, const float rcut_smth, const std::vector<int> sec);
template void prod_env_mat_r_gpu_cuda<double>(double * em, double * em_deriv, double * rij, int * nlist, const double * coord, const int * type, const InputNlist & gpu_inlist, int * array_int, unsigned long long * array_longlong, const int max_nbor_size, const double * avg, const double * std, const int nloc, const int nall, const float rcut, const float rcut_smth, const std::vector<int> sec);
}
