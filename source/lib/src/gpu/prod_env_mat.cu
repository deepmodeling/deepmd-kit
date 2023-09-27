#if GOOGLE_CUDA
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>
#elif TENSORFLOW_USE_ROCM
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#error "should not touch here"
#endif

#include "device.h"
#include "fmt_nlist.h"
#include "prod_env_mat.h"

__device__ inline double _sqrt(double x) { return sqrt(x); }
__device__ inline float _sqrt(float x) { return sqrtf(x); }
__device__ inline double _rsqrt(double x) { return rsqrt(x); }
__device__ inline float _rsqrt(float x) { return rsqrtf(x); }

// common part of prod_env_mat
template <typename Key, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS) __global__
    void BlockSortKernel(Key* d_in,
                         Key* d_out)  // Tile of output
{
  enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for
  // coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD,
                         cub::BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadT;
  // Specialize BlockRadixSort type for our thread block
  typedef cub::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD>
      BlockRadixSortT;
  // Shared memory
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockRadixSortT::TempStorage sort;
  } temp_storage;
  // Per-thread tile items
  Key items[ITEMS_PER_THREAD];
  // Our current block's offset
  int_64 block_offset = (int_64)blockIdx.x * TILE_SIZE;
  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
  // Barrier for smem reuse
  __syncthreads();
  // Sort keys
  BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(items);
  // Store output in striped fashion
  cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset,
                                         items);
}

template <typename FPTYPE>
__device__ inline FPTYPE dev_dot(FPTYPE* arr1, FPTYPE* arr2) {
  return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

template <typename FPTYPE>
__device__ inline void spline5_switch(
    FPTYPE& vv, FPTYPE& dd, FPTYPE& xx, const float& rmin, const float& rmax) {
  if (xx < rmin) {
    dd = (FPTYPE)0.;
    vv = (FPTYPE)1.;
  } else if (xx < rmax) {
    FPTYPE uu = (xx - rmin) / (rmax - rmin);
    FPTYPE du = (FPTYPE)1. / (rmax - rmin);
    vv = uu * uu * uu *
             ((FPTYPE)-6. * uu * uu + (FPTYPE)15. * uu - (FPTYPE)10.) +
         (FPTYPE)1.;
    dd = ((FPTYPE)3. * uu * uu *
              ((FPTYPE)-6. * uu * uu + (FPTYPE)15. * uu - (FPTYPE)10.) +
          uu * uu * uu * ((FPTYPE)-12. * uu + (FPTYPE)15.)) *
         du;
  } else {
    dd = (FPTYPE)0.;
    vv = (FPTYPE)0.;
  }
}

template <typename FPTYPE>
__device__ inline uint_64 encoding_nbor_info(const int type,
                                             const FPTYPE dist,
                                             const int index) {
  // nbor info checking:
  // the type of nbor atom must be smaller than 128
  // the distance of center atom between nbor atom must be smaller than 128
  // the index of nbor atom(including ghost region) must be smaller than
  // 16777216(1 << 24)
  if (type >= 128 || dist >= (FPTYPE)128.0 || index >= (1 << 24)) {
#if GOOGLE_CUDA
    asm("trap;");
#elif TENSORFLOW_USE_ROCM
    __builtin_trap();
#else
#error "should not touch here"
#endif
  }
  return ((uint_64)type << 57) +
         (uint_64)((double)dist * ((uint_64)1 << 50)) / (1 << 24) * (1 << 24) +
         index;
}

__device__ inline void decoding_nbor_info(int& type,
                                          int& index,
                                          const uint_64 key) {
  type = key >> 57;
  index = key & 0xFFFFFF;
}

template <typename FPTYPE>
__global__ void get_i_idx(FPTYPE* i_idx, const int nloc, const FPTYPE* ilist) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nloc) {
    return;
  }
  i_idx[ilist[idx]] = idx;
}

template <typename FPTYPE>
__global__ void format_nlist_fill_a(uint_64* key,
                                    const FPTYPE* coord,
                                    const int* type,
                                    const int* numneigh,
                                    int** firstneigh,
                                    const float rcut,
                                    int* i_idx,
                                    const int MAX_NBOR_SIZE) {
  // <<<nloc, MAX_NBOR_SIZE>>>
  const int_64 idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  const int nsize = numneigh[i_idx[idx]];
  if (idy >= nsize) {
    return;
  }

  const int* nei_idx = firstneigh[i_idx[idx]];
  // dev_copy(nei_idx, &jlist[jrange[i_idx]], nsize);
  uint_64* key_in = key + idx * MAX_NBOR_SIZE;
  FPTYPE diff[3];
  const int& j_idx = nei_idx[idy];
  if (type[j_idx] < 0) {
    return;
  }
  for (int dd = 0; dd < 3; dd++) {
    diff[dd] = coord[j_idx * 3 + dd] - coord[idx * 3 + dd];
  }
  FPTYPE rr = _sqrt(dev_dot(diff, diff));
  if (rr <= rcut) {
    key_in[idy] = encoding_nbor_info(type[j_idx], rr, j_idx);
  }
}

template <typename FPTYPE>
__global__ void fill_nei_iter(int* nei_iter_dev,
                              const FPTYPE* key,
                              const int nloc,
                              const int max_nbor_size,
                              const int sec_size) {
  int_64 row = blockIdx.x;
  int col = blockIdx.y * blockDim.x + threadIdx.x;
  const FPTYPE* key_out = key + nloc * max_nbor_size + row * max_nbor_size;
  int nei_type_cur = -1, nbor_idx_cur = 0;
  int nei_type_pre = -1, nbor_idx_pre = 0;
  if (col < max_nbor_size && key_out[col] != key_out[max_nbor_size - 1]) {
    if (col >= 1) {
      decoding_nbor_info(nei_type_pre, nbor_idx_pre, key_out[col - 1]);
    }
    decoding_nbor_info(nei_type_cur, nbor_idx_cur, key_out[col]);
  }
  if (nei_type_cur != nei_type_pre) {
    nei_iter_dev[row * sec_size + nei_type_cur] = col;
  }
}

template <typename FPTYPE>
__global__ void format_nlist_fill_b(int* nlist,
                                    const int nlist_size,
                                    const int nloc,
                                    FPTYPE* key,
                                    const int* sec,
                                    const int sec_size,
                                    int* nei_iter_dev,
                                    const int max_nbor_size) {
  int_64 row = blockIdx.x;
  int col = blockIdx.y * blockDim.x + threadIdx.x;
  int* nei_iter = nei_iter_dev + row * sec_size;
  FPTYPE* key_out = key + nloc * max_nbor_size + row * max_nbor_size;
  int* row_nlist = nlist + row * nlist_size;
  if (col < max_nbor_size) {
    if (key_out[col] != key_out[max_nbor_size - 1]) {
      int nei_type = 0, nbor_idx = 0;
      decoding_nbor_info(nei_type, nbor_idx, key_out[col]);
      int out_indx = col - nei_iter[nei_type] + sec[nei_type];
      if (out_indx < sec[nei_type + 1]) {
        row_nlist[out_indx] = nbor_idx;
      }
    }
  }
}

template <typename FPTYPE>
__global__ void encoding_decoding_nbor_info(uint_64* key,
                                            int* out_type,
                                            int* out_index,
                                            const int* in_type,
                                            const FPTYPE* in_dist,
                                            const int* in_index,
                                            const int size_of_array) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size_of_array) {
    return;
  }

  key[idx] = encoding_nbor_info(in_type[idx], in_dist[idx], in_index[idx]);
  decoding_nbor_info(out_type[idx], out_index[idx], key[idx]);
}

template <typename FPTYPE>
void format_nbor_list_256(uint_64* key,
                          const FPTYPE* coord,
                          const int* type,
                          const deepmd::InputNlist& gpu_inlist,
                          const int& nloc,
                          const float& rcut,
                          int* i_idx) {
  const int LEN = 256;
  const int MAX_NBOR_SIZE = 256;
  const int nblock = (MAX_NBOR_SIZE + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, LEN);
  format_nlist_fill_a<<<block_grid, thread_grid>>>(
      key, coord, type, gpu_inlist.numneigh, gpu_inlist.firstneigh, rcut, i_idx,
      MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int ITEMS_PER_THREAD = 4;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS,
  // ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<uint_64, BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<nloc, BLOCK_THREADS>>>(key, key + nloc * MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void format_nbor_list_512(uint_64* key,
                          const FPTYPE* coord,
                          const int* type,
                          const deepmd::InputNlist& gpu_inlist,
                          const int& nloc,
                          const float& rcut,
                          int* i_idx) {
  const int LEN = 256;
  const int MAX_NBOR_SIZE = 512;
  const int nblock = (MAX_NBOR_SIZE + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, LEN);
  format_nlist_fill_a<<<block_grid, thread_grid>>>(
      key, coord, type, gpu_inlist.numneigh, gpu_inlist.firstneigh, rcut, i_idx,
      MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int ITEMS_PER_THREAD = 4;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS,
  // ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<uint_64, BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<nloc, BLOCK_THREADS>>>(key, key + nloc * MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void format_nbor_list_1024(uint_64* key,
                           const FPTYPE* coord,
                           const int* type,
                           const deepmd::InputNlist& gpu_inlist,
                           const int& nloc,
                           const float& rcut,
                           int* i_idx) {
  const int LEN = 256;
  const int MAX_NBOR_SIZE = 1024;
  const int nblock = (MAX_NBOR_SIZE + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, LEN);
  format_nlist_fill_a<<<block_grid, thread_grid>>>(
      key, coord, type, gpu_inlist.numneigh, gpu_inlist.firstneigh, rcut, i_idx,
      MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int ITEMS_PER_THREAD = 8;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS,
  // ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<uint_64, BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<nloc, BLOCK_THREADS>>>(key, key + nloc * MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void format_nbor_list_2048(uint_64* key,
                           const FPTYPE* coord,
                           const int* type,
                           const deepmd::InputNlist& gpu_inlist,
                           const int& nloc,
                           const float& rcut,
                           int* i_idx) {
  const int LEN = 256;
  const int MAX_NBOR_SIZE = 2048;
  const int nblock = (MAX_NBOR_SIZE + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, LEN);
  format_nlist_fill_a<<<block_grid, thread_grid>>>(
      key, coord, type, gpu_inlist.numneigh, gpu_inlist.firstneigh, rcut, i_idx,
      MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int ITEMS_PER_THREAD = 8;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS,
  // ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<uint_64, BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<nloc, BLOCK_THREADS>>>(key, key + nloc * MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void format_nbor_list_4096(uint_64* key,
                           const FPTYPE* coord,
                           const int* type,
                           const deepmd::InputNlist& gpu_inlist,
                           const int& nloc,
                           const float& rcut,
                           int* i_idx) {
  const int LEN = 256;
  const int MAX_NBOR_SIZE = 4096;
  const int nblock = (MAX_NBOR_SIZE + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, LEN);
  format_nlist_fill_a<<<block_grid, thread_grid>>>(
      key, coord, type, gpu_inlist.numneigh, gpu_inlist.firstneigh, rcut, i_idx,
      MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int ITEMS_PER_THREAD = 16;
  const int BLOCK_THREADS = MAX_NBOR_SIZE / ITEMS_PER_THREAD;
  // BlockSortKernel<NeighborInfo, BLOCK_THREADS,
  // ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>> (
  BlockSortKernel<uint_64, BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<nloc, BLOCK_THREADS>>>(key, key + nloc * MAX_NBOR_SIZE);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void compute_env_mat_a(FPTYPE* em,
                                  FPTYPE* em_deriv,
                                  FPTYPE* rij,
                                  const FPTYPE* coord,
                                  const FPTYPE* avg,
                                  const FPTYPE* std,
                                  const int* type,
                                  const int* nlist,
                                  const int nnei,
                                  const float rmin,
                                  const float rmax) {
  // <<<nloc, TPB>>>
  const int_64 bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (type[bid] < 0) {
    return;
  }
  if (tid >= nnei) {
    return;
  }
  const int ndescrpt = nnei * 4;
  const int* row_nlist = nlist + bid * nnei;
  FPTYPE* row_rij = rij + bid * nnei * 3;
  FPTYPE* row_descript = em + bid * nnei * 4;
  FPTYPE* row_descript_deriv = em_deriv + bid * nnei * 12;
  for (int ii = tid; ii < nnei; ii += THREADS_PER_BLOCK) {
    const int idx_value = ii * 4;   // 4 components
    const int idx_deriv = ii * 12;  // 4 components time 3 directions
    if (row_nlist[ii] >= 0) {
      FPTYPE rr[3] = {(FPTYPE)0.};
      FPTYPE dd[4] = {(FPTYPE)0.};
      FPTYPE vv[12] = {(FPTYPE)0.};
      const int j_idx = row_nlist[ii];
      for (int kk = 0; kk < 3; kk++) {
        rr[kk] = coord[j_idx * 3 + kk] - coord[bid * 3 + kk];
        row_rij[ii * 3 + kk] = rr[kk];
      }
      // const FPTYPE * rr = &row_rij[ii * 3];
      FPTYPE nr2 = dev_dot(rr, rr);
      FPTYPE inr = _rsqrt(nr2);
      FPTYPE nr = nr2 * inr;
      FPTYPE inr2 = inr * inr;
      FPTYPE inr4 = inr2 * inr2;
      FPTYPE inr3 = inr4 * nr;
      FPTYPE sw, dsw;
      spline5_switch(sw, dsw, nr, rmin, rmax);
      dd[0] = ((FPTYPE)1. / nr);  //* sw;
      dd[1] = (rr[0] / nr2);      //* sw;
      dd[2] = (rr[1] / nr2);      //* sw;
      dd[3] = (rr[2] / nr2);      //* sw;
      vv[0] = (rr[0] * inr3 * sw -
               dd[0] * dsw * rr[0] *
                   inr);  // avg[type[(idx_deriv + 0) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 0) % (ndescrpt * 3)) / 3];
      vv[1] = (rr[1] * inr3 * sw -
               dd[0] * dsw * rr[1] *
                   inr);  // avg[type[(idx_deriv + 1) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 1) % (ndescrpt * 3)) / 3];
      vv[2] = (rr[2] * inr3 * sw -
               dd[0] * dsw * rr[2] *
                   inr);  // avg[type[(idx_deriv + 2) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 2) % (ndescrpt * 3)) / 3];
      // ****deriv of component x/r2
      vv[3] = (((FPTYPE)2. * rr[0] * rr[0] * inr4 - inr2) * sw -
               dd[1] * dsw * rr[0] *
                   inr);  // avg[type[(idx_deriv + 3) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 3) % (ndescrpt * 3)) / 3];
      vv[4] = (((FPTYPE)2. * rr[0] * rr[1] * inr4) * sw -
               dd[1] * dsw * rr[1] *
                   inr);  // avg[type[(idx_deriv + 4) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 4) % (ndescrpt * 3)) / 3];
      vv[5] = (((FPTYPE)2. * rr[0] * rr[2] * inr4) * sw -
               dd[1] * dsw * rr[2] *
                   inr);  // avg[type[(idx_deriv + 5) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 5) % (ndescrpt * 3)) / 3];
      // ***deriv of component y/r2
      vv[6] = (((FPTYPE)2. * rr[1] * rr[0] * inr4) * sw -
               dd[2] * dsw * rr[0] *
                   inr);  // avg[type[(idx_deriv + 6) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 6) % (ndescrpt * 3)) / 3];
      vv[7] = (((FPTYPE)2. * rr[1] * rr[1] * inr4 - inr2) * sw -
               dd[2] * dsw * rr[1] *
                   inr);  // avg[type[(idx_deriv + 7) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 7) % (ndescrpt * 3)) / 3];
      vv[8] = (((FPTYPE)2. * rr[1] * rr[2] * inr4) * sw -
               dd[2] * dsw * rr[2] *
                   inr);  // avg[type[(idx_deriv + 8) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 8) % (ndescrpt * 3)) / 3];
      // ***deriv of component z/r2
      vv[9] = (((FPTYPE)2. * rr[2] * rr[0] * inr4) * sw -
               dd[3] * dsw * rr[0] *
                   inr);  // avg[type[(idx_deriv + 9) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 9) % (ndescrpt * 3)) / 3];
      vv[10] =
          (((FPTYPE)2. * rr[2] * rr[1] * inr4) * sw -
           dd[3] * dsw * rr[1] *
               inr);  // avg[type[(idx_deriv + 10) / (ndescrpt * 3)] * ndescrpt
                      // + ((idx_deriv + 10) % (ndescrpt * 3)) / 3];
      vv[11] =
          (((FPTYPE)2. * rr[2] * rr[2] * inr4 - inr2) * sw -
           dd[3] * dsw * rr[2] *
               inr);  // avg[type[(idx_deriv + 11) / (ndescrpt * 3)] * ndescrpt
                      // + ((idx_deriv + 11) % (ndescrpt * 3)) / 3];
      // 4 value components
      dd[0] *= sw;  // * em[idx * ndescrpt + idx_value + 0]);// - avg[type[idx]
                    // * ndescrpt + idx_value + 0]) / std[type[idx] * ndescrpt +
                    // idx_value + 0];
      dd[1] *= sw;  // * em[idx * ndescrpt + idx_value + 1]);// - avg[type[idx]
                    // * ndescrpt + idx_value + 1]) / std[type[idx] * ndescrpt +
                    // idx_value + 1];
      dd[2] *= sw;  // * em[idx * ndescrpt + idx_value + 2]);// - avg[type[idx]
                    // * ndescrpt + idx_value + 2]) / std[type[idx] * ndescrpt +
                    // idx_value + 2];
      dd[3] *= sw;  // * em[idx * ndescrpt + idx_value + 3]);// - avg[type[idx]
                    // * ndescrpt + idx_value + 3]) / std[type[idx] * ndescrpt +
                    // idx_value + 3];
      for (int ii = 0; ii < 12; ii++) {
        row_descript_deriv[idx_deriv + ii] =
            vv[ii] / std[type[bid] * ndescrpt + idx_value + ii / 3];
      }
      for (int ii = 0; ii < 4; ii++) {
        row_descript[idx_value + ii] =
            (dd[ii] - avg[type[bid] * ndescrpt + idx_value + ii]) /
            std[type[bid] * ndescrpt + idx_value + ii];
      }
    } else {
      // TODO: move it to the memset.
      row_descript[idx_value] -= avg[type[bid] * ndescrpt + idx_value] /
                                 std[type[bid] * ndescrpt + idx_value];
    }
  }
}

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void compute_env_mat_r(FPTYPE* em,
                                  FPTYPE* em_deriv,
                                  FPTYPE* rij,
                                  const FPTYPE* coord,
                                  const FPTYPE* avg,
                                  const FPTYPE* std,
                                  const int* type,
                                  const int* nlist,
                                  const int nnei,
                                  const float rmin,
                                  const float rmax) {
  // <<<nloc, TPB>>>
  const int_64 bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (tid >= nnei) {
    return;
  }
  const int ndescrpt = nnei;
  const int* row_nlist = nlist + bid * nnei;
  FPTYPE* row_rij = rij + bid * nnei * 3;
  FPTYPE* row_em = em + bid * nnei;
  FPTYPE* row_em_deriv = em_deriv + bid * nnei * 3;
  for (int ii = tid; ii < nnei; ii += THREADS_PER_BLOCK) {
    const int idx_value = ii;      // 4 components
    const int idx_deriv = ii * 3;  // 4 components time 3 directions
    if (row_nlist[ii] >= 0) {
      FPTYPE rr[3] = {0};
      FPTYPE vv[3] = {0};
      FPTYPE dd = 0;
      const int& j_idx = row_nlist[ii];
      for (int kk = 0; kk < 3; kk++) {
        rr[kk] = coord[j_idx * 3 + kk] - coord[bid * 3 + kk];
        row_rij[ii * 3 + kk] = rr[kk];
      }
      // const FPTYPE * rr = &row_rij[ii * 3];
      FPTYPE nr2 = dev_dot(rr, rr);
      FPTYPE inr = _rsqrt(nr2);
      FPTYPE nr = nr2 * inr;
      FPTYPE inr2 = inr * inr;
      FPTYPE inr4 = inr2 * inr2;
      FPTYPE inr3 = inr4 * nr;
      FPTYPE sw, dsw;
      spline5_switch(sw, dsw, nr, rmin, rmax);
      dd = ((FPTYPE)1. / nr);  //* sw;
      vv[0] = (rr[0] * inr3 * sw -
               dd * dsw * rr[0] *
                   inr);  // avg[type[(idx_deriv + 0) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 0) % (ndescrpt * 3)) / 3];
      vv[1] = (rr[1] * inr3 * sw -
               dd * dsw * rr[1] *
                   inr);  // avg[type[(idx_deriv + 1) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 1) % (ndescrpt * 3)) / 3];
      vv[2] = (rr[2] * inr3 * sw -
               dd * dsw * rr[2] *
                   inr);  // avg[type[(idx_deriv + 2) / (ndescrpt * 3)] *
                          // ndescrpt + ((idx_deriv + 2) % (ndescrpt * 3)) / 3];

      // 4 value components
      dd *= sw;  // * em[idx * ndescrpt + idx_value + 0]);// - avg[type[idx] *
                 // ndescrpt + idx_value + 0]) / std[type[idx] * ndescrpt +
                 // idx_value + 0];
      for (int ii = 0; ii < 3; ii++) {
        row_em_deriv[idx_deriv + ii] =
            vv[ii] / std[type[bid] * ndescrpt + idx_value + ii / 3];
      }
      row_em[idx_value] = (dd - avg[type[bid] * ndescrpt + idx_value]) /
                          std[type[bid] * ndescrpt + idx_value];
    } else {
      // TODO: move it to the memset.
      row_em[idx_value] -= avg[type[bid] * ndescrpt + idx_value] /
                           std[type[bid] * ndescrpt + idx_value];
    }
  }
}

namespace deepmd {
template <typename FPTYPE>
void format_nbor_list_gpu(int* nlist,
                          const FPTYPE* coord,
                          const int* type,
                          const deepmd::InputNlist& gpu_inlist,
                          int* array_int,
                          uint_64* array_longlong,
                          const int max_nbor_size,
                          const int nloc,
                          const int nall,
                          const float rcut,
                          const std::vector<int> sec) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int LEN = 256;
  const int nnei = sec.back();
  const int nblock = (nloc + LEN - 1) / LEN;
  int* sec_dev = array_int;
  int* nei_iter = array_int + sec.size();  // = new int[sec_size];
  int* i_idx = array_int + sec.size() + nloc * sec.size();
  uint_64* key = array_longlong;
  assert(max_nbor_size == 256 || max_nbor_size == 512 ||
         max_nbor_size == 1024 || max_nbor_size == 2048 ||
         max_nbor_size == 4096);
  DPErrcheck(gpuMemset(nlist, -1, sizeof(int) * int_64(nloc) * nnei));
  DPErrcheck(gpuMemset(key, 0xffffffff,
                       sizeof(uint_64) * int_64(nloc) * max_nbor_size));
  DPErrcheck(gpuMemcpy(sec_dev, &sec[0], sizeof(int) * sec.size(),
                       gpuMemcpyHostToDevice));

  get_i_idx<<<nblock, LEN>>>(i_idx, nloc, gpu_inlist.ilist);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());

  if (max_nbor_size == 256) {
    format_nbor_list_256(key, coord, type, gpu_inlist, nloc, rcut, i_idx);
  } else if (max_nbor_size == 512) {
    format_nbor_list_512(key, coord, type, gpu_inlist, nloc, rcut, i_idx);
  } else if (max_nbor_size == 1024) {
    format_nbor_list_1024(key, coord, type, gpu_inlist, nloc, rcut, i_idx);
  } else if (max_nbor_size == 2048) {
    format_nbor_list_2048(key, coord, type, gpu_inlist, nloc, rcut, i_idx);
  } else if (max_nbor_size == 4096) {
    format_nbor_list_4096(key, coord, type, gpu_inlist, nloc, rcut, i_idx);
  }

  fill_nei_iter<<<dim3(nloc, (max_nbor_size + LEN - 1) / LEN), LEN>>>(
      nei_iter, key, nloc, max_nbor_size, sec.size());

  format_nlist_fill_b<<<dim3(nloc, (max_nbor_size + LEN - 1) / LEN), LEN>>>(
      nlist, nnei, nloc, key, sec_dev, sec.size(), nei_iter, max_nbor_size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void prod_env_mat_a_gpu(FPTYPE* em,
                        FPTYPE* em_deriv,
                        FPTYPE* rij,
                        int* nlist,
                        const FPTYPE* coord,
                        const int* type,
                        const InputNlist& gpu_inlist,
                        int* array_int,
                        uint_64* array_longlong,
                        const int max_nbor_size,
                        const FPTYPE* avg,
                        const FPTYPE* std,
                        const int nloc,
                        const int nall,
                        const float rcut,
                        const float rcut_smth,
                        const std::vector<int> sec,
                        const int* f_type) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  if (f_type == NULL) {
    f_type = type;
  }
  const int nnei = sec.back();
  const int ndescrpt = nnei * 4;
  DPErrcheck(gpuMemset(em, 0, sizeof(FPTYPE) * int_64(nloc) * ndescrpt));
  DPErrcheck(
      gpuMemset(em_deriv, 0, sizeof(FPTYPE) * int_64(nloc) * ndescrpt * 3));
  DPErrcheck(gpuMemset(rij, 0, sizeof(FPTYPE) * int_64(nloc) * nnei * 3));

  format_nbor_list_gpu(nlist, coord, f_type, gpu_inlist, array_int,
                       array_longlong, max_nbor_size, nloc, nall, rcut, sec);
  nborErrcheck(gpuGetLastError());
  nborErrcheck(gpuDeviceSynchronize());

  compute_env_mat_a<FPTYPE, TPB><<<nloc, TPB>>>(
      em, em_deriv, rij, coord, avg, std, type, nlist, nnei, rcut_smth, rcut);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void prod_env_mat_r_gpu(FPTYPE* em,
                        FPTYPE* em_deriv,
                        FPTYPE* rij,
                        int* nlist,
                        const FPTYPE* coord,
                        const int* type,
                        const InputNlist& gpu_inlist,
                        int* array_int,
                        uint_64* array_longlong,
                        const int max_nbor_size,
                        const FPTYPE* avg,
                        const FPTYPE* std,
                        const int nloc,
                        const int nall,
                        const float rcut,
                        const float rcut_smth,
                        const std::vector<int> sec) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int nnei = sec.back();
  const int ndescrpt = nnei * 1;
  DPErrcheck(gpuMemset(em, 0, sizeof(FPTYPE) * int_64(nloc) * ndescrpt));
  DPErrcheck(
      gpuMemset(em_deriv, 0, sizeof(FPTYPE) * int_64(nloc) * ndescrpt * 3));
  DPErrcheck(gpuMemset(rij, 0, sizeof(FPTYPE) * int_64(nloc) * nnei * 3));

  format_nbor_list_gpu(nlist, coord, type, gpu_inlist, array_int,
                       array_longlong, max_nbor_size, nloc, nall, rcut, sec);
  nborErrcheck(gpuGetLastError());
  nborErrcheck(gpuDeviceSynchronize());

  compute_env_mat_r<FPTYPE, TPB><<<nloc, TPB>>>(
      em, em_deriv, rij, coord, avg, std, type, nlist, nnei, rcut_smth, rcut);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void test_encoding_decoding_nbor_info_gpu(uint_64* key,
                                          int* out_type,
                                          int* out_index,
                                          const int* in_type,
                                          const FPTYPE* in_dist,
                                          const int* in_index,
                                          const int size_of_array) {
  const int nblock = (size_of_array + TPB - 1) / TPB;
  encoding_decoding_nbor_info<<<nblock, TPB>>>(
      key, out_type, out_index, in_type, in_dist, in_index, size_of_array);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template void prod_env_mat_a_gpu<float>(float* em,
                                        float* em_deriv,
                                        float* rij,
                                        int* nlist,
                                        const float* coord,
                                        const int* type,
                                        const InputNlist& gpu_inlist,
                                        int* array_int,
                                        unsigned long long* array_longlong,
                                        const int max_nbor_size,
                                        const float* avg,
                                        const float* std,
                                        const int nloc,
                                        const int nall,
                                        const float rcut,
                                        const float rcut_smth,
                                        const std::vector<int> sec,
                                        const int* f_type);
template void prod_env_mat_a_gpu<double>(double* em,
                                         double* em_deriv,
                                         double* rij,
                                         int* nlist,
                                         const double* coord,
                                         const int* type,
                                         const InputNlist& gpu_inlist,
                                         int* array_int,
                                         unsigned long long* array_longlong,
                                         const int max_nbor_size,
                                         const double* avg,
                                         const double* std,
                                         const int nloc,
                                         const int nall,
                                         const float rcut,
                                         const float rcut_smth,
                                         const std::vector<int> sec,
                                         const int* f_type);
template void prod_env_mat_r_gpu<float>(float* em,
                                        float* em_deriv,
                                        float* rij,
                                        int* nlist,
                                        const float* coord,
                                        const int* type,
                                        const InputNlist& gpu_inlist,
                                        int* array_int,
                                        unsigned long long* array_longlong,
                                        const int max_nbor_size,
                                        const float* avg,
                                        const float* std,
                                        const int nloc,
                                        const int nall,
                                        const float rcut,
                                        const float rcut_smth,
                                        const std::vector<int> sec);
template void prod_env_mat_r_gpu<double>(double* em,
                                         double* em_deriv,
                                         double* rij,
                                         int* nlist,
                                         const double* coord,
                                         const int* type,
                                         const InputNlist& gpu_inlist,
                                         int* array_int,
                                         unsigned long long* array_longlong,
                                         const int max_nbor_size,
                                         const double* avg,
                                         const double* std,
                                         const int nloc,
                                         const int nall,
                                         const float rcut,
                                         const float rcut_smth,
                                         const std::vector<int> sec);
template void format_nbor_list_gpu<float>(int* nlist,
                                          const float* coord,
                                          const int* type,
                                          const deepmd::InputNlist& gpu_inlist,
                                          int* array_int,
                                          uint_64* array_longlong,
                                          const int max_nbor_size,
                                          const int nloc,
                                          const int nall,
                                          const float rcut,
                                          const std::vector<int> sec);
template void format_nbor_list_gpu<double>(int* nlist,
                                           const double* coord,
                                           const int* type,
                                           const deepmd::InputNlist& gpu_inlist,
                                           int* array_int,
                                           uint_64* array_longlong,
                                           const int max_nbor_size,
                                           const int nloc,
                                           const int nall,
                                           const float rcut,
                                           const std::vector<int> sec);
template void test_encoding_decoding_nbor_info_gpu(uint_64* key,
                                                   int* out_type,
                                                   int* out_index,
                                                   const int* in_type,
                                                   const float* in_dist,
                                                   const int* in_index,
                                                   const int size_of_array);
template void test_encoding_decoding_nbor_info_gpu(uint_64* key,
                                                   int* out_type,
                                                   int* out_index,
                                                   const int* in_type,
                                                   const double* in_dist,
                                                   const int* in_index,
                                                   const int size_of_array);
}  // namespace deepmd
