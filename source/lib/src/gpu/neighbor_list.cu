#if GOOGLE_CUDA
#include <cub/block/block_scan.cuh>
#elif TENSORFLOW_USE_ROCM
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#error "should not touch here"
#endif

#include "device.h"
#include "neighbor_list.h"
// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct parallel_prefix_scan_op {
  // Running prefix
  int running_total;
  // Constructor
  __device__ parallel_prefix_scan_op(int running_total)
      : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ int operator()(int block_aggregate) {
    int old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <int THREADS_PER_BLOCK>
__global__ void parallel_prefix_scan(int *numneigh,
                                     int *nei_order,
                                     const int *temp_nlist,
                                     const int mem_size,
                                     const int nloc,
                                     const int nall) {
  // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128
  // threads, 4 ints per thread
  typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;
  // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Initialize running total
  parallel_prefix_scan_op prefix_op(0);

  // Have the block iterate over segments of items
  for (int ii = threadIdx.x; ii < nall; ii += THREADS_PER_BLOCK) {
    int block_offset = blockIdx.x * mem_size;
    // Load a segment of consecutive items that are blocked across threads
    int i_data = temp_nlist[block_offset + ii];
    int o_data = i_data == -1 ? 0 : 1;

    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).ExclusiveSum(o_data, o_data, prefix_op);

    __syncthreads();
    // Store scanned items to output segment
    if (i_data != -1) {
      nei_order[block_offset + ii] = o_data;
    }
    // Store numneigh into the output array
    if (ii == nall - 1) {
      o_data += i_data == -1 ? 0 : 1;
      numneigh[blockIdx.x] = o_data;
    }
  }
}

template <typename FPTYPE>
__device__ inline FPTYPE dev_dot(FPTYPE *arr1, FPTYPE *arr2) {
  return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

template <typename FPTYPE>
__global__ void build_nlist(int *ilist,
                            int *temp_nlist,
                            const FPTYPE *c_cpy,
                            const FPTYPE rcut2,
                            const int nloc,
                            const int nall,
                            const int mem_size) {
  const unsigned int atom_idx = blockIdx.x;
  const unsigned int neighbor_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (neighbor_idx < nall) {
    int *neighbor_row = temp_nlist + atom_idx * mem_size;
    if (neighbor_idx == atom_idx) {
      ilist[atom_idx] = atom_idx;
    } else {
      const FPTYPE *ccoord = c_cpy + atom_idx * 3;
      const FPTYPE *ncoord = c_cpy + neighbor_idx * 3;
      FPTYPE diff[3];
      for (int kk = 0; kk < 3; kk++) {
        diff[kk] = ccoord[kk] - ncoord[kk];
      }
      FPTYPE r2 = dev_dot(diff, diff);
      if (r2 < rcut2) {
        neighbor_row[neighbor_idx] = neighbor_idx;
      }
    }
  }
}

__global__ void fill_nlist(int **firstneigh,
                           const int *temp_nlist,
                           const int *nei_order,
                           const int mem_size,
                           const int nall) {
  const unsigned int atom_idx = blockIdx.x;
  const unsigned int neighbor_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (neighbor_idx < nall) {
    const int *in_row = temp_nlist + atom_idx * mem_size;
    int *out_row = firstneigh[atom_idx];
    int nei = in_row[neighbor_idx];
    if (nei != -1) {
      out_row[nei_order[atom_idx * mem_size + neighbor_idx]] = nei;
    }
  }
}

__global__ void map_nlist(int *nlist,
                          const int *nlist_map,
                          const int nloc,
                          const int nnei) {
  int atom_idx = blockIdx.x;
  int nei_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (nei_idx >= nnei) {
    return;
  }
  int nlist_idx = atom_idx * nnei + nei_idx;
  int nlist_item = nlist[nlist_idx];
  if (nlist_item != -1) {
    nlist[nlist_idx] = nlist_map[nlist_item];
  }
}

__global__ void map_nei_info(int *nlist,
                             int *ntype,
                             bool *nmask,
                             const int *type,
                             const int *nlist_map,
                             const int nloc,
                             const int nnei,
                             const int ntypes) {
  int atom_idx = blockIdx.x;
  int nei_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (nei_idx >= nnei) {
    return;
  }
  int nlist_idx = atom_idx * nnei + nei_idx;
  int nlist_item = nlist[nlist_idx];
  int temp = 0;
  if (nlist_item != -1) {
    temp = nlist_map[nlist_item];
    nlist[nlist_idx] = temp;
    ntype[nlist_idx] = type[temp];
    nmask[nlist_idx] = true;
  } else {
    ntype[nlist_idx] = ntypes;
  }
}

__global__ void map_nei_info_noconvert(int *nlist,
                                       int *ntype,
                                       bool *nmask,
                                       const int *type,
                                       const int nloc,
                                       const int nnei,
                                       const int ntypes) {
  int atom_idx = blockIdx.x;
  int nei_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (nei_idx >= nnei) {
    return;
  }
  int nlist_idx = atom_idx * nnei + nei_idx;
  int nlist_item = nlist[nlist_idx];
  if (nlist_item != -1) {
    ntype[nlist_idx] = type[nlist_item];
    nmask[nlist_idx] = true;
  } else {
    ntype[nlist_idx] = ntypes;
  }
}

namespace deepmd {
template <typename FPTYPE>
int build_nlist_gpu(InputNlist &nlist,
                    int *max_list_size,
                    int *nlist_data,
                    const FPTYPE *c_cpy,
                    const int &nloc,
                    const int &nall,
                    const int &mem_size,
                    const float &rcut) {
  if (mem_size < nall) {
    return 1;
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int nblock = (nall + TPB - 1) / TPB;
  int *ilist = nlist.ilist;
  int *numneigh = nlist.numneigh;
  int **firstneigh = nlist.firstneigh;
  DPErrcheck(gpuMemset(nlist_data, -1, sizeof(int) * 2 * nloc * mem_size));
  int *temp_nlist = nlist_data;  // nloc*mem_size
  int *nei_order = temp_nlist + nloc * mem_size;
  nlist.inum = nloc;
  FPTYPE rcut2 = rcut * rcut;

  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, TPB);
  build_nlist<<<block_grid, thread_grid>>>(ilist, temp_nlist, c_cpy, rcut2,
                                           nloc, nall, mem_size);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  parallel_prefix_scan<TPB>
      <<<nloc, TPB>>>(numneigh, nei_order, temp_nlist, mem_size, nloc, nall);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  fill_nlist<<<block_grid, thread_grid>>>(firstneigh, temp_nlist, nei_order,
                                          mem_size, nall);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  int *numneigh_host = new int[nloc];
  DPErrcheck(gpuMemcpy(numneigh_host, numneigh, sizeof(int) * nloc,
                       gpuMemcpyDeviceToHost));
  int max_nei = 0;
  for (int ii = 0; ii < nloc; ii++) {
    if (numneigh_host[ii] > max_nei) {
      max_nei = numneigh_host[ii];
    }
  }
  *max_list_size = max_nei;
  delete[] numneigh_host;
  return 0;
}

void use_nlist_map(int *nlist,
                   const int *nlist_map,
                   const int nloc,
                   const int nnei) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  int nblock = (nnei + TPB - 1) / TPB;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, TPB);
  map_nlist<<<block_grid, thread_grid>>>(nlist, nlist_map, nloc, nnei);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

void use_nei_info_gpu(int *nlist,
                      int *ntype,
                      bool *nmask,
                      const int *type,
                      const int *nlist_map,
                      const int nloc,
                      const int nnei,
                      const int ntypes,
                      const bool b_nlist_map) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  int nblock = (nnei + TPB - 1) / TPB;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(1, TPB);
  DPErrcheck(gpuMemset(ntype, 0, sizeof(int) * nloc * nnei));
  DPErrcheck(gpuMemset(nmask, 0, sizeof(bool) * nloc * nnei));
  if (b_nlist_map) {
    map_nei_info<<<block_grid, thread_grid>>>(nlist, ntype, nmask, type,
                                              nlist_map, nloc, nnei, ntypes);
  } else {
    map_nei_info_noconvert<<<block_grid, thread_grid>>>(
        nlist, ntype, nmask, type, nloc, nnei, ntypes);
  }
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template int build_nlist_gpu<float>(InputNlist &nlist,
                                    int *max_list_size,
                                    int *nlist_data,
                                    const float *c_cpy,
                                    const int &nloc,
                                    const int &nall,
                                    const int &mem_size,
                                    const float &rcut);
template int build_nlist_gpu<double>(InputNlist &nlist,
                                     int *max_list_size,
                                     int *nlist_data,
                                     const double *c_cpy,
                                     const int &nloc,
                                     const int &nall,
                                     const int &mem_size,
                                     const float &rcut);

__global__ void map_filter_ftype(int *ftype_out,
                                 const int *ftype_in,
                                 const int nloc) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < nloc) {
    ftype_out[ii] = ftype_in[ii] >= 0 ? 0 : -1;
  }
}

void filter_ftype_gpu(int *ftype_out, const int *ftype_in, const int nloc) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  int nblock = (nloc + TPB - 1) / TPB;
  map_filter_ftype<<<nblock, TPB>>>(ftype_out, ftype_in, nloc);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

}  // namespace deepmd
