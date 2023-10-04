#include <cmath>

#if GOOGLE_CUDA
#include <cub/cub.cuh>
#elif TENSORFLOW_USE_ROCM
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#error "should not touch here"
#endif

#include "device.h"
#include "neighbor_list.h"

template <typename FPTYPE>
__global__ void neighbor_stat_g(const FPTYPE* coord,
                                const int* type,
                                const int nloc,
                                const int* ilist,
                                int** firstneigh,
                                const int* numneigh,
                                int* max_nbor_size,
                                FPTYPE* min_nbor_dist,
                                const int ntypes,
                                const int MAX_NNEI) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii >= nloc) {
    return;
  }
  int idx_i = ilist[ii];
  if (type[idx_i] < 0) {
    // set all to 10000
    for (int jj = 0; jj < MAX_NNEI; jj++) {
      min_nbor_dist[ii * MAX_NNEI + jj] = INFINITY;
    }
    return;  // virtual atom
  }
  for (int jj = 0; jj < numneigh[ii]; jj++) {
    int idx_j = firstneigh[ii][jj];
    int type_j = type[idx_j];
    if (type_j < 0) {
      min_nbor_dist[ii * MAX_NNEI + jj] = INFINITY;
      continue;  // virtual atom
    }
    ++max_nbor_size[ii * ntypes + type_j];
    FPTYPE rij[3] = {coord[idx_j * 3 + 0] - coord[idx_i * 3 + 0],
                     coord[idx_j * 3 + 1] - coord[idx_i * 3 + 1],
                     coord[idx_j * 3 + 2] - coord[idx_i * 3 + 2]};
    // we do not need to use the real index
    min_nbor_dist[ii * MAX_NNEI + jj] =
        sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);
  }
  // set others to 10000
  for (int jj = numneigh[ii]; jj < MAX_NNEI; ++jj) {
    min_nbor_dist[ii * MAX_NNEI + jj] = INFINITY;
  }
}

namespace deepmd {

template <typename FPTYPE>
void neighbor_stat_gpu(const FPTYPE* coord,
                       const int* type,
                       const int nloc,
                       const deepmd::InputNlist& gpu_nlist,
                       int* max_nbor_size,
                       FPTYPE* min_nbor_dist,
                       const int ntypes,
                       const int MAX_NNEI) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());

  DPErrcheck(gpuMemset(max_nbor_size, 0, sizeof(int) * int_64(nloc) * ntypes));
  const int nblock_loc = (nloc + TPB - 1) / TPB;
  neighbor_stat_g<<<nblock_loc, TPB>>>(
      coord, type, nloc, gpu_nlist.ilist, gpu_nlist.firstneigh,
      gpu_nlist.numneigh, max_nbor_size, min_nbor_dist, ntypes, MAX_NNEI);

  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template void neighbor_stat_gpu<float>(const float* coord,
                                       const int* type,
                                       const int nloc,
                                       const deepmd::InputNlist& gpu_nlist,
                                       int* max_nbor_size,
                                       float* min_nbor_dist,
                                       const int ntypes,
                                       const int MAX_NNEI);

template void neighbor_stat_gpu<double>(const double* coord,
                                        const int* type,
                                        const int nloc,
                                        const deepmd::InputNlist& gpu_nlist,
                                        int* max_nbor_size,
                                        double* min_nbor_dist,
                                        const int ntypes,
                                        const int MAX_NNEI);
}  // namespace deepmd
