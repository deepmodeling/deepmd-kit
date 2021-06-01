#include "hip/hip_runtime.h"
#include "device.h"
#include "prod_virial.h"
#include "rocprim/rocprim.hpp"

template <
    typename FPTYPE,
    int      THREADS_PER_BLOCK>
__global__ void atom_virial_reduction(
    FPTYPE * virial, 
    const FPTYPE * atom_virial,
    const int nall)
{
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ FPTYPE data[THREADS_PER_BLOCK];
    data[tid] = 0.f;
    for (int ii = tid; ii < nall; ii += THREADS_PER_BLOCK) {
        data[tid] += atom_virial[ii * 9 + bid];
    }
    __syncthreads(); 
    // do reduction in shared memory
    for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
        if (tid < ii) {
            data[tid] += data[tid + ii];
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) virial[bid] = data[0];
}

template<typename FPTYPE>
__global__ void virial_deriv_wrt_neighbors_a(
    FPTYPE * virial, 
    FPTYPE * atom_virial,
    const FPTYPE * net_deriv,
    const FPTYPE * in_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int nloc,
    const int nnei) 
{
  // idx -> nloc
  // idy -> nnei
  // idz = dd0 * 3 + dd1
  // dd0 = idz / 3
  // dd1 = idz % 3
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y;
  const unsigned int idz = threadIdx.y;
  const unsigned int idw = threadIdx.z;
  const int ndescrpt = nnei * 4;
  if (idx >= nloc) {
      return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
      return;
  }
  // atomicAdd(
  //    virial + idz, 
  //    net_deriv[idx * ndescrpt + idy * 4 + idw] * rij[idx * nnei * 3 + idy * 3 + idz / 3] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz % 3]);
  atomicAdd(
      atom_virial + j_idx * 9 + idz, 
      net_deriv[idx * ndescrpt + idy * 4 + idw] * rij[idx * nnei * 3 + idy * 3 + idz % 3] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz / 3]);
}

template<typename FPTYPE>
__global__ void virial_deriv_wrt_neighbors_r(
    FPTYPE * virial, 
    FPTYPE * atom_virial,
    const FPTYPE * net_deriv,
    const FPTYPE * in_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int nloc,
    const int nnei) 
{
    // idx -> nloc
    // idy -> nnei
    // idz = dd0 * 3 + dd1
    // dd0 = idz / 3
    // dd1 = idz % 3
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;
    const int ndescrpt = nnei * 1;

    if (idx >= nloc) {
        return;
    }
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    // atomicAdd(
    //    virial + idz, 
    //    net_deriv[idx * ndescrpt + idy * 4 + idw] * rij[idx * nnei * 3 + idy * 3 + idz / 3] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz % 3]);
    atomicAdd(
        atom_virial + j_idx * 9 + idz, 
        net_deriv[idx * ndescrpt + idy] * rij[idx * nnei * 3 + idy * 3 + idz % 3] * in_deriv[idx * ndescrpt * 3 + idy * 3 + idz / 3]);
}

namespace deepmd {
template<typename FPTYPE>
void prod_virial_a_gpu_rocm(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
{
  hipErrcheck(hipMemset(
      virial, 
      0.0, sizeof(FPTYPE) * 9));
  hipErrcheck(hipMemset(
      atom_virial, 
      0.0, sizeof(FPTYPE) * 9 * nall));
    
  const int LEN = 16;
  int nblock = (nloc + LEN -1) / LEN;
  dim3 block_grid(nblock, nnei);
  dim3 thread_grid(LEN, 9, 4);
  // compute virial of a frame
  hipLaunchKernelGGL(virial_deriv_wrt_neighbors_a, block_grid, thread_grid, 0, 0, 
      virial, atom_virial, 
      net_deriv, in_deriv, rij, nlist, nloc, nnei);
  // reduction atom_virial to virial
  hipLaunchKernelGGL(HIP_KERNEL_NAME(atom_virial_reduction<FPTYPE, TPB>), 9, TPB, 0, 0, 
      virial, 
      atom_virial, nall);
}

template<typename FPTYPE>
void prod_virial_r_gpu_rocm(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
{
  hipErrcheck(hipMemset(
      virial, 
      0.0, sizeof(FPTYPE) * 9));
  hipErrcheck(hipMemset(
      atom_virial, 
      0.0, sizeof(FPTYPE) * 9 * nall));
    
  const int LEN = 16;
  int nblock = (nloc + LEN -1) / LEN;
  dim3 block_grid(nblock, nnei);
  dim3 thread_grid(LEN, 9);
  // compute virial of a frame
  hipLaunchKernelGGL(virial_deriv_wrt_neighbors_r, block_grid, thread_grid, 0, 0, 
      virial, atom_virial, 
      net_deriv, in_deriv, rij, nlist, nloc, nnei);
  // reduction atom_virial to virial
  hipLaunchKernelGGL(HIP_KERNEL_NAME(atom_virial_reduction<FPTYPE, TPB>), 9, TPB, 0, 0, 
    virial, 
    atom_virial, nall);
}

template void prod_virial_a_gpu_rocm<float>(float * virial, float * atom_virial, const float * net_deriv, const float * in_deriv, const float * rij, const int * nlist, const int nloc, const int nall, const int nnei);
template void prod_virial_a_gpu_rocm<double>(double * virial, double * atom_virial, const double * net_deriv, const double * in_deriv, const double * rij, const int * nlist, const int nloc, const int nall, const int nnei);
template void prod_virial_r_gpu_rocm<float>(float * virial, float * atom_virial, const float * net_deriv, const float * in_deriv, const float * rij, const int * nlist, const int nloc, const int nall, const int nnei);
template void prod_virial_r_gpu_rocm<double>(double * virial, double * atom_virial, const double * net_deriv, const double * in_deriv, const double * rij, const int * nlist, const int nloc, const int nall, const int nnei);
}
