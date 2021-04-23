#include "device.h"
#include "gpu_cuda.h"
#include "prod_virial_grad.h"

template<typename FPTYPE>
__device__ inline FPTYPE dev_dot9(
    const FPTYPE * arr1, 
    const FPTYPE * arr2) 
{
    FPTYPE result = 0.0;
    for(int ii=0; ii<9; ii++){
        result += arr1[ii] * arr2[ii];
    }
    return result;
}

template<typename FPTYPE>
__global__ void virial_grad_wrt_neighbors_a(
    FPTYPE * grad_net,
    const FPTYPE * grad,
    const FPTYPE * env_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int nloc,
    const int nnei)
{
    // idy -> nnei
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + tid;
    const unsigned int idy = blockIdx.y;
    const unsigned int idw = threadIdx.y;
    const int ndescrpt = nnei * 4;
    __shared__ FPTYPE grad_one[9];
    if(tid < 9){
        grad_one[tid] = grad[tid];
    }
    __syncthreads(); 
    if (idx >= nloc) {
        return;
    }
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    FPTYPE tmp[9];
    for (int dd0 = 0; dd0 < 3; ++dd0){
        for (int dd1 = 0; dd1 < 3; ++dd1){
            tmp[dd0 * 3 + dd1] = rij[idx * nnei * 3 + idy * 3 + dd1] * env_deriv[idx * ndescrpt * 3 + idy * 4 * 3 + idw * 3 + dd0];
        }
    }
    grad_net[idx * ndescrpt + idy * 4 + idw] -= -1.0 * dev_dot9(grad_one, tmp);
}

template<typename FPTYPE>
__global__ void virial_grad_wrt_neighbors_r(
    FPTYPE * grad_net,
    const FPTYPE * grad,
    const FPTYPE * env_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int nloc,
    const int nnei)
{
    // idy -> nnei
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + tid;
    const unsigned int idy = blockIdx.y;
    const int ndescrpt = nnei;
    __shared__ FPTYPE grad_one[9];
    if(tid < 9){
        grad_one[tid] = grad[tid];
    }
    __syncthreads(); 
    if (idx >= nloc) {
        return;
    }
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    FPTYPE tmp[9];
    for (int dd0 = 0; dd0 < 3; ++dd0){
        for (int dd1 = 0; dd1 < 3; ++dd1){
            tmp[dd0 * 3 + dd1] = rij[idx * nnei * 3 + idy * 3 + dd1] * env_deriv[idx * ndescrpt * 3 + idy * 3 + dd0];
        }
    }
    grad_net[idx * ndescrpt + idy] -= -1.0 * dev_dot9(grad_one, tmp);
}

namespace deepmd {
template<typename FPTYPE>
void prod_virial_grad_a_gpu_cuda(
    FPTYPE * grad_net,
    const FPTYPE * grad,
    const FPTYPE * env_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int nloc,
    const int nnei)
{
    const int ndescrpt = nnei * 4;
    cudaErrcheck(cudaMemset(
        grad_net, 
        0.0, sizeof(FPTYPE) * nloc * ndescrpt));
    const int LEN = 128;
    const int nblock = (nloc + LEN -1) / LEN;
    dim3 block_grid(nblock, nnei);
    dim3 thread_grid(LEN, 4);
    virial_grad_wrt_neighbors_a<<<block_grid, thread_grid>>>(
        grad_net,
        grad, env_deriv, rij, nlist, nloc, nnei);
}

template<typename FPTYPE>
void prod_virial_grad_r_gpu_cuda(
    FPTYPE * grad_net,
    const FPTYPE * grad,
    const FPTYPE * env_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int nloc,
    const int nnei)
{
    const int ndescrpt = nnei;
    cudaErrcheck(cudaMemset(
        grad_net, 
        0.0, sizeof(FPTYPE) * nloc * ndescrpt));
    const int LEN = 128;
    const int nblock = (nloc + LEN -1) / LEN;
    dim3 block_grid(nblock, nnei);
    dim3 thread_grid(LEN, 1);
    virial_grad_wrt_neighbors_r<<<block_grid, thread_grid>>>(
        grad_net,
        grad, env_deriv, rij, nlist, nloc, nnei);
}

template void prod_virial_grad_a_gpu_cuda<float>(float * grad_net, const float * grad, const float * env_deriv, const float * rij, const int * nlist, const int nloc, const int nnei);
template void prod_virial_grad_a_gpu_cuda<double>(double * grad_net, const double * grad, const double * env_deriv, const double * rij, const int * nlist, const int nloc, const int nnei);
template void prod_virial_grad_r_gpu_cuda<float>(float * grad_net, const float * grad, const float * env_deriv, const float * rij, const int * nlist, const int nloc, const int nnei);
template void prod_virial_grad_r_gpu_cuda<double>(double * grad_net, const double * grad, const double * env_deriv, const double * rij, const int * nlist, const int nloc, const int nnei);
}