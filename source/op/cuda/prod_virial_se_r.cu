#include "DeviceFunctor.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template<typename FPTYPE>
__global__ void deriv_wrt_neighbors_se_r(FPTYPE * virial, 
                        FPTYPE * atom_virial,
                        const FPTYPE * net_deriv,
                        const FPTYPE * in_deriv,
                        const FPTYPE * rij,
                        const int * nlist,
                        const int nloc,
                        const int nnei,
                        const int ndescrpt) 
{
    // idx -> nloc
    // idy -> nnei
    // idz = dd0 * 3 + dd1
    // dd0 = idz / 3
    // dd1 = idz % 3
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;

    if (idx >= nloc) {
        return;
    }
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    // atomicAdd(virial + idz, net_deriv[idx * ndescrpt + idy * 4 + idw] * rij[idx * nnei * 3 + idy * 3 + idz / 3] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz % 3]);
    atomicAdd(atom_virial + j_idx * 9 + idz, net_deriv[idx * ndescrpt + idy] * rij[idx * nnei * 3 + idy * 3 + idz % 3] * in_deriv[idx * ndescrpt * 3 + idy * 3 + idz / 3]);
}

template <typename FPTYPE>
void ProdVirialSeRGPUExecuteFunctor<FPTYPE>::operator()(FPTYPE * virial, 
                        FPTYPE * atom_virial,
                        const FPTYPE * net_deriv,
                        const FPTYPE * in_deriv,
                        const FPTYPE * rij,
                        const int * nlist,
                        const int nloc,
                        const int nall,
                        const int nnei,
                        const int ndescrpt)
{
    cudaErrcheck(cudaMemset(virial, 0.0, sizeof(FPTYPE) * 9));
    cudaErrcheck(cudaMemset(atom_virial, 0.0, sizeof(FPTYPE) * 9 * nall));

    const int LEN = 64;
    int nblock = (nloc + LEN -1) / LEN;
    dim3 block_grid(nblock, nnei);
    dim3 thread_grid(LEN, 9);
    // compute virial of a frame
    deriv_wrt_neighbors_se_r<<<block_grid, thread_grid>>>(
                        virial, 
                        atom_virial, 
                        net_deriv, 
                        in_deriv,
                        rij,
                        nlist,
                        nloc,
                        nnei,
                        ndescrpt
    );
}

template struct ProdVirialSeRGPUExecuteFunctor<float>;
template struct ProdVirialSeRGPUExecuteFunctor<double>;