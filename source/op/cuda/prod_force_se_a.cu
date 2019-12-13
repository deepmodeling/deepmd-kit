#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#ifdef HIGH_PREC
    typedef double VALUETYPE;
#else
    typedef float  VALUETYPE;
#endif

#define cudaErrcheck(res) { cudaAssert((res), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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

__global__ void deriv_wrt_center_atom_se_a(VALUETYPE * force, 
                        const VALUETYPE * net_deriv,
                        const VALUETYPE * in_deriv,
                        const int ndescrpt)
{
    const unsigned int idx = blockIdx.y;
    const unsigned int idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idz = threadIdx.y;

    if (idy >= ndescrpt) {
        return;
    }
    
    atomicAdd(force + idx * 3 + idz, -1.0 * net_deriv[idx * ndescrpt + idy] * in_deriv[idx * ndescrpt * 3 + idy * 3 + idz]);
}

__global__ void deriv_wrt_neighbors_se_a(VALUETYPE * force, 
                        const VALUETYPE * net_deriv,
                        const VALUETYPE * in_deriv,
                        const int * nlist,
                        const int nloc,
                        const int nnei,
                        const int ndescrpt,
                        const int n_a_sel,
                        const int n_a_shift)
{  
    // idy -> nnei
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;
    const unsigned int idw = threadIdx.z;
    
    if (idx >= nloc) {
        return;
    }
    // deriv wrt neighbors
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    atomicAdd(force + j_idx * 3 + idz, net_deriv[idx * ndescrpt + idy * 4 + idw] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz]);
}

void ProdForceSeALauncher(VALUETYPE * force, 
                        const VALUETYPE * net_deriv,
                        const VALUETYPE * in_deriv,
                        const int * nlist,
                        const int nloc,
                        const int nall,
                        const int ndescrpt,
                        const int nnei,
                        const int n_a_sel,
                        const int n_a_shift)
{   
    // std::cout << "I'm here!" << std::endl;
    cudaErrcheck(cudaMemset(force, 0.0, sizeof(VALUETYPE) * nall * 3));
    const int LEN1 = 256;
    const int nblock1 = (ndescrpt + LEN1 -1) / LEN1;
    dim3 grid(nblock1, nloc);
    dim3 thread(LEN1, 3);
    deriv_wrt_center_atom_se_a<<<grid, thread>>>(force, net_deriv, in_deriv, ndescrpt);
    
    const int LEN = 64;
    int nblock = (nloc + LEN -1) / LEN;
    dim3 block_grid(nblock, nnei);
    dim3 thread_grid(LEN, 3, 4);
    deriv_wrt_neighbors_se_a<<<block_grid, thread_grid>>>(force, net_deriv, in_deriv, nlist, nloc, nnei, ndescrpt, n_a_sel, n_a_shift);
}