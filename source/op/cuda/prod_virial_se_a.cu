#include <stdio.h>
#include <cuda_runtime.h>

#define MUL 512

#ifdef HIGH_PREC
    typedef double VALUETYPE;
#else
    typedef float  VALUETYPE;
#endif

#define cudaErrcheck(res) { cudaAssert((res), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
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

__global__ void deriv_wrt_neighbors_se_a(VALUETYPE * virial, 
                        VALUETYPE * atom_virial,
                        const VALUETYPE * net_deriv,
                        const VALUETYPE * in_deriv,
                        const VALUETYPE * rij,
                        const int * nlist,
                        const int nloc,
                        const int nnei,
                        const int ndescrpt,
                        const int n_a_sel,
                        const int n_a_shift) 
{
    // idx -> nloc
    // idy -> nnei
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;
    const unsigned int idw = threadIdx.z;

    if (idx >= nloc) {
        return;
    }
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    // atomicAdd(virial + idz, net_deriv[idx * ndescrpt + idy * 4 + idw] * rij[idx * nnei * 3 + idy * 3 + idz / 3] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz % 3]);
    atomicAdd(atom_virial + j_idx * 9 + idz, net_deriv[idx * ndescrpt + idy * 4 + idw] * rij[idx * nnei * 3 + idy * 3 + idz / 3] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz % 3]);
}

void ProdVirialSeALauncher(VALUETYPE * virial, 
                        VALUETYPE * atom_virial,
                        const VALUETYPE * net_deriv,
                        const VALUETYPE * in_deriv,
                        const VALUETYPE * rij,
                        const int * nlist,
                        const int nloc,
                        const int nall,
                        const int nnei,
                        const int ndescrpt,
                        const int n_a_sel,
                        const int n_a_shift) 
{
    cudaErrcheck(cudaMemset(virial, 0.0, sizeof(VALUETYPE) * 9));
    cudaErrcheck(cudaMemset(atom_virial, 0.0, sizeof(VALUETYPE) * 9 * nall));

    const int LEN = 16;
    int nblock = (nloc + LEN -1) / LEN;
    dim3 block_grid(nblock, nnei);
    dim3 thread_grid(LEN, 9, 4);
    // compute virial of a frame
    deriv_wrt_neighbors_se_a<<<block_grid, thread_grid>>>(
                        virial, 
                        atom_virial, 
                        net_deriv, 
                        in_deriv,
                        rij,
                        nlist,
                        nloc,
                        nnei,
                        ndescrpt,
                        n_a_sel,
                        n_a_shift
    );
}
