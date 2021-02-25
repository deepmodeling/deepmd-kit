#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include "DeviceFunctor.h"
#include "gpu_nv.cuh"
#include "gpu_nv.h"

template<
    typename FPTYPE,
    int      THREADS_PER_BLOCK>
__global__ void compute_descriptor_se_r(FPTYPE* descript,
                            const int ndescrpt,
                            FPTYPE* descript_deriv,
                            const int descript_deriv_size,
                            FPTYPE* rij,
                            const int rij_size,
                            const int* type,
                            const FPTYPE* avg,
                            const FPTYPE* std,
                            int* nlist,
                            const int nlist_size,
                            const FPTYPE* coord,
                            const float rmin,
                            const float rmax,
                            const int sec_a_size)
{   
    // <<<nloc, TPB>>>
    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    // usually false...
    if (tid >= sec_a_size) {
        return;
    }
    // const int idx_deriv = idy * 4 * 3;	// 4 components time 3 directions
    // const int idx_value = idy * 4;	// 4 components
    int * row_nlist = nlist + bid * nlist_size;
    FPTYPE * row_rij = rij + bid * rij_size;
    FPTYPE * row_descript = descript + bid * ndescrpt;
    FPTYPE * row_descript_deriv = descript_deriv + bid * descript_deriv_size;

    for (int ii = tid; ii < sec_a_size; ii += THREADS_PER_BLOCK) {
        const int idx_value = ii;	// 4 components
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
            dd *= sw; // * descript[idx * ndescrpt + idx_value + 0]);// - avg[type[idx] * ndescrpt + idx_value + 0]) / std[type[idx] * ndescrpt + idx_value + 0];
            for (int ii = 0; ii < 3; ii++) {
                row_descript_deriv[idx_deriv + ii] = vv[ii] / std[type[bid] * ndescrpt + idx_value + ii / 3];
            }
            row_descript[idx_value] = (dd - avg[type[bid] * ndescrpt + idx_value]) / std[type[bid] * ndescrpt + idx_value];
        }
        else {
            // TODO: move it to the memset.
            row_descript[idx_value] -= avg[type[bid] * ndescrpt + idx_value] / std[type[bid] * ndescrpt + idx_value];
        }
    }
}

template <typename FPTYPE>
void DescrptSeRGPUExecuteFunctor<FPTYPE>::operator()(const FPTYPE * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const FPTYPE * avg, const FPTYPE * std, FPTYPE * em, FPTYPE * em_deriv, FPTYPE * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int max_nbor_size) {
    prod_env_mat_common(
        em, em_deriv, rij, nlist, 
        coord, type, ilist, jrange, jlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, nall, rcut_r, rcut_r_smth, sec_a);

    compute_descriptor_se_r<FPTYPE, TPB> <<<nloc, TPB>>> (em, ndescrpt, em_deriv, ndescrpt * 3, rij, nnei * 3, type, avg, std, nlist, nnei, coord, rcut_r_smth, rcut_r, sec_a.back());
}

template struct DescrptSeRGPUExecuteFunctor<float>;
template struct DescrptSeRGPUExecuteFunctor<double>;