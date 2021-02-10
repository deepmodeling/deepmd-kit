#pragma once
#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include "MathUtilities.h"
#include "fmt_nlist.h"
#include "env_mat.h"
#if GOOGLE_CUDA
#include "DeviceFunctor.h"
#endif // GOOGLE_CUDA

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

struct NeighborInfo {
    int type;
    double dist;
    int index;
    NeighborInfo () : type (0), dist(0), index(0) {}
    NeighborInfo (int tt, double dd, int ii) : type (tt), dist(dd), index(ii) {}
    
    bool operator < (const NeighborInfo & b) const {
	    return (type < b.type || (type == b.type && (dist < b.dist || (dist == b.dist && index < b.index))));
    }
};





#if GOOGLE_CUDA
template<typename FPTYPE>
void DescrptSeAGPULauncher(const FPTYPE * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const FPTYPE * avg, const FPTYPE * std, FPTYPE * descrpt, FPTYPE * descrpt_deriv, FPTYPE * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int max_nbor_size) {
    DescrptSeAGPUExecuteFunctor<FPTYPE>()(coord, type, ilist, jrange, jlist, array_int, array_longlong, avg, std, descrpt, descrpt_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, rcut_r, rcut_r_smth, sec_a, fill_nei_a, max_nbor_size);
}
#endif // GOOGLE_CUDA
// ******************************************************************************
// end of custome op DescrptSeA
// ******************************************************************************

inline void make_descript_range (int & idx_start, int & idx_end, const int & nei_idx, const int& n_a_sel, const int n_a_shift) {
    if (nei_idx < n_a_sel) {
        idx_start = nei_idx * 4;
        idx_end   = nei_idx * 4 + 4;
    }
    else {
        idx_start = n_a_shift + (nei_idx - n_a_sel);
        idx_end   = n_a_shift + (nei_idx - n_a_sel) + 1;
    }
}

template<typename FPTYPE>
void ProdForceSeACPULauncher(FPTYPE * force, const FPTYPE * net_deriv, const FPTYPE * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift) {
    memset(force, 0.0, sizeof(FPTYPE) * nall * 3);
    // compute force of a frame
    for (int i_idx = 0; i_idx < nloc; ++i_idx) {
	    // deriv wrt center atom
	    for (int aa = 0; aa < ndescrpt; ++aa) {
	        force[i_idx * 3 + 0] -= net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
	        force[i_idx * 3 + 1] -= net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
	        force[i_idx * 3 + 2] -= net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
	    }
	    // deriv wrt neighbors
	    for (int jj = 0; jj < nnei; ++jj) {
	        int j_idx = nlist[i_idx * nnei + jj];
	        if (j_idx < 0) continue;
	        int aa_start, aa_end;
	        make_descript_range (aa_start, aa_end, jj, n_a_sel, n_a_shift);
	        for (int aa = aa_start; aa < aa_end; ++aa) {
	            force[j_idx * 3 + 0] += net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
	            force[j_idx * 3 + 1] += net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
	            force[j_idx * 3 + 2] += net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
	        }
	    }
    }
}

#if GOOGLE_CUDA
template<typename FPTYPE>
void ProdForceSeAGPULauncher(FPTYPE * force, const FPTYPE * net_deriv, const FPTYPE * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift) {
    ProdForceSeAGPUExecuteFunctor<FPTYPE>()(force, net_deriv, in_deriv, nlist, nloc, nall, nnei, ndescrpt, n_a_sel, n_a_shift);
}
#endif // GOOGLE_CUDA

// ******************************************************************************
// end of custome op ProdForceSeA
// ******************************************************************************

template<typename FPTYPE>
void ProdVirialSeACPULauncher(FPTYPE * virial, FPTYPE * atom_virial, const FPTYPE * net_deriv, const FPTYPE * in_deriv, const FPTYPE * rij, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift) {
    memset(virial, 0.0, sizeof(FPTYPE) * 9);
    memset(atom_virial, 0.0, sizeof(FPTYPE) * nall * 9);

    // compute virial of a frame
    for (int i_idx = 0; i_idx < nloc; ++i_idx) {
	    // deriv wrt neighbors
	    for (int jj = 0; jj < nnei; ++jj) {
	        int j_idx = nlist[i_idx * nnei + jj];
	        if (j_idx < 0) continue;
	        int aa_start, aa_end;
	        make_descript_range (aa_start, aa_end, jj, n_a_sel, n_a_shift);
	        for (int aa = aa_start; aa < aa_end; ++aa) {
	            FPTYPE pref = -1.0 * net_deriv[i_idx * ndescrpt + aa];
	            for (int dd0 = 0; dd0 < 3; ++dd0)
	                for (int dd1 = 0; dd1 < 3; ++dd1) {
		                FPTYPE tmp_v = pref * rij[i_idx * nnei * 3 + jj * 3 + dd1] *  in_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd0];
		                virial[dd0 * 3 + dd1] -= tmp_v;
		                atom_virial[j_idx * 9 + dd0 * 3 + dd1] -= tmp_v;
	                }
	        }
	    }
	}
}

#if GOOGLE_CUDA
template<typename FPTYPE>
void ProdVirialSeAGPULauncher(FPTYPE * virial, FPTYPE * atom_virial, const FPTYPE * net_deriv, const FPTYPE * in_deriv, const FPTYPE * rij, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift) {
    ProdVirialSeAGPUExecuteFunctor<FPTYPE>()(virial, atom_virial, net_deriv, in_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, n_a_sel, n_a_shift);
}
#endif // GOOGLE_CUDA
// ******************************************************************************
// end of custome op ProdVirialSeA
// ******************************************************************************

template<typename FPTYPE>
void GeluCPULauncher(const FPTYPE * in, FPTYPE * out, int const size) {
    for (int ii = 0; ii < size; ii++) {
        out[ii] = in[ii] * 0.5 * (1.0 + tanh(SQRT_2_PI * (in[ii] + 0.044715 * in[ii] * in[ii] *in[ii])));
    }
}

template<typename FPTYPE>
void GeluGradCPULauncher(const FPTYPE * dy, const FPTYPE * in, FPTYPE * out, int const size) {
    for (int ii = 0; ii < size; ii++) {
        FPTYPE const var1 = tanh(SQRT_2_PI * (in[ii] + 0.044715 * in[ii] * in[ii] *in[ii]));
        out[ii] = dy[ii] * (0.5 * SQRT_2_PI * in[ii] * (1 - var1 * var1) * (0.134145 * in[ii] * in[ii] + 1) + 0.5 * var1 + 0.5);
    }
}

template <typename FPTYPE>
void GeluGradGradCPULauncher(const FPTYPE * dy, const FPTYPE * dy_, const FPTYPE * in, FPTYPE * out, int const size) {
    for (int ii = 0; ii < size; ii++) {
        FPTYPE const var1 = tanh(SQRT_2_PI * (in[ii] + 0.044715 * in[ii] * in[ii] *in[ii]));
        FPTYPE const var2 = SQRT_2_PI * (1 - var1 * var1) * (0.134145 * in[ii] * in[ii] + 1);
	    out[ii] = dy[ii] * dy_[ii] * (0.134145 * SQRT_2_PI * in[ii] * in[ii] * (1 - var1 * var1) - SQRT_2_PI * in[ii] * var2 * (0.134145 * in[ii] * in[ii] + 1) * var1 + var2);
    }
}

#if GOOGLE_CUDA
template<typename FPTYPE>
void GeluGPULauncher(const FPTYPE * in, FPTYPE * out, int const size) {
    GeluGPUExecuteFunctor<FPTYPE>()(in, out, size);
}

template<typename FPTYPE>
void GeluGradGPULauncher(const FPTYPE * dy, const FPTYPE * in, FPTYPE * out, int const size) {
    GeluGradGPUExecuteFunctor<FPTYPE>()(dy, in, out, size);
}

template <typename FPTYPE>
void GeluGradGradGPULauncher(const FPTYPE * dy, const FPTYPE * dy_, const FPTYPE * in, FPTYPE * out, int const size) {
    GeluGradGradGPUExecuteFunctor<FPTYPE>()(dy, dy_, in, out, size);
}
#endif // GOOGLE_CUDA
// ******************************************************************************
// end of custome op Gelu
// ******************************************************************************


template<typename FPTYPE>
void DescrptSeRCPULauncher(const FPTYPE * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, const FPTYPE * avg, const FPTYPE * std, FPTYPE * descrpt, FPTYPE * descrpt_deriv, FPTYPE * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ntypes, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int max_nbor_size) {
    // set & normalize coord
    std::vector<FPTYPE> d_coord3(nall * 3);
    for (int ii = 0; ii < nall; ++ii) {
	    for (int dd = 0; dd < 3; ++dd) {
	        d_coord3[ii * 3 + dd] = coord[ii * 3 + dd];
	    }
    }

    // set type
    std::vector<int> d_type (nall);
    for (int ii = 0; ii < nall; ++ii) {
        d_type[ii] = type[ii];
    }
    
    // build nlist
    std::vector<std::vector<int > > d_nlist_a(nloc);

	for (unsigned ii = 0; ii < nloc; ++ii) {
	    d_nlist_a.reserve (jrange[nloc] / nloc + 10);
	}
	for (unsigned ii = 0; ii < nloc; ++ii) {
	    int i_idx = ilist[ii];
	    for (unsigned jj = jrange[ii]; jj < jrange[ii+1]; ++jj) {
	        int j_idx = jlist[jj];
	        d_nlist_a[i_idx].push_back (j_idx);
	    }
	}
    
    #pragma omp parallel for 
    for (int ii = 0; ii < nloc; ++ii) {
	    std::vector<int> fmt_nlist_a;
	    int ret = -1;
	    if (fill_nei_a) {
	        format_nlist_cpu(fmt_nlist_a, d_coord3, ntypes, d_type, ii, d_nlist_a[ii], rcut_r, sec_a);
	    }
	    std::vector<FPTYPE> d_descrpt_a;
	    std::vector<FPTYPE> d_descrpt_a_deriv;
	    std::vector<FPTYPE> d_descrpt_r;
	    std::vector<FPTYPE> d_descrpt_r_deriv;
	    std::vector<FPTYPE> d_rij_a;
	    env_mat_r_cpu (d_descrpt_a, d_descrpt_a_deriv, d_rij_a, d_coord3, ntypes, d_type, ii, fmt_nlist_a, sec_a, rcut_r_smth, rcut_r);

	    // check sizes
	    assert (d_descrpt_a.size() == ndescrpt);
	    assert (d_descrpt_a_deriv.size() == ndescrpt * 3);
	    assert (d_rij_a.size() == nnei * 3);
	    assert (fmt_nlist_a.size() == nnei);
	    // record outputs
	    for (int jj = 0; jj < ndescrpt; ++jj) {
            descrpt[ii * ndescrpt + jj] = (d_descrpt_a[jj] - avg[d_type[ii] * ndescrpt + jj]) / std[d_type[ii] * ndescrpt + jj];
        }
	    for (int jj = 0; jj < ndescrpt * 3; ++jj) {
	        descrpt_deriv[ii * ndescrpt * 3 + jj] = d_descrpt_a_deriv[jj] / std[d_type[ii] * ndescrpt + jj / 3];
	    }
	    for (int jj = 0; jj < nnei * 3; ++jj) {
	        rij[ii * nnei * 3 + jj] = d_rij_a[jj];
	    }
	    for (int jj = 0; jj < nnei; ++jj) {
	        nlist[ii * nnei + jj] = fmt_nlist_a[jj];
	    }
    }
}

#if GOOGLE_CUDA
template<typename FPTYPE>
void DescrptSeRGPULauncher(const FPTYPE * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const FPTYPE * avg, const FPTYPE * std, FPTYPE * descrpt, FPTYPE * descrpt_deriv, FPTYPE * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int max_nbor_size) {
    DescrptSeRGPUExecuteFunctor<FPTYPE>()(coord, type, ilist, jrange, jlist, array_int, array_longlong, avg, std, descrpt, descrpt_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, rcut_r, rcut_r_smth, sec_a, fill_nei_a, max_nbor_size);
}
#endif // GOOGLE_CUDA
// ******************************************************************************
// end of custome op DescrptSeR
// ******************************************************************************

template<typename FPTYPE>
void ProdForceSeRCPULauncher(FPTYPE * force, const FPTYPE * net_deriv, const FPTYPE * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt) {
    memset(force, 0.0, sizeof(FPTYPE) * nall * 3);
    // compute force of a frame
    for (int i_idx = 0; i_idx < nloc; ++i_idx) {
	    // deriv wrt center atom
	    for (int aa = 0; aa < ndescrpt; ++aa) {
	        force[i_idx * 3 + 0] -= net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
	        force[i_idx * 3 + 1] -= net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
	        force[i_idx * 3 + 2] -= net_deriv[i_idx * ndescrpt + aa] * in_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
	    }
	    // deriv wrt neighbors
	    for (int jj = 0; jj < nnei; ++jj) {
	        int j_idx = nlist[i_idx * nnei + jj];
	        if (j_idx < 0) continue;
	        force[j_idx * 3 + 0] += net_deriv[i_idx * ndescrpt + jj] * in_deriv[i_idx * ndescrpt * 3 + jj * 3 + 0];
	        force[j_idx * 3 + 1] += net_deriv[i_idx * ndescrpt + jj] * in_deriv[i_idx * ndescrpt * 3 + jj * 3 + 1];
	        force[j_idx * 3 + 2] += net_deriv[i_idx * ndescrpt + jj] * in_deriv[i_idx * ndescrpt * 3 + jj * 3 + 2];
	    }
    }
}

#if GOOGLE_CUDA
template<typename FPTYPE>
void ProdForceSeRGPULauncher(FPTYPE * force, const FPTYPE * net_deriv, const FPTYPE * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt) {
    ProdForceSeRGPUExecuteFunctor<FPTYPE>()(force, net_deriv, in_deriv, nlist, nloc, nall, nnei, ndescrpt);
}
#endif // GOOGLE_CUDA

// ******************************************************************************
// end of custome op ProdForceSeR
// ******************************************************************************

template<typename FPTYPE>
void ProdVirialSeRCPULauncher(FPTYPE * virial, FPTYPE * atom_virial, const FPTYPE * net_deriv, const FPTYPE * in_deriv, const FPTYPE * rij, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt) {
    memset(virial, 0.0, sizeof(FPTYPE) * 9);
    memset(atom_virial, 0.0, sizeof(FPTYPE) * nall * 9);

    // compute virial of a frame
    for (int i_idx = 0; i_idx < nloc; ++i_idx) {
	    // deriv wrt neighbors
	    for (int jj = 0; jj < nnei; ++jj) {
	        int j_idx = nlist[i_idx * nnei + jj];
	        if (j_idx < 0) continue;
	        FPTYPE pref = -1.0 * net_deriv[i_idx * ndescrpt + jj];
	        for (int dd0 = 0; dd0 < 3; ++dd0)
	            for (int dd1 = 0; dd1 < 3; ++dd1) {
		            FPTYPE tmp_v = pref * rij[i_idx * nnei * 3 + jj * 3 + dd1] *  in_deriv[i_idx * ndescrpt * 3 + jj * 3 + dd0];
		            virial[dd0 * 3 + dd1] -= tmp_v;
		            atom_virial[j_idx * 9 + dd0 * 3 + dd1] -= tmp_v;
	            }
	    }
	}
}

#if GOOGLE_CUDA
template<typename FPTYPE>
void ProdVirialSeRGPULauncher(FPTYPE * virial, FPTYPE * atom_virial, const FPTYPE * net_deriv, const FPTYPE * in_deriv, const FPTYPE * rij, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt) {
    ProdVirialSeRGPUExecuteFunctor<FPTYPE>()(virial, atom_virial, net_deriv, in_deriv, rij, nlist, nloc, nall, nnei, ndescrpt);
}
#endif // GOOGLE_CUDA
// ******************************************************************************
// end of custome op ProdVirialSeR
// ******************************************************************************
