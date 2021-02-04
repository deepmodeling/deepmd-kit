#pragma once
#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include "MathUtilities.h"
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

template <typename FPTYPE>
inline void spline5_switch (
    FPTYPE & vv,
	FPTYPE & dd,
	const FPTYPE & xx, 
	const float & rmin, 
	const float & rmax)
{
    if (xx < rmin) {
        dd = 0;
        vv = 1;
    }
    else if (xx < rmax) {
        FPTYPE uu = (xx - rmin) / (rmax - rmin) ;
        FPTYPE du = 1. / (rmax - rmin) ;
        vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1;
        dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
    }
    else {
        dd = 0;
        vv = 0;
    }
}

template<typename FPTYPE> 
int format_nlist_fill_se_a_cpu (
    std::vector<int > &		    fmt_nei_idx_a,
	const std::vector<FPTYPE > &	    posi,
	const int &			    ntypes,
	const std::vector<int > &    type,
	const int &			    i_idx,
	const std::vector<int > &    nei_idx_a, 
	const float &		    rcut,
	const std::vector<int > &    sec_a)
{
    fmt_nei_idx_a.resize (sec_a.back());
    fill (fmt_nei_idx_a.begin(), fmt_nei_idx_a.end(), -1);
  
    // gether all neighbors
    std::vector<int > nei_idx (nei_idx_a);
    // allocate the information for all neighbors
    std::vector<NeighborInfo > sel_nei;
    sel_nei.reserve (nei_idx_a.size());
    for (unsigned kk = 0; kk < nei_idx.size(); ++kk) {
        FPTYPE diff[3];
        const int & j_idx = nei_idx[kk];
        for (int dd = 0; dd < 3; ++dd) {
            diff[dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
        }
        FPTYPE rr = sqrt(MathUtilities::dot<FPTYPE> (diff, diff));    
        if (rr <= rcut) {
            sel_nei.push_back(NeighborInfo(type[j_idx], rr, j_idx));
        }
    }
    sort(sel_nei.begin(), sel_nei.end());  
  
    std::vector<int > nei_iter = sec_a;
    int overflowed = -1;
    for (unsigned kk = 0; kk < sel_nei.size(); ++kk) {
        const int & nei_type = sel_nei[kk].type;
        if (nei_iter[nei_type] < sec_a[nei_type+1]) {
            fmt_nei_idx_a[nei_iter[nei_type] ++] = sel_nei[kk].index;
        }
    }
    return overflowed;
}

template<typename FPTYPE> 
void compute_descriptor_se_a_cpu (
    std::vector<FPTYPE > &	        descrpt_a,
	std::vector<FPTYPE > &	        descrpt_a_deriv,
	std::vector<FPTYPE > &	        rij_a,
	const std::vector<FPTYPE > &	    posi,
	const int &				ntypes,
	const std::vector<int > &	type,
	const int &				i_idx,
	const std::vector<int > &	fmt_nlist_a,
	const std::vector<int > &	sec_a, 
	const float &			rmin,
	const float &			rmax) 
{
    // compute the diff of the neighbors
    rij_a.resize (sec_a.back() * 3);
    fill (rij_a.begin(), rij_a.end(), 0.0);
    for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii) {
        for (int jj = sec_a[ii]; jj < sec_a[ii + 1]; ++jj) {
            if (fmt_nlist_a[jj] < 0) break;
            const int & j_idx = fmt_nlist_a[jj];

            for (int dd = 0; dd < 3; ++dd) {
                rij_a[jj * 3 + dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
            }
        }
    }
    // 1./rr, cos(theta), cos(phi), sin(phi)
    descrpt_a.resize (sec_a.back() * 4);
    fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
    // deriv wrt center: 3
    descrpt_a_deriv.resize (sec_a.back() * 4 * 3);
    fill (descrpt_a_deriv.begin(), descrpt_a_deriv.end(), 0.0);

    for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter+1]; ++nei_iter) {      
            if (fmt_nlist_a[nei_iter] < 0) break;
            const FPTYPE * rr = &rij_a[nei_iter * 3];
            FPTYPE nr2 = MathUtilities::dot(rr, rr);
            FPTYPE inr = 1./sqrt(nr2);
            FPTYPE nr = nr2 * inr;
            FPTYPE inr2 = inr * inr;
            FPTYPE inr4 = inr2 * inr2;
            FPTYPE inr3 = inr4 * nr;
            FPTYPE sw, dsw;
            spline5_switch(sw, dsw, nr, rmin, rmax);
            int idx_deriv = nei_iter * 4 * 3;	// 4 components time 3 directions
            int idx_value = nei_iter * 4;	// 4 components
            // 4 value components
            descrpt_a[idx_value + 0] = 1./nr;
            descrpt_a[idx_value + 1] = rr[0] / nr2;
            descrpt_a[idx_value + 2] = rr[1] / nr2;
            descrpt_a[idx_value + 3] = rr[2] / nr2;
            // deriv of component 1/r
            descrpt_a_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[2] * inr;
            // deriv of component x/r2
            descrpt_a_deriv[idx_deriv + 3] = (2. * rr[0] * rr[0] * inr4 - inr2) * sw - descrpt_a[idx_value + 1] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv + 4] = (2. * rr[0] * rr[1] * inr4	) * sw - descrpt_a[idx_value + 1] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv + 5] = (2. * rr[0] * rr[2] * inr4	) * sw - descrpt_a[idx_value + 1] * dsw * rr[2] * inr;
            // deriv of component y/r2
            descrpt_a_deriv[idx_deriv + 6] = (2. * rr[1] * rr[0] * inr4	) * sw - descrpt_a[idx_value + 2] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv + 7] = (2. * rr[1] * rr[1] * inr4 - inr2) * sw - descrpt_a[idx_value + 2] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv + 8] = (2. * rr[1] * rr[2] * inr4	) * sw - descrpt_a[idx_value + 2] * dsw * rr[2] * inr;
            // deriv of component z/r2
            descrpt_a_deriv[idx_deriv + 9] = (2. * rr[2] * rr[0] * inr4	) * sw - descrpt_a[idx_value + 3] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv +10] = (2. * rr[2] * rr[1] * inr4	) * sw - descrpt_a[idx_value + 3] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv +11] = (2. * rr[2] * rr[2] * inr4 - inr2) * sw - descrpt_a[idx_value + 3] * dsw * rr[2] * inr;
            // 4 value components
            descrpt_a[idx_value + 0] *= sw;
            descrpt_a[idx_value + 1] *= sw;
            descrpt_a[idx_value + 2] *= sw;
            descrpt_a[idx_value + 3] *= sw;
        }
    }
}

template<typename FPTYPE>
void DescrptSeACPULauncher(const FPTYPE * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, const FPTYPE * avg, const FPTYPE * std, FPTYPE * descrpt, FPTYPE * descrpt_deriv, FPTYPE * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ntypes, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int magic_number) {
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
	        format_nlist_fill_se_a_cpu(fmt_nlist_a, d_coord3, ntypes, d_type, ii, d_nlist_a[ii], rcut_r, sec_a);
	    }
	    std::vector<FPTYPE> d_descrpt_a;
	    std::vector<FPTYPE> d_descrpt_a_deriv;
	    std::vector<FPTYPE> d_descrpt_r;
	    std::vector<FPTYPE> d_descrpt_r_deriv;
	    std::vector<FPTYPE> d_rij_a;
	    compute_descriptor_se_a_cpu (d_descrpt_a, d_descrpt_a_deriv, d_rij_a, d_coord3, ntypes, d_type, ii, fmt_nlist_a, sec_a, rcut_r_smth, rcut_r);

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
void DescrptSeAGPULauncher(const FPTYPE * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const FPTYPE * avg, const FPTYPE * std, FPTYPE * descrpt, FPTYPE * descrpt_deriv, FPTYPE * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int magic_number) {
    DescrptSeAGPUExecuteFunctor<FPTYPE>()(coord, type, ilist, jrange, jlist, array_int, array_longlong, avg, std, descrpt, descrpt_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, rcut_r, rcut_r_smth, sec_a, fill_nei_a, magic_number);
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
void compute_descriptor_se_r_cpu (
    std::vector<FPTYPE > &	        descrpt_a,
	std::vector<FPTYPE > &	        descrpt_a_deriv,
	std::vector<FPTYPE > &	        rij_a,
	const std::vector<FPTYPE > &	    posi,
	const int &				ntypes,
	const std::vector<int > &	type,
	const int &				i_idx,
	const std::vector<int > &	fmt_nlist_a,
	const std::vector<int > &	sec_a, 
	const float &			rmin,
	const float &			rmax) 
{
    // compute the diff of the neighbors
    rij_a.resize (sec_a.back() * 3);
    fill (rij_a.begin(), rij_a.end(), 0.0);
    for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii) {
        for (int jj = sec_a[ii]; jj < sec_a[ii + 1]; ++jj) {
            if (fmt_nlist_a[jj] < 0) break;
            const int & j_idx = fmt_nlist_a[jj];

            for (int dd = 0; dd < 3; ++dd) {
                rij_a[jj * 3 + dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
            }
        }
    }
    // 1./rr, cos(theta), cos(phi), sin(phi)
    descrpt_a.resize (sec_a.back());
    fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
    // deriv wrt center: 3
    descrpt_a_deriv.resize (sec_a.back() * 3);
    fill (descrpt_a_deriv.begin(), descrpt_a_deriv.end(), 0.0);

    for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter+1]; ++nei_iter) {      
            if (fmt_nlist_a[nei_iter] < 0) break;
            const FPTYPE * rr = &rij_a[nei_iter * 3];
            FPTYPE nr2 = MathUtilities::dot(rr, rr);
            FPTYPE inr = 1./sqrt(nr2);
            FPTYPE nr = nr2 * inr;
            FPTYPE inr2 = inr * inr;
            FPTYPE inr4 = inr2 * inr2;
            FPTYPE inr3 = inr4 * nr;
            FPTYPE sw, dsw;
            spline5_switch(sw, dsw, nr, rmin, rmax);
            int idx_deriv = nei_iter * 3;	// 1 components time 3 directions
            int idx_value = nei_iter;	    // 1 components
            // 4 value components
            descrpt_a[idx_value + 0] = 1./nr;
            // deriv of component 1/r
            descrpt_a_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[2] * inr;
            // 4 value components
            descrpt_a[idx_value + 0] *= sw;
        }
    }
}

template<typename FPTYPE>
void DescrptSeRCPULauncher(const FPTYPE * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, const FPTYPE * avg, const FPTYPE * std, FPTYPE * descrpt, FPTYPE * descrpt_deriv, FPTYPE * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ntypes, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int magic_number) {
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
	        format_nlist_fill_se_a_cpu(fmt_nlist_a, d_coord3, ntypes, d_type, ii, d_nlist_a[ii], rcut_r, sec_a);
	    }
	    std::vector<FPTYPE> d_descrpt_a;
	    std::vector<FPTYPE> d_descrpt_a_deriv;
	    std::vector<FPTYPE> d_descrpt_r;
	    std::vector<FPTYPE> d_descrpt_r_deriv;
	    std::vector<FPTYPE> d_rij_a;
	    compute_descriptor_se_r_cpu (d_descrpt_a, d_descrpt_a_deriv, d_rij_a, d_coord3, ntypes, d_type, ii, fmt_nlist_a, sec_a, rcut_r_smth, rcut_r);

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
void DescrptSeRGPULauncher(const FPTYPE * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const FPTYPE * avg, const FPTYPE * std, FPTYPE * descrpt, FPTYPE * descrpt_deriv, FPTYPE * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int magic_number) {
    DescrptSeRGPUExecuteFunctor<FPTYPE>()(coord, type, ilist, jrange, jlist, array_int, array_longlong, avg, std, descrpt, descrpt_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, rcut_r, rcut_r_smth, sec_a, fill_nei_a, magic_number);
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

template <typename FPTYPE>
inline FPTYPE dot(FPTYPE a[4], FPTYPE b[4]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; 
}

/*
    This inline function was designed to get the table info and bias value for current input xx!
    lower:      indicate the lower boundary of the first table;
    upper:      indicate the upper boundary of the first table as well as the lower boundary of the second table;
    max:        indicate the upper boundary of the second table;
    stride0:    indicate the stride of the first table;
    stride1:    indicate the stride of the second table;
    xx:         indicate the inputs value;
    table_idx:  indicate the location of table info of input value xx;
*/
template <typename FPTYPE>
inline void locate_xx(const FPTYPE& lower, const FPTYPE& upper,  const FPTYPE& max, const FPTYPE& stride0, const FPTYPE& stride1, FPTYPE& xx, int& table_idx) {
    if (xx < lower) {
        table_idx = 0;
        xx = 0;
    }
    else if (xx < upper) {
        table_idx = (int)((xx - lower) / stride0);
        xx -= (table_idx * stride0 + lower);
    }
    else if (xx < max) {
        int first_stride = int((upper - lower) / stride0);
        table_idx = first_stride + (int)((xx - upper) / stride1);
        xx -= ((table_idx - first_stride) * stride1 + upper);
    }
    else {
        table_idx = int((upper - lower) / stride0) + (int)((max - upper) / stride1) - 1;
        xx = 0;
    }
}

template <typename FPTYPE>
void TabulateFusionCPULauncher(const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const int nloc, const int nnei, const int last_layer_size, FPTYPE * out) {
    //Currently, Do nothing at all! 
    // std::cout << "I'm in tabulate @CPU!" << std::endl;
    memset(out, 0.0, sizeof(FPTYPE) * nloc * 4 * last_layer_size);
    FPTYPE const lower   = table_info[0];
    FPTYPE const upper   = table_info[1];
    FPTYPE const _max    = table_info[2];
    FPTYPE const stride0 = table_info[3];
    FPTYPE const stride1 = table_info[4];
    // for every atom, execute a small gemm~
    // FPTYPE * res = new FPTYPE[4 * last_layer_size];
    #pragma omp parallel for
    for (int ii = 0; ii < nloc; ii++) {
        FPTYPE ll[4] = {0};
        FPTYPE ago = in[ii * nnei + nnei - 1];
        bool unloop = false; 
        for (int jj = 0; jj < nnei; jj++) { 
            ll[0] = ff[ii * nnei * 4 + jj * 4 + 0];
            ll[1] = ff[ii * nnei * 4 + jj * 4 + 1];
            ll[2] = ff[ii * nnei * 4 + jj * 4 + 2];
            ll[3] = ff[ii * nnei * 4 + jj * 4 + 3];
            FPTYPE xx = in[ii * nnei + jj]; 
            if (ago == xx) {
                unloop = true;
            }
            int table_idx = 0;
            locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
            for (int kk = 0; kk < last_layer_size; kk++) {
                // 1.094 timesteps/s                                       
                FPTYPE a0  = table[table_idx * last_layer_size * 6 + 6 * kk + 0]; 
                FPTYPE a1  = table[table_idx * last_layer_size * 6 + 6 * kk + 1]; 
                FPTYPE a2  = table[table_idx * last_layer_size * 6 + 6 * kk + 2]; 
                FPTYPE a3  = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
                FPTYPE a4  = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
                FPTYPE a5  = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
                FPTYPE var = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
                if (unloop) {
                    out[ii * last_layer_size * 4 + 0 * last_layer_size + kk] += (nnei - jj) * var * ll[0];
                    out[ii * last_layer_size * 4 + 1 * last_layer_size + kk] += (nnei - jj) * var * ll[1];
                    out[ii * last_layer_size * 4 + 2 * last_layer_size + kk] += (nnei - jj) * var * ll[2];
                    out[ii * last_layer_size * 4 + 3 * last_layer_size + kk] += (nnei - jj) * var * ll[3];
                }
                else {
                    out[ii * last_layer_size * 4 + 0 * last_layer_size + kk] += var * ll[0];
                    out[ii * last_layer_size * 4 + 1 * last_layer_size + kk] += var * ll[1];
                    out[ii * last_layer_size * 4 + 2 * last_layer_size + kk] += var * ll[2];
                    out[ii * last_layer_size * 4 + 3 * last_layer_size + kk] += var * ll[3];
                }
            }
            if (unloop) break;
        }
    }
}

template <typename FPTYPE>
void TabulateFusionGradCPULauncher(const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, const int nloc, const int nnei, const int last_layer_size, FPTYPE * dy_dx, FPTYPE * dy_df) {
    // std::cout << "I'm in tabulate gradient @CPU!" << std::endl;
    memset(dy_dx, 0.0, sizeof(FPTYPE) * nloc * nnei);
    memset(dy_df, 0.0, sizeof(FPTYPE) * nloc * nnei * 4);
    FPTYPE const lower   = table_info[0];
    FPTYPE const upper   = table_info[1];
    FPTYPE const _max    = table_info[2];
    FPTYPE const stride0 = table_info[3];
    FPTYPE const stride1 = table_info[4];
    // for every atom, execute a small gemm~
    // FPTYPE * res = new FPTYPE[4 * last_layer_size];
    #pragma omp parallel for
    for (int ii = 0; ii < nloc; ii++) {
        FPTYPE ll[4];
        FPTYPE rr[4];
        FPTYPE ago = in[ii * nnei + nnei - 1];
        bool unloop = false;
        for (int jj = 0; jj < nnei; jj++) {
            // construct the dy/dx
            ll[0] = ff[ii * nnei * 4 + jj * 4 + 0];
            ll[1] = ff[ii * nnei * 4 + jj * 4 + 1];
            ll[2] = ff[ii * nnei * 4 + jj * 4 + 2];
            ll[3] = ff[ii * nnei * 4 + jj * 4 + 3];
            FPTYPE xx = in[ii * nnei + jj]; 
            if (ago == xx) {
                unloop = true;
            }
            int table_idx = 0;
            locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
            FPTYPE grad = 0.0;
            for (int kk = 0; kk < last_layer_size; kk++) {
                rr[0] = dy[ii * last_layer_size * 4 + 0 * last_layer_size + kk];
                rr[1] = dy[ii * last_layer_size * 4 + 1 * last_layer_size + kk];
                rr[2] = dy[ii * last_layer_size * 4 + 2 * last_layer_size + kk];
                rr[3] = dy[ii * last_layer_size * 4 + 3 * last_layer_size + kk];
                // 1.094 timesteps/s
                FPTYPE a0  = table[table_idx * last_layer_size * 6 + 6 * kk + 0]; 
                FPTYPE a1  = table[table_idx * last_layer_size * 6 + 6 * kk + 1]; 
                FPTYPE a2  = table[table_idx * last_layer_size * 6 + 6 * kk + 2]; 
                FPTYPE a3  = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
                FPTYPE a4  = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
                FPTYPE a5  = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
                FPTYPE res = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
                
                if (unloop) {
                    grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr) * (nnei - jj);
                    dy_df[ii * nnei * 4 + jj * 4 + 0] += res * rr[0] * (nnei - jj);
                    dy_df[ii * nnei * 4 + jj * 4 + 1] += res * rr[1] * (nnei - jj);
                    dy_df[ii * nnei * 4 + jj * 4 + 2] += res * rr[2] * (nnei - jj);
                    dy_df[ii * nnei * 4 + jj * 4 + 3] += res * rr[3] * (nnei - jj);
                }
                else {
                    grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr);
                    dy_df[ii * nnei * 4 + jj * 4 + 0] += res * rr[0];
                    dy_df[ii * nnei * 4 + jj * 4 + 1] += res * rr[1];
                    dy_df[ii * nnei * 4 + jj * 4 + 2] += res * rr[2];
                    dy_df[ii * nnei * 4 + jj * 4 + 3] += res * rr[3];
                }
            }
            dy_dx[ii * nnei + jj] = grad;
            if (unloop) break;
        }
    }
}

template <typename FPTYPE>
void TabulateCheckerCPULauncher(const FPTYPE * table_info, const FPTYPE * in, int * out, const int nloc, const int nnei) {
    FPTYPE const lower   = table_info[0];
    FPTYPE const upper   = table_info[1];
    FPTYPE const _max    = table_info[2];
    FPTYPE const stride0 = table_info[3];
    FPTYPE const stride1 = table_info[4];
    // for every atom, execute a small gemm~
    // FPTYPE * res = new FPTYPE[4 * last_layer_size];
    int Csub = 0;    // summation of second table approximate;
    int Dsub = 0;    // summation of the endpoint approximate;
    for (int ii = 0; ii < nloc; ii++) {
        for (int jj = 0; jj < nnei; jj++) {
            FPTYPE xx = in[ii * nnei + jj];
            if (xx < lower || xx > _max) {
                Csub += 1;
            }
            else if (xx >= upper && xx <= _max) {
                Dsub += 1;
            }
        }
    }
    if(Csub > 0) {
        std::cout << "# DEEPMD: warning! some values [" << Csub << "/" << nloc * nnei << "] overflow the range of the table, using the endpoint approximate processing.." << std::endl;
    }
    if(Dsub > 0) {
        std::cout << "# DEEPMD: warning! some values [" << Dsub << "/" << nloc * nnei << "] overflow the range of the table, using second table approximate processing.." << std::endl;
    }
}

#if GOOGLE_CUDA
template<typename FPTYPE>
void TabulateFusionGPULauncher(const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const int nloc, const int nnei, const int last_layer_size, FPTYPE * out) {
    TabulateFusionGPUExecuteFunctor<FPTYPE>()(table, table_info, in, ff, nloc, nnei, last_layer_size, out);
}

template<typename FPTYPE>
void TabulateFusionGradGPULauncher(const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, const int nloc, const int nnei, const int last_layer_size, FPTYPE * dy_dx, FPTYPE * dy_df) {
    TabulateFusionGradGPUExecuteFunctor<FPTYPE>()(table, table_info, in, ff, dy, nloc, nnei, last_layer_size, dy_dx, dy_df);
}

template <typename FPTYPE>
void TabulateCheckerGPULauncher(const FPTYPE * table_info, const FPTYPE * in, int * out, const int nloc, const int nnei) {
    TabulateCheckerGPUExecuteFunctor<FPTYPE>()(table_info, in, out, nloc, nnei);
}
#endif // GOOGLE_CUDA
// ******************************************************************************
// end of custome op Tabulate
// ******************************************************************************
