/*
//==================================================
 _   _  __     __  _   _   __  __   ____  
| \ | | \ \   / / | \ | | |  \/  | |  _ \ 
|  \| |  \ \ / /  |  \| | | |\/| | | | | |
| |\  |   \ V /   | |\  | | |  | | | |_| |
|_| \_|    \_/    |_| \_| |_|  |_| |____/ 

//==================================================

code: nvnmd
reference: deepmd
author: mph (pinghui_mo@outlook.com)
date: 2021-12-6

*/


#include "env_mat_nvnmd.h"
#include "switcher.h"


/*
//==================================================
  env_mat_a_nvnmd_cpu
//==================================================
*/


template<typename FPTYPE> 
void 
deepmd::
env_mat_a_nvnmd_cpu (
    std::vector<FPTYPE > &	        descrpt_a,
    std::vector<FPTYPE > &	        descrpt_a_deriv,
    std::vector<FPTYPE > &	        rij_a,
    const std::vector<FPTYPE > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			rmin,
    const float &			rmax,
    const FPTYPE precs[3])
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

    /*
    precs: NBIT_DATA_FL, NBIT_FEA_X, NBIT_FEA_X_FL
    */
   const double rc2 = rmax * rmax;


    for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter+1]; ++nei_iter) {      
            if (fmt_nlist_a[nei_iter] < 0) break;
            const FPTYPE * rr = &rij_a[nei_iter * 3];

            // NVNMD
            // FPTYPE rij[3];
            // rij[0] = round(rr[0] * precs[0]) / precs[0];
            // rij[1] = round(rr[1] * precs[0]) / precs[0];
            // rij[2] = round(rr[2] * precs[0]) / precs[0];
            // FPTYPE nr2 = deepmd::dot3(rij, rij);
            // nr2 = floor(nr2 * precs[0]) / precs[0];
            
            FPTYPE nr2 = deepmd::dot3(rr, rr);
            FPTYPE inr = 1./sqrt(nr2);
            FPTYPE nr = nr2 * inr;
            FPTYPE inr2 = inr * inr;
            FPTYPE inr4 = inr2 * inr2;
            FPTYPE inr3 = inr4 * nr;
            FPTYPE sw, dsw;
            deepmd::spline5_switch(sw, dsw, nr, rmin, rmax);
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



template
void 
deepmd::
env_mat_a_nvnmd_cpu<double> (
    std::vector<double > &	        descrpt_a,
    std::vector<double > &	        descrpt_a_deriv,
    std::vector<double > &	        rij_a,
    const std::vector<double > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax,
    const double      precs[3]) ;


template
void 
deepmd::
env_mat_a_nvnmd_cpu<float> (
    std::vector<float > &	        descrpt_a,
    std::vector<float > &	        descrpt_a_deriv,
    std::vector<float > &	        rij_a,
    const std::vector<float > &		posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax,
    const float       precs[3]);

/*
//==================================================
  env_mat_a_nvnmd_quantize_cpu
//==================================================
*/


template<typename FPTYPE> 
void 
deepmd::
env_mat_a_nvnmd_quantize_cpu (
    std::vector<FPTYPE > &	        descrpt_a,
    std::vector<FPTYPE > &	        descrpt_a_deriv,
    std::vector<FPTYPE > &	        rij_a,
    const std::vector<FPTYPE > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			rmin,
    const float &			rmax,
    const FPTYPE precs[3])
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

    /*
    precs: NBIT_DATA_FL, NBIT_FEA_X, NBIT_FEA_FL
    */
   const double rc2 = rmax * rmax;


    for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter+1]; ++nei_iter) {      
            if (fmt_nlist_a[nei_iter] < 0) break;
            const FPTYPE * rr = &rij_a[nei_iter * 3];

            // NVNMD
            FPTYPE rij[3];
            rij[0] = round(rr[0] * precs[0]) / precs[0];
            rij[1] = round(rr[1] * precs[0]) / precs[0];
            rij[2] = round(rr[2] * precs[0]) / precs[0];
            FPTYPE nr2 = deepmd::dot3(rij, rij);
            nr2 = floor(nr2 * precs[0]) / precs[0];

            // FPTYPE nr2 = deepmd::dot3(rr, rr);
            // FPTYPE inr = 1./sqrt(nr2);
            // FPTYPE nr = nr2 * inr;
            // FPTYPE inr2 = inr * inr;
            // FPTYPE inr4 = inr2 * inr2;
            // FPTYPE inr3 = inr4 * nr;
            // FPTYPE sw, dsw;
            // deepmd::spline5_switch(sw, dsw, nr, rmin, rmax);
            int idx_deriv = nei_iter * 4 * 3;	// 4 components time 3 directions
            int idx_value = nei_iter * 4;	// 4 components
            // 4 value components
            descrpt_a[idx_value + 0] = nr2;
            descrpt_a[idx_value + 1] = rij[0];
            descrpt_a[idx_value + 2] = rij[1];
            descrpt_a[idx_value + 3] = rij[2];
            // deriv of component 1/r
            descrpt_a_deriv[idx_deriv + 0] = -2 * rij[0];
            descrpt_a_deriv[idx_deriv + 1] = -2 * rij[1];
            descrpt_a_deriv[idx_deriv + 2] = -2 * rij[2];
            /*
            d(sw*x/r)_d(x) = x * d(sw/r)_d(x) + sw/r
            d(sw*y/r)_d(x) = y * d(sw/r)_d(x)
            */
            // deriv of component x/r
            descrpt_a_deriv[idx_deriv + 3] = -1;
            descrpt_a_deriv[idx_deriv + 4] =  0;
            descrpt_a_deriv[idx_deriv + 5] =  0;
            // deriv of component y/r2
            descrpt_a_deriv[idx_deriv + 6] =  0;
            descrpt_a_deriv[idx_deriv + 7] = -1;
            descrpt_a_deriv[idx_deriv + 8] =  0;
            // deriv of component z/r2
            descrpt_a_deriv[idx_deriv + 9] =  0;
            descrpt_a_deriv[idx_deriv +10] =  0;
            descrpt_a_deriv[idx_deriv +11] = -1;
        }
    }
}



template
void 
deepmd::
env_mat_a_nvnmd_quantize_cpu<double> (
    std::vector<double > &	        descrpt_a,
    std::vector<double > &	        descrpt_a_deriv,
    std::vector<double > &	        rij_a,
    const std::vector<double > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax,
    const double      precs[3]);


template
void 
deepmd::
env_mat_a_nvnmd_quantize_cpu<float> (
    std::vector<float > &	        descrpt_a,
    std::vector<float > &	        descrpt_a_deriv,
    std::vector<float > &	        rij_a,
    const std::vector<float > &		posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax,
    const float       precs[3]);


