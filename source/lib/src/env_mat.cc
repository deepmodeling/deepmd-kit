#include "env_mat.h"
#include "switcher.h"

// output deriv size: n_sel_a_nei x 4 x 12				    
//		      (1./rr, cos_theta, cos_phi, sin_phi)  x 4 x (x, y, z) 
void env_mat_a (
    std::vector<double > &		descrpt_a,
    std::vector<double > &		descrpt_a_deriv,
    std::vector<double > &		rij_a,
    const std::vector<double > &	posi,
    const int &				ntypes,
    const std::vector<int > &		type,
    const SimulationRegion<double> &	region,
    const bool &			b_pbc,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const double &			rmin, 
    const double &			rmax)
{  
  // compute the diff of the neighbors
  std::vector<std::vector<double > > sel_a_diff (sec_a.back());
  rij_a.resize (sec_a.back() * 3);
  fill (rij_a.begin(), rij_a.end(), 0.0);
  for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii){
    for (int jj = sec_a[ii]; jj < sec_a[ii+1]; ++jj){
      if (fmt_nlist_a[jj] < 0) break;
      sel_a_diff[jj].resize(3);
      const int & j_idx = fmt_nlist_a[jj];
      if (b_pbc){
	region.diffNearestNeighbor (posi[j_idx*3+0], posi[j_idx*3+1], posi[j_idx*3+2], 
				    posi[i_idx*3+0], posi[i_idx*3+1], posi[i_idx*3+2], 
				    sel_a_diff[jj][0], sel_a_diff[jj][1], sel_a_diff[jj][2]);
      }
      else {
	for (int dd = 0; dd < 3; ++dd) sel_a_diff[jj][dd] = posi[j_idx*3+dd] - posi[i_idx*3+dd];
      }
      for (int dd = 0; dd < 3; ++dd) rij_a[jj*3+dd] = sel_a_diff[jj][dd];
    }
  }
  
  // 1./rr, cos(theta), cos(phi), sin(phi)
  descrpt_a.resize (sec_a.back() * 4);
  fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
  // deriv wrt center: 3
  descrpt_a_deriv.resize (sec_a.back() * 4 * 3);
  fill (descrpt_a_deriv.begin(), descrpt_a_deriv.end(), 0.0);

  for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter){
    for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter+1]; ++nei_iter) {      
      if (fmt_nlist_a[nei_iter] < 0) break;
      const double * rr = &sel_a_diff[nei_iter][0];
      double nr2 = deepmd::dot3(rr, rr);
      double inr = 1./sqrt(nr2);
      double nr = nr2 * inr;
      double inr2 = inr * inr;
      double inr4 = inr2 * inr2;
      double inr3 = inr4 * nr;
      double sw, dsw;
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


template<typename FPTYPE> 
void 
deepmd::
env_mat_a_cpu (
    std::vector<FPTYPE > &	        descrpt_a,
    std::vector<FPTYPE > &	        descrpt_a_deriv,
    std::vector<FPTYPE > &	        rij_a,
    const std::vector<FPTYPE > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist_a,
    const std::vector<int > &		sec_a, 
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


void env_mat_r (
    std::vector<double > &		descrpt,
    std::vector<double > &		descrpt_deriv,
    std::vector<double > &		rij,
    const std::vector<double > &	posi,
    const int &				ntypes,
    const std::vector<int > &		type,
    const SimulationRegion<double> &	region,
    const bool &			b_pbc,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec,
    const double &			rmin, 
    const double &			rmax)
{  
  // compute the diff of the neighbors
  std::vector<std::vector<double > > sel_diff (sec.back());
  rij.resize (sec.back() * 3);
  fill (rij.begin(), rij.end(), 0.0);
  for (int ii = 0; ii < int(sec.size()) - 1; ++ii){
    for (int jj = sec[ii]; jj < sec[ii+1]; ++jj){
      if (fmt_nlist[jj] < 0) break;
      sel_diff[jj].resize(3);
      const int & j_idx = fmt_nlist[jj];
      if (b_pbc){
	region.diffNearestNeighbor (posi[j_idx*3+0], posi[j_idx*3+1], posi[j_idx*3+2], 
				    posi[i_idx*3+0], posi[i_idx*3+1], posi[i_idx*3+2], 
				    sel_diff[jj][0], sel_diff[jj][1], sel_diff[jj][2]);
      }
      else {
	for (int dd = 0; dd < 3; ++dd) sel_diff[jj][dd] = posi[j_idx*3+dd] - posi[i_idx*3+dd];
      }
      for (int dd = 0; dd < 3; ++dd) rij[jj*3+dd] = sel_diff[jj][dd];
    }
  }
  
  // 1./rr
  descrpt.resize (sec.back());
  fill (descrpt.begin(), descrpt.end(), 0.0);
  // deriv wrt center: 3
  descrpt_deriv.resize (sec.back() * 3);
  fill (descrpt_deriv.begin(), descrpt_deriv.end(), 0.0);

  for (int sec_iter = 0; sec_iter < int(sec.size()) - 1; ++sec_iter){
    for (int nei_iter = sec[sec_iter]; nei_iter < sec[sec_iter+1]; ++nei_iter) {      
      if (fmt_nlist[nei_iter] < 0) break;
      const double * rr = &sel_diff[nei_iter][0];
      double nr2 = deepmd::dot3(rr, rr);
      double inr = 1./sqrt(nr2);
      double nr = nr2 * inr;
      double inr2 = inr * inr;
      double inr4 = inr2 * inr2;
      double inr3 = inr4 * nr;
      double sw, dsw;
      deepmd::spline5_switch(sw, dsw, nr, rmin, rmax);
      int idx_deriv = nei_iter * 3;	// 1 components time 3 directions
      int idx_value = nei_iter;		// 1 components
      // value components
      descrpt[idx_value + 0] = 1./nr;
      // deriv of component 1/r
      descrpt_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - descrpt[idx_value + 0] * dsw * rr[0] * inr;
      descrpt_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - descrpt[idx_value + 0] * dsw * rr[1] * inr;
      descrpt_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - descrpt[idx_value + 0] * dsw * rr[2] * inr;
      // value components
      descrpt[idx_value + 0] *= sw;
    }
  }
}

template<typename FPTYPE> 
void 
deepmd::
env_mat_r_cpu (
    std::vector<FPTYPE > &		descrpt_a,
    std::vector<FPTYPE > &	        descrpt_a_deriv,
    std::vector<FPTYPE > &	        rij_a,
    const std::vector<FPTYPE > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax) 
{
    // compute the diff of the neighbors
    rij_a.resize (sec.back() * 3);
    fill (rij_a.begin(), rij_a.end(), 0.0);
    for (int ii = 0; ii < int(sec.size()) - 1; ++ii) {
        for (int jj = sec[ii]; jj < sec[ii + 1]; ++jj) {
            if (fmt_nlist[jj] < 0) break;
            const int & j_idx = fmt_nlist[jj];

            for (int dd = 0; dd < 3; ++dd) {
                rij_a[jj * 3 + dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
            }
        }
    }
    // 1./rr, cos(theta), cos(phi), sin(phi)
    descrpt_a.resize (sec.back());
    fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
    // deriv wrt center: 3
    descrpt_a_deriv.resize (sec.back() * 3);
    fill (descrpt_a_deriv.begin(), descrpt_a_deriv.end(), 0.0);

    for (int sec_iter = 0; sec_iter < int(sec.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec[sec_iter]; nei_iter < sec[sec_iter+1]; ++nei_iter) {      
            if (fmt_nlist[nei_iter] < 0) break;
            const FPTYPE * rr = &rij_a[nei_iter * 3];
            FPTYPE nr2 = deepmd::dot3(rr, rr);
            FPTYPE inr = 1./sqrt(nr2);
            FPTYPE nr = nr2 * inr;
            FPTYPE inr2 = inr * inr;
            FPTYPE inr4 = inr2 * inr2;
            FPTYPE inr3 = inr4 * nr;
            FPTYPE sw, dsw;
            deepmd::spline5_switch(sw, dsw, nr, rmin, rmax);
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


template
void 
deepmd::
env_mat_a_cpu<double> (
    std::vector<double > &	        descrpt_a,
    std::vector<double > &	        descrpt_a_deriv,
    std::vector<double > &	        rij_a,
    const std::vector<double > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax) ;


template
void 
deepmd::
env_mat_a_cpu<float> (
    std::vector<float > &	        descrpt_a,
    std::vector<float > &	        descrpt_a_deriv,
    std::vector<float > &	        rij_a,
    const std::vector<float > &		posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax) ;


template
void 
deepmd::
env_mat_r_cpu<double> (
    std::vector<double > &	        descrpt_r,
    std::vector<double > &	        descrpt_r_deriv,
    std::vector<double > &	        rij_r,
    const std::vector<double > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax) ;


template
void 
deepmd::
env_mat_r_cpu<float> (
    std::vector<float > &	        descrpt_r,
    std::vector<float > &	        descrpt_r_deriv,
    std::vector<float > &	        rij_r,
    const std::vector<float > &		posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax) ;


