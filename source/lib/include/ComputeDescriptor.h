#pragma once

#include <algorithm>
#include <iterator>
#include <cassert>

#include "SimulationRegion.h"
#include "utilities.h"
#include "switcher.h"


inline
void compute_descriptor (std::vector<double > &			descrpt_a,
			 std::vector<double > &			descrpt_r,
			 std::vector<double > &			rot_mat,
			 const std::vector<double > &		posi,
			 const int &				ntypes,
			 const std::vector<int > &		type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const std::vector<int > &		fmt_nlist_a,
			 const std::vector<int > &		fmt_nlist_r,
			 const std::vector<int > &		sec_a,
			 const std::vector<int > &		sec_r,
			 const int				axis0_type,
			 const int				axis0_idx,
			 const int				axis1_type,
			 const int				axis1_idx);

inline
void compute_descriptor (std::vector<double > &			descrpt_a,
			 std::vector<double > &			descrpt_a_deriv,
			 std::vector<double > &			descrpt_r,
			 std::vector<double > &			descrpt_r_deriv,
			 std::vector<double > &			rij_a,
			 std::vector<double > &			rij_r,
			 std::vector<double > &			rot_mat,
			 const std::vector<double > &		posi,
			 const int &				ntypes,
			 const std::vector<int > &		type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const std::vector<int > &		fmt_nlist_a,
			 const std::vector<int > &		fmt_nlist_r,
			 const std::vector<int > &		sec_a,
			 const std::vector<int > &		sec_r,
			 const int				axis0_type,
			 const int				axis0_idx,
			 const int				axis1_type,
			 const int				axis1_idx);


inline
void compute_descriptor_se_a_extf (std::vector<double > &	descrpt_a,
				   std::vector<double > &	descrpt_a_deriv,
				   std::vector<double > &	rij_a,
				   const std::vector<double > &	posi,
				   const int &			ntypes,
				   const std::vector<int > &	type,
				   const SimulationRegion<double> &	region,
				   const bool &			b_pbc,
				   const std::vector<double > &	efield,
				   const int &			i_idx,
				   const std::vector<int > &	fmt_nlist_a,
				   const std::vector<int > &	sec_a, 
				   const double &		rmin, 
				   const double &		rmax);
inline
void compute_descriptor_se_a_ef_para (std::vector<double > &			descrpt_a,
				      std::vector<double > &			descrpt_a_deriv,
				      std::vector<double > &			rij_a,
				      const std::vector<double > &		posi,
				      const int &				ntypes,
				      const std::vector<int > &			type,
				      const SimulationRegion<double> &		region,
				      const bool &				b_pbc,
				      const std::vector<double > &		efield,
				      const int &				i_idx,
				      const std::vector<int > &			fmt_nlist_a,
				      const std::vector<int > &			sec_a, 
				      const double &				rmin, 
				      const double &				rmax);
inline
void compute_descriptor_se_a_ef_vert (std::vector<double > &			descrpt_a,
				      std::vector<double > &			descrpt_a_deriv,
				      std::vector<double > &			rij_a,
				      const std::vector<double > &		posi,
				      const int &				ntypes,
				      const std::vector<int > &			type,
				      const SimulationRegion<double> &		region,
				      const bool &				b_pbc,
				      const std::vector<double > &		efield,
				      const int &				i_idx,
				      const std::vector<int > &			fmt_nlist_a,
				      const std::vector<int > &			sec_a, 
				      const double &				rmin, 
				      const double &				rmax);

static void 
compute_dRdT (double (* dRdT)[9], 
	      const double * r1, 
	      const double * r2, 
	      const double * rot)
{
  double * dRdT0 = dRdT[0];
  double * dRdT1 = dRdT[1];
  double * dRdT2 = dRdT[2];
  const double *xx = rot;
  const double *yy = rot+3;

  double nr1 = sqrt(deepmd::dot3(r1, r1));
  double nr12 = nr1  * nr1;
  double nr13 = nr1  * nr12;
  double nr14 = nr12 * nr12;
  double r1dr2 = deepmd::dot3(r1, r2);

  // dRdT0
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      dRdT0[ii*3+jj] = r1[ii] * r1[jj] / nr13;
      if (ii == jj) dRdT0[ii*3+jj] -= 1./nr1;
    }
  }
  
  // dRdT1  
  double dRdy[9];
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      dRdy[ii*3+jj] = (- 2 * r1dr2 / nr14 * r1[ii] * r1[jj] 
		       + (r1[ii] + r2[ii]) * r1[jj] / nr12 );
      if (ii == jj) {
	dRdy[ii*3+jj] += r1dr2 / nr12 - 1.;
      }
    }
  }
  double tmpy[3];
  for (int dd = 0; dd < 3; ++dd) tmpy[dd] = r2[dd] - r1dr2 / nr12 * r1[dd];
  double ntmpy = sqrt(deepmd::dot3(tmpy, tmpy));
  double ydRdy [3] = {0};
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      ydRdy[ii] += tmpy[jj] * dRdy[ii*3 + jj];
    }
  }
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      dRdT1[ii*3+jj] = (- ydRdy[ii] * tmpy[jj] / (ntmpy * ntmpy * ntmpy) 
			+ dRdy[3*ii+jj] / ntmpy );
    }
  }
  // dRdT2
  for (int ii = 0; ii < 3; ++ii){
    double res[3];
    deepmd::cprod(dRdT0 + ii*3, yy, dRdT2 + ii*3);
    deepmd::cprod(xx, dRdT1 + ii*3, res);
    for (int dd = 0; dd < 3; ++dd) dRdT2[ii*3+dd] += res[dd];
  }
}

static void 
compute_dRdT_1 (double (* dRdT)[9], 
		const double * r1, 
		const double * r2, 
		const double * rot)
{
  double * dRdT0 = dRdT[0];
  double * dRdT1 = dRdT[1];
  double * dRdT2 = dRdT[2];
  const double *xx = rot;
  const double *yy = rot+3;

  double nr1 = sqrt(deepmd::dot3(r1, r1));
  double nr12 = nr1  * nr1;
  double nr13 = nr1  * nr12;
  double nr14 = nr12 * nr12;
  double r1dr2 = deepmd::dot3(r1, r2);

  // dRdT0
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      dRdT0[ii*3+jj] = -r1[ii] * r1[jj] / nr13;
      if (ii == jj) dRdT0[ii*3+jj] += 1./nr1;
    }
  }
  
  // dRdT1  
  double dRdy[9];
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      dRdy[ii*3+jj] = (+ 2 * r1dr2 / nr14 * r1[ii] * r1[jj] 
		       - r2[ii] * r1[jj] / nr12 );
      if (ii == jj) {
	dRdy[ii*3+jj] -= r1dr2 / nr12;
      }
    }
  }
  double tmpy[3];
  for (int dd = 0; dd < 3; ++dd) tmpy[dd] = r2[dd] - r1dr2 / nr12 * r1[dd];
  double ntmpy = sqrt(deepmd::dot3(tmpy, tmpy));
  double ydRdy [3] = {0};
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      ydRdy[ii] += tmpy[jj] * dRdy[ii*3 + jj];
    }
  }
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      dRdT1[ii*3+jj] = (- ydRdy[ii] * tmpy[jj] / (ntmpy * ntmpy * ntmpy) 
			+ dRdy[3*ii+jj] / ntmpy );
    }
  }
  // dRdT2
  for (int ii = 0; ii < 3; ++ii){
    double res[3];
    deepmd::cprod(dRdT0 + ii*3, yy, dRdT2 + ii*3);
    deepmd::cprod(xx, dRdT1 + ii*3, res);
    for (int dd = 0; dd < 3; ++dd) dRdT2[ii*3+dd] += res[dd];
  }
}


static void 
compute_dRdT_2 (double (* dRdT)[9], 
		const double * r1, 
		const double * r2, 
		const double * rot)
{
  double * dRdT0 = dRdT[0];
  double * dRdT1 = dRdT[1];
  double * dRdT2 = dRdT[2];
  const double *xx = rot;
  const double *yy = rot+3;

  double nr1 = sqrt(deepmd::dot3(r1, r1));
  double nr12 = nr1  * nr1;
  double r1dr2 = deepmd::dot3(r1, r2);

  // dRdT0
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      dRdT0[ii*3+jj] = 0.;
    }
  }
  
  // dRdT1  
  double dRdy[9];
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj) {
      dRdy[ii*3+jj] = - r1[ii] * r1[jj] / nr12;
      if (ii == jj) {
	dRdy[ii*3+jj] += 1;
      }
    }
  }
  double tmpy[3];
  for (int dd = 0; dd < 3; ++dd) tmpy[dd] = r2[dd] - r1dr2 / nr12 * r1[dd];
  double ntmpy = sqrt(deepmd::dot3(tmpy, tmpy));
  double ydRdy [3] = {0};
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      ydRdy[ii] += tmpy[jj] * dRdy[ii*3 + jj];
    }
  }
  for (int ii = 0; ii < 3; ++ii){
    for (int jj = 0; jj < 3; ++jj){
      dRdT1[ii*3+jj] = (- ydRdy[ii] * tmpy[jj] / (ntmpy * ntmpy * ntmpy) 
			+ dRdy[3*ii+jj] / ntmpy );
    }
  }
  // dRdT2
  for (int ii = 0; ii < 3; ++ii){
    double res[3];
    deepmd::cprod(dRdT0 + ii*3, yy, dRdT2 + ii*3);
    deepmd::cprod(xx, dRdT1 + ii*3, res);
    for (int dd = 0; dd < 3; ++dd) dRdT2[ii*3+dd] += res[dd];
  }
}



// output deriv size: n_sel_a_nei x 4 x 12				    + n_sel_r_nei x 12
//		      (1./rr, cos_theta, cos_phi, sin_phi)  x 4 x (x, y, z) + (1./rr) x 4 x (x, y, z)
void compute_descriptor (std::vector<double > &			descrpt_a,
			 std::vector<double > &			descrpt_a_deriv,
			 std::vector<double > &			descrpt_r,
			 std::vector<double > &			descrpt_r_deriv,
			 std::vector<double > &			rij_a,
			 std::vector<double > &			rij_r,
			 std::vector<double > &			rot_mat,
			 const std::vector<double > &		posi,
			 const int &				ntypes,
			 const std::vector<int > &		type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const std::vector<int > &		fmt_nlist_a,
			 const std::vector<int > &		fmt_nlist_r,
			 const std::vector<int > &		sec_a,
			 const std::vector<int > &		sec_r,
			 const int				axis0_type,
			 const int				axis0_idx,
			 const int				axis1_type,
			 const int				axis1_idx)
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

  std::vector<std::vector<double > > sel_r_diff (sec_r.back());
  rij_r.resize (sec_r.back() * 3);
  fill (rij_r.begin(), rij_r.end(), 0.0);
  for (int ii = 0; ii < int(sec_r.size()) - 1; ++ii){
    for (int jj = sec_r[ii]; jj < sec_r[ii+1]; ++jj){
      if (fmt_nlist_r[jj] < 0) break;
      sel_r_diff[jj].resize(3);
      const int & j_idx = fmt_nlist_r[jj];
      if (b_pbc){
	region.diffNearestNeighbor (posi[j_idx*3+0], posi[j_idx*3+1], posi[j_idx*3+2], 
				    posi[i_idx*3+0], posi[i_idx*3+1], posi[i_idx*3+2], 
				    sel_r_diff[jj][0], sel_r_diff[jj][1], sel_r_diff[jj][2]);
      }
      else {
	for (int dd = 0; dd < 3; ++dd) sel_r_diff[jj][dd] = posi[j_idx*3+dd] - posi[i_idx*3+dd];
      }
      for (int dd = 0; dd < 3; ++dd) rij_r[jj*3+dd] = sel_r_diff[jj][dd];
    }
  }
  
  // if (i_idx == 0){
  //   for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii){
  //     for (int jj = sec_a[ii]; jj < sec_a[ii+1]; ++jj){
  // 	int j_idx = fmt_nlist_a[jj];
  // 	cout << "a list " ;
  // 	cout << jj << "\t  jidx " << j_idx;
  // 	if (j_idx >= 0){
  // 	  cout << "\t type " << type[j_idx];
  // 	  cout << "\t " << sqrt(deepmd::dot3(&sel_a_diff[jj][0], &sel_a_diff[jj][0]));
  // 	}
  // 	cout << endl;
  //     }
  //   }
  //   for (int ii = 0; ii < int(sec_r.size()) - 1; ++ii){
  //     for (int jj = sec_r[ii]; jj < sec_r[ii+1]; ++jj){
  // 	int j_idx = fmt_nlist_r[jj];
  // 	cout << "r list " ;
  // 	cout << jj << "\t  jidx " << j_idx;
  // 	if (j_idx >= 0){
  // 	  cout << "\t type " << type[j_idx];
  // 	  cout << "\t " << sqrt(deepmd::dot3(&sel_r_diff[jj][0], &sel_r_diff[jj][0]));
  // 	}
  // 	cout << endl;
  //     }
  //   }
  // }

  // record axis vectors
  double r1[3], r2[3];
  for (unsigned dd = 0; dd < 3; ++dd){
    if (axis0_type == 0){
      assert  (sel_a_diff[axis0_idx].size() == 3);
      r1[dd] = sel_a_diff[axis0_idx][dd];
    }
    else {
      assert  (sel_r_diff[axis0_idx].size() == 3);
      r1[dd] = sel_r_diff[axis0_idx][dd];
    }
    if (axis1_type == 0){
      assert  (sel_a_diff[axis1_idx].size() == 3);
      r2[dd] = sel_a_diff[axis1_idx][dd];
    }
    else {
      assert  (sel_r_diff[axis1_idx].size() == 3);
      r2[dd] = sel_r_diff[axis1_idx][dd];
    }
  }  

  // rotation matrix
  double rot [9];
  double *xx = rot;
  double *yy = rot+3;
  double *zz = rot+6;
  for (unsigned dd = 0; dd < 3; ++dd){
    xx[dd] = r1[dd];
    yy[dd] = r2[dd];
  }
  double norm_xx = sqrt(deepmd::dot3(xx, xx));
  for (unsigned dd = 0; dd < 3; ++dd) xx[dd] /= norm_xx;
  double dxy = deepmd::dot3(xx, yy);
  for (unsigned dd = 0; dd < 3; ++dd) yy[dd] -= dxy * xx[dd];
  double norm_yy = sqrt(deepmd::dot3(yy, yy));
  for (unsigned dd = 0; dd < 3; ++dd) yy[dd] /= norm_yy;
  deepmd::cprod(xx, yy, zz);  
  rot_mat.resize (9);
  for (int dd = 0; dd < 9; ++dd) rot_mat[dd] = rot[dd];

  // 1./rr, cos(theta), cos(phi), sin(phi)
  descrpt_a.resize (sec_a.back() * 4);
  fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
  for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii){
    for (int jj = sec_a[ii]; jj < sec_a[ii+1]; ++jj){
      if (fmt_nlist_a[jj] < 0) break;
      double rdiff[3] ;
      deepmd::dotmv3(rdiff, rot, &sel_a_diff[jj][0]);
      double rr2 = deepmd::dot3(rdiff, rdiff);
      double rr = sqrt(rr2);
#ifdef DESCRPT_THETAPHI
      double cos_theta = rdiff[2] / rr;
      double rxy = sqrt(rdiff[0] * rdiff[0] + rdiff[1] * rdiff[1]);
      double cos_phi = rdiff[0] / rxy;
      double sin_phi = rdiff[1] / rxy;
#else
      double cos_theta = rdiff[2] / rr2;
      double cos_phi = rdiff[0] / rr2;
      double sin_phi = rdiff[1] / rr2;
#endif
      descrpt_a[jj * 4 + 0] = 1./rr;
      descrpt_a[jj * 4 + 1] = cos_theta;
      descrpt_a[jj * 4 + 2] = cos_phi;
      descrpt_a[jj * 4 + 3] = sin_phi;      
    }
  }
  // 1./rr
  descrpt_r.resize (sec_r.back());
  fill (descrpt_r.begin(), descrpt_r.end(), 0.0);
  for (int ii = 0; ii < int(sec_r.size()) - 1; ++ii){
    for (int jj = sec_r[ii]; jj < sec_r[ii+1]; ++jj){
      if (fmt_nlist_r[jj] < 0) break;
      const double *rdiff = &sel_r_diff[jj][0];
      double rr = sqrt (deepmd::dot3(rdiff, rdiff));
      descrpt_r[jj] = 1./rr;      
    }
  }
  
  // first_dim: T_i, second_dim: R_k (T_i)_j
  double dRdT_0[3][9];
  double dRdT_1[3][9];
  double dRdT_2[3][9];
  if (sec_a.back() > 0) {
    compute_dRdT   (dRdT_0, r1, r2, rot);
    compute_dRdT_1 (dRdT_1, r1, r2, rot);
    compute_dRdT_2 (dRdT_2, r1, r2, rot);
  }

  // deriv wrt center: 3
  // deriv wrt axis 1: 3
  // deriv wrt axis 2: 3
  // deriv wrt atom k: 3
  // if k == 1 or k == 2, 2 copies of data stored.
  descrpt_a_deriv.resize (sec_a.back() * 4 * 12);
  fill (descrpt_a_deriv.begin(), descrpt_a_deriv.end(), 0.);
  for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter){
    for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter+1]; ++nei_iter) {
      if (fmt_nlist_a[nei_iter] < 0) break;
      // drdS, stored in tranposed form
      double dtrdST[4][3];
      double * rr = &sel_a_diff[nei_iter][0];
      double tr[3] ;
      deepmd::dotmv3(tr, rot, rr);
      double nr2 = deepmd::dot3(tr, tr);
      double nr = sqrt(nr2);
      double nr3 = nr * nr2;
      for (int dd = 0; dd < 3; ++dd){
	dtrdST[0][dd] = -tr[dd] / nr3;
      }
#ifdef DESCRPT_THETAPHI
      for (int dd = 0; dd < 3; ++dd){
	dtrdST[1][dd] = -tr[dd] / nr3 * tr[2];
      }
      dtrdST[1][2] += 1./nr;
      double nr01 = sqrt(tr[0] * tr[0] + tr[1] * tr[1]);
      double nr013 = nr01 * nr01 * nr01;
      dtrdST[2][0] = -tr[0] * tr[0] / nr013 + 1./nr01;
      dtrdST[2][1] = -tr[1] * tr[0] / nr013;
      dtrdST[2][2] = 0.;
      dtrdST[3][0] = -tr[0] * tr[1] / nr013;
      dtrdST[3][1] = -tr[1] * tr[1] / nr013 + 1./nr01;
      dtrdST[3][2] = 0.;
#else
      double nr4 = nr2 * nr2;
      for (int dd = 0; dd < 3; ++dd){
	dtrdST[1][dd] = -2. * tr[dd] / nr4 * tr[2];
	dtrdST[2][dd] = -2. * tr[dd] / nr4 * tr[0];
	dtrdST[3][dd] = -2. * tr[dd] / nr4 * tr[1];
      }
      dtrdST[1][2] += 1./nr2;
      dtrdST[2][0] += 1./nr2;
      dtrdST[3][1] += 1./nr2;
#endif
      // dRdTr
      double dRdTr_0[3][3];
      for (int ii = 0; ii < 3; ++ii){
	for (int jj = 0; jj < 3; ++jj){
	  dRdTr_0[ii][jj] = 0;	
	  for (int ll = 0; ll < 3; ++ll){
	    dRdTr_0[ii][jj] += dRdT_0[jj][ii*3+ll] * rr[ll];
	  }
	  dRdTr_0[ii][jj] -= rot[jj*3 + ii];
	}
      }
      // dRdTr_1
      double dRdTr_1[3][3];
      for (int ii = 0; ii < 3; ++ii){
	for (int jj = 0; jj < 3; ++jj){
	  dRdTr_1[ii][jj] = 0;	
	  for (int ll = 0; ll < 3; ++ll){
	    dRdTr_1[ii][jj] += dRdT_1[jj][ii*3+ll] * rr[ll];
	  }
	  if (axis0_type == 0 && nei_iter == axis0_idx) dRdTr_1[ii][jj] += rot[jj*3 + ii];
	}
      }
      // dRdTr_2
      double dRdTr_2[3][3];
      for (int ii = 0; ii < 3; ++ii){
	for (int jj = 0; jj < 3; ++jj){
	  dRdTr_2[ii][jj] = 0;	
	  for (int ll = 0; ll < 3; ++ll){
	    dRdTr_2[ii][jj] += dRdT_2[jj][ii*3+ll] * rr[ll];
	  }
	  if (axis1_type == 0 && nei_iter == axis1_idx) dRdTr_2[ii][jj] += rot[jj*3 + ii];
	}
      }
      // dRdTr_k
      double dRdTr_k[3][3];
      for (int ii = 0; ii < 3; ++ii){
	for (int jj = 0; jj < 3; ++jj){
	  dRdTr_k[ii][jj] = 0;	
	  if (axis0_type == 0 && nei_iter == axis0_idx){
	    for (int ll = 0; ll < 3; ++ll){
	      dRdTr_k[ii][jj] += dRdT_1[jj][ii*3+ll] * rr[ll];
	    }
	  }
	  if (axis1_type == 0 && nei_iter == axis1_idx){
	    for (int ll = 0; ll < 3; ++ll){
	      dRdTr_k[ii][jj] += dRdT_2[jj][ii*3+ll] * rr[ll];
	    }
	  }
	  dRdTr_k[ii][jj] += rot[jj*3 + ii];
	}
      }

      // assemble
      // 4 components times 12 derivs
      int idx_start = nei_iter * 4 * 12;
      // loop over components
      for (int ii = 0; ii < 4; ++ii){
	for (int jj = 0; jj < 3; ++jj){
	  int idx = idx_start + ii * 12 + jj;
	  descrpt_a_deriv[idx] = 0;
	  for (int ll = 0; ll < 3; ++ll){
	    descrpt_a_deriv[idx] += dtrdST[ii][ll] * dRdTr_0[jj][ll];
	  }
	}
	for (int jj = 0; jj < 3; ++jj){
	  int idx = idx_start + ii * 12 + jj + 3;
	  descrpt_a_deriv[idx] = 0;
	  for (int ll = 0; ll < 3; ++ll){
	    descrpt_a_deriv[idx] += dtrdST[ii][ll] * dRdTr_1[jj][ll];
	  }
	}
	for (int jj = 0; jj < 3; ++jj){
	  int idx = idx_start + ii * 12 + jj + 6;
	  descrpt_a_deriv[idx] = 0;
	  for (int ll = 0; ll < 3; ++ll){
	    descrpt_a_deriv[idx] += dtrdST[ii][ll] * dRdTr_2[jj][ll];
	  }
	}
	for (int jj = 0; jj < 3; ++jj){
	  int idx = idx_start + ii * 12 + jj + 9;
	  descrpt_a_deriv[idx] = 0;
	  for (int ll = 0; ll < 3; ++ll){
	    descrpt_a_deriv[idx] += dtrdST[ii][ll] * dRdTr_k[jj][ll];
	  }
	}
      }
    }
  } 

  descrpt_r_deriv.resize (sec_r.back() * 1 * 12);
  fill (descrpt_r_deriv.begin(), descrpt_r_deriv.end(), 0.);
  for (int sec_iter = 0; sec_iter < int(sec_r.size()) - 1; ++sec_iter){
    for (int nei_iter = sec_r[sec_iter]; nei_iter < sec_r[sec_iter+1]; ++nei_iter) {
      if (fmt_nlist_r[nei_iter] < 0) break;      

      const double * rr = &sel_r_diff[nei_iter][0];
      double nr = sqrt(deepmd::dot3(rr, rr));
      double nr3 = nr * nr * nr;
      int idx = nei_iter * 12;

      for (int jj = 0; jj < 3; ++jj){
	double value = rr[jj] / nr3;
	descrpt_r_deriv[idx+0+jj] =  value;
	descrpt_r_deriv[idx+9+jj] = -value;
	if (nei_iter == axis0_idx) {
	  descrpt_r_deriv[idx+3+jj] = -value;
	}
	if (nei_iter == axis1_idx) {
	  descrpt_r_deriv[idx+6+jj] = -value;
	}
      }
    }
  }
}


void compute_descriptor (std::vector<double > &			descrpt_a,
			 std::vector<double > &			descrpt_r,
			 std::vector<double > &			rot_mat,
			 const std::vector<double > &		posi,
			 const int &				ntypes,
			 const std::vector<int > &		type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const std::vector<int > &		fmt_nlist_a,
			 const std::vector<int > &		fmt_nlist_r,
			 const std::vector<int > &		sec_a,
			 const std::vector<int > &		sec_r,
			 const int				axis0_type,
			 const int				axis0_idx,
			 const int				axis1_type,
			 const int				axis1_idx)
{  
  // compute the diff of the neighbors
  std::vector<std::vector<double > > sel_a_diff (sec_a.back());
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
    }
  }
  std::vector<std::vector<double > > sel_r_diff (sec_r.back());
  for (int ii = 0; ii < int(sec_r.size()) - 1; ++ii){
    for (int jj = sec_r[ii]; jj < sec_r[ii+1]; ++jj){
      if (fmt_nlist_r[jj] < 0) break;
      sel_r_diff[jj].resize(3);
      const int & j_idx = fmt_nlist_r[jj];
      if (b_pbc){
	region.diffNearestNeighbor (posi[j_idx*3+0], posi[j_idx*3+1], posi[j_idx*3+2], 
				    posi[i_idx*3+0], posi[i_idx*3+1], posi[i_idx*3+2], 
				    sel_r_diff[jj][0], sel_r_diff[jj][1], sel_r_diff[jj][2]);
      }
      else {
	for (int dd = 0; dd < 3; ++dd) sel_r_diff[jj][dd] = posi[j_idx*3+dd] - posi[i_idx*3+dd];
      }
    }
  }

  // record axis vectors
  double r1[3], r2[3];
  for (unsigned dd = 0; dd < 3; ++dd){
    if (axis0_type == 0){
      r1[dd] = sel_a_diff[axis0_idx][dd];
    }
    else {
      r1[dd] = sel_r_diff[axis0_idx][dd];
    }
    if (axis1_type == 0){
      r2[dd] = sel_a_diff[axis1_idx][dd];
    }
    else {
      r2[dd] = sel_r_diff[axis1_idx][dd];
    }
  }  

  // rotation matrix
  double rot [9];
  double *xx = rot;
  double *yy = rot+3;
  double *zz = rot+6;
  for (unsigned dd = 0; dd < 3; ++dd){
    xx[dd] = r1[dd];
    yy[dd] = r2[dd];
  }
  double norm_xx = sqrt(deepmd::dot3(xx, xx));
  for (unsigned dd = 0; dd < 3; ++dd) xx[dd] /= norm_xx;
  double dxy = deepmd::dot3(xx, yy);
  for (unsigned dd = 0; dd < 3; ++dd) yy[dd] -= dxy * xx[dd];
  double norm_yy = sqrt(deepmd::dot3(yy, yy));
  for (unsigned dd = 0; dd < 3; ++dd) yy[dd] /= norm_yy;
  deepmd::cprod(xx, yy, zz);  
  rot_mat.resize (9);
  for (int dd = 0; dd < 9; ++dd) rot_mat[dd] = rot[dd];

  // 1./rr, cos(theta), cos(phi), sin(phi)
  descrpt_a.resize (sec_a.back() * 4);
  fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
  for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii){
    for (int jj = sec_a[ii]; jj < sec_a[ii+1]; ++jj){
      if (fmt_nlist_a[jj] < 0) break;
      double rdiff[3] ;
      deepmd::dotmv3(rdiff, rot, &sel_a_diff[jj][0]);
      double rr2 = deepmd::dot3(rdiff, rdiff);
      double rr = sqrt(rr2);
#ifdef DESCRPT_THETAPHI
      double cos_theta = rdiff[2] / rr;
      double rxy = sqrt(rdiff[0] * rdiff[0] + rdiff[1] * rdiff[1]);
      double cos_phi = rdiff[0] / rxy;
      double sin_phi = rdiff[1] / rxy;
#else
      double cos_theta = rdiff[2] / rr2;
      double cos_phi = rdiff[0] / rr2;
      double sin_phi = rdiff[1] / rr2;
#endif
      descrpt_a[jj * 4 + 0] = 1./rr;
      descrpt_a[jj * 4 + 1] = cos_theta;
      descrpt_a[jj * 4 + 2] = cos_phi;
      descrpt_a[jj * 4 + 3] = sin_phi;      
    }
  }
  // 1./rr
  descrpt_r.resize (sec_r.back());
  fill (descrpt_r.begin(), descrpt_r.end(), 0.0);
  for (int ii = 0; ii < int(sec_r.size()) - 1; ++ii){
    for (int jj = sec_r[ii]; jj < sec_r[ii+1]; ++jj){
      if (fmt_nlist_r[jj] < 0) break;
      double rdiff[3] ;
      deepmd::dotmv3(rdiff, rot, &sel_r_diff[jj][0]);
      double rr = sqrt (deepmd::dot3(rdiff, rdiff));
      descrpt_r[jj] = 1./rr;
    }
  }  
}








// output deriv size: n_sel_a_nei x 4 x 12				    
//		      (1./rr, cos_theta, cos_phi, sin_phi)  x 4 x (x, y, z) 
void compute_descriptor_se_a_extf (std::vector<double > &		descrpt_a,
				   std::vector<double > &		descrpt_a_deriv,
				   std::vector<double > &		rij_a,
				   const std::vector<double > &		posi,
				   const int &				ntypes,
				   const std::vector<int > &		type,
				   const SimulationRegion<double> &	region,
				   const bool &				b_pbc,
				   const std::vector<double > &		efield,
				   const int &				i_idx,
				   const std::vector<int > &		fmt_nlist_a,
				   const std::vector<int > &		sec_a, 
				   const double &			rmin, 
				   const double &			rmax)
{
  const double * ef_ = &efield[i_idx*3+0];
  double ef[3] = {0.};
  if (std::isnan(ef_[0]) || std::isnan(ef_[1]) || std::isnan(ef_[2])){
    ef[0] = 1.;
    ef[1] = ef[2] = 0.;
  }
  else {
    for (int ii = 0; ii < 3; ++ii){
      ef[ii] = ef_[ii];
    }
  }
  assert( fabs(deepmd::dot3(ef, ef) - 1.0) < 1e-12 ), "ef should be a normalized std::vector";

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
      // check validity of ef
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
      // projections
      double rp = deepmd::dot3(rr, ef);
      double rv[3];
      rv[0] = rr[0] - rp * ef[0];
      rv[1] = rr[1] - rp * ef[1];
      rv[2] = rr[2] - rp * ef[2];
      // 4 value components
      descrpt_a[idx_value + 0] = rp / nr2;
      descrpt_a[idx_value + 1] = rv[0] / nr2;
      descrpt_a[idx_value + 2] = rv[1] / nr2;
      descrpt_a[idx_value + 3] = rv[2] / nr2;
      // deriv of component rp/r2
      descrpt_a_deriv[idx_deriv + 0] = (2. * inr4 * rp * rr[0] - inr2 * ef[0]) * sw - descrpt_a[idx_value + 0] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 1] = (2. * inr4 * rp * rr[1] - inr2 * ef[1]) * sw - descrpt_a[idx_value + 0] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 2] = (2. * inr4 * rp * rr[2] - inr2 * ef[2]) * sw - descrpt_a[idx_value + 0] * dsw * rr[2] * inr;
      // deriv of component rvx/r2
      descrpt_a_deriv[idx_deriv + 3] = (2. * inr4 * rv[0] * rr[0] - inr2 * (1. - ef[0] * ef[0])) * sw - descrpt_a[idx_value + 1] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 4] = (2. * inr4 * rv[0] * rr[1] - inr2 * (   - ef[0] * ef[1])) * sw - descrpt_a[idx_value + 1] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 5] = (2. * inr4 * rv[0] * rr[2] - inr2 * (   - ef[0] * ef[2])) * sw - descrpt_a[idx_value + 1] * dsw * rr[2] * inr;
      // deriv of component rvy/r2
      descrpt_a_deriv[idx_deriv + 6] = (2. * inr4 * rv[1] * rr[0] - inr2 * (   - ef[1] * ef[0])) * sw - descrpt_a[idx_value + 2] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 7] = (2. * inr4 * rv[1] * rr[1] - inr2 * (1. - ef[1] * ef[1])) * sw - descrpt_a[idx_value + 2] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 8] = (2. * inr4 * rv[1] * rr[2] - inr2 * (   - ef[1] * ef[2])) * sw - descrpt_a[idx_value + 2] * dsw * rr[2] * inr;
      // deriv of component rvz/r2
      descrpt_a_deriv[idx_deriv + 9] = (2. * inr4 * rv[2] * rr[0] - inr2 * (   - ef[2] * ef[0])) * sw - descrpt_a[idx_value + 3] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv +10] = (2. * inr4 * rv[2] * rr[1] - inr2 * (   - ef[2] * ef[1])) * sw - descrpt_a[idx_value + 3] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv +11] = (2. * inr4 * rv[2] * rr[2] - inr2 * (1. - ef[2] * ef[2])) * sw - descrpt_a[idx_value + 3] * dsw * rr[2] * inr;
      // 4 value components
      descrpt_a[idx_value + 0] *= sw;
      descrpt_a[idx_value + 1] *= sw;
      descrpt_a[idx_value + 2] *= sw;
      descrpt_a[idx_value + 3] *= sw;
    }
  }
}

// output deriv size: n_sel_a_nei x 4 x 12				    
//		      (1./rr, cos_theta, cos_phi, sin_phi)  x 4 x (x, y, z) 
void compute_descriptor_se_a_ef_para (std::vector<double > &		descrpt_a,
				      std::vector<double > &		descrpt_a_deriv,
				      std::vector<double > &		rij_a,
				      const std::vector<double > &	posi,
				      const int &			ntypes,
				      const std::vector<int > &		type,
				      const SimulationRegion<double> &	region,
				      const bool &			b_pbc,
				      const std::vector<double > &	efield,
				      const int &			i_idx,
				      const std::vector<int > &		fmt_nlist_a,
				      const std::vector<int > &		sec_a, 
				      const double &			rmin, 
				      const double &			rmax)
{
  const double * ef_ = &efield[i_idx*3+0];
  double ef[3] = {0.};
  if (std::isnan(ef_[0]) || std::isnan(ef_[1]) || std::isnan(ef_[2])){
    ef[0] = 1.;
    ef[1] = ef[2] = 0.;
  }
  else {
    for (int ii = 0; ii < 3; ++ii){
      ef[ii] = ef_[ii];
    }
  }
  assert( fabs(deepmd::dot3(ef, ef) - 1.0) < 1e-12 ), "ef should be a normalized vector";

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
      // check validity of ef
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
      // projections
      double rp[3];
      rp[0] = deepmd::dot3(rr, ef) * ef[0];
      rp[1] = deepmd::dot3(rr, ef) * ef[1];
      rp[2] = deepmd::dot3(rr, ef) * ef[2];
      // 4 value components
      descrpt_a[idx_value + 0] = 1 / nr;
      descrpt_a[idx_value + 1] = rp[0] / nr2;
      descrpt_a[idx_value + 2] = rp[1] / nr2;
      descrpt_a[idx_value + 3] = rp[2] / nr2;
      // deriv of component 1/r
      descrpt_a_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[2] * inr;
      // deriv of component rpx/r2
      descrpt_a_deriv[idx_deriv + 3] = (2. * inr4 * rp[0] * rr[0] - inr2 * (ef[0] * ef[0])) * sw - descrpt_a[idx_value + 1] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 4] = (2. * inr4 * rp[0] * rr[1] - inr2 * (ef[0] * ef[1])) * sw - descrpt_a[idx_value + 1] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 5] = (2. * inr4 * rp[0] * rr[2] - inr2 * (ef[0] * ef[2])) * sw - descrpt_a[idx_value + 1] * dsw * rr[2] * inr;
      // deriv of component rpy/r2
      descrpt_a_deriv[idx_deriv + 6] = (2. * inr4 * rp[1] * rr[0] - inr2 * (ef[1] * ef[0])) * sw - descrpt_a[idx_value + 2] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 7] = (2. * inr4 * rp[1] * rr[1] - inr2 * (ef[1] * ef[1])) * sw - descrpt_a[idx_value + 2] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 8] = (2. * inr4 * rp[1] * rr[2] - inr2 * (ef[1] * ef[2])) * sw - descrpt_a[idx_value + 2] * dsw * rr[2] * inr;
      // deriv of component rpz/r2
      descrpt_a_deriv[idx_deriv + 9] = (2. * inr4 * rp[2] * rr[0] - inr2 * (ef[2] * ef[0])) * sw - descrpt_a[idx_value + 3] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv +10] = (2. * inr4 * rp[2] * rr[1] - inr2 * (ef[2] * ef[1])) * sw - descrpt_a[idx_value + 3] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv +11] = (2. * inr4 * rp[2] * rr[2] - inr2 * (ef[2] * ef[2])) * sw - descrpt_a[idx_value + 3] * dsw * rr[2] * inr;
      // 4 value components
      descrpt_a[idx_value + 0] *= sw;
      descrpt_a[idx_value + 1] *= sw;
      descrpt_a[idx_value + 2] *= sw;
      descrpt_a[idx_value + 3] *= sw;
    }
  }
}

// output deriv size: n_sel_a_nei x 4 x 12				    
//		      (1./rr, cos_theta, cos_phi, sin_phi)  x 4 x (x, y, z) 
void compute_descriptor_se_a_ef_vert (std::vector<double > &		descrpt_a,
				      std::vector<double > &		descrpt_a_deriv,
				      std::vector<double > &		rij_a,
				      const std::vector<double > &	posi,
				      const int &			ntypes,
				      const std::vector<int > &		type,
				      const SimulationRegion<double> &	region,
				      const bool &			b_pbc,
				      const std::vector<double > &	efield,
				      const int &			i_idx,
				      const std::vector<int > &		fmt_nlist_a,
				      const std::vector<int > &		sec_a, 
				      const double &			rmin, 
				      const double &			rmax)
{
  const double * ef_ = &efield[i_idx*3+0];
  double ef[3] = {0.};
  if (std::isnan(ef_[0]) || std::isnan(ef_[1]) || std::isnan(ef_[2])){
    ef[0] = 1.;
    ef[1] = ef[2] = 0.;
  }
  else {
    for (int ii = 0; ii < 3; ++ii){
      ef[ii] = ef_[ii];
    }
  }
  assert( fabs(deepmd::dot3(ef, ef) - 1.0) < 1e-12 ), "ef should be a normalized vector";

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
      // check validity of ef
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
      // projections
      double rp = deepmd::dot3(rr, ef);
      double rv[3];
      rv[0] = rr[0] - rp * ef[0];
      rv[1] = rr[1] - rp * ef[1];
      rv[2] = rr[2] - rp * ef[2];
      // 4 value components
      descrpt_a[idx_value + 0] = 1 / nr;
      descrpt_a[idx_value + 1] = rv[0] / nr2;
      descrpt_a[idx_value + 2] = rv[1] / nr2;
      descrpt_a[idx_value + 3] = rv[2] / nr2;
      // deriv of component 1/r
      descrpt_a_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[2] * inr;
      // deriv of component rvx/r2
      descrpt_a_deriv[idx_deriv + 3] = (2. * inr4 * rv[0] * rr[0] - inr2 * (1. - ef[0] * ef[0])) * sw - descrpt_a[idx_value + 1] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 4] = (2. * inr4 * rv[0] * rr[1] - inr2 * (   - ef[0] * ef[1])) * sw - descrpt_a[idx_value + 1] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 5] = (2. * inr4 * rv[0] * rr[2] - inr2 * (   - ef[0] * ef[2])) * sw - descrpt_a[idx_value + 1] * dsw * rr[2] * inr;
      // deriv of component rvy/r2
      descrpt_a_deriv[idx_deriv + 6] = (2. * inr4 * rv[1] * rr[0] - inr2 * (   - ef[1] * ef[0])) * sw - descrpt_a[idx_value + 2] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv + 7] = (2. * inr4 * rv[1] * rr[1] - inr2 * (1. - ef[1] * ef[1])) * sw - descrpt_a[idx_value + 2] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv + 8] = (2. * inr4 * rv[1] * rr[2] - inr2 * (   - ef[1] * ef[2])) * sw - descrpt_a[idx_value + 2] * dsw * rr[2] * inr;
      // deriv of component rvz/r2
      descrpt_a_deriv[idx_deriv + 9] = (2. * inr4 * rv[2] * rr[0] - inr2 * (   - ef[2] * ef[0])) * sw - descrpt_a[idx_value + 3] * dsw * rr[0] * inr;
      descrpt_a_deriv[idx_deriv +10] = (2. * inr4 * rv[2] * rr[1] - inr2 * (   - ef[2] * ef[1])) * sw - descrpt_a[idx_value + 3] * dsw * rr[1] * inr;
      descrpt_a_deriv[idx_deriv +11] = (2. * inr4 * rv[2] * rr[2] - inr2 * (1. - ef[2] * ef[2])) * sw - descrpt_a[idx_value + 3] * dsw * rr[2] * inr;
      // 4 value components
      descrpt_a[idx_value + 0] *= sw;
      descrpt_a[idx_value + 1] *= sw;
      descrpt_a[idx_value + 2] *= sw;
      descrpt_a[idx_value + 3] *= sw;
    }
  }
}



