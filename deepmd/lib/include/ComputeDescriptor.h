#pragma once

#include <algorithm>
#include <iterator>
#include <cassert>

#include "SimulationRegion.h"
#include "MathUtilities.h"


// return:	-1	OK
//		> 0	the type of unsuccessful neighbor list
inline
int format_nlist_fill_a (vector<int > &				fmt_nei_idx_a,
			 vector<int > &				fmt_nei_idx_r,
			 const vector<double > &		posi,
			 const int &				ntypes,
			 const vector<int > &			type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const vector<int > &			nei_idx_a, 
			 const vector<int > &			nei_idx_r, 
			 const double &				rcut,
			 const vector<int > &			sec_a, 
			 const vector<int > &			sec_r);

inline
void compute_descriptor (vector<double > &			descrpt_a,
			 vector<double > &			descrpt_r,
			 vector<double > &			rot_mat,
			 const vector<double > &		posi,
			 const int &				ntypes,
			 const vector<int > &			type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const vector<int > &			fmt_nlist_a,
			 const vector<int > &			fmt_nlist_r,
			 const vector<int > &			sec_a,
			 const vector<int > &			sec_r,
			 const int				axis0_type,
			 const int				axis0_idx,
			 const int				axis1_type,
			 const int				axis1_idx);

inline
void compute_descriptor (vector<double > &			descrpt_a,
			 vector<double > &			descrpt_a_deriv,
			 vector<double > &			descrpt_r,
			 vector<double > &			descrpt_r_deriv,
			 vector<double > &			rij_a,
			 vector<double > &			rij_r,
			 vector<double > &			rot_mat,
			 const vector<double > &		posi,
			 const int &				ntypes,
			 const vector<int > &			type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const vector<int > &			fmt_nlist_a,
			 const vector<int > &			fmt_nlist_r,
			 const vector<int > &			sec_a,
			 const vector<int > &			sec_r,
			 const int				axis0_type,
			 const int				axis0_idx,
			 const int				axis1_type,
			 const int				axis1_idx);

inline
void compute_descriptor_se_a (vector<double > &			descrpt_a,
			       vector<double > &			descrpt_a_deriv,
			       vector<double > &			rij_a,
			       const vector<double > &			posi,
			       const int &				ntypes,
			       const vector<int > &			type,
			       const SimulationRegion<double> &		region,
			       const bool &				b_pbc,
			       const int &				i_idx,
			       const vector<int > &			fmt_nlist_a,
			       const vector<int > &			sec_a, 
			       const double &				rmin,
			       const double &				rmax);

inline
void compute_descriptor_se_r (vector<double > &			descrpt_r,
			      vector<double > &			descrpt_r_deriv,
			      vector<double > &			rij_r,
			      const vector<double > &		posi,
			      const int &			ntypes,
			      const vector<int > &		type,
			      const SimulationRegion<double> &	region,
			      const bool &			b_pbc,
			      const int &			i_idx,
			      const vector<int > &		fmt_nlist_r,
			      const vector<int > &		sec_r,
			      const double &			rmin, 
			      const double &			rmax);


struct NeighborInfo 
{
  int type;
  double dist;
  int index;
  NeighborInfo () 
      : type (0), dist(0), index(0) 
      {
      }
  NeighborInfo (int tt, double dd, int ii) 
      : type (tt), dist(dd), index(ii) 
      {
      }
  bool operator < (const NeighborInfo & b) const 
      {
	return (type < b.type || 
		(type == b.type && 
		 (dist < b.dist || 
		  (dist == b.dist && index < b.index) ) ) );
      }
};

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

  double nr1 = sqrt(MathUtilities::dot (r1, r1));
  double nr12 = nr1  * nr1;
  double nr13 = nr1  * nr12;
  double nr14 = nr12 * nr12;
  double r1dr2 = MathUtilities::dot (r1, r2);

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
  double ntmpy = sqrt(MathUtilities::dot(tmpy, tmpy));
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
    MathUtilities::cprod (dRdT0 + ii*3, yy, dRdT2 + ii*3);
    MathUtilities::cprod (xx, dRdT1 + ii*3, res);
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

  double nr1 = sqrt(MathUtilities::dot (r1, r1));
  double nr12 = nr1  * nr1;
  double nr13 = nr1  * nr12;
  double nr14 = nr12 * nr12;
  double r1dr2 = MathUtilities::dot (r1, r2);

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
  double ntmpy = sqrt(MathUtilities::dot(tmpy, tmpy));
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
    MathUtilities::cprod (dRdT0 + ii*3, yy, dRdT2 + ii*3);
    MathUtilities::cprod (xx, dRdT1 + ii*3, res);
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

  double nr1 = sqrt(MathUtilities::dot (r1, r1));
  double nr12 = nr1  * nr1;
  double r1dr2 = MathUtilities::dot (r1, r2);

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
  double ntmpy = sqrt(MathUtilities::dot(tmpy, tmpy));
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
    MathUtilities::cprod (dRdT0 + ii*3, yy, dRdT2 + ii*3);
    MathUtilities::cprod (xx, dRdT1 + ii*3, res);
    for (int dd = 0; dd < 3; ++dd) dRdT2[ii*3+dd] += res[dd];
  }
}

int format_nlist_fill_a (vector<int > &				fmt_nei_idx_a,
			 vector<int > &				fmt_nei_idx_r,
			 const vector<double > &		posi,
			 const int &				ntypes,
			 const vector<int > &			type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const vector<int > &			nei_idx_a, 
			 const vector<int > &			nei_idx_r, 
			 const double &				rcut,
			 const vector<int > &			sec_a, 
			 const vector<int > &			sec_r)
{
#ifdef DEBUG
  assert (sec_a.size() == ntypes + 1);
  assert (sec_r.size() == ntypes + 1);
#endif
  
  fmt_nei_idx_a.resize (sec_a.back());
  fmt_nei_idx_r.resize (sec_r.back());
  fill (fmt_nei_idx_a.begin(), fmt_nei_idx_a.end(), -1);
  fill (fmt_nei_idx_r.begin(), fmt_nei_idx_r.end(), -1);  
  
  // gether all neighbors
  vector<int > nei_idx (nei_idx_a);
  nei_idx.insert (nei_idx.end(), nei_idx_r.begin(), nei_idx_r.end());
  assert (nei_idx.size() == nei_idx_a.size() + nei_idx_r.size());
  // allocate the information for all neighbors
  vector<NeighborInfo > sel_nei ;
  sel_nei.reserve (nei_idx_a.size() + nei_idx_r.size());
  for (unsigned kk = 0; kk < nei_idx.size(); ++kk){
    double diff[3];
    const int & j_idx = nei_idx[kk];
    if (b_pbc){
      region.diffNearestNeighbor (posi[j_idx*3+0], posi[j_idx*3+1], posi[j_idx*3+2], 
				  posi[i_idx*3+0], posi[i_idx*3+1], posi[i_idx*3+2], 
				  diff[0], diff[1], diff[2]);
    }
    else {
      for (int dd = 0; dd < 3; ++dd) diff[dd] = posi[j_idx*3+dd] - posi[i_idx*3+dd];
    }
    double rr = sqrt(MathUtilities::dot<double> (diff, diff));    
    if (rr <= rcut) {
      sel_nei.push_back(NeighborInfo (type[j_idx], rr, j_idx));
    }
  }
  sort (sel_nei.begin(), sel_nei.end());  
  
  vector<int > nei_iter = sec_a;
  int overflowed = -1;
  for (unsigned kk = 0; kk < sel_nei.size(); ++kk){
    const int & nei_type = sel_nei[kk].type;
    if (nei_iter[nei_type] >= sec_a[nei_type+1]) {
      int r_idx_iter = (nei_iter[nei_type] ++) - sec_a[nei_type+1] + sec_r[nei_type];
      if (r_idx_iter >= sec_r[nei_type+1]) {
	// return nei_type;
	overflowed = nei_type;
      }
      else {
	fmt_nei_idx_r[r_idx_iter] = sel_nei[kk].index;
      }
    }
    else {
      fmt_nei_idx_a[nei_iter[nei_type] ++] = sel_nei[kk].index;
    }
  }
  
  return overflowed;
}


// output deriv size: n_sel_a_nei x 4 x 12				    + n_sel_r_nei x 12
//		      (1./rr, cos_theta, cos_phi, sin_phi)  x 4 x (x, y, z) + (1./rr) x 4 x (x, y, z)
void compute_descriptor (vector<double > &			descrpt_a,
			 vector<double > &			descrpt_a_deriv,
			 vector<double > &			descrpt_r,
			 vector<double > &			descrpt_r_deriv,
			 vector<double > &			rij_a,
			 vector<double > &			rij_r,
			 vector<double > &			rot_mat,
			 const vector<double > &		posi,
			 const int &				ntypes,
			 const vector<int > &			type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const vector<int > &			fmt_nlist_a,
			 const vector<int > &			fmt_nlist_r,
			 const vector<int > &			sec_a,
			 const vector<int > &			sec_r,
			 const int				axis0_type,
			 const int				axis0_idx,
			 const int				axis1_type,
			 const int				axis1_idx)
{  
  // compute the diff of the neighbors
  vector<vector<double > > sel_a_diff (sec_a.back());
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

  vector<vector<double > > sel_r_diff (sec_r.back());
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
  // 	  cout << "\t " << sqrt(MathUtilities::dot (&sel_a_diff[jj][0], &sel_a_diff[jj][0]));
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
  // 	  cout << "\t " << sqrt(MathUtilities::dot (&sel_r_diff[jj][0], &sel_r_diff[jj][0]));
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
  double norm_xx = sqrt(MathUtilities::dot (xx, xx));
  for (unsigned dd = 0; dd < 3; ++dd) xx[dd] /= norm_xx;
  double dxy = MathUtilities::dot (xx, yy);
  for (unsigned dd = 0; dd < 3; ++dd) yy[dd] -= dxy * xx[dd];
  double norm_yy = sqrt(MathUtilities::dot (yy, yy));
  for (unsigned dd = 0; dd < 3; ++dd) yy[dd] /= norm_yy;
  MathUtilities::cprod (xx, yy, zz);  
  rot_mat.resize (9);
  for (int dd = 0; dd < 9; ++dd) rot_mat[dd] = rot[dd];

  // 1./rr, cos(theta), cos(phi), sin(phi)
  descrpt_a.resize (sec_a.back() * 4);
  fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
  for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii){
    for (int jj = sec_a[ii]; jj < sec_a[ii+1]; ++jj){
      if (fmt_nlist_a[jj] < 0) break;
      double rdiff[3] ;
      MathUtilities::dot (rdiff, rot, &sel_a_diff[jj][0]);
      double rr2 = MathUtilities::dot(rdiff, rdiff);
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
      double rr = sqrt (MathUtilities::dot(rdiff, rdiff));
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
      MathUtilities::dot (tr, rot, rr);
      double nr2 = MathUtilities::dot(tr, tr);
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
      double nr = sqrt(MathUtilities::dot(rr, rr));
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


void compute_descriptor (vector<double > &			descrpt_a,
			 vector<double > &			descrpt_r,
			 vector<double > &			rot_mat,
			 const vector<double > &		posi,
			 const int &				ntypes,
			 const vector<int > &			type,
			 const SimulationRegion<double> &	region,
			 const bool &				b_pbc,
			 const int &				i_idx,
			 const vector<int > &			fmt_nlist_a,
			 const vector<int > &			fmt_nlist_r,
			 const vector<int > &			sec_a,
			 const vector<int > &			sec_r,
			 const int				axis0_type,
			 const int				axis0_idx,
			 const int				axis1_type,
			 const int				axis1_idx)
{  
  // compute the diff of the neighbors
  vector<vector<double > > sel_a_diff (sec_a.back());
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
  vector<vector<double > > sel_r_diff (sec_r.back());
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
  double norm_xx = sqrt(MathUtilities::dot (xx, xx));
  for (unsigned dd = 0; dd < 3; ++dd) xx[dd] /= norm_xx;
  double dxy = MathUtilities::dot (xx, yy);
  for (unsigned dd = 0; dd < 3; ++dd) yy[dd] -= dxy * xx[dd];
  double norm_yy = sqrt(MathUtilities::dot (yy, yy));
  for (unsigned dd = 0; dd < 3; ++dd) yy[dd] /= norm_yy;
  MathUtilities::cprod (xx, yy, zz);  
  rot_mat.resize (9);
  for (int dd = 0; dd < 9; ++dd) rot_mat[dd] = rot[dd];

  // 1./rr, cos(theta), cos(phi), sin(phi)
  descrpt_a.resize (sec_a.back() * 4);
  fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
  for (int ii = 0; ii < int(sec_a.size()) - 1; ++ii){
    for (int jj = sec_a[ii]; jj < sec_a[ii+1]; ++jj){
      if (fmt_nlist_a[jj] < 0) break;
      double rdiff[3] ;
      MathUtilities::dot (rdiff, rot, &sel_a_diff[jj][0]);
      double rr2 = MathUtilities::dot(rdiff, rdiff);
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
      MathUtilities::dot (rdiff, rot, &sel_r_diff[jj][0]);
      double rr = sqrt (MathUtilities::dot(rdiff, rdiff));
      descrpt_r[jj] = 1./rr;
    }
  }  
}

inline double
cos_switch (const double & xx, 
	    const double & rmin, 
	    const double & rmax) 
{
  if (xx < rmin) {
    return 1.;
  }
  else if (xx < rmax) {
    const double value = (xx - rmin) / (rmax - rmin) * M_PI;
    return 0.5 * (cos(value) + 1);  
  }
  else {
    return 0.;
  }
}

inline void
cos_switch (double & vv,
	    double & dd,
	    const double & xx, 
	    const double & rmin, 
	    const double & rmax) 
{
  if (xx < rmin) {
    dd = 0;
    vv = 1;
  }
  else if (xx < rmax) {
    double value = (xx - rmin) / (rmax - rmin) * M_PI;
    dd = -0.5 * sin(value) * M_PI / (rmax - rmin);
    vv = 0.5 * (cos(value) + 1);    
  }
  else {
    dd = 0;
    vv = 0;
  }
}

inline void
spline3_switch (double & vv,
		double & dd,
		const double & xx, 
		const double & rmin, 
		const double & rmax) 
{
  if (xx < rmin) {
    dd = 0;
    vv = 1;
  }
  else if (xx < rmax) {
    double uu = (xx - rmin) / (rmax - rmin) ;
    double du = 1. / (rmax - rmin) ;
    // s(u) = (1+2u)(1-u)^2
    // s'(u) = 2(2u+1)(u-1) + 2(u-1)^2
    vv = (1 + 2*uu) * (1-uu) * (1-uu);
    dd = (2 * (2*uu + 1) * (uu-1) + 2 * (uu-1) * (uu-1) ) * du;
  }
  else {
    dd = 0;
    vv = 0;
  }
}

template <typename TYPE>
inline void
spline5_switch (TYPE & vv,
		TYPE & dd,
		const TYPE & xx, 
		const TYPE & rmin, 
		const TYPE & rmax) 
{
  if (xx < rmin) {
    dd = 0;
    vv = 1;
  }
  else if (xx < rmax) {
    double uu = (xx - rmin) / (rmax - rmin) ;
    double du = 1. / (rmax - rmin) ;
    vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1;
    dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
  }
  else {
    dd = 0;
    vv = 0;
  }
}

// output deriv size: n_sel_a_nei x 4 x 12				    
//		      (1./rr, cos_theta, cos_phi, sin_phi)  x 4 x (x, y, z) 
void compute_descriptor_se_a (vector<double > &			descrpt_a,
			       vector<double > &			descrpt_a_deriv,
			       vector<double > &			rij_a,
			       const vector<double > &			posi,
			       const int &				ntypes,
			       const vector<int > &			type,
			       const SimulationRegion<double> &		region,
			       const bool &				b_pbc,
			       const int &				i_idx,
			       const vector<int > &			fmt_nlist_a,
			       const vector<int > &			sec_a, 
			       const double &				rmin, 
			       const double &				rmax)
{  
  // compute the diff of the neighbors
  vector<vector<double > > sel_a_diff (sec_a.back());
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
      double nr2 = MathUtilities::dot(rr, rr);
      double inr = 1./sqrt(nr2);
      double nr = nr2 * inr;
      double inr2 = inr * inr;
      double inr4 = inr2 * inr2;
      double inr3 = inr4 * nr;
      double sw, dsw;
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


void compute_descriptor_se_r (vector<double > &			descrpt,
			      vector<double > &			descrpt_deriv,
			      vector<double > &			rij,
			      const vector<double > &		posi,
			      const int &			ntypes,
			      const vector<int > &		type,
			      const SimulationRegion<double> &	region,
			      const bool &			b_pbc,
			      const int &			i_idx,
			      const vector<int > &		fmt_nlist,
			      const vector<int > &		sec,
			      const double &			rmin, 
			      const double &			rmax)
{  
  // compute the diff of the neighbors
  vector<vector<double > > sel_diff (sec.back());
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
      double nr2 = MathUtilities::dot(rr, rr);
      double inr = 1./sqrt(nr2);
      double nr = nr2 * inr;
      double inr2 = inr * inr;
      double inr4 = inr2 * inr2;
      double inr3 = inr4 * nr;
      double sw, dsw;
      spline5_switch(sw, dsw, nr, rmin, rmax);
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



