#ifndef __SimulationRegion_Impl_h_wanghan__
#define __SimulationRegion_Impl_h_wanghan__

// #include <iomanip>
#include <iostream>
#include <limits>
#include <typeinfo>
#include <stdexcept>

// using namespace std;

template<typename VALUETYPE>
SimulationRegion<VALUETYPE>::
~SimulationRegion ()
{
}

template<typename VALUETYPE>
SimulationRegion<VALUETYPE>::
SimulationRegion ()
{
  is_periodic[0] = is_periodic[1] = is_periodic[2] = true;
  std::fill (boxt,		boxt    + SPACENDIM*SPACENDIM, 0);
  std::fill (boxt_bk,		boxt_bk + SPACENDIM*SPACENDIM, 0);
  std::fill (origin,		origin  + SPACENDIM, 0);
}

template <typename VALUETYPE>
void
SimulationRegion<VALUETYPE>::
defaultInitBox (double * my_boxv, double * my_orig, bool * period) const
{
  // by default is a 1,1,1 logical box
  for (int ii = 0; ii < SPACENDIM; ++ii){
    for (int jj = 0; jj < SPACENDIM; ++jj){
      my_boxv[ii*3+jj] = 0.;
    }
  }
  // origin is at 0,0,0
  for (int jj = 0; jj < SPACENDIM; ++jj){
    my_boxv[jj*3+jj] = 1.;
  }
  for (int ii = 0; ii < SPACENDIM; ++ii) {
    my_orig[ii] = 0.;
    period[ii] = true;
  }
}


template<typename VALUETYPE>
void
SimulationRegion<VALUETYPE>::
backup ()
{
  for (int ii = 0; ii < SPACENDIM * SPACENDIM; ++ii){
    boxt_bk[ii] = boxt[ii];
  }  
}

template<typename VALUETYPE>
void
SimulationRegion<VALUETYPE>::
recover ()
{
  reinitBox (boxt_bk);
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
reinitBox (const double * boxv_)
{
  for (int ii = 0; ii < SPACENDIM * SPACENDIM; ++ii){
    boxt[ii] = boxv_[ii];
  }
  computeVolume();
  computeRecBox();
  computeShiftVec();
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
affineTransform (const double * affine_map)
{
  tensorDotVector (boxt+SPACENDIM*0, affine_map, boxt+SPACENDIM*0);
  tensorDotVector (boxt+SPACENDIM*1, affine_map, boxt+SPACENDIM*1);
  tensorDotVector (boxt+SPACENDIM*2, affine_map, boxt+SPACENDIM*2);
  computeVolume();
  computeRecBox();
  computeShiftVec();
}


template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
reinitOrigin (const double * orig)
{
  for (int ii = 0; ii < SPACENDIM ; ++ii){
    origin[ii] = orig[ii];
  }  
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
reinitOrigin (const std::vector<double>& orig)
{
  for (int ii = 0; ii < SPACENDIM ; ++ii){
    origin[ii] = orig[ii];
  }  
}

template <typename VALUETYPE>
void
SimulationRegion<VALUETYPE>::
computeShiftVec ()
{
  int tmp_idx[3];
  int & ii (tmp_idx[0]);
  int & jj (tmp_idx[1]);
  int & kk (tmp_idx[2]);
  for (ii = -DBOX_XX; ii <= DBOX_XX; ++ii){
    for (jj = -DBOX_YY; jj <= DBOX_YY; ++jj){
      for (kk = -DBOX_ZZ; kk <= DBOX_ZZ; ++kk){
	double *posi = getShiftVec(getShiftIndex(tmp_idx));
	double *inter_posi = getInterShiftVec(getShiftIndex(tmp_idx));
	inter_posi[0] = ii;
	inter_posi[1] = jj;
	inter_posi[2] = kk;
	// inter2Phys (posi, inter_posi);
	tensorTransDotVector (posi, boxt, inter_posi);
      }
    }
  }
}

template <typename VALUETYPE>
inline double *
SimulationRegion<VALUETYPE>::
getShiftVec (const int index)
{
  return shift_vec + SPACENDIM*index;
}

template <typename VALUETYPE>
inline const double *
SimulationRegion<VALUETYPE>::
getShiftVec (const int index) const
{
  return shift_vec + SPACENDIM*index;
}

template <typename VALUETYPE>
inline double *
SimulationRegion<VALUETYPE>::
getInterShiftVec (const int index)
{
  return inter_shift_vec + SPACENDIM*index;
}

template <typename VALUETYPE>
inline const double *
SimulationRegion<VALUETYPE>::
getInterShiftVec (const int index) const
{
  return inter_shift_vec + SPACENDIM*index;
}

template <typename VALUETYPE>
inline int
SimulationRegion<VALUETYPE>::
getShiftIndex (const int * idx) const
{
  return index3to1(idx[0], idx[1], idx[2]);
}

template <typename VALUETYPE>
inline int
SimulationRegion<VALUETYPE>::
getNullShiftIndex () const
{
  return index3to1(0,0,0);
}

template <typename VALUETYPE>
inline int
SimulationRegion<VALUETYPE>::
compactIndex (const int * idx) 
{
  return index3to1(idx[0], idx[1], idx[2]);
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
shiftCoord (const int * idx,
	    VALUETYPE &x,
	    VALUETYPE &y,
	    VALUETYPE &z) const
{
  const double * shift = getShiftVec(getShiftIndex(idx));
  x += shift[0];
  y += shift[1];
  z += shift[2];
}

// template<typename VALUETYPE>
// inline void
// SimulationRegion<VALUETYPE>::
// diffNearestNeighbor (const VALUETYPE x0,
// 		     const VALUETYPE y0,
// 		     const VALUETYPE z0,
// 		     const VALUETYPE x1,
// 		     const VALUETYPE y1,
// 		     const VALUETYPE z1,
// 		     VALUETYPE & dx,
// 		     VALUETYPE & dy,
// 		     VALUETYPE & dz) const
// {
//   dx = x0 - x1;
//   dy = y0 - y1;
//   dz = z0 - z1;
// }

// template<typename VALUETYPE>
// inline void
// SimulationRegion<VALUETYPE>::
// diffNearestNeighbor (const VALUETYPE x0,
// 		     const VALUETYPE y0,
// 		     const VALUETYPE z0,
// 		     const VALUETYPE x1,
// 		     const VALUETYPE y1,
// 		     const VALUETYPE z1,
// 		     VALUETYPE & dx,
// 		     VALUETYPE & dy,
// 		     VALUETYPE & dz,
// 		     int & shift_x,
// 		     int & shift_y,
// 		     int & shift_z) const 
// {
//   shift_x = shift_y = shift_z = 0;
//   diffNearestNeighbor (x0, y0, z0, x1, y1, z1, dx, dy, dz);
// }

template<typename VALUETYPE> 
inline void
SimulationRegion<VALUETYPE>::
apply_periodic (int dim, double * dd) const
{
  if (!is_periodic[dim]) return;
  if      (dd[dim] >= static_cast<double>(0.5)) dd[dim] -= static_cast<double>(1.);
  else if (dd[dim] < -static_cast<double>(0.5)) dd[dim] += static_cast<double>(1.);
}

template<typename VALUETYPE> 
inline void
SimulationRegion<VALUETYPE>::
apply_periodic (int dim,
		double * dd,
		int & shift) const
{
  shift = 0;
  if (!is_periodic[dim]) return;
  if      (dd[dim] >= static_cast<double>(0.5)) {
    dd[dim] -= static_cast<double>(1.);
    shift = -1;
  }
  else if (dd[dim] < -static_cast<double>(0.5)) {
    dd[dim] += static_cast<double>(1.);
    shift = 1;
  }
}

template<typename VALUETYPE> 
inline void
SimulationRegion<VALUETYPE>::
diffNearestNeighbor (const VALUETYPE * r0,
		     const VALUETYPE * r1,
		     VALUETYPE * phys) const
{
  double inter[3];
  for (int dd = 0; dd < 3; ++dd) phys[dd] = r0[dd] - r1[dd];
  SimulationRegion<VALUETYPE>::phys2Inter (inter, phys);
  for (int dd = 0; dd < 3; ++dd) apply_periodic (dd, inter);
  SimulationRegion<VALUETYPE>::inter2Phys (phys, inter);
}

template<typename VALUETYPE> 
inline void
SimulationRegion<VALUETYPE>::
diffNearestNeighbor (const VALUETYPE x0,
		     const VALUETYPE y0,
		     const VALUETYPE z0,
		     const VALUETYPE x1,
		     const VALUETYPE y1,
		     const VALUETYPE z1,
		     VALUETYPE & dx,
		     VALUETYPE & dy,
		     VALUETYPE & dz) const
{
  // diffNearestNeighbor (0, x0, x1, dx);
  // diffNearestNeighbor (1, y0, y1, dy);
  // diffNearestNeighbor (2, z0, z1, dz);
  VALUETYPE phys [3];
  double inter[3];
  phys[0] = x0 - x1;
  phys[1] = y0 - y1;
  phys[2] = z0 - z1;  
  SimulationRegion<VALUETYPE>::phys2Inter (inter, phys);
  apply_periodic (0, inter);
  apply_periodic (1, inter);
  apply_periodic (2, inter);
  SimulationRegion<VALUETYPE>::inter2Phys (phys, inter);
  dx = phys[0];
  dy = phys[1];
  dz = phys[2];
}

template<typename VALUETYPE> 
inline void
SimulationRegion<VALUETYPE>::
diffNearestNeighbor (const VALUETYPE x0,
		     const VALUETYPE y0,
		     const VALUETYPE z0,
		     const VALUETYPE x1,
		     const VALUETYPE y1,
		     const VALUETYPE z1,
		     VALUETYPE & dx,
		     VALUETYPE & dy,
		     VALUETYPE & dz,
		     int & shift_x,
		     int & shift_y,
		     int & shift_z) const 
{
  // diffNearestNeighbor (0, x0, x1, dx, shift_x);
  // diffNearestNeighbor (1, y0, y1, dy, shift_y);
  // diffNearestNeighbor (2, z0, z1, dz, shift_z);
  VALUETYPE phys [3];
  double inter[3];
  phys[0] = x0 - x1;
  phys[1] = y0 - y1;
  phys[2] = z0 - z1;  
  SimulationRegion<VALUETYPE>::phys2Inter (inter, phys);
  apply_periodic (0, inter, shift_x);
  apply_periodic (1, inter, shift_y);
  apply_periodic (2, inter, shift_z);
  SimulationRegion<VALUETYPE>::inter2Phys (phys, inter);
  dx = phys[0];
  dy = phys[1];
  dz = phys[2];  
}

template<typename VALUETYPE> 
inline void
SimulationRegion<VALUETYPE>::
diffNearestNeighbor (const VALUETYPE x0,
		     const VALUETYPE y0,
		     const VALUETYPE z0,
		     const VALUETYPE x1,
		     const VALUETYPE y1,
		     const VALUETYPE z1,
		     VALUETYPE & dx,
		     VALUETYPE & dy,
		     VALUETYPE & dz,
		     VALUETYPE & shift_x,
		     VALUETYPE & shift_y,
		     VALUETYPE & shift_z) const 
{
  // diffNearestNeighbor (0, x0, x1, dx, shift_x);
  // diffNearestNeighbor (1, y0, y1, dy, shift_y);
  // diffNearestNeighbor (2, z0, z1, dz, shift_z);
  VALUETYPE phys [3];
  double inter[3];
  phys[0] = x0 - x1;
  phys[1] = y0 - y1;
  phys[2] = z0 - z1;  
  SimulationRegion<VALUETYPE>::phys2Inter (inter, phys);
  int i_shift_x, i_shift_y, i_shift_z;
  apply_periodic (0, inter, i_shift_x);
  apply_periodic (1, inter, i_shift_y);
  apply_periodic (2, inter, i_shift_z);
  SimulationRegion<VALUETYPE>::inter2Phys (phys, inter);
  dx = phys[0];
  dy = phys[1];
  dz = phys[2];
  const double * tmp_shift (getShiftVec (index3to1 (i_shift_x, i_shift_y, i_shift_z) ) );
  shift_x = tmp_shift[0];
  shift_y = tmp_shift[1];
  shift_z = tmp_shift[2];
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
phys2Inter (double * i_v, const VALUETYPE * p_v_) const
{
  double p_v[3];
  for (int dd = 0; dd < 3; ++dd) p_v[dd] = p_v_[dd];
  tensorDotVector (i_v, rec_boxt, p_v);
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
inter2Phys (VALUETYPE * p_v_, const double * i_v) const
{
  double p_v[3];
  tensorTransDotVector (p_v, boxt, i_v);
  for (int dd = 0; dd < 3; ++dd) p_v_[dd] = p_v[dd];
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
toFaceDistance	(double * dd) const
{
  double tmp[3];
  deepmd::cprod(boxt+3, boxt+6, tmp);
  dd[0] = volume * deepmd::invsqrt(deepmd::dot3(tmp,tmp));
  deepmd::cprod(boxt+6, boxt+0, tmp);
  dd[1] = volume * deepmd::invsqrt(deepmd::dot3(tmp,tmp));
  deepmd::cprod(boxt+0, boxt+3, tmp);
  dd[2] = volume * deepmd::invsqrt(deepmd::dot3(tmp,tmp));
}

// static int tmp_count = 0;

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
copy (double * o_v, const double * i_v) const
{
#ifdef DEBUG_CHECK_ASSERTIONS
  assert (o_v != i_v);
#endif
  o_v[0] = i_v[0];
  o_v[1] = i_v[1];
  o_v[2] = i_v[2];
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
naiveTensorDotVector (double * o_v,
		      const double * i_t,
		      const double * i_v) const
{
  o_v[0] = i_v[0] * i_t[0*3+0] + i_v[1] * i_t[0*3+1] + i_v[2] * i_t[0*3+2];
  o_v[1] = i_v[0] * i_t[1*3+0] + i_v[1] * i_t[1*3+1] + i_v[2] * i_t[1*3+2];
  o_v[2] = i_v[0] * i_t[2*3+0] + i_v[1] * i_t[2*3+1] + i_v[2] * i_t[2*3+2];
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
naiveTensorTransDotVector (double * o_v,
			   const double * i_t,
			   const double * i_v) const
{
  o_v[0] = i_v[0] * i_t[0*3+0] + i_v[1] * i_t[1*3+0] + i_v[2] * i_t[2*3+0];
  o_v[1] = i_v[0] * i_t[0*3+1] + i_v[1] * i_t[1*3+1] + i_v[2] * i_t[2*3+1];
  o_v[2] = i_v[0] * i_t[0*3+2] + i_v[1] * i_t[1*3+2] + i_v[2] * i_t[2*3+2];
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
tensorDotVector (double * o_v,
		 const double * i_t,
		 const double * i_v) const
{
  // the compiler will auto-matically optimize the following code away...
  // const double * tmp_v (i_v);
  // if (o_v == i_v){
  //   double ii_v[3];
  //   copy (ii_v, i_v);
  //   tmp_v = ii_v;
  // }
  naiveTensorDotVector (o_v, i_t, i_v);
}

template <typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
tensorTransDotVector (double * o_v,
		      const double * i_t,
		      const double * i_v) const
{
  naiveTensorTransDotVector (o_v, i_t, i_v);
}

template<typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
computeVolume()
{
  volume =
      boxt[0*3+0] * (boxt[1*3+1]*boxt[2*3+2] - boxt[2*3+1]*boxt[1*3+2]) - 
      boxt[0*3+1] * (boxt[1*3+0]*boxt[2*3+2] - boxt[2*3+0]*boxt[1*3+2]) +
      boxt[0*3+2] * (boxt[1*3+0]*boxt[2*3+1] - boxt[2*3+0]*boxt[1*3+1]);
  volumei = static_cast<double>(1.)/volume;
  if (volume < 0) {
    throw std::runtime_error("Negative volume detected. Please make sure the simulation cell obeys the right-hand rule.");
  }
}

template<typename VALUETYPE>
inline void
SimulationRegion<VALUETYPE>::
computeRecBox	()
{
  // rec_boxt[0*3+0] =( boxt[1*3+1]*boxt[2*3+2] - boxt[2*3+1]*boxt[1*3+2]) * volumei;
  // rec_boxt[1*3+1] =( boxt[0*3+0]*boxt[2*3+2] - boxt[2*3+0]*boxt[0*3+2]) * volumei;
  // rec_boxt[2*3+2] =( boxt[0*3+0]*boxt[1*3+1] - boxt[1*3+0]*boxt[0*3+1]) * volumei;
  // rec_boxt[1*3+0] =(-boxt[1*3+0]*boxt[2*3+2] + boxt[2*3+0]*boxt[1*3+2]) * volumei;
  // rec_boxt[2*3+0] =( boxt[1*3+0]*boxt[2*3+1] - boxt[2*3+0]*boxt[1*3+1]) * volumei;
  // rec_boxt[0*3+1] =(-boxt[0*3+1]*boxt[2*3+2] + boxt[2*3+1]*boxt[0*3+2]) * volumei;
  // rec_boxt[2*3+1] =(-boxt[0*3+0]*boxt[2*3+1] + boxt[2*3+0]*boxt[0*3+1]) * volumei;
  // rec_boxt[0*3+2] =( boxt[0*3+1]*boxt[1*3+2] - boxt[1*3+1]*boxt[0*3+2]) * volumei;
  // rec_boxt[1*3+2] =(-boxt[0*3+0]*boxt[1*3+2] + boxt[1*3+0]*boxt[0*3+2]) * volumei;  

  rec_boxt[0*3+0] =( boxt[1*3+1]*boxt[2*3+2] - boxt[2*3+1]*boxt[1*3+2]) * volumei;
  rec_boxt[1*3+1] =( boxt[0*3+0]*boxt[2*3+2] - boxt[2*3+0]*boxt[0*3+2]) * volumei;
  rec_boxt[2*3+2] =( boxt[0*3+0]*boxt[1*3+1] - boxt[1*3+0]*boxt[0*3+1]) * volumei;
  rec_boxt[0*3+1] =(-boxt[1*3+0]*boxt[2*3+2] + boxt[2*3+0]*boxt[1*3+2]) * volumei;
  rec_boxt[0*3+2] =( boxt[1*3+0]*boxt[2*3+1] - boxt[2*3+0]*boxt[1*3+1]) * volumei;
  rec_boxt[1*3+0] =(-boxt[0*3+1]*boxt[2*3+2] + boxt[2*3+1]*boxt[0*3+2]) * volumei;
  rec_boxt[1*3+2] =(-boxt[0*3+0]*boxt[2*3+1] + boxt[2*3+0]*boxt[0*3+1]) * volumei;
  rec_boxt[2*3+0] =( boxt[0*3+1]*boxt[1*3+2] - boxt[1*3+1]*boxt[0*3+2]) * volumei;
  rec_boxt[2*3+1] =(-boxt[0*3+0]*boxt[1*3+2] + boxt[1*3+0]*boxt[0*3+2]) * volumei;
}




#endif


