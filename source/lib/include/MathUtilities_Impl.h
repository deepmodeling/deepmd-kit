#pragma once

#include <vector>
#include <cmath>

#include <math.h>

template <typename TYPE>
inline TYPE
MathUtilities::
max (const TYPE & v0, const TYPE & v1)
{
  return (v0 > v1) ? v0 : v1;
}

template <typename TYPE>
inline TYPE
MathUtilities::
min (const TYPE & v0, const TYPE & v1)
{
  return (v0 < v1) ? v0 : v1;
}

template <typename TYPE>
inline
void 
MathUtilities::
dot (TYPE * vec_o, const TYPE * tensor, const TYPE * vec_i)
{
  vec_o[0] = dot(tensor+0, vec_i);
  vec_o[1] = dot(tensor+3, vec_i);
  vec_o[2] = dot(tensor+6, vec_i);
}

template <typename TYPE>
inline TYPE
MathUtilities::
dot (const TYPE& x0, const TYPE& y0, const TYPE& z0,
     const TYPE& x1, const TYPE& y1, const TYPE& z1)
{
  return x0 * x1 + y0 * y1 + z0 * z1;
}

template <typename TYPE>
inline TYPE
MathUtilities::
dot (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2];
}

template <typename TYPE>
inline void
MathUtilities::
cprod (const TYPE& x0, const TYPE& y0, const TYPE& z0,
       const TYPE& x1, const TYPE& y1, const TYPE& z1,
       TYPE& x2, TYPE& y2, TYPE& z2)
{
  x2 = y0 * z1 - z0 * y1;
  y2 = z0 * x1 - x0 * z1;
  z2 = x0 * y1 - y0 * x1;
}

template <typename TYPE>
inline void
MathUtilities::
cprod (const TYPE * r0,
       const TYPE * r1,
       TYPE* r2)
{
  r2[0] = r0[1] * r1[2] - r0[2] * r1[1];
  r2[1] = r0[2] * r1[0] - r0[0] * r1[2];
  r2[2] = r0[0] * r1[1] - r0[1] * r1[0];
}


template <typename TYPE>
inline TYPE
MathUtilities::
cos (const TYPE& x0, const TYPE& y0, const TYPE& z0,
     const TYPE& x1, const TYPE& y1, const TYPE& z1)
{
  double dblx0 = (double) (x0);
  double dblx1 = (double) (x1);
  double dbly0 = (double) (y0);
  double dbly1 = (double) (y1);
  double dblz0 = (double) (z0);
  double dblz1 = (double) (z1);
  
  double ip  = dot<double> (dblx0, dbly0, dblz0, dblx1, dbly1, dblz1);
  double ip0 = dot<double> (dblx0, dbly0, dblz0, dblx0, dbly0, dblz0);
  double ip1 = dot<double> (dblx1, dbly1, dblz1, dblx1, dbly1, dblz1);  
  double ip01 = ip0 * ip1;
  
  double cosval;
  if (ip01 > 0) {
    cosval = ip * invsqrt(ip01);
  }
  else {
    cosval = 1.0;
  }
  if (cosval > 1.0) {
    return 1.0;
  }
  if (cosval < -1.0) {
    return -1.0;
  }
  return cosval;
}

template <typename TYPE>
inline TYPE
MathUtilities::
angle (const TYPE& x0_, const TYPE& y0_, const TYPE& z0_,
       const TYPE& x1_, const TYPE& y1_, const TYPE& z1_)
{
  double x0 = (double) (x0_);
  double x1 = (double) (x1_);
  double y0 = (double) (y0_);
  double y1 = (double) (y1_);
  double z0 = (double) (z0_);
  double z1 = (double) (z1_);

  double x2, y2, z2;
  cprod<double> (x0, y0, z0, x1, y1, z1, x2, y2, z2);

  double length = sqrt (dot<double>(x2, y2, z2, x2, y2, z2));

  double s = dot<double> (x0, y0, z0, x1, y1, z1);

  return atan2 (length, s);
}



template <typename TYPE>
inline TYPE
MathUtilities::
det1d (const TYPE * tensor)
{
  return (tensor[0]);
}

template <typename TYPE>
inline TYPE
MathUtilities::
det2d (const TYPE * tensor)
{
  return ((tensor[0*2+0]*tensor[1*2+1] - tensor[1*2+0]*tensor[0*2+1]));
}

template <typename TYPE>
inline TYPE
MathUtilities::
det3d (const TYPE * tensor)
{
  return (tensor[0*3+0] * (tensor[1*3+1]*tensor[2*3+2] - tensor[2*3+1]*tensor[1*3+2]) - 
	  tensor[0*3+1] * (tensor[1*3+0]*tensor[2*3+2] - tensor[2*3+0]*tensor[1*3+2]) +
	  tensor[0*3+2] * (tensor[1*3+0]*tensor[2*3+1] - tensor[2*3+0]*tensor[1*3+1]) );
}

template <typename TYPE>
inline TYPE
MathUtilities::
det4d (const TYPE * mat)
{
  return
      ( + mat[0*4+0] *   
	( mat[1*4+1] * (mat[2*4+2]*mat[3*4+3]-mat[3*4+2]*mat[2*4+3] ) - 
	  mat[1*4+2] * (mat[2*4+1]*mat[3*4+3]-mat[3*4+1]*mat[2*4+3] ) + 
	  mat[1*4+3] * (mat[2*4+1]*mat[3*4+2]-mat[3*4+1]*mat[2*4+2] ) ) 
	- mat[0*4+1] * 
	( mat[1*4+0] * (mat[2*4+2]*mat[3*4+3]-mat[3*4+2]*mat[2*4+3] ) - 
	  mat[1*4+2] * (mat[2*4+0]*mat[3*4+3]-mat[3*4+0]*mat[2*4+3] ) + 
	  mat[1*4+3] * (mat[2*4+0]*mat[3*4+2]-mat[3*4+0]*mat[2*4+2] ) ) 
	+ mat[0*4+2] * 
	( mat[1*4+0] * (mat[2*4+1]*mat[3*4+3]-mat[3*4+1]*mat[2*4+3] ) - 
	  mat[1*4+1] * (mat[2*4+0]*mat[3*4+3]-mat[3*4+0]*mat[2*4+3] ) + 
	  mat[1*4+3] * (mat[2*4+0]*mat[3*4+1]-mat[3*4+0]*mat[2*4+1] ) ) 
	- mat[0*4+3] * 
	( mat[1*4+0] * (mat[2*4+1]*mat[3*4+2]-mat[3*4+1]*mat[2*4+2] ) - 
	  mat[1*4+1] * (mat[2*4+0]*mat[3*4+2]-mat[3*4+0]*mat[2*4+2] ) + 
	  mat[1*4+2] * (mat[2*4+0]*mat[3*4+1]-mat[3*4+0]*mat[2*4+1] ) )
	  );
}

  namespace MathUtilities{
    template<typename TYPE>
    struct det <TYPE, 1>
    {
      inline TYPE operator () (const TYPE * tensor) const {return det1d (tensor);}
    };

    template<typename TYPE>
    struct det <TYPE, 2>
    {
      inline TYPE operator () (const TYPE * tensor) const {return det2d (tensor);}
    };

    template<typename TYPE>
    struct det <TYPE, 3>
    {
      inline TYPE operator () (const TYPE * tensor) const {return det3d (tensor);}
    };

    template<typename TYPE>
    struct det <TYPE, 4>
    {
      inline TYPE operator () (const TYPE * tensor) const {return det4d (tensor);}
    };
  }

template<typename TYPE, typename VALUETYPE>
inline int
MathUtilities::
searchVec (const vector<TYPE> & vec,
	   const int sta_,
	   const int end_,
	   const VALUETYPE & val)
{
  int sta (sta_);
  int end (end_);
  if (sta == end) return -1;
  while (end - sta > 1){
    int mid = (sta + end) >> 1;
    if ((vec[mid] < val)) sta = mid;
    else end = mid;
  }
  return sta;
}

template<typename TYPE, typename VALUETYPE>
inline int
MathUtilities::
lowerBound (const vector<TYPE> & vec,
	    const int sta_,
	    const int end_,
	    const VALUETYPE & val)
{
  int sta (sta_);
  int iter, step;
  int count = end_ - sta;
  while (count > 0){
    iter = sta;
    step = count >> 1;
    iter += step;
    if (vec[iter] < val){
      sta = ++iter;
      count -= step + 1;
    }
    else count = step;
  }
  return sta;
}

template<typename TYPE, typename VALUETYPE>
inline int
MathUtilities::
upperBound (const vector<TYPE> & vec,
	    const int sta_,
	    const int end_,
	    const VALUETYPE & val)
{
  int sta (sta_);
  int iter, step;
  int count = end_ - sta;
  while (count > 0){
    iter = sta;
    step = count >> 1;
    iter += step;
    if ( ! (val < vec[iter]) ){
      sta = ++iter;
      count -= step + 1;
    }
    else count = step;
  }
  return sta;
}

template<typename TYPE, typename VALUETYPE, typename COMPARE>
inline int
MathUtilities::
upperBound (const vector<TYPE> & vec,
	    const int sta_,
	    const int end_,
	    const VALUETYPE & val,
	    const COMPARE & less)
{
  int sta (sta_);
  int iter, step;
  int count = end_ - sta;
  while (count > 0){
    iter = sta;
    step = count >> 1;
    iter += step;
    if ( ! less (val , vec[iter]) ){
      sta = ++iter;
      count -= step + 1;
    }
    else count = step;
  }
  return sta;
}

  namespace MathUtilities{
    template <>
    inline double
    msp_sqrt<double> (const double x) 
    {
      return ::sqrt (x);
    }

    template <>
    inline float
    msp_sqrt<float> (const float x) 
    {
      return ::sqrtf (x);
    }

    template <>
    inline double
    invsqrt<double> (const double x) 
    {
      return 1./sqrt (x);
    }

    template <>
    inline float
    invsqrt<float> (const float x) 
    {
      return 1./sqrtf (x);
    }
  }

// template<typename TYPE, typename VALUETYPE, typename COMPARE>
// int
// searchVec (const vector<TYPE> & vec,
// 	   const int sta_,
// 	   const int end_,
// 	   const VALUETYPE & val,
// 	   const COMPARE & lessOrEq)
// {
//   int sta (sta_);
//   int end (end_);
//   if (sta == end) return -1;
//   while (end - sta > 1){
//     int mid = (sta + end) >> 1;
//     if (lessOrEq (vec[mid], val)) sta = mid;
//     else end = mid;
//   }
//   return sta;
// }

