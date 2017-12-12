#pragma once

template <typename VALUETYPE>
inline VALUETYPE
dot (const VALUETYPE * r0,
     const VALUETYPE * r1)
{
  return ( r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2] );
}

template <typename TYPE>
inline TYPE
dot (const TYPE& x0, const TYPE& y0, const TYPE& z0,
     const TYPE& x1, const TYPE& y1, const TYPE& z1)
{
  return x0 * x1 + y0 * y1 + z0 * z1;
}

template <typename VALUETYPE>
inline VALUETYPE
cos (const VALUETYPE * r0,
     const VALUETYPE * r1)
{
  double ip  = dot<VALUETYPE> (r0, r1);
  double ip0 = dot<VALUETYPE> (r0, r0);
  double ip1 = dot<VALUETYPE> (r1, r1);
  double ip01 = ip0 * ip1;
  
  double cosval;
  if (ip01 > 0) {
    cosval = ip / sqrt(ip01);
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
    cosval = ip / sqrt(ip01);
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
