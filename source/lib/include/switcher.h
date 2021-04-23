#pragma once

namespace deepmd{

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

template <typename FPTYPE>
inline void 
spline5_switch (
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

}
