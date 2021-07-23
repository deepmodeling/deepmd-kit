#pragma once
#include <cmath>

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class CosSwitch 
{
public:
  CosSwitch (const VALUETYPE & rmin_ = 0, 
	     const VALUETYPE & rmax_ = 0)
      {reinit (rmin_, rmax_); }
  void reinit (const VALUETYPE & rmin_, 
	       const VALUETYPE & rmax_);
public:
  void eval (VALUETYPE & vv,
	     const VALUETYPE xx) const;
private:
  VALUETYPE rmin, rmax;
};


void
CosSwitch::
reinit (const VALUETYPE & rmin_, 
	const VALUETYPE & rmax_)
{
  rmin = rmin_;
  rmax = rmax_;
}
    
void
CosSwitch::
eval (VALUETYPE & vv,
      const VALUETYPE xx) const
{
  VALUETYPE dd;
  if (xx >= 0){
    if (xx < rmin) {
      dd = 0;
      vv = 1;
    }
    else if (xx < rmax){
      VALUETYPE value = (xx - rmin) / (rmax - rmin) * M_PI;
      dd = -0.5 * sin(value) * M_PI / (rmax - rmin);
      vv = 0.5 * (cos(value) + 1);
    }
    else {
      dd = 0;
      vv = 0;
    }
  }
  else {
    if (xx > -rmin){
      dd = 0;
      vv = 1;
    }
    else if (xx > -rmax){
      VALUETYPE value = (-xx - rmin) / (rmax - rmin) * M_PI;
      dd = 0.5 * sin(value) * M_PI / (rmax - rmin);
      vv = 0.5 * (cos(value) + 1);      
    }
    else {
      dd = 0;
      vv = 0;
    }
  }
}

