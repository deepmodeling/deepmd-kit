#include "LJTab.h"

LJTab::
LJTab (const VALUETYPE & c6,
       const VALUETYPE & c12,
       const VALUETYPE & rc)
{
  VALUETYPE rcp = rc + 1;
  VALUETYPE hh = 2e-3;
  int nn = rcp / hh;
  vector<VALUETYPE> tab;
  VALUETYPE rc6 = rc * rc * rc * rc * rc * rc ;
  VALUETYPE one_over_rc6 = 1./rc6;
  VALUETYPE one_over_rc12 = 1./rc6/rc6;
  for (int ii = 0; ii < nn; ++ii){
    VALUETYPE xx = ii * hh;
    VALUETYPE value, deriv;
    if (xx <= rc) {
      VALUETYPE xx3 = xx * xx * xx;
      VALUETYPE xx6 = xx3 * xx3;
      VALUETYPE xx12 = xx6 * xx6;    
      value = - c6 / xx6 + c12 / xx12 + c6 * one_over_rc6 - c12 * one_over_rc12;
      deriv = - (6. * c6 / xx6 - 12. * c12 / xx12) / xx;
    }
    else {
      value = deriv = 0;
    }
    tab.push_back (value);
    tab.push_back (deriv);
  }
  lj_tab.reinit (rcp, hh, tab);
}




