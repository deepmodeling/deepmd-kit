#include "TF.h"
#include "Interpolation.h"
#include "TableFileLoader.h"
#include <iostream>

TF::
TF (const string & filename)
{
  vector<vector<double> > tmpdata;
  TableFileLoader tfl (filename.c_str());
  tfl.setColumns ({1, 3});
  tfl.loadAll (tmpdata);
  data = tmpdata[1];
  hh = tmpdata[0][1] - tmpdata[0][0];
  xup = tmpdata[0].back();
  xup *= b2m_l;
  hh *= b2m_l;
  for (unsigned ii = 0; ii < data.size(); ++ii){
    data[ii] *= b2m_e / b2m_l;
  }
}

VALUETYPE
TF::
meas (const VALUETYPE & xx) const
{
  VALUETYPE ff = 0;
  if (xx >= xup) {
    ff = 0;
  }
  else {
    int posi = int (xx / hh);
    if (posi < 0) posi = 0;
    else if (posi >= data.size()-1) posi = data.size() - 2;
    Poly p;
    Interpolation::pieceLinearInterpol (posi*hh, (posi+1)*hh, data[posi], data[posi+1], p);
    ff = p.value (xx);
  }
  return ff;
}

void
TF::
apply (vector<VALUETYPE> & dforce,
       const vector<VALUETYPE> & dcoord,
       const AdWeight & adw) const
{
  vector<VALUETYPE> weight, weight_x;
  adw.atom_weight (weight, weight_x, dcoord);
  vector<VALUETYPE> center = adw.get_center();
  
  for (unsigned ii = 0; ii < weight_x.size(); ++ii){
    VALUETYPE ff = meas (weight_x[ii]);
    if (dcoord[ii*3] <  center[0]) {
      ff=-ff;
    }
    dforce [ii*3] += ff;
  }
}



