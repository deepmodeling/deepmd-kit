#pragma once

#include <string>
#include <vector>

#include "AdWeight.h"
#include "common.h"

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class TF {
 public:
  TF(const string& filename);

 public:
  void apply(vector<VALUETYPE>& force,
             const vector<VALUETYPE>& coord,
             const AdWeight& weight) const;

 private:
  VALUETYPE meas(const VALUETYPE& xx) const;
  vector<double> data;
  double hh;
  double xup;
};
