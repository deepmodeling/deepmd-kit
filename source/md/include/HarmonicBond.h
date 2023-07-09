// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#include "SimulationRegion.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class HarmonicBond {
 public:
  HarmonicBond(const VALUETYPE& kk, const VALUETYPE& bb);

 public:
  void compute(VALUETYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const SimulationRegion<VALUETYPE>& region,
               const std::vector<int>& blist);

 private:
  VALUETYPE kk, bb;
  void hb_inner(VALUETYPE& ae, VALUETYPE& af, const VALUETYPE& r2);
};
