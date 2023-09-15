// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#include "SimulationRegion.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class LJInter {
 public:
  LJInter(const VALUETYPE& c6, const VALUETYPE& c12, const VALUETYPE& rc);

 public:
  void compute(VALUETYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const SimulationRegion<VALUETYPE>& region,
               const std::vector<std::vector<int> >& nlist);

 private:
  VALUETYPE c6, c12, rc, rc2, one_over_6, one_over_12, one_over_rc6,
      one_over_rc12;
  void lj_inner(VALUETYPE& ae, VALUETYPE& af, const VALUETYPE& r2);
};
