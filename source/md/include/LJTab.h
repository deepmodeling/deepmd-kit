// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#include "Tabulated.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class LJTab {
 public:
  LJTab(const VALUETYPE& c6, const VALUETYPE& c12, const VALUETYPE& rc);

 public:
  void compute(VALUETYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const SimulationRegion<VALUETYPE>& region,
               const std::vector<std::vector<int> >& nlist) {
    lj_tab.compute(ener, force, virial, coord, atype, region, nlist);
  };

 private:
  Tabulated lj_tab;
};
