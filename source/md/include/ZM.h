// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#include "SimulationRegion.h"
#include "Tabulated.h"
#include "ZMFunctions.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class ZM {
 public:
  ZM(const int& order, const VALUETYPE& alpha, const VALUETYPE& rc);

 public:
  void compute(VALUETYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& charge,
               const std::vector<int>& atype,
               const SimulationRegion<VALUETYPE>& region,
               const std::vector<std::vector<int> >& nlist) {
    zm_tab.compute(ener, force, virial, coord, charge, atype, region, nlist);
  };
  void exclude(VALUETYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& charge,
               const std::vector<int>& atype,
               const SimulationRegion<VALUETYPE>& region,
               const std::vector<int>& elist);
  VALUETYPE e_corr(const std::vector<VALUETYPE>& charge) const;

 private:
  Tabulated zm_tab;
  void ex_inner(VALUETYPE& ae, VALUETYPE& af, const VALUETYPE& r2);
  ZeroMultipole::Potential potzm;
};
