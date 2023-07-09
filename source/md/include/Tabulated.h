// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#include "SimulationRegion.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class Tabulated {
 public:
  Tabulated(){};
  Tabulated(const VALUETYPE rc,
            const VALUETYPE hh,
            const std::vector<VALUETYPE>& tab);
  void reinit(const VALUETYPE rc,
              const VALUETYPE hh,
              const std::vector<VALUETYPE>& tab);

 public:
  void compute(VALUETYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const SimulationRegion<VALUETYPE>& region,
               const std::vector<std::vector<int> >& nlist);
  void compute(VALUETYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& charge,
               const std::vector<int>& atype,
               const SimulationRegion<VALUETYPE>& region,
               const std::vector<std::vector<int> >& nlist);
  void tb_inner(VALUETYPE& ae, VALUETYPE& af, const VALUETYPE& r2);

 private:
  VALUETYPE rc2, hi;
  std::vector<VALUETYPE> data;
  void compute_posi(int& idx, VALUETYPE& eps, const VALUETYPE& rr);
};
