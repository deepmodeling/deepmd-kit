// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#include "SimulationRegion.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class MaxShift {
 public:
  MaxShift(const std::vector<VALUETYPE>& dcoord, const VALUETYPE& shell);

  bool rebuild(const std::vector<VALUETYPE>& coord,
               const SimulationRegion<VALUETYPE>& region);

 private:
  VALUETYPE
  max_shift2(const std::vector<VALUETYPE>& coord,
             const SimulationRegion<VALUETYPE>& region);
  std::vector<VALUETYPE> record;
  VALUETYPE shell;
  VALUETYPE max_allow2;
};
