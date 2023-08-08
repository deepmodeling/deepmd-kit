// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#include "SimulationRegion.h"

const double b2m_l = 10;
const double b2m_e = 1.660539040e-21 / 1.602176621e-19;

template <typename VALUETYPE>
void clear(VALUETYPE& ener,
           std::vector<VALUETYPE>& force,
           std::vector<VALUETYPE>& virial) {
  ener = 0;
  std::fill(force.begin(), force.end(), 0.);
  std::fill(virial.begin(), virial.end(), 0.);
}

template <typename VALUETYPE>
void normalize_coord(std::vector<VALUETYPE>& coord,
                     const SimulationRegion<VALUETYPE>& region) {
  int natoms = coord.size() / 3;
  for (int ii = 0; ii < natoms; ++ii) {
    double phys[3];
    for (int dd = 0; dd < 3; ++dd) {
      phys[dd] = coord[ii * 3 + dd];
    }
    double inter[3];
    region.phys2Inter(inter, phys);
    for (int dd = 0; dd < 3; ++dd) {
      if (inter[dd] < 0) {
        inter[dd] += 1.;
      } else if (inter[dd] >= 1) {
        inter[dd] -= 1.;
      }
    }
    region.inter2Phys(phys, inter);
    for (int dd = 0; dd < 3; ++dd) {
      coord[ii * 3 + dd] = phys[dd];
    }
  }
}
