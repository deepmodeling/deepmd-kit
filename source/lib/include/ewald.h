// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#else
int omp_get_num_threads() { return 1; }
int omp_get_thread_num() { return 0; }
#endif

#include "region.h"
#include "utilities.h"

namespace deepmd {

// 8.988e9 / pc.electron_volt / pc.angstrom * (1.602e-19)**2
const double ElectrostaticConvertion = 14.39964535475696995031;

template <typename VALUETYPE>
struct EwaldParameters {
  VALUETYPE rcut = 6.0;
  VALUETYPE beta = 2;
  VALUETYPE spacing = 4;
};

// compute the reciprocal part of the Ewald sum.
// outputs: energy force virial
// inputs: coordinates charges region
template <typename VALUETYPE>
void ewald_recp(VALUETYPE& ener,
                std::vector<VALUETYPE>& force,
                std::vector<VALUETYPE>& virial,
                const std::vector<VALUETYPE>& coord,
                const std::vector<VALUETYPE>& charge,
                const deepmd::Region<VALUETYPE>& region,
                const EwaldParameters<VALUETYPE>& param);

}  // namespace deepmd
