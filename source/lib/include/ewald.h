#pragma once

#include<algorithm>
#include<cassert>
#include<omp.h>

#include "utilities.h"
#include "region.h"

namespace deepmd{

// 8.988e9 / pc.electron_volt / pc.angstrom * (1.602e-19)**2
const double ElectrostaticConvertion = 14.39964535475696995031;

template <typename VALUETYPE>
struct EwaldParameters 
{
  VALUETYPE rcut = 6.0;
  VALUETYPE beta = 2;
  VALUETYPE spacing = 4;
};

// compute the reciprocal part of the Ewald sum.
// outputs: energy force virial
// inputs: coordinates charges region
template <typename VALUETYPE>
void 
ewald_recp(
    VALUETYPE &				ener, 
    std::vector<VALUETYPE> &		force,
    std::vector<VALUETYPE> &		virial,
    const std::vector<VALUETYPE>&	coord,
    const std::vector<VALUETYPE>&	charge,
    const deepmd::Region<VALUETYPE>&	region, 
    const EwaldParameters<VALUETYPE>&	param);

}
