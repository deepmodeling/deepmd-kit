#pragma once

namespace deepmd{
  
template<typename FPTYPE>
void soft_min_switch_virial_cpu(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * du, 
    const FPTYPE * sw_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

}
