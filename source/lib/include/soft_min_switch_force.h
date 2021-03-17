#pragma once

namespace deepmd{
  
template<typename FPTYPE>
void soft_min_switch_force_cpu(
    FPTYPE * force, 
    const FPTYPE * du, 
    const FPTYPE * sw_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

}
