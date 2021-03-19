#pragma once

namespace deepmd{
  
template <typename FPTYPE>
void soft_min_switch_cpu(
    FPTYPE * sw_value,
    FPTYPE * sw_deriv,
    const FPTYPE * rij,
    const int * nlist,
    const int & nloc,
    const int & nnei, 
    const FPTYPE & alpha,
    const FPTYPE & rmin,
    const FPTYPE & rmax);

}
