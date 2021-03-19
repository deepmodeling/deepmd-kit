#pragma once

namespace deepmd{
  
template<typename FPTYPE>
void soft_min_switch_virial_grad_cpu(
    FPTYPE * grad_net, 
    const FPTYPE * grad,
    const FPTYPE * sw_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nnei);

}
