#pragma once

template<typename FPTYPE>
void prod_virial_a_cpu(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

