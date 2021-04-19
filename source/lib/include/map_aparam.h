#pragma once

namespace deepmd{
  
template <typename FPTYPE>
void map_aparam_cpu (
    FPTYPE * output,
    const FPTYPE * aparam,
    const int * nlist,
    const int & nloc,
    const int & nnei,
    const int & numb_aparam
    );

}
