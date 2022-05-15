
/*
//==================================================
 _   _  __     __  _   _   __  __   ____  
| \ | | \ \   / / | \ | | |  \/  | |  _ \ 
|  \| |  \ \ / /  |  \| | | |\/| | | | | |
| |\  |   \ V /   | |\  | | |  | | | |_| |
|_| \_|    \_/    |_| \_| |_|  |_| |____/ 

//==================================================

code: nvnmd
reference: deepmd
author: mph (pinghui_mo@outlook.com)
date: 2021-12-6

*/

#pragma once

#include <cmath>
#include <vector>
#include "utilities.h"

namespace deepmd{

template<typename FPTYPE> 
void env_mat_a_nvnmd_quantize_cpu (
    std::vector<FPTYPE > &	        descrpt_a,
    std::vector<FPTYPE > &	        descrpt_a_deriv,
    std::vector<FPTYPE > &	        rij_a,
    const std::vector<FPTYPE > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax,
    const FPTYPE            precs[3]);

}
