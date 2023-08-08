// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <vector>

namespace deepmd {

template <typename FPTYPE>
void pair_tab_cpu(FPTYPE* energy,
                  FPTYPE* force,
                  FPTYPE* virial,
                  const double* table_info,
                  const double* table_data,
                  const FPTYPE* rij,
                  const FPTYPE* scale,
                  const int* type,
                  const int* nlist,
                  const int* natoms,
                  const std::vector<int>& sel_a,
                  const std::vector<int>& sel_r);

}
