#pragma once

#include "region.h"

namespace deepmd{

// normalize coords
template <typename FPTYPE>
void
normalize_coord_cpu(
    FPTYPE * coord,
    const int natom,
    const deepmd::Region<FPTYPE> & region);

// copy coordinates
// outputs:
//	out_c, out_t, mapping, nall
// inputs:
//	in_c, in_t, nloc, mem_nall, rc, region
//	mem_nall is the size of allocated memory for out_c, out_t, mapping
// returns
//	0: succssful
//	1: the memory is not large enough to hold all copied coords and types.
//	   i.e. nall > mem_nall
template <typename FPTYPE>
int
copy_coord_cpu(
    FPTYPE * out_c,
    int * out_t,
    int * mapping,
    int * nall,
    const FPTYPE * in_c,
    const int * in_t,
    const int & nloc,
    const int & mem_nall,
    const float & rcut,
    const deepmd::Region<FPTYPE> & region);

}
