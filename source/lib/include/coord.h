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

// compute cell information
// output:
// cell_info: nat_stt,ncell,ext_stt,ext_end,ngcell,cell_shift,cell_iter,total_cellnum,loc_cellnum
// input:
// boxt
template <typename FPTYPE>
void
compute_cell_info(
    int * cell_info,
    const float & rcut,
    const deepmd::Region<FPTYPE> & region);

#if GOOGLE_CUDA
// normalize coords
// output:
// coord
// input:
// natom, box_info: boxt, rec_boxt
template <typename FPTYPE>
void
normalize_coord_gpu(
    FPTYPE * coord,
    const int natom,
    const deepmd::Region<FPTYPE> & region);

// copy coordinates
// outputs:
//	out_c, out_t, mapping, nall, 
//  int_data(temp cuda memory):idx_map,idx_map_noshift,temp_idx_order,loc_cellnum_map,total_cellnum_map,mask_cellnum_map,
//                             cell_map,cell_shift_map,sec_loc_cellnum_map,sec_total_cellnum_map,loc_clist
// inputs:
//	in_c, in_t, nloc, mem_nall, loc_cellnum, total_cellnum, cell_info, box_info
//	mem_nall is the size of allocated memory for out_c, out_t, mapping
// returns
//	0: succssful
//	1: the memory is not large enough to hold all copied coords and types.
//	   i.e. nall > mem_nall
template <typename FPTYPE>
int
copy_coord_gpu(
    FPTYPE * out_c,
    int * out_t,
    int * mapping,
    int * nall,
    int * int_data,
    const FPTYPE * in_c,
    const int * in_t,
    const int & nloc,
    const int & mem_nall,
    const int & loc_cellnum,
    const int & total_cellnum,
    const int * cell_info,
    const deepmd::Region<FPTYPE> & region);
#endif // GOOGLE_CUDA

}
