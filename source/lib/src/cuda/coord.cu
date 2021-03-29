#include "device.h"
#include "gpu_cuda.h"
#include "coord.h"
#include "region.cuh"

__device__ inline int collapse_index(
    const int * idx,
    const int * size)
{
    return (idx[0] * size[1] + idx[1]) * size[2] + idx[2];
}
__device__ inline void index_recover(
    const int in_idx,
    const int * size, 
    int * idx)
{
    idx[2]=in_idx%size[2];
    idx[1]=int(in_idx/size[2])%size[1];
    idx[0]=int(int(in_idx/size[2])/size[1]);
}
__device__ inline void idx_addshift(
    int * idx, 
    const int * shift)
{
    for(int dd=0;dd<3;dd++)
    {
        idx[dd]+=shift[dd];
    }
}
__device__ inline void idx_unshift(
    int * idx, 
    const int * shift)
{
    for(int dd=0;dd<3;dd++)
    {
        idx[dd]-=shift[dd];
    }
}
__device__ inline int compute_pbc_shift(
    int idx, 
    int ncell)
{
    int shift = 0;
    if (idx < 0) {
    shift = 1;
    while (idx + shift * ncell < 0) shift ++;
    }
    else if (idx >= ncell) {
    shift = -1;
    while (idx + shift * ncell >= ncell) shift --;
    }
    return shift;
}

template<typename FPTYPE>
__global__ void normalize_one(
    FPTYPE *out_c,
    const FPTYPE *boxt,
    const FPTYPE *rec_boxt,
    const int nall)
{
    // <<<nall/TPB, TPB>>>
    int idy=blockIdx.x*blockDim.x+threadIdx.x;
    if (idy>=nall){return;}
    FPTYPE inter[3];
    phys2Inter(inter,out_c+idy*3,rec_boxt);
    for (int dd = 0; dd < 3; ++dd) {
        while(inter[dd] >= 1.) inter[dd] -= 1.;
        while(inter[dd] < 0.) inter[dd] += 1.;
    }
    inter2Phys(out_c+idy*3,inter,boxt);
}

template<typename FPTYPE>
__global__ void _compute_int_data(
    const FPTYPE *in_c,
    const int *nat_stt,
    const int *nat_end,
    const int *ext_stt,
    const int *ext_end,
    const int *ngcell,
    const FPTYPE *boxt,
    const FPTYPE *rec_boxt,
    int * idx_cellmap,
    int * idx_cellmap_noshift,
    int * total_cellnum_map,
    int * mask_cellnum_map,
    int * cell_map,
    int * loc_cellnum_map,
    int * cell_shift_map,
    int * temp_idx_order,
    const int nloc,
    const int loc_cellnum,
    const int total_cellnum)
{
    int idy = blockIdx.x*blockDim.x+threadIdx.x;
    int ext_ncell[3];
    int global_grid[3];
    int idx_orig_shift[3];
    FPTYPE cell_size[3];
    FPTYPE nat_orig[3];
    for (int dd = 0; dd < 3; ++dd) 
    {
        ext_ncell[dd] = ext_end[dd] - ext_stt[dd];
        global_grid[dd] = nat_end[dd] - nat_stt[dd];
        idx_orig_shift[dd] = nat_stt[dd] - ext_stt[dd];
        cell_size[dd] = 1./global_grid[dd];
        nat_orig[dd] = nat_stt[dd] * cell_size[dd];
    }
    if (idy<nloc)
    {
        int idx_noshift[3]; 
        int idx[3];
        FPTYPE inter[3];
        phys2Inter(inter,in_c+idy*3,rec_boxt);
        for (int dd = 0; dd < 3; ++dd){
            idx_noshift[dd] = (inter[dd] - nat_orig[dd]) / cell_size[dd];
            if (inter[dd] - nat_orig[dd] < 0.) idx_noshift[dd] --;
            if (idx_noshift[dd] < nat_stt[dd]) 
            {
                idx_noshift[dd] = nat_stt[dd];
            }
            else if (idx_noshift[dd] >= nat_end[dd]) 
            {
                idx_noshift[dd] = nat_end[dd] - 1;
            }
            idx[dd] = idx_noshift[dd]+idx_orig_shift[dd];
        }
        idx_cellmap_noshift[idy]=collapse_index(idx_noshift, global_grid);
        idx_cellmap[idy]=collapse_index(idx, ext_ncell);
    }
    __syncthreads();
    if (idy<loc_cellnum)
    {
        int num=0;
        for(int ii=0;ii<nloc;ii++)
        {
            if(idx_cellmap_noshift[ii]==idy)
            {
                temp_idx_order[ii]=num;
                num++;
            }
        }
        loc_cellnum_map[idy]=num;
    }
    __syncthreads();
    if(idy<total_cellnum)
    {
        int * shift=cell_shift_map+idy*3;
        int idx[3];
        index_recover(idy, ext_ncell, idx);
        idx_unshift(idx, idx_orig_shift);
        shift[0]=compute_pbc_shift(idx[0],global_grid[0]);
        shift[1]=compute_pbc_shift(idx[1],global_grid[1]);
        shift[2]=compute_pbc_shift(idx[2],global_grid[2]);
        bool loc=false;
        if(shift[0]==0&&shift[1]==0&&shift[2]==0)loc=true;
        for(int dd=0;dd<3;dd++)
        {
            idx[dd]+=shift[dd]*global_grid[dd];
        }
        int orig_idy=collapse_index(idx, global_grid);
        mask_cellnum_map[idy]=loc_cellnum_map[orig_idy];
        total_cellnum_map[idy]=mask_cellnum_map[idy];
        if(loc)mask_cellnum_map[idy]=0;
        cell_map[idy]=orig_idy;
    }
}

__global__ void _build_loc_clist(
    int *clist,
    const int *idx_cellmap, 
    const int *idx_order,
    const int *sec_num_map,
    const int nloc)
{
    int idy = blockIdx.x*blockDim.x+threadIdx.x;
    if(idy>=nloc){return;}
    int cell_idx=idx_cellmap[idy];
    int * clist_row = clist+sec_num_map[cell_idx];
    clist_row[idx_order[idy]]=idy;
}

template<typename FPTYPE>
__global__ void _copy_coord(
    FPTYPE * out_c, 
    int * out_t, 
    int * mapping, 
    const FPTYPE * in_c, 
    const int * in_t, 
    const int * cell_map, 
    const int * cell_shift_map, 
    const int * sec_loc_cellnum_map, 
    const int * sec_total_cellnum_map, 
    const int * loc_clist, 
    const int nloc, 
    const int nall, 
    const int total_cellnum, 
    const FPTYPE * boxt, 
    const FPTYPE * rec_boxt)
{
    int idy = blockIdx.x*blockDim.x+threadIdx.x;
    if(idy>=nall){return;}
    if(idy<nloc)
    {
        mapping[idy]=idy;
        out_t[idy]=in_t[idy];
        for(int dd=0;dd<3;dd++)
        {
            out_c[idy*3+dd]=in_c[idy*3+dd];
        }
    }
    else
    {
        int cell_idx=0;
        int atom_idx=0;
        int orig_cell_idx=0;
        int orig_idy=0;
        int shift[3];
        FPTYPE d_shift[3];
        for(int ii=0;ii<total_cellnum;ii++)
        {
            if(idy>=sec_total_cellnum_map[ii+1])cell_idx++;
            else break;
        }
        for(int dd=0;dd<3;dd++)
        {
            shift[dd]=cell_shift_map[cell_idx*3+dd];
            d_shift[dd]=shift[dd];
        }
        atom_idx=idy-sec_total_cellnum_map[cell_idx];
        orig_cell_idx=cell_map[cell_idx];
        orig_idy=loc_clist[sec_loc_cellnum_map[orig_cell_idx]+atom_idx];
        mapping[idy]=orig_idy;
        out_t[idy]=in_t[orig_idy];
        FPTYPE shift_v[3];
        inter2Phys(shift_v,d_shift,boxt);
        for(int dd=0;dd<3;dd++)
        {
            out_c[idy*3+dd]=in_c[orig_idy*3+dd]-shift_v[dd];
        }
    }
}

template <typename FPTYPE>
void compute_int_data(
    int * int_data, 
    const FPTYPE * in_c, 
    const int * cell_info, 
    const deepmd::Region<FPTYPE> & region, 
    const int nloc, 
    const int loc_cellnum, 
    const int total_cellnum)
{
    const int nn=(nloc>=total_cellnum)?nloc:total_cellnum; 
    const int nblock=(nn+TPB-1)/TPB;
    int * idx_cellmap=int_data;
    int * idx_cellmap_noshift=idx_cellmap+nloc;
    int * temp_idx_order=idx_cellmap_noshift+nloc;
    int * loc_cellnum_map=temp_idx_order+nloc;
    int * total_cellnum_map=loc_cellnum_map+loc_cellnum;
    int * mask_cellnum_map=total_cellnum_map+total_cellnum;
    int * cell_map=mask_cellnum_map+total_cellnum;
    int * cell_shift_map=cell_map+total_cellnum;
    
    const int * nat_stt=cell_info;
    const int * nat_end=cell_info+3;
    const int * ext_stt=cell_info+6;
    const int * ext_end=cell_info+9;
    const int * ngcell=cell_info+12;
    const FPTYPE * boxt = region.boxt;
    const FPTYPE * rec_boxt = region.rec_boxt;
    _compute_int_data<<<nblock, TPB>>>(in_c, nat_stt, nat_end, ext_stt, ext_end, ngcell,
        boxt, rec_boxt, idx_cellmap, idx_cellmap_noshift, total_cellnum_map, mask_cellnum_map,
        cell_map, loc_cellnum_map, cell_shift_map, temp_idx_order, nloc, loc_cellnum, total_cellnum);
}

void build_loc_clist(
    int * int_data, 
    const int nloc, 
    const int loc_cellnum, 
    const int total_cellnum)
{
    const int nblock=(nloc+TPB-1)/TPB;
    const int * idx_cellmap_noshift=int_data+nloc;
    const int * temp_idx_order=idx_cellmap_noshift+nloc;
    const int * sec_loc_cellnum_map=temp_idx_order+nloc+loc_cellnum+2*total_cellnum+total_cellnum+3*total_cellnum;
    int * loc_clist=int_data+nloc*3+loc_cellnum+total_cellnum*3+total_cellnum*3+loc_cellnum+1+total_cellnum+1;
    _build_loc_clist<<<nblock, TPB>>>(loc_clist, idx_cellmap_noshift, temp_idx_order, sec_loc_cellnum_map, nloc);
}

template <typename FPTYPE>
void copy_coord(
    FPTYPE * out_c, 
    int * out_t, 
    int * mapping, 
    const int * int_data, 
    const FPTYPE * in_c, 
    const int * in_t, 
    const int nloc, 
    const int nall, 
    const int loc_cellnum, 
    const int total_cellnum, 
    const deepmd::Region<FPTYPE> & region)
{
    const int nblock=(nall+TPB-1)/TPB;
    const int * cell_map=int_data+3*nloc+loc_cellnum+2*total_cellnum;
    const int * cell_shift_map=cell_map+total_cellnum;
    const int * sec_loc_cellnum_map=cell_shift_map+3*total_cellnum;
    const int * sec_total_cellnum_map=sec_loc_cellnum_map+loc_cellnum+1;
    const int * loc_clist=sec_total_cellnum_map+total_cellnum+1;

    const FPTYPE *boxt = region.boxt;
    const FPTYPE *rec_boxt = region.rec_boxt;
    _copy_coord<<<nblock, TPB>>>(out_c, out_t, mapping, in_c, in_t, cell_map, cell_shift_map, 
        sec_loc_cellnum_map, sec_total_cellnum_map, loc_clist, nloc, nall, total_cellnum, boxt, rec_boxt);
}

namespace deepmd {
template <typename FPTYPE>
void
normalize_coord_gpu(
    FPTYPE * coord,
    const int natom,
    const Region<FPTYPE> & region)
{
    const FPTYPE * boxt=region.boxt;
    const FPTYPE * rec_boxt=region.rec_boxt;
    const int nblock=(natom+TPB-1)/TPB;
    normalize_one<<<nblock, TPB>>>(coord, boxt, rec_boxt, natom);
}

//  int_data(temp cuda memory):idx_map,idx_map_noshift,temp_idx_order,loc_cellnum_map,total_cellnum_map,mask_cellnum_map,
//                             cell_map,cell_shift_map,sec_loc_cellnum_map,sec_total_cellnum_map,loc_clist
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
    const Region<FPTYPE> & region)
{
    compute_int_data(int_data, in_c, cell_info, region, nloc, loc_cellnum, total_cellnum);
    int * int_data_cpu=new int [loc_cellnum+2*total_cellnum+loc_cellnum+1+total_cellnum+1];//loc_cellnum_map,total_cellnum_map,mask_cellnum_map,sec_loc_cellnum_map,sec_total_cellnum_map
    cudaErrcheck(cudaMemcpy(int_data_cpu, int_data+3*nloc, sizeof(int) * (loc_cellnum + 2 * total_cellnum), cudaMemcpyDeviceToHost));
    int * loc_cellnum_map=int_data_cpu;
    int * total_cellnum_map=loc_cellnum_map+loc_cellnum;
    int * mask_cellnum_map=total_cellnum_map+total_cellnum;
    int * sec_loc_cellnum_map=mask_cellnum_map+total_cellnum;
    int * sec_total_cellnum_map=sec_loc_cellnum_map+loc_cellnum+1;
    sec_loc_cellnum_map[0]=0;
    sec_total_cellnum_map[0]=nloc;
    int max_cell=0;
    for(int iii=0;iii<total_cellnum;iii++)
    {
        if(max_cell<total_cellnum_map[iii]){max_cell=total_cellnum_map[iii];}
        if(iii<loc_cellnum){sec_loc_cellnum_map[iii+1]=sec_loc_cellnum_map[iii]+loc_cellnum_map[iii];}
        sec_total_cellnum_map[iii+1]=sec_total_cellnum_map[iii]+mask_cellnum_map[iii];
    }
    *nall=sec_total_cellnum_map[total_cellnum];
    if(*nall > mem_nall){
        delete[] int_data_cpu;
        // size of the output arrays is not large enough
        return 1;
    }
    else{
        cudaErrcheck(cudaMemcpy(int_data+nloc*3+loc_cellnum+total_cellnum*3+total_cellnum*3, 
            sec_loc_cellnum_map, sizeof(int) * (loc_cellnum+1+total_cellnum+1), cudaMemcpyHostToDevice));
        delete[] int_data_cpu;
        build_loc_clist(int_data, nloc, loc_cellnum, total_cellnum);
        copy_coord(out_c, out_t, mapping, int_data, in_c, in_t, nloc, *nall, loc_cellnum, total_cellnum, region);
    }
    return 0;
}

template void normalize_coord_gpu<float>(float * coord, const int natom, const Region<float> & region);
template void normalize_coord_gpu<double>(double * coord, const int natom, const Region<double> & region);
template int copy_coord_gpu<float>(float * out_c, int * out_t, int * mapping, int * nall, int * int_data, const float * in_c, const int * in_t, const int & nloc, const int & mem_nall, const int & loc_cellnum, const int & total_cellnum, const int * cell_info, const Region<float> & region);
template int copy_coord_gpu<double>(double * out_c, int * out_t, int * mapping, int * nall, int * int_data, const double * in_c, const int * in_t, const int & nloc, const int & mem_nall, const int & loc_cellnum, const int & total_cellnum, const int * cell_info, const Region<double> & region);
}