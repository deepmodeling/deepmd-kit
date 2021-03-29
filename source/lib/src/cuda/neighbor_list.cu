#include "device.h"
#include "gpu_cuda.h"
#include "neighbor_list.h"

template<typename FPTYPE>
__device__ inline FPTYPE dev_dot(
    FPTYPE * arr1, 
    FPTYPE * arr2) 
{
    return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

template<typename FPTYPE>
__global__ void build_nlist(
    int * ilist, 
    int * temp_nlist,
    const FPTYPE * c_cpy, 
    const FPTYPE rcut2,
    const int nloc,
    const int nall,
    const int mem_size)
{
    const unsigned int atom_idx = blockIdx.x;
    const unsigned int neighbor_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if(neighbor_idx<nall)
    {
        int * neighbor_row = temp_nlist + atom_idx * mem_size;
        if(neighbor_idx==atom_idx)
        {
            ilist[atom_idx]=atom_idx;
        }
        else
        {
            const FPTYPE * ccoord=c_cpy+atom_idx*3;
            const FPTYPE * ncoord=c_cpy+neighbor_idx*3;
            FPTYPE diff[3];
            for(int kk=0;kk<3;kk++){
                diff[kk] = ccoord[kk] - ncoord[kk];
            }
            FPTYPE r2 = dev_dot(diff, diff);
            if(r2<rcut2){
                neighbor_row[neighbor_idx]=neighbor_idx;
            }
        }
    }
}

__global__ void scan_nlist(
    int * numneigh, 
    int * nei_order, 
    const int * temp_nlist, 
    const int mem_size, 
    const int nloc,
    const int nall)
{
    const unsigned int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(atom_idx<nloc){
        const int * row_nlist = temp_nlist + atom_idx * mem_size;
        int * row_order = nei_order + atom_idx * mem_size;
        int nei_num=0;
        for(int i=0;i<nall;i++){
            if(row_nlist[i]!=-1){
                row_order[i]=nei_num;
                nei_num++;
            }
        }
        numneigh[atom_idx]=nei_num;
    }
}

__global__ void fill_nlist(
    int ** firstneigh,
    const int * temp_nlist,
    const int * nei_order,
    const int mem_size,
    const int nall)
{
    const unsigned int atom_idx = blockIdx.x;
    const unsigned int neighbor_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if(neighbor_idx<nall)
    {
        const int * in_row = temp_nlist + atom_idx * mem_size;
        int * out_row = firstneigh[atom_idx];
        int nei = in_row[neighbor_idx];
        if(nei!=-1){
            out_row[nei_order[atom_idx * mem_size + neighbor_idx]]=nei;
        }
    }
}

__global__ void map_nlist(
    int *nlist,
    const int *nlist_map,
    const int nloc,
    const int nnei
)
{
    int atom_idx=blockIdx.x;
    int nei_idx=blockIdx.y*blockDim.y+threadIdx.y;
    if(nei_idx>=nnei){return;}
    int nlist_idx=atom_idx*nnei+nei_idx;
    int nlist_item=nlist[nlist_idx];
    if(nlist_item!=-1){
        nlist[nlist_idx]=nlist_map[nlist_item];
    }
}

namespace deepmd {
template <typename FPTYPE>
int build_nlist_gpu(
    InputNlist & nlist,
    int * max_list_size,
    int * nlist_data,
    const FPTYPE * c_cpy, 
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut)
{
    if(mem_size < nall){
        return 1;
    }
    const int nblock = (nall+TPB-1)/TPB;
    int * ilist = nlist.ilist;
    int * numneigh = nlist.numneigh;
    int ** firstneigh = nlist.firstneigh;
    cudaErrcheck(cudaMemset(nlist_data, -1, sizeof(int) * 2 * nloc * mem_size));
    int * temp_nlist = nlist_data; //nloc*mem_size
    int * nei_order = temp_nlist + nloc * mem_size;
    nlist.inum = nloc;
    FPTYPE rcut2 = rcut * rcut;
    
    
    dim3 block_grid(nloc, nblock);
    dim3 thread_grid(1, TPB);
    build_nlist<<<block_grid, thread_grid>>>(
                ilist, 
                temp_nlist,
                c_cpy, 
                rcut2,
                nloc,
                nall,
                mem_size);
    const int nblock_ = (nloc+TPB-1)/TPB;
    scan_nlist<<<nblock_, TPB>>>(
                numneigh, 
                nei_order, 
                temp_nlist, 
                mem_size, 
                nloc, 
                nall);
    fill_nlist<<<block_grid, thread_grid>>>(
                firstneigh,
                temp_nlist,
                nei_order,
                mem_size,
                nall
    );
    int * numneigh_host = new int[nloc];
    cudaErrcheck(cudaMemcpy(numneigh_host, numneigh, sizeof(int) * nloc, cudaMemcpyDeviceToHost));
    int max_nei = 0;
    for(int ii=0;ii<nloc;ii++){
        if(numneigh_host[ii]>max_nei)max_nei=numneigh_host[ii];
    }
    *max_list_size = max_nei;
    delete [] numneigh_host;
    return 0;
}

void use_nlist_map(
    int * nlist, 
    const int * nlist_map, 
    const int nloc, 
    const int nnei)
{
    int nblock=(nnei+TPB-1)/TPB;
    dim3 block_grid(nloc, nblock);
    dim3 thread_grid(1, TPB);
    map_nlist<<<block_grid,thread_grid>>>(nlist, nlist_map, nloc, nnei);
}

template int build_nlist_gpu<float>(InputNlist & nlist, int * max_list_size, int * nlist_data, const float * c_cpy, const int & nloc, const int & nall, const int & mem_size, const float & rcut);
template int build_nlist_gpu<double>(InputNlist & nlist, int * max_list_size, int * nlist_data, const double * c_cpy, const int & nloc, const int & nall, const int & mem_size, const float & rcut);
}