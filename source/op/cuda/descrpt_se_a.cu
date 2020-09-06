#include <vector>
#include <climits>
#include <stdio.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

#define cudaErrcheck(res) {cudaAssert((res), __FILE__, __LINE__);}
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

typedef unsigned long long int_64;

template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void BlockSortKernel(
    Key * d_in,
    Key * d_out)            // Tile of output
{   
    enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef cub::BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    // Specialize BlockRadixSort type for our thread block
    typedef cub::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage        load;
        typename BlockRadixSortT::TempStorage   sort;
    } temp_storage;
    // Per-thread tile items
    Key items[ITEMS_PER_THREAD];
    // Our current block's offset
    int block_offset = blockIdx.x * TILE_SIZE;
    // Load items into a blocked arrangement
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
    // Barrier for smem reuse
    __syncthreads();
    // Sort keys
    BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(items);
    // Store output in striped fashion
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items);
}

template<typename T>
__device__ inline T dev_dot(T * arr1, T * arr2) {
    return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

template<typename T>
__device__ inline void spline5_switch(T & vv, T & dd, T & xx, const float & rmin, const float & rmax) 
{
    if (xx < rmin) {
        dd = 0;
        vv = 1;
    }
    else if (xx < rmax) {
        T uu = (xx - rmin) / (rmax - rmin) ;
        T du = 1. / (rmax - rmin) ;
        vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1;
        dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
    }
    else {
        dd = 0;
        vv = 0;
    }
}

__global__ void get_i_idx_se_a(const int nloc, const int * ilist, int * i_idx)
{
    const unsigned int idy = blockIdx.x * blockDim.x + threadIdx.x;
    if(idy >= nloc) {return;}

    i_idx[ilist[idy]] = idy;
}

template<typename T>
__global__ void format_nlist_fill_a_se_a(
    const T * coord,
    const int  * type,
    const int  * jrange,
    const int  * jlist,
    const float rcut,
    int_64 * key,
    int * i_idx,
    const int MAGIC_NUMBER)
{   
    // <<<nloc, MAGIC_NUMBER>>>
    const unsigned int idx = blockIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int nsize = jrange[i_idx[idx] + 1] - jrange[i_idx[idx]];
    if (idy >= nsize) {return;}

    const int * nei_idx = jlist + jrange[i_idx[idx]];
    // dev_copy(nei_idx, &jlist[jrange[i_idx]], nsize);

    int_64 * key_in = key + idx * MAGIC_NUMBER;

    T diff[3];
    const int & j_idx = nei_idx[idy];
    for (int dd = 0; dd < 3; dd++) {
        diff[dd] = coord[j_idx * 3 + dd] - coord[idx * 3 + dd];
    }
    T rr = sqrt(dev_dot(diff, diff)); 
    if (rr <= rcut) {
        key_in[idy] = (int_64)(type[j_idx] * 1E15) + (int_64)(rr * 1.0E13) / 100000 * 100000 + j_idx;
    }
}

    // bubble_sort(sel_nei, num_nei);
__global__ void format_nlist_fill_b_se_a(
    int * nlist,
    const int nlist_size,
    const int nloc,
    const int * jrange,
    const int * jlist,
    int_64 * key,
    const int * sec_a,
    const int sec_a_size,
    int * nei_iter_dev,
    const int MAGIC_NUMBER)
{ 

    const unsigned int idy = blockIdx.x * blockDim.x + threadIdx.x;

    if(idy >= nloc) {
        return;
    }
    
    int * row_nlist = nlist + idy * nlist_size;
    int * nei_iter = nei_iter_dev + idy * sec_a_size;
    int_64 * key_out = key + nloc * MAGIC_NUMBER + idy * MAGIC_NUMBER;

    for (int ii = 0; ii < sec_a_size; ii++) {
        nei_iter[ii] = sec_a[ii];
    }
    
    for (unsigned int kk = 0; key_out[kk] != key_out[MAGIC_NUMBER - 1]; kk++) {
        const int & nei_type = key_out[kk] / 1E15;
        if (nei_iter[nei_type] < sec_a[nei_type + 1]) {
            row_nlist[nei_iter[nei_type]++] = key_out[kk] % 100000;
        }
    }
}
//it's ok!

template<typename T>
__global__ void compute_descriptor_se_a (
    T* descript,
    const int ndescrpt,
    T* descript_deriv,
    const int descript_deriv_size,
    T* rij,
    const int rij_size,
    const int* type,
    const T* avg,
    const T* std,
    int* nlist,
    const int nlist_size,
    const T* coord,
    const float rmin,
    const float rmax,
    const int sec_a_size)
{   
    // <<<nloc, sec_a.back()>>>
    const unsigned int idx = blockIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx_deriv = idy * 4 * 3;	// 4 components time 3 directions
    const int idx_value = idy * 4;	// 4 components
    if (idy >= sec_a_size) {return;}

    // else {return;}
    T * row_descript = descript + idx * ndescrpt;
    T * row_descript_deriv = descript_deriv + idx * descript_deriv_size;
    T * row_rij = rij + idx * rij_size;
    int * row_nlist = nlist + idx * nlist_size;

    if (row_nlist[idy] >= 0) {
        const int & j_idx = row_nlist[idy];
        for (int kk = 0; kk < 3; kk++) {
            row_rij[idy * 3 + kk] = coord[j_idx * 3 + kk] - coord[idx * 3 + kk];
        }
        const T * rr = &row_rij[idy * 3 + 0];
        T nr2 = dev_dot(rr, rr);
        T inr = 1./sqrt(nr2);
        T nr = nr2 * inr;
        T inr2 = inr * inr;
        T inr4 = inr2 * inr2;
        T inr3 = inr4 * nr;
        T sw, dsw;
        spline5_switch(sw, dsw, nr, rmin, rmax);
        row_descript[idx_value + 0] = (1./nr)       ;//* sw;
        row_descript[idx_value + 1] = (rr[0] / nr2) ;//* sw;
        row_descript[idx_value + 2] = (rr[1] / nr2) ;//* sw;
        row_descript[idx_value + 3] = (rr[2] / nr2) ;//* sw;

        row_descript_deriv[idx_deriv + 0] = (rr[0] * inr3 * sw - row_descript[idx_value + 0] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 0) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 0) % (ndescrpt * 3)) / 3];
        row_descript_deriv[idx_deriv + 1] = (rr[1] * inr3 * sw - row_descript[idx_value + 0] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 1) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 1) % (ndescrpt * 3)) / 3];
        row_descript_deriv[idx_deriv + 2] = (rr[2] * inr3 * sw - row_descript[idx_value + 0] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 2) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 2) % (ndescrpt * 3)) / 3];
        // ****deriv of component x/r2
        row_descript_deriv[idx_deriv + 3] = ((2. * rr[0] * rr[0] * inr4 - inr2) * sw - row_descript[idx_value + 1] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 3) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 3) % (ndescrpt * 3)) / 3];
        row_descript_deriv[idx_deriv + 4] = ((2. * rr[0] * rr[1] * inr4	) * sw - row_descript[idx_value + 1] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 4) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 4) % (ndescrpt * 3)) / 3];
        row_descript_deriv[idx_deriv + 5] = ((2. * rr[0] * rr[2] * inr4	) * sw - row_descript[idx_value + 1] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 5) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 5) % (ndescrpt * 3)) / 3];
        // ***deriv of component y/r2
        row_descript_deriv[idx_deriv + 6] = ((2. * rr[1] * rr[0] * inr4	) * sw - row_descript[idx_value + 2] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 6) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 6) % (ndescrpt * 3)) / 3];
        row_descript_deriv[idx_deriv + 7] = ((2. * rr[1] * rr[1] * inr4 - inr2) * sw - row_descript[idx_value + 2] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 7) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 7) % (ndescrpt * 3)) / 3];
        row_descript_deriv[idx_deriv + 8] = ((2. * rr[1] * rr[2] * inr4	) * sw - row_descript[idx_value + 2] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 8) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 8) % (ndescrpt * 3)) / 3];
        // ***deriv of component z/r2
        row_descript_deriv[idx_deriv + 9] = ((2. * rr[2] * rr[0] * inr4	) * sw - row_descript[idx_value + 3] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 9) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 9) % (ndescrpt * 3)) / 3];
        row_descript_deriv[idx_deriv +10] = ((2. * rr[2] * rr[1] * inr4	) * sw - row_descript[idx_value + 3] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 10) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 10) % (ndescrpt * 3)) / 3];
        row_descript_deriv[idx_deriv +11] = ((2. * rr[2] * rr[2] * inr4 - inr2) * sw - row_descript[idx_value + 3] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 11) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 11) % (ndescrpt * 3)) / 3];
        // 4 value components
        row_descript[idx_value + 0] *= sw; // * descript[idx * ndescrpt + idx_value + 0]);// - avg[type[idx] * ndescrpt + idx_value + 0]) / std[type[idx] * ndescrpt + idx_value + 0];
        row_descript[idx_value + 1] *= sw; // * descript[idx * ndescrpt + idx_value + 1]);// - avg[type[idx] * ndescrpt + idx_value + 1]) / std[type[idx] * ndescrpt + idx_value + 1];
        row_descript[idx_value + 2] *= sw; // * descript[idx * ndescrpt + idx_value + 2]);// - avg[type[idx] * ndescrpt + idx_value + 2]) / std[type[idx] * ndescrpt + idx_value + 2];
        row_descript[idx_value + 3] *= sw; // * descript[idx * ndescrpt + idx_value + 3]);// - avg[type[idx] * ndescrpt + idx_value + 3]) / std[type[idx] * ndescrpt + idx_value + 3];
    }

    for (int ii = 0; ii < 4; ii++) {
        row_descript[idx_value + ii] = (row_descript[idx_value + ii] - avg[type[idx] * ndescrpt + idx_value + ii]) / std[type[idx] * ndescrpt + idx_value + ii];
    }
    for (int ii = 0; ii < 12; ii++) {
        row_descript_deriv[idx_deriv + ii] /= std[type[idx] * ndescrpt + (idx_deriv + ii) / 3];
    }
}

template<typename T>
void format_nbor_list_256 (const T* coord, const int* type, const int* jrange, const int* jlist, const int& nloc, const float& rcut_r, int * i_idx, int_64 * key)  {   
    const int LEN = 256;
    const int MAGIC_NUMBER = 256;
    const int nblock = (MAGIC_NUMBER + LEN - 1) / LEN;
    dim3 block_grid(nloc, nblock);
    dim3 thread_grid(1, LEN);
    format_nlist_fill_a_se_a<<<block_grid, thread_grid>>> (coord, type, jrange, jlist, rcut_r, key, i_idx, MAGIC_NUMBER);
    
    const int ITEMS_PER_THREAD = 4;
    const int BLOCK_THREADS = MAGIC_NUMBER / ITEMS_PER_THREAD;
    BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (key, key + nloc * MAGIC_NUMBER);
}

template<typename T>
void format_nbor_list_512 (const T* coord, const int* type, const int* jrange, const int* jlist, const int& nloc, const float& rcut_r, int * i_idx, int_64 * key)  {   
    const int LEN = 256;
    const int MAGIC_NUMBER = 512;
    const int nblock = (MAGIC_NUMBER + LEN - 1) / LEN;
    dim3 block_grid(nloc, nblock);
    dim3 thread_grid(1, LEN);
    format_nlist_fill_a_se_a<<<block_grid, thread_grid>>> (coord, type, jrange, jlist, rcut_r, key, i_idx, MAGIC_NUMBER);

    const int ITEMS_PER_THREAD = 4;
    const int BLOCK_THREADS = MAGIC_NUMBER / ITEMS_PER_THREAD;
    BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (key, key + nloc * MAGIC_NUMBER);
}

template<typename T>
void format_nbor_list_1024 (const T* coord, const int* type, const int* jrange, const int* jlist, const int& nloc, const float& rcut_r, int * i_idx, int_64 * key)  {   
    const int LEN = 256;
    const int MAGIC_NUMBER = 1024;
    const int nblock = (MAGIC_NUMBER + LEN - 1) / LEN;
    dim3 block_grid(nloc, nblock);
    dim3 thread_grid(1, LEN);
    format_nlist_fill_a_se_a<<<block_grid, thread_grid>>> (coord, type, jrange, jlist, rcut_r, key, i_idx, MAGIC_NUMBER);

    const int ITEMS_PER_THREAD = 8;
    const int BLOCK_THREADS = MAGIC_NUMBER / ITEMS_PER_THREAD;
    BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (key, key + nloc * MAGIC_NUMBER);
}

template<typename T>
void format_nbor_list_2048 (const T* coord, const int* type, const int* jrange, const int* jlist, const int& nloc, const float& rcut_r, int * i_idx, int_64 * key)  {   
    const int LEN = 256;
    const int MAGIC_NUMBER = 2048;
    const int nblock = (MAGIC_NUMBER + LEN - 1) / LEN;
    dim3 block_grid(nloc, nblock);
    dim3 thread_grid(1, LEN);
    format_nlist_fill_a_se_a<<<block_grid, thread_grid>>> (coord, type, jrange, jlist, rcut_r, key, i_idx, MAGIC_NUMBER);

    const int ITEMS_PER_THREAD = 8;
    const int BLOCK_THREADS = MAGIC_NUMBER / ITEMS_PER_THREAD;
    BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (key, key + nloc * MAGIC_NUMBER);
}

template<typename T>
void format_nbor_list_4096 (const T* coord, const int* type, const int* jrange, const int* jlist, const int& nloc, const float& rcut_r, int * i_idx, int_64 * key)  {   
    const int LEN = 256;
    const int MAGIC_NUMBER = 4096;
    const int nblock = (MAGIC_NUMBER + LEN - 1) / LEN;
    dim3 block_grid(nloc, nblock);
    dim3 thread_grid(1, LEN);
    format_nlist_fill_a_se_a<<<block_grid, thread_grid>>> (coord, type, jrange, jlist, rcut_r, key, i_idx, MAGIC_NUMBER);

    const int ITEMS_PER_THREAD = 16;
    const int BLOCK_THREADS = MAGIC_NUMBER / ITEMS_PER_THREAD;
    BlockSortKernel<int_64, BLOCK_THREADS, ITEMS_PER_THREAD> <<<nloc, BLOCK_THREADS>>> (key, key + nloc * MAGIC_NUMBER);
}

void DescrptSeAGPUExecuteLauncher(const float * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const float * avg, const float * std, float * descript, float * descript_deriv, float * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int MAGIC_NUMBER)
{   
    const int LEN = 256;
    int nblock = (nloc + LEN -1) / LEN;
    int * sec_a_dev = array_int;
    int * nei_iter = array_int + sec_a.size(); // = new int[sec_a_size];
    int * i_idx = array_int + sec_a.size() + nloc * sec_a.size();
    int_64 * key = array_longlong;
    
    cudaError_t res = cudaSuccess;
    res = cudaMemcpy(sec_a_dev, &sec_a[0], sizeof(int) * sec_a.size(), cudaMemcpyHostToDevice); cudaErrcheck(res);    
    res = cudaMemset(key, 0xffffffff, sizeof(int_64) * nloc * MAGIC_NUMBER); cudaErrcheck(res);
    res = cudaMemset(nlist, -1, sizeof(int) * nloc * nnei); cudaErrcheck(res);
    res = cudaMemset(descript, 0.0, sizeof(float) * nloc * ndescrpt); cudaErrcheck(res);
    res = cudaMemset(descript_deriv, 0.0, sizeof(float) * nloc * ndescrpt * 3); cudaErrcheck(res);

    if (fill_nei_a) {
        // ~~~
        // cudaProfilerStart();
        get_i_idx_se_a<<<nblock, LEN>>> (nloc, ilist, i_idx);

        if (nnei <= 256) {
            format_nbor_list_256 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        } else if (nnei <= 512) {
            format_nbor_list_512 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        } else if (nnei <= 1024) {
            format_nbor_list_1024 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        } else if (nnei <= 2048) {
            format_nbor_list_2048 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        } else if (nnei <= 4096) {
            format_nbor_list_4096 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        } 

        format_nlist_fill_b_se_a<<<nblock, LEN>>> (nlist, nnei, nloc, jrange, jlist, key, sec_a_dev, sec_a.size(), nei_iter, MAGIC_NUMBER);
    }

    const int nblock_ = (sec_a.back() + LEN -1) / LEN;
    dim3 block_grid(nloc, nblock_);
    dim3 thread_grid(1, LEN);
    compute_descriptor_se_a<<<block_grid, thread_grid>>> (descript, ndescrpt, descript_deriv, ndescrpt * 3, rij, nnei * 3, type, avg, std, nlist, nnei, coord, rcut_r_smth, rcut_r, sec_a.back());
}

void DescrptSeAGPUExecuteLauncher(const double * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const double * avg, const double * std, double * descript, double * descript_deriv, double * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int MAGIC_NUMBER)
{   
    const int LEN = 256;
    int nblock = (nloc + LEN -1) / LEN;
    int * sec_a_dev = array_int;
    int * nei_iter = array_int + sec_a.size(); // = new int[sec_a_size];
    int * i_idx = array_int + sec_a.size() + nloc * sec_a.size();
    int_64 * key = array_longlong;
    
    cudaError_t res = cudaSuccess;
    res = cudaMemcpy(sec_a_dev, &sec_a[0], sizeof(int) * sec_a.size(), cudaMemcpyHostToDevice); cudaErrcheck(res);    
    res = cudaMemset(key, 0xffffffff, sizeof(int_64) * nloc * MAGIC_NUMBER); cudaErrcheck(res);
    res = cudaMemset(nlist, -1, sizeof(int) * nloc * nnei); cudaErrcheck(res);
    res = cudaMemset(descript, 0.0, sizeof(double) * nloc * ndescrpt); cudaErrcheck(res);
    res = cudaMemset(descript_deriv, 0.0, sizeof(double) * nloc * ndescrpt * 3); cudaErrcheck(res);

    if (fill_nei_a) {
        // ~~~
        // cudaProfilerStart();
        get_i_idx_se_a<<<nblock, LEN>>> (nloc, ilist, i_idx);

        if (nnei <= 256) {
            format_nbor_list_256 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        }  else if (nnei <= 512) {
            format_nbor_list_512 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        } else if (nnei <= 1024) {
            format_nbor_list_1024 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        } else if (nnei <= 2048) {
            format_nbor_list_2048 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        } else if (nnei <= 4096) {
            format_nbor_list_4096 (coord, type, jrange, jlist, nloc, rcut_r, i_idx, key); 
        }

        format_nlist_fill_b_se_a<<<nblock, LEN>>> (nlist, nnei, nloc, jrange, jlist, key, sec_a_dev, sec_a.size(), nei_iter, MAGIC_NUMBER);
    }

    const int nblock_ = (sec_a.back() + LEN -1) / LEN;
    dim3 block_grid(nloc, nblock_);
    dim3 thread_grid(1, LEN);
    compute_descriptor_se_a<<<block_grid, thread_grid>>> (descript, ndescrpt, descript_deriv, ndescrpt * 3, rij, nnei * 3, type, avg, std, nlist, nnei, coord, rcut_r_smth, rcut_r, sec_a.back());

    cudaErrcheck(cudaGetLastError());

    #if GPU_DEBUG
    #include<fstream>
    std::fstream fout1("nlist_gpu.txt", std::ios::out);
    int * nlist_host = (int *)malloc(sizeof(int) * nloc * nnei);
    cudaErrcheck(cudaMemcpy(nlist_host, nlist, sizeof(int) * nloc * nnei, cudaMemcpyDeviceToHost));
    fout1 << "nlist info:\t" << "height:\t" << nloc << "\twidth:\t" << nnei << std::endl;
    for (int ii = 0; ii < nloc; ii++) {
        for (int jj = 0; jj < nnei; jj++) {
            fout1 << nlist_host[ii * nnei + jj] << "\t";
        }
        fout1 << std::endl;
    }
    fout1.close();

    std::fstream fout2("rij_gpu.txt", std::ios::out);
    double * rij_host = (double *)malloc(sizeof(double) * nloc * nnei * 3);
    cudaErrcheck(cudaMemcpy(rij_host, rij, sizeof(double) * nloc * nnei * 3, cudaMemcpyDeviceToHost));
    fout2 << "rij info:\t" << "height:\t" << nloc << "\twidth:\t" << nnei * 3 << std::endl;
    for (int ii = 0; ii < nloc; ii++) {
        for (int jj = 0; jj < nnei * 3; jj++) {
            fout2 << rij_host[ii * nnei * 3 + jj] << "\t";
        }
        fout2 << std::endl;
    }
    fout2.close();

    std::fstream fout3("descrpt_gpu.txt", std::ios::out);
    double * descrpt_host = (double *)malloc(sizeof(double) * nloc * ndescrpt);
    cudaErrcheck(cudaMemcpy(descrpt_host, descript, sizeof(double) * nloc * ndescrpt, cudaMemcpyDeviceToHost));
    fout3 << "descrpt info:\t" << "height:\t" << nloc << "\twidth:\t" << ndescrpt << std::endl;
    for (int ii = 0; ii < nloc; ii++) {
        for (int jj = 0; jj < ndescrpt; jj++) {
            fout3 << descrpt_host[ii * ndescrpt + jj] << "\t";
        }
        fout3 << std::endl;
    }
    fout3.close();


    std::fstream fout4("descrpt_deriv_gpu.txt", std::ios::out);
    double * descrpt_deriv_host = (double *)malloc(sizeof(double) * nloc * ndescrpt * 3);
    cudaErrcheck(cudaMemcpy(descrpt_deriv_host, descript_deriv, sizeof(double) * nloc * ndescrpt * 3, cudaMemcpyDeviceToHost));
    fout4 << "descrpt_deriv info:\t" << "height:\t" << nloc << "\twidth:\t" << ndescrpt * 3 << std::endl;
    for (int ii = 0; ii < nloc; ii++) {
        for (int jj = 0; jj < ndescrpt * 3; jj++) {
            fout4 << descrpt_deriv_host[ii * ndescrpt * 3 + jj] << "\t";
        }
        fout4 << std::endl;
    }
    fout4.close();
    #endif // GPU_DEBUG
}