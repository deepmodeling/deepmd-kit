#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "DeviceFunctor.h"
#include "gpu_nv.h"

#define MM 4
#define KK 4
#define TPB 256
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

template <typename FPTYPE>
__forceinline__ 
__device__
void locate_xx(const FPTYPE& lower, const FPTYPE& upper,  const FPTYPE& max, const FPTYPE& stride0, const FPTYPE& stride1, FPTYPE& xx, int& table_idx) {
    if (xx < lower) {
        table_idx = 0;
        xx = 0;
    }
    else if (xx < upper) {
        table_idx = (int)((xx - lower) / stride0);
        xx -= (table_idx * stride0 + lower);
    }
    else if (xx < max) {
        int first_stride = int((upper - lower) / stride0);
        table_idx = first_stride + (int)((xx - upper) / stride1);
        xx -= ((table_idx - first_stride) * stride1 + upper);
    }
    else {
        table_idx = int((upper - lower) / stride0) + (int)((max - upper) / stride1) - 1;
        xx = 0;
    }
}

template <typename FPTYPE>
__forceinline__ 
__device__ 
FPTYPE dot(FPTYPE ll[4], FPTYPE rr[4]) {
    return ll[0] * rr[0] + ll[1] * rr[1] + ll[2] * rr[2] + ll[3] * rr[3];
}

template <typename FPTYPE>
__forceinline__ 
__device__
void warp_reduce(FPTYPE & val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
}

// last_layer_size must larger than MTILE * KTILE!
// TODO: A more flexible implementation of sparse 
template <
    typename FPTYPE,
    int      MTILE,
    int      KTILE> 
__global__ void tabulate_fusion(const FPTYPE * table, const FPTYPE * in, const FPTYPE * ff, FPTYPE * out, const FPTYPE lower, const FPTYPE upper, const FPTYPE max, const FPTYPE stride0, const FPTYPE stride1, const int nnei, const int last_layer_size) {
    extern __shared__ int _data[];
    int const block_idx = blockIdx.x; // nloc
    int const thread_idx = threadIdx.x; // last_layer_size
    FPTYPE ago = __shfl_sync(0xffffffff, in[block_idx * nnei + nnei - 1], 0);
    bool unloop = false;
    int breakpoint = nnei - 1;
    // int const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    // int const lane_idx = threadIdx.x % 32;
    // iteratorC for data reuse...
    FPTYPE * iteratorC = (FPTYPE*) &_data[0];
    for (int kk = 0; kk < MTILE; kk++)
        iteratorC[kk * last_layer_size + thread_idx] = 0.f;
    __syncthreads();  

    for (int ii = 0; ii < nnei; ii++) {
        FPTYPE var[4]; 
        FPTYPE xx = in[block_idx * nnei + ii];

        if (ago == xx) {
            unloop = true;
            breakpoint = ii;
        } 
        int table_idx = 0;
        locate_xx(lower, upper, max, stride0, stride1, xx, table_idx);
        var[0] = table[table_idx * last_layer_size * 4 + thread_idx * 4 + 0];
        var[1] = table[table_idx * last_layer_size * 4 + thread_idx * 4 + 1];
        var[2] = table[table_idx * last_layer_size * 4 + thread_idx * 4 + 2];
        var[3] = table[table_idx * last_layer_size * 4 + thread_idx * 4 + 3];
        FPTYPE res = ((var[0] * xx + var[1]) * xx + var[2]) * xx + var[3];
        for (int kk = 0; kk < MTILE; kk++) {
            iteratorC[kk * last_layer_size + thread_idx] += (nnei - breakpoint) * ff[block_idx * nnei * MTILE + ii * MTILE + kk] * res;
        }
        if (unloop) break;
    }
    for (int ii = 0; ii < MTILE; ii++) {
        out[block_idx * MTILE * last_layer_size + ii * last_layer_size + thread_idx] = iteratorC[ii * last_layer_size + thread_idx];
    }
}

// last_layer_size must larger than MTILE * KTILE!
// TODO: A more flexible implementation of sparse 


template <
    typename FPTYPE,
    int      MTILE,
    int      KTILE>
__global__ void tabulate_fusion_grad_warp_reduce(const FPTYPE * table, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, FPTYPE * dy_dx, FPTYPE * dy_df, const FPTYPE lower, const FPTYPE upper, const FPTYPE max, const FPTYPE stride0, const FPTYPE stride1, const int nnei, const int last_layer_size) {
    extern __shared__ int _data[];
    int const block_idx = blockIdx.x;  // nloc
    int const thread_idx = threadIdx.x; // KTILE * WARP_SIZE, usally 128 here~
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    int breakpoint = nnei - 1;
    bool unloop = false;

    FPTYPE * iteratorA = (FPTYPE *)&_data[0]; // dy
    for (int ii = 0; ii < MTILE; ii++) {
        if (thread_idx < last_layer_size) {
            iteratorA[ii * last_layer_size + thread_idx] = dy[block_idx * MTILE * last_layer_size + ii * last_layer_size + thread_idx];
        }
    }
    __syncthreads();
    FPTYPE ago = __shfl_sync(0xffffffff, in[block_idx * nnei + nnei - 1], 0);
    for (int ii = 0; ii < nnei; ii += KTILE) {
        FPTYPE xx = in[block_idx * nnei + ii + warp_idx];
        // if (ago == xx) {
        //     unloop = true;
        //     breakpoint = ii;
        // }
        
        int table_idx = 0;
        locate_xx(lower, upper, max, stride0, stride1, xx, table_idx);
        FPTYPE sum[KTILE] = {0.f};
        FPTYPE Csub = 0.f;
        for (int jj = lane_idx; jj < last_layer_size; jj += WARP_SIZE) {
            // load iteratorB through table 
            FPTYPE var[KTILE];
            var[0] = table[table_idx * last_layer_size * 4 + jj * 4 + 0];
            var[1] = table[table_idx * last_layer_size * 4 + jj * 4 + 1];
            var[2] = table[table_idx * last_layer_size * 4 + jj * 4 + 2];
            var[3] = table[table_idx * last_layer_size * 4 + jj * 4 + 3];
            FPTYPE tmp = (var[0] * xx + var[1]) * xx + var[2];
            for (int kk = 0; kk < KTILE; kk++) {
                sum[kk] += (nnei - breakpoint) * iteratorA[kk * last_layer_size + jj] * (tmp * xx + var[3]);
            }
            var[2]  = ff[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + 0] * iteratorA[0 * last_layer_size + jj];
            var[2] += ff[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + 1] * iteratorA[1 * last_layer_size + jj];
            var[2] += ff[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + 2] * iteratorA[2 * last_layer_size + jj];
            var[2] += ff[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + 3] * iteratorA[3 * last_layer_size + jj];
            Csub += (nnei - breakpoint) * ((2.0 * var[0] * xx + var[1]) * xx + tmp) * var[2];
        }
        __syncwarp();
        for (int kk = 0; kk < KTILE; kk++) {
            warp_reduce(sum[kk]);
        }
        warp_reduce(Csub);
        if (lane_idx == 0) {
            for (int kk = 0; kk < KTILE; kk++) {
                dy_df[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + kk] = sum[kk];
            }
            dy_dx[block_idx * nnei + ii + warp_idx] = Csub;
        }
        if (unloop) break;
    }
}

template <
    typename FPTYPE,
    int      MTILE,
    int      KTILE> 
__global__ void tabulate_fusion_special(const FPTYPE * table, const FPTYPE * in, const FPTYPE * ff, FPTYPE * out, const FPTYPE lower, const FPTYPE upper, const FPTYPE max, const FPTYPE stride0, const FPTYPE stride1, const int nnei, const int last_layer_size) {
    extern __shared__ int _data[];
    int const block_idx = blockIdx.x; // nloc
    int const thread_idx = threadIdx.x; // last_layer_size
    FPTYPE ago = __shfl_sync(0xffffffff, in[block_idx * nnei + nnei - 1], 0);
    bool unloop = false;
    int breakpoint = nnei - 1;

    FPTYPE * iteratorC = (FPTYPE*) &_data[0];
    for (int kk = 0; kk < MTILE; kk++)
        iteratorC[kk * last_layer_size + thread_idx] = 0.f;
    __syncthreads();
 
    for (int ii = 0; ii < nnei; ii++) {
        FPTYPE var[6]; 
        FPTYPE xx = in[block_idx * nnei + ii];
        if (xx == ago) {
            unloop = true;
            breakpoint = ii;
        }
        int table_idx = 0;
        locate_xx(lower, upper, max, stride0, stride1, xx, table_idx);
        var[0] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 0];
        var[1] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 1];
        var[2] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 2];
        var[3] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 3];
        var[4] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 4];
        var[5] = table[table_idx * last_layer_size * 6 + thread_idx * 6 + 5];
        FPTYPE res = var[0] + (var[1] + (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) * xx;
        
        for (int kk = 0; kk < MTILE; kk++) {
            iteratorC[kk * last_layer_size + thread_idx] += (nnei - breakpoint) * ff[block_idx * nnei * MTILE + ii * MTILE + kk] * res;
        }
        if (unloop) break;
    }
    for (int ii = 0; ii < MTILE; ii++) {
        out[block_idx * MTILE * last_layer_size + ii * last_layer_size + thread_idx] = iteratorC[ii * last_layer_size + thread_idx];
    }
}

template <
    typename FPTYPE,
    int      MTILE,
    int      KTILE> 
__global__ void tabulate_fusion_grad_warp_reduce_special(const FPTYPE * table, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, FPTYPE * dy_dx, FPTYPE * dy_df, const FPTYPE lower, const FPTYPE upper, const FPTYPE max, const FPTYPE stride0, const FPTYPE stride1, const int nnei, const int last_layer_size) {
    extern __shared__ int _data[];
    int const block_idx = blockIdx.x;  // nloc
    int const thread_idx = threadIdx.x; // KTILE * WARP_SIZE, usally 128 here~
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    int breakpoint = nnei - 1;
    bool unloop = false;

    FPTYPE * iteratorA = (FPTYPE *)&_data[0]; // dy
    for (int ii = 0; ii < MTILE; ii++) {
        if (thread_idx < last_layer_size) {
            iteratorA[ii * last_layer_size + thread_idx] = dy[block_idx * MTILE * last_layer_size + ii * last_layer_size + thread_idx];
        }
    }
    __syncthreads();
    FPTYPE ago = __shfl_sync(0xffffffff, in[block_idx * nnei + nnei - 1], 0);
    for (int ii = 0; ii < nnei; ii += KTILE) {
        FPTYPE xx = in[block_idx * nnei + ii + warp_idx];
        if (ago == xx) { 
            unloop = true;
            breakpoint = ii;
        }
        
        int table_idx = 0;
        locate_xx(lower, upper, max, stride0, stride1, xx, table_idx);
        FPTYPE sum[KTILE] = {0.f};
        FPTYPE Csub = 0.f;
        for (int jj = lane_idx; jj < last_layer_size; jj += WARP_SIZE) {
            FPTYPE var[6]; 
            // load iteratorB through table 
            var[0]  = table[table_idx * last_layer_size * 6 + 6 * jj + 0]; 
            var[1]  = table[table_idx * last_layer_size * 6 + 6 * jj + 1]; 
            var[2]  = table[table_idx * last_layer_size * 6 + 6 * jj + 2]; 
            var[3]  = table[table_idx * last_layer_size * 6 + 6 * jj + 3];
            var[4]  = table[table_idx * last_layer_size * 6 + 6 * jj + 4];
            var[5]  = table[table_idx * last_layer_size * 6 + 6 * jj + 5];
            FPTYPE res = var[0] + (var[1] + (var[2] + (var[3] + (var[4] + var[5] * xx) * xx) * xx) * xx) * xx;
            
            for (int kk = 0; kk < KTILE; kk++) {
                sum[kk] += (nnei - breakpoint) * iteratorA[kk * last_layer_size + jj] * res;
            }
            res  = ff[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + 0] * iteratorA[0 * last_layer_size + jj];
            res += ff[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + 1] * iteratorA[1 * last_layer_size + jj];
            res += ff[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + 2] * iteratorA[2 * last_layer_size + jj];
            res += ff[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + 3] * iteratorA[3 * last_layer_size + jj];
            Csub += (nnei - breakpoint) * (var[1] + (2 * var[2] + (3 * var[3] + (4 * var[4] + 5 * var[5] * xx) * xx) * xx) * xx) * res;
        }
        __syncwarp();
        for (int kk = 0; kk < KTILE; kk++) {
            warp_reduce(sum[kk]);
        }
        warp_reduce(Csub);
        if (lane_idx == 0) {
            for (int kk = 0; kk < KTILE; kk++) {
                dy_df[block_idx * nnei * MTILE + (ii + warp_idx) * 4 + kk] = sum[kk];
            }
            dy_dx[block_idx * nnei + ii + warp_idx] = Csub;
        }
        if (unloop) break;
    }
}

template <typename FPTYPE,
          int      THREADS_PER_BLOCK>
__global__ void tabulate_checker(const FPTYPE * in, int * out, const FPTYPE lower, const FPTYPE upper, const FPTYPE max, const int nloc, const int nnei) {
    __shared__ int Csub[THREADS_PER_BLOCK];
    __shared__ int Dsub[THREADS_PER_BLOCK];
    int const bid = blockIdx.x;
    int const tid = threadIdx.x;
    
    Csub[tid] = 0;
    Dsub[tid] = 0;
    __syncthreads();

    for (int ii = tid; ii < nnei; ii += THREADS_PER_BLOCK) {
        FPTYPE xx = in[bid * nnei + ii];
        if (xx < lower || xx > max) {
            Csub[tid] += 1;
            // printf("# DEEPMD: level 2 overflow, xx:\t%f\n", xx);
        }
        else if (xx >= upper && xx <= max) {
            Dsub[tid] += 1;
            // printf("# DEEPMD: level 1 overflow, xx:\t%f\n", xx);
        }
    }
    __syncthreads();
    // do reduction in shared memory
    for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
        if (tid < ii) {
            Csub[tid] += Csub[tid + ii];
            Dsub[tid] += Dsub[tid + ii];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[bid] = Csub[0];
        out[nloc + bid] = Dsub[0];
    }
}

void TabulateFusionLauncher(const double * table, const double * table_info, const double * in, const double * ff, const int nloc, const int nnei, const int last_layer_size, double * out) {
    // std::cout << "I'm in tabulate GPU!" << std::endl;
    tabulate_fusion_special<double, MM, KK> <<<nloc, last_layer_size, sizeof(double) * MM * last_layer_size>>>(table, in, ff, out, table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
}
void TabulateFusionLauncher(const float * table, const float * table_info, const float * in, const float * ff, const int nloc, const int nnei, const int last_layer_size, float * out) {
    tabulate_fusion_special<float, MM, KK> <<<nloc, last_layer_size, sizeof(float) * MM * last_layer_size>>>(table, in, ff, out, table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
}

void TabulateFusionGradLauncher(const double * table, const double * table_info, const double * in, const double * ff, const double * dy, const int nloc, const int nnei, const int last_layer_size, double * dy_dx, double * dy_df) {
    // cudaMemset(dy_df, 0.0, sizeof(double) * nloc * nnei * 4);
    cudaMemset(dy_dx, 0.0, sizeof(double) * nloc * nnei);
    cudaMemset(dy_df, 0.0, sizeof(double) * nloc * nnei * 4);
    tabulate_fusion_grad_warp_reduce_special<double, MM, KK> <<<nloc, KK * WARP_SIZE, sizeof(double) * MM * last_layer_size>>>(table, in, ff, dy, dy_dx, dy_df, table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
}
void TabulateFusionGradLauncher(const float * table, const float * table_info, const float * in, const float * ff, const float * dy, const int nloc, const int nnei, const int last_layer_size, float * dy_dx, float * dy_df) {
    // cudaMemset(dy_df, 0.0, sizeof(float) * nloc * nnei * 4);
    cudaMemset(dy_dx, 0.0, sizeof(float) * nloc * nnei);
    cudaMemset(dy_df, 0.0, sizeof(float) * nloc * nnei * 4);
    tabulate_fusion_grad_warp_reduce_special<float, MM, KK> <<<nloc, KK * WARP_SIZE, sizeof(float) * MM * last_layer_size>>>(table, in, ff, dy, dy_dx, dy_df, table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
}

void TabulateCheckerLauncher(const double * table_info, const double * in, int * out, const int nloc, const int nnei) {
    tabulate_checker <double, TPB> <<<nloc, TPB>>>(in, out, table_info[0], table_info[1], table_info[2], nloc, nnei);
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int * d_out = NULL;
    int * h_out = NULL;
    cudaMalloc((void **)&d_out, sizeof(int));
    h_out = (int*)malloc(sizeof(int));
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out, d_out, nloc);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out, d_out, nloc);

    // d_out <-- [38]
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    if(h_out[0] > 0) {
        std::cout << "# DEEPMD: warning! some values [" << h_out[0] << "/" << nloc * nnei << "] overflow the range of the table, using the endpoint approximate processing.." << std::endl;
    }

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out + nloc, d_out, nloc);
    
    // d_out <-- [38]
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    if(h_out[0] > 0) {
        std::cout << "# DEEPMD: warning! some values [" << h_out[0] << "/" << nloc * nnei << "] overflow the range of the table, using second table approximate processing.." << std::endl;
    }

    // free the temperary storage
    cudaFree(d_out);
    cudaFree(d_temp_storage);
    free(h_out);
}

void TabulateCheckerLauncher(const float * table_info, const float * in, int * out, const int nloc, const int nnei) {
    tabulate_checker <float, TPB> <<<nloc, TPB>>>(in, out, table_info[0], table_info[1], table_info[2], nloc, nnei);
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int * d_out = NULL;
    int * h_out = NULL;
    cudaMalloc((void **)&d_out, sizeof(int));
    h_out = (int*)malloc(sizeof(int));
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out, d_out, nloc);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out, d_out, nloc);

    // d_out <-- [38]
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if(h_out[0] > 0) {
        std::cout << "# DEEPMD: warning! some values [" << h_out[0] << "/" << nloc * nnei << "] overflow the range of the table, using the endpoint approximate processing.." << std::endl;
    }

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out + nloc, d_out, nloc);
    
    // d_out <-- [38]
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    if(h_out[0] > 0) {
        std::cout << "# DEEPMD: warning! some values [" << h_out[0] << "/" << nloc * nnei << "] overflow the range of the table, using second table approximate processing.." << std::endl;
    }

    // free the temperary storage
    cudaFree(d_out);
    cudaFree(d_temp_storage);
    free(h_out);
}

template<typename FPTYPE>
void TabulateFusionGPUExecuteFunctor<FPTYPE>::operator()(const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const int nloc, const int nnei, const int last_layer_size, FPTYPE * out) {
    tabulate_fusion_special<FPTYPE, MM, KK> <<<nloc, last_layer_size, sizeof(FPTYPE) * MM * last_layer_size>>>(table, in, ff, out, table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
}

template<typename FPTYPE>
void TabulateFusionGradGPUExecuteFunctor<FPTYPE>::operator()(const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, const int nloc, const int nnei, const int last_layer_size, FPTYPE * dy_dx, FPTYPE * dy_df) {
    cudaErrcheck(cudaMemset(dy_dx, 0.0, sizeof(FPTYPE) * nloc * nnei));
    cudaErrcheck(cudaMemset(dy_df, 0.0, sizeof(FPTYPE) * nloc * nnei * 4));
    tabulate_fusion_grad_warp_reduce_special<FPTYPE, MM, KK> <<<nloc, KK * WARP_SIZE, sizeof(FPTYPE) * MM * last_layer_size>>>(table, in, ff, dy, dy_dx, dy_df, table_info[0], table_info[1], table_info[2], table_info[3], table_info[4], nnei, last_layer_size);
}

template <typename FPTYPE>
void TabulateCheckerGPUExecuteFunctor<FPTYPE>::operator()(const FPTYPE * table_info, const FPTYPE * in, int * out, const int nloc, const int nnei) {
    tabulate_checker <FPTYPE, TPB> <<<nloc, TPB>>>(in, out, table_info[0], table_info[1], table_info[2], nloc, nnei);
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int * d_out = NULL;
    int * h_out = NULL;
    cudaMalloc((void **)&d_out, sizeof(int));
    h_out = (int*)malloc(sizeof(int));
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out, d_out, nloc);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out, d_out, nloc);

    // d_out <-- [38]
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if(h_out[0] > 0) {
        std::cout << "# DEEPMD: warning! some values [" << h_out[0] << "/" << nloc * nnei << "] overflow the range of the table, using the endpoint approximate processing.." << std::endl;
    }

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, out + nloc, d_out, nloc);
    
    // d_out <-- [38]
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    if(h_out[0] > 0) {
        std::cout << "# DEEPMD: warning! some values [" << h_out[0] << "/" << nloc * nnei << "] overflow the range of the table, using second table approximate processing.." << std::endl;
    }

    // free the temperary storage
    cudaFree(d_out);
    cudaFree(d_temp_storage);
    free(h_out);
}

template struct TabulateFusionGPUExecuteFunctor<float>;
template struct TabulateFusionGPUExecuteFunctor<double>;
template struct TabulateFusionGradGPUExecuteFunctor<float>;
template struct TabulateFusionGradGPUExecuteFunctor<double>;
template struct TabulateCheckerGPUExecuteFunctor<float>;
template struct TabulateCheckerGPUExecuteFunctor<double>;