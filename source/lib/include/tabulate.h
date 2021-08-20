#pragma once

namespace deepmd{

template<typename FPTYPE>
void tabulate_fusion_cpu(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_cpu(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_grad_cpu(
    FPTYPE * dz_dy,
    const FPTYPE * table,
    const FPTYPE * table_info,
    const FPTYPE * em_x,
    const FPTYPE * em,
    const FPTYPE * dz_dy_dem_x,
    const FPTYPE * dz_dy_dem,
    const int nloc,
    const int nnei,
    const int last_layer_size);

#if GOOGLE_CUDA
template<typename FPTYPE>
void tabulate_fusion_gpu_cuda(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc,
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_gpu_cuda(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_grad_gpu_cuda(
    FPTYPE * dz_dy,
    const FPTYPE * table,
    const FPTYPE * table_info,
    const FPTYPE * em_x,
    const FPTYPE * em,
    const FPTYPE * dz_dy_dem_x,
    const FPTYPE * dz_dy_dem,
    const int nloc,
    const int nnei,
    const int last_layer_size);
#endif // GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
template<typename FPTYPE>
void tabulate_fusion_gpu_rocm(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc,
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_gpu_rocm(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_grad_gpu_rocm(
    FPTYPE * dz_dy,
    const FPTYPE * table,
    const FPTYPE * table_info,
    const FPTYPE * em_x,
    const FPTYPE * em,
    const FPTYPE * dz_dy_dem_x,
    const FPTYPE * dz_dy_dem,
    const int nloc,
    const int nnei,
    const int last_layer_size);
#endif // TENSORFLOW_USE_ROCM
}

