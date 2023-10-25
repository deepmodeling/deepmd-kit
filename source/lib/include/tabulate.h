// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

namespace deepmd {

template <typename FPTYPE>
void tabulate_fusion_se_a_cpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em_x,
                              const FPTYPE* em,
                              const FPTYPE* two_embed,
                              const int nloc,
                              const int nnei,
                              const int last_layer_size,
                              const bool is_sorted = true);

template <typename FPTYPE>
void tabulate_fusion_se_a_grad_cpu(FPTYPE* dy_dem_x,
                                   FPTYPE* dy_dem,
                                   FPTYPE* dy_dtwo,
                                   const FPTYPE* table,
                                   const FPTYPE* table_info,
                                   const FPTYPE* em_x,
                                   const FPTYPE* em,
                                   const FPTYPE* two_embed,
                                   const FPTYPE* dy,
                                   const int nloc,
                                   const int nnei,
                                   const int last_layer_size,
                                   const bool is_sorted = true);

template <typename FPTYPE>
void tabulate_fusion_se_a_grad_grad_cpu(FPTYPE* dz_dy,
                                        const FPTYPE* table,
                                        const FPTYPE* table_info,
                                        const FPTYPE* em_x,
                                        const FPTYPE* em,
                                        const FPTYPE* two_embed,
                                        const FPTYPE* dz_dy_dem_x,
                                        const FPTYPE* dz_dy_dem,
                                        const FPTYPE* dz_dy_dtwo,
                                        const int nloc,
                                        const int nnei,
                                        const int last_layer_size,
                                        const bool is_sorted = true);

template <typename FPTYPE>
void tabulate_fusion_se_t_cpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em_x,
                              const FPTYPE* em,
                              const int nloc,
                              const int nnei_i,
                              const int nnei_j,
                              const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_t_grad_cpu(FPTYPE* dy_dem_x,
                                   FPTYPE* dy_dem,
                                   const FPTYPE* table,
                                   const FPTYPE* table_info,
                                   const FPTYPE* em_x,
                                   const FPTYPE* em,
                                   const FPTYPE* dy,
                                   const int nloc,
                                   const int nnei_i,
                                   const int nnei_j,
                                   const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_t_grad_grad_cpu(FPTYPE* dz_dy,
                                        const FPTYPE* table,
                                        const FPTYPE* table_info,
                                        const FPTYPE* em_x,
                                        const FPTYPE* em,
                                        const FPTYPE* dz_dy_dem_x,
                                        const FPTYPE* dz_dy_dem,
                                        const int nloc,
                                        const int nnei_i,
                                        const int nnei_j,
                                        const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_r_cpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em,
                              const int nloc,
                              const int nnei,
                              const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_r_grad_cpu(FPTYPE* dy_dem,
                                   const FPTYPE* table,
                                   const FPTYPE* table_info,
                                   const FPTYPE* em,
                                   const FPTYPE* dy,
                                   const int nloc,
                                   const int nnei,
                                   const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_r_grad_grad_cpu(FPTYPE* dz_dy,
                                        const FPTYPE* table,
                                        const FPTYPE* table_info,
                                        const FPTYPE* em,
                                        const FPTYPE* dz_dy_dem,
                                        const int nloc,
                                        const int nnei,
                                        const int last_layer_size);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename FPTYPE>
void tabulate_fusion_se_a_gpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em_x,
                              const FPTYPE* em,
                              const FPTYPE* two_embed,
                              const int nloc,
                              const int nnei,
                              const int last_layer_size,
                              const bool is_sorted = true);

template <typename FPTYPE>
void tabulate_fusion_se_a_grad_gpu(FPTYPE* dy_dem_x,
                                   FPTYPE* dy_dem,
                                   FPTYPE* dy_dtwo,
                                   const FPTYPE* table,
                                   const FPTYPE* table_info,
                                   const FPTYPE* em_x,
                                   const FPTYPE* em,
                                   const FPTYPE* two_embed,
                                   const FPTYPE* dy,
                                   const int nloc,
                                   const int nnei,
                                   const int last_layer_size,
                                   const bool is_sorted = true);

template <typename FPTYPE>
void tabulate_fusion_se_a_grad_grad_gpu(FPTYPE* dz_dy,
                                        const FPTYPE* table,
                                        const FPTYPE* table_info,
                                        const FPTYPE* em_x,
                                        const FPTYPE* em,
                                        const FPTYPE* two_embed,
                                        const FPTYPE* dz_dy_dem_x,
                                        const FPTYPE* dz_dy_dem,
                                        const FPTYPE* dz_dy_dtwo,
                                        const int nloc,
                                        const int nnei,
                                        const int last_layer_size,
                                        const bool is_sorted = true);

template <typename FPTYPE>
void tabulate_fusion_se_t_gpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em_x,
                              const FPTYPE* em,
                              const int nloc,
                              const int nnei_i,
                              const int nnei_j,
                              const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_t_grad_gpu(FPTYPE* dy_dem_x,
                                   FPTYPE* dy_dem,
                                   const FPTYPE* table,
                                   const FPTYPE* table_info,
                                   const FPTYPE* em_x,
                                   const FPTYPE* em,
                                   const FPTYPE* dy,
                                   const int nloc,
                                   const int nnei_i,
                                   const int nnei_j,
                                   const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_t_grad_grad_gpu(FPTYPE* dz_dy,
                                        const FPTYPE* table,
                                        const FPTYPE* table_info,
                                        const FPTYPE* em_x,
                                        const FPTYPE* em,
                                        const FPTYPE* dz_dy_dem_x,
                                        const FPTYPE* dz_dy_dem,
                                        const int nloc,
                                        const int nnei_i,
                                        const int nnei_j,
                                        const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_r_gpu(FPTYPE* out,
                              const FPTYPE* table,
                              const FPTYPE* table_info,
                              const FPTYPE* em,
                              const int nloc,
                              const int nnei,
                              const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_r_grad_gpu(FPTYPE* dy_dem,
                                   const FPTYPE* table,
                                   const FPTYPE* table_info,
                                   const FPTYPE* em,
                                   const FPTYPE* dy,
                                   const int nloc,
                                   const int nnei,
                                   const int last_layer_size);

template <typename FPTYPE>
void tabulate_fusion_se_r_grad_grad_gpu(FPTYPE* dz_dy,
                                        const FPTYPE* table,
                                        const FPTYPE* table_info,
                                        const FPTYPE* em,
                                        const FPTYPE* dz_dy_dem,
                                        const int nloc,
                                        const int nnei,
                                        const int last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace deepmd
