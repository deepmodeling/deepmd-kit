// SPDX-License-Identifier: LGPL-3.0-or-later
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "tabulate.h"
#include "tabulate_validation.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "device.h"
#endif

void GetTensorDevice(const torch::Tensor& t, std::string& str) {
  if (t.device().is_cuda()) {
    str = "GPU";
  } else {
    str = "CPU";
  }
}

void CheckTabulateDataTensor(const torch::Tensor& tensor,
                             const torch::Tensor& table_tensor,
                             const char* name) {
  TORCH_CHECK(tensor.scalar_type() == table_tensor.scalar_type(), name,
              " must have the same dtype as table");
  TORCH_CHECK(tensor.device() == table_tensor.device(), name,
              " must be on the same device as table");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template <typename FPTYPE>
void CheckTabulateTable(const torch::Tensor& table_tensor,
                        const torch::Tensor& table_info_tensor,
                        const int64_t last_layer_size,
                        const bool symmetric_range) {
  TORCH_CHECK(table_tensor.dim() == 2, "table must be rank 2");
  TORCH_CHECK(table_tensor.scalar_type() == torch::kFloat ||
                  table_tensor.scalar_type() == torch::kDouble,
              "table must use float32 or float64");
  TORCH_CHECK(table_tensor.device().is_cpu() || table_tensor.device().is_cuda(),
              "table must be on a CPU or CUDA/ROCm device");
  TORCH_CHECK(table_tensor.is_contiguous(), "table must be contiguous");
  TORCH_CHECK(last_layer_size > 0, "last_layer_size must be positive");
  TORCH_CHECK(table_info_tensor.device().is_cpu(),
              "table_info must be on the CPU");
  TORCH_CHECK(table_info_tensor.scalar_type() == table_tensor.scalar_type(),
              "table_info must have the same dtype as table");
  TORCH_CHECK(table_info_tensor.is_contiguous(),
              "table_info must be contiguous");
  TORCH_CHECK(table_info_tensor.numel() >= 5,
              "table_info must contain at least 5 values");

  int64_t required_rows = 0;
  std::string error;
  TORCH_CHECK(deepmd::tabulate_required_table_rows<FPTYPE>(
                  table_info_tensor.data_ptr<FPTYPE>(), symmetric_range,
                  required_rows, error),
              error);
  int64_t required_elements = 0;
  TORCH_CHECK(deepmd::tabulate_required_table_elements(
                  required_rows, last_layer_size, required_elements, error),
              error);
  TORCH_CHECK(table_tensor.numel() >= required_elements,
              "table does not contain enough coefficients for table_info and "
              "last_layer_size");
}

template <typename FPTYPE>
void CheckTabulateSeAInputs(const torch::Tensor& table_tensor,
                            const torch::Tensor& table_info_tensor,
                            const torch::Tensor& em_x_tensor,
                            const torch::Tensor& em_tensor,
                            const torch::Tensor& two_embed_tensor,
                            const int64_t last_layer_size) {
  CheckTabulateTable<FPTYPE>(table_tensor, table_info_tensor, last_layer_size,
                             false);
  TORCH_CHECK(em_tensor.dim() == 3 && em_tensor.size(2) == 4,
              "em must have shape [nloc, nnei, 4]");
  const int64_t neighbor_count = em_tensor.numel() / 4;
  TORCH_CHECK(em_x_tensor.dim() == 2 && em_x_tensor.numel() == neighbor_count,
              "em_x must be rank 2 and contain nloc * nnei values");
  CheckTabulateDataTensor(em_x_tensor, table_tensor, "em_x");
  CheckTabulateDataTensor(em_tensor, table_tensor, "em");
  if (two_embed_tensor.defined()) {
    TORCH_CHECK(two_embed_tensor.dim() == 2, "two_embed must be rank 2");
    int64_t expected_two_embed_elements = 0;
    TORCH_CHECK(
        deepmd::tabulate_checked_product(neighbor_count, last_layer_size,
                                         expected_two_embed_elements),
        "two_embed element count exceeds the supported integer range");
    TORCH_CHECK(two_embed_tensor.numel() == expected_two_embed_elements,
                "two_embed must contain nloc * nnei * last_layer_size values");
    CheckTabulateDataTensor(two_embed_tensor, table_tensor, "two_embed");
  }
}

template <typename FPTYPE>
void CheckTabulateSeTInputs(const torch::Tensor& table_tensor,
                            const torch::Tensor& table_info_tensor,
                            const torch::Tensor& em_x_tensor,
                            const torch::Tensor& em_tensor,
                            const int64_t last_layer_size) {
  CheckTabulateTable<FPTYPE>(table_tensor, table_info_tensor, last_layer_size,
                             true);
  TORCH_CHECK(em_tensor.dim() == 3, "em must be rank 3");
  TORCH_CHECK(
      em_x_tensor.dim() == 2 && em_x_tensor.numel() == em_tensor.numel(),
      "em_x must be rank 2 and contain the same number of values as em");
  CheckTabulateDataTensor(em_x_tensor, table_tensor, "em_x");
  CheckTabulateDataTensor(em_tensor, table_tensor, "em");
}

template <typename FPTYPE>
void CheckTabulateSeRInputs(const torch::Tensor& table_tensor,
                            const torch::Tensor& table_info_tensor,
                            const torch::Tensor& em_tensor,
                            const int64_t last_layer_size) {
  CheckTabulateTable<FPTYPE>(table_tensor, table_info_tensor, last_layer_size,
                             false);
  TORCH_CHECK(em_tensor.dim() == 2, "em must be rank 2");
  CheckTabulateDataTensor(em_tensor, table_tensor, "em");
}

template <typename FPTYPE>
void TabulateFusionSeAForward(const torch::Tensor& table_tensor,
                              const torch::Tensor& table_info_tensor,
                              const torch::Tensor& em_x_tensor,
                              const torch::Tensor& em_tensor,
                              const torch::Tensor& two_embed_tensor,
                              int64_t last_layer_size,
                              bool is_sorted,
                              torch::Tensor& descriptor_tensor) {
  // check input shape
  if (table_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of table should be 2");
  }
  if (em_x_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of input should be 2");
  }
  if (em_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of input should be 3");
  }
  if (two_embed_tensor.defined() && two_embed_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of input should be 2");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* descriptor = descriptor_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* two_embed =
      (!two_embed_tensor.defined())
          ? nullptr
          : two_embed_tensor.view({-1}).data_ptr<FPTYPE>();

  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_a_gpu(descriptor, table, table_info, em_x, em,
                                     two_embed, nloc, nnei, last_layer_size,
                                     is_sorted);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_a_cpu(descriptor, table, table_info, em_x, em,
                                     two_embed, nloc, nnei, last_layer_size,
                                     is_sorted);
  }
}

template <typename FPTYPE>
void TabulateFusionSeAGradForward(const torch::Tensor& table_tensor,
                                  const torch::Tensor& table_info_tensor,
                                  const torch::Tensor& em_x_tensor,
                                  const torch::Tensor& em_tensor,
                                  const torch::Tensor& two_embed_tensor,
                                  const torch::Tensor& dy_tensor,
                                  const torch::Tensor& descriptor_tensor,
                                  bool is_sorted,
                                  torch::Tensor& dy_dem_x_tensor,
                                  torch::Tensor& dy_dem_tensor,
                                  torch::Tensor& dy_dtwo_tensor) {
  // check input shape
  if (dy_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of dy_tensor should be 3");
  }
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dy_dem_x = dy_dem_x_tensor.view({-1}).data_ptr<FPTYPE>();
  FPTYPE* dy_dem = dy_dem_tensor.view({-1}).data_ptr<FPTYPE>();
  FPTYPE* dy_dtwo = (!dy_dtwo_tensor.defined())
                        ? nullptr
                        : dy_dtwo_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* two_embed =
      (!two_embed_tensor.defined())
          ? nullptr
          : two_embed_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dy = dy_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  const int64_t last_layer_size = descriptor_tensor.size(2);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_a_grad_gpu(
        dy_dem_x, dy_dem, dy_dtwo, table, table_info, em_x, em, two_embed, dy,
        nloc, nnei, last_layer_size, is_sorted);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_a_grad_cpu(
        dy_dem_x, dy_dem, dy_dtwo, table, table_info, em_x, em, two_embed, dy,
        nloc, nnei, last_layer_size, is_sorted);
  }
}

template <typename FPTYPE>
void TabulateFusionSeAGradGradForward(const torch::Tensor& table_tensor,
                                      const torch::Tensor& table_info_tensor,
                                      const torch::Tensor& em_x_tensor,
                                      const torch::Tensor& em_tensor,
                                      const torch::Tensor& two_embed_tensor,
                                      const torch::Tensor& dz_dy_dem_x_tensor,
                                      const torch::Tensor& dz_dy_dem_tensor,
                                      const torch::Tensor& dz_dy_dtwo_tensor,
                                      const torch::Tensor& descriptor_tensor,
                                      bool is_sorted,
                                      torch::Tensor& dz_dy_tensor) {
  // Check input shape
  if (dz_dy_dem_x_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of dz_dy_dem_x should be 2");
  }
  if (dz_dy_dem_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of dz_dy_dem should be 3");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dz_dy = dz_dy_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* two_embed =
      (!two_embed_tensor.defined())
          ? nullptr
          : two_embed_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dem_x = dz_dy_dem_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dem = dz_dy_dem_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dtwo =
      (!dz_dy_dtwo_tensor.defined())
          ? nullptr
          : dz_dy_dtwo_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  const int64_t last_layer_size = descriptor_tensor.size(2);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_a_grad_grad_gpu(
        dz_dy, table, table_info, em_x, em, two_embed, dz_dy_dem_x, dz_dy_dem,
        dz_dy_dtwo, nloc, nnei, last_layer_size, is_sorted);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    TORCH_CHECK(last_layer_size <= 1024,
                "In the process of model compression, the size of the "
                "last layer of embedding net must be less than 1024!");
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_a_grad_grad_cpu(
        dz_dy, table, table_info, em_x, em, two_embed, dz_dy_dem_x, dz_dy_dem,
        dz_dy_dtwo, nloc, nnei, last_layer_size, is_sorted);
  }
}

template <typename FPTYPE>
void TabulateFusionSeTForward(const torch::Tensor& table_tensor,
                              const torch::Tensor& table_info_tensor,
                              const torch::Tensor& em_x_tensor,
                              const torch::Tensor& em_tensor,
                              int64_t last_layer_size,
                              torch::Tensor& descriptor_tensor) {
  // check input shape
  if (table_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of table should be 2");
  }
  if (em_x_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of input should be 2");
  }
  if (em_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of input should be 3");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* descriptor = descriptor_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei_i = em_tensor.size(1);
  const int64_t nnei_j = em_tensor.size(2);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_t_gpu(descriptor, table, table_info, em_x, em,
                                     nloc, nnei_i, nnei_j, last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_t_cpu(descriptor, table, table_info, em_x, em,
                                     nloc, nnei_i, nnei_j, last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeTGradForward(const torch::Tensor& table_tensor,
                                  const torch::Tensor& table_info_tensor,
                                  const torch::Tensor& em_x_tensor,
                                  const torch::Tensor& em_tensor,
                                  const torch::Tensor& dy_tensor,
                                  const torch::Tensor& descriptor_tensor,
                                  torch::Tensor& dy_dem_x_tensor,
                                  torch::Tensor& dy_dem_tensor) {
  // check input shape
  if (dy_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of dy_tensor should be 2");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dy_dem_x = dy_dem_x_tensor.view({-1}).data_ptr<FPTYPE>();
  FPTYPE* dy_dem = dy_dem_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dy = dy_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei_i = em_tensor.size(1);
  const int64_t nnei_j = em_tensor.size(2);
  const int64_t last_layer_size = descriptor_tensor.size(1);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_t_grad_gpu(dy_dem_x, dy_dem, table, table_info,
                                          em_x, em, dy, nloc, nnei_i, nnei_j,
                                          last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_t_grad_cpu(dy_dem_x, dy_dem, table, table_info,
                                          em_x, em, dy, nloc, nnei_i, nnei_j,
                                          last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeTGradGradForward(const torch::Tensor& table_tensor,
                                      const torch::Tensor& table_info_tensor,
                                      const torch::Tensor& em_x_tensor,
                                      const torch::Tensor& em_tensor,
                                      const torch::Tensor& dz_dy_dem_x_tensor,
                                      const torch::Tensor& dz_dy_dem_tensor,
                                      const torch::Tensor& descriptor_tensor,
                                      torch::Tensor& dz_dy_tensor) {
  // Check input shape
  if (dz_dy_dem_x_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of dz_dy_dem_x should be 2");
  }
  if (dz_dy_dem_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of dz_dy_dem should be 3");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dz_dy = dz_dy_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dem_x = dz_dy_dem_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dem = dz_dy_dem_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei_i = em_tensor.size(1);
  const int64_t nnei_j = em_tensor.size(2);
  const int64_t last_layer_size = descriptor_tensor.size(1);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_t_grad_grad_gpu(dz_dy, table, table_info, em_x,
                                               em, dz_dy_dem_x, dz_dy_dem, nloc,
                                               nnei_i, nnei_j, last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    TORCH_CHECK(last_layer_size <= 1024,
                "In the process of model compression, the size of the "
                "last layer of embedding net must be less than 1024!");
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_t_grad_grad_cpu(dz_dy, table, table_info, em_x,
                                               em, dz_dy_dem_x, dz_dy_dem, nloc,
                                               nnei_i, nnei_j, last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeTTebdForward(const torch::Tensor& table_tensor,
                                  const torch::Tensor& table_info_tensor,
                                  const torch::Tensor& em_x_tensor,
                                  const torch::Tensor& em_tensor,
                                  int64_t last_layer_size,
                                  torch::Tensor& descriptor_tensor) {
  // check input shape
  if (table_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of table should be 2");
  }
  if (em_x_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of em_x should be 2");
  }
  if (em_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of em should be 3");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* descriptor = descriptor_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();

  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei_i = em_tensor.size(1);
  const int64_t nnei_j = em_tensor.size(2);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_t_tebd_gpu(descriptor, table, table_info, em_x,
                                          em, nloc, nnei_i, nnei_j,
                                          last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_t_tebd_cpu(descriptor, table, table_info, em_x,
                                          em, nloc, nnei_i, nnei_j,
                                          last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeTTebdGradForward(const torch::Tensor& table_tensor,
                                      const torch::Tensor& table_info_tensor,
                                      const torch::Tensor& em_x_tensor,
                                      const torch::Tensor& em_tensor,
                                      const torch::Tensor& dy_tensor,
                                      const torch::Tensor& descriptor_tensor,
                                      torch::Tensor& dy_dem_x_tensor) {
  // check input shape
  if (dy_tensor.dim() != 4) {
    throw std::invalid_argument("Dim of dy_tensor should be 4");
  }
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dy_dem_x = dy_dem_x_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dy = dy_tensor.view({-1}).data_ptr<FPTYPE>();

  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei_i = em_tensor.size(1);
  const int64_t nnei_j = em_tensor.size(2);
  const int64_t last_layer_size = descriptor_tensor.size(3);

  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_t_tebd_grad_gpu(dy_dem_x, table, table_info,
                                               em_x, em, dy, nloc, nnei_i,
                                               nnei_j, last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_t_tebd_grad_cpu(dy_dem_x, table, table_info,
                                               em_x, em, dy, nloc, nnei_i,
                                               nnei_j, last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeTTebdGradGradForward(
    const torch::Tensor& table_tensor,
    const torch::Tensor& table_info_tensor,
    const torch::Tensor& em_x_tensor,
    const torch::Tensor& em_tensor,
    const torch::Tensor& dz_dy_dem_x_tensor,
    const torch::Tensor& descriptor_tensor,
    torch::Tensor& dz_dy_tensor) {
  // Check input shape
  if (dz_dy_dem_x_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of dz_dy_dem_x should be 3");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dz_dy = dz_dy_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dem_x = dz_dy_dem_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei_i = em_tensor.size(1);
  const int64_t nnei_j = em_tensor.size(2);
  const int64_t last_layer_size = descriptor_tensor.size(3);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_t_tebd_grad_grad_gpu(
        dz_dy, table, table_info, em_x, em, dz_dy_dem_x, nloc, nnei_i, nnei_j,
        last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    TORCH_CHECK(last_layer_size <= 1024,
                "In the process of model compression, the size of the "
                "last layer of embedding net must be less than 1024!");
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_t_tebd_grad_grad_cpu(
        dz_dy, table, table_info, em_x, em, dz_dy_dem_x, nloc, nnei_i, nnei_j,
        last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeRForward(const torch::Tensor& table_tensor,
                              const torch::Tensor& table_info_tensor,
                              const torch::Tensor& em_tensor,
                              int64_t last_layer_size,
                              torch::Tensor& descriptor_tensor) {
  // check input shape
  if (table_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of table should be 2");
  }
  if (em_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of input should be 2");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* descriptor = descriptor_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_r_gpu(descriptor, table, table_info, em, nloc,
                                     nnei, last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_r_cpu(descriptor, table, table_info, em, nloc,
                                     nnei, last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeRGradForward(const torch::Tensor& table_tensor,
                                  const torch::Tensor& table_info_tensor,
                                  const torch::Tensor& em_tensor,
                                  const torch::Tensor& dy_tensor,
                                  const torch::Tensor& descriptor_tensor,
                                  torch::Tensor& dy_dem_tensor) {
  // check input shape
  if (dy_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of dy_tensor should be 3");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dy_dem = dy_dem_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dy = dy_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  const int64_t last_layer_size = descriptor_tensor.size(2);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_r_grad_gpu(dy_dem, table, table_info, em, dy,
                                          nloc, nnei, last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_r_grad_cpu(dy_dem, table, table_info, em, dy,
                                          nloc, nnei, last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeRGradGradForward(const torch::Tensor& table_tensor,
                                      const torch::Tensor& table_info_tensor,
                                      const torch::Tensor& em_tensor,
                                      const torch::Tensor& dz_dy_dem_tensor,
                                      const torch::Tensor& descriptor_tensor,
                                      torch::Tensor& dz_dy_tensor) {
  // Check input shape
  if (dz_dy_dem_tensor.dim() != 2) {
    throw std::invalid_argument("Dim of dz_dy_dem should be 2");
  }
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dz_dy = dz_dy_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dem = dz_dy_dem_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  const int64_t last_layer_size = descriptor_tensor.size(2);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    deepmd::tabulate_fusion_se_r_grad_grad_gpu(
        dz_dy, table, table_info, em, dz_dy_dem, nloc, nnei, last_layer_size);
#else
    throw std::runtime_error(
        "The input tensor is on the GPU, but the GPU support for the "
        "customized OP library is not enabled.");
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    TORCH_CHECK(last_layer_size <= 1024,
                "In the process of model compression, the size of the "
                "last layer of embedding net must be less than 1024!");
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_r_grad_grad_cpu(
        dz_dy, table, table_info, em, dz_dy_dem, nloc, nnei, last_layer_size);
  }
}

class TabulateFusionSeAGradOp
    : public torch::autograd::Function<TabulateFusionSeAGradOp> {
 private:
  std::string device;

 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, dy_tensor,
                               descriptor_tensor);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, dy_tensor, descriptor_tensor);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor) {
    // Allocate output tensors
    torch::Tensor dy_dem_x_tensor = torch::zeros_like(em_x_tensor);
    torch::Tensor dy_dem_tensor = torch::zeros_like(em_tensor);
    torch::Tensor dy_dtwo_tensor = at::Tensor();
    // compute
    // The non-attention se_a path invokes this op per type-pair block, so
    // exclusions cannot interleave zero rows. Compressed forward_lower also
    // requests a sorted nlist through the Python-side
    // DescrptBlockSeA.need_sorted_nlist_for_lower() contract.
    TabulateFusionSeAGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor, at::Tensor(),
        dy_tensor, descriptor_tensor, true, dy_dem_x_tensor, dy_dem_tensor,
        dy_dtwo_tensor);
    // save data
    ctx->save_for_backward({table_tensor, table_info_tensor, em_x_tensor,
                            em_tensor, descriptor_tensor});

    return torch::autograd::variable_list{dy_dem_x_tensor, dy_dem_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // load data
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // load data
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_x_tensor = saved_variables[2];
    torch::Tensor em_tensor = saved_variables[3];
    torch::Tensor descriptor_tensor = saved_variables[4];

    bool is_sorted = true;

    torch::Tensor dz_dy_dem_x_tensor = grad_output[0].defined()
                                           ? grad_output[0].contiguous()
                                           : torch::zeros_like(em_x_tensor);
    torch::Tensor dz_dy_dem_tensor = grad_output[1].defined()
                                         ? grad_output[1].contiguous()
                                         : torch::zeros_like(em_tensor);
    // allocate output tensors
    torch::Tensor dz_dy_tensor = torch::empty_like(descriptor_tensor);
    // compute
    TabulateFusionSeAGradGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor, at::Tensor(),
        dz_dy_dem_x_tensor, dz_dy_dem_tensor, at::Tensor(), descriptor_tensor,
        is_sorted, dz_dy_tensor);

    return torch::autograd::variable_list{at::Tensor(), at::Tensor(),
                                          at::Tensor(), at::Tensor(),
                                          dz_dy_tensor, at::Tensor()};
  }
};

class TabulateFusionSeAGradGradOp
    : public torch::autograd::Function<TabulateFusionSeAGradGradOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dz_dy_dem_x_tensor,
      const torch::Tensor& dz_dy_dem_tensor,
      const torch::Tensor& descriptor_tensor,
      bool is_sorted) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, dz_dy_dem_x_tensor,
                               dz_dy_dem_tensor, descriptor_tensor, is_sorted);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, dz_dy_dem_x_tensor, dz_dy_dem_tensor,
                              descriptor_tensor, is_sorted);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dz_dy_dem_x_tensor,
      const torch::Tensor& dz_dy_dem_tensor,
      const torch::Tensor& descriptor_tensor,
      bool is_sorted) {
    // Allocate output tensor
    torch::Tensor dz_dy_tensor = torch::empty_like(descriptor_tensor);
    // compute
    TabulateFusionSeAGradGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor, at::Tensor(),
        dz_dy_dem_x_tensor, dz_dy_dem_tensor, at::Tensor(), descriptor_tensor,
        is_sorted, dz_dy_tensor);

    return torch::autograd::variable_list{dz_dy_tensor};
  }
};

class TabulateFusionSeAOp
    : public torch::autograd::Function<TabulateFusionSeAOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, last_layer_size);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, last_layer_size);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size) {
    CheckTabulateSeAInputs<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
                                   em_tensor, at::Tensor(), last_layer_size);
    // allocate output tensors
    auto options = torch::TensorOptions()
                       .dtype(table_tensor.dtype())
                       .device(table_tensor.device());
    torch::Tensor descriptor_tensor =
        torch::empty({em_tensor.size(0), 4, last_layer_size}, options);
    // compute
    // Keep the sorted fold enabled: exclusions are uniform within each se_a
    // type-pair invocation, and compressed forward_lower sorts its nlist first.
    TabulateFusionSeAForward<FPTYPE>(table_tensor, table_info_tensor,
                                     em_x_tensor, em_tensor, at::Tensor(),
                                     last_layer_size, true, descriptor_tensor);
    // save data
    ctx->save_for_backward({table_tensor, table_info_tensor, em_x_tensor,
                            em_tensor, descriptor_tensor});
    return {descriptor_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // load data
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_x_tensor = saved_variables[2];
    torch::Tensor em_tensor = saved_variables[3];
    torch::Tensor descriptor_tensor = saved_variables[4];

    // ensure the gradient output is contiguous
    torch::Tensor dy_tensor = grad_output[0].contiguous();
    torch::autograd::variable_list dy_dem_tensors =
        TabulateFusionSeAGradOp::apply(table_tensor, table_info_tensor,
                                       em_x_tensor, em_tensor, dy_tensor,
                                       descriptor_tensor);

    return {at::Tensor(), at::Tensor(), dy_dem_tensors[0], dy_dem_tensors[1],
            at::Tensor()};
  }
};

class TabulateFusionSeAttenGradOp
    : public torch::autograd::Function<TabulateFusionSeAttenGradOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& two_embed_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor,
      bool is_sorted) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, two_embed_tensor,
                               dy_tensor, descriptor_tensor, is_sorted);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, two_embed_tensor, dy_tensor,
                              descriptor_tensor, is_sorted);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& two_embed_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor,
      bool is_sorted) {
    torch::Tensor dy_dem_x_tensor = torch::zeros_like(em_x_tensor);
    torch::Tensor dy_dem_tensor = torch::zeros_like(em_tensor);
    torch::Tensor dy_dtwo_tensor = torch::zeros_like(two_embed_tensor);
    TabulateFusionSeAGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor,
        two_embed_tensor, dy_tensor, descriptor_tensor, is_sorted,
        dy_dem_x_tensor, dy_dem_tensor, dy_dtwo_tensor);

    ctx->save_for_backward({table_tensor, table_info_tensor, em_x_tensor,
                            em_tensor, two_embed_tensor, descriptor_tensor});
    ctx->saved_data["is_sorted"] = is_sorted;

    return torch::autograd::variable_list{dy_dem_x_tensor, dy_dem_tensor,
                                          dy_dtwo_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_x_tensor = saved_variables[2];
    torch::Tensor em_tensor = saved_variables[3];
    torch::Tensor two_embed_tensor = saved_variables[4];
    torch::Tensor descriptor_tensor = saved_variables[5];
    bool is_sorted = ctx->saved_data["is_sorted"].toBool();

    torch::Tensor dz_dy_dem_x_tensor = grad_output[0].defined()
                                           ? grad_output[0].contiguous()
                                           : torch::zeros_like(em_x_tensor);
    torch::Tensor dz_dy_dem_tensor = grad_output[1].defined()
                                         ? grad_output[1].contiguous()
                                         : torch::zeros_like(em_tensor);
    torch::Tensor dz_dy_dtwo_tensor = grad_output[2].defined()
                                          ? grad_output[2].contiguous()
                                          : torch::zeros_like(two_embed_tensor);
    torch::Tensor dz_dy_tensor = torch::empty_like(descriptor_tensor);
    TabulateFusionSeAGradGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor,
        two_embed_tensor, dz_dy_dem_x_tensor, dz_dy_dem_tensor,
        dz_dy_dtwo_tensor, descriptor_tensor, is_sorted, dz_dy_tensor);

    return torch::autograd::variable_list{
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), dz_dy_tensor, at::Tensor(), at::Tensor()};
  }
};

class TabulateFusionSeAttenOp
    : public torch::autograd::Function<TabulateFusionSeAttenOp> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& two_embed_tensor,
      int64_t last_layer_size,
      bool is_sorted) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, two_embed_tensor,
                               last_layer_size, is_sorted);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, two_embed_tensor, last_layer_size,
                              is_sorted);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& two_embed_tensor,
      int64_t last_layer_size,
      bool is_sorted) {
    CheckTabulateSeAInputs<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
                                   em_tensor, two_embed_tensor,
                                   last_layer_size);
    // allocate output tensors
    auto options = torch::TensorOptions()
                       .dtype(table_tensor.dtype())
                       .device(table_tensor.device());
    torch::Tensor descriptor_tensor =
        torch::empty({em_tensor.size(0), 4, last_layer_size}, options);
    // compute
    TabulateFusionSeAForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor,
        two_embed_tensor, last_layer_size, is_sorted, descriptor_tensor);
    // save data
    ctx->save_for_backward({table_tensor, table_info_tensor, em_x_tensor,
                            em_tensor, two_embed_tensor, descriptor_tensor});
    ctx->saved_data["is_sorted"] = is_sorted;
    return {descriptor_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // load data
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_x_tensor = saved_variables[2];
    torch::Tensor em_tensor = saved_variables[3];
    torch::Tensor two_embed_tensor = saved_variables[4];
    torch::Tensor descriptor_tensor = saved_variables[5];
    bool is_sorted = ctx->saved_data["is_sorted"].toBool();

    torch::Tensor dy_tensor = grad_output[0].contiguous();
    torch::autograd::variable_list dy_dem_tensors =
        TabulateFusionSeAttenGradOp::apply(
            table_tensor, table_info_tensor, em_x_tensor, em_tensor,
            two_embed_tensor, dy_tensor, descriptor_tensor, is_sorted);

    return {at::Tensor(),      at::Tensor(),      dy_dem_tensors[0],
            dy_dem_tensors[1], dy_dem_tensors[2], at::Tensor(),
            at::Tensor()};
  }
};

class TabulateFusionSeTGradOp
    : public torch::autograd::Function<TabulateFusionSeTGradOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, dy_tensor,
                               descriptor_tensor);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, dy_tensor, descriptor_tensor);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor) {
    torch::Tensor dy_dem_x_tensor = torch::zeros_like(em_x_tensor);
    torch::Tensor dy_dem_tensor = torch::zeros_like(em_tensor);
    TabulateFusionSeTGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor, dy_tensor,
        descriptor_tensor, dy_dem_x_tensor, dy_dem_tensor);

    ctx->save_for_backward({table_tensor, table_info_tensor, em_x_tensor,
                            em_tensor, descriptor_tensor});

    return torch::autograd::variable_list{dy_dem_x_tensor, dy_dem_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_x_tensor = saved_variables[2];
    torch::Tensor em_tensor = saved_variables[3];
    torch::Tensor descriptor_tensor = saved_variables[4];

    torch::Tensor dz_dy_dem_x_tensor = grad_output[0].defined()
                                           ? grad_output[0].contiguous()
                                           : torch::zeros_like(em_x_tensor);
    torch::Tensor dz_dy_dem_tensor = grad_output[1].defined()
                                         ? grad_output[1].contiguous()
                                         : torch::zeros_like(em_tensor);
    torch::Tensor dz_dy_tensor = torch::empty_like(descriptor_tensor);
    TabulateFusionSeTGradGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor,
        dz_dy_dem_x_tensor, dz_dy_dem_tensor, descriptor_tensor, dz_dy_tensor);

    return torch::autograd::variable_list{at::Tensor(), at::Tensor(),
                                          at::Tensor(), at::Tensor(),
                                          dz_dy_tensor, at::Tensor()};
  }
};

class TabulateFusionSeTOp
    : public torch::autograd::Function<TabulateFusionSeTOp> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, last_layer_size);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, last_layer_size);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size) {
    CheckTabulateSeTInputs<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
                                   em_tensor, last_layer_size);
    // allocate output tensors
    auto options = torch::TensorOptions()
                       .dtype(table_tensor.dtype())
                       .device(table_tensor.device());
    torch::Tensor descriptor_tensor =
        torch::empty({em_tensor.size(0), last_layer_size}, options);
    // compute
    TabulateFusionSeTForward<FPTYPE>(table_tensor, table_info_tensor,
                                     em_x_tensor, em_tensor, last_layer_size,
                                     descriptor_tensor);
    // save data
    ctx->save_for_backward({table_tensor, table_info_tensor, em_x_tensor,
                            em_tensor, descriptor_tensor});
    return {descriptor_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // load data
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_x_tensor = saved_variables[2];
    torch::Tensor em_tensor = saved_variables[3];
    torch::Tensor descriptor_tensor = saved_variables[4];

    torch::Tensor dy_tensor = grad_output[0].contiguous();
    torch::autograd::variable_list dy_dem_tensors =
        TabulateFusionSeTGradOp::apply(table_tensor, table_info_tensor,
                                       em_x_tensor, em_tensor, dy_tensor,
                                       descriptor_tensor);

    return {at::Tensor(), at::Tensor(), dy_dem_tensors[0], dy_dem_tensors[1],
            at::Tensor()};
  }
};

class TabulateFusionSeRGradOp
    : public torch::autograd::Function<TabulateFusionSeRGradOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor, em_tensor,
                               dy_tensor, descriptor_tensor);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_tensor,
                              dy_tensor, descriptor_tensor);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor) {
    torch::Tensor dy_dem_tensor = torch::zeros_like(em_tensor);
    TabulateFusionSeRGradForward<FPTYPE>(table_tensor, table_info_tensor,
                                         em_tensor, dy_tensor,
                                         descriptor_tensor, dy_dem_tensor);

    ctx->save_for_backward(
        {table_tensor, table_info_tensor, em_tensor, descriptor_tensor});

    return torch::autograd::variable_list{dy_dem_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_tensor = saved_variables[2];
    torch::Tensor descriptor_tensor = saved_variables[3];

    torch::Tensor dz_dy_dem_tensor = grad_output[0].defined()
                                         ? grad_output[0].contiguous()
                                         : torch::zeros_like(em_tensor);
    torch::Tensor dz_dy_tensor = torch::empty_like(descriptor_tensor);
    TabulateFusionSeRGradGradForward<FPTYPE>(table_tensor, table_info_tensor,
                                             em_tensor, dz_dy_dem_tensor,
                                             descriptor_tensor, dz_dy_tensor);

    return torch::autograd::variable_list{
        at::Tensor(), at::Tensor(), at::Tensor(), dz_dy_tensor, at::Tensor()};
  }
};

class TabulateFusionSeROp
    : public torch::autograd::Function<TabulateFusionSeROp> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor, em_tensor,
                               last_layer_size);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_tensor,
                              last_layer_size);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size) {
    CheckTabulateSeRInputs<FPTYPE>(table_tensor, table_info_tensor, em_tensor,
                                   last_layer_size);
    // allocate output tensors
    auto options = torch::TensorOptions()
                       .dtype(table_tensor.dtype())
                       .device(table_tensor.device());
    torch::Tensor descriptor_tensor = torch::empty(
        {em_tensor.size(0), em_tensor.size(1), last_layer_size}, options);
    // compute
    TabulateFusionSeRForward<FPTYPE>(table_tensor, table_info_tensor, em_tensor,
                                     last_layer_size, descriptor_tensor);
    // save data
    ctx->save_for_backward(
        {table_tensor, table_info_tensor, em_tensor, descriptor_tensor});
    return {descriptor_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // load data
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_tensor = saved_variables[2];
    torch::Tensor descriptor_tensor = saved_variables[3];

    torch::Tensor dy_tensor = grad_output[0].contiguous();
    torch::autograd::variable_list dy_dem_tensors =
        TabulateFusionSeRGradOp::apply(table_tensor, table_info_tensor,
                                       em_tensor, dy_tensor, descriptor_tensor);

    return {at::Tensor(), at::Tensor(), dy_dem_tensors[0], at::Tensor()};
  }
};

class TabulateFusionSeTTebdGradOp
    : public torch::autograd::Function<TabulateFusionSeTTebdGradOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, dy_tensor,
                               descriptor_tensor);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, dy_tensor, descriptor_tensor);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& descriptor_tensor) {
    torch::Tensor dy_dem_x_tensor = torch::zeros_like(em_x_tensor);
    TabulateFusionSeTTebdGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor, dy_tensor,
        descriptor_tensor, dy_dem_x_tensor);

    ctx->save_for_backward({table_tensor, table_info_tensor, em_x_tensor,
                            em_tensor, descriptor_tensor});

    return torch::autograd::variable_list{dy_dem_x_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_x_tensor = saved_variables[2];
    torch::Tensor em_tensor = saved_variables[3];
    torch::Tensor descriptor_tensor = saved_variables[4];

    torch::Tensor dz_dy_dem_x_tensor =
        grad_output[0].defined()
            ? grad_output[0].contiguous().view(
                  {em_tensor.size(0), em_tensor.size(1), em_tensor.size(2)})
            : torch::zeros_like(em_tensor);
    torch::Tensor dz_dy_tensor = torch::empty_like(descriptor_tensor);
    TabulateFusionSeTTebdGradGradForward<FPTYPE>(
        table_tensor, table_info_tensor, em_x_tensor, em_tensor,
        dz_dy_dem_x_tensor, descriptor_tensor, dz_dy_tensor);

    return torch::autograd::variable_list{at::Tensor(), at::Tensor(),
                                          at::Tensor(), at::Tensor(),
                                          dz_dy_tensor, at::Tensor()};
  }
};

class TabulateFusionSeTTebdOp
    : public torch::autograd::Function<TabulateFusionSeTTebdOp> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size) {
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, table_tensor, table_info_tensor,
                               em_x_tensor, em_tensor, last_layer_size);
    } else {
      return forward_t<float>(ctx, table_tensor, table_info_tensor, em_x_tensor,
                              em_tensor, last_layer_size);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size) {
    CheckTabulateSeTInputs<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
                                   em_tensor, last_layer_size);
    // allocate output tensors
    auto options = torch::TensorOptions()
                       .dtype(table_tensor.dtype())
                       .device(table_tensor.device());
    torch::Tensor descriptor_tensor =
        torch::empty({em_tensor.size(0), em_tensor.size(1), em_tensor.size(2),
                      last_layer_size},
                     options);
    // compute
    TabulateFusionSeTTebdForward<FPTYPE>(table_tensor, table_info_tensor,
                                         em_x_tensor, em_tensor,
                                         last_layer_size, descriptor_tensor);
    // save data
    ctx->save_for_backward({table_tensor, table_info_tensor, em_x_tensor,
                            em_tensor, descriptor_tensor});
    return {descriptor_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return backward_t<double>(ctx, grad_output);
    } else {
      return backward_t<float>(ctx, grad_output);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // load data
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor table_tensor = saved_variables[0];
    torch::Tensor table_info_tensor = saved_variables[1];
    torch::Tensor em_x_tensor = saved_variables[2];
    torch::Tensor em_tensor = saved_variables[3];
    torch::Tensor descriptor_tensor = saved_variables[4];

    torch::Tensor dy_tensor = grad_output[0].contiguous();
    torch::autograd::variable_list dy_dem_tensors =
        TabulateFusionSeTTebdGradOp::apply(table_tensor, table_info_tensor,
                                           em_x_tensor, em_tensor, dy_tensor,
                                           descriptor_tensor);

    return {at::Tensor(), at::Tensor(), dy_dem_tensors[0], at::Tensor(),
            at::Tensor()};
  }
};

std::vector<torch::Tensor> tabulate_fusion_se_a(
    const torch::Tensor& table_tensor,
    const torch::Tensor& table_info_tensor,  // only cpu
    const torch::Tensor& em_x_tensor,
    const torch::Tensor& em_tensor,
    int64_t last_layer_size) {
  return TabulateFusionSeAOp::apply(table_tensor, table_info_tensor,
                                    em_x_tensor, em_tensor, last_layer_size);
}

std::vector<torch::Tensor> tabulate_fusion_se_atten(
    const torch::Tensor& table_tensor,
    const torch::Tensor& table_info_tensor,  // only cpu
    const torch::Tensor& em_x_tensor,
    const torch::Tensor& em_tensor,
    const torch::Tensor& two_embed_tensor,
    int64_t last_layer_size,
    bool is_sorted) {
  return TabulateFusionSeAttenOp::apply(
      table_tensor, table_info_tensor, em_x_tensor, em_tensor, two_embed_tensor,
      last_layer_size, is_sorted);
}

std::vector<torch::Tensor> tabulate_fusion_se_t(
    const torch::Tensor& table_tensor,
    const torch::Tensor& table_info_tensor,  // only cpu
    const torch::Tensor& em_x_tensor,
    const torch::Tensor& em_tensor,
    int64_t last_layer_size) {
  return TabulateFusionSeTOp::apply(table_tensor, table_info_tensor,
                                    em_x_tensor, em_tensor, last_layer_size);
}

std::vector<torch::Tensor> tabulate_fusion_se_t_tebd(
    const torch::Tensor& table_tensor,
    const torch::Tensor& table_info_tensor,  // only cpu
    const torch::Tensor& em_x_tensor,
    const torch::Tensor& em_tensor,
    int64_t last_layer_size) {
  return TabulateFusionSeTTebdOp::apply(
      table_tensor, table_info_tensor, em_x_tensor, em_tensor, last_layer_size);
}

std::vector<torch::Tensor> tabulate_fusion_se_r(
    const torch::Tensor& table_tensor,
    const torch::Tensor& table_info_tensor,  // only cpu
    const torch::Tensor& em_tensor,
    int64_t last_layer_size) {
  return TabulateFusionSeROp::apply(table_tensor, table_info_tensor, em_tensor,
                                    last_layer_size);
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("tabulate_fusion_se_a", tabulate_fusion_se_a);
}
TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("tabulate_fusion_se_atten", tabulate_fusion_se_atten);
}
TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("tabulate_fusion_se_t", tabulate_fusion_se_t);
}
TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("tabulate_fusion_se_t_tebd", tabulate_fusion_se_t_tebd);
}
TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("tabulate_fusion_se_r", tabulate_fusion_se_r);
}
