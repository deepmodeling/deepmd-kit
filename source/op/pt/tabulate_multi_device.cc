#include <torch/torch.h>
#include "tabulate.h"
#include <vector>
#include <string>

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "device.h"
#endif

void GetTensorDevice(const torch::Tensor& t, std::string& str){
  if(t.device().is_cuda())
    str = "GPU";
  else
    str = "CPU";
}

void print_first_five_elements(const torch::Tensor& tensor, const std::string& name) {
    std::cout << "First five elements of " << name << ":" << std::endl;
    std::cout << tensor.slice(0, 0, 5) << std::endl;
}

template <typename FPTYPE>
void TabulateFusionSeAForward(
  const torch::Tensor& table_tensor,
  const torch::Tensor& table_info_tensor,
  const torch::Tensor& em_x_tensor,
  const torch::Tensor& em_tensor,
  int64_t last_layer_size,
  torch::Tensor& descriptor_tensor
){
  // check input shape
  if(table_tensor.dim() != 2)
    throw std::invalid_argument("Dim of table should be 2");
  if(em_x_tensor.dim() != 2)
    throw std::invalid_argument("Dim of input should be 2");
  if(em_tensor.dim() != 3)
    throw std::invalid_argument("Dim of input should be 3");
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* descriptor = descriptor_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* two_embed = nullptr;
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_a_gpu(descriptor, table, table_info, em_x, em,
                                       two_embed, nloc, nnei, last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_a_cpu(descriptor, table, table_info, em_x, em,
                                      two_embed, nloc, nnei, last_layer_size);
  }
}

template <typename FPTYPE>
void TabulateFusionSeAGradForward(
  const torch::Tensor& table_tensor,
  const torch::Tensor& table_info_tensor,
  const torch::Tensor& em_x_tensor,
  const torch::Tensor& em_tensor,
  const torch::Tensor& dy_tensor,
  const torch::Tensor& descriptor_tensor,
  torch::Tensor& dy_dem_x_tensor,
  torch::Tensor& dy_dem_tensor,
  torch::Tensor& dy_dtwo_tensor
){
  // check input shape
  if (dy_tensor.dim() != 3) {
    throw std::invalid_argument("Dim of dy_tensor should be 3");
  }
  // check input value
  // print_first_five_elements(table_tensor, "table_tensor");
  // print_first_five_elements(table_info_tensor, "table_info_tensor");
  // print_first_five_elements(em_x_tensor, "em_x_tensor");
  // print_first_five_elements(em_tensor, "em_tensor");
  print_first_five_elements(dy_tensor, "dy_tensor");
  print_first_five_elements(descriptor_tensor, "descriptor_tensor");
  print_first_five_elements(dy_dem_x_tensor, "dy_dem_x_tensor");
  print_first_five_elements(dy_dem_tensor, "dy_dem_tensor");
  print_first_five_elements(dy_dtwo_tensor, "dy_dtwo_tensor");

  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dy_dem_x = dy_dem_x_tensor.view({-1}).data_ptr<FPTYPE>();
  FPTYPE* dy_dem = dy_dem_tensor.view({-1}).data_ptr<FPTYPE>();
  FPTYPE* dy_dtwo = dy_dtwo_tensor.view({-1}).data_ptr<FPTYPE>();

  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* two_embed = nullptr;
  const FPTYPE* dy = dy_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* descriptor = descriptor_tensor.view({-1}).data_ptr<FPTYPE>();
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  const int64_t last_layer_size = descriptor_tensor.size(2);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_a_grad_gpu(dy_dem_x, dy_dem, dy_dtwo, table,
                                            table_info, em_x, em, two_embed, dy,
                                            nloc, nnei, last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device == "CPU") {
    deepmd::tabulate_fusion_se_a_grad_cpu(dy_dem_x, dy_dem, dy_dtwo, table,
                                          table_info, em_x, em, two_embed, dy,
                                          nloc, nnei, last_layer_size);
  }
  print_first_five_elements(dy_dem_x_tensor, "dy_dem_x_tensor");
  print_first_five_elements(dy_dem_tensor, "dy_dem_tensor");
  print_first_five_elements(dy_dtwo_tensor, "dy_dtwo_tensor");

}

template <typename FPTYPE>
void TabulateFusionSeAGradGradForward(
  const torch::Tensor& table_tensor,
  const torch::Tensor& table_info_tensor,
  const torch::Tensor& em_x_tensor,
  const torch::Tensor& em_tensor,
  const torch::Tensor& dz_dy_dem_x_tensor,
  const torch::Tensor& dz_dy_dem_tensor,
  const torch::Tensor& descriptor_tensor,
  bool is_sorted,
  torch::Tensor& dz_dy_tensor
){
  // Check input shape
  if (dz_dy_dem_x_tensor.dim() != 2)
    throw std::invalid_argument("Dim of dz_dy_dem_x should be 2");
  if (dz_dy_dem_tensor.dim() != 3)
    throw std::invalid_argument("Dim of dz_dy_dem should be 3");
  // get the device
  std::string device;
  GetTensorDevice(table_tensor, device);
  // flat the tensors
  FPTYPE* dz_dy = dz_dy_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table = table_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* table_info = table_info_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em_x = em_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* em = em_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* two_embed = nullptr;
  const FPTYPE* dz_dy_dem_x = dz_dy_dem_x_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dem = dz_dy_dem_tensor.view({-1}).data_ptr<FPTYPE>();
  const FPTYPE* dz_dy_dtwo = nullptr;
  const int64_t nloc = em_tensor.size(0);
  const int64_t nnei = em_tensor.size(1);
  const int64_t last_layer_size = descriptor_tensor.size(2);
  // compute
  if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_a_grad_grad_gpu(
          dz_dy, table, table_info, em_x, em, two_embed, dz_dy_dem_x, dz_dy_dem,
          dz_dy_dtwo, nloc, nnei, last_layer_size, is_sorted);
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


class TabulateFusionSeAOp : public torch::autograd::Function<TabulateFusionSeAOp> {
  public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& table_tensor,
        const torch::Tensor& table_info_tensor,
        const torch::Tensor& em_x_tensor,
        const torch::Tensor& em_tensor,
        int64_t last_layer_size
    ){
      bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
      if(type_flag) {
        return forward_t<double>(ctx, table_tensor, table_info_tensor,
                                 em_x_tensor, em_tensor, last_layer_size);
      } else {
        return forward_t<float>(ctx, table_tensor, table_info_tensor,
                                em_x_tensor, em_tensor, last_layer_size);
      }
    }

    // std::vector<torch::Tensor>
    template <typename FPTYPE>
    static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& table_tensor,
      const torch::Tensor& table_info_tensor,
      const torch::Tensor& em_x_tensor,
      const torch::Tensor& em_tensor,
      int64_t last_layer_size
    ){
      // allocate output tensors
      auto options = 
        torch::TensorOptions()
          .dtype(table_tensor.dtype())
          .device(table_tensor.device());
      torch::Tensor descriptor_tensor = torch::empty(
        {em_tensor.size(0), 4, last_layer_size}, options);
      // compute
      TabulateFusionSeAForward<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
                               em_tensor, last_layer_size, descriptor_tensor);
      // save data
      ctx->save_for_backward({
        table_tensor, table_info_tensor, em_x_tensor, 
        em_tensor, descriptor_tensor
      });
      return { descriptor_tensor };
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_output
    ){
      // load data
      torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
      torch::Tensor table_tensor = saved_variables[0];
      bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
      if(type_flag) {
        return backward_t<double>(ctx, grad_output);
      } else{
        return backward_t<float>(ctx, grad_output);
      }
    }

    template <typename FPTYPE>
    static torch::autograd::variable_list backward_t(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output
    ){
      // load data
      torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
      torch::Tensor table_tensor = saved_variables[0];
      torch::Tensor table_info_tensor = saved_variables[1];
      torch::Tensor em_x_tensor = saved_variables[2];
      torch::Tensor em_tensor = saved_variables[3];
      torch::Tensor descriptor_tensor = saved_variables[4];

      torch::Tensor dy_tensor = grad_output[0];
      // allocate output tensors
      auto options = torch::TensorOptions()
                      .dtype(em_x_tensor.dtype())
                      .device(em_x_tensor.device());
      torch::Tensor dy_dem_x_tensor = torch::zeros(
        em_x_tensor.sizes(), options
      );
      torch::Tensor dy_dem_tensor = torch::zeros(
        em_tensor.sizes(), options
      );
      torch::Tensor dy_dtwo_tensor = torch::zeros(
        dy_tensor.sizes(), options
      );
      // compute
      TabulateFusionSeAGradForward<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
                                          em_tensor, dy_tensor, descriptor_tensor,
                                          dy_dem_x_tensor, dy_dem_tensor, dy_dtwo_tensor);
      
      return {at::Tensor(), at::Tensor(), dy_dem_x_tensor, dy_dem_tensor, at::Tensor()};
    }
};

class TabulateFusionSeAGradOp : public torch::autograd::Function<TabulateFusionSeAGradOp> {
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
      const torch::Tensor& descriptor_tensor
    ) {
      bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
      if(type_flag) {
        return forward_t<double>(ctx, table_tensor, table_info_tensor,
                                 em_x_tensor, em_tensor, dy_tensor,
                                 descriptor_tensor);
      } else{
        return forward_t<float>(ctx, table_tensor, table_info_tensor,
                                 em_x_tensor, em_tensor, dy_tensor,
                                 descriptor_tensor);
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
      const torch::Tensor& descriptor_tensor
    ){
      // Allocate output tensors
      torch::Tensor dy_dem_x_tensor = torch::empty_like(em_x_tensor);
      torch::Tensor dy_dem_tensor = torch::empty_like(em_tensor);
      // compute
      // TabulateFusionSeAGradForward<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
      //                              em_tensor, dy_tensor, descriptor_tensor,
      //                              dy_dem_x_tensor, dy_dem_tensor);
      // save data
      ctx->save_for_backward({
        table_tensor, table_info_tensor, em_x_tensor,
        em_tensor, descriptor_tensor
      });

      return torch::autograd::variable_list{ dy_dem_x_tensor, dy_dem_tensor };
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_output
    ){
      // load data
      torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
      torch::Tensor table_tensor = saved_variables[0];
      bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
      if(type_flag) {
        return backward_t<double>(ctx, grad_output);
      } else{
        return backward_t<float>(ctx, grad_output);
      }
    }

    template <typename FPTYPE>
    static torch::autograd::variable_list backward_t(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_output
    ){
      // load data
      torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
      torch::Tensor table_tensor = saved_variables[0];
      torch::Tensor table_info_tensor = saved_variables[1];
      torch::Tensor em_x_tensor = saved_variables[2];
      torch::Tensor em_tensor = saved_variables[3];
      torch::Tensor descriptor_tensor = saved_variables[4];

      bool is_sorted = true;

      torch::Tensor dz_dy_dem_x_tensor = grad_output[0];
      torch::Tensor dz_dy_dem_tensor = grad_output[1];
      // allocate output tensors
      torch::Tensor dz_dy_tensor = torch::empty_like(descriptor_tensor);
      // compute
      TabulateFusionSeAGradGradForward<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
                                       em_tensor, dz_dy_dem_x_tensor, dz_dy_dem_tensor,
                                       descriptor_tensor, is_sorted, dz_dy_tensor);

      return torch::autograd::variable_list{at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), dz_dy_tensor, at::Tensor()};
    }

};

class TabulateFusionSeAGradGradOp : public torch::autograd::Function<TabulateFusionSeAGradGradOp> { 
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
      bool is_sorted
    ){
      bool type_flag = (table_tensor.dtype() == torch::kDouble) ? true : false;
      if(type_flag) {
        return forward_t<double>(ctx, table_tensor, table_info_tensor,
                                 em_x_tensor, em_tensor, dz_dy_dem_x_tensor,
                                 dz_dy_dem_tensor, descriptor_tensor, is_sorted);
      } else{
        return forward_t<float>(ctx, table_tensor, table_info_tensor,
                                 em_x_tensor, em_tensor, dz_dy_dem_x_tensor,
                                 dz_dy_dem_tensor, descriptor_tensor, is_sorted);
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
      bool is_sorted
    ) {
      // Allocate output tensor
      torch::Tensor dz_dy_tensor = torch::empty_like(descriptor_tensor);
      // compute
      TabulateFusionSeAGradGradForward<FPTYPE>(table_tensor, table_info_tensor, em_x_tensor,
                                       em_tensor, dz_dy_dem_x_tensor, dz_dy_dem_tensor,
                                       descriptor_tensor, is_sorted, dz_dy_tensor);
      
      return torch::autograd::variable_list{ dz_dy_tensor };
    }
};

// class TabulateFusionSeAttenOp : public torch::autograd::Function<TabulateFusionSeAttenOp>{
//   private:
//     std::string device;
  
//   public:
//     explicit TabulateFusionSeAttenOp(int last_layer_size, bool is_sorted)
//       : last_layer_size(last_layer_size), is_sorted(is_sorted) {}
    
//     static std::vector<torch::Tensor> forward(
//         torch::autograd::AutogradContext* ctx,
//         const torch::Tensor& table_tensor,
//         const torch::Tensor& table_info_tensor,
//         const torch::Tensor& em_x_tensor,
//         const torch::Tensor& em_tensor,
//         int last_layer_size,
//         bool is_sorted
//     ){
      
//     }
// };

std::vector<torch::Tensor> tabluate_fusion_se_a(
  const torch::Tensor& table_tensor,
  const torch::Tensor& table_info_tensor,
  const torch::Tensor& em_x_tensor,
  const torch::Tensor& em_tensor,
  int64_t last_layer_size
){
  return TabulateFusionSeAOp::apply(table_tensor, table_info_tensor, em_x_tensor,
                                    em_tensor, last_layer_size);
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) { m.def("tabulate_fusion_se_a", tabluate_fusion_se_a); }
