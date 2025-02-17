// SPDX-License-Identifier: LGPL-3.0-or-later
#include "thsilu.h"

#include <torch/torch.h>

#include <string>
#include <vector>

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "device.h"
#endif

inline void GetTensorDevice(const torch::Tensor& t, std::string& str) {
  if (t.device().is_cuda()) {
    str = "GPU";
  } else {
    str = "CPU";
  }
}

class ThsiluGradGradOp : public torch::autograd::Function<ThsiluGradGradOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x_tensor,
      const torch::Tensor& grad_output,
      const torch::Tensor& grad_output2,
      const double& w,
      const double& a) {
    bool type_flag = (x_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, x_tensor, grad_output, grad_output2, w, a);
    } else {
      return forward_t<float>(ctx, x_tensor, grad_output, grad_output2, w, a);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x_tensor,
      const torch::Tensor& dy_tensor,
      const torch::Tensor& dy2_tensor,
      const FPTYPE& w,
      const FPTYPE& a) {
    // allocate output tensors
    auto options = torch::TensorOptions()
                       .dtype(x_tensor.dtype())
                       .device(x_tensor.device());
    torch::Tensor y_tensor = torch::empty_like(x_tensor);
    int64_t tensor_size = x_tensor.numel();
    // get the device
    std::string device;
    GetTensorDevice(x_tensor, device);
    // flat the tensors
    FPTYPE* y = y_tensor.view({-1}).data_ptr<FPTYPE>();
    const FPTYPE* dy = dy_tensor.view({-1}).data_ptr<FPTYPE>();
    const FPTYPE* dy_2 = dy2_tensor.view({-1}).data_ptr<FPTYPE>();
    const FPTYPE* x = x_tensor.view({-1}).data_ptr<FPTYPE>();
    // compute
    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::thsilu_grad_grad_gpu(y, x, dy, dy_2, tensor_size, w, a);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::thsilu_grad_grad_cpu(y, x, dy, dy_2, tensor_size, w, a);
    }
    // save data
    // ctx->save_for_backward({x_tensor, grad_output});
    // save w, a, b
    // ctx->saved_data["w"] = w;
    // ctx->saved_data["a"] = a;
    return {y_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    throw std::runtime_error("Not Implemented");
  }
};

class ThsiluGradOp : public torch::autograd::Function<ThsiluGradOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x_tensor,
      const torch::Tensor& grad_output,
      const double& w,
      const double& a) {
    bool type_flag = (x_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, x_tensor, grad_output, w, a);
    } else {
      return forward_t<float>(ctx, x_tensor, grad_output, w, a);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x_tensor,
      const torch::Tensor& dy_tensor,
      const FPTYPE& w,
      const FPTYPE& a) {
    // allocate output tensors
    auto options = torch::TensorOptions()
                       .dtype(x_tensor.dtype())
                       .device(x_tensor.device());
    torch::Tensor y_tensor = torch::empty_like(x_tensor);
    int64_t tensor_size = x_tensor.numel();
    // get the device
    std::string device;
    GetTensorDevice(x_tensor, device);
    // flat the tensors
    FPTYPE* y = y_tensor.view({-1}).data_ptr<FPTYPE>();
    const FPTYPE* dy = dy_tensor.view({-1}).data_ptr<FPTYPE>();
    const FPTYPE* x = x_tensor.view({-1}).data_ptr<FPTYPE>();
    // compute
    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::thsilu_grad_gpu(y, x, dy, tensor_size, w, a);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::thsilu_grad_cpu(y, x, dy, tensor_size, w, a);
    }
    // save data
    ctx->save_for_backward({x_tensor, y_tensor, dy_tensor});
    // save w, a, b
    ctx->saved_data["w"] = w;
    ctx->saved_data["a"] = a;
    return {y_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor x_tensor = saved_variables[0];
    bool type_flag = (x_tensor.dtype() == torch::kDouble) ? true : false;
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
    torch::Tensor x_tensor = saved_variables[0];
    torch::Tensor y_tensor = saved_variables[1];
    torch::Tensor dy_tensor = saved_variables[2];
    torch::Tensor dy2_tensor = grad_output[0];
    FPTYPE w = ctx->saved_data["w"].toDouble();
    FPTYPE a = ctx->saved_data["a"].toDouble();
    return {ThsiluGradGradOp::apply(x_tensor, dy_tensor, dy2_tensor, w, a)[0],
            ThsiluGradOp::apply(x_tensor, dy_tensor, w, a)[0], at::Tensor(),
            at::Tensor()};
  }
};

class ThsiluOp : public torch::autograd::Function<ThsiluOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x_tensor,
      const double& w,
      const double& a,
      const double& b) {
    bool type_flag = (x_tensor.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, x_tensor, w, a, b);
    } else {
      return forward_t<float>(ctx, x_tensor, w, a, b);
    }
  }

  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x_tensor,
      const FPTYPE& w,
      const FPTYPE& a,
      const FPTYPE& b) {
    // allocate output tensors
    auto options = torch::TensorOptions()
                       .dtype(x_tensor.dtype())
                       .device(x_tensor.device());
    torch::Tensor y_tensor = torch::empty_like(x_tensor);
    int64_t tensor_size = x_tensor.numel();
    // get the device
    std::string device;
    GetTensorDevice(x_tensor, device);
    // flat the tensors
    FPTYPE* y = y_tensor.view({-1}).data_ptr<FPTYPE>();
    const FPTYPE* x = x_tensor.view({-1}).data_ptr<FPTYPE>();
    // compute
    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::thsilu_gpu(y, x, tensor_size, w, a, b);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::thsilu_cpu(y, x, tensor_size, w, a, b);
    }
    // save data
    ctx->save_for_backward({x_tensor, y_tensor});
    // save w, a, b
    ctx->saved_data["w"] = w;
    ctx->saved_data["a"] = a;
    return {y_tensor};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor x_tensor = saved_variables[0];
    bool type_flag = (x_tensor.dtype() == torch::kDouble) ? true : false;
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
    torch::Tensor x_tensor = saved_variables[0];
    torch::Tensor dy_tensor = grad_output[0];
    FPTYPE w = ctx->saved_data["w"].toDouble();
    FPTYPE a = ctx->saved_data["a"].toDouble();
    return {ThsiluGradOp::apply(x_tensor, dy_tensor, w, a)[0], at::Tensor(),
            at::Tensor(), at::Tensor()};
  }
};

torch::Tensor thsilu(const torch::Tensor& x,
                     const double& w,
                     const double& a,
                     const double& b) {
  return ThsiluOp::apply(x, w, a, b)[0];
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) { m.def("thsilu", thsilu); }
