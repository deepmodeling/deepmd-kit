
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "custom_op.h"
#include "tabulate.h"

REGISTER_OP("TabulateFusion")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Attr("last_layer_size: int")
    .Output("descriptor: T");

REGISTER_OP("TabulateFusionGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("dy: T")
    .Input("descriptor: T")
    .Output("dy_dem_x: T")
    .Output("dy_dem: T");

REGISTER_OP("TabulateFusionGradGrad")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("dz_dy_dem_x: T")
    .Input("dz_dy_dem: T")
    .Input("descriptor: T")
    .Output("dz_dy: T");

REGISTER_OP("TabulateFusionSeA")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Attr("last_layer_size: int")
    .Output("descriptor: T");

REGISTER_OP("TabulateFusionSeAGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("dy: T")
    .Input("descriptor: T")
    .Output("dy_dem_x: T")
    .Output("dy_dem: T");

REGISTER_OP("TabulateFusionSeAGradGrad")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("dz_dy_dem_x: T")
    .Input("dz_dy_dem: T")
    .Input("descriptor: T")
    .Output("dz_dy: T")
    .Attr("is_sorted: bool = true");

REGISTER_OP("TabulateFusionSeAtten")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("two_embed: T")
    .Attr("last_layer_size: int")
    .Attr("is_sorted: bool = true")
    .Output("descriptor: T");

REGISTER_OP("TabulateFusionSeAttenGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("two_embed: T")
    .Input("dy: T")
    .Input("descriptor: T")
    .Output("dy_dem_x: T")
    .Output("dy_dem: T")
    .Output("dy_dtwo: T")
    .Attr("is_sorted: bool = true");

REGISTER_OP("TabulateFusionSeAttenGradGrad")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("two_embed: T")
    .Input("dz_dy_dem_x: T")
    .Input("dz_dy_dem: T")
    .Input("dz_dy_dtwo: T")
    .Input("descriptor: T")
    .Output("dz_dy: T")
    .Attr("is_sorted: bool = true");

REGISTER_OP("TabulateFusionSeT")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Attr("last_layer_size: int")
    .Output("descriptor: T");

REGISTER_OP("TabulateFusionSeTGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("dy: T")
    .Input("descriptor: T")
    .Output("dy_dem_x: T")
    .Output("dy_dem: T");

REGISTER_OP("TabulateFusionSeTGradGrad")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("dz_dy_dem_x: T")
    .Input("dz_dy_dem: T")
    .Input("descriptor: T")
    .Output("dz_dy: T");

REGISTER_OP("TabulateFusionSeR")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em: T")
    .Attr("last_layer_size: int")
    .Output("descriptor: T");

REGISTER_OP("TabulateFusionSeRGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em: T")
    .Input("dy: T")
    .Input("descriptor: T")
    .Output("dy_dem: T");

REGISTER_OP("TabulateFusionSeRGradGrad")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em: T")
    .Input("dz_dy_dem: T")
    .Input("descriptor: T")
    .Output("dz_dy: T");

template <typename Device, typename FPTYPE>
class TabulateFusionSeAOp : public OpKernel {
 public:
  explicit TabulateFusionSeAOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("last_layer_size", &last_layer_size));
  }
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (table_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of table should be 2"));
    OP_REQUIRES(context, (em_x_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (em_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of input should be 3"));
    TensorShape descriptor_shape;
    descriptor_shape.AddDim(em_tensor.shape().dim_size(0));
    descriptor_shape.AddDim(4);  // TODO: be careful here;
    descriptor_shape.AddDim(last_layer_size);
    int context_output_index = 0;
    Tensor* descriptor_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descriptor_shape,
                                                     &descriptor_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());
    // flat the tensors
    FPTYPE* descriptor = descriptor_tensor->flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* two_embed = nullptr;
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);

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

 private:
  int last_layer_size;
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeAGradOp : public OpKernel {
 public:
  explicit TabulateFusionSeAGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);

    const Tensor& dy_tensor = context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (dy_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of table should be 3"));
    int context_output_index = 0;
    Tensor* dy_dem_x_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     em_x_tensor.shape(),
                                                     &dy_dem_x_tensor));
    Tensor* dy_dem_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++,
                                            em_tensor.shape(), &dy_dem_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());

    // flat the tensors
    FPTYPE* dy_dem_x = dy_dem_x_tensor->flat<FPTYPE>().data();
    FPTYPE* dy_dem = dy_dem_tensor->flat<FPTYPE>().data();
    FPTYPE* dy_dtwo = nullptr;

    const FPTYPE* descriptor = descriptor_tensor.flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* two_embed = nullptr;
    const FPTYPE* dy = dy_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);
    const int last_layer_size = descriptor_tensor.shape().dim_size(2);

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
  }

 private:
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeAGradGradOp : public OpKernel {
 public:
  explicit TabulateFusionSeAGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_sorted", &is_sorted));
  }
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    const Tensor& dz_dy_dem_x_tensor = context->input(context_input_index++);
    const Tensor& dz_dy_dem_tensor = context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (dz_dy_dem_x_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (dz_dy_dem_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of input should be 3"));
    int context_output_index = 0;
    Tensor* dz_dy_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descriptor_tensor.shape(),
                                                     &dz_dy_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());

    // flat the tensors
    FPTYPE* dz_dy = dz_dy_tensor->flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* two_embed = nullptr;
    const FPTYPE* dz_dy_dem_x = dz_dy_dem_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* dz_dy_dem = dz_dy_dem_tensor.flat<FPTYPE>().data();
    const FPTYPE* dz_dy_dtwo = nullptr;
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);
    const int last_layer_size = descriptor_tensor.shape().dim_size(2);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_a_grad_grad_gpu(
          dz_dy, table, table_info, em_x, em, two_embed, dz_dy_dem_x, dz_dy_dem,
          dz_dy_dtwo, nloc, nnei, last_layer_size, is_sorted);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      OP_REQUIRES(context, (last_layer_size <= 1024),
                  errors::InvalidArgument(
                      "In the process of model compression, the size of the "
                      "last layer of embedding net must be less than 1024!"));
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_a_grad_grad_cpu(
          dz_dy, table, table_info, em_x, em, two_embed, dz_dy_dem_x, dz_dy_dem,
          dz_dy_dtwo, nloc, nnei, last_layer_size, is_sorted);
    }
  }

 private:
  bool is_sorted;
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeAttenOp : public OpKernel {
 public:
  explicit TabulateFusionSeAttenOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("last_layer_size", &last_layer_size));
    OP_REQUIRES_OK(context, context->GetAttr("is_sorted", &is_sorted));
  }
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    const Tensor& two_embed_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (table_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of table should be 2"));
    OP_REQUIRES(context, (em_x_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (em_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of input should be 3"));
    OP_REQUIRES(context, (two_embed_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    TensorShape descriptor_shape;
    descriptor_shape.AddDim(em_tensor.shape().dim_size(0));
    descriptor_shape.AddDim(4);  // TODO: be careful here;
    descriptor_shape.AddDim(last_layer_size);
    int context_output_index = 0;
    Tensor* descriptor_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descriptor_shape,
                                                     &descriptor_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());
    // flat the tensors
    FPTYPE* descriptor = descriptor_tensor->flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* two_embed = two_embed_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_a_gpu(descriptor, table, table_info, em_x, em,
                                       two_embed, nloc, nnei, last_layer_size,
                                       is_sorted);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_a_cpu(descriptor, table, table_info, em_x, em,
                                       two_embed, nloc, nnei, last_layer_size,
                                       is_sorted);
    }
  }

 private:
  int last_layer_size;
  bool is_sorted;
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeAttenGradOp : public OpKernel {
 public:
  explicit TabulateFusionSeAttenGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_sorted", &is_sorted));
  }
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    const Tensor& two_embed_tensor = context->input(context_input_index++);

    const Tensor& dy_tensor = context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (dy_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of table should be 3"));
    int context_output_index = 0;
    Tensor* dy_dem_x_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     em_x_tensor.shape(),
                                                     &dy_dem_x_tensor));
    Tensor* dy_dem_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++,
                                            em_tensor.shape(), &dy_dem_tensor));
    Tensor* dy_dtwo_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     two_embed_tensor.shape(),
                                                     &dy_dtwo_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());

    // flat the tensors
    FPTYPE* dy_dem_x = dy_dem_x_tensor->flat<FPTYPE>().data();
    FPTYPE* dy_dem = dy_dem_tensor->flat<FPTYPE>().data();
    FPTYPE* dy_dtwo = dy_dtwo_tensor->flat<FPTYPE>().data();

    const FPTYPE* descriptor = descriptor_tensor.flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* two_embed = two_embed_tensor.flat<FPTYPE>().data();
    const FPTYPE* dy = dy_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);
    const int last_layer_size = descriptor_tensor.shape().dim_size(2);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_a_grad_gpu(
          dy_dem_x, dy_dem, dy_dtwo, table, table_info, em_x, em, two_embed, dy,
          nloc, nnei, last_layer_size, is_sorted);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_a_grad_cpu(
          dy_dem_x, dy_dem, dy_dtwo, table, table_info, em_x, em, two_embed, dy,
          nloc, nnei, last_layer_size, is_sorted);
    }
  }

 private:
  bool is_sorted;
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeAttenGradGradOp : public OpKernel {
 public:
  explicit TabulateFusionSeAttenGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_sorted", &is_sorted));
  }
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    const Tensor& two_embed_tensor = context->input(context_input_index++);
    const Tensor& dz_dy_dem_x_tensor = context->input(context_input_index++);
    const Tensor& dz_dy_dem_tensor = context->input(context_input_index++);
    const Tensor& dz_dy_dtwo_tensor = context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (dz_dy_dem_x_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (dz_dy_dem_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of input should be 3"));
    int context_output_index = 0;
    Tensor* dz_dy_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descriptor_tensor.shape(),
                                                     &dz_dy_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());

    // flat the tensors
    FPTYPE* dz_dy = dz_dy_tensor->flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* two_embed = two_embed_tensor.flat<FPTYPE>().data();
    const FPTYPE* dz_dy_dem_x = dz_dy_dem_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* dz_dy_dem = dz_dy_dem_tensor.flat<FPTYPE>().data();
    const FPTYPE* dz_dy_dtwo = dz_dy_dtwo_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);
    const int last_layer_size = descriptor_tensor.shape().dim_size(2);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_a_grad_grad_gpu(
          dz_dy, table, table_info, em_x, em, two_embed, dz_dy_dem_x, dz_dy_dem,
          dz_dy_dtwo, nloc, nnei, last_layer_size, is_sorted);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      OP_REQUIRES(context, (last_layer_size <= 1024),
                  errors::InvalidArgument(
                      "In the process of model compression, the size of the "
                      "last layer of embedding net must be less than 1024!"));
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_a_grad_grad_cpu(
          dz_dy, table, table_info, em_x, em, two_embed, dz_dy_dem_x, dz_dy_dem,
          dz_dy_dtwo, nloc, nnei, last_layer_size, is_sorted);
    }
  }

 private:
  bool is_sorted;
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeTOp : public OpKernel {
 public:
  explicit TabulateFusionSeTOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("last_layer_size", &last_layer_size));
  }
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (table_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of table should be 2"));
    OP_REQUIRES(context, (em_x_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of em_x_tensor should be 2"));
    OP_REQUIRES(context, (em_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of em_tensor should be 3"));
    TensorShape descriptor_shape;
    descriptor_shape.AddDim(em_tensor.shape().dim_size(0));
    descriptor_shape.AddDim(last_layer_size);
    int context_output_index = 0;
    Tensor* descriptor_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descriptor_shape,
                                                     &descriptor_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());
    // flat the tensors
    FPTYPE* descriptor = descriptor_tensor->flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei_i = em_tensor.shape().dim_size(1);
    const int nnei_j = em_tensor.shape().dim_size(2);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_t_gpu(descriptor, table, table_info, em_x, em,
                                       nloc, nnei_i, nnei_j, last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_t_cpu(descriptor, table, table_info, em_x, em,
                                       nloc, nnei_i, nnei_j, last_layer_size);
    }
  }

 private:
  int last_layer_size;
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeTGradOp : public OpKernel {
 public:
  explicit TabulateFusionSeTGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    const Tensor& dy_tensor = context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (dy_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of dy_tensor should be 2"));
    int context_output_index = 0;
    Tensor* dy_dem_x_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     em_x_tensor.shape(),
                                                     &dy_dem_x_tensor));
    Tensor* dy_dem_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++,
                                            em_tensor.shape(), &dy_dem_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());

    // flat the tensors
    FPTYPE* dy_dem_x = dy_dem_x_tensor->flat<FPTYPE>().data();
    FPTYPE* dy_dem = dy_dem_tensor->flat<FPTYPE>().data();
    const FPTYPE* descriptor = descriptor_tensor.flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* dy = dy_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei_i = em_tensor.shape().dim_size(1);
    const int nnei_j = em_tensor.shape().dim_size(2);
    const int last_layer_size = descriptor_tensor.shape().dim_size(1);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_t_grad_gpu(dy_dem_x, dy_dem, table, table_info,
                                            em_x, em, dy, nloc, nnei_i, nnei_j,
                                            last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_t_grad_cpu(dy_dem_x, dy_dem, table, table_info,
                                            em_x, em, dy, nloc, nnei_i, nnei_j,
                                            last_layer_size);
    }
  }

 private:
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeTGradGradOp : public OpKernel {
 public:
  explicit TabulateFusionSeTGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    const Tensor& dz_dy_dem_x_tensor = context->input(context_input_index++);
    const Tensor& dz_dy_dem_tensor = context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (dz_dy_dem_x_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (dz_dy_dem_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of input should be 3"));
    int context_output_index = 0;
    Tensor* dz_dy_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descriptor_tensor.shape(),
                                                     &dz_dy_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());

    // flat the tensors
    FPTYPE* dz_dy = dz_dy_tensor->flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* dz_dy_dem_x = dz_dy_dem_x_tensor.flat<FPTYPE>().data();
    const FPTYPE* dz_dy_dem = dz_dy_dem_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei_i = em_tensor.shape().dim_size(1);
    const int nnei_j = em_tensor.shape().dim_size(2);
    const int last_layer_size = descriptor_tensor.shape().dim_size(1);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_t_grad_grad_gpu(
          dz_dy, table, table_info, em_x, em, dz_dy_dem_x, dz_dy_dem, nloc,
          nnei_i, nnei_j, last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      OP_REQUIRES(context, (last_layer_size <= 1024),
                  errors::InvalidArgument(
                      "In the process of model compression, the size of the "
                      "last layer of embedding net must be less than 1024!"));
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_t_grad_grad_cpu(
          dz_dy, table, table_info, em_x, em, dz_dy_dem_x, dz_dy_dem, nloc,
          nnei_i, nnei_j, last_layer_size);
    }
  }

 private:
  std::string device;
};
template <typename Device, typename FPTYPE>
class TabulateFusionSeROp : public OpKernel {
 public:
  explicit TabulateFusionSeROp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("last_layer_size", &last_layer_size));
  }
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (table_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of table should be 2"));
    OP_REQUIRES(context, (em_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    TensorShape descriptor_shape;
    descriptor_shape.AddDim(em_tensor.shape().dim_size(0));
    descriptor_shape.AddDim(
        em_tensor.shape().dim_size(1));  // TODO: be careful here;
    descriptor_shape.AddDim(last_layer_size);
    int context_output_index = 0;
    Tensor* descriptor_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descriptor_shape,
                                                     &descriptor_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());
    // flat the tensors
    FPTYPE* descriptor = descriptor_tensor->flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_r_gpu(descriptor, table, table_info, em, nloc,
                                       nnei, last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_r_cpu(descriptor, table, table_info, em, nloc,
                                       nnei, last_layer_size);
    }
  }

 private:
  int last_layer_size;
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeRGradOp : public OpKernel {
 public:
  explicit TabulateFusionSeRGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    const Tensor& dy_tensor = context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (dy_tensor.shape().dims() == 3),
                errors::InvalidArgument("Dim of table should be 3"));
    int context_output_index = 0;
    Tensor* dy_dem_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++,
                                            em_tensor.shape(), &dy_dem_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());

    // flat the tensors
    FPTYPE* dy_dem = dy_dem_tensor->flat<FPTYPE>().data();
    const FPTYPE* descriptor = descriptor_tensor.flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* dy = dy_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);
    const int last_layer_size = descriptor_tensor.shape().dim_size(2);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_r_grad_gpu(dy_dem, table, table_info, em, dy,
                                            nloc, nnei, last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_r_grad_cpu(dy_dem, table, table_info, em, dy,
                                            nloc, nnei, last_layer_size);
    }
  }

 private:
  std::string device;
};

template <typename Device, typename FPTYPE>
class TabulateFusionSeRGradGradOp : public OpKernel {
 public:
  explicit TabulateFusionSeRGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor = context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_tensor = context->input(context_input_index++);
    const Tensor& dz_dy_dem_tensor = context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (dz_dy_dem_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    int context_output_index = 0;
    Tensor* dz_dy_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descriptor_tensor.shape(),
                                                     &dz_dy_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());

    // flat the tensors
    FPTYPE* dz_dy = dz_dy_tensor->flat<FPTYPE>().data();
    const FPTYPE* table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE* table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE* em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE* dz_dy_dem = dz_dy_dem_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);
    const int last_layer_size = descriptor_tensor.shape().dim_size(2);

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::tabulate_fusion_se_r_grad_grad_gpu(
          dz_dy, table, table_info, em, dz_dy_dem, nloc, nnei, last_layer_size);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      OP_REQUIRES(context, (last_layer_size <= 1024),
                  errors::InvalidArgument(
                      "In the process of model compression, the size of the "
                      "last layer of embedding net must be less than 1024!"));
    } else if (device == "CPU") {
      deepmd::tabulate_fusion_se_r_grad_grad_cpu(
          dz_dy, table, table_info, em, dz_dy_dem, nloc, nnei, last_layer_size);
    }
  }

 private:
  std::string device;
};

#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusion").Device(DEVICE_CPU).TypeConstraint<T>("T"),        \
      TabulateFusionSeAOp<CPUDevice, T>);                                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusionGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),    \
      TabulateFusionSeAGradOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionGradGrad")                       \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          TabulateFusionSeAGradGradOp<CPUDevice, T>);          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusionSeA").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      TabulateFusionSeAOp<CPUDevice, T>);                                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusionSeAGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TabulateFusionSeAGradOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeAGradGrad")                    \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          TabulateFusionSeAGradGradOp<CPUDevice, T>);          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusionSeAtten").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TabulateFusionSeAttenOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeAttenGrad")                    \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          TabulateFusionSeAttenGradOp<CPUDevice, T>);          \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeAttenGradGrad")                \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          TabulateFusionSeAttenGradGradOp<CPUDevice, T>);      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusionSeT").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      TabulateFusionSeTOp<CPUDevice, T>);                                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusionSeTGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TabulateFusionSeTGradOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeTGradGrad")                    \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          TabulateFusionSeTGradGradOp<CPUDevice, T>);          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusionSeR").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      TabulateFusionSeROp<CPUDevice, T>);                                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TabulateFusionSeRGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TabulateFusionSeRGradOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeRGradGrad")                    \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          TabulateFusionSeRGradGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusion")                          \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAOp<GPUDevice, T>);             \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionGrad")                      \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAGradOp<GPUDevice, T>);         \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionGradGrad")                  \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAGradGradOp<GPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeA")                       \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAOp<GPUDevice, T>);             \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeAGrad")                   \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAGradOp<GPUDevice, T>);         \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeAGradGrad")               \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAGradGradOp<GPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeAtten")                   \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAttenOp<GPUDevice, T>);         \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeAttenGrad")               \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAttenGradOp<GPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeAttenGradGrad")           \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeAttenGradGradOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeT")                       \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeTOp<GPUDevice, T>);             \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeTGrad")                   \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeTGradOp<GPUDevice, T>);         \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeTGradGrad")               \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeTGradGradOp<GPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeR")                       \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeROp<GPUDevice, T>);             \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeRGrad")                   \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeRGradOp<GPUDevice, T>);         \
  REGISTER_KERNEL_BUILDER(Name("TabulateFusionSeRGradGrad")               \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .HostMemory("table_info"),                  \
                          TabulateFusionSeRGradGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
