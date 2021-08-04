#include "custom_op.h"
#include "prod_virial_grad.h"

REGISTER_OP("ProdVirialSeAGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("grad: T")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("rij: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("grad_net: T");

REGISTER_OP("ProdVirialSeRGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("grad: T")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("rij: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("grad_net: T");

template<typename Device, typename FPTYPE>
class ProdVirialSeAGradOp : public OpKernel 
{
public:
  explicit ProdVirialSeAGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));    
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));    
    n_a_shift = n_a_sel * 4;
  }

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(context, [this](OpKernelContext* context) {this->_Compute(context);});
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& grad_tensor		= context->input(context_input_index++);
    const Tensor& net_deriv_tensor	= context->input(context_input_index++);
    const Tensor& in_deriv_tensor	= context->input(context_input_index++);
    const Tensor& rij_tensor		= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    TensorShape grad_shape		= grad_tensor.shape();
    TensorShape net_deriv_shape		= net_deriv_tensor.shape();
    TensorShape in_deriv_shape		= in_deriv_tensor.shape();
    TensorShape rij_shape		= rij_tensor.shape();
    TensorShape nlist_shape		= nlist_tensor.shape();

    OP_REQUIRES (context, (grad_shape.dims() == 2),	errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES (context, (net_deriv_shape.dims() == 2),errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_shape.dims() == 2), errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (rij_shape.dims() == 2),	errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_shape.dims() == 2),	errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = net_deriv_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;

    // check the sizes
    OP_REQUIRES (context, (nframes == grad_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == in_deriv_shape.dim_size(0)),	errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == rij_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == nlist_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    
    OP_REQUIRES (context, (9 == grad_shape.dim_size(1)),		errors::InvalidArgument ("input grad shape should be 3 x natoms"));
    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_shape.dim_size(1)),errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_shape.dim_size(1)),	errors::InvalidArgument ("dim of rij should be  nnei * 3"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),			errors::InvalidArgument ("number of neighbors should match"));

    // Create an output tensor
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nframes);
    grad_net_shape.AddDim (nloc * ndescrpt);

    // allocate the output tensor
    Tensor* grad_net_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
        grad_net_shape, 
        &grad_net_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    assert (nframes == grad_net_shape.dim_size(0));
    assert (nframes == grad_shape.dim_size(0));
    assert (nframes == net_deriv_tensor.shape().dim_size(0));
    assert (nframes == in_deriv_tensor.shape().dim_size(0));
    assert (nframes == rij_tensor.shape().dim_size(0));
    assert (nframes == nlist_tensor.shape().dim_size(0));
    assert (nloc * ndescrpt == grad_net_shape.dim_size(1));
    assert (9 == grad_shape.dim_size(1));
    assert (nloc * ndescrpt == net_deriv_tensor.shape().dim_size(1));
    assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1));
    assert (nloc * nnei * 3 == rij_tensor.shape().dim_size(1));
    assert (nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert (nnei * 4 == ndescrpt);
    
    // flat the tensors
    FPTYPE * p_grad_net = grad_net_tensor->flat<FPTYPE>().data();
    const FPTYPE * p_grad = grad_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_in_deriv = in_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_rij = rij_tensor.flat<FPTYPE>().data();
    const int * p_nlist	= nlist_tensor.flat<int>().data();

    // loop over frames
    for (int kk = 0; kk < nframes; ++kk){
      FPTYPE * grad_net = p_grad_net + kk * nloc * ndescrpt;
      const FPTYPE * grad = p_grad + kk * 9;
      const FPTYPE * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
      const FPTYPE * rij = p_rij + kk * nloc * nnei * 3;
      const int * nlist = p_nlist + kk * nloc * nnei; 
      if (device == "GPU") {
        #if GOOGLE_CUDA
        deepmd::prod_virial_grad_a_gpu_cuda(    
          grad_net, 
          grad, in_deriv, rij, nlist, nloc, nnei);
        #endif // GOOGLE_CUDA
        
        #if TENSORFLOW_USE_ROCM
        deepmd::prod_virial_grad_a_gpu_rocm(    
          grad_net, 
          grad, in_deriv, rij, nlist, nloc, nnei);
        #endif // TENSORFLOW_USE_ROCM
      }
      else if (device == "CPU") {
        deepmd::prod_virial_grad_a_cpu(    
          grad_net, 
          grad, in_deriv, rij, nlist, nloc, nnei);
      }
    }
  }
private:
  std::string device;
  int n_r_sel, n_a_sel, n_a_shift;
};

template<typename Device, typename FPTYPE>
class ProdVirialSeRGradOp : public OpKernel 
{
public:
  explicit ProdVirialSeRGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(context, [this](OpKernelContext* context) {this->_Compute(context);});
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& grad_tensor		= context->input(context_input_index++);
    const Tensor& net_deriv_tensor	= context->input(context_input_index++);
    const Tensor& in_deriv_tensor	= context->input(context_input_index++);
    const Tensor& rij_tensor		= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    TensorShape grad_shape		= grad_tensor.shape();
    TensorShape net_deriv_shape		= net_deriv_tensor.shape();
    TensorShape in_deriv_shape		= in_deriv_tensor.shape();
    TensorShape rij_shape		= rij_tensor.shape();
    TensorShape nlist_shape		= nlist_tensor.shape();

    OP_REQUIRES (context, (grad_shape.dims() == 2),	errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES (context, (net_deriv_shape.dims() == 2),errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_shape.dims() == 2), errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (rij_shape.dims() == 2),	errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_shape.dims() == 2),	errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = net_deriv_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;

    // check the sizes
    OP_REQUIRES (context, (nframes == grad_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == in_deriv_shape.dim_size(0)),	errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == rij_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == nlist_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    
    OP_REQUIRES (context, (9 == grad_shape.dim_size(1)),		errors::InvalidArgument ("input grad shape should be 3 x natoms"));
    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_shape.dim_size(1)),errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_shape.dim_size(1)),	errors::InvalidArgument ("dim of rij should be  nnei * 3"));

    // Create an output tensor
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nframes);
    grad_net_shape.AddDim (nloc * ndescrpt);

    // allocate the output tensor
    Tensor* grad_net_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
        grad_net_shape, 
        &grad_net_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    assert (nframes == grad_net_shape.dim_size(0));
    assert (nframes == grad_shape.dim_size(0));
    assert (nframes == net_deriv_tensor.shape().dim_size(0));
    assert (nframes == in_deriv_tensor.shape().dim_size(0));
    assert (nframes == rij_tensor.shape().dim_size(0));
    assert (nframes == nlist_tensor.shape().dim_size(0));
    assert (nloc * ndescrpt == grad_net_shape.dim_size(1));
    assert (9 == grad_shape.dim_size(1));
    assert (nloc * ndescrpt == net_deriv_tensor.shape().dim_size(1));
    assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1));
    assert (nloc * nnei * 3 == rij_tensor.shape().dim_size(1));
    assert (nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert (nnei * 1 == ndescrpt);
    
    // flat the tensors
    FPTYPE * p_grad_net = grad_net_tensor->flat<FPTYPE>().data();
    const FPTYPE * p_grad = grad_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_in_deriv = in_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_rij = rij_tensor.flat<FPTYPE>().data();
    const int * p_nlist	= nlist_tensor.flat<int>().data();

    // loop over frames
    for (int kk = 0; kk < nframes; ++kk){
      FPTYPE * grad_net = p_grad_net + kk * nloc * ndescrpt;
      const FPTYPE * grad = p_grad + kk * 9;
      const FPTYPE * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
      const FPTYPE * rij = p_rij + kk * nloc * nnei * 3;
      const int * nlist = p_nlist + kk * nloc * nnei; 
      if (device == "GPU") {
        #if GOOGLE_CUDA
        deepmd::prod_virial_grad_r_gpu_cuda(    
          grad_net, 
          grad, in_deriv, rij, nlist, nloc, nnei);
        #endif // GOOGLE_CUDA
        
        #if TENSORFLOW_USE_ROCM
        deepmd::prod_virial_grad_r_gpu_rocm(    
          grad_net, 
          grad, in_deriv, rij, nlist, nloc, nnei);
        #endif // TENSORFLOW_USE_ROCM
      }
      else if (device == "CPU") {
        deepmd::prod_virial_grad_r_cpu(    
          grad_net, 
          grad, in_deriv, rij, nlist, nloc, nnei);
      }
    }
  }
private:
  std::string device;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                        \
REGISTER_KERNEL_BUILDER(                                                                       \
    Name("ProdVirialSeAGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    ProdVirialSeAGradOp<CPUDevice, T>);                                                        \
REGISTER_KERNEL_BUILDER(                                                                       \
    Name("ProdVirialSeRGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    ProdVirialSeRGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
// Register the GPU kernels.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(T)                                                                       \
REGISTER_KERNEL_BUILDER(                                                                      \
    Name("ProdVirialSeAGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdVirialSeAGradOp<GPUDevice, T>);                                                       \
REGISTER_KERNEL_BUILDER(                                                                      \
    Name("ProdVirialSeRGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdVirialSeRGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
