#include "custom_op.h"
#include "prod_force_grad.h"

REGISTER_OP("ProdForceSeAGrad")
    .Attr("T: {float, double}")
    .Input("grad: T")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("grad_net: T");

REGISTER_OP("ProdForceSeRGrad")
    .Attr("T: {float, double}")
    .Input("grad: T")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("grad_net: T");

template<typename Device, typename FPTYPE>
class ProdForceSeAGradOp : public OpKernel {
public:
  explicit ProdForceSeAGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));    
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));    
    n_a_shift = n_a_sel * 4;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& grad_tensor		= context->input(context_input_index++);
    const Tensor& net_deriv_tensor	= context->input(context_input_index++);
    const Tensor& in_deriv_tensor	= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    TensorShape grad_shape		= grad_tensor.shape();
    TensorShape net_deriv_shape		= net_deriv_tensor.shape();
    TensorShape in_deriv_shape		= in_deriv_tensor.shape();
    TensorShape nlist_shape		= nlist_tensor.shape();

    OP_REQUIRES (context, (grad_shape.dims() == 2),	errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES (context, (net_deriv_shape.dims() == 2),errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_shape.dims() == 2), errors::InvalidArgument ("Dim of input deriv should be 2"));
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
    OP_REQUIRES (context, (nframes == nlist_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    
    OP_REQUIRES (context, (nloc * 3 == grad_shape.dim_size(1)),		errors::InvalidArgument ("input grad shape should be 3 x natoms"));
    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_shape.dim_size(1)),errors::InvalidArgument ("number of descriptors should match"));
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
    assert (nframes == nlist_tensor.shape().dim_size(0));
    assert (nloc * ndescrpt == grad_net_shape.dim_size(1));
    assert (nloc * 3 == grad_shape.dim_size(1));
    assert (nloc * ndescrpt == net_deriv_tensor.shape().dim_size(1));
    assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1));
    assert (nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert (nnei * 4 == ndescrpt);	
    // flat the tensors
    FPTYPE * p_grad_net = grad_net_tensor->flat<FPTYPE>().data();
    const FPTYPE * p_grad = grad_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_in_deriv = in_deriv_tensor.flat<FPTYPE>().data();
    const int * p_nlist	= nlist_tensor.flat<int>().data();

    for (int kk = 0; kk < nframes; ++kk){
        FPTYPE * grad_net = p_grad_net + kk * nloc * ndescrpt;
        const FPTYPE * grad = p_grad + kk * nloc * 3;
        const FPTYPE * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
        const int * nlist = p_nlist + kk * nloc * nnei; 
        if (device == "GPU") {
        #if GOOGLE_CUDA
        deepmd::prod_force_grad_a_gpu_cuda(    
            grad_net, 
            grad, in_deriv, nlist, nloc, nnei);
        #endif // GOOGLE_CUDA
        }
        else if (device == "CPU") {
        deepmd::prod_force_grad_a_cpu(    
            grad_net, 
            grad, in_deriv, nlist, nloc, nnei);
        }
    }
  }
private:
  std::string device;
  int n_r_sel, n_a_sel, n_a_shift;
};

template<typename Device, typename FPTYPE>
class ProdForceSeRGradOp : public OpKernel 
{
public:
  explicit ProdForceSeRGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& grad_tensor		= context->input(context_input_index++);
    const Tensor& net_deriv_tensor	= context->input(context_input_index++);
    const Tensor& in_deriv_tensor	= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    TensorShape grad_shape		= grad_tensor.shape();
    TensorShape net_deriv_shape		= net_deriv_tensor.shape();
    TensorShape in_deriv_shape		= in_deriv_tensor.shape();
    TensorShape nlist_shape		= nlist_tensor.shape();

    OP_REQUIRES (context, (grad_shape.dims() == 2),	errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES (context, (net_deriv_shape.dims() == 2),errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_shape.dims() == 2), errors::InvalidArgument ("Dim of input deriv should be 2"));
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
    OP_REQUIRES (context, (nframes == nlist_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    
    OP_REQUIRES (context, (nloc * 3 == grad_shape.dim_size(1)),		errors::InvalidArgument ("input grad shape should be 3 x natoms"));
    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_shape.dim_size(1)),errors::InvalidArgument ("number of descriptors should match"));

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
    assert (nframes == nlist_tensor.shape().dim_size(0));
    assert (nloc * ndescrpt == grad_net_shape.dim_size(1));
    assert (nloc * 3 == grad_shape.dim_size(1));
    assert (nloc * ndescrpt == net_deriv_tensor.shape().dim_size(1));
    assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1));
    assert (nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert (nnei * 1 == ndescrpt);	
    // flat the tensors
    FPTYPE * p_grad_net = grad_net_tensor->flat<FPTYPE>().data();
    const FPTYPE * p_grad = grad_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_in_deriv = in_deriv_tensor.flat<FPTYPE>().data();
    const int * p_nlist	= nlist_tensor.flat<int>().data();

    // loop over frames
    for (int kk = 0; kk < nframes; ++kk){
        FPTYPE * grad_net = p_grad_net + kk * nloc * ndescrpt;
        const FPTYPE * grad = p_grad + kk * nloc * 3;
        const FPTYPE * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
        const int * nlist = p_nlist + kk * nloc * nnei; 
        if (device == "GPU") {
          #if GOOGLE_CUDA
          deepmd::prod_force_grad_r_gpu_cuda(    
              grad_net, 
              grad, in_deriv, nlist, nloc, nnei);
          #endif // GOOGLE_CUDA
        }
        else if (device == "CPU") {
          deepmd::prod_force_grad_r_cpu(    
              grad_net, 
              grad, in_deriv, nlist, nloc, nnei);
        }
    }
  }
  private:
  std::string device;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                       \
REGISTER_KERNEL_BUILDER(                                                                      \
    Name("ProdForceSeAGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    ProdForceSeAGradOp<CPUDevice, T>);                                                        \
REGISTER_KERNEL_BUILDER(                                                                      \
    Name("ProdForceSeRGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    ProdForceSeRGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                      \
REGISTER_KERNEL_BUILDER(                                                                     \
    Name("ProdForceSeAGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdForceSeAGradOp<GPUDevice, T>);                                                       \
REGISTER_KERNEL_BUILDER(                                                                     \
    Name("ProdForceSeRGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdForceSeRGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA