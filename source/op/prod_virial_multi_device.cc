#include "custom_op.h"
#include "prod_virial.h"

REGISTER_OP("ProdVirialSeA")
    .Attr("T: {float, double}")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("rij: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("virial: T")
    .Output("atom_virial: T");

REGISTER_OP("ProdVirialSeR")
    .Attr("T: {float, double}")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("rij: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("virial: T")
    .Output("atom_virial: T");

template<typename Device, typename FPTYPE>
class ProdVirialSeAOp : public OpKernel {
 public:
  explicit ProdVirialSeAOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& net_deriv_tensor  = context->input(context_input_index++);
    const Tensor& in_deriv_tensor   = context->input(context_input_index++);
    const Tensor& rij_tensor        = context->input(context_input_index++);
    const Tensor& nlist_tensor      = context->input(context_input_index++);
    const Tensor& natoms_tensor     = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES (context, (net_deriv_tensor.shape().dims() == 2),   errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_tensor.shape().dims() == 2),    errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (rij_tensor.shape().dims() == 2),         errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),       errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),      errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    const int * natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;
    int nframes = net_deriv_tensor.shape().dim_size(0);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    // check the sizes
    OP_REQUIRES (context, (nframes == in_deriv_tensor.shape().dim_size(0)), errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == rij_tensor.shape().dim_size(0)),      errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),    errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_tensor.shape().dim_size(1)),  errors::InvalidArgument ("dim of rij should be nnei * 3"));
    // Create an output tensor
    TensorShape virial_shape ;
    virial_shape.AddDim (nframes);
    virial_shape.AddDim (9);
    TensorShape atom_virial_shape;
    atom_virial_shape.AddDim (nframes);
    atom_virial_shape.AddDim (9 * nall);
    int context_output_index = 0;
    Tensor* virial_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
        virial_shape, 
        &virial_tensor));
    Tensor* atom_virial_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
        atom_virial_shape, 
        &atom_virial_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    // flat the tensors
    FPTYPE * p_virial = virial_tensor->flat<FPTYPE>().data();
    FPTYPE * p_atom_virial = atom_virial_tensor->flat<FPTYPE>().data();
    const FPTYPE * p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_in_deriv = in_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_rij = rij_tensor.flat<FPTYPE>().data();
    const int * p_nlist = nlist_tensor.flat<int>().data();
    
    for(int kk = 0; kk < nframes; ++kk){
      FPTYPE * virial = p_virial + kk * 9;
      FPTYPE * atom_virial = p_atom_virial + kk * nall * 9;
      const FPTYPE * net_deriv = p_net_deriv + kk * nloc * ndescrpt;
      const FPTYPE * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
      const FPTYPE * rij = p_rij + kk * nloc * nnei * 3;
      const int * nlist = p_nlist + kk * nloc * nnei;      
    if (device == "GPU") {
      #if GOOGLE_CUDA
      deepmd::prod_virial_a_gpu_cuda(    
          virial, atom_virial,
          net_deriv, in_deriv, rij, nlist, nloc, nall, nnei);
      #endif // GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::prod_virial_a_cpu(    
          virial, atom_virial,
          net_deriv, in_deriv, rij, nlist, nloc, nall, nnei);
    }
    }
  }
 private:
  std::string device;
};

template<typename Device, typename FPTYPE>
class ProdVirialSeROp : public OpKernel {
 public:
  explicit ProdVirialSeROp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& net_deriv_tensor  = context->input(context_input_index++);
    const Tensor& in_deriv_tensor   = context->input(context_input_index++);
    const Tensor& rij_tensor        = context->input(context_input_index++);
    const Tensor& nlist_tensor      = context->input(context_input_index++);
    const Tensor& natoms_tensor     = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES (context, (net_deriv_tensor.shape().dims() == 2),   errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_tensor.shape().dims() == 2),    errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (rij_tensor.shape().dims() == 2),         errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),       errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),      errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    const int * natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;
    int nframes = net_deriv_tensor.shape().dim_size(0);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    // check the sizes
    OP_REQUIRES (context, (nframes == in_deriv_tensor.shape().dim_size(0)), errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == rij_tensor.shape().dim_size(0)),      errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),    errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_tensor.shape().dim_size(1)),  errors::InvalidArgument ("dim of rij should be nnei * 3"));
    // Create an output tensor
    TensorShape virial_shape ;
    virial_shape.AddDim (nframes);
    virial_shape.AddDim (9);
    TensorShape atom_virial_shape;
    atom_virial_shape.AddDim (nframes);
    atom_virial_shape.AddDim (9 * nall);
    int context_output_index = 0;
    Tensor* virial_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
        virial_shape, 
        &virial_tensor));
    Tensor* atom_virial_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
        atom_virial_shape, 
        &atom_virial_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    // flat the tensors
    FPTYPE * p_virial = virial_tensor->flat<FPTYPE>().data();
    FPTYPE * p_atom_virial = atom_virial_tensor->flat<FPTYPE>().data();
    const FPTYPE * p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_in_deriv = in_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_rij = rij_tensor.flat<FPTYPE>().data();
    const int * p_nlist = nlist_tensor.flat<int>().data();
    
    for(int kk = 0; kk < nframes; ++kk){
      FPTYPE * virial = p_virial + kk * 9;
      FPTYPE * atom_virial = p_atom_virial + kk * nall * 9;
      const FPTYPE * net_deriv = p_net_deriv + kk * nloc * ndescrpt;
      const FPTYPE * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
      const FPTYPE * rij = p_rij + kk * nloc * nnei * 3;
      const int * nlist = p_nlist + kk * nloc * nnei;      
    if (device == "GPU") {
      #if GOOGLE_CUDA
      deepmd::prod_virial_r_gpu_cuda(    
          virial, atom_virial,
          net_deriv, in_deriv, rij, nlist, nloc, nall, nnei);
      #endif // GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::prod_virial_r_cpu(    
          virial, atom_virial,
          net_deriv, in_deriv, rij, nlist, nloc, nall, nnei);
    }
    }
  }
 private:
  std::string device;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                   \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("ProdVirialSeA").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    ProdVirialSeAOp<CPUDevice, T>);                                                       \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("ProdVirialSeR").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    ProdVirialSeROp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                   \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("ProdVirialSeA").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdVirialSeAOp<GPUDevice, T>);                                                       \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("ProdVirialSeR").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdVirialSeROp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA
