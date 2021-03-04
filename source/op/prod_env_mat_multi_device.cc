#include "custom_op.h"
#include "utilities.h"
#include "prod_env_mat.h"

REGISTER_OP("DescrptSeA")
    .Attr("T: {float, double}")
    .Input("coord: T")          //atomic coordinates
    .Input("type: int32")       //atomic type
    .Input("natoms: int32")     //local atomic number; each type atomic number; daizheyingxiangqude atomic numbers
    .Input("box : T")
    .Input("mesh : int32")
    .Input("davg: T")           //average value of data
    .Input("dstd: T")           //standard deviation
    .Attr("rcut_a: float")      //no use
    .Attr("rcut_r: float")
    .Attr("rcut_r_smth: float")
    .Attr("sel_a: list(int)")
    .Attr("sel_r: list(int)")   //all zero
    .Output("descrpt: T")
    .Output("descrpt_deriv: T")
    .Output("rij: T")
    .Output("nlist: int32");
    // only sel_a and rcut_r uesd.

REGISTER_OP("DescrptSeR")
    .Attr("T: {float, double}")
    .Input("coord: T")
    .Input("type: int32")
    .Input("natoms: int32")
    .Input("box: T")
    .Input("mesh: int32")
    .Input("davg: T")
    .Input("dstd: T")
    .Attr("rcut: float")
    .Attr("rcut_smth: float")
    .Attr("sel: list(int)")
    .Output("descrpt: T")
    .Output("descrpt_deriv: T")
    .Output("rij: T")
    .Output("nlist: int32");

template <typename Device, typename FPTYPE>
class DescrptSeAOp : public OpKernel {
public:
  explicit DescrptSeAOp(OpKernelConstruction* context) : OpKernel(context) {
    float nloc_f, nall_f;
    OP_REQUIRES_OK(context, context->GetAttr("rcut_a", &rcut_a));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r", &rcut_r));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r_smth", &rcut_r_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
    OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
    // OP_REQUIRES_OK(context, context->GetAttr("nloc", &nloc_f));
    // OP_REQUIRES_OK(context, context->GetAttr("nall", &nall_f));
    cum_sum (sec_a, sel_a);
    cum_sum (sec_r, sel_r);
    ndescrpt_a = sec_a.back() * 4;
    ndescrpt_r = sec_r.back() * 1;
    ndescrpt = ndescrpt_a + ndescrpt_r;
    nnei_a = sec_a.back();
    nnei_r = sec_r.back();
    nnei = nnei_a + nnei_r;
    max_nbor_size = 1024;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& coord_tensor	= context->input(context_input_index++);
    const Tensor& type_tensor	= context->input(context_input_index++);
    const Tensor& natoms_tensor	= context->input(context_input_index++);
    const Tensor& box_tensor	= context->input(context_input_index++);
    const Tensor& mesh_tensor   = context->input(context_input_index++);
    const Tensor& avg_tensor	= context->input(context_input_index++);
    const Tensor& std_tensor	= context->input(context_input_index++);
    // set size of the sample. assume 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], then shape(t) ==> [2, 2, 3]
    OP_REQUIRES (context, (coord_tensor.shape().dims() == 2),       errors::InvalidArgument ("Dim of coord should be 2"));
    OP_REQUIRES (context, (type_tensor.shape().dims() == 2),        errors::InvalidArgument ("Dim of type should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),      errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (box_tensor.shape().dims() == 2),         errors::InvalidArgument ("Dim of box should be 2"));
    OP_REQUIRES (context, (mesh_tensor.shape().dims() == 1),        errors::InvalidArgument ("Dim of mesh should be 1"));
    OP_REQUIRES (context, (avg_tensor.shape().dims() == 2),         errors::InvalidArgument ("Dim of avg should be 2"));
    OP_REQUIRES (context, (std_tensor.shape().dims() == 2),         errors::InvalidArgument ("Dim of std should be 2"));
    OP_REQUIRES (context, (sec_r.back() == 0),                      errors::InvalidArgument ("Rotational free descriptor only support all-angular information: sel_r should be all zero."));
    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    const int * natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int ntypes = natoms_tensor.shape().dim_size(0) - 2; //nloc and nall mean something.
    int nsamples = coord_tensor.shape().dim_size(0);
    //// check the sizes
    OP_REQUIRES (context, (nsamples == type_tensor.shape().dim_size(0)),  errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nsamples == box_tensor.shape().dim_size(0)),   errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (ntypes == avg_tensor.shape().dim_size(0)),     errors::InvalidArgument ("number of avg should be ntype"));
    OP_REQUIRES (context, (ntypes == std_tensor.shape().dim_size(0)),     errors::InvalidArgument ("number of std should be ntype"));
    
    OP_REQUIRES (context, (nall * 3 == coord_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (nall == type_tensor.shape().dim_size(1)),      errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (9 == box_tensor.shape().dim_size(1)),          errors::InvalidArgument ("number of box should be 9"));
    OP_REQUIRES (context, (ndescrpt == avg_tensor.shape().dim_size(1)),   errors::InvalidArgument ("number of avg should be ndescrpt"));
    OP_REQUIRES (context, (ndescrpt == std_tensor.shape().dim_size(1)),   errors::InvalidArgument ("number of std should be ndescrpt"));   
    
    OP_REQUIRES (context, (ntypes == int(sel_a.size())),  errors::InvalidArgument ("number of types should match the length of sel array"));
    OP_REQUIRES (context, (ntypes == int(sel_r.size())),  errors::InvalidArgument ("number of types should match the length of sel array"));
    // Create output tensors
    TensorShape descrpt_shape ;
    descrpt_shape.AddDim (nsamples);
    descrpt_shape.AddDim (nloc * ndescrpt);
    TensorShape descrpt_deriv_shape ;
    descrpt_deriv_shape.AddDim (nsamples);
    descrpt_deriv_shape.AddDim (nloc * ndescrpt * 3);
    TensorShape rij_shape ;
    rij_shape.AddDim (nsamples);
    rij_shape.AddDim (nloc * nnei * 3);
    TensorShape nlist_shape ;
    nlist_shape.AddDim (nsamples);
    nlist_shape.AddDim (nloc * nnei);
    // define output tensor
    int context_output_index = 0;
    Tensor* descrpt_tensor = NULL;
    Tensor* descrpt_deriv_tensor = NULL;
    Tensor* rij_tensor = NULL;
    Tensor* nlist_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
        descrpt_shape,
        &descrpt_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
        descrpt_deriv_shape,
        &descrpt_deriv_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
        rij_shape,
        &rij_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
        nlist_shape,
        &nlist_tensor));

    FPTYPE * em = descrpt_tensor->flat<FPTYPE>().data();
    FPTYPE * em_deriv = descrpt_deriv_tensor->flat<FPTYPE>().data();
    FPTYPE * rij = rij_tensor->flat<FPTYPE>().data();
    int * nlist = nlist_tensor->flat<int>().data();
    const FPTYPE * coord = coord_tensor.flat<FPTYPE>().data();
    const FPTYPE * avg = avg_tensor.flat<FPTYPE>().data();
    const FPTYPE * std = std_tensor.flat<FPTYPE>().data();
    const int * type = type_tensor.flat<int>().data();

    if(device == "GPU") {
      // allocate temp memory, temp memory must not be used after this operation!
      Tensor int_temp;
      TensorShape int_shape;
      int_shape.AddDim(sec_a.size() + nloc * sec_a.size() + nloc);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, int_shape, &int_temp));
      Tensor uint64_temp;
      TensorShape uint64_shape;
      uint64_shape.AddDim(nloc * GPU_MAX_NBOR_SIZE * 2);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape, &uint64_temp));
      array_int = int_temp.flat<int>().data(); 
      array_longlong = uint64_temp.flat<unsigned long long>().data();
      env_mat_nbor_update(
          init, ilist, jrange, jlist, ilist_size, jrange_size, jlist_size, max_nbor_size,
          mesh_tensor.flat<int>().data(), static_cast<int>(mesh_tensor.NumElements()));
      OP_REQUIRES (context, (max_nbor_size <= GPU_MAX_NBOR_SIZE), errors::InvalidArgument ("Assert failed, max neighbor size of atom(lammps) " + std::to_string(max_nbor_size) + " is larger than " + std::to_string(GPU_MAX_NBOR_SIZE) + ", which currently is not supported by deepmd-kit."));
      #if GOOGLE_CUDA
      // launch the gpu(nv) compute function
      prod_env_mat_a_gpu_cuda(
          em, em_deriv, rij, nlist, 
          coord, type, ilist, jrange, jlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, nall, rcut_r, rcut_r_smth, sec_a);
      #endif //GOOGLE_CUDA
    }
    else if (device == "CPU") {
      memcpy (&ilist,  4  + mesh_tensor.flat<int>().data(), sizeof(int *));
      memcpy (&jrange, 8  + mesh_tensor.flat<int>().data(), sizeof(int *));
      memcpy (&jlist,  12 + mesh_tensor.flat<int>().data(), sizeof(int *));
      // launch the cpu compute function
      prod_env_mat_a_cpu(
          em, em_deriv, rij, nlist, 
          coord, type, ilist, jrange, jlist, max_nbor_size, avg, std, nloc, nall, rcut_r, rcut_r_smth, sec_a);
    }
  }

/////////////////////////////////////////////////////////////////////////////////////////////
private:
  float rcut_a;
  float rcut_r;
  float rcut_r_smth;
  std::vector<int32> sel_r;
  std::vector<int32> sel_a;
  std::vector<int> sec_a;
  std::vector<int> sec_r;
  int ndescrpt, ndescrpt_a, ndescrpt_r;
  int nnei, nnei_a, nnei_r, nloc, nall, max_nbor_size;
  std::string device;
  int * array_int = NULL;
  unsigned long long * array_longlong = NULL;
  bool init = false;
  int * ilist = NULL, * jrange = NULL, * jlist = NULL;
  int ilist_size = 0, jrange_size = 0, jlist_size = 0;
};

template<typename Device, typename FPTYPE>
class DescrptSeROp : public OpKernel {
public:
  explicit DescrptSeROp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rcut", &rcut));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_smth", &rcut_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel", &sel));
    cum_sum (sec, sel);
    sel_null.resize(3, 0);
    cum_sum (sec_null, sel_null);
    ndescrpt = sec.back() * 1;
    nnei = sec.back();
    max_nbor_size = 1024;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& coord_tensor  = context->input(context_input_index++);
    const Tensor& type_tensor   = context->input(context_input_index++);
    const Tensor& natoms_tensor = context->input(context_input_index++);
    const Tensor& box_tensor    = context->input(context_input_index++);
    const Tensor& mesh_tensor   = context->input(context_input_index++);
    const Tensor& avg_tensor    = context->input(context_input_index++);
    const Tensor& std_tensor    = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES (context, (coord_tensor.shape().dims() == 2),	      errors::InvalidArgument ("Dim of coord should be 2"));
    OP_REQUIRES (context, (type_tensor.shape().dims() == 2),	      errors::InvalidArgument ("Dim of type should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),      errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (box_tensor.shape().dims() == 2),         errors::InvalidArgument ("Dim of box should be 2"));
    OP_REQUIRES (context, (mesh_tensor.shape().dims() == 1),        errors::InvalidArgument ("Dim of mesh should be 1"));
    OP_REQUIRES (context, (avg_tensor.shape().dims() == 2),         errors::InvalidArgument ("Dim of avg should be 2"));
    OP_REQUIRES (context, (std_tensor.shape().dims() == 2),         errors::InvalidArgument ("Dim of std should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    const int * natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int ntypes = natoms_tensor.shape().dim_size(0) - 2; //nloc and nall mean something.
    int nsamples = coord_tensor.shape().dim_size(0);
    //
    //// check the sizes
    // check the sizes
    OP_REQUIRES (context, (nsamples == type_tensor.shape().dim_size(0)),  errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nsamples == box_tensor.shape().dim_size(0)),   errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (ntypes == avg_tensor.shape().dim_size(0)),     errors::InvalidArgument ("number of avg should be ntype"));
    OP_REQUIRES (context, (ntypes == std_tensor.shape().dim_size(0)),     errors::InvalidArgument ("number of std should be ntype"));
    OP_REQUIRES (context, (nall * 3 == coord_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (nall == type_tensor.shape().dim_size(1)),      errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (9 == box_tensor.shape().dim_size(1)),          errors::InvalidArgument ("number of box should be 9"));
    OP_REQUIRES (context, (ndescrpt == avg_tensor.shape().dim_size(1)),   errors::InvalidArgument ("number of avg should be ndescrpt"));
    OP_REQUIRES (context, (ndescrpt == std_tensor.shape().dim_size(1)),   errors::InvalidArgument ("number of std should be ndescrpt"));
    // Create an output tensor
    TensorShape descrpt_shape ;
    descrpt_shape.AddDim (nsamples);
    descrpt_shape.AddDim (nloc * ndescrpt);
    TensorShape descrpt_deriv_shape ;
    descrpt_deriv_shape.AddDim (nsamples);
    descrpt_deriv_shape.AddDim (nloc * ndescrpt * 3);
    TensorShape rij_shape ;
    rij_shape.AddDim (nsamples);
    rij_shape.AddDim (nloc * nnei * 3);
    TensorShape nlist_shape ;
    nlist_shape.AddDim (nsamples);
    nlist_shape.AddDim (nloc * nnei);

    int context_output_index = 0;
    Tensor* descrpt_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
        descrpt_shape, 
        &descrpt_tensor));
    Tensor* descrpt_deriv_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
        descrpt_deriv_shape, 
        &descrpt_deriv_tensor));
    Tensor* rij_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
        rij_shape,
        &rij_tensor));
    Tensor* nlist_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
        nlist_shape,
        &nlist_tensor));

    FPTYPE * em = descrpt_tensor->flat<FPTYPE>().data();
    FPTYPE * em_deriv = descrpt_deriv_tensor->flat<FPTYPE>().data();
    FPTYPE * rij = rij_tensor->flat<FPTYPE>().data();
    int * nlist = nlist_tensor->flat<int>().data();
    const FPTYPE * coord = coord_tensor.flat<FPTYPE>().data();
    const FPTYPE * avg = avg_tensor.flat<FPTYPE>().data();
    const FPTYPE * std = std_tensor.flat<FPTYPE>().data();
    const int * type = type_tensor.flat<int>().data();

    if(device == "GPU") {
      // allocate temp memory, temp memory must not be used after this operation!
      Tensor int_temp;
      TensorShape int_shape;
      int_shape.AddDim(sec.size() + nloc * sec.size() + nloc);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, int_shape, &int_temp));
      Tensor uint64_temp;
      TensorShape uint64_shape;
      uint64_shape.AddDim(nloc * GPU_MAX_NBOR_SIZE * 2);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape, &uint64_temp));
      array_int = int_temp.flat<int>().data(); 
      array_longlong = uint64_temp.flat<unsigned long long>().data();
      env_mat_nbor_update(
          init, ilist, jrange, jlist, ilist_size, jrange_size, jlist_size, max_nbor_size,
          mesh_tensor.flat<int>().data(), static_cast<int>(mesh_tensor.NumElements()));
      OP_REQUIRES (context, (max_nbor_size <= GPU_MAX_NBOR_SIZE), errors::InvalidArgument ("Assert failed, max neighbor size of atom(lammps) " + std::to_string(max_nbor_size) + " is larger than " + std::to_string(GPU_MAX_NBOR_SIZE) + ", which currently is not supported by deepmd-kit."));
      #if GOOGLE_CUDA
      // launch the gpu(nv) compute function
      prod_env_mat_r_gpu_cuda(
          em, em_deriv, rij, nlist, 
          coord, type, ilist, jrange, jlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, nall, rcut, rcut_smth, sec);
      #endif //GOOGLE_CUDA
    }
    else if (device == "CPU") {
      memcpy (&ilist,  4  + mesh_tensor.flat<int>().data(), sizeof(int *));
      memcpy (&jrange, 8  + mesh_tensor.flat<int>().data(), sizeof(int *));
      memcpy (&jlist,  12 + mesh_tensor.flat<int>().data(), sizeof(int *));
      // launch the cpu compute function
      prod_env_mat_r_cpu(
          em, em_deriv, rij, nlist, 
          coord, type, ilist, jrange, jlist, max_nbor_size, avg, std, nloc, nall, rcut, rcut_smth, sec);
    }
  }

/////////////////////////////////////////////////////////////////////////////////////////////

private:
  float rcut;
  float rcut_smth;
  std::vector<int32> sel;
  std::vector<int32> sel_null;
  std::vector<int> sec;
  std::vector<int> sec_null;
  int nnei, ndescrpt, nloc, nall, max_nbor_size;
  std::string device;
  int * array_int = NULL;
  unsigned long long * array_longlong = NULL;
  bool init = false;
  int * ilist = NULL, * jrange = NULL, * jlist = NULL;
  int ilist_size = 0, jrange_size = 0, jlist_size = 0;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("DescrptSeA").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    DescrptSeAOp<CPUDevice, T>);                                                        \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("DescrptSeR").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    DescrptSeROp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("DescrptSeA").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"),  \
    DescrptSeAOp<GPUDevice, T>);                                                        \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("DescrptSeR").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"),  \
    DescrptSeROp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA
