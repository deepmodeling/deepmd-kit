#include "custom_op.h"
#include "utilities.h"
#include "coord.h"
#include "region.h"
#include "neighbor_list.h"
#include "prod_env_mat.h"

REGISTER_OP("ProdEnvMatA")
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

REGISTER_OP("ProdEnvMatR")
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


template<typename FPTYPE>
static int
_norm_copy_coord_cpu(
    std::vector<FPTYPE> & coord_cpy,
    std::vector<int> & type_cpy,
    std::vector<int> & mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r);

template<typename FPTYPE>
static int
_build_nlist_cpu(
    std::vector<int> &ilist, 
    std::vector<int> &numneigh,
    std::vector<int*> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r);

static void
_map_nlist_cpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei);

template <typename FPTYPE>
static void
_prepare_coord_nlist_cpu(
    OpKernelContext* context,
    FPTYPE const ** coord,
    std::vector<FPTYPE> & coord_cpy,
    int const** type,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    deepmd::InputNlist & inlist,
    std::vector<int> & ilist,
    std::vector<int> & numneigh,
    std::vector<int*> & firstneigh,
    std::vector<std::vector<int>> & jlist,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial);



template <typename Device, typename FPTYPE>
class ProdEnvMatAOp : public OpKernel {
public:
  explicit ProdEnvMatAOp(OpKernelConstruction* context) : OpKernel(context) {
    float nloc_f, nall_f;
    OP_REQUIRES_OK(context, context->GetAttr("rcut_a", &rcut_a));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r", &rcut_r));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r_smth", &rcut_r_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
    OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
    // OP_REQUIRES_OK(context, context->GetAttr("nloc", &nloc_f));
    // OP_REQUIRES_OK(context, context->GetAttr("nall", &nall_f));
    deepmd::cum_sum (sec_a, sel_a);
    deepmd::cum_sum (sec_r, sel_r);
    ndescrpt_a = sec_a.back() * 4;
    ndescrpt_r = sec_r.back() * 1;
    ndescrpt = ndescrpt_a + ndescrpt_r;
    nnei_a = sec_a.back();
    nnei_r = sec_r.back();
    nnei = nnei_a + nnei_r;
    max_nbor_size = 1024;
    max_cpy_trial = 100;
    mem_cpy = 256;
    max_nnei_trial = 100;
    mem_nnei = 256;
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

    int nei_mode = 0;
    bool b_nlist_map = false;
    if (mesh_tensor.shape().dim_size(0) == 16) {
      // lammps neighbor list
      nei_mode = 3;
    }
    else if (mesh_tensor.shape().dim_size(0) == 6) {
      // manual copied pbc
      assert (nloc == nall);
      nei_mode = 1;
      b_nlist_map = true;
    }
    else if (mesh_tensor.shape().dim_size(0) == 0) {
      // no pbc
      assert (nloc == nall);
      nei_mode = -1;
    }
    else {
      throw std::runtime_error("invalid mesh tensor");
    }

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

    FPTYPE * p_em = descrpt_tensor->flat<FPTYPE>().data();
    FPTYPE * p_em_deriv = descrpt_deriv_tensor->flat<FPTYPE>().data();
    FPTYPE * p_rij = rij_tensor->flat<FPTYPE>().data();
    int * p_nlist = nlist_tensor->flat<int>().data();
    const FPTYPE * p_coord = coord_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_box = box_tensor.flat<FPTYPE>().data();
    const FPTYPE * avg = avg_tensor.flat<FPTYPE>().data();
    const FPTYPE * std = std_tensor.flat<FPTYPE>().data();
    const int * p_type = type_tensor.flat<int>().data();

    // loop over samples
    for(int ff = 0; ff < nsamples; ++ff){
      FPTYPE * em = p_em + ff*nloc*ndescrpt;
      FPTYPE * em_deriv = p_em_deriv + ff*nloc*ndescrpt*3;
      FPTYPE * rij = p_rij + ff*nloc*nnei*3;
      int * nlist = p_nlist + ff*nloc*nnei;
      const FPTYPE * coord = p_coord + ff*nall*3;
      const FPTYPE * box = p_box + ff*9;
      const int * type = p_type + ff*nall;

    if(device == "GPU") {
      #if GOOGLE_CUDA
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

      // update nbor list
      deepmd::InputNlist inlist;
      inlist.inum = nloc;
      deepmd::env_mat_nbor_update(
          inlist, gpu_inlist, max_nbor_size, nbor_list_dev,
          mesh_tensor.flat<int>().data(), static_cast<int>(mesh_tensor.NumElements()));
      OP_REQUIRES (context, (max_numneigh(inlist) <= GPU_MAX_NBOR_SIZE), errors::InvalidArgument ("Assert failed, max neighbor size of atom(lammps) " + std::to_string(max_numneigh(inlist)) + " is larger than " + std::to_string(GPU_MAX_NBOR_SIZE) + ", which currently is not supported by deepmd-kit."));
      // launch the gpu(nv) compute function
      deepmd::prod_env_mat_a_gpu_cuda(
          em, em_deriv, rij, nlist, 
          coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, nall, rcut_r, rcut_r_smth, sec_a);
      #endif //GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::InputNlist inlist;
      // some buffers, be freed after the evaluation of this frame
      std::vector<int> idx_mapping;
      std::vector<int> ilist(nloc), numneigh(nloc);
      std::vector<int*> firstneigh(nloc);
      std::vector<std::vector<int>> jlist(nloc);
      std::vector<FPTYPE> coord_cpy;
      std::vector<int> type_cpy;
      int frame_nall = nall;
      // prepare coord and nlist
      _prepare_coord_nlist_cpu<FPTYPE>(
	  context, &coord, coord_cpy, &type, type_cpy, idx_mapping, 
	  inlist, ilist, numneigh, firstneigh, jlist,
	  frame_nall, mem_cpy, mem_nnei, max_nbor_size,
	  box, mesh_tensor.flat<int>().data(), nloc, nei_mode, rcut_r, max_cpy_trial, max_nnei_trial);
      // launch the cpu compute function
      deepmd::prod_env_mat_a_cpu(
	  em, em_deriv, rij, nlist, 
	  coord, type, inlist, max_nbor_size, avg, std, nloc, frame_nall, rcut_r, rcut_r_smth, sec_a);
      // do nlist mapping if coords were copied
      if(b_nlist_map) _map_nlist_cpu(nlist, &idx_mapping[0], nloc, nnei);
    }
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
  int mem_cpy, max_cpy_trial;
  int mem_nnei, max_nnei_trial;
  std::string device;
  int * array_int = NULL;
  unsigned long long * array_longlong = NULL;
  deepmd::InputNlist gpu_inlist;
  int * nbor_list_dev = NULL;
};

template<typename Device, typename FPTYPE>
class ProdEnvMatROp : public OpKernel {
public:
  explicit ProdEnvMatROp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rcut", &rcut));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_smth", &rcut_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel", &sel));
    deepmd::cum_sum (sec, sel);
    sel_null.resize(3, 0);
    deepmd::cum_sum (sec_null, sel_null);
    ndescrpt = sec.back() * 1;
    nnei = sec.back();
    max_nbor_size = 1024;
    max_cpy_trial = 100;
    mem_cpy = 256;
    max_nnei_trial = 100;
    mem_nnei = 256;
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

    int nei_mode = 0;
    bool b_nlist_map = false;
    if (mesh_tensor.shape().dim_size(0) == 16) {
      // lammps neighbor list
      nei_mode = 3;
    }
    else if (mesh_tensor.shape().dim_size(0) == 6) {
      // manual copied pbc
      assert (nloc == nall);
      nei_mode = 1;
      b_nlist_map = true;
    }
    else if (mesh_tensor.shape().dim_size(0) == 0) {
      // no pbc
      assert (nloc == nall);
      nei_mode = -1;
    }
    else {
      throw std::runtime_error("invalid mesh tensor");
    }

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

    FPTYPE * p_em = descrpt_tensor->flat<FPTYPE>().data();
    FPTYPE * p_em_deriv = descrpt_deriv_tensor->flat<FPTYPE>().data();
    FPTYPE * p_rij = rij_tensor->flat<FPTYPE>().data();
    int * p_nlist = nlist_tensor->flat<int>().data();
    const FPTYPE * p_coord = coord_tensor.flat<FPTYPE>().data();
    const FPTYPE * p_box = box_tensor.flat<FPTYPE>().data();
    const FPTYPE * avg = avg_tensor.flat<FPTYPE>().data();
    const FPTYPE * std = std_tensor.flat<FPTYPE>().data();
    const int * p_type = type_tensor.flat<int>().data();

    // loop over samples
    for(int ff = 0; ff < nsamples; ++ff){
      FPTYPE * em = p_em + ff*nloc*ndescrpt;
      FPTYPE * em_deriv = p_em_deriv + ff*nloc*ndescrpt*3;
      FPTYPE * rij = p_rij + ff*nloc*nnei*3;
      int * nlist = p_nlist + ff*nloc*nnei;
      const FPTYPE * coord = p_coord + ff*nall*3;
      const FPTYPE * box = p_box + ff*9;
      const int * type = p_type + ff*nall;

    if(device == "GPU") {
      #if GOOGLE_CUDA
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
      
      // update nbor list
      deepmd::InputNlist inlist;
      inlist.inum = nloc;
      deepmd::env_mat_nbor_update(
          inlist, gpu_inlist, max_nbor_size, nbor_list_dev,
          mesh_tensor.flat<int>().data(), static_cast<int>(mesh_tensor.NumElements()));
      OP_REQUIRES (context, (max_numneigh(inlist) <= GPU_MAX_NBOR_SIZE), errors::InvalidArgument ("Assert failed, max neighbor size of atom(lammps) " + std::to_string(max_numneigh(inlist)) + " is larger than " + std::to_string(GPU_MAX_NBOR_SIZE) + ", which currently is not supported by deepmd-kit."));
      // launch the gpu(nv) compute function
      deepmd::prod_env_mat_r_gpu_cuda(
          em, em_deriv, rij, nlist, 
          coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, nall, rcut, rcut_smth, sec);
      #endif //GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::InputNlist inlist;
      // some buffers, be freed after the evaluation of this frame
      std::vector<int> idx_mapping;
      std::vector<int> ilist(nloc), numneigh(nloc);
      std::vector<int*> firstneigh(nloc);
      std::vector<std::vector<int>> jlist(nloc);
      std::vector<FPTYPE> coord_cpy;
      std::vector<int> type_cpy;
      int frame_nall = nall;
      // prepare coord and nlist
      _prepare_coord_nlist_cpu<FPTYPE>(
	  context, &coord, coord_cpy, &type, type_cpy, idx_mapping, 
	  inlist, ilist, numneigh, firstneigh, jlist,
	  frame_nall, mem_cpy, mem_nnei, max_nbor_size,
	  box, mesh_tensor.flat<int>().data(), nloc, nei_mode, rcut, max_cpy_trial, max_nnei_trial);
      // launch the cpu compute function
      prod_env_mat_r_cpu(
          em, em_deriv, rij, nlist, 
          coord, type, inlist, max_nbor_size, avg, std, nloc, frame_nall, rcut, rcut_smth, sec);
      if(b_nlist_map) _map_nlist_cpu(nlist, &idx_mapping[0], nloc, nnei);
    }
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
  int mem_cpy, max_cpy_trial;
  int mem_nnei, max_nnei_trial;
  std::string device;
  int * array_int = NULL;
  unsigned long long * array_longlong = NULL;
  deepmd::InputNlist gpu_inlist;
  int * nbor_list_dev = NULL;
};




template<typename FPTYPE>
static int
_norm_copy_coord_cpu(
    std::vector<FPTYPE> & coord_cpy,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r)
{
  std::vector<FPTYPE> tmp_coord(nall*3);
  std::copy(coord, coord+nall*3, tmp_coord.begin());
  deepmd::Region<FPTYPE> region;
  init_region_cpu(region, box);
  normalize_coord_cpu(&tmp_coord[0], nall, region);
  int tt;
  for(tt = 0; tt < max_cpy_trial; ++tt){
    coord_cpy.resize(mem_cpy*3);
    type_cpy.resize(mem_cpy);
    idx_mapping.resize(mem_cpy);
    int ret = copy_coord_cpu(
	&coord_cpy[0], &type_cpy[0], &idx_mapping[0], &nall, 
	&tmp_coord[0], type, nloc, mem_cpy, rcut_r, region);
    if(ret == 0){
      break;
    }
    else{
      mem_cpy *= 2;
    }
  }
  return (tt != max_cpy_trial);
}

template<typename FPTYPE>
static int
_build_nlist_cpu(
    std::vector<int> &ilist, 
    std::vector<int> &numneigh,
    std::vector<int*> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r)
{
  int tt;
  for(tt = 0; tt < max_nnei_trial; ++tt){
    for(int ii = 0; ii < nloc; ++ii){
      jlist[ii].resize(mem_nnei);
      firstneigh[ii] = &jlist[ii][0];
    }
    deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
    int ret = build_nlist_cpu(
	inlist, &max_nnei, 
	coord, nloc, new_nall, mem_nnei, rcut_r);
    if(ret == 0){
      break;
    }
    else{
      mem_nnei *= 2;
    }
  }
  return (tt != max_nnei_trial);
}
    
static void
_map_nlist_cpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei)
{
  for (int ii = 0; ii < nloc; ++ii){
    for (int jj = 0; jj < nnei; ++jj){
      int record = nlist[ii*nnei+jj];
      if (record >= 0) {		
	nlist[ii*nnei+jj] = idx_mapping[record];	      
      }
    }
  }  
}

template <typename FPTYPE>
static void
_prepare_coord_nlist_cpu(
    OpKernelContext* context,
    FPTYPE const ** coord,
    std::vector<FPTYPE> & coord_cpy,
    int const** type,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    deepmd::InputNlist & inlist,
    std::vector<int> & ilist,
    std::vector<int> & numneigh,
    std::vector<int*> & firstneigh,
    std::vector<std::vector<int>> & jlist,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial)
{    
  inlist.inum = nloc;
  if(nei_mode != 3){
    // build nlist by myself
    // normalize and copy coord
    if(nei_mode == 1){
      int copy_ok = _norm_copy_coord_cpu(
	  coord_cpy, type_cpy, idx_mapping, new_nall, mem_cpy,
	  *coord, box, *type, nloc, max_cpy_trial, rcut_r);
      OP_REQUIRES (context, copy_ok, errors::Aborted("cannot allocate mem for copied coords"));
      *coord = &coord_cpy[0];
      *type = &type_cpy[0];
    }
    // build nlist
    int build_ok = _build_nlist_cpu(
	ilist, numneigh, firstneigh, jlist, max_nbor_size, mem_nnei,
	*coord, nloc, new_nall, max_nnei_trial, rcut_r);
    OP_REQUIRES (context, build_ok, errors::Aborted("cannot allocate mem for nlist"));
    inlist.ilist = &ilist[0];
    inlist.numneigh = &numneigh[0];
    inlist.firstneigh = &firstneigh[0];
  }
  else{
    // copy pointers to nlist data
    memcpy(&inlist.ilist, 4 + mesh_tensor_data, sizeof(int *));
    memcpy(&inlist.numneigh, 8 + mesh_tensor_data, sizeof(int *));
    memcpy(&inlist.firstneigh, 12 + mesh_tensor_data, sizeof(int **));
    max_nbor_size = max_numneigh(inlist);
  }
}


// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("ProdEnvMatA").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    ProdEnvMatAOp<CPUDevice, T>);                                                       \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("ProdEnvMatR").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    ProdEnvMatROp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("ProdEnvMatA").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdEnvMatAOp<GPUDevice, T>);                                                       \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("ProdEnvMatR").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdEnvMatROp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA
