// SPDX-License-Identifier: LGPL-3.0-or-later
/*
//==================================================
 _   _  __     __  _   _   __  __   ____
| \ | | \ \   / / | \ | | |  \/  | |  _ \
|  \| |  \ \ / /  |  \| | | |\/| | | | | |
| |\  |   \ V /   | |\  | | |  | | | |_| |
|_| \_|    \_/    |_| \_| |_|  |_| |____/

//==================================================

code: nvnmd
reference: deepmd
author: mph (pinghui_mo@outlook.com)
date: 2021-12-6

*/

#include "coord.h"
#include "custom_op.h"
#include "errors.h"
#include "neighbor_list.h"
#include "prod_env_mat_nvnmd.h"
#include "region.h"
#include "utilities.h"

// ProdEnvMatANvnmd
// have been remove for the same function

REGISTER_OP("ProdEnvMatANvnmdQuantize")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("coord: T")       // atomic coordinates
    .Input("type: int32")    // atomic type
    .Input("natoms: int32")  // local atomic number; each type atomic number
    .Input("box : T")
    .Input("mesh : int32")
    .Input("davg: T")       // average value of data
    .Input("dstd: T")       // standard deviation
    .Attr("rcut_a: float")  // no use
    .Attr("rcut_r: float")
    .Attr("rcut_r_smth: float")
    .Attr("sel_a: list(int)")
    .Attr("sel_r: list(int)")  // all zero
    .Output("descrpt: T")
    .Output("descrpt_deriv: T")
    .Output("rij: T")
    .Output("nlist: int32");
// only sel_a and rcut_r uesd.

// ProdEnvMatAMixNvnmd
REGISTER_OP("ProdEnvMatAMixNvnmdQuantize")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("coord: T")       // atomic coordinates
    .Input("type: int32")    // atomic type
    .Input("natoms: int32")  // local atomic number; each type atomic number
    .Input("box : T")
    .Input("mesh : int32")
    .Input("davg: T")       // average value of data
    .Input("dstd: T")       // standard deviation
    .Attr("rcut_a: float")  // no use
    .Attr("rcut_r: float")
    .Attr("rcut_r_smth: float")
    .Attr("sel_a: list(int)")
    .Attr("sel_r: list(int)")  // all zero
    .Output("descrpt: T")
    .Output("descrpt_deriv: T")
    .Output("rij: T")
    .Output("nlist: int32")
    .Output("ntype: int32")
    .Output("nmask: bool");
// only sel_a and rcut_r uesd.

template <typename FPTYPE>
static int _norm_copy_coord_cpu(std::vector<FPTYPE>& coord_cpy,
                                std::vector<int>& type_cpy,
                                std::vector<int>& mapping,
                                int& nall,
                                int& mem_cpy,
                                const FPTYPE* coord,
                                const FPTYPE* box,
                                const int* type,
                                const int& nloc,
                                const int& max_cpy_trial,
                                const float& rcut_r);

template <typename FPTYPE>
static int _build_nlist_cpu(std::vector<int>& ilist,
                            std::vector<int>& numneigh,
                            std::vector<int*>& firstneigh,
                            std::vector<std::vector<int>>& jlist,
                            int& max_nnei,
                            int& mem_nnei,
                            const FPTYPE* coord,
                            const int& nloc,
                            const int& new_nall,
                            const int& max_nnei_trial,
                            const float& rcut_r);

static void _map_nlist_cpu(int* nlist,
                           const int* idx_mapping,
                           const int& nloc,
                           const int& nnei);

static void _map_nei_info_cpu(int* nlist,
                              int* ntype,
                              bool* nmask,
                              const int* type,
                              const int* idx_mapping,
                              const int& nloc,
                              const int& nnei,
                              const int& ntypes,
                              const bool& b_nlist_map);

template <typename FPTYPE>
static void _prepare_coord_nlist_cpu(OpKernelContext* context,
                                     FPTYPE const** coord,
                                     std::vector<FPTYPE>& coord_cpy,
                                     int const** type,
                                     std::vector<int>& type_cpy,
                                     std::vector<int>& idx_mapping,
                                     deepmd::InputNlist& inlist,
                                     std::vector<int>& ilist,
                                     std::vector<int>& numneigh,
                                     std::vector<int*>& firstneigh,
                                     std::vector<std::vector<int>>& jlist,
                                     int& new_nall,
                                     int& mem_cpy,
                                     int& mem_nnei,
                                     int& max_nbor_size,
                                     const FPTYPE* box,
                                     const int* mesh_tensor_data,
                                     const int& nloc,
                                     const int& nei_mode,
                                     const float& rcut_r,
                                     const int& max_cpy_trial,
                                     const int& max_nnei_trial);

// instance of function

template <typename FPTYPE>
static int _norm_copy_coord_cpu(std::vector<FPTYPE>& coord_cpy,
                                std::vector<int>& type_cpy,
                                std::vector<int>& idx_mapping,
                                int& nall,
                                int& mem_cpy,
                                const FPTYPE* coord,
                                const FPTYPE* box,
                                const int* type,
                                const int& nloc,
                                const int& max_cpy_trial,
                                const float& rcut_r) {
  std::vector<FPTYPE> tmp_coord(nall * 3);
  std::copy(coord, coord + nall * 3, tmp_coord.begin());
  deepmd::Region<FPTYPE> region;
  init_region_cpu(region, box);
  normalize_coord_cpu(&tmp_coord[0], nall, region);
  int tt;
  for (tt = 0; tt < max_cpy_trial; ++tt) {
    coord_cpy.resize(static_cast<size_t>(mem_cpy) * 3);
    type_cpy.resize(mem_cpy);
    idx_mapping.resize(mem_cpy);
    int ret =
        copy_coord_cpu(&coord_cpy[0], &type_cpy[0], &idx_mapping[0], &nall,
                       &tmp_coord[0], type, nloc, mem_cpy, rcut_r, region);
    if (ret == 0) {
      break;
    } else {
      mem_cpy *= 2;
    }
  }
  return (tt != max_cpy_trial);
}

template <typename FPTYPE>
static int _build_nlist_cpu(std::vector<int>& ilist,
                            std::vector<int>& numneigh,
                            std::vector<int*>& firstneigh,
                            std::vector<std::vector<int>>& jlist,
                            int& max_nnei,
                            int& mem_nnei,
                            const FPTYPE* coord,
                            const int& nloc,
                            const int& new_nall,
                            const int& max_nnei_trial,
                            const float& rcut_r) {
  int tt;
  for (tt = 0; tt < max_nnei_trial; ++tt) {
    for (int ii = 0; ii < nloc; ++ii) {
      jlist[ii].resize(mem_nnei);
      firstneigh[ii] = &jlist[ii][0];
    }
    deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
    int ret = build_nlist_cpu(inlist, &max_nnei, coord, nloc, new_nall,
                              mem_nnei, rcut_r);
    if (ret == 0) {
      break;
    } else {
      mem_nnei *= 2;
    }
  }
  return (tt != max_nnei_trial);
}

static void _map_nlist_cpu(int* nlist,
                           const int* idx_mapping,
                           const int& nloc,
                           const int& nnei) {
  for (int ii = 0; ii < nloc; ++ii) {
    for (int jj = 0; jj < nnei; ++jj) {
      int record = nlist[ii * nnei + jj];
      if (record >= 0) {
        nlist[ii * nnei + jj] = idx_mapping[record];
      }
    }
  }
}

static void _map_nei_info_cpu(int* nlist,
                              int* ntype,
                              bool* nmask,
                              const int* type,
                              const int* idx_mapping,
                              const int& nloc,
                              const int& nnei,
                              const int& ntypes,
                              const bool& b_nlist_map) {
  deepmd::use_nei_info_cpu(nlist, ntype, nmask, type, idx_mapping, nloc, nnei,
                           ntypes, b_nlist_map);
}

template <typename FPTYPE>
static void _prepare_coord_nlist_cpu(OpKernelContext* context,
                                     FPTYPE const** coord,
                                     std::vector<FPTYPE>& coord_cpy,
                                     int const** type,
                                     std::vector<int>& type_cpy,
                                     std::vector<int>& idx_mapping,
                                     deepmd::InputNlist& inlist,
                                     std::vector<int>& ilist,
                                     std::vector<int>& numneigh,
                                     std::vector<int*>& firstneigh,
                                     std::vector<std::vector<int>>& jlist,
                                     int& new_nall,
                                     int& mem_cpy,
                                     int& mem_nnei,
                                     int& max_nbor_size,
                                     const FPTYPE* box,
                                     const int* mesh_tensor_data,
                                     const int& nloc,
                                     const int& nei_mode,
                                     const float& rcut_r,
                                     const int& max_cpy_trial,
                                     const int& max_nnei_trial) {
  inlist.inum = nloc;
  if (nei_mode != 3) {
    // build nlist by myself
    // normalize and copy coord
    if (nei_mode == 1) {
      int copy_ok = _norm_copy_coord_cpu(coord_cpy, type_cpy, idx_mapping,
                                         new_nall, mem_cpy, *coord, box, *type,
                                         nloc, max_cpy_trial, rcut_r);
      OP_REQUIRES(context, copy_ok,
                  errors::Aborted("cannot allocate mem for copied coords"));
      *coord = &coord_cpy[0];
      *type = &type_cpy[0];
    }
    // build nlist
    int build_ok = _build_nlist_cpu(ilist, numneigh, firstneigh, jlist,
                                    max_nbor_size, mem_nnei, *coord, nloc,
                                    new_nall, max_nnei_trial, rcut_r);
    OP_REQUIRES(context, build_ok,
                errors::Aborted("cannot allocate mem for nlist"));
    inlist.ilist = &ilist[0];
    inlist.numneigh = &numneigh[0];
    inlist.firstneigh = &firstneigh[0];
  } else {
    // copy pointers to nlist data
    memcpy(&inlist.ilist, 4 + mesh_tensor_data, sizeof(int*));
    memcpy(&inlist.numneigh, 8 + mesh_tensor_data, sizeof(int*));
    memcpy(&inlist.firstneigh, 12 + mesh_tensor_data, sizeof(int**));
    max_nbor_size = max_numneigh(inlist);
  }
}

/*
//==================================================
  ProdEnvMatANvnmdOp
//==================================================
*/

// have been remove for the same function

/*
//==================================================
  ProdEnvMatANvnmdQuantizeOp
//==================================================
*/

template <typename Device, typename FPTYPE>
class ProdEnvMatANvnmdQuantizeOp : public OpKernel {
 public:
  explicit ProdEnvMatANvnmdQuantizeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float nloc_f, nall_f;
    OP_REQUIRES_OK(context, context->GetAttr("rcut_a", &rcut_a));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r", &rcut_r));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r_smth", &rcut_r_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
    OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
    // OP_REQUIRES_OK(context, context->GetAttr("nloc", &nloc_f));
    // OP_REQUIRES_OK(context, context->GetAttr("nall", &nall_f));
    deepmd::cum_sum(sec_a, sel_a);
    deepmd::cum_sum(sec_r, sel_r);
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
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& coord_tensor = context->input(context_input_index++);
    const Tensor& type_tensor = context->input(context_input_index++);
    const Tensor& natoms_tensor = context->input(context_input_index++);
    const Tensor& box_tensor = context->input(context_input_index++);
    const Tensor& mesh_tensor = context->input(context_input_index++);
    const Tensor& avg_tensor = context->input(context_input_index++);
    const Tensor& std_tensor = context->input(context_input_index++);
    // set size of the sample. assume 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3,
    // 3], [4, 4, 4]]], then shape(t) ==> [2, 2, 3]
    OP_REQUIRES(context, (coord_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of coord should be 2"));
    OP_REQUIRES(context, (type_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of type should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1),
                errors::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (box_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of box should be 2"));
    OP_REQUIRES(context, (mesh_tensor.shape().dims() == 1),
                errors::InvalidArgument("Dim of mesh should be 1"));
    OP_REQUIRES(context, (avg_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of avg should be 2"));
    OP_REQUIRES(context, (std_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of std should be 2"));
    OP_REQUIRES(context, (sec_r.back() == 0),
                errors::InvalidArgument(
                    "Rotational free descriptor only support all-angular "
                    "information: sel_r should be all zero."));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3),
                errors::InvalidArgument(
                    "number of atoms should be larger than (or equal to) 3"));
    DeviceFunctor()(device, context->eigen_device<Device>());
    const int* natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int ntypes =
        natoms_tensor.shape().dim_size(0) - 2;  // nloc and nall mean something.
    int nsamples = coord_tensor.shape().dim_size(0);
    //// check the sizes
    OP_REQUIRES(context, (nsamples == type_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nsamples == box_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (ntypes == avg_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of avg should be ntype"));
    OP_REQUIRES(context, (ntypes == std_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of std should be ntype"));

    OP_REQUIRES(context, (nall * 3 == coord_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context, (nall == type_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context, (9 == box_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of box should be 9"));
    OP_REQUIRES(context, (ndescrpt == avg_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of avg should be ndescrpt"));
    OP_REQUIRES(context, (ndescrpt == std_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of std should be ndescrpt"));

    OP_REQUIRES(context, (ntypes == int(sel_a.size())),
                errors::InvalidArgument(
                    "number of types should match the length of sel array"));
    OP_REQUIRES(context, (ntypes == int(sel_r.size())),
                errors::InvalidArgument(
                    "number of types should match the length of sel array"));

    int nei_mode = 0;
    bool b_nlist_map = false;
    if (mesh_tensor.shape().dim_size(0) == 16) {
      // lammps neighbor list
      nei_mode = 3;
    } else if (mesh_tensor.shape().dim_size(0) == 6) {
      // manual copied pbc
      assert(nloc == nall);
      nei_mode = 1;
      b_nlist_map = true;
    } else if (mesh_tensor.shape().dim_size(0) == 0) {
      // no pbc
      assert(nloc == nall);
      nei_mode = -1;
    } else if (mesh_tensor.shape().dim_size(0) > 16) {
      // pass neighbor list inside the tensor
      nei_mode = 4;
    } else if (mesh_tensor.shape().dim_size(0) == 7 ||
               mesh_tensor.shape().dim_size(0) == 1) {
      throw deepmd::deepmd_exception(
          "Mixed types are not supported by this OP.");
    } else {
      throw deepmd::deepmd_exception("invalid mesh tensor");
    }

    // Create output tensors
    TensorShape descrpt_shape;
    descrpt_shape.AddDim(nsamples);
    descrpt_shape.AddDim(int_64(nloc) * ndescrpt);
    TensorShape descrpt_deriv_shape;
    descrpt_deriv_shape.AddDim(nsamples);
    descrpt_deriv_shape.AddDim(int_64(nloc) * ndescrpt * 3);
    TensorShape rij_shape;
    rij_shape.AddDim(nsamples);
    rij_shape.AddDim(int_64(nloc) * nnei * 3);
    TensorShape nlist_shape;
    nlist_shape.AddDim(nsamples);
    nlist_shape.AddDim(int_64(nloc) * nnei);
    // define output tensor
    int context_output_index = 0;
    Tensor* descrpt_tensor = NULL;
    Tensor* descrpt_deriv_tensor = NULL;
    Tensor* rij_tensor = NULL;
    Tensor* nlist_tensor = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(context_output_index++, descrpt_shape,
                                          &descrpt_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descrpt_deriv_shape,
                                                     &descrpt_deriv_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     rij_shape, &rij_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++, nlist_shape,
                                            &nlist_tensor));

    FPTYPE* p_em = descrpt_tensor->flat<FPTYPE>().data();
    FPTYPE* p_em_deriv = descrpt_deriv_tensor->flat<FPTYPE>().data();
    FPTYPE* p_rij = rij_tensor->flat<FPTYPE>().data();
    int* p_nlist = nlist_tensor->flat<int>().data();
    const FPTYPE* p_coord = coord_tensor.flat<FPTYPE>().data();
    const FPTYPE* p_box = box_tensor.flat<FPTYPE>().data();
    const FPTYPE* avg = avg_tensor.flat<FPTYPE>().data();
    const FPTYPE* std = std_tensor.flat<FPTYPE>().data();
    const int* p_type = type_tensor.flat<int>().data();

    // must declar out of if, otherwise the memory will be destroyed!
    Tensor int_temp;
    Tensor uint64_temp;
    std::vector<Tensor> tensor_list(7);
    if (device == "GPU") {
      // UNDEFINE
    }

    // loop over samples
    for (int_64 ff = 0; ff < nsamples; ++ff) {
      FPTYPE* em = p_em + ff * nloc * ndescrpt;
      FPTYPE* em_deriv = p_em_deriv + ff * nloc * ndescrpt * 3;
      FPTYPE* rij = p_rij + ff * nloc * nnei * 3;
      int* nlist = p_nlist + ff * nloc * nnei;
      const FPTYPE* coord = p_coord + ff * nall * 3;
      const FPTYPE* box = p_box + ff * 9;
      const int* type = p_type + ff * nall;

      if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// UNDEFINE
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      } else if (device == "CPU") {
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
            context, &coord, coord_cpy, &type, type_cpy, idx_mapping, inlist,
            ilist, numneigh, firstneigh, jlist, frame_nall, mem_cpy, mem_nnei,
            max_nbor_size, box, mesh_tensor.flat<int>().data(), nloc, nei_mode,
            rcut_r, max_cpy_trial, max_nnei_trial);
        // launch the cpu compute function
        deepmd::prod_env_mat_a_nvnmd_quantize_cpu(
            em, em_deriv, rij, nlist, coord, type, inlist, max_nbor_size, avg,
            std, nloc, frame_nall, rcut_r, rcut_r_smth, sec_a);
        // do nlist mapping if coords were copied
        if (b_nlist_map) {
          _map_nlist_cpu(nlist, &idx_mapping[0], nloc, nnei);
        }
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
  int* array_int = NULL;
  unsigned long long* array_longlong = NULL;
  deepmd::InputNlist gpu_inlist;
  int* nbor_list_dev = NULL;
};

/*
//==================================================
  ProdEnvMatAMixNvnmdQuantizeOp
//==================================================
*/

template <typename Device, typename FPTYPE>
class ProdEnvMatAMixNvnmdQuantizeOp : public OpKernel {
 public:
  explicit ProdEnvMatAMixNvnmdQuantizeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float nloc_f, nall_f;
    OP_REQUIRES_OK(context, context->GetAttr("rcut_a", &rcut_a));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r", &rcut_r));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r_smth", &rcut_r_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
    OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
    // OP_REQUIRES_OK(context, context->GetAttr("nloc", &nloc_f));
    // OP_REQUIRES_OK(context, context->GetAttr("nall", &nall_f));
    deepmd::cum_sum(sec_a, sel_a);
    deepmd::cum_sum(sec_r, sel_r);
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
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& coord_tensor = context->input(context_input_index++);
    const Tensor& type_tensor = context->input(context_input_index++);
    const Tensor& natoms_tensor = context->input(context_input_index++);
    const Tensor& box_tensor = context->input(context_input_index++);
    const Tensor& mesh_tensor = context->input(context_input_index++);
    const Tensor& avg_tensor = context->input(context_input_index++);
    const Tensor& std_tensor = context->input(context_input_index++);
    // set size of the sample. assume 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3,
    // 3], [4, 4, 4]]], then shape(t) ==> [2, 2, 3]
    OP_REQUIRES(context, (coord_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of coord should be 2"));
    OP_REQUIRES(context, (type_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of type should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1),
                errors::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (box_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of box should be 2"));
    OP_REQUIRES(context, (mesh_tensor.shape().dims() == 1),
                errors::InvalidArgument("Dim of mesh should be 1"));
    OP_REQUIRES(context, (avg_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of avg should be 2"));
    OP_REQUIRES(context, (std_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of std should be 2"));
    OP_REQUIRES(context, (sec_r.back() == 0),
                errors::InvalidArgument(
                    "Rotational free descriptor only support all-angular "
                    "information: sel_r should be all zero."));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3),
                errors::InvalidArgument(
                    "number of atoms should be larger than (or equal to) 3"));
    DeviceFunctor()(device, context->eigen_device<Device>());
    const int* natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int ntypes = natoms_tensor.shape().dim_size(0) - 2;
    int nsamples = coord_tensor.shape().dim_size(0);
    //// check the sizes
    OP_REQUIRES(context, (nsamples == type_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nsamples == box_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (ntypes == avg_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of avg should be ntype"));
    OP_REQUIRES(context, (ntypes == std_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of std should be ntype"));

    OP_REQUIRES(context, (nall * 3 == coord_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context, (nall == type_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context, (9 == box_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of box should be 9"));
    OP_REQUIRES(context, (ndescrpt == avg_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of avg should be ndescrpt"));
    OP_REQUIRES(context, (ndescrpt == std_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of std should be ndescrpt"));

    OP_REQUIRES(context, (1 == int(sel_a.size())),
                errors::InvalidArgument(
                    "the length of sel array should be 1 in this op"));
    OP_REQUIRES(context, (1 == int(sel_r.size())),
                errors::InvalidArgument(
                    "the length of sel array should be 1 in this op"));

    int nei_mode = 0;
    bool b_nlist_map = false;
    if (mesh_tensor.shape().dim_size(0) == 16) {
      // lammps neighbor list
      nei_mode = 3;
    } else if (mesh_tensor.shape().dim_size(0) == 6 ||
               mesh_tensor.shape().dim_size(0) == 7) {
      // manual copied pbc
      nei_mode = 1;
      b_nlist_map = true;
    } else if (mesh_tensor.shape().dim_size(0) == 0 ||
               mesh_tensor.shape().dim_size(0) == 1) {
      // no pbc
      nei_mode = -1;
    } else if (mesh_tensor.shape().dim_size(0) > 16) {
      // pass neighbor list inside the tensor
      nei_mode = 4;
    } else {
      throw deepmd::deepmd_exception("invalid mesh tensor");
    }

    // Create output tensors
    TensorShape descrpt_shape;
    descrpt_shape.AddDim(nsamples);
    descrpt_shape.AddDim(int_64(nloc) * ndescrpt);
    TensorShape descrpt_deriv_shape;
    descrpt_deriv_shape.AddDim(nsamples);
    descrpt_deriv_shape.AddDim(int_64(nloc) * ndescrpt * 3);
    TensorShape rij_shape;
    rij_shape.AddDim(nsamples);
    rij_shape.AddDim(int_64(nloc) * nnei * 3);
    TensorShape nlist_shape;
    nlist_shape.AddDim(nsamples);
    nlist_shape.AddDim(int_64(nloc) * nnei);
    TensorShape ntype_shape;
    ntype_shape.AddDim(nsamples);
    ntype_shape.AddDim(static_cast<int64_t>(nloc) * nnei);
    TensorShape nmask_shape;
    nmask_shape.AddDim(nsamples);
    nmask_shape.AddDim(static_cast<int64_t>(nloc) * nnei);
    // define output tensor
    int context_output_index = 0;
    Tensor* descrpt_tensor = NULL;
    Tensor* descrpt_deriv_tensor = NULL;
    Tensor* rij_tensor = NULL;
    Tensor* nlist_tensor = NULL;
    Tensor* ntype_tensor = NULL;
    Tensor* nmask_tensor = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(context_output_index++, descrpt_shape,
                                          &descrpt_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descrpt_deriv_shape,
                                                     &descrpt_deriv_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     rij_shape, &rij_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++, nlist_shape,
                                            &nlist_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++, ntype_shape,
                                            &ntype_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++, nmask_shape,
                                            &nmask_tensor));

    Tensor fake_type_tensor;  // all zeros
    TensorShape fake_type_shape;
    fake_type_shape.AddDim(static_cast<int64_t>(nsamples) * nall);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, fake_type_shape,
                                                   &fake_type_tensor));

    FPTYPE* p_em = descrpt_tensor->flat<FPTYPE>().data();
    FPTYPE* p_em_deriv = descrpt_deriv_tensor->flat<FPTYPE>().data();
    FPTYPE* p_rij = rij_tensor->flat<FPTYPE>().data();
    int* p_nlist = nlist_tensor->flat<int>().data();
    int* p_ntype = ntype_tensor->flat<int>().data();
    bool* p_nmask = nmask_tensor->flat<bool>().data();
    const FPTYPE* p_coord = coord_tensor.flat<FPTYPE>().data();
    const FPTYPE* p_box = box_tensor.flat<FPTYPE>().data();
    const FPTYPE* avg = avg_tensor.flat<FPTYPE>().data();
    const FPTYPE* std = std_tensor.flat<FPTYPE>().data();
    const int* p_type = type_tensor.flat<int>().data();
    int* p_f_type = fake_type_tensor.flat<int>().data();

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// UNDEFINE
#endif
    } else if (device == "CPU") {
      for (int ii = 0; ii < nsamples * nall; ii++) {
        p_f_type[ii] = (p_type[ii] < 0) ? -1 : 0;
      }
    }

    // must declar out of if, otherwise the memory will be destroyed!
    Tensor int_temp;
    Tensor uint64_temp;
    std::vector<Tensor> tensor_list(7);
    if (device == "GPU") {
      // UNDEFINE
    }
    // loop over samples
    for (int_64 ff = 0; ff < nsamples; ++ff) {
      FPTYPE* em = p_em + ff * nloc * ndescrpt;
      FPTYPE* em_deriv = p_em_deriv + ff * nloc * ndescrpt * 3;
      FPTYPE* rij = p_rij + ff * nloc * nnei * 3;
      int* nlist = p_nlist + ff * nloc * nnei;
      int* ntype = p_ntype + ff * nloc * nnei;
      bool* nmask = p_nmask + ff * nloc * nnei;
      const FPTYPE* coord = p_coord + ff * nall * 3;
      const FPTYPE* box = p_box + ff * 9;
      const int* type = p_type + ff * nall;
      const int* f_type = p_f_type + ff * nall;

      if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// UNDEFINE
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      } else if (device == "CPU") {
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
            context, &coord, coord_cpy, &f_type, type_cpy, idx_mapping, inlist,
            ilist, numneigh, firstneigh, jlist, frame_nall, mem_cpy, mem_nnei,
            max_nbor_size, box, mesh_tensor.flat<int>().data(), nloc, nei_mode,
            rcut_r, max_cpy_trial, max_nnei_trial);
        // launch the cpu compute function
        deepmd::prod_env_mat_a_nvnmd_quantize_cpu(
            em, em_deriv, rij, nlist, coord, type, inlist, max_nbor_size, avg,
            std, nloc, frame_nall, rcut_r, rcut_r_smth, sec_a, f_type);
        // do nlist mapping if coords were copied
        _map_nei_info_cpu(nlist, ntype, nmask, type, &idx_mapping[0], nloc,
                          nnei, ntypes, b_nlist_map);
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
  int* array_int = NULL;
  unsigned long long* array_longlong = NULL;
  deepmd::InputNlist gpu_inlist;
  int* nbor_list_dev = NULL;
};

// Register the CPU kernels.
// Compatible with v1.3
#define REGISTER_CPU(T)                                              \
  REGISTER_KERNEL_BUILDER(Name("ProdEnvMatANvnmdQuantize")           \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T"),               \
                          ProdEnvMatANvnmdQuantizeOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("ProdEnvMatAMixNvnmdQuantize")        \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T"),               \
                          ProdEnvMatAMixNvnmdQuantizeOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
// Compatible with v1.3
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// UNDEFINE
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
