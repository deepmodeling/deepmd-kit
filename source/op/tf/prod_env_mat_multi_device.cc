// SPDX-License-Identifier: LGPL-3.0-or-later
#include "coord.h"
#include "custom_op.h"
#include "device.h"
#include "errors.h"
#include "neighbor_list.h"
#include "prod_env_mat.h"
#include "region.h"
#include "utilities.h"

REGISTER_OP("ProdEnvMatA")
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
    .Doc(R"(Compute the environment matrix for descriptor se_e2_a.
Each row of the environment matrix :math:`\mathcal{R}^i` can be constructed as follows

    .. math::
        (\mathcal{R}^i)_j = [
        \begin{array}{c}
            s(r_{ji}) & \frac{s(r_{ji})x_{ji}}{r_{ji}} & \frac{s(r_{ji})y_{ji}}{r_{ji}} & \frac{s(r_{ji})z_{ji}}{r_{ji}}
        \end{array}
        ]

In the above equation, :math:`\mathbf{R}_{ji}=\mathbf{R}_j-\mathbf{R}_i = (x_{ji}, y_{ji}, z_{ji})` is
the relative coordinate and :math:`r_{ji}=\lVert \mathbf{R}_{ji} \lVert` is its norm.
The switching function :math:`s(r)` is defined as:

    .. math::
        s(r)=
        \begin{cases}
        \frac{1}{r}, & r<r_s \\
        \frac{1}{r} \{ {(\frac{r - r_s}{ r_c - r_s})}^3 (-6 {(\frac{r - r_s}{ r_c - r_s})}^2 +15 \frac{r - r_s}{ r_c - r_s} -10) +1 \}, & r_s \leq r<r_c \\
        0, & r \geq r_c
        \end{cases}

Note that the environment matrix is normalized by davg and dstd.
coord: The coordinates of atoms.
type: The types of atoms.
natoms: The number of atoms. This tensor has the length of Ntypes + 2.
  natoms[0]: number of local atoms.
  natoms[1]: total number of atoms held by this processor.
  natoms[i]: 2 <= i < Ntypes+2, number of type i atoms.
box: The box of frames.
mesh: Gor historical reasons, only the length of the Tensor matters.
  If size of mesh == 6, pbc is assumed.
  If size of mesh == 0, no-pbc is assumed.
davg: Average value of the environment matrix for normalization.
dstd: Standard deviation of the environment matrix for normalization.
rcut_a: This argument is not used.
rcut_r: The cutoff radius for the environment matrix.
rcut_r_smth: From where the environment matrix should be smoothed.
sel_a: sel_a[i] specifies the maxmum number of type i atoms in the cut-off radius.
sel_r: This argument is not used.
descrpt: The environment matrix.
descrpt_deriv: The derivative of the environment matrix.
rij: The distance between the atoms.
nlist: The neighbor list of each atom.)");
// only sel_a and rcut_r used.

// an alias of ProdEnvMatA -- Compatible with v1.3
REGISTER_OP("DescrptSeA")
    .Attr("T: {float, double} = DT_DOUBLE")
    // give a default value to T, compatible with v1.2
    // See https://www.tensorflow.org/guide/create_op#backwards_compatibility
    .Input("coord: T")
    .Input("type: int32")
    .Input("natoms: int32")
    .Input("box : T")
    .Input("mesh : int32")
    .Input("davg: T")
    .Input("dstd: T")
    .Attr("rcut_a: float")
    .Attr("rcut_r: float")
    .Attr("rcut_r_smth: float")
    .Attr("sel_a: list(int)")
    .Attr("sel_r: list(int)")
    .Output("descrpt: T")
    .Output("descrpt_deriv: T")
    .Output("rij: T")
    .Output("nlist: int32");

// alias of ProdEnvMatA -- compatible with v0.12
REGISTER_OP("DescrptNorot")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("coord: T")
    .Input("type: int32")
    .Input("natoms: int32")
    .Input("box : T")
    .Input("mesh : int32")
    .Input("davg: T")
    .Input("dstd: T")
    .Attr("rcut_a: float")
    .Attr("rcut_r: float")
    .Attr("rcut_r_smth: float")
    .Attr("sel_a: list(int)")
    .Attr("sel_r: list(int)")
    .Output("descrpt: T")
    .Output("descrpt_deriv: T")
    .Output("rij: T")
    .Output("nlist: int32");

REGISTER_OP("ProdEnvMatR")
    .Attr("T: {float, double} = DT_DOUBLE")
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

// an alias of ProdEnvMatR -- Compatible with v1.3
REGISTER_OP("DescrptSeR")
    .Attr("T: {float, double} = DT_DOUBLE")
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

REGISTER_OP("ProdEnvMatAMix")
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
    .Output("nmask: bool")
    .Doc(R"(Compute the environment matrix mixing the atom types.
The sorting of neighbor atoms depends not on atom types, but on the distance and index.
The atoms in nlist matrix will gather forward and thus save space for gaps of types in ProdEnvMatA,
resulting in optimized and relative small sel_a.

The additional outputs are listed as following:
ntype: The corresponding atom types in nlist.
nmask: The atom mask in nlist.
)");

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename FPTYPE>
static int _norm_copy_coord_gpu(OpKernelContext* context,
                                Tensor* tensor_list,
                                FPTYPE*& coord_cpy,
                                int*& type_cpy,
                                int*& idx_mapping,
                                int& nall,
                                int& mem_cpy,
                                const FPTYPE* coord,
                                const FPTYPE* box,
                                const int* type,
                                const int& nloc,
                                const int& max_cpy_trial,
                                const float& rcut_r);

template <typename FPTYPE>
static int _build_nlist_gpu(OpKernelContext* context,
                            Tensor* tensor_list,
                            int*& ilist,
                            int*& numneigh,
                            int**& firstneigh,
                            int*& jlist,
                            int& max_nnei,
                            int& mem_nnei,
                            const FPTYPE* coord,
                            const int& nloc,
                            const int& new_nall,
                            const int& max_nnei_trial,
                            const float& rcut_r);

static void _map_nlist_gpu(int* nlist,
                           const int* idx_mapping,
                           const int& nloc,
                           const int& nnei);

static void _map_nei_info_gpu(int* nlist,
                              int* ntype,
                              bool* nmask,
                              const int* type,
                              const int* idx_mapping,
                              const int& nloc,
                              const int& nnei,
                              const int& ntypes,
                              const bool& b_nlist_map);

template <typename FPTYPE>
void _prepare_coord_nlist_gpu(OpKernelContext* context,
                              Tensor* tensor_list,
                              FPTYPE const** coord,
                              FPTYPE*& coord_cpy,
                              int const** type,
                              int*& type_cpy,
                              int*& idx_mapping,
                              deepmd::InputNlist& inlist,
                              int*& ilist,
                              int*& numneigh,
                              int**& firstneigh,
                              int*& jlist,
                              int*& nbor_list_dev,
                              int& new_nall,
                              int& mem_cpy,
                              int& mem_nnei,
                              int& max_nbor_size,
                              const FPTYPE* box,
                              const int* mesh_tensor_data,
                              const int mesh_tensor_size,
                              const int& nloc,
                              const int& nei_mode,
                              const float& rcut_r,
                              const int& max_cpy_trial,
                              const int& max_nnei_trial);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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
      // allocate temp memory only once for multiple frames
      // allocate temp memory, temp memory must not be used after this
      // operation!
      if (nei_mode != 3) {
        if (nei_mode == 1) {
          // Tensor FPTYPE_temp;
          TensorShape FPTYPE_shape;
          FPTYPE_shape.AddDim(static_cast<int64_t>(nall) * 3);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                FPTYPE_shape, &tensor_list[0]));

          // Tensor double_temp;
          TensorShape double_shape;
          double_shape.AddDim(18);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                double_shape, &tensor_list[1]));
          // Tensor cpy_temp;
          TensorShape cpy_shape;
          cpy_shape.AddDim(static_cast<int64_t>(mem_cpy) * 3);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                cpy_shape, &tensor_list[3]));
          // Tensor t_temp;
          TensorShape t_shape;
          t_shape.AddDim(static_cast<int64_t>(mem_cpy) * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, t_shape,
                                                         &tensor_list[4]));
        }

        // Tensor nlist_temp;
        TensorShape nlist_shape;
        nlist_shape.AddDim(static_cast<int64_t>(nloc) * 2);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, nlist_shape,
                                                       &tensor_list[5]));

        TensorShape jlist_shape;
        jlist_shape.AddDim(3 * int_64(nloc) * mem_nnei);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, jlist_shape,
                                                       &tensor_list[6]));
      }

      // used for format_nbor_list_gpu_cuda

      TensorShape int_shape;
      int_shape.AddDim(sec_a.size() + int_64(nloc) * sec_a.size() + nloc);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT32, int_shape, &int_temp));

      TensorShape uint64_shape;
      uint64_shape.AddDim(int_64(nloc) * max_nbor_size * 2);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape,
                                                     &uint64_temp));
      array_int = int_temp.flat<int>().data();
      array_longlong = uint64_temp.flat<unsigned long long>().data();
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
        int* idx_mapping = NULL;
        int *ilist = NULL, *numneigh = NULL;
        int** firstneigh = NULL;
        deepmd::malloc_device_memory(firstneigh, nloc);
        int* jlist = NULL;
        FPTYPE* coord_cpy;
        int* type_cpy;
        int frame_nall = nall;
        int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
        // prepare coord and nlist
        _prepare_coord_nlist_gpu<FPTYPE>(
            context, &tensor_list[0], &coord, coord_cpy, &type, type_cpy,
            idx_mapping, gpu_inlist, ilist, numneigh, firstneigh, jlist,
            nbor_list_dev, frame_nall, mem_cpy, mem_nnei, max_nbor_size, box,
            mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc, nei_mode,
            rcut_r, max_cpy_trial, max_nnei_trial);

        // max_nbor_size may be changed after _prepare_coord_nlist_gpu
        // So we need to update the uint64_temp tensor if necessary
        if (uint64_temp.NumElements() < int_64(nloc) * max_nbor_size * 2) {
          TensorShape uint64_shape;
          uint64_shape.AddDim(int_64(nloc) * max_nbor_size * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DT_UINT64, uint64_shape, &uint64_temp));
          array_longlong = uint64_temp.flat<unsigned long long>().data();
        }
        // launch the gpu(nv) compute function
        deepmd::prod_env_mat_a_gpu(em, em_deriv, rij, nlist, coord, type,
                                   gpu_inlist, array_int, array_longlong,
                                   max_nbor_size, avg, std, nloc, frame_nall,
                                   rcut_r, rcut_r_smth, sec_a);
        if (b_nlist_map) {
          _map_nlist_gpu(nlist, idx_mapping, nloc, nnei);
        }
        deepmd::delete_device_memory(firstneigh);
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
        deepmd::prod_env_mat_a_cpu(em, em_deriv, rij, nlist, coord, type,
                                   inlist, max_nbor_size, avg, std, nloc,
                                   frame_nall, rcut_r, rcut_r_smth, sec_a);
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

template <typename Device, typename FPTYPE>
class ProdEnvMatROp : public OpKernel {
 public:
  explicit ProdEnvMatROp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rcut", &rcut));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_smth", &rcut_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel", &sel));
    deepmd::cum_sum(sec, sel);
    sel_null.resize(3, 0);
    deepmd::cum_sum(sec_null, sel_null);
    ndescrpt = sec.back() * 1;
    nnei = sec.back();
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
    // set size of the sample
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
    //
    //// check the sizes
    // check the sizes
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

    // Create an output tensor
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

    int context_output_index = 0;
    Tensor* descrpt_tensor = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(context_output_index++, descrpt_shape,
                                          &descrpt_tensor));
    Tensor* descrpt_deriv_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descrpt_deriv_shape,
                                                     &descrpt_deriv_tensor));
    Tensor* rij_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     rij_shape, &rij_tensor));
    Tensor* nlist_tensor = NULL;
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
      // allocate temp memory only once for multiple frames
      // allocate temp memory, temp memory must not be used after this
      // operation!
      if (nei_mode != 3) {
        if (nei_mode == 1) {
          // Tensor FPTYPE_temp;
          TensorShape FPTYPE_shape;
          FPTYPE_shape.AddDim(static_cast<int64_t>(nall) * 3);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                FPTYPE_shape, &tensor_list[0]));

          // Tensor double_temp;
          TensorShape double_shape;
          double_shape.AddDim(18);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                double_shape, &tensor_list[1]));
          // Tensor cpy_temp;
          TensorShape cpy_shape;
          cpy_shape.AddDim(static_cast<int64_t>(mem_cpy) * 3);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                cpy_shape, &tensor_list[3]));
          // Tensor t_temp;
          TensorShape t_shape;
          t_shape.AddDim(static_cast<int64_t>(mem_cpy) * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, t_shape,
                                                         &tensor_list[4]));
        }

        // Tensor nlist_temp;
        TensorShape nlist_shape;
        nlist_shape.AddDim(static_cast<int64_t>(nloc) * 2);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, nlist_shape,
                                                       &tensor_list[5]));

        TensorShape jlist_shape;
        jlist_shape.AddDim(3 * int_64(nloc) * mem_nnei);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, jlist_shape,
                                                       &tensor_list[6]));
      }

      // used for format_nbor_list_gpu_cuda

      TensorShape int_shape;
      int_shape.AddDim(sec.size() + int_64(nloc) * sec.size() + nloc);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT32, int_shape, &int_temp));

      TensorShape uint64_shape;
      uint64_shape.AddDim(int_64(nloc) * max_nbor_size * 2);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape,
                                                     &uint64_temp));

      array_int = int_temp.flat<int>().data();
      array_longlong = uint64_temp.flat<unsigned long long>().data();
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
        int* idx_mapping = NULL;
        int *ilist = NULL, *numneigh = NULL;
        int** firstneigh = NULL;
        deepmd::malloc_device_memory(firstneigh, nloc);
        int* jlist = NULL;
        FPTYPE* coord_cpy;
        int* type_cpy;
        int frame_nall = nall;
        int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
        // prepare coord and nlist
        _prepare_coord_nlist_gpu<FPTYPE>(
            context, &tensor_list[0], &coord, coord_cpy, &type, type_cpy,
            idx_mapping, gpu_inlist, ilist, numneigh, firstneigh, jlist,
            nbor_list_dev, frame_nall, mem_cpy, mem_nnei, max_nbor_size, box,
            mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc, nei_mode,
            rcut, max_cpy_trial, max_nnei_trial);

        // max_nbor_size may be changed after _prepare_coord_nlist_gpu
        // So we need to update the uint64_temp tensor if necessary
        if (uint64_temp.NumElements() < int_64(nloc) * max_nbor_size * 2) {
          TensorShape uint64_shape;
          uint64_shape.AddDim(int_64(nloc) * max_nbor_size * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DT_UINT64, uint64_shape, &uint64_temp));
          array_longlong = uint64_temp.flat<unsigned long long>().data();
        }

        // launch the gpu(nv) compute function
        deepmd::prod_env_mat_r_gpu(em, em_deriv, rij, nlist, coord, type,
                                   gpu_inlist, array_int, array_longlong,
                                   max_nbor_size, avg, std, nloc, frame_nall,
                                   rcut, rcut_smth, sec);
        if (b_nlist_map) {
          _map_nlist_gpu(nlist, idx_mapping, nloc, nnei);
        }
        deepmd::delete_device_memory(firstneigh);
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
            rcut, max_cpy_trial, max_nnei_trial);
        // launch the cpu compute function
        deepmd::prod_env_mat_r_cpu(em, em_deriv, rij, nlist, coord, type,
                                   inlist, max_nbor_size, avg, std, nloc,
                                   frame_nall, rcut, rcut_smth, sec);
        if (b_nlist_map) {
          _map_nlist_cpu(nlist, &idx_mapping[0], nloc, nnei);
        }
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
  int* array_int = NULL;
  unsigned long long* array_longlong = NULL;
  deepmd::InputNlist gpu_inlist;
  int* nbor_list_dev = NULL;
};

template <typename Device, typename FPTYPE>
class ProdEnvMatAMixOp : public OpKernel {
 public:
  explicit ProdEnvMatAMixOp(OpKernelConstruction* context) : OpKernel(context) {
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
      deepmd::filter_ftype_gpu(p_f_type, p_type, nsamples * nall);
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
      // allocate temp memory only once for multiple frames
      // allocate temp memory, temp memory must not be used after this
      // operation!
      if (nei_mode != 3) {
        if (nei_mode == 1) {
          // Tensor FPTYPE_temp;
          TensorShape FPTYPE_shape;
          FPTYPE_shape.AddDim(static_cast<int64_t>(nall) * 3);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                FPTYPE_shape, &tensor_list[0]));

          // Tensor double_temp;
          TensorShape double_shape;
          double_shape.AddDim(18);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                double_shape, &tensor_list[1]));
          // Tensor cpy_temp;
          TensorShape cpy_shape;
          cpy_shape.AddDim(static_cast<int64_t>(mem_cpy) * 3);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                cpy_shape, &tensor_list[3]));
          // Tensor t_temp;
          TensorShape t_shape;
          t_shape.AddDim(static_cast<int64_t>(mem_cpy) * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, t_shape,
                                                         &tensor_list[4]));
        }

        // Tensor nlist_temp;
        TensorShape nlist_shape;
        nlist_shape.AddDim(static_cast<int64_t>(nloc) * 2);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, nlist_shape,
                                                       &tensor_list[5]));

        TensorShape jlist_shape;
        jlist_shape.AddDim(3 * int_64(nloc) * mem_nnei);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, jlist_shape,
                                                       &tensor_list[6]));
      }

      // used for format_nbor_list_gpu_cuda

      TensorShape int_shape;
      int_shape.AddDim(sec_a.size() + int_64(nloc) * sec_a.size() + nloc);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT32, int_shape, &int_temp));

      TensorShape uint64_shape;
      uint64_shape.AddDim(int_64(nloc) * max_nbor_size * 2);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape,
                                                     &uint64_temp));

      array_int = int_temp.flat<int>().data();
      array_longlong = uint64_temp.flat<unsigned long long>().data();
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
        int* idx_mapping = NULL;
        int *ilist = NULL, *numneigh = NULL;
        int** firstneigh = NULL;
        deepmd::malloc_device_memory(firstneigh, nloc);
        int* jlist = NULL;
        FPTYPE* coord_cpy;
        int* type_cpy;
        int frame_nall = nall;
        int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
        // prepare coord and nlist
        _prepare_coord_nlist_gpu<FPTYPE>(
            context, &tensor_list[0], &coord, coord_cpy, &f_type, type_cpy,
            idx_mapping, gpu_inlist, ilist, numneigh, firstneigh, jlist,
            nbor_list_dev, frame_nall, mem_cpy, mem_nnei, max_nbor_size, box,
            mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc, nei_mode,
            rcut_r, max_cpy_trial, max_nnei_trial);

        // max_nbor_size may be changed after _prepare_coord_nlist_gpu
        // So we need to update the uint64_temp tensor if necessary
        if (uint64_temp.NumElements() < int_64(nloc) * max_nbor_size * 2) {
          TensorShape uint64_shape;
          uint64_shape.AddDim(int_64(nloc) * max_nbor_size * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DT_UINT64, uint64_shape, &uint64_temp));
          array_longlong = uint64_temp.flat<unsigned long long>().data();
        }

        // launch the gpu(nv) compute function
        deepmd::prod_env_mat_a_gpu(em, em_deriv, rij, nlist, coord, type,
                                   gpu_inlist, array_int, array_longlong,
                                   max_nbor_size, avg, std, nloc, frame_nall,
                                   rcut_r, rcut_r_smth, sec_a, f_type);
        _map_nei_info_gpu(nlist, ntype, nmask, type, idx_mapping, nloc, nnei,
                          ntypes, b_nlist_map);
        deepmd::delete_device_memory(firstneigh);
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
        deepmd::prod_env_mat_a_cpu(
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
  for (int_64 ii = 0; ii < nloc; ++ii) {
    for (int_64 jj = 0; jj < nnei; ++jj) {
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

/**
 * @param[in] nei_mode -1, 1, 3, or 4.
 *   - -1: Build neighbor list without PBC. The size of mesh should
 *     be 0 (no mixed) or 1 (mixed).
 *   - 1: Build neighbor list with PBC. The size of mesh should
 *     be 6 (no mixed) or 7 (mixed).
 *   - 3：Use neighbor list from given pointers. The size of mesh should be 16.
 *     The first element is ago (whether update the internal neighbour list).
 *     The second element is the number of local atoms. The 5th-8th, 9th-12th,
 *     and 13th-16th elements are the pointer (int*, 4x size of int) to
 *     ilist, numneigh, firstneigh. The pointer should be valid during the
 *     execution of this op, so it may be created and given by an external
 *     program calling the TensorFlow session.
 *   - 4: Use neighbor list stored in the tensor. The size of mesh should be
 *     16 + 2 * nloc + sum(numneigh). Starting from the 17th element, the
 *     elements are ilist (size of nloc), numneigh (size of nloc), and neighbors
 *     (size of numneigh[i] for each i).
 */
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
  if (nei_mode != 3 && nei_mode != 4) {
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
  } else if (nei_mode == 4) {
    std::memcpy(&ilist[0], 16 + mesh_tensor_data, sizeof(int) * nloc);
    std::memcpy(&numneigh[0], 16 + nloc + mesh_tensor_data, sizeof(int) * nloc);
    for (int ii = 0, kk = 0; ii < nloc; ++ii) {
      jlist[ii].resize(numneigh[ii]);
      std::memcpy(&jlist[ii][0], 16 + 2 * nloc + kk + mesh_tensor_data,
                  sizeof(int) * numneigh[ii]);
      firstneigh[ii] = &jlist[ii][0];
      kk += numneigh[ii];
    }
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename FPTYPE>
static int _norm_copy_coord_gpu(OpKernelContext* context,
                                Tensor* tensor_list,
                                FPTYPE*& coord_cpy,
                                int*& type_cpy,
                                int*& idx_mapping,
                                int& nall,
                                int& mem_cpy,
                                const FPTYPE* coord,
                                const FPTYPE* box,
                                const int* type,
                                const int& nloc,
                                const int& max_cpy_trial,
                                const float& rcut_r) {
  FPTYPE* tmp_coord = (*tensor_list).flat<FPTYPE>().data();
  DPErrcheck(gpuMemcpy(tmp_coord, coord, sizeof(FPTYPE) * nall * 3,
                       gpuMemcpyDeviceToDevice));

  deepmd::Region<FPTYPE> region;
  init_region_cpu(region, box);
  FPTYPE box_info[18];
  std::copy(region.boxt, region.boxt + 9, box_info);
  std::copy(region.rec_boxt, region.rec_boxt + 9, box_info + 9);
  int cell_info[23];
  deepmd::compute_cell_info(cell_info, rcut_r, region);
  const int loc_cellnum = cell_info[21];
  const int total_cellnum = cell_info[22];
  // Tensor int_temp;
  TensorShape int_shape;
  int_shape.AddDim(23 + nloc * 3 + loc_cellnum + total_cellnum * 3 +
                   total_cellnum * 3 + loc_cellnum + 1 + total_cellnum + 1 +
                   nloc);
  tensorflow::Status status =
      context->allocate_temp(DT_INT32, int_shape, tensor_list + 2);
  if (!status.ok()) {
    return false;
  }
  FPTYPE* box_info_dev = (*(tensor_list + 1)).flat<FPTYPE>().data();
  int* cell_info_dev = (*(tensor_list + 2)).flat<int>().data();
  int* int_data_dev = cell_info_dev + 23;
  deepmd::memcpy_host_to_device(box_info_dev, box_info, 18);
  deepmd::memcpy_host_to_device(cell_info_dev, cell_info, 23);
  deepmd::Region<FPTYPE> region_dev(box_info_dev, box_info_dev + 9);
  deepmd::normalize_coord_gpu(tmp_coord, nall, region_dev);
  int tt;
  for (tt = 0; tt < max_cpy_trial; ++tt) {
    coord_cpy = (*(tensor_list + 3)).flat<FPTYPE>().data();
    type_cpy = (*(tensor_list + 4)).flat<int>().data();
    idx_mapping = type_cpy + mem_cpy;
    int ret = deepmd::copy_coord_gpu(
        coord_cpy, type_cpy, idx_mapping, &nall, int_data_dev, tmp_coord, type,
        nloc, mem_cpy, loc_cellnum, total_cellnum, cell_info_dev, region_dev);
    if (ret == 0) {
      break;
    } else {
      mem_cpy *= 2;
      // Tensor cpy_temp;
      TensorShape cpy_shape;
      cpy_shape.AddDim(static_cast<int64_t>(mem_cpy) * 3);
      status = context->allocate_temp(DataTypeToEnum<FPTYPE>::value, cpy_shape,
                                      tensor_list + 3);
      if (!status.ok()) {
        return false;
      }
      // Tensor t_temp;
      TensorShape t_shape;
      t_shape.AddDim(static_cast<int64_t>(mem_cpy) * 2);
      status = context->allocate_temp(DT_INT32, t_shape, tensor_list + 4);
      if (!status.ok()) {
        return false;
      }
    }
  }
  return (tt != max_cpy_trial);
}

template <typename FPTYPE>
static int _build_nlist_gpu(OpKernelContext* context,
                            Tensor* tensor_list,
                            int*& ilist,
                            int*& numneigh,
                            int**& firstneigh,
                            int*& jlist,
                            int& max_nnei,
                            int& mem_nnei,
                            const FPTYPE* coord,
                            const int& nloc,
                            const int& new_nall,
                            const int& max_nnei_trial,
                            const float& rcut_r) {
  ilist = (*tensor_list).flat<int>().data();
  numneigh = ilist + nloc;
  // Tensor jlist_temp;
  int* ind_data = NULL;

  std::vector<int*> firstneigh_host(nloc);
  int tt;
  for (tt = 0; tt < max_nnei_trial; ++tt) {
    jlist = (*(tensor_list + 1)).flat<int>().data();
    ind_data = jlist + nloc * mem_nnei;
    for (int_64 ii = 0; ii < nloc; ++ii) {
      firstneigh_host[ii] = jlist + ii * mem_nnei;
    }
    deepmd::memcpy_host_to_device(firstneigh, firstneigh_host);
    deepmd::InputNlist inlist(nloc, ilist, numneigh, firstneigh);
    int ret = deepmd::build_nlist_gpu(inlist, &max_nnei, ind_data, coord, nloc,
                                      new_nall, mem_nnei, rcut_r);
    if (ret == 0) {
      break;
    } else {
      mem_nnei *= 2;
      TensorShape jlist_shape;
      jlist_shape.AddDim(3 * int_64(nloc) * mem_nnei);
      tensorflow::Status status =
          context->allocate_temp(DT_INT32, jlist_shape, tensor_list + 1);
      if (!status.ok()) {
        return false;
      }
    }
  }
  return (tt != max_nnei_trial);
}

static void _map_nlist_gpu(int* nlist,
                           const int* idx_mapping,
                           const int& nloc,
                           const int& nnei) {
  deepmd::use_nlist_map(nlist, idx_mapping, nloc, nnei);
}

static void _map_nei_info_gpu(int* nlist,
                              int* ntype,
                              bool* nmask,
                              const int* type,
                              const int* idx_mapping,
                              const int& nloc,
                              const int& nnei,
                              const int& ntypes,
                              const bool& b_nlist_map) {
  deepmd::use_nei_info_gpu(nlist, ntype, nmask, type, idx_mapping, nloc, nnei,
                           ntypes, b_nlist_map);
}

template <typename FPTYPE>
void _prepare_coord_nlist_gpu(OpKernelContext* context,
                              Tensor* tensor_list,
                              FPTYPE const** coord,
                              FPTYPE*& coord_cpy,
                              int const** type,
                              int*& type_cpy,
                              int*& idx_mapping,
                              deepmd::InputNlist& inlist,
                              int*& ilist,
                              int*& numneigh,
                              int**& firstneigh,
                              int*& jlist,
                              int*& nbor_list_dev,
                              int& new_nall,
                              int& mem_cpy,
                              int& mem_nnei,
                              int& max_nbor_size,
                              const FPTYPE* box,
                              const int* mesh_tensor_data,
                              const int mesh_tensor_size,
                              const int& nloc,
                              const int& nei_mode,
                              const float& rcut_r,
                              const int& max_cpy_trial,
                              const int& max_nnei_trial) {
  if (nei_mode != 3 && nei_mode != 4) {
    inlist.inum = nloc;
    // build nlist by myself
    // normalize and copy coord
    if (nei_mode == 1) {
      int copy_ok = _norm_copy_coord_gpu(
          context, tensor_list, coord_cpy, type_cpy, idx_mapping, new_nall,
          mem_cpy, *coord, box, *type, nloc, max_cpy_trial, rcut_r);
      OP_REQUIRES(context, copy_ok,
                  errors::Aborted("cannot allocate mem for copied coords"));
      *coord = coord_cpy;
      *type = type_cpy;
    }
    // build nlist
    int build_ok =
        _build_nlist_gpu(context, tensor_list + 5, ilist, numneigh, firstneigh,
                         jlist, max_nbor_size, mem_nnei, *coord, nloc, new_nall,
                         max_nnei_trial, rcut_r);
    OP_REQUIRES(context, build_ok,
                errors::Aborted("cannot allocate mem for nlist"));
    if (max_nbor_size <= 1024) {
      max_nbor_size = 1024;
    } else if (max_nbor_size <= 2048) {
      max_nbor_size = 2048;
    } else {
      max_nbor_size = 4096;
    }
    inlist.ilist = ilist;
    inlist.numneigh = numneigh;
    inlist.firstneigh = firstneigh;
  } else if (nei_mode == 4) {
    // TODO: in theory, it will be faster to put everything on GPUs...
    std::vector<int> mesh_tensor_data_host(mesh_tensor_size);
    std::vector<int> ilist_host(nloc);
    std::vector<int> numneigh_host(nloc);
    std::vector<int*> firstneigh_host(nloc);
    std::vector<int> fake_mesh(16);

    // copy from gpu to cpu
    deepmd::memcpy_device_to_host(mesh_tensor_data, mesh_tensor_data_host);
    std::memcpy(&ilist_host[0], &mesh_tensor_data_host[16], sizeof(int) * nloc);
    std::memcpy(&numneigh_host[0], &mesh_tensor_data_host[16 + nloc],
                sizeof(int) * nloc);
    for (int ii = 0, kk = 0; ii < nloc; ++ii) {
      firstneigh_host[ii] = &mesh_tensor_data_host[16 + 2 * nloc + kk];
      kk += numneigh_host[ii];
    }
    // make a fake mesh
    fake_mesh[0] = 0;
    fake_mesh[1] = nloc;
    std::memcpy(&fake_mesh[4], &ilist_host, sizeof(int*));
    std::memcpy(&fake_mesh[8], &numneigh_host, sizeof(int*));
    std::memcpy(&fake_mesh[12], &firstneigh_host, sizeof(int**));
    // copy from cpu to gpu
    int* fake_mesh_dev = NULL;
    deepmd::malloc_device_memory(fake_mesh_dev, 16);
    deepmd::memcpy_host_to_device(fake_mesh_dev, fake_mesh);

    deepmd::InputNlist inlist_temp;
    inlist_temp.inum = nloc;
    // everything should be copied to GPU...
    deepmd::env_mat_nbor_update(inlist_temp, inlist, max_nbor_size,
                                nbor_list_dev, fake_mesh_dev, 16);
    OP_REQUIRES(context, (max_numneigh(inlist_temp) <= max_nbor_size),
                errors::InvalidArgument(
                    "Assert failed, max neighbor size of atom(lammps) " +
                    std::to_string(max_numneigh(inlist_temp)) +
                    " is larger than " + std::to_string(max_nbor_size) +
                    ", which currently is not supported by deepmd-kit."));
    deepmd::delete_device_memory(fake_mesh_dev);
  } else {
    // update nbor list
    deepmd::InputNlist inlist_temp;
    inlist_temp.inum = nloc;
    deepmd::env_mat_nbor_update(inlist_temp, inlist, max_nbor_size,
                                nbor_list_dev, mesh_tensor_data,
                                mesh_tensor_size);
    OP_REQUIRES(context, (max_numneigh(inlist_temp) <= max_nbor_size),
                errors::InvalidArgument(
                    "Assert failed, max neighbor size of atom(lammps) " +
                    std::to_string(max_numneigh(inlist_temp)) +
                    " is larger than " + std::to_string(max_nbor_size) +
                    ", which currently is not supported by deepmd-kit."));
  }
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Register the CPU kernels.
// Compatible with v1.3
#define REGISTER_CPU(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ProdEnvMatA").Device(DEVICE_CPU).TypeConstraint<T>("T"),    \
      ProdEnvMatAOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ProdEnvMatR").Device(DEVICE_CPU).TypeConstraint<T>("T"),    \
      ProdEnvMatROp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ProdEnvMatAMix").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ProdEnvMatAMixOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DescrptSeA").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      ProdEnvMatAOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DescrptNorot").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      ProdEnvMatAOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DescrptSeR").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      ProdEnvMatROp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
// Compatible with v1.3
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("ProdEnvMatA")              \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("natoms")        \
                              .HostMemory("box"),          \
                          ProdEnvMatAOp<GPUDevice, T>);    \
  REGISTER_KERNEL_BUILDER(Name("ProdEnvMatR")              \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("natoms")        \
                              .HostMemory("box"),          \
                          ProdEnvMatROp<GPUDevice, T>);    \
  REGISTER_KERNEL_BUILDER(Name("ProdEnvMatAMix")           \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("natoms")        \
                              .HostMemory("box"),          \
                          ProdEnvMatAMixOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("DescrptSeA")               \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("natoms")        \
                              .HostMemory("box"),          \
                          ProdEnvMatAOp<GPUDevice, T>);    \
  REGISTER_KERNEL_BUILDER(Name("DescrptNorot")             \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("natoms")        \
                              .HostMemory("box"),          \
                          ProdEnvMatAOp<GPUDevice, T>);    \
  REGISTER_KERNEL_BUILDER(Name("DescrptSeR")               \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("natoms")        \
                              .HostMemory("box"),          \
                          ProdEnvMatROp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
