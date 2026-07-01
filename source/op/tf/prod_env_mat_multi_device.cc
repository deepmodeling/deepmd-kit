// SPDX-License-Identifier: LGPL-3.0-or-later
#include <algorithm>

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
static int _norm_copy_coord_cpu_frame(FPTYPE* coord_cpy,
                                      int* type_cpy,
                                      int* idx_mapping,
                                      int& frame_nall,
                                      const int& mem_cpy,
                                      const FPTYPE* coord,
                                      const FPTYPE* box,
                                      const int* type,
                                      const int& nall,
                                      const int& nloc,
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
                            const float& rcut_r,
                            const int& nframes = 1,
                            const int* type = NULL);

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

static tensorflow::Status _prepare_mesh_nlist_cpu_batch(
    deepmd::InputNlist& inlist,
    std::vector<int>& ilist,
    std::vector<int>& numneigh,
    std::vector<int*>& firstneigh,
    std::vector<std::vector<int>>& jlist,
    int& max_nbor_size,
    const int* mesh_tensor_data,
    const int mesh_tensor_size,
    const int nloc,
    const int nframes);

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

template <typename FPTYPE>
static tensorflow::Status _prepare_coord_nlist_cpu_batch(
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
    const int mesh_tensor_size,
    const int& nloc,
    const int& nall,
    const int& nframes,
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
static int _norm_copy_coord_gpu_frame(OpKernelContext* context,
                                      FPTYPE* coord_cpy,
                                      int* type_cpy,
                                      int* idx_mapping,
                                      int& frame_nall,
                                      const int& mem_cpy,
                                      const FPTYPE* coord,
                                      const FPTYPE* box,
                                      const int* type,
                                      const int& nloc,
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
                            const float& rcut_r,
                            const int& nframes = 1,
                            const int* type = NULL);

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

static tensorflow::Status _prepare_mesh_nlist_gpu_batch(
    OpKernelContext* context,
    Tensor* tensor_list,
    deepmd::InputNlist& gpu_inlist,
    int** firstneigh,
    int& max_nbor_size,
    const int* mesh_tensor_data,
    const int mesh_tensor_size,
    const int nloc,
    const int nframes);

static tensorflow::Status _round_built_gpu_nbor_size(int& max_nbor_size);

template <typename T>
struct DeviceMemoryGuard {
  T* ptr = NULL;
  DeviceMemoryGuard() = default;
  DeviceMemoryGuard(const DeviceMemoryGuard&) = delete;
  DeviceMemoryGuard& operator=(const DeviceMemoryGuard&) = delete;
  ~DeviceMemoryGuard() {
    if (ptr != NULL) {
      deepmd::delete_device_memory(ptr);
    }
  }
};

template <typename FPTYPE>
tensorflow::Status _prepare_coord_nlist_gpu(OpKernelContext* context,
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
                deepmd::tf_compat::InvalidArgument("Dim of coord should be 2"));
    OP_REQUIRES(context, (type_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of type should be 2"));
    OP_REQUIRES(
        context, (natoms_tensor.shape().dims() == 1),
        deepmd::tf_compat::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (box_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of box should be 2"));
    OP_REQUIRES(context, (mesh_tensor.shape().dims() == 1),
                deepmd::tf_compat::InvalidArgument("Dim of mesh should be 1"));
    OP_REQUIRES(context, (avg_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of avg should be 2"));
    OP_REQUIRES(context, (std_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of std should be 2"));
    OP_REQUIRES(context, (sec_r.back() == 0),
                deepmd::tf_compat::InvalidArgument(
                    "Rotational free descriptor only support all-angular "
                    "information: sel_r should be all zero."));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3),
                deepmd::tf_compat::InvalidArgument(
                    "number of atoms should be larger than (or equal to) 3"));
    DeviceFunctor()(device, context->eigen_device<Device>());
    const int* natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int ntypes =
        natoms_tensor.shape().dim_size(0) - 2;  // nloc and nall mean something.
    int nsamples = coord_tensor.shape().dim_size(0);
    //// check the sizes
    OP_REQUIRES(
        context, (nsamples == type_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of samples should match"));
    OP_REQUIRES(
        context, (nsamples == box_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of samples should match"));
    OP_REQUIRES(
        context, (ntypes == avg_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of avg should be ntype"));
    OP_REQUIRES(
        context, (ntypes == std_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of std should be ntype"));

    OP_REQUIRES(
        context, (nall * 3 == coord_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(
        context, (nall == type_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(
        context, (9 == box_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of box should be 9"));
    OP_REQUIRES(
        context, (ndescrpt == avg_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of avg should be ndescrpt"));
    OP_REQUIRES(
        context, (ndescrpt == std_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of std should be ndescrpt"));

    OP_REQUIRES(context, (ntypes == int(sel_a.size())),
                deepmd::tf_compat::InvalidArgument(
                    "number of types should match the length of sel array"));
    OP_REQUIRES(context, (ntypes == int(sel_r.size())),
                deepmd::tf_compat::InvalidArgument(
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

    if (device == "CPU" && nei_mode != 3) {
      const FPTYPE* coord = p_coord;
      const int* type = p_type;
      int frame_nall = nall;
      std::vector<FPTYPE> coord_cpy;
      std::vector<int> type_cpy;
      std::vector<int> idx_mapping;
      std::vector<int> ilist;
      std::vector<int> numneigh;
      std::vector<int*> firstneigh;
      std::vector<std::vector<int>> jlist;
      deepmd::InputNlist batch_inlist;
      int batch_max_nbor_size = max_nbor_size;
      OP_REQUIRES_OK(
          context,
          _prepare_coord_nlist_cpu_batch<FPTYPE>(
              &coord, coord_cpy, &type, type_cpy, idx_mapping, batch_inlist,
              ilist, numneigh, firstneigh, jlist, frame_nall, mem_cpy, mem_nnei,
              batch_max_nbor_size, p_box, mesh_tensor.flat<int>().data(),
              static_cast<int>(mesh_tensor.NumElements()), nloc, nall, nsamples,
              nei_mode, rcut_r, max_cpy_trial, max_nnei_trial));
      max_nbor_size = std::max(max_nbor_size, batch_max_nbor_size);

      deepmd::prod_env_mat_a_cpu(p_em, p_em_deriv, p_rij, p_nlist, coord, type,
                                 batch_inlist, batch_max_nbor_size, avg, std,
                                 nloc, frame_nall, nsamples, rcut_r,
                                 rcut_r_smth, sec_a);
      if (nei_mode == 1) {
        for (int kk = 0; kk < nsamples; ++kk) {
          _map_nlist_cpu(
              p_nlist + static_cast<int_64>(kk) * nloc * nnei,
              idx_mapping.data() + static_cast<int_64>(kk) * frame_nall, nloc,
              nnei);
        }
      }
      return;
    }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (device == "GPU" && (nei_mode == -1 || nei_mode == 1 || nei_mode == 4)) {
      const int frame_chunk = nsamples;
      for (int ff = 0; ff < nsamples; ff += frame_chunk) {
        const int chunk_nframes = std::min(frame_chunk, nsamples - ff);
        const int nrows = chunk_nframes * nloc;
        FPTYPE* em = p_em + static_cast<int_64>(ff) * nloc * ndescrpt;
        FPTYPE* em_deriv =
            p_em_deriv + static_cast<int_64>(ff) * nloc * ndescrpt * 3;
        FPTYPE* rij = p_rij + static_cast<int_64>(ff) * nloc * nnei * 3;
        int* nlist = p_nlist + static_cast<int_64>(ff) * nloc * nnei;
        const FPTYPE* coord = p_coord + static_cast<int_64>(ff) * nall * 3;
        const int* type = p_type + static_cast<int_64>(ff) * nall;
        int* idx_mapping = NULL;
        int frame_nall = nall;
        Tensor coord_cpy_tensor;
        Tensor type_cpy_tensor;
        if (nei_mode == 1) {
          int copy_ok = 0;
          for (int tt = 0; tt < max_cpy_trial; ++tt) {
            TensorShape cpy_shape;
            cpy_shape.AddDim(static_cast<int64_t>(chunk_nframes) * mem_cpy * 3);
            OP_REQUIRES_OK(
                context, context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                cpy_shape, &coord_cpy_tensor));
            TensorShape t_shape;
            t_shape.AddDim(static_cast<int64_t>(chunk_nframes) * mem_cpy * 2);
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, t_shape,
                                                           &type_cpy_tensor));
            FPTYPE* coord_cpy = coord_cpy_tensor.flat<FPTYPE>().data();
            int* type_cpy = type_cpy_tensor.flat<int>().data();
            idx_mapping = type_cpy + int_64(chunk_nframes) * mem_cpy;
            DPErrcheck(
                gpuMemset(type_cpy, -1,
                          sizeof(int) * int_64(chunk_nframes) * mem_cpy * 2));

            copy_ok = 1;
            for (int kk = 0; kk < chunk_nframes; ++kk) {
              int frame_copied_nall = nall;
              int ret = _norm_copy_coord_gpu_frame(
                  context, coord_cpy + int_64(kk) * mem_cpy * 3,
                  type_cpy + int_64(kk) * mem_cpy,
                  idx_mapping + int_64(kk) * mem_cpy, frame_copied_nall,
                  mem_cpy, p_coord + static_cast<int_64>(ff + kk) * nall * 3,
                  p_box + static_cast<int_64>(ff + kk) * 9,
                  p_type + static_cast<int_64>(ff + kk) * nall, nloc, rcut_r);
              OP_REQUIRES(
                  context, ret >= 0,
                  errors::Aborted("cannot allocate mem for copied coords"));
              if (ret != 0) {
                copy_ok = 0;
                break;
              }
            }
            if (copy_ok) {
              coord = coord_cpy;
              type = type_cpy;
              frame_nall = mem_cpy;
              break;
            }
            mem_cpy *= 2;
          }
          OP_REQUIRES(context, copy_ok,
                      errors::Aborted("cannot allocate mem for copied coords"));
        }

        std::vector<Tensor> nlist_tensors(2);
        int *ilist = NULL, *numneigh = NULL, *jlist = NULL;
        DeviceMemoryGuard<int*> firstneigh_guard;
        deepmd::malloc_device_memory(firstneigh_guard.ptr, nrows);
        int** firstneigh = firstneigh_guard.ptr;
        int chunk_max_nbor_size = max_nbor_size;
        deepmd::InputNlist chunk_gpu_inlist;
        if (nei_mode == 4) {
          OP_REQUIRES_OK(
              context,
              _prepare_mesh_nlist_gpu_batch(
                  context, nlist_tensors.data(), chunk_gpu_inlist, firstneigh,
                  chunk_max_nbor_size, mesh_tensor.flat<int>().data(),
                  static_cast<int>(mesh_tensor.NumElements()), nloc,
                  chunk_nframes));
        } else {
          TensorShape ilist_shape;
          ilist_shape.AddDim(static_cast<int64_t>(nrows) * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, ilist_shape,
                                                         &nlist_tensors[0]));
          TensorShape jlist_shape;
          jlist_shape.AddDim(3 * int_64(nrows) * mem_nnei);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, jlist_shape,
                                                         &nlist_tensors[1]));
          int build_ok =
              _build_nlist_gpu(context, nlist_tensors.data(), ilist, numneigh,
                               firstneigh, jlist, chunk_max_nbor_size, mem_nnei,
                               coord, nloc, frame_nall, max_nnei_trial, rcut_r,
                               chunk_nframes, nei_mode == 1 ? type : NULL);
          OP_REQUIRES(context, build_ok,
                      errors::Aborted("cannot allocate mem for nlist"));
          OP_REQUIRES_OK(context,
                         _round_built_gpu_nbor_size(chunk_max_nbor_size));
          chunk_gpu_inlist =
              deepmd::InputNlist(nrows, ilist, numneigh, firstneigh);
        }
        max_nbor_size = std::max(max_nbor_size, chunk_max_nbor_size);

        Tensor int_temp;
        TensorShape int_shape;
        int_shape.AddDim(sec_a.size() + int_64(nrows) * sec_a.size() + nrows);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DT_INT32, int_shape, &int_temp));
        Tensor uint64_temp;
        TensorShape uint64_shape;
        uint64_shape.AddDim(int_64(nrows) * chunk_max_nbor_size * 2);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape,
                                                       &uint64_temp));
        array_int = int_temp.flat<int>().data();
        array_longlong = uint64_temp.flat<unsigned long long>().data();

        deepmd::prod_env_mat_a_gpu(
            em, em_deriv, rij, nlist, coord, type, chunk_gpu_inlist, array_int,
            array_longlong, chunk_max_nbor_size, avg, std, nloc, frame_nall,
            chunk_nframes, rcut_r, rcut_r_smth, sec_a);
        if (nei_mode == 1) {
          for (int kk = 0; kk < chunk_nframes; ++kk) {
            _map_nlist_gpu(nlist + int_64(kk) * nloc * nnei,
                           idx_mapping + int_64(kk) * mem_cpy, nloc, nnei);
          }
        }
      }
      return;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

    // must declare out of if, otherwise the memory will be destroyed!
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

    // LAMMPS external nlists are updated outside this op, so keep their
    // existing per-sample implementation.
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
        DeviceMemoryGuard<int*> firstneigh_guard;
        deepmd::malloc_device_memory(firstneigh_guard.ptr, nloc);
        int** firstneigh = firstneigh_guard.ptr;
        int* jlist = NULL;
        FPTYPE* coord_cpy;
        int* type_cpy;
        int frame_nall = nall;
        int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
        // prepare coord and nlist
        OP_REQUIRES_OK(
            context,
            _prepare_coord_nlist_gpu<FPTYPE>(
                context, &tensor_list[0], &coord, coord_cpy, &type, type_cpy,
                idx_mapping, gpu_inlist, ilist, numneigh, firstneigh, jlist,
                nbor_list_dev, frame_nall, mem_cpy, mem_nnei, max_nbor_size,
                box, mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc,
                nei_mode, rcut_r, max_cpy_trial, max_nnei_trial));

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
                                   max_nbor_size, avg, std, nloc, frame_nall, 1,
                                   rcut_r, rcut_r_smth, sec_a);
        if (b_nlist_map) {
          _map_nlist_gpu(nlist, idx_mapping, nloc, nnei);
        }
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
                deepmd::tf_compat::InvalidArgument("Dim of coord should be 2"));
    OP_REQUIRES(context, (type_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of type should be 2"));
    OP_REQUIRES(
        context, (natoms_tensor.shape().dims() == 1),
        deepmd::tf_compat::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (box_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of box should be 2"));
    OP_REQUIRES(context, (mesh_tensor.shape().dims() == 1),
                deepmd::tf_compat::InvalidArgument("Dim of mesh should be 1"));
    OP_REQUIRES(context, (avg_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of avg should be 2"));
    OP_REQUIRES(context, (std_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of std should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3),
                deepmd::tf_compat::InvalidArgument(
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
    OP_REQUIRES(
        context, (nsamples == type_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of samples should match"));
    OP_REQUIRES(
        context, (nsamples == box_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of samples should match"));
    OP_REQUIRES(
        context, (ntypes == avg_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of avg should be ntype"));
    OP_REQUIRES(
        context, (ntypes == std_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of std should be ntype"));
    OP_REQUIRES(
        context, (nall * 3 == coord_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(
        context, (nall == type_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(
        context, (9 == box_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of box should be 9"));
    OP_REQUIRES(
        context, (ndescrpt == avg_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of avg should be ndescrpt"));
    OP_REQUIRES(
        context, (ndescrpt == std_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of std should be ndescrpt"));

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

    if (device == "CPU" && nei_mode != 3) {
      const FPTYPE* coord = p_coord;
      const int* type = p_type;
      int frame_nall = nall;
      std::vector<FPTYPE> coord_cpy;
      std::vector<int> type_cpy;
      std::vector<int> idx_mapping;
      std::vector<int> ilist;
      std::vector<int> numneigh;
      std::vector<int*> firstneigh;
      std::vector<std::vector<int>> jlist;
      deepmd::InputNlist batch_inlist;
      int batch_max_nbor_size = max_nbor_size;
      OP_REQUIRES_OK(
          context,
          _prepare_coord_nlist_cpu_batch<FPTYPE>(
              &coord, coord_cpy, &type, type_cpy, idx_mapping, batch_inlist,
              ilist, numneigh, firstneigh, jlist, frame_nall, mem_cpy, mem_nnei,
              batch_max_nbor_size, p_box, mesh_tensor.flat<int>().data(),
              static_cast<int>(mesh_tensor.NumElements()), nloc, nall, nsamples,
              nei_mode, rcut, max_cpy_trial, max_nnei_trial));
      max_nbor_size = std::max(max_nbor_size, batch_max_nbor_size);

      deepmd::prod_env_mat_r_cpu(p_em, p_em_deriv, p_rij, p_nlist, coord, type,
                                 batch_inlist, batch_max_nbor_size, avg, std,
                                 nloc, frame_nall, nsamples, rcut, rcut_smth,
                                 sec);
      if (nei_mode == 1) {
        for (int kk = 0; kk < nsamples; ++kk) {
          _map_nlist_cpu(
              p_nlist + static_cast<int_64>(kk) * nloc * nnei,
              idx_mapping.data() + static_cast<int_64>(kk) * frame_nall, nloc,
              nnei);
        }
      }
      return;
    }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (device == "GPU" && (nei_mode == -1 || nei_mode == 1 || nei_mode == 4)) {
      const int frame_chunk = nsamples;
      for (int ff = 0; ff < nsamples; ff += frame_chunk) {
        const int chunk_nframes = std::min(frame_chunk, nsamples - ff);
        const int nrows = chunk_nframes * nloc;
        FPTYPE* em = p_em + static_cast<int_64>(ff) * nloc * ndescrpt;
        FPTYPE* em_deriv =
            p_em_deriv + static_cast<int_64>(ff) * nloc * ndescrpt * 3;
        FPTYPE* rij = p_rij + static_cast<int_64>(ff) * nloc * nnei * 3;
        int* nlist = p_nlist + static_cast<int_64>(ff) * nloc * nnei;
        const FPTYPE* coord = p_coord + static_cast<int_64>(ff) * nall * 3;
        const int* type = p_type + static_cast<int_64>(ff) * nall;
        int* idx_mapping = NULL;
        int frame_nall = nall;
        Tensor coord_cpy_tensor;
        Tensor type_cpy_tensor;
        if (nei_mode == 1) {
          int copy_ok = 0;
          for (int tt = 0; tt < max_cpy_trial; ++tt) {
            TensorShape cpy_shape;
            cpy_shape.AddDim(static_cast<int64_t>(chunk_nframes) * mem_cpy * 3);
            OP_REQUIRES_OK(
                context, context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                cpy_shape, &coord_cpy_tensor));
            TensorShape t_shape;
            t_shape.AddDim(static_cast<int64_t>(chunk_nframes) * mem_cpy * 2);
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, t_shape,
                                                           &type_cpy_tensor));
            FPTYPE* coord_cpy = coord_cpy_tensor.flat<FPTYPE>().data();
            int* type_cpy = type_cpy_tensor.flat<int>().data();
            idx_mapping = type_cpy + int_64(chunk_nframes) * mem_cpy;
            DPErrcheck(
                gpuMemset(type_cpy, -1,
                          sizeof(int) * int_64(chunk_nframes) * mem_cpy * 2));

            copy_ok = 1;
            for (int kk = 0; kk < chunk_nframes; ++kk) {
              int frame_copied_nall = nall;
              int ret = _norm_copy_coord_gpu_frame(
                  context, coord_cpy + int_64(kk) * mem_cpy * 3,
                  type_cpy + int_64(kk) * mem_cpy,
                  idx_mapping + int_64(kk) * mem_cpy, frame_copied_nall,
                  mem_cpy, p_coord + static_cast<int_64>(ff + kk) * nall * 3,
                  p_box + static_cast<int_64>(ff + kk) * 9,
                  p_type + static_cast<int_64>(ff + kk) * nall, nloc, rcut);
              OP_REQUIRES(
                  context, ret >= 0,
                  errors::Aborted("cannot allocate mem for copied coords"));
              if (ret != 0) {
                copy_ok = 0;
                break;
              }
            }
            if (copy_ok) {
              coord = coord_cpy;
              type = type_cpy;
              frame_nall = mem_cpy;
              break;
            }
            mem_cpy *= 2;
          }
          OP_REQUIRES(context, copy_ok,
                      errors::Aborted("cannot allocate mem for copied coords"));
        }

        std::vector<Tensor> nlist_tensors(2);
        int *ilist = NULL, *numneigh = NULL, *jlist = NULL;
        DeviceMemoryGuard<int*> firstneigh_guard;
        deepmd::malloc_device_memory(firstneigh_guard.ptr, nrows);
        int** firstneigh = firstneigh_guard.ptr;
        int chunk_max_nbor_size = max_nbor_size;
        deepmd::InputNlist chunk_gpu_inlist;
        if (nei_mode == 4) {
          OP_REQUIRES_OK(
              context,
              _prepare_mesh_nlist_gpu_batch(
                  context, nlist_tensors.data(), chunk_gpu_inlist, firstneigh,
                  chunk_max_nbor_size, mesh_tensor.flat<int>().data(),
                  static_cast<int>(mesh_tensor.NumElements()), nloc,
                  chunk_nframes));
        } else {
          TensorShape ilist_shape;
          ilist_shape.AddDim(static_cast<int64_t>(nrows) * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, ilist_shape,
                                                         &nlist_tensors[0]));
          TensorShape jlist_shape;
          jlist_shape.AddDim(3 * int_64(nrows) * mem_nnei);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, jlist_shape,
                                                         &nlist_tensors[1]));
          int build_ok = _build_nlist_gpu(
              context, nlist_tensors.data(), ilist, numneigh, firstneigh, jlist,
              chunk_max_nbor_size, mem_nnei, coord, nloc, frame_nall,
              max_nnei_trial, rcut, chunk_nframes, nei_mode == 1 ? type : NULL);
          OP_REQUIRES(context, build_ok,
                      errors::Aborted("cannot allocate mem for nlist"));
          OP_REQUIRES_OK(context,
                         _round_built_gpu_nbor_size(chunk_max_nbor_size));
          chunk_gpu_inlist =
              deepmd::InputNlist(nrows, ilist, numneigh, firstneigh);
        }
        max_nbor_size = std::max(max_nbor_size, chunk_max_nbor_size);

        Tensor int_temp;
        TensorShape int_shape;
        int_shape.AddDim(sec.size() + int_64(nrows) * sec.size() + nrows);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DT_INT32, int_shape, &int_temp));
        Tensor uint64_temp;
        TensorShape uint64_shape;
        uint64_shape.AddDim(int_64(nrows) * chunk_max_nbor_size * 2);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape,
                                                       &uint64_temp));
        array_int = int_temp.flat<int>().data();
        array_longlong = uint64_temp.flat<unsigned long long>().data();

        deepmd::prod_env_mat_r_gpu(
            em, em_deriv, rij, nlist, coord, type, chunk_gpu_inlist, array_int,
            array_longlong, chunk_max_nbor_size, avg, std, nloc, frame_nall,
            chunk_nframes, rcut, rcut_smth, sec);
        if (nei_mode == 1) {
          for (int kk = 0; kk < chunk_nframes; ++kk) {
            _map_nlist_gpu(nlist + int_64(kk) * nloc * nnei,
                           idx_mapping + int_64(kk) * mem_cpy, nloc, nnei);
          }
        }
      }
      return;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

    // must declare out of if, otherwise the memory will be destroyed!
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

    // LAMMPS external nlists are updated outside this op, so keep their
    // existing per-sample implementation.
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
        DeviceMemoryGuard<int*> firstneigh_guard;
        deepmd::malloc_device_memory(firstneigh_guard.ptr, nloc);
        int** firstneigh = firstneigh_guard.ptr;
        int* jlist = NULL;
        FPTYPE* coord_cpy;
        int* type_cpy;
        int frame_nall = nall;
        int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
        // prepare coord and nlist
        OP_REQUIRES_OK(
            context,
            _prepare_coord_nlist_gpu<FPTYPE>(
                context, &tensor_list[0], &coord, coord_cpy, &type, type_cpy,
                idx_mapping, gpu_inlist, ilist, numneigh, firstneigh, jlist,
                nbor_list_dev, frame_nall, mem_cpy, mem_nnei, max_nbor_size,
                box, mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc,
                nei_mode, rcut, max_cpy_trial, max_nnei_trial));

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
                                   max_nbor_size, avg, std, nloc, frame_nall, 1,
                                   rcut, rcut_smth, sec);
        if (b_nlist_map) {
          _map_nlist_gpu(nlist, idx_mapping, nloc, nnei);
        }
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
                deepmd::tf_compat::InvalidArgument("Dim of coord should be 2"));
    OP_REQUIRES(context, (type_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of type should be 2"));
    OP_REQUIRES(
        context, (natoms_tensor.shape().dims() == 1),
        deepmd::tf_compat::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (box_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of box should be 2"));
    OP_REQUIRES(context, (mesh_tensor.shape().dims() == 1),
                deepmd::tf_compat::InvalidArgument("Dim of mesh should be 1"));
    OP_REQUIRES(context, (avg_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of avg should be 2"));
    OP_REQUIRES(context, (std_tensor.shape().dims() == 2),
                deepmd::tf_compat::InvalidArgument("Dim of std should be 2"));
    OP_REQUIRES(context, (sec_r.back() == 0),
                deepmd::tf_compat::InvalidArgument(
                    "Rotational free descriptor only support all-angular "
                    "information: sel_r should be all zero."));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3),
                deepmd::tf_compat::InvalidArgument(
                    "number of atoms should be larger than (or equal to) 3"));
    DeviceFunctor()(device, context->eigen_device<Device>());
    const int* natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int ntypes = natoms_tensor.shape().dim_size(0) - 2;
    int nsamples = coord_tensor.shape().dim_size(0);
    //// check the sizes
    OP_REQUIRES(
        context, (nsamples == type_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of samples should match"));
    OP_REQUIRES(
        context, (nsamples == box_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of samples should match"));
    OP_REQUIRES(
        context, (ntypes == avg_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of avg should be ntype"));
    OP_REQUIRES(
        context, (ntypes == std_tensor.shape().dim_size(0)),
        deepmd::tf_compat::InvalidArgument("number of std should be ntype"));

    OP_REQUIRES(
        context, (nall * 3 == coord_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(
        context, (nall == type_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(
        context, (9 == box_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of box should be 9"));
    OP_REQUIRES(
        context, (ndescrpt == avg_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of avg should be ndescrpt"));
    OP_REQUIRES(
        context, (ndescrpt == std_tensor.shape().dim_size(1)),
        deepmd::tf_compat::InvalidArgument("number of std should be ndescrpt"));

    OP_REQUIRES(context, (1 == int(sel_a.size())),
                deepmd::tf_compat::InvalidArgument(
                    "the length of sel array should be 1 in this op"));
    OP_REQUIRES(context, (1 == int(sel_r.size())),
                deepmd::tf_compat::InvalidArgument(
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

    if (device == "CPU" && nei_mode != 3) {
      const FPTYPE* coord = p_coord;
      const int* type = p_type;
      const int* f_type = p_f_type;
      int frame_nall = nall;
      std::vector<FPTYPE> coord_cpy;
      std::vector<int> f_type_cpy;
      std::vector<int> real_type_cpy;
      std::vector<int> idx_mapping;
      std::vector<int> ilist;
      std::vector<int> numneigh;
      std::vector<int*> firstneigh;
      std::vector<std::vector<int>> jlist;
      deepmd::InputNlist batch_inlist;
      int batch_max_nbor_size = max_nbor_size;
      OP_REQUIRES_OK(
          context,
          _prepare_coord_nlist_cpu_batch<FPTYPE>(
              &coord, coord_cpy, &f_type, f_type_cpy, idx_mapping, batch_inlist,
              ilist, numneigh, firstneigh, jlist, frame_nall, mem_cpy, mem_nnei,
              batch_max_nbor_size, p_box, mesh_tensor.flat<int>().data(),
              static_cast<int>(mesh_tensor.NumElements()), nloc, nall, nsamples,
              nei_mode, rcut_r, max_cpy_trial, max_nnei_trial));
      max_nbor_size = std::max(max_nbor_size, batch_max_nbor_size);

      if (nei_mode == 1) {
        real_type_cpy.assign(static_cast<size_t>(nsamples) * frame_nall, -1);
        for (int kk = 0; kk < nsamples; ++kk) {
          std::copy(
              p_type + static_cast<int_64>(kk) * nall,
              p_type + static_cast<int_64>(kk + 1) * nall,
              real_type_cpy.begin() + static_cast<size_t>(kk) * frame_nall);
        }
        type = real_type_cpy.data();
      }

      deepmd::prod_env_mat_a_cpu(p_em, p_em_deriv, p_rij, p_nlist, coord, type,
                                 batch_inlist, batch_max_nbor_size, avg, std,
                                 nloc, frame_nall, nsamples, rcut_r,
                                 rcut_r_smth, sec_a, f_type);
      for (int kk = 0; kk < nsamples; ++kk) {
        _map_nei_info_cpu(
            p_nlist + static_cast<int_64>(kk) * nloc * nnei,
            p_ntype + static_cast<int_64>(kk) * nloc * nnei,
            p_nmask + static_cast<int_64>(kk) * nloc * nnei,
            p_type + static_cast<int_64>(kk) * nall,
            nei_mode == 1
                ? idx_mapping.data() + static_cast<int_64>(kk) * frame_nall
                : NULL,
            nloc, nnei, ntypes, nei_mode == 1);
      }
      return;
    }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (device == "GPU" && (nei_mode == -1 || nei_mode == 1 || nei_mode == 4)) {
      const int frame_chunk = nsamples;
      for (int ff = 0; ff < nsamples; ff += frame_chunk) {
        const int chunk_nframes = std::min(frame_chunk, nsamples - ff);
        const int nrows = chunk_nframes * nloc;
        FPTYPE* em = p_em + static_cast<int_64>(ff) * nloc * ndescrpt;
        FPTYPE* em_deriv =
            p_em_deriv + static_cast<int_64>(ff) * nloc * ndescrpt * 3;
        FPTYPE* rij = p_rij + static_cast<int_64>(ff) * nloc * nnei * 3;
        int* nlist = p_nlist + static_cast<int_64>(ff) * nloc * nnei;
        int* ntype = p_ntype + static_cast<int_64>(ff) * nloc * nnei;
        bool* nmask = p_nmask + static_cast<int_64>(ff) * nloc * nnei;
        const FPTYPE* coord = p_coord + static_cast<int_64>(ff) * nall * 3;
        const int* type = p_type + static_cast<int_64>(ff) * nall;
        const int* f_type = p_f_type + static_cast<int_64>(ff) * nall;
        int* idx_mapping = NULL;
        int frame_nall = nall;
        Tensor coord_cpy_tensor;
        Tensor f_type_cpy_tensor;
        Tensor real_type_cpy_tensor;
        if (nei_mode == 1) {
          int copy_ok = 0;
          for (int tt = 0; tt < max_cpy_trial; ++tt) {
            TensorShape cpy_shape;
            cpy_shape.AddDim(static_cast<int64_t>(chunk_nframes) * mem_cpy * 3);
            OP_REQUIRES_OK(
                context, context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                                cpy_shape, &coord_cpy_tensor));
            TensorShape fake_type_cpy_shape;
            fake_type_cpy_shape.AddDim(static_cast<int64_t>(chunk_nframes) *
                                       mem_cpy * 2);
            OP_REQUIRES_OK(context,
                           context->allocate_temp(DT_INT32, fake_type_cpy_shape,
                                                  &f_type_cpy_tensor));
            TensorShape real_type_shape;
            real_type_shape.AddDim(static_cast<int64_t>(chunk_nframes) *
                                   mem_cpy);
            OP_REQUIRES_OK(context,
                           context->allocate_temp(DT_INT32, real_type_shape,
                                                  &real_type_cpy_tensor));
            FPTYPE* coord_cpy = coord_cpy_tensor.flat<FPTYPE>().data();
            int* f_type_cpy = f_type_cpy_tensor.flat<int>().data();
            int* real_type_cpy = real_type_cpy_tensor.flat<int>().data();
            idx_mapping = f_type_cpy + int_64(chunk_nframes) * mem_cpy;
            DPErrcheck(
                gpuMemset(f_type_cpy, -1,
                          sizeof(int) * int_64(chunk_nframes) * mem_cpy * 2));
            DPErrcheck(
                gpuMemset(real_type_cpy, -1,
                          sizeof(int) * int_64(chunk_nframes) * mem_cpy));

            copy_ok = 1;
            for (int kk = 0; kk < chunk_nframes; ++kk) {
              int frame_copied_nall = nall;
              int ret = _norm_copy_coord_gpu_frame(
                  context, coord_cpy + int_64(kk) * mem_cpy * 3,
                  f_type_cpy + int_64(kk) * mem_cpy,
                  idx_mapping + int_64(kk) * mem_cpy, frame_copied_nall,
                  mem_cpy, p_coord + static_cast<int_64>(ff + kk) * nall * 3,
                  p_box + static_cast<int_64>(ff + kk) * 9,
                  p_f_type + static_cast<int_64>(ff + kk) * nall, nloc, rcut_r);
              OP_REQUIRES(
                  context, ret >= 0,
                  errors::Aborted("cannot allocate mem for copied coords"));
              if (ret != 0) {
                copy_ok = 0;
                break;
              }
              DPErrcheck(gpuMemcpy(real_type_cpy + int_64(kk) * mem_cpy,
                                   p_type + static_cast<int_64>(ff + kk) * nall,
                                   sizeof(int) * nall,
                                   gpuMemcpyDeviceToDevice));
            }
            if (copy_ok) {
              coord = coord_cpy;
              type = real_type_cpy;
              f_type = f_type_cpy;
              frame_nall = mem_cpy;
              break;
            }
            mem_cpy *= 2;
          }
          OP_REQUIRES(context, copy_ok,
                      errors::Aborted("cannot allocate mem for copied coords"));
        }

        std::vector<Tensor> nlist_tensors(2);
        int *ilist = NULL, *numneigh = NULL, *jlist = NULL;
        DeviceMemoryGuard<int*> firstneigh_guard;
        deepmd::malloc_device_memory(firstneigh_guard.ptr, nrows);
        int** firstneigh = firstneigh_guard.ptr;
        int chunk_max_nbor_size = max_nbor_size;
        deepmd::InputNlist chunk_gpu_inlist;
        if (nei_mode == 4) {
          OP_REQUIRES_OK(
              context,
              _prepare_mesh_nlist_gpu_batch(
                  context, nlist_tensors.data(), chunk_gpu_inlist, firstneigh,
                  chunk_max_nbor_size, mesh_tensor.flat<int>().data(),
                  static_cast<int>(mesh_tensor.NumElements()), nloc,
                  chunk_nframes));
        } else {
          TensorShape ilist_shape;
          ilist_shape.AddDim(static_cast<int64_t>(nrows) * 2);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, ilist_shape,
                                                         &nlist_tensors[0]));
          TensorShape jlist_shape;
          jlist_shape.AddDim(3 * int_64(nrows) * mem_nnei);
          OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, jlist_shape,
                                                         &nlist_tensors[1]));
          int build_ok =
              _build_nlist_gpu(context, nlist_tensors.data(), ilist, numneigh,
                               firstneigh, jlist, chunk_max_nbor_size, mem_nnei,
                               coord, nloc, frame_nall, max_nnei_trial, rcut_r,
                               chunk_nframes, nei_mode == 1 ? f_type : NULL);
          OP_REQUIRES(context, build_ok,
                      errors::Aborted("cannot allocate mem for nlist"));
          OP_REQUIRES_OK(context,
                         _round_built_gpu_nbor_size(chunk_max_nbor_size));
          chunk_gpu_inlist =
              deepmd::InputNlist(nrows, ilist, numneigh, firstneigh);
        }
        max_nbor_size = std::max(max_nbor_size, chunk_max_nbor_size);

        Tensor int_temp;
        TensorShape int_shape;
        int_shape.AddDim(sec_a.size() + int_64(nrows) * sec_a.size() + nrows);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DT_INT32, int_shape, &int_temp));
        Tensor uint64_temp;
        TensorShape uint64_shape;
        uint64_shape.AddDim(int_64(nrows) * chunk_max_nbor_size * 2);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape,
                                                       &uint64_temp));
        array_int = int_temp.flat<int>().data();
        array_longlong = uint64_temp.flat<unsigned long long>().data();

        deepmd::prod_env_mat_a_gpu(
            em, em_deriv, rij, nlist, coord, type, chunk_gpu_inlist, array_int,
            array_longlong, chunk_max_nbor_size, avg, std, nloc, frame_nall,
            chunk_nframes, rcut_r, rcut_r_smth, sec_a, f_type);
        for (int kk = 0; kk < chunk_nframes; ++kk) {
          _map_nei_info_gpu(
              nlist + int_64(kk) * nloc * nnei,
              ntype + int_64(kk) * nloc * nnei,
              nmask + int_64(kk) * nloc * nnei,
              p_type + static_cast<int_64>(ff + kk) * nall,
              nei_mode == 1 ? idx_mapping + int_64(kk) * mem_cpy : NULL, nloc,
              nnei, ntypes, nei_mode == 1);
        }
      }
      return;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

    // must declare out of if, otherwise the memory will be destroyed!
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

    // LAMMPS external nlists are updated outside this op, so keep their
    // existing per-sample implementation.
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
        DeviceMemoryGuard<int*> firstneigh_guard;
        deepmd::malloc_device_memory(firstneigh_guard.ptr, nloc);
        int** firstneigh = firstneigh_guard.ptr;
        int* jlist = NULL;
        FPTYPE* coord_cpy;
        int* type_cpy;
        int frame_nall = nall;
        int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
        // prepare coord and nlist
        OP_REQUIRES_OK(
            context,
            _prepare_coord_nlist_gpu<FPTYPE>(
                context, &tensor_list[0], &coord, coord_cpy, &f_type, type_cpy,
                idx_mapping, gpu_inlist, ilist, numneigh, firstneigh, jlist,
                nbor_list_dev, frame_nall, mem_cpy, mem_nnei, max_nbor_size,
                box, mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc,
                nei_mode, rcut_r, max_cpy_trial, max_nnei_trial));

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
                                   max_nbor_size, avg, std, nloc, frame_nall, 1,
                                   rcut_r, rcut_r_smth, sec_a, f_type);
        _map_nei_info_gpu(nlist, ntype, nmask, type, idx_mapping, nloc, nnei,
                          ntypes, b_nlist_map);
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
static int _norm_copy_coord_cpu_frame(FPTYPE* coord_cpy,
                                      int* type_cpy,
                                      int* idx_mapping,
                                      int& frame_nall,
                                      const int& mem_cpy,
                                      const FPTYPE* coord,
                                      const FPTYPE* box,
                                      const int* type,
                                      const int& nall,
                                      const int& nloc,
                                      const float& rcut_r) {
  std::vector<FPTYPE> tmp_coord(static_cast<size_t>(nall) * 3);
  std::copy(coord, coord + static_cast<size_t>(nall) * 3, tmp_coord.begin());
  deepmd::Region<FPTYPE> region;
  init_region_cpu(region, box);
  normalize_coord_cpu(&tmp_coord[0], nall, region);
  frame_nall = nall;
  return copy_coord_cpu(coord_cpy, type_cpy, idx_mapping, &frame_nall,
                        &tmp_coord[0], type, nloc, mem_cpy, rcut_r, region);
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
                            const float& rcut_r,
                            const int& nframes,
                            const int* type) {
  const int nrows = nframes * nloc;
  int tt;
  for (tt = 0; tt < max_nnei_trial; ++tt) {
    for (int ii = 0; ii < nrows; ++ii) {
      jlist[ii].resize(mem_nnei);
      firstneigh[ii] = &jlist[ii][0];
    }
    deepmd::InputNlist inlist(nrows, &ilist[0], &numneigh[0], &firstneigh[0]);
    int ret = build_nlist_cpu(inlist, &max_nnei, coord, nloc, new_nall,
                              mem_nnei, rcut_r, nframes, type);
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

static tensorflow::Status _validate_mesh_neighbor_counts(
    int& max_numneigh,
    int_64& neighbor_count,
    const int* numneigh_in,
    const int mesh_tensor_size,
    const int_64 header_size,
    const int nloc) {
  max_numneigh = 0;
  neighbor_count = 0;
  for (int ii = 0; ii < nloc; ++ii) {
    const int_64 numneigh = numneigh_in[ii];
    if (numneigh < 0 || neighbor_count > static_cast<int_64>(mesh_tensor_size) -
                                             header_size - numneigh) {
      return errors::InvalidArgument("invalid mesh tensor");
    }
    max_numneigh = std::max(max_numneigh, static_cast<int>(numneigh_in[ii]));
    neighbor_count += numneigh;
  }
  return tensorflow::Status();
}

static tensorflow::Status _prepare_mesh_nlist_cpu_batch(
    deepmd::InputNlist& inlist,
    std::vector<int>& ilist,
    std::vector<int>& numneigh,
    std::vector<int*>& firstneigh,
    std::vector<std::vector<int>>& jlist,
    int& max_nbor_size,
    const int* mesh_tensor_data,
    const int mesh_tensor_size,
    const int nloc,
    const int nframes) {
  const int_64 header_size = 16 + static_cast<int_64>(2) * nloc;
  if (static_cast<int_64>(mesh_tensor_size) < header_size) {
    return errors::InvalidArgument("invalid mesh tensor");
  }

  const int* ilist_in = mesh_tensor_data + 16;
  const int* numneigh_in = mesh_tensor_data + 16 + nloc;
  const int* neighbors_in = mesh_tensor_data + header_size;

  int max_numneigh = 0;
  int_64 neighbor_count = 0;
  tensorflow::Status count_status =
      _validate_mesh_neighbor_counts(max_numneigh, neighbor_count, numneigh_in,
                                     mesh_tensor_size, header_size, nloc);
  if (!count_status.ok()) {
    return count_status;
  }

  const int nrows = nframes * nloc;
  ilist.resize(nrows);
  numneigh.resize(nrows);
  firstneigh.resize(nrows);
  jlist.resize(nrows);
  max_nbor_size = std::max(max_nbor_size, max_numneigh);

  std::vector<int_64> neighbor_offset(nloc + 1, 0);
  for (int ii = 0; ii < nloc; ++ii) {
    neighbor_offset[ii + 1] = neighbor_offset[ii] + numneigh_in[ii];
  }
  for (int ff = 0; ff < nframes; ++ff) {
    for (int ii = 0; ii < nloc; ++ii) {
      const int row = ff * nloc + ii;
      ilist[row] = ilist_in[ii];
      numneigh[row] = numneigh_in[ii];
      jlist[row].resize(numneigh_in[ii]);
      std::copy(neighbors_in + neighbor_offset[ii],
                neighbors_in + neighbor_offset[ii + 1], jlist[row].begin());
      firstneigh[row] = jlist[row].data();
    }
  }
  inlist = deepmd::InputNlist(nrows, ilist.data(), numneigh.data(),
                              firstneigh.data());
  return tensorflow::Status();
}

template <typename FPTYPE>
static tensorflow::Status _prepare_coord_nlist_cpu_batch(
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
    const int mesh_tensor_size,
    const int& nloc,
    const int& nall,
    const int& nframes,
    const int& nei_mode,
    const float& rcut_r,
    const int& max_cpy_trial,
    const int& max_nnei_trial) {
  const int nrows = nframes * nloc;
  new_nall = nall;
  if (nei_mode == 1) {
    int copy_ok = 0;
    for (int tt = 0; tt < max_cpy_trial; ++tt) {
      coord_cpy.assign(static_cast<size_t>(nframes) * mem_cpy * 3,
                       static_cast<FPTYPE>(0));
      type_cpy.assign(static_cast<size_t>(nframes) * mem_cpy, -1);
      idx_mapping.assign(static_cast<size_t>(nframes) * mem_cpy, -1);

      copy_ok = 1;
      for (int kk = 0; kk < nframes; ++kk) {
        int frame_copied_nall = nall;
        int ret = _norm_copy_coord_cpu_frame(
            coord_cpy.data() + static_cast<size_t>(kk) * mem_cpy * 3,
            type_cpy.data() + static_cast<size_t>(kk) * mem_cpy,
            idx_mapping.data() + static_cast<size_t>(kk) * mem_cpy,
            frame_copied_nall, mem_cpy,
            *coord + static_cast<size_t>(kk) * nall * 3,
            box + static_cast<size_t>(kk) * 9,
            *type + static_cast<size_t>(kk) * nall, nall, nloc, rcut_r);
        if (ret != 0) {
          copy_ok = 0;
          break;
        }
      }
      if (copy_ok) {
        *coord = coord_cpy.data();
        *type = type_cpy.data();
        new_nall = mem_cpy;
        break;
      }
      mem_cpy *= 2;
    }
    if (!copy_ok) {
      return errors::Aborted("cannot allocate mem for copied coords");
    }
  }

  ilist.resize(nrows);
  numneigh.resize(nrows);
  firstneigh.resize(nrows);
  jlist.resize(nrows);
  if (nei_mode == 4) {
    tensorflow::Status status = _prepare_mesh_nlist_cpu_batch(
        inlist, ilist, numneigh, firstneigh, jlist, max_nbor_size,
        mesh_tensor_data, mesh_tensor_size, nloc, nframes);
    if (!status.ok()) {
      return status;
    }
  } else {
    int build_ok =
        _build_nlist_cpu(ilist, numneigh, firstneigh, jlist, max_nbor_size,
                         mem_nnei, *coord, nloc, new_nall, max_nnei_trial,
                         rcut_r, nframes, nei_mode == 1 ? *type : NULL);
    if (!build_ok) {
      return errors::Aborted("cannot allocate mem for nlist");
    }
    inlist = deepmd::InputNlist(nrows, ilist.data(), numneigh.data(),
                                firstneigh.data());
  }
  return tensorflow::Status();
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
static int _norm_copy_coord_gpu_frame(OpKernelContext* context,
                                      FPTYPE* coord_cpy,
                                      int* type_cpy,
                                      int* idx_mapping,
                                      int& frame_nall,
                                      const int& mem_cpy,
                                      const FPTYPE* coord,
                                      const FPTYPE* box,
                                      const int* type,
                                      const int& nloc,
                                      const float& rcut_r) {
  Tensor tmp_coord_tensor;
  TensorShape tmp_coord_shape;
  tmp_coord_shape.AddDim(static_cast<int64_t>(nloc) * 3);
  tensorflow::Status status = context->allocate_temp(
      DataTypeToEnum<FPTYPE>::value, tmp_coord_shape, &tmp_coord_tensor);
  if (!status.ok()) {
    return -1;
  }
  FPTYPE* tmp_coord = tmp_coord_tensor.flat<FPTYPE>().data();
  DPErrcheck(gpuMemcpy(tmp_coord, coord, sizeof(FPTYPE) * nloc * 3,
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

  Tensor box_info_tensor;
  TensorShape box_info_shape;
  box_info_shape.AddDim(18);
  status = context->allocate_temp(DataTypeToEnum<FPTYPE>::value, box_info_shape,
                                  &box_info_tensor);
  if (!status.ok()) {
    return -1;
  }

  Tensor int_tensor;
  TensorShape int_shape;
  int_shape.AddDim(23 + nloc * 3 + loc_cellnum + total_cellnum * 3 +
                   total_cellnum * 3 + loc_cellnum + 1 + total_cellnum + 1 +
                   nloc);
  status = context->allocate_temp(DT_INT32, int_shape, &int_tensor);
  if (!status.ok()) {
    return -1;
  }

  FPTYPE* box_info_dev = box_info_tensor.flat<FPTYPE>().data();
  int* cell_info_dev = int_tensor.flat<int>().data();
  int* int_data_dev = cell_info_dev + 23;
  deepmd::memcpy_host_to_device(box_info_dev, box_info, 18);
  deepmd::memcpy_host_to_device(cell_info_dev, cell_info, 23);
  deepmd::Region<FPTYPE> region_dev(box_info_dev, box_info_dev + 9);
  deepmd::normalize_coord_gpu(tmp_coord, nloc, region_dev);

  frame_nall = nloc;
  return deepmd::copy_coord_gpu(coord_cpy, type_cpy, idx_mapping, &frame_nall,
                                int_data_dev, tmp_coord, type, nloc, mem_cpy,
                                loc_cellnum, total_cellnum, cell_info_dev,
                                region_dev);
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
                            const float& rcut_r,
                            const int& nframes,
                            const int* type) {
  const int nrows = nframes * nloc;
  ilist = (*tensor_list).flat<int>().data();
  numneigh = ilist + nrows;
  // Tensor jlist_temp;
  int* ind_data = NULL;

  std::vector<int*> firstneigh_host(nrows);
  int tt;
  for (tt = 0; tt < max_nnei_trial; ++tt) {
    jlist = (*(tensor_list + 1)).flat<int>().data();
    ind_data = jlist + int_64(nrows) * mem_nnei;
    for (int_64 ii = 0; ii < nrows; ++ii) {
      firstneigh_host[ii] = jlist + ii * mem_nnei;
    }
    deepmd::memcpy_host_to_device(firstneigh, firstneigh_host);
    deepmd::InputNlist inlist(nrows, ilist, numneigh, firstneigh);
    int ret =
        deepmd::build_nlist_gpu(inlist, &max_nnei, ind_data, coord, nloc,
                                new_nall, mem_nnei, rcut_r, nframes, type);
    if (ret == 0) {
      break;
    } else {
      mem_nnei *= 2;
      TensorShape jlist_shape;
      jlist_shape.AddDim(3 * int_64(nrows) * mem_nnei);
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

static tensorflow::Status _prepare_mesh_nlist_gpu_batch(
    OpKernelContext* context,
    Tensor* tensor_list,
    deepmd::InputNlist& gpu_inlist,
    int** firstneigh,
    int& max_nbor_size,
    const int* mesh_tensor_data,
    const int mesh_tensor_size,
    const int nloc,
    const int nframes) {
  const int_64 header_size = 16 + static_cast<int_64>(2) * nloc;
  if (static_cast<int_64>(mesh_tensor_size) < header_size) {
    return errors::InvalidArgument("invalid mesh tensor");
  }

  // Decode the external mesh on host first so malformed neighbor counts are
  // rejected before allocating the flattened GPU list.
  std::vector<int> mesh_tensor_data_host(mesh_tensor_size);
  deepmd::memcpy_device_to_host(mesh_tensor_data, mesh_tensor_data_host);
  const int* ilist_in = mesh_tensor_data_host.data() + 16;
  const int* numneigh_in = mesh_tensor_data_host.data() + 16 + nloc;
  const int* neighbors_in = mesh_tensor_data_host.data() + header_size;

  int max_numneigh = 0;
  int_64 neighbor_count = 0;
  tensorflow::Status count_status =
      _validate_mesh_neighbor_counts(max_numneigh, neighbor_count, numneigh_in,
                                     mesh_tensor_size, header_size, nloc);
  if (!count_status.ok()) {
    return count_status;
  }
  if (max_numneigh > GPU_MAX_NBOR_SIZE) {
    return errors::InvalidArgument(
        "Assert failed, max neighbor size of atom(lammps) " +
        std::to_string(max_numneigh) + " is larger than " +
        std::to_string(GPU_MAX_NBOR_SIZE) +
        ", which currently is not supported by deepmd-kit.");
  }

  if (max_numneigh <= 256) {
    max_nbor_size = 256;
  } else if (max_numneigh <= 512) {
    max_nbor_size = 512;
  } else if (max_numneigh <= 1024) {
    max_nbor_size = 1024;
  } else if (max_numneigh <= 2048) {
    max_nbor_size = 2048;
  } else {
    max_nbor_size = 4096;
  }

  const int nrows = nframes * nloc;
  TensorShape ilist_shape;
  ilist_shape.AddDim(static_cast<int64_t>(nrows) * 2);
  tensorflow::Status status =
      context->allocate_temp(DT_INT32, ilist_shape, tensor_list);
  if (!status.ok()) {
    return status;
  }
  TensorShape jlist_shape;
  jlist_shape.AddDim(int_64(nrows) * max_nbor_size);
  status = context->allocate_temp(DT_INT32, jlist_shape, tensor_list + 1);
  if (!status.ok()) {
    return status;
  }

  // Repeat the single-frame external mesh layout for each frame in the batch.
  std::vector<int> ilist_host(nrows);
  std::vector<int> numneigh_host(nrows);
  std::vector<int> nbor_list_host(static_cast<size_t>(nrows) * max_nbor_size,
                                  0);
  std::vector<int_64> neighbor_offset(nloc + 1, 0);
  for (int ii = 0; ii < nloc; ++ii) {
    neighbor_offset[ii + 1] = neighbor_offset[ii] + numneigh_in[ii];
  }
  for (int ff = 0; ff < nframes; ++ff) {
    for (int ii = 0; ii < nloc; ++ii) {
      const int row = ff * nloc + ii;
      ilist_host[row] = ilist_in[ii];
      numneigh_host[row] = numneigh_in[ii];
      for (int jj = 0; jj < numneigh_in[ii]; ++jj) {
        nbor_list_host[static_cast<size_t>(row) * max_nbor_size + jj] =
            neighbors_in[neighbor_offset[ii] + jj];
      }
    }
  }

  int* ilist = (*tensor_list).flat<int>().data();
  int* numneigh = ilist + nrows;
  int* nbor_list = (*(tensor_list + 1)).flat<int>().data();
  deepmd::memcpy_host_to_device(ilist, ilist_host);
  deepmd::memcpy_host_to_device(numneigh, numneigh_host);
  deepmd::memcpy_host_to_device(nbor_list, nbor_list_host);

  // Store device-side row pointers separately; InputNlist expects firstneigh
  // to point to each row in the contiguous neighbor buffer.
  std::vector<int*> firstneigh_host(nrows);
  for (int ii = 0; ii < nrows; ++ii) {
    firstneigh_host[ii] = nbor_list + static_cast<int_64>(ii) * max_nbor_size;
  }
  deepmd::memcpy_host_to_device(firstneigh, firstneigh_host);
  gpu_inlist = deepmd::InputNlist(nrows, ilist, numneigh, firstneigh);
  return tensorflow::Status();
}

static tensorflow::Status _round_built_gpu_nbor_size(int& max_nbor_size) {
  if (max_nbor_size > GPU_MAX_NBOR_SIZE) {
    return errors::InvalidArgument(
        "Assert failed, max neighbor size of atom(lammps) " +
        std::to_string(max_nbor_size) + " is larger than " +
        std::to_string(GPU_MAX_NBOR_SIZE) +
        ", which currently is not supported by deepmd-kit.");
  }
  if (max_nbor_size <= 1024) {
    max_nbor_size = 1024;
  } else if (max_nbor_size <= 2048) {
    max_nbor_size = 2048;
  } else {
    max_nbor_size = 4096;
  }
  return tensorflow::Status();
}

template <typename FPTYPE>
tensorflow::Status _prepare_coord_nlist_gpu(OpKernelContext* context,
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
      if (!copy_ok) {
        return errors::Aborted("cannot allocate mem for copied coords");
      }
      *coord = coord_cpy;
      *type = type_cpy;
    }
    // build nlist
    int build_ok =
        _build_nlist_gpu(context, tensor_list + 5, ilist, numneigh, firstneigh,
                         jlist, max_nbor_size, mem_nnei, *coord, nloc, new_nall,
                         max_nnei_trial, rcut_r);
    if (!build_ok) {
      return errors::Aborted("cannot allocate mem for nlist");
    }
    tensorflow::Status status = _round_built_gpu_nbor_size(max_nbor_size);
    if (!status.ok()) {
      return status;
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
    DeviceMemoryGuard<int> fake_mesh_guard;
    deepmd::malloc_device_memory(fake_mesh_guard.ptr, 16);
    deepmd::memcpy_host_to_device(fake_mesh_guard.ptr, fake_mesh);

    deepmd::InputNlist inlist_temp;
    inlist_temp.inum = nloc;
    // everything should be copied to GPU...
    deepmd::env_mat_nbor_update(inlist_temp, inlist, max_nbor_size,
                                nbor_list_dev, fake_mesh_guard.ptr, 16);
    if (max_numneigh(inlist_temp) > max_nbor_size) {
      return deepmd::tf_compat::InvalidArgument(
          "Assert failed, max neighbor size of atom(lammps) " +
          std::to_string(max_numneigh(inlist_temp)) + " is larger than " +
          std::to_string(max_nbor_size) +
          ", which currently is not supported by deepmd-kit.");
    }
  } else {
    // update nbor list
    deepmd::InputNlist inlist_temp;
    inlist_temp.inum = nloc;
    deepmd::env_mat_nbor_update(inlist_temp, inlist, max_nbor_size,
                                nbor_list_dev, mesh_tensor_data,
                                mesh_tensor_size);
    if (max_numneigh(inlist_temp) > max_nbor_size) {
      return deepmd::tf_compat::InvalidArgument(
          "Assert failed, max neighbor size of atom(lammps) " +
          std::to_string(max_numneigh(inlist_temp)) + " is larger than " +
          std::to_string(max_nbor_size) +
          ", which currently is not supported by deepmd-kit.");
    }
  }
  return tensorflow::Status();
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
