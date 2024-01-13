#ifdef ON_INFER
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif
// #include "paddle/include/experimental/ext_all.h"
#include "utilities.h"
#include "coord.h"
#include "region.h"
#include "neighbor_list.h"
#include "prod_env_mat.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")
#define CHECK_INPUT_READY(x) PD_CHECK(x.is_initialized(), #x " must be initialized before usage.")
#define CHECK_INPUT_DIM(x, value) PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
template <typename FPTYPE>
static int
_norm_copy_coord_cpu(
    std::vector<FPTYPE> &coord_cpy,
    std::vector<int> &type_cpy,
    std::vector<int> &mapping,
    int &nall,
    int &mem_cpy,
    const FPTYPE *coord,
    const FPTYPE *box,
    const int *type,
    const int &nloc,
    const int &max_cpy_trial,
    const float &rcut_r);

template <typename FPTYPE>
static int
_build_nlist_cpu(
    std::vector<int> &ilist,
    std::vector<int> &numneigh,
    std::vector<int *> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int &max_nnei,
    int &mem_nnei,
    const FPTYPE *coord,
    const int &nloc,
    const int &new_nall,
    const int &max_nnei_trial,
    const float &rcut_r);

static void
_map_nlist_cpu(
    int *nlist,
    const int *idx_mapping,
    const int &nloc,
    const int &nnei);

template <typename FPTYPE>
static void
_prepare_coord_nlist_cpu(
    FPTYPE const **coord,
    std::vector<FPTYPE> &coord_cpy,
    int const **type,
    std::vector<int> &type_cpy,
    std::vector<int> &idx_mapping,
    deepmd::InputNlist &inlist,
    std::vector<int> &ilist,
    std::vector<int> &numneigh,
    std::vector<int *> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int &new_nall,
    int &mem_cpy,
    int &mem_nnei,
    int &max_nbor_size,
    const FPTYPE *box,
    const int *mesh_tensor_data,
    const int &nloc,
    const int &nei_mode,
    const float &rcut_r,
    const int &max_cpy_trial,
    const int &max_nnei_trial);

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> PdProdEnvMatAOpCUDAForward(
    const paddle::Tensor &coord_tensor,
    const paddle::Tensor &type_tensor,
    const paddle::Tensor &natoms_tensor,
    const paddle::Tensor &box_tensor,
    const paddle::Tensor &mesh_tensor,
    const paddle::Tensor &avg_tensor,
    const paddle::Tensor &std_tensor,
    float rcut_a,
    float rcut_r,
    float rcut_r_smth,
    std::vector<int> sel_a,
    std::vector<int> sel_r);
#endif

template <typename data_t>
void PdProdEnvMatAOpCPUForwardKernel(
    int nsamples, int nloc, int ndescrpt, int nnei, int nall, int mem_cpy, int mem_nnei,
    int max_nbor_size, const int *mesh_tensor_data, int nei_mode, float rcut_a, float rcut_r, float rcut_r_smth, int max_cpy_trial,
    int max_nnei_trial, bool b_nlist_map, const std::vector<int>& sec_a,
    const std::vector<int>& sec_r, data_t *p_em, data_t *p_em_deriv, data_t *p_rij, int *p_nlist,
    const data_t *p_coord, const data_t *p_box, const data_t *avg, const data_t *std, const int *p_type)
{
  for (size_t ff = 0; ff < nsamples; ++ff)
  {
    data_t *em = p_em + ff * nloc * ndescrpt;
    data_t *em_deriv = p_em_deriv + ff * nloc * ndescrpt * 3;
    data_t *rij = p_rij + ff * nloc * nnei * 3;
    int *nlist = p_nlist + ff * nloc * nnei;
    const data_t *coord = p_coord + ff * nall * 3;
    const data_t *box = p_box + ff * 9;
    const int *type = p_type + ff * nall;
    
    deepmd::InputNlist inlist;
    // some buffers, be freed after the evaluation of this frame
    std::vector<int> idx_mapping;
    std::vector<int> ilist(nloc), numneigh(nloc);
    std::vector<int *> firstneigh(nloc);
    std::vector<std::vector<int>> jlist(nloc);
    std::vector<data_t> coord_cpy;
    std::vector<int> type_cpy;
    int frame_nall = nall;
    // prepare coord and nlist
    _prepare_coord_nlist_cpu<data_t>(&coord, coord_cpy, &type, type_cpy, idx_mapping,
                                     inlist, ilist, numneigh, firstneigh, jlist,
                                     frame_nall, mem_cpy, mem_nnei, max_nbor_size,
                                     box, mesh_tensor_data, nloc, nei_mode, rcut_r, max_cpy_trial, max_nnei_trial);
    // launch the cpu compute function
    deepmd::prod_env_mat_a_cpu(
        em, em_deriv, rij, nlist,
        coord, type, inlist, max_nbor_size, avg, std, nloc, frame_nall, rcut_r, rcut_r_smth, sec_a);
    // do nlist mapping if coords were copied
    if (b_nlist_map)
      _map_nlist_cpu(nlist, &idx_mapping[0], nloc, nnei);
  }
}

std::vector<paddle::Tensor> PdProdEnvMatAOpCPUForward(
    const paddle::Tensor &coord_tensor,
    const paddle::Tensor &type_tensor,
    const paddle::Tensor &natoms_tensor,
    const paddle::Tensor &box_tensor,
    const paddle::Tensor &mesh_tensor,
    const paddle::Tensor &avg_tensor,
    const paddle::Tensor &std_tensor,
    float rcut_a,
    float rcut_r,
    float rcut_r_smth,
    std::vector<int> sel_a,
    std::vector<int> sel_r)
{
  CHECK_INPUT(coord_tensor);
  CHECK_INPUT(type_tensor);
  CHECK_INPUT(natoms_tensor);
  CHECK_INPUT(box_tensor);
  CHECK_INPUT(mesh_tensor);
  CHECK_INPUT(avg_tensor);
  CHECK_INPUT(std_tensor);
  
  std::vector<int> sec_a;
  std::vector<int> sec_r;
  int ndescrpt, ndescrpt_a, ndescrpt_r;
  int nnei, nnei_a, nnei_r, max_nbor_size;
  int mem_cpy, max_cpy_trial;
  int mem_nnei, max_nnei_trial;
  std::string device;
  int *array_int = NULL;
  unsigned long long *array_longlong = NULL;
  deepmd::InputNlist gpu_inlist;
  int *nbor_list_dev = NULL;
  float nloc_f, nall_f;

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

  CHECK_INPUT_DIM(coord_tensor, 2);
  CHECK_INPUT_DIM(type_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);
  CHECK_INPUT_DIM(box_tensor, 2);
  CHECK_INPUT_DIM(mesh_tensor, 1);
  CHECK_INPUT_DIM(avg_tensor, 2);
  CHECK_INPUT_DIM(std_tensor, 2);

  PD_CHECK(sec_r.back() == 0, "Rotational free descriptor only support all-angular information: sel_r should be all zero.");
  PD_CHECK(natoms_tensor.shape()[0] >= 3, "Number of atoms should be larger than (or equal to) 3");
  // Paddle Set device on Python not in custom op
  
  // TODO: This code should be removed once cuda issue fixed.
  const int* natoms = nullptr;
  natoms = natoms_tensor.data<int>();

  int nloc = natoms[0];
  int nall = natoms[1];
  int ntypes = natoms_tensor.shape()[0] - 2; //nloc and nall mean something.
  int nsamples = coord_tensor.shape()[0];
  // check the sizes
  PD_CHECK(nsamples == type_tensor.shape()[0], "number of samples should match");
  PD_CHECK(nsamples == box_tensor.shape()[0], "number of samples should match");
  PD_CHECK(ntypes == avg_tensor.shape()[0], "number of avg should be ntype");
  PD_CHECK(ntypes == std_tensor.shape()[0], "number of std should be ntype");
  PD_CHECK(nall * 3 == coord_tensor.shape()[1], "number of atoms should match");
  PD_CHECK(nall == type_tensor.shape()[1], "number of atoms should match");
  PD_CHECK(9 == box_tensor.shape()[1], "number of box should be 9");
  PD_CHECK(ndescrpt == avg_tensor.shape()[1], "number of avg should be ndescrpt");
  PD_CHECK(ndescrpt == std_tensor.shape()[1], "number of std should be ndescrpt");
  PD_CHECK(ntypes == int(sel_a.size()), "number of types should match the length of sel array");
  PD_CHECK(ntypes == int(sel_r.size()), "number of types should match the length of sel array");
  int nei_mode = 0;
  bool b_nlist_map = false;
  if (mesh_tensor.shape()[0] == 16)
  {
    // lammps neighbor list
    nei_mode = 3;
  }
  else if (mesh_tensor.shape()[0] == 6)
  {
    // manual copied pbc
    assert(nloc == nall);
    nei_mode = 1;
    b_nlist_map = true;
  }
  else if (mesh_tensor.shape()[0] == 0)
  {
    // no pbc
    assert(nloc == nall);
    nei_mode = -1;
  }
  else
  {
    PD_THROW("Invalid mesh tensor");
  }

  // Create output tensors shape
  std::vector<int64_t> descrpt_shape{nsamples, nloc * ndescrpt};
  std::vector<int64_t> descrpt_deriv_shape{nsamples, nloc * ndescrpt * 3};
  std::vector<int64_t> rij_shape{nsamples, nloc * nnei * 3};
  std::vector<int64_t> nlist_shape{nsamples, nloc * nnei};

  // define output tensor
  paddle::Tensor descrpt_tensor = paddle::Tensor(paddle::PlaceType::kCPU, descrpt_shape);
  paddle::Tensor descrpt_deriv_tensor = paddle::Tensor(paddle::PlaceType::kCPU, descrpt_deriv_shape);
  paddle::Tensor rij_tensor = paddle::Tensor(paddle::PlaceType::kCPU, rij_shape);
  paddle::Tensor nlist_tensor = paddle::Tensor(paddle::PlaceType::kCPU, nlist_shape);
  PD_DISPATCH_FLOATING_TYPES(
      coord_tensor.type(), "pd_prod_env_mat_a_cpu_forward_kernel", ([&] {
        PdProdEnvMatAOpCPUForwardKernel<data_t>(
            nsamples, nloc, ndescrpt, nnei, nall, mem_cpy, mem_nnei, max_nbor_size,
            mesh_tensor.data<int>(), nei_mode, rcut_a, rcut_r, rcut_r_smth, max_cpy_trial, max_nnei_trial, b_nlist_map, sec_a, sec_r,
            descrpt_tensor.mutable_data<data_t>(),
            descrpt_deriv_tensor.mutable_data<data_t>(),
            rij_tensor.mutable_data<data_t>(),
            nlist_tensor.mutable_data<int>(),
            coord_tensor.data<data_t>(),
            box_tensor.data<data_t>(),
            avg_tensor.data<data_t>(),
            std_tensor.data<data_t>(),
            type_tensor.data<int>());
      }));
  return {descrpt_tensor, descrpt_deriv_tensor, rij_tensor, nlist_tensor};
}
std::vector<paddle::Tensor> PdProdEnvMatAOpForward(    
    const paddle::Tensor &coord_tensor,
    const paddle::Tensor &type_tensor,
    const paddle::Tensor &natoms_tensor,
    const paddle::Tensor &box_tensor,
    const paddle::Tensor &mesh_tensor,
    const paddle::Tensor &avg_tensor,
    const paddle::Tensor &std_tensor,
    float rcut_a,
    float rcut_r,
    float rcut_r_smth,
    std::vector<int> sel_a,
    std::vector<int> sel_r) {
  // TODO(chenweihang): Check Input
  CHECK_INPUT_READY(coord_tensor);
  CHECK_INPUT_READY(type_tensor);
  CHECK_INPUT_READY(natoms_tensor);
  CHECK_INPUT_READY(box_tensor);
  CHECK_INPUT_READY(mesh_tensor);
  CHECK_INPUT_READY(avg_tensor);
  CHECK_INPUT_READY(std_tensor);
  if (coord_tensor.place() == paddle::PlaceType::kCPU) {
    return PdProdEnvMatAOpCPUForward(
      coord_tensor, 
      type_tensor, 
      natoms_tensor, 
      box_tensor, 
      mesh_tensor, 
      avg_tensor, 
      std_tensor,
      rcut_a,
      rcut_r,
      rcut_r_smth,
      sel_a,
      sel_r
    );
#ifdef PADDLE_WITH_CUDA
  } else if (coord_tensor.place() == paddle::PlaceType::kGPU) {
    return PdProdEnvMatAOpCUDAForward(
      coord_tensor, 
      type_tensor, 
      natoms_tensor, 
      box_tensor, 
      mesh_tensor, 
      avg_tensor, 
      std_tensor,
      rcut_a,
      rcut_r,
      rcut_r_smth,
      sel_a,
      sel_r
    );
#endif
  } else {
    PD_THROW("Not implemented.");
  }
}
template <typename FPTYPE>
static void
_prepare_coord_nlist_cpu(
    FPTYPE const **coord,
    std::vector<FPTYPE> &coord_cpy,
    int const **type,
    std::vector<int> &type_cpy,
    std::vector<int> &idx_mapping,
    deepmd::InputNlist &inlist,
    std::vector<int> &ilist,
    std::vector<int> &numneigh,
    std::vector<int *> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int &new_nall,
    int &mem_cpy,
    int &mem_nnei,
    int &max_nbor_size,
    const FPTYPE *box,
    const int *mesh_tensor_data,
    const int &nloc,
    const int &nei_mode,
    const float &rcut_r,
    const int &max_cpy_trial,
    const int &max_nnei_trial)
{
  inlist.inum = nloc;
  if (nei_mode != 3)
  {
    // build nlist by myself
    // normalize and copy coord
    if (nei_mode == 1)
    {
      int copy_ok = _norm_copy_coord_cpu(
          coord_cpy, type_cpy, idx_mapping, new_nall, mem_cpy,
          *coord, box, *type, nloc, max_cpy_trial, rcut_r);
      PD_CHECK(copy_ok, "cannot allocate mem for copied coords");
      *coord = &coord_cpy[0];
      *type = &type_cpy[0];
    }
    // build nlist
    int build_ok = _build_nlist_cpu(
        ilist, numneigh, firstneigh, jlist, max_nbor_size, mem_nnei,
        *coord, nloc, new_nall, max_nnei_trial, rcut_r);
    PD_CHECK(build_ok, "cannot allocate mem for nlist");
    inlist.ilist = &ilist[0];
    inlist.numneigh = &numneigh[0];
    inlist.firstneigh = &firstneigh[0];
  }
  else
  {
    // copy pointers to nlist data
    memcpy(&inlist.ilist, 4 + mesh_tensor_data, sizeof(int *));
    memcpy(&inlist.numneigh, 8 + mesh_tensor_data, sizeof(int *));
    memcpy(&inlist.firstneigh, 12 + mesh_tensor_data, sizeof(int **));
    max_nbor_size = max_numneigh(inlist);
  }
}

template <typename FPTYPE>
static int _norm_copy_coord_cpu(
    std::vector<FPTYPE> &coord_cpy,
    std::vector<int> &type_cpy,
    std::vector<int> &idx_mapping,
    int &nall,
    int &mem_cpy,
    const FPTYPE *coord,
    const FPTYPE *box,
    const int *type,
    const int &nloc,
    const int &max_cpy_trial,
    const float &rcut_r)
{
  std::vector<FPTYPE> tmp_coord(nall * 3);
  std::copy(coord, coord + nall * 3, tmp_coord.begin());
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

template <typename FPTYPE>
static int _build_nlist_cpu(
    std::vector<int> &ilist,
    std::vector<int> &numneigh,
    std::vector<int *> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int &max_nnei,
    int &mem_nnei,
    const FPTYPE *coord,
    const int &nloc,
    const int &new_nall,
    const int &max_nnei_trial,
    const float &rcut_r)
{
  int tt;
  for (tt = 0; tt < max_nnei_trial; ++tt)
  {
    for (int ii = 0; ii < nloc; ++ii)
    {
      jlist[ii].resize(mem_nnei);
      firstneigh[ii] = &jlist[ii][0];
    }
    deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
    int ret = build_nlist_cpu(
        inlist, &max_nnei,
        coord, nloc, new_nall, mem_nnei, rcut_r);
    if (ret == 0)
    {
      break;
    }
    else
    {
      mem_nnei *= 2;
    }
  }
  return (tt != max_nnei_trial);
}

static void _map_nlist_cpu(
    int *nlist,
    const int *idx_mapping,
    const int &nloc,
    const int &nnei)
{
  for (int ii = 0; ii < nloc; ++ii)
  {
    for (int jj = 0; jj < nnei; ++jj)
    {
      int record = nlist[ii * nnei + jj];
      if (record >= 0)
      {
        nlist[ii * nnei + jj] = idx_mapping[record];
      }
    }
  }
}

std::vector<std::vector<int64_t>> PdProdEnvMatAOpInferShape(
  std::vector<int64_t> coord_shape,
  std::vector<int64_t> type_shape,
  std::vector<int64_t> natoms_shape,
  std::vector<int64_t> box_shape,
  std::vector<int64_t> mesh_shape,
  std::vector<int64_t> avg_shape,
  std::vector<int64_t> std_shape
  ) {
  int nsamples = coord_shape[0];
  std::vector<int64_t> descrpt_shape{nsamples, -1};
  std::vector<int64_t> descrpt_deriv_shape{nsamples, -1};
  std::vector<int64_t> rij_shape{nsamples, -1};
  std::vector<int64_t> nlist_shape{nsamples, -1};
  return {descrpt_shape, descrpt_deriv_shape, rij_shape, nlist_shape};
}

std::vector<paddle::DataType> PdProdEnvMatAOpInferDtype(
  paddle::DataType coord_dtype,
  paddle::DataType type_dtype,
  paddle::DataType natoms_dtype,
  paddle::DataType box_dtype,
  paddle::DataType mesh_dtype,
  paddle::DataType avg_dtype,
  paddle::DataType std_dtype) {

  return {coord_dtype, coord_dtype, coord_dtype, paddle::DataType::INT32};

}

PD_BUILD_OP(prod_env_mat_a)
    .Inputs({"coord", "type", "natoms", "box", "mesh", "davg", "dstd"})
    .Outputs({"descrpt", "descrpt_deriv", "rij", "nlist"})
    .Attrs({
    "rcut_a : float",
    "rcut_r: float", 
    "rcut_r_smth: float", 
    "sel_a: std::vector<int>",
    "sel_r: std::vector<int>"})
    .SetKernelFn(PD_KERNEL(PdProdEnvMatAOpForward))
    .SetInferShapeFn(PD_INFER_SHAPE(PdProdEnvMatAOpInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PdProdEnvMatAOpInferDtype));
