#define GOOGLE_CUDA 1
#ifdef ON_INFER
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif
#include "utilities.h"
#include "coord.h"
#include "region.h"
#include "neighbor_list.h"
#include <vector>
#include "prod_env_mat.h"

#include<iomanip>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")
#define CHECK_INPUT_DIM(x, value) PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")

template <typename FPTYPE>
static int
_norm_copy_coord_gpu(
    std::vector<paddle::Tensor>* tensor_list,
    FPTYPE *&coord_cpy,
    int *&type_cpy,
    int *&idx_mapping,
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
_build_nlist_gpu(
    std::vector<paddle::Tensor> *tensor_list,
    int *&ilist,
    int *&numneigh,
    int **&firstneigh,
    int *&jlist,
    int &max_nnei,
    int &mem_nnei,
    const FPTYPE *coord,
    const int &nloc,
    const int &new_nall,
    const int &max_nnei_trial,
    const float &rcut_r);

static void
_map_nlist_gpu(
    int *nlist,
    const int *idx_mapping,
    const int &nloc,
    const int &nnei);

template <typename FPTYPE>
static void
_prepare_coord_nlist_gpu(
    std::vector<paddle::Tensor> *tensor_list,
    FPTYPE const **coord,
    FPTYPE *&coord_cpy,
    int const **type,
    int *&type_cpy,
    int *&idx_mapping,
    deepmd::InputNlist &inlist,
    int *&ilist,
    int *&numneigh,
    int **&firstneigh,
    int *&jlist,
    int *&nbor_list_dev,
    int &new_nall,
    int &mem_cpy,
    int &mem_nnei,
    int &max_nbor_size,
    const FPTYPE *box,
    const int *mesh_tensor_data,
    const int mesh_tensor_size,
    const int &nloc,
    const int &nei_mode,
    const float &rcut_r,
    const int &max_cpy_trial,
    const int &max_nnei_trial);

template <typename data_t>
void PdProdEnvMatAOpCUDAForwardKernel(
    int nsamples, int nloc, int ndescrpt, int nnei, int nall, int mem_cpy, int mem_nnei,
    int max_nbor_size, int nei_mode, float rcut_a, float rcut_r, float rcut_r_smth, int max_cpy_trial,
    int max_nnei_trial, bool b_nlist_map, const std::vector<int>& sec_a,
    const std::vector<int>& sec_r, deepmd::InputNlist gpu_inlist, int* nbor_list_dev, int* array_int, unsigned long long* array_longlong,
    data_t *p_em, data_t *p_em_deriv, data_t *p_rij, int *p_nlist,
    const data_t *p_coord, const data_t *p_box, const data_t *avg, 
    const data_t *std, const int *p_type, const paddle::Tensor& mesh_tensor)
{
    
    for (int ff = 0; ff < nsamples; ++ff)
    {
        data_t *em = p_em + ff * nloc * ndescrpt;
        data_t *em_deriv = p_em_deriv + ff * nloc * ndescrpt * 3;
        data_t *rij = p_rij + ff * nloc * nnei * 3;
        int *nlist = p_nlist + ff * nloc * nnei;
        const data_t *coord = p_coord + ff * nall * 3;
        const data_t *box = p_box + ff * 9;
        const int *type = p_type + ff * nall;


        int *idx_mapping = NULL;
        int *ilist = NULL, *numneigh = NULL;
        int **firstneigh = NULL;
        deepmd::malloc_device_memory(firstneigh, nloc);
        int *jlist = NULL;
        data_t *coord_cpy;
        int *type_cpy;
        int frame_nall = nall;
        int mesh_tensor_size = static_cast<int>(mesh_tensor.size());
        std::vector<paddle::Tensor> tensor_list;
        _prepare_coord_nlist_gpu<data_t>(
            &tensor_list, &coord, coord_cpy, &type, type_cpy, idx_mapping,
            gpu_inlist, ilist, numneigh, firstneigh, jlist, nbor_list_dev,
            frame_nall, mem_cpy, mem_nnei, max_nbor_size,
            box, mesh_tensor.data<int>(), mesh_tensor_size, nloc, nei_mode, rcut_r, max_cpy_trial, max_nnei_trial);
        // allocate temp memory, temp memory must not be used after this operation!
        std::vector<int64_t> int_temp_shape{sec_a.size() + nloc * sec_a.size() + nloc};
        paddle::Tensor int_temp(paddle::PlaceType::kGPU, int_temp_shape);

        array_int = int_temp.mutable_data<int>();

        deepmd::malloc_device_memory(array_longlong, nloc * GPU_MAX_NBOR_SIZE * 2);
        // launch the gpu(nv) compute function

        deepmd::prod_env_mat_a_gpu_cuda(
            em, em_deriv, rij, nlist,
            coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, frame_nall, rcut_r, rcut_r_smth, sec_a);
        if (b_nlist_map)
            _map_nlist_gpu(nlist, idx_mapping, nloc, nnei);
        deepmd::delete_device_memory(firstneigh);
        deepmd::delete_device_memory(array_longlong);
        array_longlong = NULL;
    }
}


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
  paddle::Tensor natoms_cpu_tensor = natoms_tensor.copy_to<int>(paddle::PlaceType::kCPU);
//   paddle::Tensor box_cpu_tensor = box_tensor.copy_to<float>(paddle::PlaceType::kCPU);
  const int *natoms = natoms_cpu_tensor.data<int>();
  
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
  paddle::Tensor descrpt_tensor = paddle::Tensor(paddle::PlaceType::kGPU, descrpt_shape);
  paddle::Tensor descrpt_deriv_tensor = paddle::Tensor(paddle::PlaceType::kGPU, descrpt_deriv_shape);
  paddle::Tensor rij_tensor = paddle::Tensor(paddle::PlaceType::kGPU, rij_shape);
  paddle::Tensor nlist_tensor = paddle::Tensor(paddle::PlaceType::kGPU, nlist_shape);

    PD_DISPATCH_FLOATING_TYPES(
        coord_tensor.type(), "pd_prod_env_mat_a_cuda_forward_kernel", ([&] {
            PdProdEnvMatAOpCUDAForwardKernel<data_t>(
                nsamples, nloc, ndescrpt, nnei, nall, mem_cpy, mem_nnei, max_nbor_size,
                nei_mode, rcut_a, rcut_r, rcut_r_smth, max_cpy_trial, max_nnei_trial, b_nlist_map, sec_a, sec_r,
                gpu_inlist, nbor_list_dev, array_int, array_longlong,
                descrpt_tensor.mutable_data<data_t>(),
                descrpt_deriv_tensor.mutable_data<data_t>(),
                rij_tensor.mutable_data<data_t>(),
                nlist_tensor.mutable_data<int>(),
                coord_tensor.data<data_t>(),
                box_tensor.copy_to<data_t>(paddle::PlaceType::kCPU).data<data_t>(),
                avg_tensor.data<data_t>(),
                std_tensor.data<data_t>(),
                type_tensor.data<int>(),
                mesh_tensor);
        }));

    return {descrpt_tensor, descrpt_deriv_tensor, rij_tensor, nlist_tensor};
}


template <typename FPTYPE>
static int 
_norm_copy_coord_gpu(
    std::vector<paddle::Tensor>* tensor_list,
    FPTYPE *&coord_cpy,
    int *&type_cpy,
    int *&idx_mapping,
    int &nall,
    int &mem_cpy,
    const FPTYPE *coord,
    const FPTYPE *box,
    const int *type,
    const int &nloc,
    const int &max_cpy_trial,
    const float &rcut_r)
{
    // Tensor FPTYPE_temp;
    std::vector<int64_t> FPTYPE_temp_shape{nall*3};
    paddle::Tensor tmp_coord_tensor = paddle::Tensor(paddle::PlaceType::kGPU, FPTYPE_temp_shape);
    FPTYPE *tmp_coord = tmp_coord_tensor.mutable_data<FPTYPE>(paddle::PlaceType::kGPU);
    tensor_list->push_back(tmp_coord_tensor);
    cudaErrcheck(cudaMemcpy(tmp_coord, coord, sizeof(FPTYPE) * nall * 3, cudaMemcpyDeviceToDevice));

    deepmd::Region<FPTYPE> region;
    init_region_cpu(region, box);
    FPTYPE box_info[18];
    std::copy(region.boxt, region.boxt + 9, box_info);
    std::copy(region.rec_boxt, region.rec_boxt + 9, box_info + 9);
    int cell_info[23];
    deepmd::compute_cell_info(cell_info, rcut_r, region);
    const int loc_cellnum = cell_info[21];
    const int total_cellnum = cell_info[22];
    
    //Tensor double_temp;
    std::vector<int64_t> double_temp_shape {18};
    paddle::Tensor double_temp_tensor = paddle::Tensor(paddle::PlaceType::kGPU, double_temp_shape);
    FPTYPE *box_info_dev = double_temp_tensor.mutable_data<FPTYPE>(paddle::PlaceType::kGPU);
    tensor_list->push_back(double_temp_tensor);
    
    //Tensor int_temp;
    std::vector<int64_t> int_temp_shape {23+nloc*3+loc_cellnum+total_cellnum*3+total_cellnum*3+loc_cellnum+1+total_cellnum+1+nloc};
    paddle::Tensor int_temp_tensor = paddle::Tensor(paddle::PlaceType::kGPU, int_temp_shape);
    int *cell_info_dev = int_temp_tensor.mutable_data<int>(paddle::PlaceType::kGPU);
    int *int_data_dev = cell_info_dev + 23;
    tensor_list->push_back(int_temp_tensor);
    
    deepmd::memcpy_host_to_device(box_info_dev, box_info, 18);
    deepmd::memcpy_host_to_device(cell_info_dev, cell_info, 23);

    deepmd::Region<FPTYPE> region_dev;
    FPTYPE *new_boxt = region_dev.boxt;
    FPTYPE *new_rec_boxt = region_dev.rec_boxt;
    region_dev.boxt = box_info_dev;
    region_dev.rec_boxt = box_info_dev + 9;

    deepmd::normalize_coord_gpu(tmp_coord, nall, region_dev);

    
    int tt;
    paddle::Tensor cpy_temp_tensor = paddle::Tensor(paddle::PlaceType::kGPU);
    paddle::Tensor t_temp_tensor = paddle::Tensor(paddle::PlaceType::kGPU);
    for (tt = 0; tt < max_cpy_trial; ++tt)
    {
         std::vector<int64_t> cpy_temp_shape {mem_cpy * 3};
         std::vector<int64_t> t_temp_shape {mem_cpy * 2};
        cpy_temp_tensor.reshape(cpy_temp_shape);
        coord_cpy = cpy_temp_tensor.mutable_data<FPTYPE>(paddle::PlaceType::kGPU);
        t_temp_tensor.reshape(t_temp_shape);
        type_cpy = t_temp_tensor.mutable_data<int>(paddle::PlaceType::kGPU);
        
        idx_mapping = type_cpy + mem_cpy;
        int ret = deepmd::copy_coord_gpu(
            coord_cpy, type_cpy, idx_mapping, &nall, int_data_dev,
            tmp_coord, type, nloc, mem_cpy, loc_cellnum, total_cellnum, cell_info_dev, region_dev);
        if (ret == 0)
        {
            break;
        }
        else
        {
            mem_cpy *= 2;
        }
    }
    tensor_list->push_back(cpy_temp_tensor);
    tensor_list->push_back(t_temp_tensor);
    region_dev.boxt = new_boxt;
    region_dev.rec_boxt = new_rec_boxt;
    
    return (tt != max_cpy_trial);
}

template <typename FPTYPE>
static int
_build_nlist_gpu(
    std::vector<paddle::Tensor> *tensor_list,
    int *&ilist,
    int *&numneigh,
    int **&firstneigh,
    int *&jlist,
    int &max_nnei,
    int &mem_nnei,
    const FPTYPE *coord,
    const int &nloc,
    const int &new_nall,
    const int &max_nnei_trial,
    const float &rcut_r)
{
    //Tensor nlist_temp;
    std::vector<int64_t> nlist_temp_shape {nloc * 2};
    paddle::Tensor nlist_temp_tensor = paddle::Tensor(paddle::PlaceType::kGPU, nlist_temp_shape);
    ilist = nlist_temp_tensor.mutable_data<int>(paddle::PlaceType::kGPU);
    tensor_list->push_back(nlist_temp_tensor);
    numneigh = ilist + nloc;
    //Tensor jlist_temp;
    int *ind_data = NULL;

    std::vector<int *> firstneigh_host(nloc);
    int tt;
    paddle::Tensor jlist_temp_tensor = paddle::Tensor(paddle::PlaceType::kGPU);
    for (tt = 0; tt < max_nnei_trial; ++tt)
    {   
        std::vector<int64_t> jlist_temp_shape {3 * nloc * mem_nnei};
        jlist_temp_tensor.reshape(jlist_temp_shape);
        jlist = jlist_temp_tensor.mutable_data<int>(paddle::PlaceType::kGPU);
        ind_data = jlist + nloc * mem_nnei;
        for (int ii = 0; ii < nloc; ++ii)
        {
            firstneigh_host[ii] = jlist + ii * mem_nnei;
        }
        deepmd::memcpy_host_to_device(firstneigh, firstneigh_host);
        deepmd::InputNlist inlist(nloc, ilist, numneigh, firstneigh);
        int ret = deepmd::build_nlist_gpu(
            inlist, &max_nnei, ind_data,
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
    tensor_list->push_back(jlist_temp_tensor);
    return (tt != max_nnei_trial);
}

static void
_map_nlist_gpu(
    int *nlist,
    const int *idx_mapping,
    const int &nloc,
    const int &nnei)
{
    deepmd::use_nlist_map(nlist, idx_mapping, nloc, nnei);
}

template <typename FPTYPE>
static void
_prepare_coord_nlist_gpu(
    std::vector<paddle::Tensor> *tensor_list,
    FPTYPE const **coord,
    FPTYPE *&coord_cpy,
    int const **type,
    int *&type_cpy,
    int *&idx_mapping,
    deepmd::InputNlist &inlist,
    int *&ilist,
    int *&numneigh,
    int **&firstneigh,
    int *&jlist,
    int *&nbor_list_dev,
    int &new_nall,
    int &mem_cpy,
    int &mem_nnei,
    int &max_nbor_size,
    const FPTYPE *box,
    const int *mesh_tensor_data,
    const int mesh_tensor_size,
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
            int copy_ok = _norm_copy_coord_gpu(
                tensor_list, coord_cpy, type_cpy, idx_mapping, new_nall, mem_cpy,
                *coord, box, *type, nloc, max_cpy_trial, rcut_r);
            PD_CHECK(copy_ok, "cannot allocate mem for copied coords");
            *coord = coord_cpy;
            *type = type_cpy;
            
        }

        //build nlist
        int build_ok = _build_nlist_gpu(
            tensor_list, ilist, numneigh, firstneigh, jlist, max_nbor_size, mem_nnei,
            *coord, nloc, new_nall, max_nnei_trial, rcut_r);
        PD_CHECK(build_ok, "cannot allocate mem for nlist");
        if (max_nbor_size <= 1024)
        {
            max_nbor_size = 1024;
        }
        else if (max_nbor_size <= 2048)
        {
            max_nbor_size = 2048;
        }
        else
        {
            max_nbor_size = 4096;
        }
        inlist.ilist = ilist;
        inlist.numneigh = numneigh;
        inlist.firstneigh = firstneigh;
    }
    else
    {
        // update nbor list
        deepmd::InputNlist inlist_temp;
        inlist_temp.inum = nloc;
        deepmd::env_mat_nbor_update(
            inlist_temp, inlist, max_nbor_size, nbor_list_dev,
            mesh_tensor_data, mesh_tensor_size);
        PD_CHECK((max_numneigh(inlist_temp) <= GPU_MAX_NBOR_SIZE), "Assert failed, max neighbor size of atom(lammps) " + std::to_string(max_numneigh(inlist_temp)) + " is larger than " + std::to_string(GPU_MAX_NBOR_SIZE) + ", which currently is not supported by deepmd-kit.");
    }
}