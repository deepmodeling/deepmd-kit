#include "device.h"
#include "errors.h"
#include "neighbor_list.h"
#include "paddle/extension.h"

#undef PADDLE_WITH_CUDA
#define CHECK_INPUT_CPU(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
typedef double boxtensor_t;
typedef double compute_t;

std::vector<paddle::Tensor> NeighborStatOpCPUForward(
    const paddle::Tensor& coord_tensor /*fp32/64*/,
    const paddle::Tensor& type_tensor /*int32*/,
    const paddle::Tensor& natoms_tensor /*int64*/,
    const paddle::Tensor& box_tensor /*fp32/64*/,
    const paddle::Tensor& mesh_tensor /*int32*/,
    const float& rcut) {
  CHECK_INPUT_CPU(coord_tensor);
  CHECK_INPUT_CPU(type_tensor);
  CHECK_INPUT_CPU(natoms_tensor);
  CHECK_INPUT_CPU(box_tensor);
  CHECK_INPUT_CPU(mesh_tensor);

  CHECK_INPUT_DIM(coord_tensor, 2);
  CHECK_INPUT_DIM(type_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);
  CHECK_INPUT_DIM(box_tensor, 2);
  CHECK_INPUT_DIM(mesh_tensor, 1);
  PD_CHECK(natoms_tensor.shape()[0] >= 3,
           "number of atoms should be larger than (or equal to) 3");

  const int64_t* natoms = natoms_tensor.data<int64_t>();
  int64_t nloc = natoms[0];
  int64_t nall = natoms[1];
  int64_t nsamples = coord_tensor.shape()[0];
  int64_t ntypes = natoms_tensor.shape()[0] - 2;

  PD_CHECK(nsamples == type_tensor.shape()[0],
           "number of samples should match");
  PD_CHECK(nsamples == box_tensor.shape()[0], "number of samples should match");
  PD_CHECK(nall * 3 == coord_tensor.shape()[1], "number of atoms should match");
  PD_CHECK(nall == type_tensor.shape()[1], "number of atoms should match");
  PD_CHECK(9 == box_tensor.shape()[1], "number of box should be 9");

  int nei_mode = 0;
  if (mesh_tensor.shape()[0] == 6) {
    // manual copied pbc
    assert(nloc == nall);
    nei_mode = 1;
  } else if (mesh_tensor.shape()[0] == 0) {
    // no pbc
    nei_mode = -1;
  } else {
    throw deepmd::deepmd_exception("invalid mesh tensor");
  }
  // if region is given extended, do not use pbc
  bool b_pbc = (nei_mode >= 1 || nei_mode == -1) ? false : true;
  bool b_norm_atom = (nei_mode == 1) ? true : false;

  std::vector<int64_t> max_nbor_size_shape = {nloc, ntypes};
  paddle::Tensor max_nbor_size_tensor = paddle::zeros(
      max_nbor_size_shape, type_tensor.dtype(), type_tensor.place());

  const float* coord = coord_tensor.data<float>();
  const int* type = type_tensor.data<int>();
  const float* box = box_tensor.data<float>();
  const int* mesh = mesh_tensor.data<int>();
  int* max_nbor_size = max_nbor_size_tensor.data<int>();

  // set region
  boxtensor_t boxt[9] = {0};
  for (int dd = 0; dd < 9; ++dd) {
    boxt[dd] = box[dd];
  }
  SimulationRegion<compute_t> region;
  region.reinitBox(boxt);
  // set & normalize coord
  std::vector<compute_t> d_coord3(nall * 3);
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      d_coord3[ii * 3 + dd] = coord[ii * 3 + dd];
    }
    if (b_norm_atom) {
      compute_t inter[3];
      region.phys2Inter(inter, &d_coord3[3 * ii]);
      for (int dd = 0; dd < 3; ++dd) {
        if (inter[dd] < 0)
          inter[dd] += 1.;
        else if (inter[dd] >= 1)
          inter[dd] -= 1.;
      }
      region.inter2Phys(&d_coord3[3 * ii], inter);
    }
  }

  // set type
  std::vector<int> d_type(nall);
  for (int ii = 0; ii < nall; ++ii) d_type[ii] = type[ii];

  // build nlist
  std::vector<std::vector<int> > d_nlist_a;
  std::vector<std::vector<int> > d_nlist_r;
  std::vector<int> nlist_map;
  bool b_nlist_map = false;

  if (nei_mode == 1) {
    // std::cout << "I'm in nei_mode 1" << std::endl;
    std::vector<double> bk_d_coord3 = d_coord3;
    std::vector<int> bk_d_type = d_type;
    std::vector<int> ncell, ngcell;
    copy_coord(d_coord3, d_type, nlist_map, ncell, ngcell, bk_d_coord3,
               bk_d_type, rcut, region);
    b_nlist_map = true;
    std::vector<int> nat_stt(3, 0);
    std::vector<int> ext_stt(3), ext_end(3);
    for (int dd = 0; dd < 3; ++dd) {
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    ::build_nlist(d_nlist_a, d_nlist_r, d_coord3, nloc, -1, rcut, nat_stt,
                  ncell, ext_stt, ext_end, region, ncell);
  } else if (nei_mode == -1) {
    ::build_nlist(d_nlist_a, d_nlist_r, d_coord3, -1, rcut, NULL);
  } else {
    throw deepmd::deepmd_exception("unknow neighbor mode");
  }

  int MAX_NNEI = 0;
  for (int ii = 0; ii < nloc; ii++) {
    MAX_NNEI =
        MAX_NNEI < d_nlist_r[ii].size() ? d_nlist_r[ii].size() : MAX_NNEI;
  }
  // allocate output tensor for deepmd-kit
  std::vector<int64_t> min_nbor_dist_shape = {nloc * MAX_NNEI};
  paddle::Tensor min_nbor_dist_tensor = paddle::full(
      min_nbor_dist_shape, 10000.0, coord_tensor.dtype(), coord_tensor.place());
  auto* min_nbor_dist = min_nbor_dist_tensor.data<float>();

#pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    if (d_type[ii] < 0) continue;  // virtual atom
    for (int jj = 0; jj < d_nlist_r[ii].size(); jj++) {
      int type = d_type[d_nlist_r[ii][jj]];
      if (type < 0) continue;  // virtual atom
      max_nbor_size[ii * ntypes + type] += 1;
      compute_t rij[3] = {
          d_coord3[d_nlist_r[ii][jj] * 3 + 0] - d_coord3[ii * 3 + 0],
          d_coord3[d_nlist_r[ii][jj] * 3 + 1] - d_coord3[ii * 3 + 1],
          d_coord3[d_nlist_r[ii][jj] * 3 + 2] - d_coord3[ii * 3 + 2]};
      min_nbor_dist[ii * MAX_NNEI + jj] =
          sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);
    }
  }
  return {max_nbor_size_tensor, min_nbor_dist_tensor};
}

std::vector<paddle::Tensor> NeighborStatForward(
    const paddle::Tensor& coord_tensor,  /*float32*/
    const paddle::Tensor& type_tensor,   /*int32*/
    const paddle::Tensor& natoms_tensor, /*int64*/
    const paddle::Tensor& box_tensor,    /*float32*/
    const paddle::Tensor& mesh_tensor,   /*int32*/
    float rcut) {
  if (coord_tensor.is_cpu()) {
    return NeighborStatOpCPUForward(coord_tensor, type_tensor, natoms_tensor,
                                    box_tensor, mesh_tensor, rcut);
  } else {
    PD_THROW("NeighborStatForward only support CPU device.");
  }
}

PD_BUILD_OP(neighbor_stat)
    .Inputs({"coord", "type", "natoms", "box", "mesh"})
    .Outputs({"max_nbor_size", "min_nbor_dist"})
    .Attrs({"rcut: float"})
    .SetKernelFn(PD_KERNEL(NeighborStatForward));
