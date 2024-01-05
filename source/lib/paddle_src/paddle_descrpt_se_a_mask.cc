#include <cmath>

#include "ComputeDescriptor.h"
#include "errors.h"
#include "fmt_nlist.h"
#include "neighbor_list.h"
#include "paddle/extension.h"

typedef double boxtensor_t;
typedef double compute_t;

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT_READY(x) \
  PD_CHECK(x.initialized(), #x " must be initialized before usage.")
#define CHECK_INPUT_INT32(x) \
  PD_CHECK(x.dtype() == paddle::DataType::INT32, #x " dtype should be INT32.")
template <typename FPTYPE>
struct NeighborInfo {
  int type;
  FPTYPE dist;
  int index;
  NeighborInfo() : type(0), dist(0), index(0) {}
  NeighborInfo(int tt, FPTYPE dd, int ii) : type(tt), dist(dd), index(ii) {}
  bool operator<(const NeighborInfo& b) const {
    return (type < b.type ||
            (type == b.type &&
             (dist < b.dist || (dist == b.dist && index < b.index))));
  }
};

compute_t max_distance = 10000.0;

void buildAndSortNeighborList(int i_idx,
                              const std::vector<compute_t> d_coord3,
                              std::vector<int>& d_type,
                              std::vector<int>& d_mask,
                              std::vector<int>& sorted_nlist,
                              int total_atom_num) {
  // sorted_nlist.resize(total_atom_num);
  std::vector<NeighborInfo<double>> sel_nei;
  for (int jj = 0; jj < total_atom_num; jj++) {
    compute_t diff[3];
    const int j_idx = jj;
    for (int dd = 0; dd < 3; ++dd) {
      diff[dd] = d_coord3[j_idx * 3 + dd] - d_coord3[i_idx * 3 + dd];
    }
    // Check if j_idx atom is virtual particle or not.
    compute_t rr = 0.0;
    if (d_mask[j_idx] == 0 || j_idx == i_idx) {
      rr = max_distance;
    } else {
      rr = sqrt(deepmd::dot3<compute_t>(diff, diff));
    }
    sel_nei.push_back(NeighborInfo<double>(d_type[j_idx], rr, j_idx));
  }
  std::sort(sel_nei.begin(), sel_nei.end());
  // Save the sorted atom index.
  for (int jj = 0; jj < sel_nei.size(); jj++) {
    int atom_idx = sel_nei[jj].index;
    sorted_nlist[jj] = atom_idx;
  }
}

template <typename data_t>
void DescrptSeAMaskCPUKernel(int n_descrpt,
                             int total_atom_num,
                             int nsamples,
                             const data_t* coord,
                             const int* type,
                             const int* mask_matrix,
                             data_t* descrpt,
                             data_t* descrpt_deriv,
                             data_t* rij,
                             int* nlist) {
#pragma omp parallel for
  for (int kk = 0; kk < nsamples; ++kk) {
    // Iterate for each frame.
    int nloc = total_atom_num;
    int natoms = total_atom_num;

    std::vector<compute_t> d_coord3(natoms * 3);
    for (int ii = 0; ii < natoms; ++ii) {
      for (int dd = 0; dd < 3; ++dd) {
        d_coord3[ii * 3 + dd] = coord[kk * total_atom_num * 3 + ii * 3 + dd];
      }
    }

    std::vector<int> d_type(natoms);
    for (int ii = 0; ii < natoms; ++ii) {
      d_type[ii] = type[kk * total_atom_num + ii];
    }

    std::vector<int> d_mask(natoms);
    for (int ii = 0; ii < natoms; ++ii) {
      d_mask[ii] = mask_matrix[kk * total_atom_num + ii];
    }
    std::vector<int> sorted_nlist(total_atom_num);

    for (int ii = 0; ii < nloc; ii++) {
      // Check this atom is virtual atom or not. If it is, set the virtual
      // atom's environment descriptor and derivation on descriptor to be zero
      // directly.
      if (mask_matrix[kk * total_atom_num + ii] == 0) {
        for (int jj = 0; jj < natoms * 4; ++jj) {
          descrpt[kk * total_atom_num * total_atom_num * n_descrpt +
                  ii * total_atom_num * 4 + jj] = 0.;
        }
        for (int jj = 0; jj < natoms * 4 * 3; ++jj) {
          descrpt_deriv[kk * total_atom_num * total_atom_num * n_descrpt * 3 +
                        ii * total_atom_num * 4 * 3 + jj] = 0.;
        }
        // Save the neighbor list relative coordinates with center atom ii.
        for (int jj = 0; jj < natoms * 3; ++jj) {
          rij[kk * total_atom_num * total_atom_num * 3 + ii * natoms * 3 + jj] =
              0.;
        }
        // Save the neighbor atoms indicies.
        for (int jj = 0; jj < natoms; jj++) {
          nlist[kk * total_atom_num * total_atom_num + ii * natoms + jj] = -1;
        }
        continue;
      }

      // Build the neighbor list for atom ii.
      std::fill(sorted_nlist.begin(), sorted_nlist.end(), -1);
      buildAndSortNeighborList(ii, d_coord3, d_type, d_mask, sorted_nlist,
                               total_atom_num);

      // Set the center atom coordinates.
      std::vector<compute_t> rloc(3);
      for (int dd = 0; dd < 3; ++dd) {
        rloc[dd] = coord[kk * total_atom_num * 3 + ii * 3 + dd];
      }

      // Compute the descriptor and derive for the descriptor for each atom.
      std::vector<compute_t> descrpt_atom(natoms * 4);
      std::vector<compute_t> descrpt_deriv_atom(natoms * 12);
      std::vector<compute_t> rij_atom(natoms * 3);

      std::fill(descrpt_deriv_atom.begin(), descrpt_deriv_atom.end(), 0.0);
      std::fill(descrpt_atom.begin(), descrpt_atom.end(), 0.0);
      std::fill(rij_atom.begin(), rij_atom.end(), 0.0);

      // Compute the each environment std::vector for each atom.
      for (int jj = 0; jj < natoms; jj++) {
        int j_idx = sorted_nlist[jj];

        compute_t temp_rr;
        compute_t temp_diff[3];
        temp_rr = 0.;

        // Once ii == j_idx, the descriptor and derivation should be set to
        // zero. Or if the atom jj is an virtual atom. The descriptor and
        // derivation should be zero also.
        if (ii == j_idx || mask_matrix[kk * total_atom_num + j_idx] == 0) {
          // 1./rr, cos(theta), cos(phi), sin(phi)
          descrpt_atom[jj * 4 + 0] = 0.;
          descrpt_atom[jj * 4 + 1] = 0.;
          descrpt_atom[jj * 4 + 2] = 0.;
          descrpt_atom[jj * 4 + 3] = 0.;
          // derive of the component 1/r
          descrpt_deriv_atom[jj * 12 + 0] = 0.;
          descrpt_deriv_atom[jj * 12 + 1] = 0.;
          descrpt_deriv_atom[jj * 12 + 2] = 0.;
          // derive of the component x/r2
          descrpt_deriv_atom[jj * 12 + 3] = 0.;  // on x.
          descrpt_deriv_atom[jj * 12 + 4] = 0.;  // on y.
          descrpt_deriv_atom[jj * 12 + 5] = 0.;  // on z.
          // derive of the component y/r2
          descrpt_deriv_atom[jj * 12 + 6] = 0.;  // on x.
          descrpt_deriv_atom[jj * 12 + 7] = 0.;  // on y.
          descrpt_deriv_atom[jj * 12 + 8] = 0.;  // on z.
          // derive of the component z/r2
          descrpt_deriv_atom[jj * 12 + 9] = 0.;   // on x.
          descrpt_deriv_atom[jj * 12 + 10] = 0.;  // on y.
          descrpt_deriv_atom[jj * 12 + 11] = 0.;  // on z.
          rij_atom[jj * 3 + 0] = 0.;
          rij_atom[jj * 3 + 1] = 0.;
          rij_atom[jj * 3 + 2] = 0.;
          continue;
        }

        for (int dd = 0; dd < 3; dd++) {
          temp_diff[dd] = d_coord3[j_idx * 3 + dd] - rloc[dd];
          rij_atom[jj * 3 + dd] = temp_diff[dd];
        }

        temp_rr = deepmd::dot3<compute_t>(temp_diff, temp_diff);

        compute_t x = temp_diff[0];
        compute_t y = temp_diff[1];
        compute_t z = temp_diff[2];

        // r^2
        compute_t nr2 = temp_rr;
        // 1/r
        compute_t inr = 1. / sqrt(nr2);
        // r
        compute_t nr = nr2 * inr;
        // 1/r^2
        compute_t inr2 = inr * inr;
        // 1/r^4
        compute_t inr4 = inr2 * inr2;
        // 1/r^3
        compute_t inr3 = inr * inr2;
        // 1./rr, cos(theta), cos(phi), sin(phi)
        descrpt_atom[jj * 4 + 0] = 1. / nr;
        descrpt_atom[jj * 4 + 1] = x / nr2;
        descrpt_atom[jj * 4 + 2] = y / nr2;
        descrpt_atom[jj * 4 + 3] = z / nr2;
        // derive of the component 1/r
        descrpt_deriv_atom[jj * 12 + 0] = x * inr3;
        descrpt_deriv_atom[jj * 12 + 1] = y * inr3;
        descrpt_deriv_atom[jj * 12 + 2] = z * inr3;
        // derive of the component x/r2
        descrpt_deriv_atom[jj * 12 + 3] = 2. * x * x * inr4 - inr2;  // on x.
        descrpt_deriv_atom[jj * 12 + 4] = 2. * x * y * inr4;         // on y.
        descrpt_deriv_atom[jj * 12 + 5] = 2. * x * z * inr4;         // on z.
        // derive of the component y/r2
        descrpt_deriv_atom[jj * 12 + 6] = 2. * y * x * inr4;         // on x.
        descrpt_deriv_atom[jj * 12 + 7] = 2. * y * y * inr4 - inr2;  // on y.
        descrpt_deriv_atom[jj * 12 + 8] = 2. * y * z * inr4;         // on z.
        // derive of the component z/r2
        descrpt_deriv_atom[jj * 12 + 9] = 2. * z * x * inr4;          // on x.
        descrpt_deriv_atom[jj * 12 + 10] = 2. * z * y * inr4;         // on y.
        descrpt_deriv_atom[jj * 12 + 11] = 2. * z * z * inr4 - inr2;  // on z.
      }

      for (int jj = 0; jj < natoms * 4; ++jj) {
        descrpt[kk * total_atom_num * total_atom_num * n_descrpt +
                ii * total_atom_num * 4 + jj] = descrpt_atom[jj];
      }
      for (int jj = 0; jj < natoms * 4 * 3; ++jj) {
        descrpt_deriv[kk * total_atom_num * total_atom_num * n_descrpt * 3 +
                      ii * total_atom_num * 4 * 3 + jj] =
            descrpt_deriv_atom[jj];
      }
      // Save the neighbor list relative coordinates with center atom ii.
      for (int jj = 0; jj < natoms * 3; ++jj) {
        rij[kk * total_atom_num * total_atom_num * 3 + ii * natoms * 3 + jj] =
            rij_atom[jj];
      }
      // Save the neighbor atoms indicies.
      for (int jj = 0; jj < natoms; ++jj) {
        nlist[kk * total_atom_num * total_atom_num + ii * natoms + jj] =
            sorted_nlist[jj];
      }
    }
  }
}

std::vector<paddle::Tensor> DescrptSeAMaskCPU(
    const paddle::Tensor& coord_tensor,
    const paddle::Tensor& type_tensor,
    const paddle::Tensor& mask_matrix_tensor,
    const paddle::Tensor& box_tensor,
    const paddle::Tensor& natoms_tensor,
    const paddle::Tensor& mesh_tensor) {
  CHECK_INPUT(coord_tensor);
  CHECK_INPUT(type_tensor);
  CHECK_INPUT(mask_matrix_tensor);
  CHECK_INPUT(box_tensor);
  CHECK_INPUT(natoms_tensor);
  CHECK_INPUT(mesh_tensor);

  CHECK_INPUT_INT32(type_tensor);
  CHECK_INPUT_INT32(mask_matrix_tensor);
  CHECK_INPUT_INT32(natoms_tensor);
  CHECK_INPUT_INT32(mesh_tensor);
  // set size of the sample
  CHECK_INPUT_DIM(coord_tensor, 2);
  CHECK_INPUT_DIM(type_tensor, 2);
  CHECK_INPUT_DIM(mask_matrix_tensor, 2);

  int nsamples = coord_tensor.shape()[0];

  // check the sizes
  PD_CHECK(nsamples == type_tensor.shape()[0],
           "number of samples should match");
  PD_CHECK(nsamples == mask_matrix_tensor.shape()[0],
           "number of samples should match");

  // Set n_descrpt for each atom. Include 1/rr, cos(theta), cos(phi), sin(phi)
  int n_descrpt = 4;

  // Calculate the total_atom_num
  const int* natoms = natoms_tensor.data<int>();
  int total_atom_num = natoms[1];
  // check the sizes
  PD_CHECK(total_atom_num * 3 == coord_tensor.shape()[1],
           "number of samples should match");
  PD_CHECK(total_atom_num == mask_matrix_tensor.shape()[1],
           "number of samples should match");

  // create output tensor
  std::vector<int64_t> descrpt_shape{
      nsamples, total_atom_num * total_atom_num * n_descrpt};
  std::vector<int64_t> descrpt_deriv_shape{
      nsamples, total_atom_num * total_atom_num * n_descrpt * 3};
  std::vector<int64_t> rij_shape{nsamples, total_atom_num * total_atom_num * 3};
  std::vector<int64_t> nlist_shape{nsamples, total_atom_num * total_atom_num};
  paddle::Tensor descrpt_tensor =
      paddle::empty(descrpt_shape, coord_tensor.dtype(), coord_tensor.place());
  paddle::Tensor descrpt_deriv_tensor = paddle::empty(
      descrpt_deriv_shape, coord_tensor.dtype(), coord_tensor.place());
  paddle::Tensor rij_tensor =
      paddle::empty(rij_shape, coord_tensor.dtype(), coord_tensor.place());
  paddle::Tensor nlist_tensor =
      paddle::empty(nlist_shape, type_tensor.dtype(), type_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      coord_tensor.type(), "descrpt_se_a_mask_kernel", ([&] {
        DescrptSeAMaskCPUKernel<data_t>(
            n_descrpt, total_atom_num, nsamples, coord_tensor.data<data_t>(),
            type_tensor.data<int>(), mask_matrix_tensor.data<int>(),
            descrpt_tensor.data<data_t>(), descrpt_deriv_tensor.data<data_t>(),
            rij_tensor.data<data_t>(), nlist_tensor.data<int>());
      }));
  return {descrpt_tensor, descrpt_deriv_tensor, rij_tensor, nlist_tensor};
}

std::vector<paddle::Tensor> DescrptSeAMask(
    const paddle::Tensor& coord_tensor,
    const paddle::Tensor& type_tensor,
    const paddle::Tensor& mask_matrix_tensor,
    const paddle::Tensor& box_tensor,
    const paddle::Tensor& natoms_tensor,
    const paddle::Tensor& mesh_tensor) {
  if (coord_tensor.is_cpu()) {
    return DescrptSeAMaskCPU(coord_tensor, type_tensor, mask_matrix_tensor,
                             box_tensor, natoms_tensor, mesh_tensor);
  } else {
    PD_THROW("DescrptSeAMask only support CPU device.");
  }
}

PD_BUILD_OP(descrpt_se_a_mask)
    .Inputs({"coord", "type", "mask", "box", "natoms", "mesh"})
    .Outputs({"descrpt", "descrpt_deriv", "rij", "nlist"})
    .SetKernelFn(PD_KERNEL(DescrptSeAMask));
