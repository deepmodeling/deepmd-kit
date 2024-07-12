
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <cmath>

#include "ComputeDescriptor.h"
#include "custom_op.h"
#include "errors.h"
#include "fmt_nlist.h"
#include "neighbor_list.h"

typedef double boxtensor_t;
typedef double compute_t;

REGISTER_OP("DescrptSeAMask")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("coord: T")
    .Input("type: int32")
    .Input("mask: int32")
    .Input("box: T")         // Not used in practice
    .Input("natoms: int32")  // Used to fetch total_atom_num and check the size
                             // of input.
    .Input("mesh: int32")    // Not used in practice
    .Output("descrpt: T")
    .Output("descrpt_deriv: T")
    .Output("rij: T")
    .Output("nlist: int32");

template <typename FPTYPE>
struct NeighborInfo {
  int type;
  FPTYPE dist;
  int index;
  NeighborInfo() : type(0), dist(0), index(0) {}
  NeighborInfo(int tt, FPTYPE dd, int ii) : type(tt), dist(dd), index(ii) {}
  bool operator<(const NeighborInfo &b) const {
    return (type < b.type ||
            (type == b.type &&
             (dist < b.dist || (dist == b.dist && index < b.index))));
  }
};

template <typename Device, typename FPTYPE>
class DescrptSeAMaskOp : public OpKernel {
 public:
  explicit DescrptSeAMaskOp(OpKernelConstruction *context) : OpKernel(context) {
    // OP_REQUIRES_OK(context);
  }

  void Compute(OpKernelContext *context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext *context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext *context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor &coord_tensor = context->input(context_input_index++);
    const Tensor &type_tensor = context->input(context_input_index++);
    const Tensor &mask_matrix_tensor = context->input(context_input_index++);
    const Tensor &box_tensor = context->input(context_input_index++);
    const Tensor &natoms_tensor = context->input(context_input_index++);
    const Tensor &mesh_tensor = context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES(context, (coord_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of coord should be 2"));
    OP_REQUIRES(
        context, (type_tensor.shape().dims() == 2),
        errors::InvalidArgument("Dim of type for se_e2_a_mask op should be 2"));
    OP_REQUIRES(context, (mask_matrix_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of mask matrix should be 2"));

    int nsamples = coord_tensor.shape().dim_size(0);

    // check the sizes
    OP_REQUIRES(context, (nsamples == type_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nsamples == mask_matrix_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));

    // Set n_descrpt for each atom. Include 1/rr, cos(theta), cos(phi), sin(phi)
    int n_descrpt = 4;

    // Calculate the total_atom_num
    auto natoms = natoms_tensor.flat<int>();
    total_atom_num = natoms(1);
    // check the sizes
    OP_REQUIRES(context,
                (total_atom_num * 3 == coord_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context,
                (total_atom_num == mask_matrix_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of atoms should match"));

    // Create an output tensor
    TensorShape descrpt_shape;
    descrpt_shape.AddDim(nsamples);
    descrpt_shape.AddDim(static_cast<int64_t>(total_atom_num) * total_atom_num *
                         n_descrpt);
    TensorShape descrpt_deriv_shape;
    descrpt_deriv_shape.AddDim(nsamples);
    descrpt_deriv_shape.AddDim(static_cast<int64_t>(total_atom_num) *
                               total_atom_num * n_descrpt * 3);
    TensorShape rij_shape;
    rij_shape.AddDim(nsamples);
    rij_shape.AddDim(static_cast<int64_t>(total_atom_num) * total_atom_num * 3);
    TensorShape nlist_shape;
    nlist_shape.AddDim(nsamples);
    nlist_shape.AddDim(static_cast<int64_t>(total_atom_num) * total_atom_num);

    int context_output_index = 0;
    Tensor *descrpt_tensor = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(context_output_index++, descrpt_shape,
                                          &descrpt_tensor));
    Tensor *descrpt_deriv_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     descrpt_deriv_shape,
                                                     &descrpt_deriv_tensor));
    Tensor *rij_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     rij_shape, &rij_tensor));
    Tensor *nlist_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++, nlist_shape,
                                            &nlist_tensor));

    auto coord = coord_tensor.matrix<FPTYPE>();
    auto type = type_tensor.matrix<int>();
    auto mask_matrix = mask_matrix_tensor.matrix<int>();

    auto descrpt = descrpt_tensor->matrix<FPTYPE>();
    auto descrpt_deriv = descrpt_deriv_tensor->matrix<FPTYPE>();
    auto rij = rij_tensor->matrix<FPTYPE>();
    auto nlist = nlist_tensor->matrix<int>();

    // // check the type
    // int max_type_v = 0;
    // for (int ii = 0; ii < natoms; ++ii){
    //   if (type(0, ii) > max_type_v) max_type_v = type(0, ii);
    // }
    // int ntype = max_type_v + 1;

    // loop over atoms, compute descriptors for each atom
#pragma omp parallel for
    for (int kk = 0; kk < nsamples; ++kk) {
      // Iterate for each frame.
      int nloc = total_atom_num;
      int natoms = total_atom_num;

      std::vector<compute_t> d_coord3(natoms * 3);
      for (int ii = 0; ii < natoms; ++ii) {
        for (int dd = 0; dd < 3; ++dd) {
          d_coord3[ii * 3 + dd] = coord(kk, ii * 3 + dd);
        }
      }

      std::vector<int> d_type(natoms);
      for (int ii = 0; ii < natoms; ++ii) {
        d_type[ii] = type(kk, ii);
      }

      std::vector<int> d_mask(natoms);
      for (int ii = 0; ii < natoms; ++ii) {
        d_mask[ii] = mask_matrix(kk, ii);
      }
      std::vector<int> sorted_nlist(total_atom_num);

      for (int ii = 0; ii < nloc; ii++) {
        // Check this atom is virtual atom or not. If it is, set the virtual
        // atom's environment descriptor and derivation on descriptor to be zero
        // directly.
        if (mask_matrix(kk, ii) == 0) {
          for (int jj = 0; jj < natoms * 4; ++jj) {
            descrpt(kk, ii * total_atom_num * 4 + jj) = 0.;
          }
          for (int jj = 0; jj < natoms * 4 * 3; ++jj) {
            descrpt_deriv(kk, ii * total_atom_num * 4 * 3 + jj) = 0.;
          }
          // Save the neighbor list relative coordinates with center atom ii.
          for (int jj = 0; jj < natoms * 3; ++jj) {
            rij(kk, ii * natoms * 3 + jj) = 0.;
          }
          // Save the neighbor atoms indicies.
          for (int jj = 0; jj < natoms; jj++) {
            nlist(kk, ii * natoms + jj) = -1;
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
          rloc[dd] = coord(kk, ii * 3 + dd);
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
          if (ii == j_idx || mask_matrix(kk, j_idx) == 0) {
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
          descrpt(kk, ii * total_atom_num * 4 + jj) = descrpt_atom[jj];
        }
        for (int jj = 0; jj < natoms * 4 * 3; ++jj) {
          descrpt_deriv(kk, ii * total_atom_num * 4 * 3 + jj) =
              descrpt_deriv_atom[jj];
        }
        // Save the neighbor list relative coordinates with center atom ii.
        for (int jj = 0; jj < natoms * 3; ++jj) {
          rij(kk, ii * natoms * 3 + jj) = rij_atom[jj];
        }
        // Save the neighbor atoms indicies.
        for (int jj = 0; jj < natoms; ++jj) {
          nlist(kk, ii * natoms + jj) = sorted_nlist[jj];
        }
      }
    }
  }

 private:
  int total_atom_num;
  compute_t max_distance = 10000.0;
  void buildAndSortNeighborList(int i_idx,
                                const std::vector<compute_t> d_coord3,
                                std::vector<int> &d_type,
                                std::vector<int> &d_mask,
                                std::vector<int> &sorted_nlist,
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
};

#define REGISTER_CPU(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DescrptSeAMask").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DescrptSeAMaskOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
