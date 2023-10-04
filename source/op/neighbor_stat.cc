// SPDX-License-Identifier: LGPL-3.0-or-later
#include "neighbor_stat.h"

#include "custom_op.h"
#include "errors.h"
#include "neighbor_list.h"

typedef double boxtensor_t;
typedef double compute_t;

REGISTER_OP("NeighborStat")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("coord: T")
    .Input("type: int32")
    .Input("natoms: int32")
    .Input("box : T")
    .Input("mesh : int32")
    .Attr("rcut: float")
    .Output("max_nbor_size: int32")
    .Output("min_nbor_dist: T");

template <typename Device, typename FPTYPE>
class NeighborStatOp : public OpKernel {
 public:
  explicit NeighborStatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rcut", &rcut));
    max_nbor_size_nlist = 1024;
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
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3),
                errors::InvalidArgument(
                    "number of atoms should be larger than (or equal to) 3"));
    int nloc = natoms_tensor.flat<int>().data()[0];
    int nall = natoms_tensor.flat<int>().data()[1];
    int nsamples = coord_tensor.shape().dim_size(0);
    int ntypes = natoms_tensor.shape().dim_size(0) - 2;
    // check the sizes
    OP_REQUIRES(context, (nsamples == type_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nsamples == box_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nall * 3 == coord_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context, (nall == type_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context, (9 == box_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of box should be 9"));
    DeviceFunctor()(device, context->eigen_device<Device>());
    int nei_mode = 0;
    if (mesh_tensor.shape().dim_size(0) == 6 ||
        mesh_tensor.shape().dim_size(0) == 7) {
      // manual copied pbc
      assert(nloc == nall);
      nei_mode = 1;
    } else if (mesh_tensor.shape().dim_size(0) == 0 ||
               mesh_tensor.shape().dim_size(0) == 1) {
      // no pbc
      nei_mode = -1;
    } else {
      throw deepmd::deepmd_exception("invalid mesh tensor");
    }
    // if region is given extended, do not use pbc
    bool b_pbc = (nei_mode >= 1 || nei_mode == -1) ? false : true;
    bool b_norm_atom = (nei_mode == 1) ? true : false;

    TensorShape max_nbor_size_shape;
    max_nbor_size_shape.AddDim(nloc);
    max_nbor_size_shape.AddDim(ntypes);

    int context_output_index = 0;
    Tensor* max_nbor_size_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     max_nbor_size_shape,
                                                     &max_nbor_size_tensor));

    const FPTYPE* coord = coord_tensor.flat<FPTYPE>().data();
    const int* type = type_tensor.flat<int>().data();
    const FPTYPE* box = box_tensor.flat<FPTYPE>().data();
    const int* mesh = mesh_tensor.flat<int>().data();
    int* max_nbor_size = max_nbor_size_tensor->flat<int>().data();
    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      std::vector<Tensor> tensor_list(7);
      if (nei_mode == 1) {
        // Tensor FPTYPE_temp;
        TensorShape FPTYPE_shape;
        FPTYPE_shape.AddDim(nall * 3);
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
        cpy_shape.AddDim(mem_cpy * 3);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<FPTYPE>::value,
                                              cpy_shape, &tensor_list[3]));
        // Tensor t_temp;
        TensorShape t_shape;
        t_shape.AddDim(mem_cpy * 2);
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, t_shape,
                                                       &tensor_list[4]));
      }

      // Tensor nlist_temp;
      TensorShape nlist_shape;
      nlist_shape.AddDim(nloc * 2);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, nlist_shape,
                                                     &tensor_list[5]));

      TensorShape jlist_shape;
      jlist_shape.AddDim(3 * int_64(nloc) * mem_nnei);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, jlist_shape,
                                                     &tensor_list[6]));

      int* idx_mapping = NULL;
      int *ilist = NULL, *numneigh = NULL;
      int** firstneigh = NULL;
      deepmd::malloc_device_memory(firstneigh, nloc);
      int* jlist = NULL;
      FPTYPE* coord_cpy;
      int* type_cpy;
      int frame_nall = nall;
      int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
      deepmd::InputNlist gpu_inlist;
      int* nbor_list_dev = NULL;
      // prepare coord and nlist
      _prepare_coord_nlist_gpu<FPTYPE>(
          context, &tensor_list[0], &coord, coord_cpy, &type, type_cpy,
          idx_mapping, gpu_inlist, ilist, numneigh, firstneigh, jlist,
          nbor_list_dev, frame_nall, mem_cpy, mem_nnei, max_nbor_size_nlist,
          box, mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc, nei_mode,
          rcut, max_cpy_trial, max_nnei_trial);

      TensorShape min_nbor_dist_shape;
      min_nbor_dist_shape.AddDim(nloc * mem_nnei);
      Tensor* min_nbor_dist_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                       min_nbor_dist_shape,
                                                       &min_nbor_dist_tensor));
      FPTYPE* min_nbor_dist = min_nbor_dist_tensor->flat<FPTYPE>().data();

      deepmd::neighbor_stat_gpu<FPTYPE>(coord, type, nloc, gpu_inlist,
                                        max_nbor_size, min_nbor_dist, ntypes,
                                        mem_nnei);
      deepmd::delete_device_memory(firstneigh);
#endif
    } else {
      for (int ii = 0;
           ii < static_cast<int>(max_nbor_size_tensor->NumElements()); ii++) {
        max_nbor_size[ii] = 0;
      }

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
            if (inter[dd] < 0) {
              inter[dd] += 1.;
            } else if (inter[dd] >= 1) {
              inter[dd] -= 1.;
            }
          }
          region.inter2Phys(&d_coord3[3 * ii], inter);
        }
      }

      // set type
      std::vector<int> d_type(nall);
      for (int ii = 0; ii < nall; ++ii) {
        d_type[ii] = type[ii];
      }

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
      TensorShape min_nbor_dist_shape;
      min_nbor_dist_shape.AddDim(nloc * MAX_NNEI);
      Tensor* min_nbor_dist_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                       min_nbor_dist_shape,
                                                       &min_nbor_dist_tensor));
      FPTYPE* min_nbor_dist = min_nbor_dist_tensor->flat<FPTYPE>().data();
      for (int ii = 0;
           ii < static_cast<int>(min_nbor_dist_tensor->NumElements()); ii++) {
        min_nbor_dist[ii] = 10000.0;
      }

#pragma omp parallel for
      for (int ii = 0; ii < nloc; ii++) {
        if (d_type[ii] < 0) {
          continue;  // virtual atom
        }
        for (int jj = 0; jj < d_nlist_r[ii].size(); jj++) {
          int type = d_type[d_nlist_r[ii][jj]];
          if (type < 0) {
            continue;  // virtual atom
          }
          max_nbor_size[ii * ntypes + type] += 1;
          compute_t rij[3] = {
              d_coord3[d_nlist_r[ii][jj] * 3 + 0] - d_coord3[ii * 3 + 0],
              d_coord3[d_nlist_r[ii][jj] * 3 + 1] - d_coord3[ii * 3 + 1],
              d_coord3[d_nlist_r[ii][jj] * 3 + 2] - d_coord3[ii * 3 + 2]};
          // we do not need to do slow sqrt for every dist; instead do sqrt in
          // the final step
          min_nbor_dist[ii * MAX_NNEI + jj] =
              rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
        }
      }
    }
  }

 private:
  int nnei;
  float rcut;
  std::string device;
  int max_nbor_size_nlist, max_cpy_trial, mem_cpy, max_nnei_trial, mem_nnei;
};

#define REGISTER_CPU(T)                                               \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("NeighborStat").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      NeighborStatOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(T)                               \
  REGISTER_KERNEL_BUILDER(Name("NeighborStat")        \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("natoms")   \
                              .HostMemory("box"),     \
                          NeighborStatOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
