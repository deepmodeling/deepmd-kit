// SPDX-License-Identifier: LGPL-3.0-or-later
#include "pairwise.h"

#include "custom_op.h"

REGISTER_OP("DprcPairwiseIdx")
    .Input("idxs: int32")
    .Input("natoms: int32")
    .Output("forward_qm_map: int32")
    .Output("backward_qm_map: int32")
    .Output("forward_qmmm_map: int32")
    .Output("backward_qmmm_map: int32")
    .Output("natoms_qm: int32")
    .Output("natoms_qmmm: int32")
    .Output("qmmm_frame_idx: int32");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class PairwiseIdxOp : public OpKernel {
 public:
  explicit PairwiseIdxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int tmp_idx = 0;
    const Tensor& idxs_tensor = context->input(tmp_idx++);
    const Tensor& natoms_tensor = context->input(tmp_idx++);

    // set size of the sample
    OP_REQUIRES(context, (idxs_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of idxs should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1),
                errors::InvalidArgument("Dim of natoms should be 1"));

    auto idxs = idxs_tensor.matrix<int>();
    int nframes = idxs_tensor.shape().dim_size(0);
    auto natoms = natoms_tensor.vec<int>();
    int nloc = natoms(0);
    int nall = natoms(1);
    OP_REQUIRES(context, nframes > 0,
                errors::InvalidArgument("nframes should be > 0"));

    std::vector<std::vector<int>> forward_qm_maps, backward_qm_maps,
        forward_qmmm_maps, backward_qmmm_maps;
    std::vector<int> nframes_qmmm, nloc_qm, nloc_qmmm, nghost_qm, nghost_qmmm;
    for (int ii = 0; ii < nframes; ++ii) {
      std::vector<int> v_idxs(nall);
      for (int jj = 0; jj < nall; ++jj) {
        v_idxs[jj] = idxs(ii, jj);
      }
      std::vector<std::vector<int>> fragments;
      std::vector<int> forward_qm_map, backward_qm_map, forward_qmmm_map,
          backward_qmmm_map;
      int nloc_qm_ii, nloc_qmmm_ii, nall_qm_ii, nall_qmmm_ii;
      deepmd::group_atoms_cpu(fragments, v_idxs);
      deepmd::dprc_pairwise_map_cpu(forward_qm_map, backward_qm_map,
                                    forward_qmmm_map, backward_qmmm_map,
                                    nloc_qm_ii, nloc_qmmm_ii, nall_qm_ii,
                                    nall_qmmm_ii, fragments, nloc, nall);
      forward_qm_maps.push_back(forward_qm_map);
      backward_qm_maps.push_back(backward_qm_map);
      forward_qmmm_maps.push_back(forward_qmmm_map);
      backward_qmmm_maps.push_back(backward_qmmm_map);
      // get the maximun
      int nghost_qm_ii = nall_qm_ii - nloc_qm_ii,
          nghost_qmmm_ii = nall_qmmm_ii - nloc_qmmm_ii;
      nloc_qm.push_back(nloc_qm_ii);
      nloc_qmmm.push_back(nloc_qmmm_ii);
      nghost_qm.push_back(nghost_qm_ii);
      nghost_qmmm.push_back(nghost_qmmm_ii);
      nframes_qmmm.push_back(backward_qmmm_map.size() / nall);
    }
    int max_nloc_qm = 0, max_nloc_qmmm = 0, max_nghost_qm = 0,
        max_nghost_qmmm = 0;
    for (int ii = 0; ii < nframes; ++ii) {
      max_nloc_qm = std::max(max_nloc_qm, nloc_qm[ii]);
      max_nloc_qmmm = std::max(max_nloc_qmmm, nloc_qmmm[ii]);
      max_nghost_qm = std::max(max_nghost_qm, nghost_qm[ii]);
      max_nghost_qmmm = std::max(max_nghost_qmmm, nghost_qmmm[ii]);
    }
    int nframes_qmmm_tot =
        std::accumulate(nframes_qmmm.begin(), nframes_qmmm.end(), 0);
    // Create an output tensor
    TensorShape forward_qm_map_shape;
    forward_qm_map_shape.AddDim(nframes);
    forward_qm_map_shape.AddDim(max_nloc_qm + max_nghost_qm);
    TensorShape backward_qm_map_shape;
    backward_qm_map_shape.AddDim(nframes);
    backward_qm_map_shape.AddDim(nall);
    TensorShape forward_qmmm_map_shape;
    forward_qmmm_map_shape.AddDim(nframes_qmmm_tot);
    forward_qmmm_map_shape.AddDim(max_nloc_qmmm + max_nghost_qmmm);
    TensorShape backward_qmmm_map_shape;
    backward_qmmm_map_shape.AddDim(nframes_qmmm_tot);
    backward_qmmm_map_shape.AddDim(nall);
    TensorShape natoms_qm_shape;
    natoms_qm_shape.AddDim(natoms_tensor.shape().dim_size(0));
    TensorShape natoms_qmmm_shape;
    natoms_qmmm_shape.AddDim(natoms_tensor.shape().dim_size(0));
    TensorShape qmmm_frame_idx_shape;
    qmmm_frame_idx_shape.AddDim(nframes_qmmm_tot);

    Tensor* forward_qm_map_tensor = NULL;
    Tensor* backward_qm_map_tensor = NULL;
    Tensor* forward_qmmm_map_tensor = NULL;
    Tensor* backward_qmmm_map_tensor = NULL;
    Tensor* natoms_qm_tensor = NULL;
    Tensor* natoms_qmmm_tensor = NULL;
    Tensor* qmmm_frame_idx_tensor = NULL;

    tmp_idx = 0;
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, forward_qm_map_shape,
                                            &forward_qm_map_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, backward_qm_map_shape,
                                            &backward_qm_map_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, forward_qmmm_map_shape,
                                            &forward_qmmm_map_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, backward_qmmm_map_shape,
                                            &backward_qmmm_map_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(tmp_idx++, natoms_qm_shape,
                                                     &natoms_qm_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, natoms_qmmm_shape,
                                            &natoms_qmmm_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, qmmm_frame_idx_shape,
                                            &qmmm_frame_idx_tensor));
    // copy from vector to tensor
    auto m_forward_qm_map = forward_qm_map_tensor->matrix<int>();
    auto m_backward_qm_map = backward_qm_map_tensor->matrix<int>();
    auto m_forward_qmmm_map = forward_qmmm_map_tensor->matrix<int>();
    auto m_backward_qmmm_map = backward_qmmm_map_tensor->matrix<int>();
    auto m_natoms_qm = natoms_qm_tensor->vec<int>();
    auto m_natoms_qmmm = natoms_qmmm_tensor->vec<int>();
    auto m_qmmm_frame_idx = qmmm_frame_idx_tensor->vec<int>();
    for (int ii = 0, nn = 0; ii < nframes; ++ii) {
      for (int jj = 0; jj < max_nloc_qm + max_nghost_qm; ++jj) {
        if (jj < nloc_qm[ii]) {
          m_forward_qm_map(ii, jj) = forward_qm_maps[ii][jj];
        } else if (jj < max_nloc_qm) {
          m_forward_qm_map(ii, jj) = -1;
        } else if (jj < max_nloc_qm + nghost_qm[ii]) {
          m_forward_qm_map(ii, jj) =
              forward_qm_maps[ii][jj - (max_nloc_qm - nloc_qm[ii])];
        } else {
          m_forward_qm_map(ii, jj) = -1;
        }
      }
      for (int jj = 0; jj < nall; ++jj) {
        m_backward_qm_map(ii, jj) = backward_qm_maps[ii][jj];
      }
      for (int kk = 0; kk < nframes_qmmm[ii]; ++kk) {
        for (int jj = 0; jj < max_nloc_qmmm + max_nghost_qmmm; ++jj) {
          if (jj < nloc_qmmm[ii]) {
            m_forward_qmmm_map(nn, jj) =
                forward_qmmm_maps[ii]
                                 [kk * (nloc_qmmm[ii] + nghost_qmmm[ii]) + jj];
          } else if (jj < max_nloc_qmmm) {
            m_forward_qmmm_map(nn, jj) = -1;
          } else if (jj < max_nloc_qmmm + nghost_qmmm[ii]) {
            m_forward_qmmm_map(nn, jj) =
                forward_qmmm_maps[ii][kk * (nloc_qmmm[ii] + nghost_qmmm[ii]) +
                                      jj - (max_nloc_qmmm - nloc_qmmm[ii])];
          } else {
            m_forward_qmmm_map(nn, jj) = -1;
          }
        }
        for (int jj = 0; jj < nall; ++jj) {
          // max_nloc_qmmm + max_nghost_qmmm
          m_backward_qmmm_map(nn, jj) = backward_qmmm_maps[ii][kk * nall + jj];
        }
        m_qmmm_frame_idx(nn) = ii;
        nn++;
      }
    }
    m_natoms_qm(0) = max_nloc_qm;
    m_natoms_qm(1) = max_nloc_qm + max_nghost_qm;
    m_natoms_qm(2) = max_nloc_qm;
    for (int ii = 3; ii < m_natoms_qm.size(); ++ii) {
      m_natoms_qm(ii) = 0;
    }
    m_natoms_qmmm(0) = max_nloc_qmmm;
    m_natoms_qmmm(1) = max_nloc_qmmm + max_nghost_qmmm;
    m_natoms_qmmm(2) = max_nloc_qmmm;
    for (int ii = 3; ii < m_natoms_qmmm.size(); ++ii) {
      m_natoms_qmmm(ii) = 0;
    }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                               \
  REGISTER_KERNEL_BUILDER(Name("DprcPairwiseIdx").Device(DEVICE_CPU), \
                          PairwiseIdxOp<CPUDevice>);
REGISTER_CPU();
