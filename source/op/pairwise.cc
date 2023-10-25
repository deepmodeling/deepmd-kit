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

REGISTER_OP("ConvertForwardMap")
    .Input("sub_forward_map: int32")
    .Input("sub_natoms: int32")
    .Input("natoms: int32")
    .Output("forward_map: int32")
    .Output("backward_map: int32")
    .Output("new_natoms: int32")
    .Output("mesh: int32");

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
      nframes_qmmm.push_back(nall > 0 ? backward_qmmm_map.size() / nall : 0);
    }
    int max_nloc_qm = 1, max_nloc_qmmm = 1, max_nghost_qm = 0,
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
        // the ghost index should add the padding indexes
        if (m_backward_qm_map(ii, jj) >= nloc_qm[ii]) {
          m_backward_qm_map(ii, jj) += max_nloc_qm - nloc_qm[ii];
        }
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
          // the ghost index should add the padding indexes
          if (m_backward_qmmm_map(nn, jj) >= nloc_qmmm[ii]) {
            m_backward_qmmm_map(nn, jj) += max_nloc_qmmm - nloc_qmmm[ii];
          }
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

template <typename Device>
class ConvertForwardMapOp : public OpKernel {
 public:
  explicit ConvertForwardMapOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int tmp_idx = 0;
    const Tensor& sub_forward_map_tensor = context->input(tmp_idx++);
    const Tensor& sub_natoms_tensor = context->input(tmp_idx++);
    const Tensor& natoms_tensor = context->input(tmp_idx++);

    // set size of the sample
    OP_REQUIRES(context, (sub_forward_map_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of idxs should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1),
                errors::InvalidArgument("Dim of natoms should be 1"));

    auto sub_forward_map = sub_forward_map_tensor.matrix<int>();
    int sub_nframes = sub_forward_map_tensor.shape().dim_size(0);
    auto sub_natoms = sub_natoms_tensor.vec<int>();
    auto natoms = natoms_tensor.vec<int>();
    int sub_nloc = sub_natoms(0);
    int sub_nall = sub_natoms(1);
    int nloc = natoms(0);
    int nall = natoms(1);

    // merge multiple sub-frames into one frame
    // firstly, we need to get the nloc and nghost size to allocate
    int new_nloc = 0, new_nghost = 0;

    for (int ii = 0; ii < sub_nframes; ++ii) {
      for (int jj = 0; jj < sub_nloc; ++jj) {
        if (sub_forward_map(ii, jj) != -1) {
          new_nloc++;
        }
      }
      for (int jj = sub_nloc; jj < sub_nall; ++jj) {
        if (sub_forward_map(ii, jj) != -1) {
          new_nghost++;
        }
      }
    }
    if (new_nloc == 0) {
      new_nloc = 1;
    }
    int new_nall = new_nloc + new_nghost;

    // Create an output tensor
    TensorShape forward_map_shape;
    forward_map_shape.AddDim(1);
    forward_map_shape.AddDim(new_nall);
    TensorShape backward_map_shape;
    // since the atom index can not be repeated, we still need
    // to split to multiple frames
    backward_map_shape.AddDim(sub_nframes);
    backward_map_shape.AddDim(nall);
    TensorShape new_natoms_shape;
    new_natoms_shape.AddDim(natoms_tensor.shape().dim_size(0));

    Tensor* forward_map_tensor = NULL;
    Tensor* backward_map_tensor = NULL;
    Tensor* new_natoms_tensor = NULL;
    tmp_idx = 0;
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, forward_map_shape,
                                            &forward_map_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, backward_map_shape,
                                            &backward_map_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(tmp_idx++, new_natoms_shape,
                                            &new_natoms_tensor));

    auto forward_map = forward_map_tensor->matrix<int>();
    auto backward_map = backward_map_tensor->matrix<int>();
    auto new_natoms = new_natoms_tensor->vec<int>();

    // fill -1 in backward_map_tensor
    for (int ii = 0; ii < sub_nframes; ++ii) {
      for (int jj = 0; jj < nall; ++jj) {
        backward_map(ii, jj) = -1;
      }
    }

    std::vector<int> start_kk(sub_nframes),
        end_kk(sub_nframes);  // current forward map index
    int kk = 0;
    // assume nlist to contain all atoms; it should not be a problem for small
    // residues
    std::vector<std::vector<int>> jlist(new_nloc);
    for (int ii = 0; ii < sub_nframes; ++ii) {
      start_kk[ii] = kk;
      for (int jj = 0; jj < sub_nloc; ++jj) {
        if (sub_forward_map(ii, jj) != -1) {
          forward_map(0, kk) = sub_forward_map(ii, jj);
          backward_map(ii, sub_forward_map(ii, jj)) = kk;
          kk++;
        }
      }
      end_kk[ii] = kk;
      // add neighbors to each other
      for (int mm = start_kk[ii]; mm < end_kk[ii]; ++mm) {
        for (int nn = start_kk[ii]; nn < end_kk[ii]; ++nn) {
          if (mm != nn) {
            jlist[mm].push_back(nn);
          }
        }
      }
    }
    for (int ii = 0; ii < sub_nframes; ++ii) {
      int start_ghost_kk = kk;
      for (int jj = sub_nloc; jj < sub_nall; ++jj) {
        if (sub_forward_map(ii, jj) != -1) {
          forward_map(0, kk) = sub_forward_map(ii, jj);
          backward_map(ii, sub_forward_map(ii, jj)) = kk;
          kk++;
        }
      }
      int end_ghost_kk = kk;
      // add ghost neighbors to real atoms
      for (int mm = start_kk[ii]; mm < end_kk[ii]; ++mm) {
        for (int nn = start_ghost_kk; nn < end_ghost_kk; ++nn) {
          jlist[mm].push_back(nn);
        }
      }
    }

    // natoms
    new_natoms(0) = new_nloc;
    new_natoms(1) = new_nall;
    new_natoms(2) = new_nloc;
    for (int ii = 3; ii < new_natoms.size(); ++ii) {
      new_natoms(ii) = 0;
    }

    // mesh:
    //   first element: nloc (a number)
    //   2~16: empty (to distinguish from other mesh)
    //   ilist: nloc
    //   numneigh: nloc
    //   jlist: sum(numneigh)

    // calculate numneigh
    std::vector<int> numneigh(new_nloc);
    for (int ii = 0; ii < new_nloc; ++ii) {
      numneigh[ii] = jlist[ii].size();
    }
    int size_mesh =
        std::accumulate(numneigh.begin(), numneigh.end(), 2 * new_nloc + 16);

    TensorShape mesh_shape;
    mesh_shape.AddDim(size_mesh);
    Tensor* mesh_tensor = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(tmp_idx++, mesh_shape, &mesh_tensor));
    auto mesh = mesh_tensor->vec<int>();
    mesh(0) = new_nloc;
    for (int ii = 1; ii < 16; ++ii) {
      mesh(ii) = 0;
    }
    for (int ii = 0; ii < new_nloc; ++ii) {
      mesh(ii + 16) = ii;
    }
    for (int ii = 0; ii < new_nloc; ++ii) {
      mesh(ii + 16 + new_nloc) = numneigh[ii];
    }
    kk = 0;
    for (int ii = 0; ii < new_nloc; ++ii) {
      for (int jj = 0; jj < numneigh[ii]; ++jj) {
        mesh(16 + 2 * new_nloc + kk) = jlist[ii][jj];
        kk++;
      }
    }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                 \
  REGISTER_KERNEL_BUILDER(Name("DprcPairwiseIdx").Device(DEVICE_CPU),   \
                          PairwiseIdxOp<CPUDevice>);                    \
  REGISTER_KERNEL_BUILDER(Name("ConvertForwardMap").Device(DEVICE_CPU), \
                          ConvertForwardMapOp<CPUDevice>);
REGISTER_CPU();
