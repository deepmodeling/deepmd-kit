// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Entry points of the fused graph-lower operator suite, shared so the fused
// energy-force operator (dpa1_graph_energy_force.cu) can drive the forward and
// backward passes directly rather than through the dispatcher. The definitions
// live in dpa1_graph_descriptor.cu, graph_fitting.cu and edge_force_virial.cu;
// the operator schemas registered from those files are the public interface.
// The compressed descriptor (dpa1_graph_compress.cu) is not part of the
// end-to-end operator, so its entries stay private to that translation unit.

#pragma once

#include <torch/torch.h>

#include <optional>
#include <tuple>
#include <vector>

// DPA1 descriptor body (environment matrix, embedding MLP, moment, G^T G).
// Returns (grrg, rot_mat, gr, edge_order, pair_table, pre2_saved, g_saved);
// the last five are consumed by dpa1_graph_descriptor_backward.
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
dpa1_graph_descriptor(torch::Tensor edge_vec,
                      torch::Tensor edge_index,
                      torch::Tensor edge_mask,
                      torch::Tensor atype,
                      torch::Tensor type_embedding,
                      torch::Tensor davg,
                      torch::Tensor dstd,
                      torch::Tensor degree_gain,
                      torch::Tensor w1,
                      torch::Tensor b1,
                      torch::Tensor idt1,
                      torch::Tensor w2,
                      torch::Tensor b2,
                      torch::Tensor idt2,
                      torch::Tensor w3,
                      torch::Tensor b3,
                      torch::Tensor idt3,
                      torch::Tensor gate_table,
                      int64_t act,
                      int64_t type_one_side,
                      int64_t concat_tebd,
                      int64_t write_rotation,
                      int64_t smooth,
                      int64_t axis,
                      int64_t resnet2,
                      int64_t resnet3,
                      double rcut,
                      double rcut_smth,
                      double protection,
                      double nnei,
                      int64_t basis_dim);

// dE/d(edge_vec) from dE/d(grrg); consumes the saved tensors of the forward.
torch::Tensor dpa1_graph_descriptor_backward(
    torch::Tensor d_grrg,
    std::optional<torch::Tensor> d_rot_mat,
    torch::Tensor gr,
    torch::Tensor order,
    torch::Tensor pair_table,
    torch::Tensor pre2_saved,
    torch::Tensor g_saved,
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor atype,
    torch::Tensor davg,
    torch::Tensor dstd,
    torch::Tensor degree_gain,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor idt1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor idt2,
    torch::Tensor w3,
    torch::Tensor b3,
    torch::Tensor idt3,
    torch::Tensor gate_table,
    int64_t act,
    int64_t type_one_side,
    int64_t smooth,
    int64_t axis,
    int64_t resnet2,
    int64_t resnet3,
    double rcut,
    double rcut_smth,
    double protection,
    double nnei);

// Energy fitting MLP on the flat node axis. Returns (atom_energy (N, 1) fp64,
// saved layer pre-activations) for graph_fitting_backward.
std::tuple<torch::Tensor, torch::Tensor> graph_fitting(
    torch::Tensor x,
    torch::Tensor atype,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> bs,
    std::vector<int64_t> resnets,
    torch::Tensor w_head,
    torch::Tensor b_head,
    torch::Tensor bias_atom_e,
    int64_t act);

// dE/d(x) from dE/d(atom_energy); consumes the saved pre-activations.
torch::Tensor graph_fitting_backward(torch::Tensor d_e,
                                     torch::Tensor saved,
                                     std::vector<torch::Tensor> ws,
                                     std::vector<torch::Tensor> bs,
                                     std::vector<int64_t> resnets,
                                     torch::Tensor w_head,
                                     int64_t act);

// Layer geometry of one fitting network, shared by the operators that
// evaluate it over a run of nodes.
struct FittingLayerPlan {
  std::vector<long> offset;  //!< Prefix sum of the hidden widths.
  long width_max;            //!< Widest hidden layer.
  int n_layer;

  long saved_width() const { return offset[n_layer]; }
};

FittingLayerPlan fitting_layer_plan(const std::vector<torch::Tensor>& ws);

// Evaluate the fitting network over one contiguous run of nodes. Every
// full-width pointer is already indexed from the run's first node, so the same
// code serves the whole node axis and a single tile of it. ``saved`` and
// ``activation`` are sized for the run rather than for the system.
void fitting_forward_range(cudaStream_t stream,
                           const FittingLayerPlan& plan,
                           const float* x,
                           long input_width,
                           const long* atype,
                           const std::vector<torch::Tensor>& ws,
                           const std::vector<torch::Tensor>& bs,
                           const std::vector<int64_t>& resnets,
                           const torch::Tensor& w_head,
                           const torch::Tensor& b_head,
                           const torch::Tensor& bias_atom_e,
                           int64_t act,
                           long run_nodes,
                           float* saved,
                           float* const activation[2],
                           double* e);

// Propagate the head cotangent of one run of nodes back to the input.
void fitting_backward_range(cudaStream_t stream,
                            const FittingLayerPlan& plan,
                            const double* d_e,
                            const float* saved,
                            const std::vector<torch::Tensor>& ws,
                            const std::vector<torch::Tensor>& bs,
                            const std::vector<int64_t>& resnets,
                            const torch::Tensor& w_head,
                            int64_t act,
                            long run_nodes,
                            float* dh,
                            float* dh_next,
                            float* d_x);

// Scatter dE/d(edge_vec) into per-node force, per-frame virial and (optional)
// per-node virial. Returns (force (N, 3), atom_virial (N, 3, 3) or empty,
// virial (nf, 3, 3)).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> edge_force_virial(
    torch::Tensor g_e,
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor source_order,
    torch::Tensor source_row_ptr,
    torch::Tensor n_node_per_frame,
    c10::SymInt node_capacity,
    bool want_atom_virial);
