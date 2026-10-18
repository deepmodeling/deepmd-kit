// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Entry points of the fused graph-lower operator suite, shared so that the
// fused energy-force operators -- that of DPA1 (dpa1_graph_energy_force.cu) and
// the compressed DPA4C one (dpa4c/graph_compress.cu) -- drive the descriptor,
// fitting and force passes directly rather than through the dispatcher. The
// definitions live in dpa1_graph_descriptor.cu, graph_fitting.cu and
// edge_force_virial.cu; the operator schemas registered from those files are
// the public interface. The compressed DPA1 descriptor (dpa1_graph_compress.cu)
// is not part of an end-to-end operator, so its entries stay private to that
// translation unit.

#pragma once

#include <cuda_runtime_api.h>
#include <torch/types.h>

#include <optional>
#include <tuple>
#include <vector>

#include "fitting_plan.h"

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

// Evaluate the fitting network over one contiguous run of nodes. Every
// full-width pointer is already indexed from the run's first node, so the same
// code serves the whole node axis and a single tile of it. ``saved`` and
// ``activation`` are sized for the run rather than for the system. The
// kernels address the hidden rows of ``saved``, ``activation`` and, in the
// backward, ``dh`` and ``dh_next`` in aligned float4 quads, so these buffers
// are 16-byte aligned, as every torch allocation is; ``x`` and ``d_x`` may lie
// at any float boundary. The weights, biases and head weight are contiguous
// 16-byte aligned fp32 tensors on the current device (check_weights).
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

// Whether the fused fitting kernel serves a run of ``run_nodes`` nodes: an
// input of at least one feature, one to eight hidden layers of one width it is
// compiled for, every layer after the first square, a kernel image compiled for
// compute_80 or later, and 32-node panels that either split over clusters of
// CTAs the device keeps resident at once or, one CTA each, cover half of the
// multiprocessors (three quarters at a width of 256). At a width of 384 it
// serves runs of at most one panel per multiprocessor. The
// ``DP_FITTING_FUSED`` override (launch_overrides.h) leaves every run to the
// per-layer kernels.
bool fitting_fused_serves(const FittingLayerPlan& plan,
                          const std::vector<torch::Tensor>& ws,
                          long run_nodes);

// Floats of the transposed weights the fused fitting kernel streams in its
// backward, and their layout from the weights: one (width, din rounded up to
// four) row-major sheet per layer, zero past din. Both serve a network for
// which fitting_fused_serves holds; the transpose checks that the network has
// one to eight layers of one hidden width and at least one input feature.
long fitting_fused_transposed_floats(const std::vector<torch::Tensor>& ws);
void fitting_fused_transpose(cudaStream_t stream,
                             const FittingLayerPlan& plan,
                             const std::vector<torch::Tensor>& ws,
                             float* wt);

// Energy and input cotangent of one run of nodes in a single launch: the
// forward, the head and the backward from the energy cotangent ``d_e``. The
// results are bitwise those of fitting_forward_range followed by
// fitting_backward_range. ``saved`` holds n_layer sheets of (run_nodes,
// width): the pre-activations of every layer but the last, then the sheet
// through which the CTAs of a cluster exchange their column slices; like
// ``wt``, it is 16-byte aligned. ``wt`` holds the sheets that
// fitting_fused_transpose writes for the same weights, and
// fitting_fused_serves holds for the network and for a run of ``run_nodes``
// or more. ``d_x`` may alias ``x``.
void fitting_energy_gradient_range(cudaStream_t stream,
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
                                   const float* wt,
                                   const double* d_e,
                                   long run_nodes,
                                   float* saved,
                                   double* e,
                                   float* d_x);

// Source-attributed per-node virial of a canonical graph from stored edge
// cotangents, atom_virial[node] = sum(src=node) -g_e (x) edge_vec_e. The edge
// tensors are contiguous fp32 of shape (S, 3), the source CSR holds
// ``node_count + 1`` int64 row pointers and S uint32 positions, and every
// tensor resides on the current CUDA device. Returns (node_count, 3, 3) in
// fp32.
torch::Tensor canonical_source_virial(const torch::Tensor& edge_gradient,
                                      const torch::Tensor& edge_vec,
                                      const torch::Tensor& source_row_ptr,
                                      const torch::Tensor& source_order,
                                      long node_count);

// Exclusive prefix of the per-frame node counts as int64, so that frame ``f``
// spans ``[row_ptr[f], row_ptr[f + 1])`` of the node axis.
torch::Tensor frame_row_pointer(const torch::Tensor& n_node_per_frame);

// Per-frame sum of an fp32 per-node virial with shape (N, 9), accumulated and
// returned in fp64 with shape (nf, 3, 3).
torch::Tensor frame_virial_sum(const torch::Tensor& node_virial,
                               const torch::Tensor& n_node_per_frame);

// Per-frame total of a node scalar with shape (N, 1), with shape (nf, 1).
torch::Tensor frame_scalar_sum(torch::Tensor node_scalar,
                               torch::Tensor n_node_per_frame);

// Scatter dE/d(edge_vec) into per-node force, per-frame virial and (optional)
// per-node virial. A rank-two ``edge_spin_gradient`` adds the per-source total
// of the magnetic cotangent, which shares the source grouping the force
// reduction already walks; a rank-one empty tensor denotes its absence.
// Returns (force (N, 3), atom_virial (N, 3, 3) or empty, virial (nf, 3, 3),
// magnetic_force (N, 3) or a rank-one empty sentinel).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
edge_force_virial(torch::Tensor g_e,
                  torch::Tensor edge_vec,
                  torch::Tensor edge_index,
                  torch::Tensor edge_mask,
                  torch::Tensor destination_order,
                  torch::Tensor destination_row_ptr,
                  torch::Tensor source_order,
                  torch::Tensor source_row_ptr,
                  torch::Tensor n_node_per_frame,
                  torch::Tensor edge_spin_gradient,
                  c10::SymInt node_capacity,
                  bool want_atom_virial);
