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
                      double nnei);

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
// saved activations/derivatives) for graph_fitting_backward.
std::tuple<torch::Tensor, torch::Tensor> graph_fitting(
    torch::Tensor x,
    torch::Tensor atype,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> bs,
    std::vector<torch::Tensor> idts,
    std::vector<int64_t> resnets,
    torch::Tensor w_head,
    torch::Tensor b_head,
    torch::Tensor bias_atom_e,
    int64_t act);

// dE/d(x) from dE/d(atom_energy); consumes the saved activations.
torch::Tensor graph_fitting_backward(torch::Tensor d_e,
                                     torch::Tensor saved,
                                     std::vector<torch::Tensor> ws,
                                     std::vector<int64_t> resnets,
                                     torch::Tensor w_head);

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
    torch::Tensor source_row_ptr,
    torch::Tensor source_order,
    torch::Tensor n_node_per_frame,
    c10::SymInt node_capacity,
    bool want_atom_virial);
