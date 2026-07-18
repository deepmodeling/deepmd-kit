// SPDX-License-Identifier: LGPL-3.0-or-later
//
// End-to-end fused energy-force operator for the DPA1 (``se_atten``) graph
// lower: one opaque graph node that consumes the neighbor edge stream and
// returns the per-frame energy, per-atom energy, force, virial and (optional)
// atom virial. It drives the descriptor and energy-fitting mega kernels and
// their analytic backwards in sequence, computing the force from the reduced
// energy internally (dE_redu/d(atom_energy) == 1), so the graph carries no
// autograd tape and no per-output-component grad loop.
//
// It is numerically identical to the separate-operator path (the descriptor
// and fitting forwards with the force assembled from autograd.grad): the same
// operator backwards evaluate the same fp32 arithmetic in the same order. The
// fusion removes the autograd machinery and the inter-operator array glue, and
// is selected at freeze time by DP_CUDA_INFER >= 2.

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <optional>
#include <tuple>
#include <vector>

#include "graph_ops.h"

// Returns (energy (nf, 1) fp64, atom_energy (N, 1) fp64, force (N, 3),
// virial (nf, 3, 3), atom_virial (N, 3, 3) or empty).
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
dpa1_graph_energy_force(torch::Tensor edge_vec,
                        torch::Tensor edge_index,
                        torch::Tensor edge_mask,
                        torch::Tensor destination_order,
                        torch::Tensor destination_row_ptr,
                        torch::Tensor source_order,
                        torch::Tensor source_row_ptr,
                        torch::Tensor atype,
                        torch::Tensor n_node,
                        torch::Tensor ownership,
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
                        int64_t smooth,
                        int64_t axis,
                        int64_t resnet2,
                        int64_t resnet3,
                        double rcut,
                        double rcut_smth,
                        double protection,
                        double nnei,
                        std::vector<torch::Tensor> fit_ws,
                        std::vector<torch::Tensor> fit_bs,
                        std::vector<torch::Tensor> fit_idts,
                        std::vector<int64_t> fit_resnets,
                        torch::Tensor w_head,
                        torch::Tensor b_head,
                        torch::Tensor bias_atom_e,
                        int64_t fit_act,
                        c10::SymInt node_capacity,
                        bool do_atomic_virial) {
  // The descriptor / fitting kernels compute in the weight precision (fp32).
  // Cast the fp64 edge stream once and thread it through the forward, backward
  // and force / virial scatter: each sub-operator's internal ``to(float)`` is
  // then a no-op, and the backward returns the ``edge_vec`` gradient in that
  // same precision, so the force scatter needs no cast either. Passing the
  // fp64 leaf instead would re-cast the whole edge stream three times and
  // round-trip the gradient fp32 -> fp64 -> fp32. The energy reduction stays
  // fp64.
  const auto fprec = w1.scalar_type();
  const auto edge_vec_f =
      edge_vec.scalar_type() == fprec ? edge_vec : edge_vec.to(fprec);

  // === Step 1. Descriptor forward: edge stream -> (N, nd) descriptor. ===
  auto desc = dpa1_graph_descriptor(
      edge_vec_f, edge_index, edge_mask, atype, type_embedding, davg, dstd, w1,
      b1, idt1, w2, b2, idt2, w3, b3, idt3, gate_table, act, type_one_side,
      concat_tebd, /*write_rotation=*/0, smooth, axis, resnet2, resnet3, rcut,
      rcut_smth, protection, nnei);
  const torch::Tensor& grrg = std::get<0>(desc);
  const torch::Tensor& gr = std::get<2>(desc);
  const torch::Tensor& edge_order = std::get<3>(desc);
  const torch::Tensor& pair_table = std::get<4>(desc);
  const torch::Tensor& pre2_saved = std::get<5>(desc);
  const torch::Tensor& g_saved = std::get<6>(desc);

  // === Step 2. Fitting forward: descriptor -> per-atom energy. ===
  auto fit = graph_fitting(grrg, atype, fit_ws, fit_bs, fit_idts, fit_resnets,
                           w_head, b_head, bias_atom_e, fit_act);
  const torch::Tensor& atom_energy_raw = std::get<0>(fit);  // (N, 1) fp64
  const torch::Tensor& fit_saved = std::get<1>(fit);
  auto owned = ownership.reshape({-1, 1}).to(atom_energy_raw.scalar_type());
  auto energy_seed = owned;
  auto atom_energy = atom_energy_raw * owned;

  // === Step 3. Per-frame energy: segment-sum over the frame index. ===
  const int64_t nf = n_node.size(0);
  auto frame_id =
      at::repeat_interleave(at::arange(nf, n_node.options()), n_node);
  auto energy = at::zeros({nf, 1}, atom_energy.options())
                    .index_add_(0, frame_id, atom_energy);

  // === Step 4. Force = grad of the reduced energy; dE_redu/d(atom_e) == 1. ===
  std::get<0>(desc) = torch::Tensor();
  auto d_grrg = graph_fitting_backward(energy_seed, fit_saved, fit_ws,
                                       fit_resnets, w_head);
  std::get<1>(fit) = torch::Tensor();
  auto g_e = dpa1_graph_descriptor_backward(
      d_grrg, std::nullopt, gr, edge_order, pair_table, pre2_saved, g_saved,
      edge_vec_f, edge_index, edge_mask, atype, davg, dstd, w1, b1, idt1, w2,
      b2, idt2, w3, b3, idt3, gate_table, act, type_one_side, smooth, axis,
      resnet2, resnet3, rcut, rcut_smth, protection, nnei);

  // === Step 5. Scatter dE/d(edge_vec) into force / virial / atom virial. ===
  // g_e and edge_vec_f are already in the compute precision; the per-node force
  // is a short neighbor sum and the per-frame virial reduces hierarchically.
  auto fv = edge_force_virial(g_e, edge_vec_f, edge_index, edge_mask,
                              destination_order, destination_row_ptr,
                              source_order, source_row_ptr, n_node,
                              node_capacity, do_atomic_virial);
  return {energy, atom_energy, std::get<0>(fv), std::get<2>(fv),
          std::get<1>(fv)};
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def(
      "dpa1_graph_energy_force(Tensor edge_vec, Tensor edge_index, Tensor "
      "edge_mask, Tensor destination_order, Tensor destination_row_ptr, "
      "Tensor source_order, Tensor source_row_ptr, Tensor atype, Tensor "
      "n_node, Tensor ownership, Tensor type_embedding, "
      "Tensor davg, Tensor dstd, Tensor w1, Tensor b1, Tensor idt1, Tensor w2, "
      "Tensor "
      "b2, Tensor idt2, Tensor w3, Tensor b3, Tensor idt3, Tensor gate_table, "
      "int act, int type_one_side, int concat_tebd, int smooth, int axis, int "
      "resnet2, int resnet3, float rcut, float rcut_smth, float protection, "
      "float nnei, Tensor[] fit_ws, Tensor[] fit_bs, Tensor[] fit_idts, int[] "
      "fit_resnets, Tensor w_head, Tensor b_head, Tensor bias_atom_e, int "
      "fit_act, SymInt node_capacity, bool do_atomic_virial) -> (Tensor, "
      "Tensor, Tensor, Tensor, Tensor)");
  m.impl("dpa1_graph_energy_force", torch::kCUDA, &dpa1_graph_energy_force);
}
