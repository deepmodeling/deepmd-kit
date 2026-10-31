// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Operator schemas of the compressed degree-wise DPA4C descriptor.
//
// The schemas are declared here, unconditionally, while each device
// registers its own kernels: the CUDA half compiles only against a
// CUDA-enabled PyTorch (graph_compress.cu), the CPU half always
// (graph_compress_cpu.cc). Declaring a schema beside one of the two would
// make the operator disappear entirely whenever that half is absent, and the
// Python front end could no longer distinguish "library not loaded" from
// "this device has no kernel".
//
// Two schema families serve two graph forms. The generic one accepts a
// masked NeighborGraph in arbitrary edge order and takes the destination
// permutation alongside the row pointers. The canonical one is the compact
// deployment ABI: destination-major payload, identity permutation, no mask,
// and source indices only.
//
// The two operators that close an energy-force evaluation, the generic
// backward and the fused canonical energy-force, also evaluate an analytical
// pair potential when `pair_table` is not empty. Its energy and radial slope
// share their exponentials and need no saved state, so the edge scan that
// differentiates the descriptor produces both: it adds
// `pair_seed_i * dV/dr / 2` to the radial cotangent of every edge of node `i`
// and accumulates `V / 2` into the node energy (`pair_energy`, or the node
// energy of the fused operator, whose pair seed is its ownership mask).
//
// The fused canonical operator derives the ownership mask from `n_node` and
// `n_local` -- a node contributes energy when its type is real and it is among
// the leading `n_local` nodes of its frame -- and returns the frame energy, the
// masked node energy, the force, the frame virial, the per-atom virial, the
// magnetic force of a spin-conditioned descriptor (empty otherwise) and the
// mask itself. It computes the per-atom virial when it receives the source
// CSR; a two-entry placeholder in place of `source_row_ptr` and
// `source_order` declines it, which leaves it zero.
//
// The fused long-range operator evaluates a model whose atomic energy adds a
// latent-charge long-range module with a charge network of one channel. It
// takes the arguments of the fused operator up to `tile`, the network that
// predicts the charges from the descriptor (`charge_ws` to `charge_act`, in
// the layout of the energy fitting, with the per-type bias `charge_bias`),
// the inputs of deepmd::latent_charge_sum (`coord` to `spacing`, see
// source/op/pt/long_range/ops.cc), whose charged nodes are the owned real
// nodes, and the fitting routes. It returns the outputs of the fused operator
// but the magnetic force: the descriptor carries no spin condition and the
// model no pair potential.
//
// Every schema carries, next to the radial table, the low-rank table and the
// expansion that a single-type scan reads instead of the scalar-only channels
// (empty together when the descriptor admits none; see `low_rank_basis`).
//
// Every schema carries the zone-bridging window as the two dimensionless
// fractions `f_inner` and `f_outer` of a pair's own length scale, together
// with the per-element radii `contact_radius` whose pairwise sums are those
// length scales in Å. Equal fractions denote a descriptor without a window.

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "dpa4c_graph_compress(Tensor edge_vec, Tensor edge_index, "
      "Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor basis_table, Tensor basis_matrix, Tensor pair_film, "
      "Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, Tensor contact_radius, "
      "bool canonical, int lmax, float table_stride, float table_max, "
      "float rcut, float eps, float degree_floor, float f_inner, "
      "float f_outer) -> (Tensor descriptor, Tensor state)");
  library.def(
      "dpa4c_graph_compress_backward(Tensor descriptor_gradient, "
      "Tensor state, Tensor edge_vec, Tensor edge_index, Tensor edge_mask, "
      "Tensor destination_order, Tensor destination_row_ptr, Tensor atype, "
      "Tensor table, Tensor basis_table, Tensor basis_matrix, "
      "Tensor pair_film, Tensor pair_mixing, "
      "Tensor type_embedding, Tensor readout_matrices, Tensor coupling_meta, "
      "Tensor coupling_entry, Tensor coupling_value, Tensor output_mean, "
      "Tensor output_inv_std, Tensor spin, Tensor spin_pair, "
      "Tensor spin_type, Tensor contact_radius, bool canonical, int lmax, "
      "float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, float f_inner, float f_outer, Tensor pair_table, "
      "Tensor pair_seed) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient, Tensor pair_energy)");
  library.def(
      "dpa4c_canonical_compress(Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor basis_table, Tensor basis_matrix, Tensor pair_film, "
      "Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, Tensor contact_radius, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, float f_inner, float f_outer) "
      "-> (Tensor descriptor, Tensor state)");
  library.def(
      "dpa4c_canonical_compress_backward(Tensor descriptor_gradient, "
      "Tensor state, Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor basis_table, Tensor basis_matrix, Tensor pair_film, "
      "Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, Tensor contact_radius, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, float f_inner, float f_outer) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.def(
      "dpa4c_canonical_compress_backward_inplace("
      "Tensor descriptor_gradient, Tensor(a!) state, Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor basis_table, Tensor basis_matrix, Tensor pair_film, "
      "Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, Tensor contact_radius, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, float f_inner, float f_outer) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.def(
      "dpa4c_canonical_compress_energy_force(Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor source_row_ptr, "
      "Tensor source_order, Tensor n_node, Tensor n_local, Tensor atype, "
      "Tensor table, Tensor basis_table, Tensor basis_matrix, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, Tensor contact_radius, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, float f_inner, float f_outer, "
      "Tensor[] ws, Tensor[] bs, int[] resnets, "
      "Tensor w_head, Tensor b_head, Tensor bias_atom_e, int act, "
      "int tile, Tensor pair_table, int[]? fitting_routes=None) "
      "-> (Tensor energy, Tensor atom_energy, Tensor force, Tensor virial, "
      "Tensor atom_virial, Tensor force_mag, Tensor mask)");
  library.def(
      "dpa4c_canonical_compress_long_range_energy_force(Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor source_row_ptr, "
      "Tensor source_order, Tensor n_node, Tensor n_local, Tensor atype, "
      "Tensor table, Tensor basis_table, Tensor basis_matrix, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, Tensor contact_radius, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, float f_inner, float f_outer, "
      "Tensor[] ws, Tensor[] bs, int[] resnets, "
      "Tensor w_head, Tensor b_head, Tensor bias_atom_e, int act, int tile, "
      "Tensor[] charge_ws, Tensor[] charge_bs, int[] charge_resnets, "
      "Tensor charge_w_head, Tensor charge_b_head, Tensor charge_bias, "
      "int charge_act, Tensor coord, Tensor total_charge, Tensor? box, "
      "Tensor coefficient, Tensor communicator, int kernel, int channel, "
      "float prefactor, int order, float spacing, "
      "int[]? fitting_routes=None, int[]? charge_fitting_routes=None) "
      "-> (Tensor energy, Tensor atom_energy, Tensor force, Tensor virial, "
      "Tensor atom_virial, Tensor mask)");
}
