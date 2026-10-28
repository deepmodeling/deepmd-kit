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
// backward and the fused canonical energy-gradient, also evaluate an
// analytical pair potential when `pair_table` is not empty. Its energy and
// radial slope share their exponentials and need no saved state, so the edge
// scan that emits the edge gradient produces both: it adds
// `pair_seed_i * dV/dr / 2` to the radial cotangent of every edge of node `i`
// and accumulates `V / 2` into the node energy (`pair_energy`, or the
// `energy` of the fused operator, whose `seed` is the pair seed).
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
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
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
      "Tensor table, Tensor pair_film, Tensor pair_mixing, "
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
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
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
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
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
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, Tensor contact_radius, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, float f_inner, float f_outer) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.def(
      "dpa4c_canonical_compress_energy_gradient(Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, Tensor contact_radius, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, float f_inner, float f_outer, "
      "Tensor[] ws, Tensor[] bs, int[] resnets, "
      "Tensor w_head, Tensor b_head, Tensor bias_atom_e, int act, "
      "Tensor seed, int tile, Tensor pair_table) "
      "-> (Tensor energy, Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
}
