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

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "dpa4c_graph_compress(Tensor edge_vec, Tensor edge_index, "
      "Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "bool canonical, int lmax, float table_stride, float table_max, "
      "float rcut, float eps, float degree_floor) "
      "-> (Tensor descriptor, Tensor state)");
  library.def(
      "dpa4c_graph_compress_backward(Tensor descriptor_gradient, "
      "Tensor state, Tensor edge_vec, Tensor edge_index, Tensor edge_mask, "
      "Tensor destination_order, Tensor destination_row_ptr, Tensor atype, "
      "Tensor table, Tensor pair_film, Tensor pair_mixing, "
      "Tensor type_embedding, Tensor readout_matrices, Tensor coupling_meta, "
      "Tensor coupling_entry, Tensor coupling_value, Tensor output_mean, "
      "Tensor output_inv_std, Tensor spin, Tensor spin_pair, "
      "Tensor spin_type, bool canonical, int lmax, float table_stride, "
      "float table_max, float rcut, float eps, float degree_floor) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.def(
      "dpa4c_canonical_compress(Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor) -> (Tensor descriptor, Tensor state)");
  library.def(
      "dpa4c_canonical_compress_backward(Tensor descriptor_gradient, "
      "Tensor state, Tensor edge_vec, Tensor source, "
      "Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.def(
      "dpa4c_canonical_compress_backward_inplace("
      "Tensor descriptor_gradient, Tensor(a!) state, Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor) "
      "-> (Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
  library.def(
      "dpa4c_canonical_compress_energy_gradient(Tensor edge_vec, "
      "Tensor source, Tensor destination_row_ptr, Tensor atype, Tensor table, "
      "Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, "
      "Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, "
      "Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, "
      "Tensor spin, Tensor spin_pair, Tensor spin_type, "
      "int lmax, float table_stride, float table_max, float rcut, float eps, "
      "float degree_floor, Tensor[] ws, Tensor[] bs, int[] resnets, "
      "Tensor w_head, Tensor b_head, Tensor bias_atom_e, int act, "
      "Tensor seed, int tile) "
      "-> (Tensor energy, Tensor edge_gradient, Tensor spin_gradient, "
      "Tensor edge_spin_gradient)");
}
