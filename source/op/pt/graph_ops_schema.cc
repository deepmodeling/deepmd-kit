// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Operator schemas of the descriptor-agnostic graph-lower operators.
//
// The schemas are declared here, unconditionally, while each device registers
// its own kernels: the CUDA half compiles only against a CUDA-enabled PyTorch
// (graph_fitting.cu, edge_force_virial.cu), the CPU half always
// (graph_fitting_cpu.cc, edge_force_virial_cpu.cc). Declaring a schema beside
// one of the two would make the operator disappear entirely whenever that
// half is absent, and the Python front end could no longer distinguish
// "library not loaded" from "this device has no kernel".

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "graph_fitting(Tensor x, Tensor atype, Tensor[] ws, Tensor[] bs, "
      "int[] resnets, Tensor w_head, Tensor b_head, Tensor bias_atom_e, "
      "int act) -> (Tensor e, Tensor saved)");
  library.def(
      "graph_fitting_backward(Tensor d_e, Tensor saved, Tensor[] ws, "
      "Tensor[] bs, int[] resnets, Tensor w_head, int act) -> Tensor");
  library.def(
      "graph_fitting_energy_gradient(Tensor(a!) x, Tensor atype, "
      "Tensor[] ws, Tensor[] bs, int[] resnets, Tensor w_head, "
      "Tensor b_head, Tensor bias_atom_e, int act, Tensor seed, int tile) "
      "-> Tensor");
  library.def(
      "build_graph_csr(Tensor edge_index, SymInt node_count, "
      "SymInt valid_edge_count) -> "
      "(Tensor destination_order, Tensor destination_row_ptr, "
      "Tensor source_order, Tensor source_row_ptr)");
  library.def(
      "edge_force_virial(Tensor edge_gradient, Tensor edge_vec, "
      "Tensor edge_index, Tensor edge_mask, Tensor destination_order, "
      "Tensor destination_row_ptr, Tensor source_order, Tensor source_row_ptr, "
      "Tensor n_node_per_frame, Tensor edge_spin_gradient, "
      "SymInt node_capacity, bool want_atom_virial) -> "
      "(Tensor force, Tensor atom_virial, Tensor virial, "
      "Tensor magnetic_force)");
  library.def(
      "canonical_edge_force_virial(Tensor edge_gradient, Tensor edge_vec, "
      "Tensor destination_row_ptr, Tensor source_row_ptr, "
      "Tensor source_order, Tensor n_node_per_frame, "
      "Tensor edge_spin_gradient, SymInt node_capacity, "
      "bool want_atom_virial) -> "
      "(Tensor force, Tensor atom_virial, Tensor virial, "
      "Tensor magnetic_force)");
  library.def(
      "frame_scalar_sum(Tensor node_scalar, Tensor n_node_per_frame) "
      "-> Tensor");
}

// The force and virial assembly runs downstream of the energy backward and
// carries no gradient of its own, on either device.
TORCH_LIBRARY_IMPL(deepmd, Autograd, library) {
  library.impl("edge_force_virial", torch::CppFunction::makeFallthrough());
  library.impl("canonical_edge_force_virial",
               torch::CppFunction::makeFallthrough());
  library.impl("frame_scalar_sum", torch::CppFunction::makeFallthrough());
}
