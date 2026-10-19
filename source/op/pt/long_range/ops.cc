// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Operator schema of the latent-charge long-range direct sum.
//
// The schema is declared here, unconditionally, while the CUDA kernel
// registers itself only against a CUDA-enabled PyTorch (direct_sum.cu), so
// that the Python front end can tell "library not loaded" from "this device
// has no kernel".
//
// For the nodes of one frame, the operator takes the raw charges c_i of the
// charge network in C float32 channels, the nodes that carry them (w_i = 1,
// and 0 elsewhere) and float64 positions r_i. The pairs carry the charges
// q_i = w_i c_i; when `channel` is a channel index rather than -1, that
// channel is shifted uniformly over the charged nodes of every frame so that
// it sums to the total charge Q_f of the frame,
//
//   q_i^ch = w_i (c_i^ch - s_f),  s_f = (sum_{j in f} w_j c_j^ch - Q_f) /
//   max(n_f, 1),
//
// with n_f charged nodes in f. With the potential of the charges
//
//   phi_i = prefactor * sum_{j != i} q_j K(r_ij),
//
// the operator returns
//
//   atom_energy_i     = q_i . phi_i / 2                        (float64)
//   charge_gradient_i = dE / dc_i = w_i (phi_i - [c = ch] m_f), m_f the
//                       mean of phi^ch over the charged nodes of f
//   force_i           = -prefactor * sum_{j != i} (q_i . q_j) K'(r_ij) / r_ij
//                       * d_ij
//   W_i               = -prefactor / 2 * sum_{j != i} (q_i . q_j)
//                       K'(r_ij) / r_ij * d_ij (x) d_ij
//   W_f               = sum_{i in f} W_i
//
// in float32 unless noted, with d_ij = r_i - r_j and the sums running over
// the nodes of the frame of i. The frames occupy the leading
// sum(n_node_per_frame) <= N nodes, which the caller ensures; the nodes past
// them are padding and join none. The frame energy is the sum of the atomic
// energies. `kernel` selects K: 0 is
// erf(alpha r) / r with `coefficient` holding alpha; 1 is a sum of Gaussians
// a_k 2^(-c_k r^2), c_k = log2(e) / (2 sigma_k^2), with `coefficient` holding
// the amplitudes a_k and then the rates c_k. `total_charge` holds Q_f in
// float64 and is read only with a constrained channel. The force and the
// virial hold the charges fixed; the response of the charges is the
// caller's, through the charge gradient.
//
// `communicator` is an int64 CPU tensor of one element holding the address of
// the caller's MPI_Comm handle, or zero. A handle of several ranks decomposes
// one frame into domains, one per rank: the charged nodes of every rank are
// the charged nodes of the frame, and the sums run over all of them. The
// outputs of a rank are those of its charged nodes, and W_f is their share of
// the virial of the frame; the other nodes of the rank carry no charge.

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def(
      "latent_charge_direct_sum(Tensor coord, Tensor charge, "
      "Tensor node_mask, Tensor n_node_per_frame, Tensor total_charge, "
      "Tensor coefficient, Tensor communicator, int kernel, int channel, "
      "float prefactor) -> "
      "(Tensor atom_energy, Tensor charge_gradient, Tensor force, "
      "Tensor atom_virial, Tensor virial)");
}

// The operator serves inference, whose caller differentiates nothing through
// it; its outputs carry no gradient.
TORCH_LIBRARY_IMPL(deepmd, Autograd, library) {
  library.impl("latent_charge_direct_sum",
               torch::CppFunction::makeFallthrough());
}
