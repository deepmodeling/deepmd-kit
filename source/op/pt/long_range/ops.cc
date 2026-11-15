// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Operator schema of the latent-charge long-range energy.
//
// The schema is declared here, unconditionally, while the CUDA kernel
// registers itself only against a CUDA-enabled PyTorch (latent_charge_sum.cu),
// so that the Python front end can tell "library not loaded" from "this device
// has no kernel".
//
// For the nodes of the frames, the operator takes the raw charges c_i of the
// charge network in C float32 channels, the nodes that carry them (w_i = 1,
// and 0 elsewhere) and float64 positions r_i. The interactions carry the
// charges q_i = w_i c_i; when `channel` is a channel index rather than -1,
// that channel is shifted uniformly over the charged nodes of every frame so
// that it sums to the total charge Q_f of the frame,
//
//   q_i^ch = w_i (c_i^ch - s_f),  s_f = (sum_{j in f} w_j c_j^ch - Q_f) /
//   max(n_f, 1),
//
// with n_f charged nodes in f. Without `box`, every frame is a complete
// non-periodic system, in which the potential of the charges is the direct
// sum
//
//   phi_i = prefactor * sum_{j != i} q_j K(r_ij);
//
// with `box`, whose row f holds the row-major cell of frame f, the rows of the
// cell being its lattice vectors, phi_i sums over the images of the cell, by
// the smooth particle-mesh Ewald method (mesh_sum.cu), less the potential
// prefactor * K(0) q_i of every charge at itself. An all-zero row marks a
// non-periodic frame, which a call does not mix with periodic ones. Either way
// the operator returns
//
//   atom_energy_i     = q_i . phi_i / 2                        (float64)
//   charge_gradient_i = dE / dc_i = w_i (phi_i - [c = ch] m_f), m_f the
//                       mean of phi^ch over the charged nodes of f
//   force_i           = -dE / dr_i at fixed charges
//   atom_virial_i     (9 components, row-major)
//   virial_f          = sum_{i in f} atom_virial_i
//
// in float32 unless noted. In a non-periodic frame, with d_ij = r_i - r_j,
//
//   force_i  = -prefactor * sum_{j != i} (q_i . q_j) K'(r_ij) / r_ij * d_ij,
//   W_i      = -prefactor / 2 * sum_{j != i} (q_i . q_j) K'(r_ij) / r_ij
//              * d_ij (x) d_ij;
//
// in a periodic frame the virial of the frame is that of the strain of the
// cell, W_f = -dE / d(strain), and W_i = F_i (x) r_i + (W_f - sum_{j in f}
// F_j (x) r_j) / n_f on the charged nodes, the force index first, so that the
// charged nodes of the frame sum to W_f.
//
// The frames occupy the leading sum(n_node_per_frame) <= N nodes, which the
// caller ensures; the nodes past them are padding and join none. The frame
// energy is the sum of the atomic energies.
//
// `kernel` selects K: 0 is erf(alpha r) / r with `coefficient` holding alpha;
// 1 is a sum of Gaussians a_k 2^(-c_k r^2), c_k = log2(e) / (2 sigma_k^2),
// with `coefficient` holding the amplitudes a_k and then the rates c_k.
// `coefficient` is float64. `total_charge` holds Q_f in float64 and is read
// only with a constrained channel. `box` is a float64 CPU tensor of shape
// (F, 9). The mesh of a periodic frame holds along every lattice vector a_d
// the smallest even number of points without a prime factor above 7 of at
// least ceil(|a_d| / spacing), and spreads the charges with B-splines of
// order `order`. The operator sizes the meshes from `box` on the host; the
// device reads pinned cells in place, while the stream runs the call, and a
// copy of pageable ones. A CUDA graph that captures a call reads the cells
// again at every replay, so it needs `box` in pinned memory and evaluates the
// cells there at replay, provided their meshes keep the sizes of the capture.
// It captures only meshes whose size, with the channel count and the order,
// an eager call has evaluated: an eager call forms the cuFFT plans and the
// assignment factors of a mesh size, which the operator keeps for the sizes
// used last and for as long as a graph holds them. The caller keeps pinned
// cells alive and unchanged until the stream has passed a call that reads
// them. The force and the virial hold the charges fixed; the response of the
// charges is the caller's, through the charge gradient.
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
      "latent_charge_sum(Tensor coord, Tensor charge, Tensor node_mask, "
      "Tensor n_node_per_frame, Tensor total_charge, Tensor? box, "
      "Tensor coefficient, Tensor communicator, int kernel, int channel, "
      "float prefactor, int order, float spacing) -> "
      "(Tensor atom_energy, Tensor charge_gradient, Tensor force, "
      "Tensor atom_virial, Tensor virial)");
}

// The operator serves inference, whose caller differentiates nothing through
// it; its outputs carry no gradient.
TORCH_LIBRARY_IMPL(deepmd, Autograd, library) {
  library.impl("latent_charge_sum", torch::CppFunction::makeFallthrough());
}
