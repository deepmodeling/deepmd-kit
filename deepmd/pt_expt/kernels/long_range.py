# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the direct-sum operator of the latent-charge long range.

The operator ``deepmd::latent_charge_direct_sum`` evaluates, for the nodes of
complete frames, the potential that the charges of every other node of the
frame create at each node and the force between the charges, without listing
the pairs: its memory is linear in the node count. The CUDA kernel is
``source/op/pt/long_range/direct_sum.cu``; the CPU kernel registered here is
the reference it is tested against.

The operator serves inference. It applies the total-charge constraint of the
module, and returns the atomic energy, its derivative with respect to the
charges the charge network predicts, and the force and virial at fixed
charges; the caller adds the response of the charges by back-propagating the
charge gradient through the charge network.

Given the MPI communicator of a system decomposed into domains, one per rank,
the CUDA kernel evaluates the charged nodes of its rank against those of
every rank; the CPU reference evaluates a single domain.
"""

import math
from functools import (
    cache,
)
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    frame_id_from_n_node,
    unordered_pair_index,
)
from deepmd.pt_expt.kernels.utils import (
    operator_available,
)
from deepmd.pt_expt.long_range.kernel import (
    LESKernel,
    SOGKernel,
)

__all__ = [
    "MAX_CHANNELS",
    "MAX_GAUSSIANS",
    "kernel_eligible",
    "kernel_operator_arguments",
    "latent_charge_direct_sum",
    "op_available",
]

#: Charge channels and Gaussians the CUDA kernel is instantiated for.
MAX_CHANNELS = 8
MAX_GAUSSIANS = 32

_LES = 0
_SOG = 1


def op_available() -> bool:
    """Whether the backend device carries the direct-sum operator.

    The CPU kernel is the reference of this module, which is registered here
    so that the answer does not depend on earlier calls.
    """
    # Importing the module loads the operator library.
    import deepmd.pt.cxx_op  # noqa: F401

    ensure_registered()
    return operator_available("latent_charge_direct_sum")


def kernel_eligible(kernel: Any) -> bool:
    """Whether the operator evaluates this interaction kernel."""
    if isinstance(kernel, LESKernel):
        return True
    return isinstance(kernel, SOGKernel) and kernel.amplitude.numel() <= MAX_GAUSSIANS


def kernel_operator_arguments(kernel: Any) -> tuple[int, torch.Tensor]:
    """Convert an interaction kernel into the code and coefficients of the operator.

    Parameters
    ----------
    kernel
        A pt_expt interaction kernel accepted by :func:`kernel_eligible`.

    Returns
    -------
    tuple[int, torch.Tensor]
        The kernel code and its float32 coefficients: ``alpha`` for the erf
        kernel; the amplitudes ``a_k`` and then the rates ``c_k = log2(e) /
        (2 sigma_k^2)`` of ``a_k 2^(-c_k r^2)`` for the sum of Gaussians.

    Raises
    ------
    ValueError
        If the operator does not evaluate the kernel.
    """
    if not kernel_eligible(kernel):
        raise ValueError(
            f"the direct-sum operator does not evaluate {type(kernel).__name__}"
        )
    if isinstance(kernel, LESKernel):
        coefficient = torch.exp(kernel.log_alpha).reshape(1)
        return _LES, coefficient.detach().to(torch.float32).contiguous()
    rate = (0.5 / math.log(2.0)) * torch.exp(-2.0 * kernel.log_width)
    coefficient = torch.cat((kernel.amplitude, rate))
    return _SOG, coefficient.detach().to(torch.float32).contiguous()


def _kernel_and_slope(
    r_sq: torch.Tensor, coefficient: torch.Tensor, kernel: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """K(r) and K'(r) / r at the squared distances ``r_sq``."""
    coefficient = coefficient.to(dtype=r_sq.dtype, device=r_sq.device)
    if kernel == _LES:
        alpha = coefficient[0]
        # Coincident nodes, which carry no weight, are evaluated at a short
        # distance, where the closed form stays finite.
        r_sq = torch.clamp(r_sq, min=1e-12 / alpha**2)
        r = torch.sqrt(r_sq)
        value = torch.erf(alpha * r) / r
        contact = (2.0 / math.sqrt(math.pi)) * alpha
        slope = (contact * torch.exp(-(alpha**2) * r_sq) - value) / r_sq
        return value, slope
    amplitude, rate = coefficient.reshape(2, -1)
    weight = torch.exp2(-r_sq[:, None] * rate)
    return weight @ amplitude, weight @ (-2.0 * math.log(2.0) * rate * amplitude)


def _reference(
    coord: torch.Tensor,
    charge: torch.Tensor,
    node_mask: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    total_charge: torch.Tensor,
    coefficient: torch.Tensor,
    communicator: torch.Tensor,
    kernel: int,
    channel: int,
    prefactor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the operator over the explicit pairs of every frame, in float64."""
    if int(communicator.reshape(-1)[0]) != 0:
        raise NotImplementedError(
            "the CPU reference of latent_charge_direct_sum evaluates a single "
            "domain; a communicator of several ranks requires the CUDA kernel"
        )
    n = coord.shape[0]
    frame_count = n_node_per_frame.shape[0]
    # Padding nodes belong to no frame and carry no charge.
    frame = frame_id_from_n_node(n_node_per_frame, n_total=n)
    real = torch.arange(n, dtype=torch.int64, device=coord.device) < torch.sum(
        n_node_per_frame
    )
    weight = (node_mask & real).to(torch.float64)
    q = charge.to(torch.float64) * weight[:, None]
    if channel >= 0:
        columns = torch.arange(q.shape[1], dtype=torch.int64, device=coord.device)
        selector = (columns == channel).to(torch.float64)
        count = coord.new_zeros(frame_count).index_add(0, frame, weight)
        count = torch.clamp(count, min=1.0)
        total = coord.new_zeros(frame_count).index_add(0, frame, q[:, channel])
        shift = (total - total_charge.to(torch.float64)) / count
        q = q - (shift[frame] * weight)[:, None] * selector
    pairs = unordered_pair_index(n_node_per_frame)
    src, dst = pairs[0], pairs[1]
    d = coord[src] - coord[dst]
    value, slope = _kernel_and_slope(torch.sum(d * d, dim=-1), coefficient, kernel)
    potential = coord.new_zeros(q.shape)
    potential = potential.index_add(0, src, q[dst] * value[:, None])
    potential = prefactor * potential.index_add(0, dst, q[src] * value[:, None])
    strength = torch.sum(q[src] * q[dst], dim=-1) * slope
    pair_force = -strength[:, None] * d
    force = torch.zeros_like(coord).index_add(0, src, pair_force)
    force = force.index_add(0, dst, -pair_force)
    half = (-0.5 * strength)[:, None, None] * d[:, :, None] * d[:, None, :]
    atom_virial = coord.new_zeros(n, 3, 3).index_add(0, src, half)
    atom_virial = prefactor * atom_virial.index_add(0, dst, half).reshape(n, 9)
    virial = coord.new_zeros(frame_count, 9).index_add(0, frame, atom_virial)
    atom_energy = 0.5 * torch.sum(q * potential, dim=-1, keepdim=True)
    gradient = potential * weight[:, None]
    if channel >= 0:
        mean = coord.new_zeros(frame_count).index_add(0, frame, gradient[:, channel])
        gradient = gradient - ((mean / count)[frame] * weight)[:, None] * selector
    return (
        atom_energy,
        gradient.to(charge.dtype),
        (prefactor * force).to(charge.dtype),
        atom_virial.to(charge.dtype),
        virial.to(charge.dtype),
    )


def _fake(
    coord: torch.Tensor,
    charge: torch.Tensor,
    node_mask: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    total_charge: torch.Tensor,
    coefficient: torch.Tensor,
    communicator: torch.Tensor,
    kernel: int,
    channel: int,
    prefactor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = coord.shape[0]
    return (
        coord.new_empty(n, 1),
        torch.empty_like(charge),
        charge.new_empty(n, 3),
        charge.new_empty(n, 9),
        charge.new_empty(n_node_per_frame.shape[0], 9),
    )


@cache
def _register() -> None:
    torch.library.register_fake("deepmd::latent_charge_direct_sum")(_fake)
    library = torch.library.Library("deepmd", "IMPL")
    library.impl("latent_charge_direct_sum", _reference, "CPU")
    # The registration lives as long as the library object.
    _register.library = library


def ensure_registered() -> None:
    """Register the fake and CPU implementations when the schema is loaded."""
    if isinstance(
        getattr(torch.ops.deepmd, "latent_charge_direct_sum", None),
        torch._ops.OpOverloadPacket,
    ):
        _register()


def latent_charge_direct_sum(
    coord: torch.Tensor,
    charge: torch.Tensor,
    node_mask: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    total_charge: torch.Tensor,
    kernel: Any,
    channel: int,
    prefactor: float,
    communicator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy, charge gradient, force and virial of the charges of complete frames.

    Parameters
    ----------
    coord
        Node positions with shape ``(N, 3)`` in Å, float64. The frames occupy
        contiguous spans of the node axis; nodes past ``sum(n_node_per_frame)``
        are padding and join no frame.
    charge
        Raw node charges of the charge network with shape ``(N, C)``,
        float32, with ``C`` at most :data:`MAX_CHANNELS`.
    node_mask
        Nodes that carry charges with shape ``(N,)``, bool; the others carry
        none.
    n_node_per_frame
        Node count of every frame with shape ``(F,)``.
    total_charge
        Total charge of every frame with shape ``(F,)``, float64, which the
        constrained channel sums to.
    kernel
        The interaction kernel, accepted by :func:`kernel_eligible`.
    channel
        The constrained channel, which is shifted uniformly over the charged
        nodes of every frame to sum to its total charge, or -1 for none.
    prefactor
        Factor of every pair term, in eV Å.
    communicator
        The address of the MPI_Comm handle of a system decomposed into
        domains, one per rank, as an int64 CPU tensor of one element, or zero,
        the default, for a single domain. Across domains the call takes one
        frame, whose charged nodes are those of every rank; the outputs are
        those of the charged nodes of this rank, the other nodes carry no
        charge, and ``virial`` is the share of this rank.

    Returns
    -------
    atom_energy
        Per-node energy with shape ``(N, 1)`` in eV, float64, each pair
        energy split half to each of its nodes.
    charge_gradient
        Derivative of the energy with respect to ``charge`` with shape
        ``(N, C)``, in eV per unit charge, through the constraint.
    force
        Force at fixed charges with shape ``(N, 3)``, in eV/Å.
    atom_virial
        Per-node virial at fixed charges with shape ``(N, 9)`` in eV, each
        pair term split half to each of its nodes.
    virial
        Per-frame virial at fixed charges with shape ``(F, 9)`` in eV.
    """
    ensure_registered()
    code, coefficient = kernel_operator_arguments(kernel)
    if communicator is None:
        communicator = torch.zeros(1, dtype=torch.int64, device="cpu")
    return torch.ops.deepmd.latent_charge_direct_sum(
        coord,
        charge,
        node_mask,
        n_node_per_frame,
        total_charge,
        coefficient.to(coord.device),
        communicator,
        code,
        int(channel),
        float(prefactor),
    )
