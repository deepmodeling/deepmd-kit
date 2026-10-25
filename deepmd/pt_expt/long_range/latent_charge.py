# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.dpmodel.long_range.kernel import (
    COULOMB_CONSTANT,
)
from deepmd.dpmodel.long_range.latent_charge import (
    LatentChargeLongRange as LatentChargeLongRangeDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.fitting.latent_charge import (
    LatentChargeFitting,
)

from .base_long_range import (
    BaseLongRange,
)
from .kernel import (
    LESKernel,
    SOGKernel,
)


@BaseLongRange.register("latent_charge")
@torch_module
class LatentChargeLongRange(LatentChargeLongRangeDP):
    """Energy of interacting latent charges, periodic or not (pt_expt)."""

    # The dpmodel class builds the charge network and the kernel, which the
    # assignment converts to these pt_expt modules; importing their classes
    # registers the conversions.
    charge_net: LatentChargeFitting
    kernel: LESKernel | SOGKernel

    def _pair_kernel(
        self, r_sq: torch.Tensor, src: torch.Tensor, n: int
    ) -> torch.Tensor:
        # A graph traced for a symbolic pair count plans the reduction of the
        # coefficient gradients over the pairs before it sees a batch. In
        # training the coefficients therefore reach the pairs through the node
        # axis, which turns that reduction into a scatter over the pairs and a
        # sum over the nodes, parallel in the pair count of every batch. A
        # concrete pair count is reduced directly.
        if not (self.training and isinstance(r_sq.shape[0], torch.SymInt)):
            return self.kernel(r_sq)
        coefficients = {
            name: value.expand(n, *value.shape)[src]
            for name, value in self.kernel.coefficients(r_sq).items()
        }
        return self.kernel.evaluate(r_sq, coefficients)

    def operator_eligible(self) -> bool:
        """Whether :meth:`energy_force_from_coord` evaluates this module.

        The operator evaluates the kernels, channel counts and B-spline orders
        it is instantiated for. The compact lower that calls it reads no frame
        input: the charges depend on the descriptor alone, and a constrained
        channel sums to zero or to the total charge of a folded charge state.
        """
        from deepmd.pt_expt.kernels.long_range import (
            MAX_CHANNELS,
            MAX_ORDER,
            kernel_eligible,
            op_available,
        )

        return (
            op_available()
            and kernel_eligible(self.kernel)
            and self.charge_net.dim_out <= MAX_CHANNELS
            and self.pme_order <= MAX_ORDER
            and self.charge_net.numb_fparam == 0
            and self.charge_net.numb_aparam == 0
            and not self.uses_charge_spin()
        )

    def operator_constraint(
        self, n_frames: int, like: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """The total charge of every frame and the constrained channel of the operator.

        Parameters
        ----------
        n_frames
            The number of frames.
        like
            A tensor giving the device of the total charges.

        Returns
        -------
        tuple[torch.Tensor, int]
            The total charges with shape ``(n_frames,)``, float64 and
            contiguous, and the constrained channel, or -1 without a
            constraint, in which case the total charges are zero.
        """
        total_charge = like.new_zeros(n_frames, dtype=torch.float64)
        if self.charge_constraint is None:
            return total_charge, -1
        total_charge = self._total_charge(n_frames, total_charge, None, None)
        return total_charge.contiguous(), int(self.charge_constraint["channel"])

    def energy_force_from_coord(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        node_mask: torch.Tensor,
        n_node: torch.Tensor,
        coord: torch.Tensor,
        communicator: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the module for inference from the positions of the nodes.

        The operator sums a non-periodic frame directly over its pairs, which
        it enumerates itself, so that the memory is linear in the node count,
        and a periodic frame on the mesh of its cell. It applies the
        constraint and returns the force at fixed charges and the derivative
        of the energy with respect to the charges the network predicts;
        pulled back through the charge network, that derivative is the
        derivative of the energy with respect to the descriptor, through which
        the caller adds the response of the charges to the positions.

        Parameters
        ----------
        descriptor
            Invariant node features with shape ``(N, dim_descrpt)``.
        atype
            Node types with shape ``(N,)``.
        node_mask
            Nodes carrying charges with shape ``(N,)``.
        n_node
            Per-frame node counts with shape ``(nf,)``; the frames occupy
            contiguous spans of the node axis.
        coord
            Node positions with shape ``(N, 3)`` in Å.
        communicator
            The MPI communicator of a frame decomposed into domains, one per
            rank, whose charged nodes are those of every rank, in the form of
            :func:`~deepmd.pt_expt.kernels.long_range.latent_charge_sum`;
            ``None`` for a single domain.
        box
            The cells of the frames with shape ``(nf, 9)``, float64 on the
            CPU, a zero row marking a non-periodic frame, or ``None`` when no
            frame is periodic. A module that does not serve periodic systems
            evaluates every frame as non-periodic.

        Returns
        -------
        atom_energy
            Per-node energy with shape ``(N, 1)`` in eV, float64.
        descriptor_gradient
            Derivative of the energy with respect to the descriptor through
            the charges, with the shape and precision of ``descriptor``.
        force
            Force at fixed charges with shape ``(N, 3)`` in eV/Å, float32.
        atom_virial
            Per-node virial at fixed charges with shape ``(N, 9)`` in eV,
            float32, each pair term split half to each of its nodes.
        virial
            Per-frame virial at fixed charges with shape ``(nf, 9)`` in eV,
            float32; across domains, the share of the nodes of this rank.
        """
        from deepmd.pt_expt.kernels.long_range import (
            latent_charge_sum,
        )

        leaf = descriptor.detach().requires_grad_(True)
        with torch.enable_grad():
            charge = self.charge_net.call_graph(leaf, atype)["latent_charge"]
        total_charge, channel = self.operator_constraint(n_node.shape[0], coord)
        atom_energy, charge_gradient, force, atom_virial, virial = latent_charge_sum(
            coord.to(torch.float64).contiguous(),
            charge.detach().to(torch.float32).contiguous(),
            node_mask.contiguous(),
            n_node,
            total_charge,
            self.kernel,
            channel,
            COULOMB_CONSTANT,
            communicator,
            # A module that does not serve periodic systems sums every frame
            # directly.
            box if self.periodic else None,
            self.pme_order,
            self.mesh_spacing() if self.periodic else 1.0,
        )
        (descriptor_gradient,) = torch.autograd.grad(
            charge, leaf, charge_gradient.to(charge.dtype)
        )
        return (
            atom_energy,
            descriptor_gradient.to(descriptor.dtype),
            force,
            atom_virial,
            virial,
        )
