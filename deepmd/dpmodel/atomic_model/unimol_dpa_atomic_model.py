# SPDX-License-Identifier: LGPL-3.0-or-later
"""Atomic model for Uni-Mol pretraining on a DPA backbone."""

from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.fitting.unimol_dpa_pretrain import (
    UniMolDPAPretrainFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPUniMolDPAAtomicModel(DPAtomicModel):
    r"""Uni-Mol's objective wired onto a DPA backbone.

    The standard path hands a descriptor's five-tuple to a fitting. Two of the
    things these heads read are not in it: the equivariant state the coordinate
    head projects, which the descriptor otherwise discards, and the neighbour
    list the distance head uses to decide which pairs it covers. This model
    overrides one method to fetch both and hand them over. Nothing else about
    the atomic model changes.
    """

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: list[str], **kwargs: Any
    ) -> None:
        if not hasattr(descriptor, "call_with_latent"):
            raise TypeError(
                "the unimol DPA objective needs a backbone that exposes its "
                "equivariant state, which is what the coordinate head reads; "
                f"{type(descriptor).__name__} does not have call_with_latent"
            )
        if not isinstance(fitting, UniMolDPAPretrainFitting):
            raise TypeError(
                "DPUniMolDPAAtomicModel needs the unimol_dpa_pretrain fitting"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def forward_atomic(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> dict[str, Array]:
        """Run the backbone and its heads.

        Parameters
        ----------
        extended_coord
            nf x (nall x 3) coordinates.
        extended_atype
            nf x nall element types, already clamped to be non-negative.
        nlist
            nf x nloc x nnei neighbour list, which also decides which pairs the
            distance head covers.
        mapping
            Extended-to-local mapping, passed through to the backbone.
        fparam, aparam, comm_dict, charge_spin
            Unused by this model.

        Returns
        -------
        dict
            The head outputs, plus the mask saying which pairs are covered.
        """
        del fparam, aparam, comm_dict, charge_spin
        node_ebd, latent = self.descriptor.call_with_latent(
            extended_coord, extended_atype, nlist, mapping=mapping
        )
        return self.fitting_net.call_atoms(node_ebd, latent, nlist)

    def apply_out_stat(self, ret: dict[str, Array], atype: Array) -> dict[str, Array]:
        """Return the head outputs untouched.

        Self-supervised targets carry no per-element bias to add back: the
        element head predicts a distribution, and the coordinate and distance
        heads predict geometry the data already fixes.
        """
        del atype
        return ret
