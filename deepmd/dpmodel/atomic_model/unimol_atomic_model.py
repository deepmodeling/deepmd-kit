# SPDX-License-Identifier: LGPL-3.0-or-later
"""Atomic model for Uni-Mol v1 self-supervised pretraining."""

from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.descriptor.unimol import (
    DescrptUniMol,
)
from deepmd.dpmodel.fitting.unimol_pretrain import (
    UniMolPretrainFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPUniMolAtomicModel(DPAtomicModel):
    r"""Uni-Mol pretraining, wired at token resolution.

    The standard path hands a descriptor's five-tuple to a fitting, which
    cannot carry the two virtual tokens, the pair channel or the norm
    regularisers that these heads read. This model therefore overrides one
    method to route the backbone's token-resolution output straight into the
    heads. Nothing else about the atomic model changes.
    """

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: list[str], **kwargs: Any
    ) -> None:
        if not isinstance(descriptor, DescrptUniMol):
            raise TypeError(
                "DPUniMolAtomicModel needs the unimol descriptor, which is the only "
                "one producing a Uni-Mol token sequence"
            )
        if not isinstance(fitting, UniMolPretrainFitting):
            raise TypeError("DPUniMolAtomicModel needs the unimol_pretrain fitting")
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
        """Run the backbone and its heads at token resolution.

        Parameters
        ----------
        extended_coord
            nf x (nall x 3) coordinates; the descriptor rejects any frame that
            carries periodic images.
        extended_atype
            nf x nall element types, already clamped to be non-negative.
        nlist
            nf x nloc x nnei neighbour list, which is how real atoms are told
            apart from padding.
        mapping, fparam, aparam, comm_dict, charge_spin
            Unused by this model.

        Returns
        -------
        dict
            The three head outputs plus the two norm regularisers.
        """
        del mapping, fparam, aparam, comm_dict, charge_spin
        backbone = self.descriptor.forward_tokens(extended_coord, extended_atype, nlist)
        return self.fitting_net.call_tokens(backbone)

    def apply_out_stat(self, ret: dict[str, Array], atype: Array) -> dict[str, Array]:
        """Return the head outputs untouched.

        Self-supervised targets carry no per-element bias to add back: the
        element head predicts a distribution, and the coordinate and distance
        heads predict geometry the data already fixes.
        """
        del atype
        return ret
