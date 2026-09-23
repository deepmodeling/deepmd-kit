# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model wrapper for Uni-Mol pretraining on a DPA backbone."""

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model import (
    DPUniMolDPAAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)


def _reject_periodic(box) -> None:  # noqa: ANN001
    """Refuse a cell, with an explanation rather than a quiet mistraining."""
    if box is None:
        return
    xp = array_api_compat.array_namespace(box)
    if bool(xp.any(box != 0)):
        raise ValueError(
            "the unimol objective does not support periodic boundaries: its "
            "distance target is built from plain coordinate differences with no "
            "minimum-image convention, and its coverage keeps only local "
            "neighbours, so a cell would train against wrong labels rather than "
            "fail. Pass box=None"
        )


DPUniMolDPAPretrainModel_ = make_model(
    DPUniMolDPAAtomicModel, T_Bases=(NativeOP, BaseModel)
)


@BaseModel.register("unimol_dpa_pretrain")
class UniMolDPAPretrainModel(DPModelCommon, DPUniMolDPAPretrainModel_):
    r"""Uni-Mol's pretraining objective on a DPA backbone.

    Predicts the element of every corrupted atom, denoises the coordinates from
    the backbone's equivariant state, and predicts the clean pairwise distances
    from pairs of atom representations. Nothing reduces to a frame total and
    nothing is differentiated with respect to the coordinates: this is
    representation learning, not a potential energy surface.

    Periodic frames are refused. The backbone handles them, but this objective
    does not: the distance target is built from plain coordinate differences
    with no minimum-image convention, and the coverage mask keeps only local
    neighbours, so under a cell the labels are wrong by up to several Angstrom
    and some exceed the cut-off entirely. It would run and quietly mistrain.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DPModelCommon.__init__(self)
        DPUniMolDPAPretrainModel_.__init__(self, *args, **kwargs)

    def call(
        self,
        coord: Array,
        atype: Array,
        box: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
        charge_spin: Array | None = None,
    ) -> dict[str, Array]:
        """Evaluate the heads on a frame.

        Raises
        ------
        ValueError
            If a periodic cell is supplied; see the class docstring.
        """
        _reject_periodic(box)
        model_ret = self.call_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        return {k: v for k, v in model_ret.items() if v is not None}
