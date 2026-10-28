# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model wrapper for Uni-Mol v1 self-supervised pretraining."""

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model import (
    DPUniMolAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.output_def import (
    OutputVariableDef,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)


def _reject_periodic(box) -> None:  # noqa: ANN001
    """Refuse a periodic cell, with an explanation rather than an allocation."""
    if box is None:
        return
    xp = array_api_compat.array_namespace(box)
    if bool(xp.any(box != 0)):
        raise ValueError(
            "the unimol descriptor is molecular and does not support periodic "
            "boundaries; pass box=None"
        )


DPUniMolPretrainModel_ = make_model(DPUniMolAtomicModel, T_Bases=(NativeOP, BaseModel))


@BaseModel.register("unimol_pretrain")
class UniMolPretrainModel(DPModelCommon, DPUniMolPretrainModel_):
    r"""Uni-Mol v1 molecular pretraining.

    Predicts the element of every corrupted atom, denoises the coordinates and
    predicts the clean pairwise distances, and reports the two norm
    regularisers of the backbone. Nothing here reduces to a frame total and
    nothing is differentiated with respect to the coordinates: this is
    representation learning, not a potential energy surface.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DPModelCommon.__init__(self)
        DPUniMolPretrainModel_.__init__(self, *args, **kwargs)

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
        """Evaluate the pretraining heads on a frame.

        Raises
        ------
        ValueError
            If a periodic cell is supplied. Uni-Mol is molecular: it has no
            cut-off, so a cell would ask the neighbour-list builder for an
            astronomical number of images before any other check could fire.
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

    def call_lower(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
        charge_spin: Array | None = None,
    ) -> dict[str, Array]:
        """Evaluate the pretraining heads on an extended frame."""
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        return {k: v for k, v in model_ret.items() if v is not None}

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
        """The head outputs, passed through under their own names."""
        return dict(self.model_output_def().get_data())
