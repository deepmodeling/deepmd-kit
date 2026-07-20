# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
)
from deepmd.utils.spin import (
    Spin,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.atomic_model.dp_atomic_model import (
        DPAtomicModel,
    )


class DPA4NativeSpinModel(NativeOP):
    r"""DPA4/SeZM native-spin model on the NeighborGraph route.

    Unlike :class:`~deepmd.dpmodel.model.spin_model.SpinModel`, this wrapper
    creates no virtual atoms: ``spin`` is a per-local-atom ``(nf, nloc, 3)``
    input consumed directly by the descriptor's equivariant spin embedding
    (``DescrptDPA4.call_graph(..., spin=...)``), so the type map, neighbor
    selection and type count stay at the real-system sizes.

    dpmodel is energy-only for this wrapper: it forwards through the
    NeighborGraph lower (energy-only by design -- see
    :meth:`~deepmd.dpmodel.model.make_model.make_model._call_common_graph`),
    so ``call`` returns ``energy``/``atom_energy``/``mask_mag`` with
    ``force``/``force_mag``/``virial`` as ``None`` placeholders. Force and
    magnetic force are produced by autograd in the pt_expt backend.
    """

    def __init__(self, backbone_model: Any, spin: Spin) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.spin = spin
        self.ntypes_real = self.spin.ntypes_real
        # Per-real-type 0/1 spin gate.
        self.spin_mask = self.spin.get_spin_mask()

    # =========================================================================
    # Delegation (mirrors deepmd/dpmodel/model/spin_model.py's delegation
    # block, minus the virtual-atom-specific bits: no type_map doubling, no
    # nnei/nsel halving, no virtual-atom output splitting).
    # =========================================================================

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.backbone_model.get_type_map()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return len(self.get_type_map())

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.backbone_model.get_rcut()

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.backbone_model.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.backbone_model.get_dim_aparam()

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution to the
        result of the model. If returning an empty list, all atom types are
        selected.
        """
        return self.backbone_model.get_sel_type()

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return self.backbone_model.is_aparam_nall()

    def model_output_type(self) -> list[str]:
        """Get the output type for the model."""
        return self.backbone_model.model_output_type()

    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        return self.backbone_model.get_model_def_script()

    def get_min_nbor_dist(self) -> float | None:
        """Get the minimum neighbor distance."""
        return self.backbone_model.get_min_nbor_dist()

    def get_nnei(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        return self.backbone_model.get_nnei()

    def get_nsel(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        return self.backbone_model.get_nsel()

    def has_message_passing(self) -> bool:
        """Returns whether the model has message passing."""
        return self.backbone_model.has_message_passing()

    def atomic_output_def(self) -> FittingOutputDef:
        """Get the output def of the atomic model."""
        return self.backbone_model.atomic_output_def()

    @staticmethod
    def has_spin() -> bool:
        """Returns whether it has spin input and output."""
        return True

    def get_dp_atomic_model(self) -> "DPAtomicModel | None":
        """Get the underlying DPAtomicModel by delegating to the backbone model."""
        return self.backbone_model.get_dp_atomic_model()

    def __getattr__(self, name: str) -> Any:
        """Fall back to the wrapped backbone model for anything not defined above."""
        if "backbone_model" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.backbone_model, name)

    # =========================================================================
    # Output definition
    # =========================================================================

    def model_output_def(self) -> ModelOutputDef:
        """Get the output def for the model."""
        backbone_def = self.backbone_model.atomic_output_def()
        backbone_def["energy"].magnetic = True
        return ModelOutputDef(backbone_def)

    # =========================================================================
    # Forward
    # =========================================================================

    def call(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
        spin: np.ndarray,
        box: np.ndarray | None = None,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return native-spin model predictions with translated user-facing keys.

        Parameters
        ----------
        coord
            The coordinates of the atoms. shape: nf x (nloc x 3)
        atype
            The type of atoms. shape: nf x nloc
        spin
            The per-local-atom spin. shape: nf x (nloc x 3)
        box
            The simulation box. shape: nf x 9
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda
        do_atomic_virial
            If calculate the atomic virial (unused: dpmodel is energy-only
            for this wrapper).

        Returns
        -------
        ret_dict
            The result dict with translated keys: ``atom_energy``,
            ``energy``, ``mask_mag``, plus ``force``/``force_mag``/``virial``
            as ``None`` placeholders (produced by pt_expt autograd instead).
        """
        model_ret = self.backbone_model.call_common(
            coord,
            atype,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            spin=spin,
            # dpmodel: opt into the carry-all NeighborGraph builder (the
            # only lower that consumes model-level spin).
            neighbor_graph_method="dense",
        )
        out: dict[str, np.ndarray | None] = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "mask_mag": (self.spin_mask[atype] > 0)[..., None],
        }
        for kk_src, kk_dst in (
            ("energy_derv_r", "force"),
            ("energy_derv_r_mag", "force_mag"),
            ("energy_derv_c_redu", "virial"),
        ):
            src = model_ret.get(kk_src)
            out[kk_dst] = np.squeeze(src, axis=-2) if src is not None else None
        return out

    # =========================================================================
    # (De)serialization
    # =========================================================================

    def serialize(self) -> dict:
        return {
            "@class": "Model",
            "@version": 1,
            "type": "dpa4_native_spin",
            "backbone_model": self.backbone_model.serialize(),
            "spin": self.spin.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DPA4NativeSpinModel":
        data = data.copy()
        data.pop("@class", None)
        data.pop("@version", None)
        data.pop("type", None)
        spin = Spin.deserialize(data.pop("spin"))
        backbone_model = BaseModel.deserialize(data.pop("backbone_model"))
        return cls(backbone_model=backbone_model, spin=spin)
