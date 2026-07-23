# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native-spin model factory (``make_native_spin_model``) and its concrete
energy-model instantiation (``NativeSpinEnergyModel``).
"""

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
)
from deepmd.utils.spin import (
    Spin,
)


class NativeSpinModelKind:
    """Marker base identifying classes produced by ``make_native_spin_model``.

    Each backend instantiates the factory on its OWN standard model class,
    so the concrete classes (e.g. dpmodel's and pt_expt's
    ``NativeSpinEnergyModel``) are parallel products with NO subclass
    relation between them -- an ``isinstance`` against one backend's
    concrete class is silently dead in the other. Backend seams that need a
    cross-backend family test (e.g. the with-comm freeze gate: native-spin
    lowers are single-rank only) test against this shared marker instead.
    """


def make_native_spin_model(T_Model: type) -> type:
    """Make a native-spin model class from a standard model class.

    The native scheme injects the per-atom spin vector directly into the
    descriptor as an equivariant feature and obtains the magnetic force as
    the negative spin gradient of the energy. No virtual atoms are created
    (unlike :class:`~deepmd.dpmodel.model.spin_model.SpinModel`), so the
    neighbor list, type map and selection stay at the real-system sizes.

    Mirrors :func:`~deepmd.dpmodel.model.make_model.make_model`'s
    class-factory pattern: the produced class subclasses ``T_Model`` (is-a),
    serializes as the parent's flat dict plus a ``spin`` field under wire
    type ``"native_spin"``, and is meant to be registered in each backend's
    ``BaseModel`` plugin registry so ``deserialize`` dispatch stays
    backend-aware. Eligibility of a backbone is the
    ``descriptor.supports_native_spin()`` capability, checked by the config
    builders -- the factory itself is descriptor-agnostic.

    Parameters
    ----------
    T_Model : type
        The standard model class to derive from (e.g. the backend's
        ``EnergyModel``).

    Returns
    -------
    type
        The derived native-spin model class.
    """

    class NSM(T_Model, NativeSpinModelKind):
        """Native-spin variant of ``T_Model`` (see ``make_native_spin_model``)."""

        def __init__(self, *args: Any, spin: Spin, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.spin = spin
            self.ntypes_real = self.spin.ntypes_real
            # Per-real-type 0/1 spin gate.
            self.spin_mask = self.spin.get_spin_mask()

        @staticmethod
        def has_spin() -> bool:
            """Returns whether it has spin input and output."""
            return True

        def model_output_def(self) -> ModelOutputDef:
            """Get the spin-aware output def for the model."""
            atomic_output_def = self.atomic_output_def()
            atomic_output_def["energy"].magnetic = True
            return ModelOutputDef(atomic_output_def)

        def translated_output_def(self) -> dict[str, Any]:
            """Get the translated output definition with public spin keys.

            Maps internal output names to user-facing names, e.g.
            ``energy`` -> ``atom_energy``, ``energy_redu`` -> ``energy``,
            ``energy_derv_r`` -> ``force``, ``energy_derv_r_mag`` ->
            ``force_mag``. Built from this class's OWN
            :meth:`model_output_def` (which sets ``energy.magnetic = True``).
            """
            out_def_data = self.model_output_def().get_data()
            model_output_type = self.model_output_type()
            if "mask" in model_output_type:
                model_output_type.pop(model_output_type.index("mask"))
            var_name = model_output_type[0]
            output_def = {
                f"atom_{var_name}": out_def_data[var_name],
                var_name: out_def_data[f"{var_name}_redu"],
                "mask_mag": out_def_data["mask_mag"],
            }
            if self.do_grad_r(var_name):
                output_def["force"] = deepcopy(out_def_data[f"{var_name}_derv_r"])
                output_def["force"].squeeze(-2)
                output_def["force_mag"] = deepcopy(
                    out_def_data[f"{var_name}_derv_r_mag"]
                )
                output_def["force_mag"].squeeze(-2)
            if self.do_grad_c(var_name):
                output_def["virial"] = deepcopy(out_def_data[f"{var_name}_derv_c_redu"])
                output_def["virial"].squeeze(-2)
                output_def["atom_virial"] = deepcopy(out_def_data[f"{var_name}_derv_c"])
                output_def["atom_virial"].squeeze(-2)
            return output_def

        def call(
            self,
            coord: np.ndarray,
            atype: np.ndarray,
            spin: np.ndarray,
            box: np.ndarray | None = None,
            fparam: np.ndarray | None = None,
            aparam: np.ndarray | None = None,
            do_atomic_virial: bool = False,
            charge_spin: np.ndarray | None = None,
        ) -> dict[str, np.ndarray]:
            """Return native-spin model predictions with translated public keys.

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
                If calculate the atomic virial (unused: dpmodel is
                energy-only for this class).
            charge_spin
                Frame-level charge/spin FiLM conditioning, shape
                nf x dim_chg_spin (only consumed when the descriptor
                declares ``add_chg_spin_ebd``).

            Returns
            -------
            ret_dict
                The result dict with translated keys: ``atom_energy``,
                ``energy``, ``mask_mag``, plus
                ``force``/``force_mag``/``virial`` as ``None`` placeholders
                when the backend produces no derivatives (dpmodel; the
                pt_expt subclass produces real autograd tensors).
            """
            model_ret = self.call_common(
                coord,
                atype,
                box=box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                spin=spin,
                charge_spin=charge_spin,
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

        def serialize(self) -> dict:
            data = super().serialize()
            data["type"] = "native_spin"
            data["spin"] = self.spin.serialize()
            return data

        @classmethod
        def deserialize(cls, data: dict) -> "NSM":
            data = data.copy()
            data.pop("type", None)
            spin = Spin.deserialize(data.pop("spin"))
            # make_model flat shape: the remaining dict IS the standard
            # model (atomic) dict -- its @class/@version belong to the
            # atomic deserialize and must stay.
            data["type"] = "standard"
            backbone = T_Model.deserialize(data)
            return cls(atomic_model_=backbone.atomic_model, spin=spin)

    return NSM


@BaseModel.register("native_spin")
class NativeSpinEnergyModel(make_native_spin_model(EnergyModel)):
    r"""Native-spin energy model (dpmodel backend).

    dpmodel is energy-only for this model: it forwards through the
    NeighborGraph lower (energy-only by design -- see
    :meth:`~deepmd.dpmodel.model.make_model.make_model._call_common_graph`),
    so ``call`` returns ``energy``/``atom_energy``/``mask_mag`` with
    ``force``/``force_mag``/``virial`` as ``None`` placeholders. Force and
    magnetic force are produced by autograd in the pt_expt backend.
    Currently the DPA4/SeZM descriptor is the only one declaring
    ``supports_native_spin()``.
    """
