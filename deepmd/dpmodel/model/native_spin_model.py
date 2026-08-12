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

import array_api_compat
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
    cross-backend family test check against this shared marker instead. The
    motivating consumer is the with-comm freeze gate, where native spin
    excludes only the DENSE lower; native-spin GRAPH lowers do participate
    in the with-comm path and carry the nested artifact (issue #5906).
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

        CONFIG_DERIVED_ARRAYS = ("spin_mask",)

        def __init__(self, *args: Any, spin: Spin, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.spin = spin
            self.ntypes_real = self.spin.ntypes_real
            # Per-real-type 0/1 spin gate, derived from ``use_spin`` and hence
            # rebuilt here rather than adopted from a checkpoint.
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
            if "mask" in out_def_data:
                output_def["mask"] = out_def_data["mask"]
            return output_def

        def _spin_active_mask(self, atype: np.ndarray) -> np.ndarray:
            """Single owner of ``mask_mag``: ``(N|nf,nloc, 1)`` bool, True
            where the atom type carries spin. Array-api compatible; reused by
            the eager and graph-export translations.
            """
            xp = array_api_compat.array_namespace(atype)
            real_atom = atype >= 0
            safe_atype = xp.where(real_atom, atype, xp.zeros_like(atype))
            spin_mask = xp.asarray(
                self.spin_mask,
                device=array_api_compat.device(atype),
            )
            spin_active = (
                xp.reshape(
                    xp.take(spin_mask, xp.reshape(safe_atype, (-1,)), axis=0),
                    atype.shape,
                )
                > 0
            )
            return xp.logical_and(spin_active, real_atom)[..., None]

        def _translate_eager_call(
            self,
            model_ret: dict[str, np.ndarray],
            atype: np.ndarray,
            do_atomic_virial: bool = False,
        ) -> dict[str, np.ndarray | None]:
            """Single owner of the native-spin output translation, shared by
            dpmodel/pt_expt ``call``/``forward`` and the graph export. Each
            derivative key is the backend's value (pt_expt autograd) or
            ``None`` (energy-only dpmodel); ``atom_virial`` is treated like
            ``virial``, gated on ``do_atomic_virial``.
            """
            out: dict[str, np.ndarray | None] = {
                "atom_energy": model_ret["energy"],
                "energy": model_ret["energy_redu"],
                "mask_mag": self._spin_active_mask(atype),
            }
            translated = [
                ("energy_derv_r", "force"),
                ("energy_derv_r_mag", "force_mag"),
                ("energy_derv_c_redu", "virial"),
            ]
            if do_atomic_virial:
                # Per-atom virial is opt-in (2.5x cost) -- only surfaced when
                # requested, unlike the always-present reduced keys above.
                translated.append(("energy_derv_c", "atom_virial"))
            for kk_src, kk_dst in translated:
                src = model_ret.get(kk_src)
                out[kk_dst] = np.squeeze(src, axis=-2) if src is not None else None
            for key in ("mask", "n_node"):
                if key in model_ret:
                    out[key] = model_ret[key]
            return out

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
                If set, request the per-atom virial (``atom_virial`` key).
                The energy-only dpmodel backend produces no derivatives, so
                the value is ``None`` here (same as ``force``/``virial``);
                the pt_expt subclass fills it with a real autograd tensor.
            charge_spin
                Frame-level charge/spin FiLM conditioning, shape
                nf x dim_chg_spin (only consumed when the descriptor
                declares ``add_chg_spin_ebd``).

            Returns
            -------
            ret_dict
                The result dict with translated keys: ``atom_energy``,
                ``energy``, ``mask_mag``, plus
                ``force``/``force_mag``/``virial`` (and ``atom_virial`` when
                ``do_atomic_virial``) as ``None`` placeholders when the
                backend produces no derivatives (dpmodel; the pt_expt
                subclass produces real autograd tensors).
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
            return self._translate_eager_call(
                model_ret, atype, do_atomic_virial=do_atomic_virial
            )

        def serialize(self) -> dict:
            data = super().serialize()
            # The backbone's own wire type would be lost under "native_spin";
            # keep it so deserialize can rebuild the RIGHT backbone.  It is
            # not always "standard": with analytical bridging the backbone is
            # a composition ("linear"), whose dict has a different shape and
            # @version.
            data["backbone_type"] = data.get("type", "standard")
            data["type"] = "native_spin"
            data["spin"] = self.spin.serialize()
            return data

        @classmethod
        def deserialize(cls, data: dict) -> "NSM":
            data = data.copy()
            data.pop("type", None)
            spin = Spin.deserialize(data.pop("spin"))
            # make_model flat shape: the remaining dict IS the backbone
            # (atomic) dict -- its @class/@version belong to the backbone's
            # deserialize and must stay.  Archives written before
            # ``backbone_type`` existed are all plain standard models.
            backbone_type = data.pop("backbone_type", "standard")
            data["type"] = backbone_type
            backbone_cls = (
                T_Model
                if backbone_type == "standard"
                else T_Model.get_class_by_type(backbone_type)
            )
            backbone = backbone_cls.deserialize(data)
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
