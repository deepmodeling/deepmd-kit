# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native-spin SeZM energy model.

Unlike the virtual-atom :class:`SeZMSpinModel`, the native scheme injects the
per-atom spin vector directly into the descriptor as an equivariant feature
(``l = 0`` magnitude and ``l = 1`` direction) and obtains the magnetic force as
the negative spin gradient of the energy. No virtual atoms are created, so the
neighbor list, type map and selection stay at their real-system sizes, and the
analytical bridging potential needs no real/virtual masking.
"""

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import torch
from einops import (
    rearrange,
)

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.pt.model.atomic_model.sezm_atomic_model import (
    SeZMAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.sezm_model import (
    SeZMModel,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.spin import (
    Spin,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@BaseModel.register("sezm_native_spin")
class SeZMNativeSpinModel(SeZMModel):
    """SeZM energy model with native (virtual-atom-free) spin.

    The per-atom spin enters the descriptor through :class:`SpinEmbedding`, and
    the magnetic force is the negative spin gradient of the energy, produced by
    the same backward as the conservative force and virial.

    Parameters
    ----------
    spin
        Spin metadata describing which real atom types carry spin.
    *args
        Positional arguments forwarded to :class:`SeZMModel`.
    **kwargs
        Keyword arguments forwarded to :class:`SeZMModel`.
    """

    model_type = "sezm_native_spin"

    def __init__(
        self,
        *args: Any,
        spin: Spin,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spin = spin
        self.ntypes_real = self.spin.ntypes_real
        # Per-type 0/1 spin gate.
        self.register_buffer(
            "spin_mask",
            to_torch_tensor(self.spin.get_spin_mask()),
            persistent=False,
        )

    # =========================================================================
    # Forward Methods
    # =========================================================================

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return native-spin SeZM predictions with public output keys.

        ``mask_mag`` is built from the per-type spin gate on the local
        ``atype``; non-magnetic atoms already carry a zero magnetic force (the
        descriptor gates the spin embedding by type), so the force itself needs
        no re-masking. This is the runtime counterpart of the static schema in
        :meth:`translated_output_def`.
        """
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
            spin=spin,
        )
        nf, nloc = atype.shape[:2]
        model_predict: dict[str, torch.Tensor] = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "mask_mag": self.spin_mask.index_select(0, atype.reshape(-1)).reshape(
                nf, nloc, 1
            )
            > 0.0,
        }
        if self.do_grad_r("energy"):
            model_predict["force"] = rearrange(
                model_ret["energy_derv_r"], "nf n 1 three -> nf n three", three=3
            )
            model_predict["force_mag"] = rearrange(
                model_ret["energy_derv_r_mag"], "nf n 1 three -> nf n three", three=3
            )
        if self.do_grad_c("energy"):
            model_predict["virial"] = rearrange(
                model_ret["energy_derv_c_redu"], "nf 1 nine -> nf nine", nine=9
            )
            if do_atomic_virial:
                model_predict["atom_virial"] = rearrange(
                    model_ret["energy_derv_c"], "nf n 1 nine -> nf n nine", nine=9
                )
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    # =========================================================================
    # Export
    # =========================================================================

    def forward_common_lower_exportable(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_scatter_index: torch.Tensor,
        edge_mask: torch.Tensor,
        spin: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> torch.nn.Module:
        """Trace the native-spin lower interface into an exportable FX graph.

        The native scheme reuses the energy model's edge ABI (``coord``,
        ``atype``, ``edge_index``, ``edge_vec``, ``edge_scatter_index``,
        ``edge_mask``); the only addition is the per-local-atom ``spin`` leaf,
        so ``make_fx`` unfolds the single
        ``autograd.grad(energy, [edge_vec, spin])`` into the conservative force
        and the magnetic force. The C++ backend builds the edge schema (exactly
        like :class:`DeepPotPTExpt`) and feeds the owned-atom spins, so a spin
        and a non-spin ``.pt2`` archive share one inference path.

        Parameters
        ----------
        coord
            Extended coordinates with shape ``(nf, nall, 3)``.
        atype
            Local atom types with shape ``(nf, nloc)``.
        edge_index
            Local-folded source/destination indices with shape ``(2, nedge)``.
        edge_vec
            Per-edge displacement with shape ``(nedge, 3)``.
        edge_scatter_index
            Extended source/destination indices for the force scatter with
            shape ``(2, nedge)``.
        edge_mask
            Boolean per-edge validity mask with shape ``(nedge,)``.
        spin
            Per-local-atom spin vectors with shape ``(nf, nloc, 3)``.
        fparam, aparam, charge_spin
            Optional frame / atomic / charge-spin conditioning inputs.

        Returns
        -------
        torch.nn.Module
            The traced exportable lower graph.
        """
        if self.get_active_mode() == "dens":
            raise NotImplementedError(
                "SeZM export supports only the conservative `ener` path."
            )
        model = self

        def lower_fn(
            coord_: torch.Tensor,
            atype_: torch.Tensor,
            edge_index_: torch.Tensor,
            edge_vec_: torch.Tensor,
            edge_scatter_index_: torch.Tensor,
            edge_mask_: torch.Tensor,
            spin_: torch.Tensor,
            fparam_: torch.Tensor | None,
            aparam_: torch.Tensor | None,
            charge_spin_: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            # Detach the leaves inside the traced closure so the exported graph
            # owns its own force/magnetic-force autograd endpoints (edge_vec and
            # spin) rather than capturing the upstream LAMMPS tensors.
            coord_ = coord_.detach()
            edge_vec_ = edge_vec_.detach()
            model_ret = model.forward_common_lower(
                coord_,
                atype_,
                edge_index_,
                edge_vec_,
                edge_scatter_index_,
                edge_mask_,
                fparam=fparam_,
                aparam=aparam_,
                charge_spin=charge_spin_,
                spin=spin_,
                use_compile=False,
            )
            return model._attach_spin_masks(
                model_ret, atype=atype_, nall=coord_.shape[1]
            )

        if self.get_dim_chg_spin() > 0:
            charge_spin = self.convert_charge_spin(
                charge_spin,
                nf=atype.shape[0],
                dtype=coord.dtype,
                device=coord.device,
            )
        return self.trace_lower_exportable(
            lower_fn,
            coord,
            atype,
            edge_index,
            edge_vec,
            edge_scatter_index,
            edge_mask,
            spin,
            fparam,
            aparam,
            charge_spin,
        )

    def forward_common_lower_exportable_with_comm(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        extended_atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_scatter_index: torch.Tensor,
        edge_mask: torch.Tensor,
        spin: torch.Tensor,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
        send_list: torch.Tensor,
        send_proc: torch.Tensor,
        recv_proc: torch.Tensor,
        send_num: torch.Tensor,
        recv_num: torch.Tensor,
        communicator: torch.Tensor,
        nlocal: torch.Tensor,
        nghost: torch.Tensor,
    ) -> torch.nn.Module:
        """Trace the native-spin parallel (with-comm) lower interface.

        Mirrors :meth:`SeZMModel.forward_common_lower_exportable_with_comm` and
        adds the extended (nall) per-atom spin leaf. On the parallel path the
        spin is a per-extended-node feature -- the LAMMPS spin reverse/forward
        comm supplies ghost spins -- so the magnetic force ``-dE/dspin`` is
        itself extended (nall): its ghost rows are the cross-rank neighbour
        contributions that ``border_op``'s exact-VJP backward routes correctly
        and the LAMMPS spin reverse-comm folds onto owners. ``_attach_spin_masks``
        is a no-op pad here (the force is already extended) and attaches the
        extended ``mask_mag``.
        """
        if self.get_active_mode() == "dens":
            raise NotImplementedError(
                "SeZM export supports only the conservative `ener` path."
            )
        from deepmd.pt_expt.utils.comm import (
            ensure_comm_registered,
        )

        ensure_comm_registered()
        model = self

        def fn(
            coord_: torch.Tensor,
            atype_: torch.Tensor,
            extended_atype_: torch.Tensor,
            edge_index_: torch.Tensor,
            edge_vec_: torch.Tensor,
            edge_scatter_index_: torch.Tensor,
            edge_mask_: torch.Tensor,
            spin_: torch.Tensor,
            fparam_: torch.Tensor | None,
            aparam_: torch.Tensor | None,
            charge_spin_: torch.Tensor | None,
            send_list_: torch.Tensor,
            send_proc_: torch.Tensor,
            recv_proc_: torch.Tensor,
            send_num_: torch.Tensor,
            recv_num_: torch.Tensor,
            communicator_: torch.Tensor,
            nlocal_: torch.Tensor,
            nghost_: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            coord_ = coord_.detach()
            edge_vec_ = edge_vec_.detach()
            comm_dict = {
                "send_list": send_list_,
                "send_proc": send_proc_,
                "recv_proc": recv_proc_,
                "send_num": send_num_,
                "recv_num": recv_num_,
                "communicator": communicator_,
                "nlocal": nlocal_,
                "nghost": nghost_,
            }
            model_ret = model.forward_common_lower(
                coord_,
                atype_,
                edge_index_,
                edge_vec_,
                edge_scatter_index_,
                edge_mask_,
                fparam=fparam_,
                aparam=aparam_,
                comm_dict=comm_dict,
                extended_atype=extended_atype_,
                charge_spin=charge_spin_,
                spin=spin_,
                use_compile=False,
            )
            return model._attach_spin_masks(
                model_ret, atype=extended_atype_, nall=coord_.shape[1]
            )

        if self.get_dim_chg_spin() > 0:
            charge_spin = self.convert_charge_spin(
                charge_spin,
                nf=atype.shape[0],
                dtype=coord.dtype,
                device=coord.device,
            )
        return self.trace_lower_exportable(
            fn,
            coord,
            atype,
            extended_atype,
            edge_index,
            edge_vec,
            edge_scatter_index,
            edge_mask,
            spin,
            fparam,
            aparam,
            charge_spin,
            send_list,
            send_proc,
            recv_proc,
            send_num,
            recv_num,
            communicator,
            nlocal,
            nghost,
        )

    def _attach_spin_masks(
        self,
        model_ret: dict[str, torch.Tensor],
        *,
        atype: torch.Tensor,
        nall: int,
    ) -> dict[str, torch.Tensor]:
        """Express the magnetic force in the extended layout and attach ``mask_mag``.

        Parameters
        ----------
        model_ret
            Internal SeZM lower outputs; ``energy_derv_r_mag`` has the
            per-local-atom shape ``(nf, nloc, 1, 3)``.
        atype
            Local atom types with shape ``(nf, nloc)``, used to build
            ``mask_mag`` with shape ``(nf, nloc, 1)``.
        nall
            Extended atom count the magnetic force is padded to.

        Returns
        -------
        dict[str, torch.Tensor]
            ``model_ret`` with ``energy_derv_r_mag`` padded to ``nall`` and a
            ``mask_mag`` entry.

        Notes
        -----
        The magnetic force is intrinsically per-local-atom (only owned spins
        enter the descriptor); padding the ghost slots with zero lets it share
        the extended reduce / fold-back contract that
        ``communicate_extended_output`` and the LAMMPS C++ backend apply to the
        conservative force, so the native and deepspin schemes share one
        downstream path. The padding is unconditional (``nall - nloc`` is zero
        for an isolated cluster) so the closure stays free of shape-dependent
        branches under ``make_fx`` symbolic tracing.
        """
        derv_r_mag = model_ret["energy_derv_r_mag"]  # (nf, nloc, 1, 3)
        nf, nloc = derv_r_mag.shape[:2]
        ghost_pad = derv_r_mag.new_zeros(nf, nall - nloc, *derv_r_mag.shape[2:])
        model_ret["energy_derv_r_mag"] = torch.cat([derv_r_mag, ghost_pad], dim=1)
        model_ret["mask_mag"] = (
            self.spin_mask.index_select(0, atype.reshape(-1)).reshape(
                atype.shape[0], atype.shape[1], 1
            )
            > 0.0
        )
        return model_ret

    # =========================================================================
    # Mode Selection
    # =========================================================================

    def set_active_mode(self, mode: str) -> None:
        """Switch mode, allowing only the conservative energy path."""
        normalized = str(mode).lower()
        if normalized != "ener":
            raise NotImplementedError("SeZM native spin supports only the `ener` path.")
        super().set_active_mode(normalized)

    def set_active_mode_from_loss(self, loss_type: str) -> None:
        """Select execution mode from loss type."""
        normalized = str(loss_type).lower()
        if normalized == "dens":
            raise NotImplementedError("SeZM native spin supports only the `ener` path.")
        if normalized in {"ener", "ener_spin"}:
            self.set_active_mode("ener")

    # =========================================================================
    # Output Definitions and Metadata
    # =========================================================================

    def has_spin(self) -> bool:
        """Return whether this model consumes spin input."""
        return True

    def model_output_def(self) -> ModelOutputDef:
        """Return the spin-aware model output definition."""
        atomic_output_def = self.atomic_output_def()
        atomic_output_def["energy"].magnetic = True
        return ModelOutputDef(atomic_output_def)

    def translated_output_def(self) -> dict[str, Any]:
        """Translate internal output definitions to public spin keys."""
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
            "mask_mag": out_def_data["mask_mag"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
            output_def["force"].squeeze(-2)
            output_def["force_mag"] = deepcopy(out_def_data["energy_derv_r_mag"])
            output_def["force_mag"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = deepcopy(out_def_data["energy_derv_c_redu"])
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = deepcopy(out_def_data["energy_derv_c"])
            output_def["atom_virial"].squeeze(-2)
        return output_def

    # =========================================================================
    # Serialization
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize the native-spin SeZM model."""
        data = super().serialize()
        data["type"] = self.model_type
        data["spin"] = self.spin.serialize()
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SeZMNativeSpinModel":
        """Deserialize a native-spin SeZM model."""
        data = data.copy()
        version = int(data.pop("@version", 1))
        check_version_compatibility(version, 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        spin = Spin.deserialize(data.pop("spin"))
        atomic_model = SeZMAtomicModel.deserialize(data.pop("atomic_model"))
        return cls(atomic_model_=atomic_model, spin=spin, **data)
