# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.dpmodel.model.make_hessian_model import (
    make_hessian_model,
)
from deepmd.dpmodel.utils.neighbor_list import (
    NeighborList,
)

from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel, T_Bases=(BaseModel,))


def _translate_energy_keys(
    model_ret: dict[str, torch.Tensor],
    *,
    do_grad_r: bool,
    do_grad_c: bool,
    do_atomic_virial: bool,
    local: bool,
) -> dict[str, torch.Tensor]:
    """Map internal fitting keys -> public energy-model keys (shared by the
    dense and graph ``forward_lower`` export traces).

    Operates on plain dicts (make_fx-safe). ``local=True`` is the GRAPH path
    (per-node ``N == sum(n_node)`` local atoms, no ghost/extended region) and
    emits ``force``/``atom_virial``; ``local=False`` is the DENSE extended-region
    path and emits ``extended_force``/``extended_virial`` (folded to local by
    ``communicate_extended_output`` at inference).
    """
    out: dict[str, torch.Tensor] = {}
    out["atom_energy"] = model_ret["energy"]
    out["energy"] = model_ret["energy_redu"]
    if do_grad_r:
        out["force" if local else "extended_force"] = model_ret[
            "energy_derv_r"
        ].squeeze(-2)
    if do_grad_c:
        out["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        if do_atomic_virial:
            out["atom_virial" if local else "extended_virial"] = model_ret[
                "energy_derv_c"
            ].squeeze(-2)
    if "mask" in model_ret:
        out["mask"] = model_ret["mask"]
    return out


@BaseModel.register("ener")
@BaseModel.register("sezm_ener")
@BaseModel.register("dpa4_ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)
        self._hessian_enabled = False

    def enable_hessian(self) -> None:
        if self._hessian_enabled:
            return
        self.__class__ = make_hessian_model(type(self))
        self.hess_fitting_def = copy.deepcopy(
            super(type(self), self).atomic_output_def()
        )
        self.requires_hessian("energy")
        self._hessian_enabled = True

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        neighbor_list: NeighborList | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the energy model.

        Most arguments share the meaning of :meth:`call_common`.

        Parameters
        ----------
        neighbor_list
            The neighbor-list construction strategy forwarded to
            :meth:`call_common`.  ``None`` uses the default all-pairs builder
            (:class:`~deepmd.dpmodel.utils.default_neighbor_list.DefaultNeighborList`),
            reproducing the historical behavior; an alternative strategy (e.g.
            the ``vesin`` O(N) cell list) may be injected to accelerate
            neighbor-list construction without changing the model outputs.
        """
        model_ret = self.call_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
            neighbor_list=neighbor_list,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        if self.atomic_output_def()["energy"].r_hessian:
            model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-3)
        return model_predict

    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -2
                )
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"]
            output_def["atom_virial"].squeeze(-2)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        if self.atomic_output_def()["energy"].r_hessian:
            output_def["hessian"] = out_def_data["energy_derv_r_derv_r"]
        return output_def

    def forward_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace ``forward_lower`` into an exportable module.

        Delegates to ``forward_common_lower_exportable`` for tracing,
        then translates the internal keys to the ``forward_lower``
        convention.

        Parameters
        ----------
        extended_coord, extended_atype, nlist, mapping, fparam, aparam, do_atomic_virial
            Sample inputs with representative shapes (used for tracing).
        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx``
            (e.g. ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts
            ``(extended_coord, extended_atype, nlist, mapping, fparam, aparam)``
            and returns a dict with the same keys as ``forward_lower``.
        """
        traced = self.forward_common_lower_exportable(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
            **make_fx_kwargs,
        )

        # Translate internal keys to forward_lower convention.
        # Capture model config at trace time via closures.
        do_grad_r = self.do_grad_r("energy")
        do_grad_c = self.do_grad_c("energy")

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
            charge_spin: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            model_ret = traced(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
                charge_spin,
            )
            return _translate_energy_keys(
                model_ret,
                do_grad_r=do_grad_r,
                do_grad_c=do_grad_c,
                do_atomic_virial=do_atomic_virial,
                local=False,
            )

        return make_fx(fn, **make_fx_kwargs)(
            extended_coord, extended_atype, nlist, mapping, fparam, aparam, charge_spin
        )

    def forward_lower_graph_exportable(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace ``forward_common_lower_graph`` into an exportable module with
        public output keys.

        Delegates to ``forward_common_lower_graph_exportable`` for tracing,
        then translates the internal keys to the ``forward_lower`` convention.

        Parameters
        ----------
        atype
            (N,) flat local atom types, ``N == sum(n_node)``.
        n_node
            (nf,) per-frame local atom counts.
        edge_index
            (2, E) ``[src, dst]`` edge endpoints (flat local indices).
        edge_vec
            (E, 3) neighbor-minus-center edge vectors (sample for tracing).
        edge_mask
            (E,) valid-edge mask (sample for tracing).
        fparam, aparam, do_atomic_virial, charge_spin
            As in ``forward_lower``.
        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx``
            (e.g. ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts
            ``(atype, n_node, edge_index, edge_vec, edge_mask,
            fparam, aparam, charge_spin)`` and returns a dict with the
            public keys: ``atom_energy``, ``energy``, ``force``,
            ``virial``, ``atom_virial`` (the last only when
            ``do_atomic_virial``).  Unlike the dense
            :meth:`forward_lower_exportable` (which emits ``extended_force`` /
            ``extended_virial`` over the ghost-padded extended region), the
            graph path is LOCAL-only (``N == sum(n_node)`` nodes, no ghosts),
            so it emits ``force`` / ``atom_virial`` directly.
        """
        traced = self.forward_common_lower_graph_exportable(
            atype,
            n_node,
            edge_index,
            edge_vec,
            edge_mask,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
            **make_fx_kwargs,
        )

        # Translate internal keys to public convention.
        # Capture model config at trace time via closures.
        do_grad_r = self.do_grad_r("energy")
        do_grad_c = self.do_grad_c("energy")

        def fn(
            atype: torch.Tensor,
            n_node: torch.Tensor,
            edge_index: torch.Tensor,
            edge_vec: torch.Tensor,
            edge_mask: torch.Tensor,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
            charge_spin: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            model_ret = traced(
                atype,
                n_node,
                edge_index,
                edge_vec,
                edge_mask,
                fparam,
                aparam,
                charge_spin,
            )
            return _translate_energy_keys(
                model_ret,
                do_grad_r=do_grad_r,
                do_grad_c=do_grad_c,
                do_atomic_virial=do_atomic_virial,
                local=True,
            )

        return make_fx(fn, **make_fx_kwargs)(
            atype, n_node, edge_index, edge_vec, edge_mask, fparam, aparam, charge_spin
        )

    def forward_lower_graph_exportable_with_comm(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
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
        do_atomic_virial: bool = False,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace ``forward_common_lower_graph`` with comm_dict tensors as
        additional positional inputs -- the with-comm counterpart of
        :meth:`forward_lower_graph_exportable` for message-passing graph
        descriptors (dpa2's repformer block drives cross-rank halo refresh
        via ``deepmd_export::border_op``, see
        :meth:`~deepmd.pt_expt.descriptor.repformers.
        DescrptBlockRepformers._exchange_ghosts_graph`).

        Mirrors the dense ``forward_common_lower_exportable_with_comm``
        (``pt_expt/model/make_model.py``): packs the 8 trailing positional
        comm tensors into a ``comm_dict`` inside the traced function. Also
        derives ``n_local`` (the per-frame OWNED node count, reshaped to
        ``(1,)``; single-frame -- LAMMPS always drives inference with
        ``nf=1``) from the scalar ``nlocal`` tensor, so the differentiated
        reduction excludes halo (not-owned) nodes (see
        :meth:`forward_common_lower_graph`'s ``n_local`` parameter). Unlike
        the plain-graph export path (which traces
        ``forward_common_lower_graph_exportable`` and then wraps a SECOND
        make_fx trace around the key-translation closure), this method
        traces ONCE: the comm-dict packing, ``n_local`` derivation, the
        ``forward_common_lower_graph`` call and the key translation all live
        in a single traced ``fn`` -- following the dense with-comm
        precedent, which is also a single trace.

        Parameters
        ----------
        atype, n_node, edge_index, edge_vec, edge_mask, fparam, aparam, charge_spin, do_atomic_virial
            As in :meth:`forward_lower_graph_exportable`.
        send_list, send_proc, recv_proc, send_num, recv_num, communicator, nlocal, nghost
            The 8 comm tensors (see ``_make_comm_sample_inputs`` in
            ``serialization.py``), packed into ``comm_dict`` inside the
            traced function.

            Runtime device contract (asymmetric, unlike the dense with-comm
            artifact where all 8 stay on CPU): ``nlocal`` and ``nghost``
            must live ON THE MODEL DEVICE, because ``nlocal`` is consumed
            IN-GRAPH here (the ``n_local`` derivation below becomes a
            device kernel after ``move_to_device_pass``; a CPU tensor
            fed to it is read as a device pointer -- CUDA illegal memory
            access).  The other six are consumed only by the opaque
            ``border_op`` whose host code dereferences their ``data_ptr``
            (``send_list`` carries raw host pointers), so they must stay
            on CPU.  The C++ ``run_model_graph_with_comm`` implements
            this placement.
        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx``
            (e.g. ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts ``(atype, n_node,
            edge_index, edge_vec, edge_mask, fparam, aparam, charge_spin,
            send_list, send_proc, recv_proc, send_num, recv_num,
            communicator, nlocal, nghost)`` and returns a dict with the
            SAME public keys as :meth:`forward_lower_graph_exportable`
            (``atom_energy``, ``energy``, ``force``, ``virial``,
            ``atom_virial`` when ``do_atomic_virial``).
        """
        model = self
        do_grad_r = self.do_grad_r("energy")
        do_grad_c = self.do_grad_c("energy")

        def fn(
            atype: torch.Tensor,
            n_node: torch.Tensor,
            edge_index: torch.Tensor,
            edge_vec: torch.Tensor,
            edge_mask: torch.Tensor,
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
        ) -> dict[str, torch.Tensor]:
            comm_dict = {
                "send_list": send_list,
                "send_proc": send_proc,
                "recv_proc": recv_proc,
                "send_num": send_num,
                "recv_num": recv_num,
                "communicator": communicator,
                "nlocal": nlocal,
                "nghost": nghost,
            }
            # nlocal is a scalar int32 tensor; reshape to the (nf,) == (1,)
            # shape forward_common_lower_graph's n_local expects, cast to
            # n_node's dtype (node_ownership_mask compares directly against
            # n_node-derived indices).
            n_local = nlocal.reshape(1).to(n_node.dtype)
            model_ret = model.forward_common_lower_graph(
                atype,
                n_node,
                edge_index,
                edge_vec,
                edge_mask,
                do_atomic_virial=do_atomic_virial,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                n_local=n_local,
                comm_dict=comm_dict,
            )
            return _translate_energy_keys(
                model_ret,
                do_grad_r=do_grad_r,
                do_grad_c=do_grad_c,
                do_atomic_virial=do_atomic_virial,
                local=True,
            )

        return make_fx(fn, **make_fx_kwargs)(
            atype,
            n_node,
            edge_index,
            edge_vec,
            edge_mask,
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
