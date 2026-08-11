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
    _translate_energy_keys,
    make_model,
)
from .model import (
    BaseModel,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel, T_Bases=(BaseModel,))


@BaseModel.register("ener")
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

    def _to_public_keys(
        self,
        model_ret: dict[str, torch.Tensor],
        do_atomic_virial: bool,
    ) -> dict[str, torch.Tensor]:
        """Rename one ``call_common`` result to the public energy-model keys.

        The renaming is a property of the output definition, not of the node
        axis, so it serves the rectangular and the ragged entry alike.

        Parameters
        ----------
        model_ret : dict[str, torch.Tensor]
            A ``call_common`` result, in internal ``<var>_<derivative>`` keys.
        do_atomic_virial : bool
            Whether the per-atom virial was requested and should be carried.

        Returns
        -------
        dict[str, torch.Tensor]
            The same tensors under the public names.
        """
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        for key in ("mask", "n_node"):
            if key in model_ret:
                model_predict[key] = model_ret[key]
        if self.atomic_output_def()["energy"].r_hessian:
            model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-3)
        return model_predict

    def _translate_eager_call(
        self,
        model_ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Translate internal energy outputs at the public model boundary.

        Parameters
        ----------
        model_ret : dict[str, torch.Tensor]
            Result returned by a ``call_common`` entry.
        atype : torch.Tensor
            Atom types on the same node axis as the atomic outputs.
        do_atomic_virial : bool, default: False
            Whether the per-atom virial was requested.

        Returns
        -------
        dict[str, torch.Tensor]
            Public model outputs.

        Notes
        -----
        Native-spin models override this translation to add ``force_mag`` and
        ``mask_mag``. Keeping the dispatch here lets rectangular and ragged
        forwards share one implementation without duplicating a spin-specific
        ``forward_ragged``.
        """
        del atype
        return self._to_public_keys(model_ret, do_atomic_virial)

    def forward_lower_canonical_graph(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor,
        source: torch.Tensor,
        edge_vec: torch.Tensor,
        destination_row_ptr: torch.Tensor,
        source_row_ptr: torch.Tensor,
        source_order: torch.Tensor,
        *,
        do_atomic_virial: bool,
        spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate an eligible compressed canonical deployment graph.

        Parameters
        ----------
        atype
            Flat local-plus-halo atom types with shape ``(N,)``, int64.
        n_node
            Per-frame total node counts with shape ``(nf,)``, int64.
        n_local
            Per-frame owned node counts with shape ``(nf,)``, int64.
        source
            Source-node indices with shape ``(S,)``, uint32.
        edge_vec
            Destination-major edge vectors with shape ``(S, 3)``, float32.
        destination_row_ptr
            Destination CSR offsets with shape ``(N + 1,)``, int64.
        source_row_ptr
            Source CSR offsets with shape ``(N + 1,)``, int64.
        source_order
            Source-grouped edge positions with shape ``(S,)`` and the same
            dtype as ``source``.
        do_atomic_virial
            Whether to return the per-node virial.
        spin
            Per-node magnetic moments with shape ``(N, 3)`` for a native-spin
            model, or ``None``. Ghost rows carry their owner's moment.

        Returns
        -------
        dict[str, torch.Tensor]
            Public energy-model outputs on the flat node axis.
        """
        from deepmd.pt_expt.kernels.cuda.dpa1.canonical import (
            canonical_model_eligible as dpa1_canonical_eligible,
        )
        from deepmd.pt_expt.kernels.cuda.dpa1.canonical import (
            dpa1_canonical_compress_energy_force,
        )
        from deepmd.pt_expt.kernels.cuda.dpa4c.canonical import (
            canonical_model_eligible as dpa4c_canonical_eligible,
        )
        from deepmd.pt_expt.kernels.cuda.dpa4c.canonical import (
            dpa4c_canonical_compress_energy_force,
        )
        from deepmd.pt_expt.utils.canonical_graph import (
            CanonicalGraph,
            validate_canonical_graph_shapes,
        )

        use_dpa4c = dpa4c_canonical_eligible(self)
        if not use_dpa4c and not dpa1_canonical_eligible(self):
            raise ValueError("model is not eligible for compact canonical deployment")
        graph = CanonicalGraph(
            n_node=n_node,
            n_local=n_local,
            source=source,
            edge_vec=edge_vec,
            destination_row_ptr=destination_row_ptr,
            source_row_ptr=source_row_ptr,
            source_order=source_order,
        )
        validate_canonical_graph_shapes(graph, atype.shape[0])
        atype, output_mask = self.atomic_model._prepare_graph_nodes(
            n_node,
            n_local,
            atype,
            edge_vec,
        )
        descriptor = self.atomic_model.descriptor
        fitting = self.atomic_model.fitting_net
        atom_bias = fitting.bias_atom_e[:, 0] + self.atomic_model.out_bias[0, :, 0]
        if use_dpa4c:
            energy, atom_energy, force, virial, atom_virial, force_mag = (
                dpa4c_canonical_compress_energy_force(
                    descriptor,
                    fitting,
                    graph,
                    atype,
                    output_mask,
                    atom_bias,
                    do_atomic_virial,
                    spin,
                )
            )
        else:
            force_mag = None
            energy, atom_energy, force, virial, atom_virial = (
                dpa1_canonical_compress_energy_force(
                    descriptor,
                    fitting,
                    graph,
                    atype,
                    # Descriptor-owned hook: the single owner of the
                    # graph-route type-embedding table.
                    descriptor.graph_type_embedding_table(),
                    output_mask,
                    atom_bias,
                    do_atomic_virial,
                )
            )
        result = {
            "atom_energy": atom_energy,
            "energy": energy,
            "force": force,
            "virial": virial,
            "mask": output_mask.to(torch.int32),
        }
        if spin is not None:
            if force_mag is None or force_mag.ndim != 2:
                raise RuntimeError(
                    "canonical native-spin inference did not return a "
                    "per-node magnetic force"
                )
            result["force_mag"] = force_mag
        elif force_mag is not None and force_mag.ndim == 2:
            result["force_mag"] = force_mag
        if do_atomic_virial:
            result["atom_virial"] = atom_virial
        return result

    def forward_lower_canonical_graph_exportable(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor,
        source: torch.Tensor,
        edge_vec: torch.Tensor,
        destination_row_ptr: torch.Tensor,
        source_row_ptr: torch.Tensor,
        source_order: torch.Tensor,
        *,
        do_atomic_virial: bool,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace the compact canonical deployment forward with ``make_fx``.

        Parameters
        ----------
        atype, n_node, n_local, source, edge_vec
            Compact graph node and edge tensors.
        destination_row_ptr, source_row_ptr, source_order
            Compact dual-CSR topology.
        do_atomic_virial
            Whether the traced output includes per-node virial.
        **make_fx_kwargs
            Additional arguments passed to :func:`make_fx`.

        Returns
        -------
        torch.nn.Module
            Traced eight-input compact deployment module.
        """
        model = self

        def fn(
            atype: torch.Tensor,
            n_node: torch.Tensor,
            n_local: torch.Tensor,
            source: torch.Tensor,
            edge_vec: torch.Tensor,
            destination_row_ptr: torch.Tensor,
            source_row_ptr: torch.Tensor,
            source_order: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return model.forward_lower_canonical_graph(
                atype,
                n_node,
                n_local,
                source,
                edge_vec,
                destination_row_ptr,
                source_row_ptr,
                source_order,
                do_atomic_virial=do_atomic_virial,
            )

        return make_fx(fn, **make_fx_kwargs)(
            atype,
            n_node,
            n_local,
            source,
            edge_vec,
            destination_row_ptr,
            source_row_ptr,
            source_order,
        )

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
        coord
            Atomic coordinates.
        atype
            Atomic type indices.
        box
            Simulation-cell vectors, or ``None`` for a non-periodic system.
        fparam
            Optional frame parameters.
        aparam
            Optional atomic parameters.
        do_atomic_virial
            Whether to return per-atom virials.
        charge_spin
            Optional frame-level charge and spin conditioning.
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
        return self._translate_eager_call(
            model_ret,
            atype,
            do_atomic_virial=do_atomic_virial,
        )

    def forward_ragged(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the energy model over a batch whose node axis is flat.

        The counterpart of :meth:`forward` for frames held concatenated rather
        than padded to a common atom count. Arguments and results share their
        meaning with :meth:`call_common_ragged`, whose per-atom outputs keep
        the flat axis; only the keys are the public ones.

        Parameters
        ----------
        coord : torch.Tensor
            Local coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
        atype : torch.Tensor
            Local atom types with shape ``(N,)``.
        n_node : torch.Tensor
            Atoms per frame with shape ``(nf,)``.
        box : torch.Tensor or None, optional
            Simulation cell with shape ``(nf, 3, 3)``.
        fparam : torch.Tensor or None, optional
            Frame parameters with shape ``(nf, ndf)``.
        aparam : torch.Tensor or None, optional
            Atomic parameters with shape ``(N, nda)``.
        do_atomic_virial : bool, default: False
            Whether to return per-atom virials.
        charge_spin : torch.Tensor or None, optional
            Frame-level charge and spin conditioning with shape ``(nf, 2)``.
        spin : torch.Tensor or None, optional
            Per-atom native spin with shape ``(N, 3)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Public energy-model keys; per-atom entries have leading dimension
            ``N`` and per-frame entries ``nf``.
        """
        model_ret = self.call_common_ragged(
            coord,
            atype,
            n_node,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
            spin=spin,
        )
        return self._translate_eager_call(
            model_ret,
            atype,
            do_atomic_virial=do_atomic_virial,
        )

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
        extended_coord
            Extended-coordinate sample used for tracing.
        extended_atype
            Extended atom-type sample used for tracing.
        nlist
            Neighbor-list sample used for tracing.
        mapping
            Extended-to-local mapping sample used for tracing.
        fparam
            Optional frame-parameter sample.
        aparam
            Optional atomic-parameter sample.
        do_atomic_virial
            Whether the traced module returns per-atom virials.
        charge_spin
            Optional charge/spin conditioning sample.
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
        n_local: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        destination_order: torch.Tensor,
        destination_row_ptr: torch.Tensor,
        source_order: torch.Tensor,
        source_row_ptr: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        destination_sorted: bool = False,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace ``forward_common_lower_graph`` into an exportable module with
        public output keys.

        Delegates to ``forward_common_lower_graph_exportable`` for tracing,
        then translates the internal keys to the ``forward_lower`` convention.

        Parameters
        ----------
        atype
            (N,) flat local-plus-halo atom types, ``N == sum(n_node)``.
        n_node
            (nf,) per-frame total node counts.
        n_local
            (nf,) per-frame owned node counts.
        edge_index
            (2, E) ``[src, dst]`` edge endpoints (flat local indices).
        edge_vec
            (E, 3) neighbor-minus-center edge vectors (sample for tracing).
        edge_mask
            (E,) valid-edge mask (sample for tracing).
        destination_order
            (E,) destination-grouped edge permutation.
        source_order
            (E,) source-grouped edge permutation.
        destination_row_ptr, source_row_ptr
            (N + 1,) destination/source CSR offsets.
        destination_sorted
            Static export-time assertion that the payload is destination-major
            and ``destination_order`` is identity.
        fparam, aparam, do_atomic_virial, charge_spin
            As in ``forward_lower``.
        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx``
            (e.g. ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts
            ``(atype, n_node, n_local, edge_index, edge_vec, edge_mask,
            destination_order, destination_row_ptr, source_order,
            source_row_ptr, fparam, aparam, charge_spin)`` and returns a dict
            with the public keys: ``atom_energy``, ``energy``, ``force``,
            ``virial``, ``atom_virial`` (the last only when
            ``do_atomic_virial``). Unlike the dense
            :meth:`forward_lower_exportable` (which emits ``extended_force`` /
            ``extended_virial``), the graph path emits ``force`` and
            ``atom_virial`` directly on the local-plus-halo node axis.
        """
        traced = self.forward_common_lower_graph_exportable(
            atype,
            n_node,
            n_local,
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
            destination_sorted=destination_sorted,
            **make_fx_kwargs,
        )

        # Translate internal keys to public convention.
        # Capture model config at trace time via closures.
        do_grad_r = self.do_grad_r("energy")
        do_grad_c = self.do_grad_c("energy")

        def fn(
            atype: torch.Tensor,
            n_node: torch.Tensor,
            n_local: torch.Tensor,
            edge_index: torch.Tensor,
            edge_vec: torch.Tensor,
            edge_mask: torch.Tensor,
            destination_order: torch.Tensor,
            destination_row_ptr: torch.Tensor,
            source_order: torch.Tensor,
            source_row_ptr: torch.Tensor,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
            charge_spin: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            model_ret = traced(
                atype,
                n_node,
                n_local,
                edge_index,
                edge_vec,
                edge_mask,
                destination_order,
                destination_row_ptr,
                source_order,
                source_row_ptr,
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
            atype,
            n_node,
            n_local,
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            fparam,
            aparam,
            charge_spin,
        )

    def forward_lower_graph_exportable_with_comm(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        destination_order: torch.Tensor,
        destination_row_ptr: torch.Tensor,
        source_order: torch.Tensor,
        source_row_ptr: torch.Tensor,
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
        descriptors (dpa2's repformer block drives cross-rank ghost refresh
        via ``deepmd_export::border_op``, see
        :meth:`~deepmd.pt_expt.descriptor.repformers.
        DescrptBlockRepformers._exchange_ghosts_graph`).

        Mirrors the dense ``forward_common_lower_exportable_with_comm``
        (``pt_expt/model/make_model.py``): packs the 8 trailing positional
        comm tensors into a ``comm_dict`` inside the traced function. Also
        derives ``n_local`` (the per-frame OWNED node count, reshaped to
        ``(1,)``; single-frame -- LAMMPS always drives inference with
        ``nf=1``) from the scalar ``nlocal`` tensor, so the differentiated
        reduction excludes ghost (not-owned) nodes (see
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

        destination_order
            Destination-major edge permutation used by fused graph operators.

        destination_row_ptr
            Destination CSR row pointers.

        source_order
            Source-major edge permutation used by force assembly.

        source_row_ptr
            Source CSR row pointers.

        send_list, send_proc, recv_proc, send_num, recv_num, communicator, nlocal, nghost
            The 8 comm tensors (see ``_make_comm_sample_inputs`` in
            ``serialization.py``), packed into ``comm_dict`` inside the
            traced function.

            Runtime device contract: ALL 8 stay on CPU, symmetric with
            the dense with-comm artifact -- they are consumed only by the
            opaque ``border_op`` whose HOST code dereferences their
            ``data_ptr`` (``send_list`` carries raw host pointers) and
            reads ``nlocal``/``nghost`` via cheap host ``.item()`` calls.
            Deriving the in-graph owned count from a device-placed
            ``nlocal`` instead (the previous design) made every per-layer
            ``border_op`` forward AND custom backward pull the scalars
            back with synchronizing D2H reads (``4 * nlayers`` per MD
            step).  The C++ ``run_model_graph_with_comm`` implements this
            placement.

        n_local
            (1,) int64 ON THE MODEL DEVICE: the per-frame OWNED node
            count consumed IN-GRAPH by the owned-node energy mask (it
            becomes a device kernel operand after
            ``move_to_device_pass``, like ``n_node``; a CPU tensor fed
            there is read as a device pointer -- CUDA illegal memory
            access).  Carries the same value as the ``nlocal`` comm
            tensor; the two inputs exist precisely to separate the
            device-compute role from the host-MPI-control role.

        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx``
            (e.g. ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts ``(atype, n_node,
            n_local, edge_index, edge_vec, edge_mask, destination_order,
            destination_row_ptr, source_order, source_row_ptr, fparam,
            aparam, charge_spin, send_list, send_proc, recv_proc, send_num,
            recv_num, communicator, nlocal, nghost)`` and returns a dict with the
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
            n_local: torch.Tensor,
            edge_index: torch.Tensor,
            edge_vec: torch.Tensor,
            edge_mask: torch.Tensor,
            destination_order: torch.Tensor,
            destination_row_ptr: torch.Tensor,
            source_order: torch.Tensor,
            source_row_ptr: torch.Tensor,
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
            # ``n_local`` (slot 2, DEVICE) is the owned-count input consumed
            # by the in-graph owned-node mask; the CPU ``nlocal`` comm
            # tensor is host control metadata for border_op only.
            model_ret = model.forward_common_lower_graph(
                atype,
                n_node,
                n_local,
                edge_index,
                edge_vec,
                edge_mask,
                destination_order,
                destination_row_ptr,
                source_order,
                source_row_ptr,
                destination_sorted=True,
                do_atomic_virial=do_atomic_virial,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
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
            n_local,
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
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
