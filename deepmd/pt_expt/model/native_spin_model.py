# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt native-spin energy model (NeighborGraph route, autograd force_mag)."""

from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.model.native_spin_model import (
    make_native_spin_model,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
    trace_named_lower,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.utils.canonical_graph import (
    BorderPlan,
)
from deepmd.pt_expt.utils.graph_csr import (
    validate_graph_csr_for_export,
)


@BaseModel.register("native_spin")
class NativeSpinEnergyModel(make_native_spin_model(EnergyModel)):
    """pt_expt native-spin energy model.

    ``make_native_spin_model`` applied to THIS backend's
    :class:`~deepmd.pt_expt.model.ener_model.EnergyModel` (construction,
    output defs, serialization all come from the factory; the deserialize
    closure rebuilds through the pt_expt model class, so a registry round
    trip yields a real ``torch.nn.Module``), plus two torch-specific
    overrides:

    - :meth:`forward`: the pt_expt ``call_common`` produces REAL autograd
      ``energy_derv_r``/``energy_derv_r_mag``/``energy_derv_c_redu``
      tensors, unlike the dpmodel factory's energy-only ``call`` (which is
      restricted to ``force``/``force_mag``/``virial`` as ``None``
      placeholders because dpmodel has no autograd).
    - :meth:`forward_lower_graph_exportable`: the graph-spin ``.pt2``
      positional ABI (``spin`` at index 10, the border plan of further
      message-passing blocks last) over ``forward_common_lower_graph``.
    """

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
        """Return native-spin model predictions with public output keys.

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
            If calculate the atomic virial.
        charge_spin
            Frame-level charge/spin conditioning, shape nf x 2. Forwarded to
            the backbone like every other conditioning input; native spin and
            charge-spin FiLM combine freely (DPA4 declares both
            capabilities).

        Returns
        -------
        ret_dict
            The result dict with keys ``atom_energy``, ``energy``,
            ``force``, ``force_mag``, ``virial``, ``mask_mag``, and
            (when ``do_atomic_virial``) ``atom_virial``. ``force`` and
            ``force_mag`` are real autograd tensors (``-dE/dcoord`` and
            ``-dE/dspin``), NOT placeholders.
        """
        # Default neighbor_graph_method: pt_expt's default-flip picks the fast
        # graph builder (dpmodel's call forces the O(N^2) dense one).
        model_ret = self.call_common(
            coord,
            atype,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
            spin=spin,
        )
        # Same shared translation as dpmodel's call, on real autograd tensors.
        return self._translate_eager_call(
            model_ret, atype, do_atomic_virial=do_atomic_virial
        )

    def forward_ragged(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        spin: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return native-spin predictions on a flat, frame-segmented node axis."""
        model_ret = self.call_common_ragged(
            coord,
            atype,
            n_node,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
            spin=spin,
        )
        return self._translate_eager_call(
            model_ret, atype, do_atomic_virial=do_atomic_virial
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
        spin: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        communicator: torch.Tensor | None = None,
        halo_send: torch.Tensor | None = None,
        halo_swaps: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        destination_sorted: bool = False,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace the graph-spin lower into an exportable module.

        THIS METHOD OWNS the positional ``.pt2`` ABI for graph-spin models
        (mirrored verbatim by the C++ / serialization seams): ``spin`` sits
        at index 10, right after the shared ``NeighborGraph`` CSR block and
        before the conditional ``fparam``/``aparam`` tail; the conditional
        ``charge_spin`` tail follows at index 13 (combined native-spin +
        charge-spin FiLM models, review 3638047227; ``None`` otherwise).
        A model with further message-passing blocks takes the communicator
        and the border plan of a decomposed frame last, as the compact
        canonical lower does, so that one program evaluates a frame held
        whole (a plan without swaps) and a domain of a decomposed one.

        One make_fx trace holds the graph forward with ``spin`` as a SECOND
        autograd leaf next to ``edge_vec`` (``charge_spin`` is a frame-level
        conditioning input, not a leaf), the refresh of the halo rows of every
        further block through the border plan, whose registered backward
        returns the gradients of the halo rows to their owners, and the
        translation of the internal fitting keys to the public output keys by
        the shared :meth:`_translate_eager_call` (the same single owner as the
        eager :meth:`forward`, so ``mask_mag`` is emitted here too). ``spin``
        is the extended per-node moment: halo rows carry their owner's moment,
        which the LAMMPS ``sp`` forward communication delivers, so the moment
        needs no exchange of its own.

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
        destination_row_ptr
            (N + 1,) destination CSR offsets.
        source_order
            (E,) source-grouped edge permutation.
        source_row_ptr
            (N + 1,) source CSR offsets.
        spin
            (N, 3) per-node native spin (sample for tracing). ALWAYS present
            in this ABI, unlike the energy model's optional ``charge_spin``.
        fparam
            Frame parameter, ``(nf, ndf)``, or ``None`` when
            ``dim_fparam == 0``.
        aparam
            Atomic parameter, ``(N, nda)``, or ``None`` when
            ``dim_aparam == 0``.
        charge_spin
            Frame-level charge/spin FiLM conditioning, ``(nf,
            dim_chg_spin)``, or ``None`` when the descriptor has no
            ``add_chg_spin_ebd`` (conditional tail, slot 13).
        communicator, halo_send, halo_swaps
            For a model with further message-passing blocks, a sample
            communicator and border plan (see
            :class:`~deepmd.pt_expt.utils.canonical_graph.BorderPlan`), given
            together; ``None`` otherwise.
        do_atomic_virial
            Whether to also return ``atom_virial``.
        destination_sorted
            Static export-time assertion that the payload is
            destination-major and ``destination_order`` is identity.
        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx`` (e.g.
            ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts ``(atype, n_node,
            n_local, edge_index, edge_vec, edge_mask, destination_order,
            destination_row_ptr, source_order, source_row_ptr, spin,
            fparam, aparam, charge_spin)``, followed by ``(communicator,
            halo_send, halo_swaps)`` for a model with further
            message-passing blocks, and returns a dict with the public keys
            ``atom_energy``, ``energy``, ``force``, ``force_mag``,
            ``virial``, ``mask_mag``, and (when ``do_atomic_virial``)
            ``atom_virial``.
        """
        border = (communicator, halo_send, halo_swaps)
        reads_border = any(value is not None for value in border)
        if reads_border and any(value is None for value in border):
            raise ValueError(
                "the graph-spin lower takes the communicator and the border "
                "plan of further message-passing blocks together"
            )
        validate_graph_csr_for_export(
            edge_index,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            atype.shape[0],
            destination_sorted=destination_sorted,
        )
        model = self

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
            spin: torch.Tensor,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
            charge_spin: torch.Tensor | None,
            *plan: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            # ``plan`` holds the communicator, the send list and the swap
            # table of a model with further message-passing blocks.
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
                destination_sorted=destination_sorted,
                do_atomic_virial=do_atomic_virial,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                spin=spin,
                comm_dict=(
                    BorderPlan(send_nodes=plan[1], swaps=plan[2], communicator=plan[0])
                    if plan
                    else None
                ),
            )
            # Same single-owner translation as the eager forward, so the
            # exported .pt2 emits mask_mag too and consumers read it.
            return model._translate_eager_call(
                model_ret, atype, do_atomic_virial=do_atomic_virial
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
            spin,
            fparam,
            aparam,
            charge_spin,
            *(border if reads_border else ()),
        )

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
        spin: torch.Tensor,
        charge_spin: torch.Tensor | None = None,
        communicator: torch.Tensor | None = None,
        halo_send: torch.Tensor | None = None,
        halo_swaps: torch.Tensor | None = None,
        *,
        do_atomic_virial: bool,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace the compact canonical spin lower into an exportable module.

        THIS METHOD OWNS the positional ``.pt2`` ABI for compact canonical
        spin models (mirrored verbatim by the C++ / serialization seams):
        ``spin`` sits at index 8, directly after the dual-CSR block, which is
        the same placement rule the graph ABI uses at index 10. The
        charge/spin condition of a model that reads it as an input follows,
        then the communicator and the border plan of a model with further
        message-passing blocks.

        A compressed DPA4C model returns the magnetic force as a value: the
        fused operator closes the on-site part in its node kernel and
        accumulates the neighbour part of every edge onto the edge's source
        node before it returns. Any other model differentiates its graph
        forward with respect to the moment. ``mask_mag`` is derived from the
        types by the same single owner the eager translation uses.

        Parameters
        ----------
        atype, n_node, n_local, source, edge_vec
            Compact graph node and edge tensors.
        destination_row_ptr, source_row_ptr, source_order
            Compact dual-CSR topology.
        spin
            ``(N, 3)`` per-node moment. Ghost rows carry their owner's moment.
        charge_spin
            For a model that reads its charge/spin condition as an input, a
            sample condition with shape ``(nf, dim_chg_spin)``, float64;
            ``None`` otherwise.
        communicator, halo_send, halo_swaps
            For a model with further message-passing blocks, a sample
            communicator and border plan, given together; ``None`` otherwise.
        do_atomic_virial
            Whether the traced output includes per-node virial.
        **make_fx_kwargs
            Additional arguments passed to :func:`make_fx`.

        Returns
        -------
        torch.nn.Module
            Traced compact deployment module of the nine graph and moment
            inputs, followed by those of ``charge_spin``, ``communicator``,
            ``halo_send`` and ``halo_swaps`` that the model reads, whose
            output dict adds ``force_mag`` and ``mask_mag`` to the spin-free
            key set.
        """
        model = self
        border = (communicator, halo_send, halo_swaps)
        if any(value is None for value in border) and any(
            value is not None for value in border
        ):
            raise ValueError(
                "the compact spin lower takes the communicator and the border "
                "plan of further message-passing blocks together"
            )

        def lower(**inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            result = model.forward_lower_canonical_graph(
                **inputs, do_atomic_virial=do_atomic_virial
            )
            result["mask_mag"] = model._spin_active_mask(inputs["atype"])
            return result

        inputs = {
            "atype": atype,
            "n_node": n_node,
            "n_local": n_local,
            "source": source,
            "edge_vec": edge_vec,
            "destination_row_ptr": destination_row_ptr,
            "source_row_ptr": source_row_ptr,
            "source_order": source_order,
            "spin": spin,
            "charge_spin": charge_spin,
            "communicator": communicator,
            "halo_send": halo_send,
            "halo_swaps": halo_swaps,
        }
        return trace_named_lower(
            lower,
            {name: value for name, value in inputs.items() if value is not None},
            **make_fx_kwargs,
        )
