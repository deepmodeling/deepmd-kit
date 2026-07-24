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
)
from deepmd.pt_expt.model.model import (
    BaseModel,
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
      positional ABI (``spin`` at index 10) over the inherited
      ``forward_common_lower_graph_exportable``.
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
            Frame-level charge/spin conditioning, shape nf x 2. Accepted for
            call-signature compatibility with ``ModelWrapper.forward`` (which
            always forwards this keyword); charge-spin FiLM combined with
            native spin is rejected at construction time
            (``add_chg_spin_ebd`` on the descriptor), so this is always
            ``None`` in practice for a model this class can build.

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
        There is NO with-comm variant (single-rank only; multi-rank
        graph-spin is a follow-up).

        Two-layer make_fx trace, mirroring
        :meth:`~deepmd.pt_expt.model.ener_model.EnergyModel.forward_lower_graph_exportable`:
        the inner layer
        (the inherited
        :meth:`~deepmd.pt_expt.model.make_model.make_model.forward_common_lower_graph_exportable`)
        traces ``forward_common_lower_graph``
        with ``spin`` as a SECOND autograd leaf next to ``edge_vec``
        (``charge_spin`` is a frame-level conditioning input, not a leaf);
        this outer layer re-traces with the PUBLIC positional ABI above and
        translates the internal fitting keys to the public output keys via
        the shared :meth:`_translate_eager_call` (the same single owner as
        the eager :meth:`forward`, so ``mask_mag`` is emitted here too).

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
            fparam, aparam, charge_spin)`` and returns a dict with the
            public keys ``atom_energy``, ``energy``, ``force``,
            ``force_mag``, ``virial``, ``mask_mag``, and (when
            ``do_atomic_virial``) ``atom_virial``.
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
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
            spin=spin,
            destination_sorted=destination_sorted,
            **make_fx_kwargs,
        )

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
                spin,
            )
            # Same single-owner translation as the eager forward, so the
            # exported .pt2 emits mask_mag too and consumers read it.
            return self._translate_eager_call(
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
        )
