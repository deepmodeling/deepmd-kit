# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity: graph lower (forward_common_lower_graph) vs legacy dense lower.

Builds a same-weights pt_expt dpa1(attn_layer=0) EnergyModel and a small
extended system, then compares the graph-native lower (energy/force/virial/
atom_virial assembled from ``edge_energy_deriv``) against the legacy dense
``forward_common_lower`` on the SAME neighbor set (the graph is built REGIME-1
from the same extended quartet via ``from_dense_quartet``).

The graph lower is inherently LOCAL (ghost-free): its force/atom_virial live on
``nloc`` nodes, while the legacy lower returns EXTENDED (``nall``) force/
atom_virial.  The two are reconciled by folding the legacy extended force/
atom_virial onto local atoms via ``mapping`` (a scatter-add on the atom axis,
identical to ``communicate_extended_output``).  Energy, reduced energy and the
reduced (per-frame) virial are frame/local quantities and compare directly.
"""

import copy

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    apply_pair_exclusion,
    from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    apply_pair_exclusion_nlist,
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt_expt.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt_expt.fitting import (
    InvarFitting,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)

from ...seed import (
    GLOBAL_SEED,
)


def _fold_extended_to_local(
    ext: torch.Tensor, mapping: torch.Tensor, nloc: int
) -> torch.Tensor:
    """Scatter-add an extended (nf, nall, 1, K) tensor onto local atoms.

    Mirrors ``communicate_extended_output``: ``local[mapping[j]] += ext[j]``
    along the atom axis (dim 1).
    """
    nf, nall = mapping.shape
    K = ext.shape[-1]
    out = torch.zeros(nf, nloc, 1, K, dtype=ext.dtype, device=ext.device)
    idx = mapping.view(nf, nall, 1, 1).expand(nf, nall, 1, K)
    out.scatter_add_(1, idx, ext)
    return out


class TestDpa1GraphLower:
    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.natoms = 5
        self.rcut = 4.0
        self.rcut_smth = 0.5
        self.sel = 20  # mixed-type single int sel
        self.nt = 2
        self.type_map = ["foo", "bar"]

        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=self.device)
        self.cell = cell.unsqueeze(0)  # [1, 3, 3]
        coord = torch.rand(
            [self.natoms, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0).to(self.device)  # [1, natoms, 3]
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1]], dtype=torch.int64, device=self.device
        )

    def _make_model(
        self,
        attn_layer: int = 0,
        smooth: bool = False,
        pair_excl_types: list | None = None,
        descr_excl_types: list | None = None,
    ) -> EnergyModel:
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            self.sel,
            self.nt,
            neuron=[3, 6],
            axis_neuron=2,
            attn=4,
            attn_layer=attn_layer,
            attn_dotr=True,
            attn_mask=False,
            # Smooth attention keeps sel-padding in the dense softmax
            # denominator; the carry-all graph omits it, so
            # exact graph-vs-dense parity requires smooth=False here.
            smooth_type_embedding=smooth,
            activation_function="tanh",
            set_davg_zero=False,
            type_one_side=True,
            precision="float64",
            exclude_types=descr_excl_types or [],
            seed=GLOBAL_SEED,
        ).to(self.device)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            precision="float64",
            seed=GLOBAL_SEED,
        ).to(self.device)
        return EnergyModel(
            ds,
            ft,
            type_map=self.type_map,
            pair_exclude_types=pair_excl_types or [],
        ).to(self.device)

    def _prepare_lower_inputs(self, periodic: bool):
        """Build extended coords, atype, nlist, mapping as torch tensors."""
        coord_np = self.coord.detach().cpu().numpy()
        atype_np = self.atype.detach().cpu().numpy()
        if periodic:
            cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
            coord_normalized = normalize_coord(
                coord_np.reshape(1, self.natoms, 3),
                cell_np.reshape(1, 3, 3),
            )
            extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
                coord_normalized,
                atype_np,
                cell_np,
                self.rcut,
            )
            nlist = build_neighbor_list(
                extended_coord,
                extended_atype,
                self.natoms,
                self.rcut,
                [self.sel],
                distinguish_types=False,
            )
            extended_coord = extended_coord.reshape(1, -1, 3)
        else:
            extended_coord = coord_np.reshape(1, self.natoms, 3)
            extended_atype = atype_np.reshape(1, self.natoms)
            mapping = np.arange(self.natoms, dtype=np.int64).reshape(1, self.natoms)
            nlist = build_neighbor_list(
                extended_coord,
                extended_atype,
                self.natoms,
                self.rcut,
                [self.sel],
                distinguish_types=False,
            )
        ext_coord = torch.tensor(
            extended_coord, dtype=torch.float64, device=self.device
        )
        ext_atype = torch.tensor(extended_atype, dtype=torch.int64, device=self.device)
        nlist_t = torch.tensor(nlist, dtype=torch.int64, device=self.device)
        mapping_t = torch.tensor(mapping, dtype=torch.int64, device=self.device)
        return ext_coord, ext_atype, nlist_t, mapping_t

    @pytest.mark.parametrize("attn_layer", [0, 2])
    @pytest.mark.parametrize("periodic", [True, False])  # PBC vs non-PBC
    @pytest.mark.parametrize("do_av", [False, True])  # atom-virial off / on
    @pytest.mark.parametrize(
        "excl_types", [[], [(0, 1)]]
    )  # no exclusion / type-0-1 pair exclusion
    def test_force_virial_parity_vs_legacy(
        self, periodic, do_av, attn_layer, excl_types
    ) -> None:
        """Graph lower energy/force/virial/atom_virial == legacy dense lower on
        the SAME neighbor set (regime-1 graph from from_dense_quartet).
        attn_layer=2 exercises graph attention through model-level autograd
        (smooth=False: exact carry-all parity regime, NeighborGraph PR-D).
        Parametrized over exclude_types: empty list (no exclusion) and
        [(0,1)] (model-level pair exclusion applied identically on both routes).
        """
        model = self._make_model(attn_layer=attn_layer, pair_excl_types=excl_types)
        model.eval()
        tol = (
            {"rtol": 1e-12, "atol": 1e-12}
            if self.device.type == "cpu"
            else {"rtol": 1e-10, "atol": 1e-10}
        )
        ext_coord, ext_atype, nlist, mapping = self._prepare_lower_inputs(periodic)
        nf = ext_coord.shape[0]
        nloc = self.natoms

        # Model-level pair_exclude is a nlist-BUILD transform (decision
        # #18/A4): BOTH lowers consume pre-excluded inputs, so fold the
        # exclusion into the dense nlist here (mirrors the C++
        # ``applyPairExclusionNlist`` build step).
        nlist = apply_pair_exclusion_nlist(
            nlist, ext_atype, model.atomic_model.pair_excl
        )
        legacy = model.forward_common_lower(
            ext_coord.clone().requires_grad_(True),
            ext_atype,
            nlist,
            mapping,
            do_atomic_virial=do_av,
        )

        # build the regime-1 graph from the SAME extended quartet.
        # from_dense_quartet is array-API; feed torch tensors so the
        # returned edge_vec is already a torch tensor on env.DEVICE.
        ng = from_dense_quartet(ext_coord, nlist, mapping)
        atype_local = ext_atype[:, :nloc].reshape(nf * nloc)
        # Model-level pair_exclude is a BUILD-time transform (decision #18): the
        # converter does not bake it in and the lower no longer re-applies it, so
        # apply it to the graph here (mirrors the C++ ``applyPairExclusion`` step).
        ng = apply_pair_exclusion(ng, atype_local, model.atomic_model.pair_excl)
        graph = model.forward_common_lower_graph(
            atype_local,
            ng.n_node,
            ng.n_node,
            ng.edge_index,
            ng.edge_vec,
            ng.edge_mask,
            do_atomic_virial=do_av,
        )

        # forward_common_lower_graph returns flat (N = nf * nloc, *) per-atom
        # outputs. Reshape to (nf, nloc, *) to compare against the dense lower.

        # per-atom energy: flat (N, 1) -> (nf, nloc, 1)
        graph_energy = graph["energy"].reshape(nf, nloc, 1)
        torch.testing.assert_close(graph_energy, legacy["energy"], **tol)

        # reduced energy and virial: already per-frame (nf, *)
        torch.testing.assert_close(graph["energy_redu"], legacy["energy_redu"], **tol)
        torch.testing.assert_close(
            graph["energy_derv_c_redu"], legacy["energy_derv_c_redu"], **tol
        )

        # force: graph is flat (N, 1, 3); fold legacy extended (nall) -> local (nloc)
        legacy_force_local = _fold_extended_to_local(
            legacy["energy_derv_r"], mapping, nloc
        )
        graph_force = graph["energy_derv_r"].reshape(nf, nloc, 1, 3)
        torch.testing.assert_close(graph_force, legacy_force_local, **tol)

        if do_av:
            legacy_av_local = _fold_extended_to_local(
                legacy["energy_derv_c"], mapping, nloc
            )
            graph_av = graph["energy_derv_c"].reshape(nf, nloc, 1, 9)
            torch.testing.assert_close(graph_av, legacy_av_local, **tol)

    @pytest.mark.parametrize("attn_layer", [0, 2])  # factorizable AND attention
    def test_graph_lower_symbolic_trace(self, attn_layer) -> None:
        """``forward_lower_graph_exportable`` traces symbolically for BOTH the
        factorizable (attn_layer=0) and attention (attn_layer=2) graph lowers,
        and the traced module reproduces the eager graph lower bit-tight.

        attn_layer > 0 exercises the carry-all compact pair enumeration
        (``center_edge_pairs`` with ``static_nnei=None``) under make_fx
        symbolic tracing: its ``nonzero``/tensor-``repeat`` output sizes are
        UNBACKED SymInts, registered via ``xp_hint_dynamic_size`` — the
        mechanism that makes the attention graph lower ``.pt2``-exportable.
        """
        from deepmd.pt_expt.utils.serialization import (
            build_synthetic_graph_inputs,
        )

        # The real .pt2 export (``deserialize_to_file``) traces on CPU: it does
        # ``model.to("cpu")`` and builds CPU synthetic inputs. Mirror that here so
        # model params and the traced inputs share a device -- otherwise, on a
        # CUDA runner, the CUDA params meet the CPU graph tensors and FakeTensor
        # device propagation raises for aten.index_select.
        model = self._make_model(attn_layer=attn_layer).to("cpu")
        model.eval()
        sample = build_synthetic_graph_inputs(
            model,
            e_max=175,
            nframes=2,
            nloc=7,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        (
            atype,
            n_node,
            n_local,
            ei,
            ev,
            em,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            fp,
            ap,
            cs,
        ) = sample
        traced = model.forward_lower_graph_exportable(
            atype,
            n_node,
            n_local,
            ei,
            ev,
            em,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
            charge_spin=cs,
            destination_sorted=True,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        out = traced(
            atype,
            n_node,
            n_local,
            ei,
            ev,
            em,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            fp,
            ap,
            cs,
        )
        ref = model.forward_common_lower_graph(
            atype,
            n_node,
            n_local,
            ei,
            ev,
            em,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
        )
        tol = {"rtol": 1e-12, "atol": 1e-12}
        torch.testing.assert_close(out["energy"], ref["energy_redu"], **tol)
        torch.testing.assert_close(
            out["force"], ref["energy_derv_r"].reshape(out["force"].shape), **tol
        )
        torch.testing.assert_close(
            out["virial"], ref["energy_derv_c_redu"].reshape(out["virial"].shape), **tol
        )

    def test_smooth_attention_divergence_pinned(self) -> None:
        """End-to-end: the pt_expt DEFAULT route (carry-all graph) diverges
        from the dense route for ``smooth_type_embedding=True`` + attention —
        nonzero and bounded by the documented ~1e-4 magnitude.

        The carry-all graph drops sel-padding phantom terms from the smooth
        attention softmax denominator, while
        the dense path keeps them, so dense output is sel-dependent.  This
        test pins that divergence at the public model forward so a future
        refactor cannot silently change the carry-all smooth semantics.
        ``neighbor_graph_method="legacy"`` is the escape hatch restoring the
        dense numbers; the parity tests above cover the smooth=False regime
        where the two routes agree bit-tight.
        """
        model = self._make_model(attn_layer=2, smooth=True)
        model.eval()
        coord = self.coord.clone().requires_grad_(True)
        box = self.cell.reshape(1, 9)
        # None = the default flip: graph-eligible mixed_types -> carry-all graph
        graph = model.call_common(coord, self.atype, box, neighbor_graph_method=None)
        dense = model.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="legacy",
        )
        e_diff = (graph["energy_redu"] - dense["energy_redu"]).abs().max().item()
        f_diff = (graph["energy_derv_r"] - dense["energy_derv_r"]).abs().max().item()
        # nonzero: well above fp64 accumulation noise of a bit-tight parity
        assert e_diff > 1e-10, f"expected smooth divergence, got {e_diff:.3e}"
        # bounded: the documented magnitude is ~1e-4; 1e-3 leaves headroom
        assert e_diff < 1e-3, f"smooth divergence too large: {e_diff:.3e}"
        assert f_diff < 1e-3, f"smooth force divergence too large: {f_diff:.3e}"

    def test_pair_exclude_types_graph_vs_legacy(self) -> None:
        """Model-level pair_exclude_types: graph route and legacy dense agree
        bit-tight (fp64, 1e-12), AND the excluded model output differs from the
        no-exclude baseline (exclusion is not vacuous).

        Strategy: build the no-exclude model, serialize it, inject
        ``pair_exclude_types=[[0,1]]`` into the serialized dict, deserialize
        to get an exclude model with IDENTICAL weights, then run both routes.
        """
        import copy

        # 1. build the reference (no-exclude) model
        model_ref = self._make_model(attn_layer=0)
        model_ref.eval()

        # 2. derive the exclude model by patching the serialized dict
        data = copy.deepcopy(model_ref.serialize())
        data["pair_exclude_types"] = [[0, 1]]
        model_excl = EnergyModel.deserialize(data).to(self.device)
        model_excl.eval()

        tol = (
            {"rtol": 1e-12, "atol": 1e-12}
            if self.device.type == "cpu"
            else {"rtol": 1e-10, "atol": 1e-10}
        )
        box = self.cell.reshape(1, 9)

        # 3. graph route (build-time pair exclusion)
        graph_out = model_excl.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="dense",
        )
        # 4. legacy dense route (seam backstop in forward_atomic_graph)
        legacy_out = model_excl.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="legacy",
        )
        # parity: graph == legacy
        torch.testing.assert_close(
            graph_out["energy_redu"], legacy_out["energy_redu"], **tol
        )
        torch.testing.assert_close(
            graph_out["energy_derv_r"], legacy_out["energy_derv_r"], **tol
        )

        # 5. reference (no-exclude) via graph route
        ref_out = model_ref.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="dense",
        )
        # exclusion must have an effect
        e_diff = (graph_out["energy_redu"] - ref_out["energy_redu"]).abs().max().item()
        assert e_diff > 1e-10, (
            f"pair_exclude_types had no effect on energy; diff={e_diff:.3e}"
        )

    @pytest.mark.parametrize("attn_layer", [0, 2])  # factorizable AND attention
    def test_descriptor_exclude_types_graph_vs_legacy(self, attn_layer) -> None:
        """Descriptor-level exclude_types: graph route and legacy dense agree
        bit-tight (fp64, 1e-12) when exclusion is on the DESCRIPTOR (not the
        model pair_exclude_types).  Uses identical weights across both routes;
        also checks exclusion is non-vacuous vs a no-exclude baseline.
        """
        # 1. no-exclude model (graph route = reference)
        model_ref = self._make_model(attn_layer=attn_layer)
        model_ref.eval()

        # 2. exclude model: inject exclude_types into the serialized dict
        data = copy.deepcopy(model_ref.serialize())
        data["descriptor"]["exclude_types"] = [[0, 1]]
        model_excl = EnergyModel.deserialize(data).to(self.device)
        model_excl.eval()

        tol = (
            {"rtol": 1e-12, "atol": 1e-12}
            if self.device.type == "cpu"
            else {"rtol": 1e-10, "atol": 1e-10}
        )
        box = self.cell.reshape(1, 9)

        # 3. graph route
        graph_out = model_excl.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="dense",
        )
        # 4. legacy dense route
        legacy_out = model_excl.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="legacy",
        )
        torch.testing.assert_close(
            graph_out["energy_redu"], legacy_out["energy_redu"], **tol
        )
        torch.testing.assert_close(
            graph_out["energy_derv_r"], legacy_out["energy_derv_r"], **tol
        )
        torch.testing.assert_close(
            graph_out["energy_derv_c_redu"], legacy_out["energy_derv_c_redu"], **tol
        )

        # 5. exclusion must be non-vacuous
        ref_out = model_ref.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="dense",
        )
        e_diff = (graph_out["energy_redu"] - ref_out["energy_redu"]).abs().max().item()
        assert e_diff > 1e-10, (
            f"descriptor exclude_types had no effect on energy; diff={e_diff:.3e}"
        )

    @pytest.mark.parametrize("attn_layer", [0, 2])  # factorizable AND attention
    def test_graph_route_float32(self, attn_layer) -> None:
        """A float32 model runs the graph route and matches the dense route.

        The descriptor-level ``call_graph`` casts ``edge_vec`` to the
        descriptor precision manually (``@cast_precision`` cannot see inside
        the NeighborGraph dataclass); without it, fp32 models crash with a
        double-vs-float matmul on the graph route while the dense route works.
        fp32 accumulation-order differences bound the tolerance (1e-6/1e-5),
        per the fp32-computation guidance.
        """
        from deepmd.pt_expt.descriptor.dpa1 import DescrptDPA1 as _D
        from deepmd.pt_expt.fitting import InvarFitting as _F

        ds = _D(
            self.rcut,
            self.rcut_smth,
            self.sel,
            self.nt,
            neuron=[3, 6],
            axis_neuron=2,
            attn=4,
            attn_layer=attn_layer,
            attn_dotr=True,
            smooth_type_embedding=False,
            precision="float32",
            seed=GLOBAL_SEED,
        ).to(self.device)
        ft = _F(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=True,
            precision="float32",
            seed=GLOBAL_SEED,
        ).to(self.device)
        model = EnergyModel(ds, ft, type_map=self.type_map).to(self.device)
        model.eval()
        graph = model.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            self.cell.reshape(1, 9),
            neighbor_graph_method="dense",
        )
        dense = model.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            self.cell.reshape(1, 9),
            neighbor_graph_method="legacy",
        )
        tol = {"rtol": 1e-5, "atol": 1e-6}
        torch.testing.assert_close(graph["energy_redu"], dense["energy_redu"], **tol)
        # The graph lower assembles force / virial in the model compute precision
        # (fp32 here) while the dense reference stays fp64; compare values only.
        torch.testing.assert_close(
            graph["energy_derv_r"], dense["energy_derv_r"], check_dtype=False, **tol
        )
