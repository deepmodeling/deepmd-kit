# SPDX-License-Identifier: LGPL-3.0-or-later
"""Issue #5906 Task 2: SFPG completion across ranks, eager self-comm rung.

Reference = the folded single-rank graph forward (periodic carry-all graph,
src folded onto owners, gate complete by construction). Candidate = the
UNFOLDED extended-layout forward (owners + ghost images, edges restricted to
owned destinations -- the LAMMPS layout) with a 1-swap self-send comm_dict.

Geometry is the load-bearing part: the cell places a close pair ACROSS the
periodic x boundary so the close contact's src is a ghost image row and its
``log w``/``zero_count`` partial lives on a different row than the owner --
without that, every edge contributes ``log w = 0`` and the exchange is
untestable (the issue's vacuous-test warning). Parametrized over a
hard-freeze pair (r < r_inner, exercises the ``zero_count`` channel) and a
transition-zone pair (r_inner < r < r_outer, exercises ``log_eta``).
"""

import copy
import ctypes

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils.comm import (
    ensure_comm_registered,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)
from .test_zbl_bridging import (
    ZBL_CONFIG,
)

# bridging window of ZBL_CONFIG: r_inner=0.8, r_outer=1.2 (Angstrom)
_L = 6.0  # cubic box edge; > rcut so self-images never bond


def _close_pair_coords(gap: float) -> np.ndarray:
    """4 atoms; atoms 0/1 straddle the x periodic boundary at ``gap``.

    Atom 0 sits at ``x = gap/2`` and atom 1 at ``x = L - gap/2``, so their
    minimum-image distance is exactly ``gap``. Atoms 2/3 give the
    descriptor a normal environment (their 1.118 A contact sits inside the
    transition zone but does NOT straddle the boundary -- its partial is
    rank-local and pins the no-exchange-needed case alongside).
    """
    half = gap / 2.0
    return np.array(
        [
            [half, 3.0, 3.0],
            [_L - half, 3.0, 3.0],
            [2.5, 3.0, 3.0],
            [3.6, 3.2, 3.0],
        ],
        dtype=np.float64,
    ).reshape(1, 4, 3)


def _addr_of(np_arr: np.ndarray) -> int:
    return np_arr.ctypes.data_as(ctypes.c_void_p).value


def _build_self_comm_dict(
    *,
    nloc: int,
    nghost: int,
    sendlist_indices: np.ndarray,
    keepalive: list,
) -> dict:
    """Single-rank self-exchange comm_dict (per-file copy, repo precedent)."""
    sendlist_indices = np.ascontiguousarray(sendlist_indices, dtype=np.int32)
    keepalive.append(sendlist_indices)
    addr = _addr_of(sendlist_indices)
    return {
        "send_list": torch.tensor([addr], dtype=torch.int64, device="cpu"),
        "send_proc": torch.zeros(1, dtype=torch.int32, device="cpu"),
        "recv_proc": torch.zeros(1, dtype=torch.int32, device="cpu"),
        "send_num": torch.tensor([nghost], dtype=torch.int32, device="cpu"),
        "recv_num": torch.tensor([nghost], dtype=torch.int32, device="cpu"),
        "communicator": torch.zeros(1, dtype=torch.int64, device="cpu"),
        "nlocal": torch.tensor(nloc, dtype=torch.int32, device="cpu"),
        "nghost": torch.tensor(nghost, dtype=torch.int32, device="cpu"),
    }


def _make_bridged_model():
    """Bridged ZBL model with jittered zero-init residuals.

    A fresh DPA4 is architecturally edge-independent (zero-init residual
    projections -- see ``jitter_zero_arrays``), so an un-jittered model
    makes BOTH the parity and the ablation vacuous: the SFPG gate would
    multiply edge messages that never reach the output.
    """
    model = get_model(copy.deepcopy(ZBL_CONFIG))
    learned = model.atomic_model.models[0]
    data = jitter_zero_arrays(learned.descriptor.serialize(), np.random.default_rng(99))
    learned.descriptor = DescrptDPA4.deserialize(data)
    return model.to(torch.float64).to("cpu").eval()


def _extended_quartet(coord: np.ndarray):
    """LAMMPS-layout quartet: ghosts materialized, nlist over owned dst."""
    atype = np.array([[0, 0, 1, 1]], dtype=np.int64)
    box = (_L * np.eye(3, dtype=np.float64)).reshape(1, 3, 3)
    rcut = 4.0
    ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
        coord, atype, box.reshape(1, 9), rcut
    )
    # sel=128 is deliberately non-binding: this dense nlist only supplies
    # the owned-dst edge list; truncation would silently desynchronize the
    # two routes' neighbor sets.
    nlist = build_neighbor_list(
        ext_coord,
        ext_atype,
        4,
        rcut,
        [128],
        distinguish_types=False,
    )
    return (
        np.asarray(ext_coord).reshape(1, -1, 3),
        np.asarray(ext_atype),
        np.asarray(nlist),
        np.asarray(mapping),
    )


def _unfolded_graph_inputs(ext_coord, ext_atype, nlist):
    """Unfolded graph over owners+ghosts, edges for OWNED dst only."""
    nloc = nlist.shape[1]
    nall = ext_coord.shape[1]
    idx = nlist[0]  # (nloc, nnei), extended indices, -1 padding
    valid = idx >= 0
    src = idx[valid].astype(np.int64)
    dst = np.repeat(np.arange(nloc, dtype=np.int64), idx.shape[1])[valid.ravel()]
    edge_vec = ext_coord[0, src] - ext_coord[0, dst]
    return {
        "atype": torch.tensor(ext_atype[0], dtype=torch.int64),
        "n_node": torch.tensor([nall], dtype=torch.int64),
        "n_local": torch.tensor([nloc], dtype=torch.int64),
        "edge_index": torch.tensor(np.stack([src, dst]), dtype=torch.int64),
        "edge_vec": torch.tensor(edge_vec, dtype=torch.float64),
        "edge_mask": torch.ones(len(src), dtype=torch.bool),
    }


def _fold_forces(per_node_force: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Sum ghost-image force rows onto their owners (LAMMPS reverse comm)."""
    nloc = 4
    out = np.zeros((nloc, 3), dtype=np.float64)
    np.add.at(out, mapping[0], per_node_force)
    return out


class TestBridgedGraphSelfComm:
    @pytest.fixture(autouse=True)
    def _setup(self):
        ensure_comm_registered()
        self.model = _make_bridged_model()

    def _run_folded(self, coord: np.ndarray):
        atype = np.array([[0, 0, 1, 1]], dtype=np.int64)
        box = (_L * np.eye(3, dtype=np.float64)).reshape(1, 3, 3)
        graph = build_neighbor_graph(coord, atype, box, 4.0, canonicalize=True)
        out = self.model.forward_common_lower_graph(
            torch.tensor(atype.reshape(-1), dtype=torch.int64),
            torch.as_tensor(np.asarray(graph.n_node), dtype=torch.int64),
            torch.as_tensor(np.asarray(graph.n_node), dtype=torch.int64),
            torch.as_tensor(np.asarray(graph.edge_index), dtype=torch.int64),
            torch.as_tensor(np.asarray(graph.edge_vec), dtype=torch.float64),
            torch.as_tensor(np.asarray(graph.edge_mask), dtype=torch.bool),
        )
        e = out["energy_redu"].detach().numpy().reshape(-1)
        f = -out["energy_derv_r"].detach().numpy().reshape(-1, 3)
        return e, f

    def _run_self_comm(self, coord: np.ndarray):
        ext_coord, ext_atype, nlist, mapping = _extended_quartet(coord)
        gi = _unfolded_graph_inputs(ext_coord, ext_atype, nlist)
        nall = ext_coord.shape[1]
        keepalive: list = []
        comm_dict = _build_self_comm_dict(
            nloc=4,
            nghost=nall - 4,
            sendlist_indices=mapping[0, 4:].astype(np.int32),
            keepalive=keepalive,
        )
        out = self.model.forward_common_lower_graph(
            gi["atype"],
            gi["n_node"],
            gi["n_local"],
            gi["edge_index"],
            gi["edge_vec"],
            gi["edge_mask"],
            comm_dict=comm_dict,
        )
        e = out["energy_redu"].detach().numpy().reshape(-1)
        f_ext = -out["energy_derv_r"].detach().numpy().reshape(-1, 3)
        return e, _fold_forces(f_ext, mapping)

    @pytest.mark.parametrize(
        "gap",
        [
            0.4,  # r < r_inner=0.8: hard freeze, exercises the zero_count channel
            1.0,  # r_inner < r < r_outer=1.2: exercises the log_eta channel
        ],
    )
    def test_self_comm_matches_folded_reference(self, gap: float) -> None:
        coord = _close_pair_coords(gap)
        e_ref, f_ref = self._run_folded(coord)
        e_par, f_par = self._run_self_comm(coord)
        np.testing.assert_allclose(e_par, e_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f_par, f_ref, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "gap",
        [
            0.4,  # zero_count channel: ablation loses the hard freeze
            1.0,  # log_eta channel: ablation loses the transition attenuation
        ],
    )
    def test_ablation_identity_exchange_diverges(
        self, gap: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative contract: with the exchange stubbed to identity the
        parity BREAKS -- proves the geometry actually exercises the gate.
        """
        from deepmd.pt_expt.descriptor import dpa4 as pe_dpa4

        monkeypatch.setattr(
            pe_dpa4.DescrptDPA4,
            "_gate_partial_exchange",
            lambda self, partials, comm_dict: partials,
        )
        coord = _close_pair_coords(gap)
        e_ref, _ = self._run_folded(coord)
        e_par, _ = self._run_self_comm(coord)
        assert np.abs(e_par - e_ref).max() > 1e-6


class TestBridgedGraphWithCommExport:
    """Issue #5906 Task 2: the bridged composition's with-comm export rungs."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        ensure_comm_registered()
        self.model = _make_bridged_model()

    def test_make_fx_traces_with_comm(self) -> None:
        """The exchange (border_op_backward + border_op) traces symbolically
        into the with-comm graph forward.
        """
        from deepmd.pt_expt.utils.serialization import (
            _trace_and_export,
        )

        data = {"model": self.model.serialize()}
        exported, _meta, _dj, _keys = _trace_and_export(
            copy.deepcopy(data),
            model_json_override=None,
            with_comm_dict=True,
            lower_kind="graph",
        )
        loaded = exported.module()
        placeholders = loaded.graph.find_nodes(op="placeholder")
        assert len(placeholders) == 21, (
            f"graph with-comm program must accept 21 positional inputs "
            f"(13 graph-base incl. n_local + 8 comm); got {len(placeholders)}"
        )
        gm_code = str(loaded.code)
        assert "deepmd_export.border_op_backward" in gm_code
        assert (
            "deepmd_export.border_op." in gm_code
            or "deepmd_export.border_op(" in gm_code
        )

    def test_freeze_embeds_with_comm_artifact(self, tmp_path) -> None:
        """Freezing the bridged model to a graph ``.pt2`` embeds the nested
        with-comm artifact (mirrors the plain-DPA4 embed test).
        """
        import json
        import zipfile

        from deepmd.pt_expt.utils.serialization import (
            deserialize_to_file,
        )

        data = {"model": self.model.serialize()}
        p = str(tmp_path / "m_dpa4_zbl_graph.pt2")
        deserialize_to_file(p, copy.deepcopy(data), lower_kind="graph")
        with zipfile.ZipFile(p, "r") as zf:
            names = zf.namelist()
            meta = json.loads(zf.read("model/extra/metadata.json"))
        assert "model/extra/forward_lower_with_comm.pt2" in names
        assert meta["has_comm_artifact"] is True
        assert meta["lower_input_kind"] == "graph"


SPIN_ZBL_CONFIG = {
    **copy.deepcopy(ZBL_CONFIG),
    "spin": {"use_spin": [True, False], "scheme": "native"},
}


def _make_bridged_spin_model():
    """Native-spin + ZBL composition with jittered residuals (see above)."""
    model = get_model(copy.deepcopy(SPIN_ZBL_CONFIG))
    learned = model.atomic_model.models[0]
    data = jitter_zero_arrays(learned.descriptor.serialize(), np.random.default_rng(99))
    learned.descriptor = DescrptDPA4.deserialize(data)
    return model.to(torch.float64).to("cpu").eval()


class TestBridgedSpinGraphSelfComm:
    """Issue #5906 Task 3: native spin + ZBL, same ladder as the non-spin
    class. The gate exchange sits below the spin wrapper
    (``NativeSpinEnergyModel`` re-classes the SAME composed atomic model),
    so no spin-specific production change is expected -- these tests pin
    that the machinery composes.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        ensure_comm_registered()
        self.model = _make_bridged_spin_model()

    def _spins(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(11)
        sp = rng.normal(size=(n, 3))
        return sp / np.linalg.norm(sp, axis=-1, keepdims=True)

    def _run_folded(self, coord: np.ndarray):
        atype = np.array([[0, 0, 1, 1]], dtype=np.int64)
        box = (_L * np.eye(3, dtype=np.float64)).reshape(1, 3, 3)
        graph = build_neighbor_graph(coord, atype, box, 4.0, canonicalize=True)
        spin = torch.tensor(self._spins(4), dtype=torch.float64)
        out = self.model.forward_common_lower_graph(
            torch.tensor(atype.reshape(-1), dtype=torch.int64),
            torch.as_tensor(np.asarray(graph.n_node), dtype=torch.int64),
            torch.as_tensor(np.asarray(graph.n_node), dtype=torch.int64),
            torch.as_tensor(np.asarray(graph.edge_index), dtype=torch.int64),
            torch.as_tensor(np.asarray(graph.edge_vec), dtype=torch.float64),
            torch.as_tensor(np.asarray(graph.edge_mask), dtype=torch.bool),
            spin=spin,
        )
        e = out["energy_redu"].detach().numpy().reshape(-1)
        f = -out["energy_derv_r"].detach().numpy().reshape(-1, 3)
        fm = -out["energy_derv_r_mag"].detach().numpy().reshape(-1, 3)
        return e, f, fm

    def _run_self_comm(self, coord: np.ndarray):
        ext_coord, ext_atype, nlist, mapping = _extended_quartet(coord)
        gi = _unfolded_graph_inputs(ext_coord, ext_atype, nlist)
        nall = ext_coord.shape[1]
        # ghost spins mirror their owners (LAMMPS forwards ``sp``)
        spin_ext = torch.tensor(self._spins(4)[mapping[0]], dtype=torch.float64)
        keepalive: list = []
        comm_dict = _build_self_comm_dict(
            nloc=4,
            nghost=nall - 4,
            sendlist_indices=mapping[0, 4:].astype(np.int32),
            keepalive=keepalive,
        )
        out = self.model.forward_common_lower_graph(
            gi["atype"],
            gi["n_node"],
            gi["n_local"],
            gi["edge_index"],
            gi["edge_vec"],
            gi["edge_mask"],
            spin=spin_ext,
            comm_dict=comm_dict,
        )
        e = out["energy_redu"].detach().numpy().reshape(-1)
        f = _fold_forces(-out["energy_derv_r"].detach().numpy().reshape(-1, 3), mapping)
        fm = _fold_forces(
            -out["energy_derv_r_mag"].detach().numpy().reshape(-1, 3), mapping
        )
        return e, f, fm

    @pytest.mark.parametrize(
        "gap",
        [
            0.4,  # zero_count channel across the boundary
            1.0,  # log_eta channel across the boundary
        ],
    )
    def test_self_comm_matches_folded_reference(self, gap: float) -> None:
        coord = _close_pair_coords(gap)
        e_ref, f_ref, fm_ref = self._run_folded(coord)
        e_par, f_par, fm_par = self._run_self_comm(coord)
        np.testing.assert_allclose(e_par, e_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f_par, f_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fm_par, fm_ref, rtol=1e-12, atol=1e-12)

    def test_freeze_embeds_with_comm_artifact(self, tmp_path) -> None:
        """Freezing the bridged spin model embeds the nested artifact."""
        import json
        import zipfile

        from deepmd.pt_expt.utils.serialization import (
            deserialize_to_file,
        )

        data = {"model": self.model.serialize()}
        p = str(tmp_path / "m_dpa4_spin_zbl_graph.pt2")
        deserialize_to_file(p, copy.deepcopy(data), lower_kind="graph")
        with zipfile.ZipFile(p, "r") as zf:
            names = zf.namelist()
            meta = json.loads(zf.read("model/extra/metadata.json"))
        assert "model/extra/forward_lower_with_comm.pt2" in names
        assert meta["has_comm_artifact"] is True
