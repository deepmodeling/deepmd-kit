# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-form ``.pt2`` export: ``lower_input_kind`` metadata branch.

Covers both branches of the ``lower_kind`` selector on
``deserialize_to_file``: ``"graph"`` traces ``forward_lower_graph_exportable``
over the NeighborGraph schema and records ``lower_input_kind == "graph"`` in
``metadata.json``; the default (``"nlist"``) traces the dense quartet and
records ``lower_input_kind == "nlist"``.
"""

import copy
import json
import os
import tempfile
import zipfile

import pytest
import torch

from deepmd.pt_expt.utils.serialization import (
    _graph_edge_dtype,
    _needs_with_comm_artifact,
    _supports_graph_export,
    deserialize_to_file,
)

# dpa1 with attn_layer == 0 — the energy model exercised by the graph path.
DPA1_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_atten",
        "sel": 30,
        "rcut_smth": 2.0,
        "rcut": 6.0,
        "neuron": [2, 4, 8],
        "axis_neuron": 4,
        "attn": 5,
        "attn_layer": 0,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "scaling_factor": 1.0,
        "normalize": True,
        "temperature": 1.0,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [5, 5, 5],
        "resnet_dt": True,
        "seed": 1,
    },
}


def _build_dpa1_data(config: dict | None = None) -> dict:
    """Build a serialized dpmodel data dict for a dpa1(attn_layer=0) energy model.

    Parameters
    ----------
    config : dict, optional
        Model config to build from.  Defaults to ``DPA1_CONFIG``.
    """
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    if config is None:
        config = DPA1_CONFIG
    model = get_model(copy.deepcopy(config))
    return {
        "model": model.serialize(),
        "model_def_script": copy.deepcopy(config),
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }


def _read_metadata(pt2_path: str) -> dict:
    """Read ``model/extra/metadata.json`` from a ``.pt2`` ZIP archive."""
    with zipfile.ZipFile(pt2_path, "r") as zf:
        raw = zf.read("model/extra/metadata.json").decode("utf-8")
    return json.loads(raw)


@pytest.fixture(scope="module")
def dpa1_dpmodel_data() -> dict:
    return _build_dpa1_data()


def test_graph_pt2_has_lower_input_kind_graph(dpa1_dpmodel_data) -> None:
    """``lower_kind="graph"`` -> metadata ``lower_input_kind == "graph"``."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m_graph.pt2")
        deserialize_to_file(
            p,
            copy.deepcopy(dpa1_dpmodel_data),
            do_atomic_virial=True,
            lower_kind="graph",
        )
        meta = _read_metadata(p)
    assert meta["lower_input_kind"] == "graph"
    assert meta["graph_edge_dtype"] == "float64"
    # A dynamic edge axis has no persisted static capacity.
    assert "edge_capacity" not in meta


def test_graph_pt2_small_sel_exports() -> None:
    """Graph-form ``.pt2`` export succeeds for a small-``sel`` model.

    The graph trace capacity derives from the synthetic trace system's
    REAL edge count; the former sel-derived estimate
    (``ceil(1.25 * nloc * sum(sel))``) overflowed the sel-free carry-all
    builder whenever the actual degree exceeded ``sel`` (``edge overflow:
    36 real edges > edge_capacity 18`` at ``sel=2``).
    """
    cfg = copy.deepcopy(DPA1_CONFIG)
    cfg["descriptor"]["sel"] = 2
    data = _build_dpa1_data(cfg)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m_graph_small_sel.pt2")
        deserialize_to_file(
            p,
            data,
            do_atomic_virial=True,
            lower_kind="graph",
        )
        meta = _read_metadata(p)
    assert meta["lower_input_kind"] == "graph"


@pytest.mark.parametrize(
    ("statistics_dtype", "expected"),
    [(torch.float32, "float32"), (torch.float64, "float64")],
)
def test_compressed_graph_uses_compute_precision_edge_geometry(
    statistics_dtype: torch.dtype, expected: str
) -> None:
    """Compressed DPA1 graph geometry follows descriptor compute precision."""

    class _Descriptor:
        geo_compress = True

        class _Block:
            mean = torch.empty(0, dtype=statistics_dtype)

        se_atten = _Block()

        def _fused_eligible(self, backend: str) -> bool:
            return backend == "cuda" and self.se_atten.mean.dtype == torch.float32

    class _AtomicModel:
        descriptor = _Descriptor()

    class _Model:
        atomic_model = _AtomicModel()

    assert _graph_edge_dtype(_Model(), "graph") == expected
    assert _graph_edge_dtype(_Model(), "nlist") == "float64"
    assert _supports_graph_export(_Model()) is (statistics_dtype == torch.float32)


def test_dense_pt2_has_lower_input_kind_nlist(dpa1_dpmodel_data) -> None:
    """Default (``lower_kind="nlist"``) -> metadata ``lower_input_kind == "nlist"``."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m_dense.pt2")
        deserialize_to_file(
            p,
            copy.deepcopy(dpa1_dpmodel_data),
            do_atomic_virial=True,
        )
        meta = _read_metadata(p)
    assert meta["lower_input_kind"] == "nlist"
    assert meta["graph_edge_dtype"] == "float64"
    # edge_capacity is a graph-only artifact constant; the dense path omits it.
    assert "edge_capacity" not in meta


def test_neighbor_graph_method_rejected_on_nlist_artifact(dpa1_dpmodel_data) -> None:
    """A non-default ``neighbor_graph_method`` on a NLIST-form artifact raises.

    The knob is consumed only by graph-form ``.pt2`` eval; silently ignoring
    it on nlist-form artifacts misled users into thinking they selected an
    O(N) builder. The nlist-path knob is ``nlist_backend``.
    """
    from deepmd.infer import (
        DeepPot,
    )

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m_dense.pt2")
        deserialize_to_file(
            p,
            copy.deepcopy(dpa1_dpmodel_data),
            do_atomic_virial=True,
        )
        with pytest.raises(ValueError, match="graph-form"):
            DeepPot(p, neighbor_graph_method="vesin")
        # The default remains valid for nlist-form artifacts.
        DeepPot(p)


class _FakeDesc:
    def __init__(self, n_attn: int) -> None:
        self._n = n_attn

    def uses_compact_edge_pairs(self) -> bool:
        # mirrors dpa1's capability: attention rides center_edge_pairs
        return self._n > 0


class _FakeAtomicModel:
    def __init__(self, n_attn: int) -> None:
        self.descriptor = _FakeDesc(n_attn)


class _FakeModel:
    def __init__(self, n_attn: int) -> None:
        self.atomic_model = _FakeAtomicModel(n_attn)


@pytest.mark.parametrize(
    "version", ["2.5.1", "2.5.1+cu124"]
)  # torch below the 2.6 floor
def test_graph_trace_version_guard_rejects_attention_on_old_torch(
    monkeypatch, version
) -> None:
    """attn_layer > 0 on torch < 2.6 fails fast with a clear message."""
    import torch

    from deepmd.pt_expt.utils.serialization import (
        check_graph_trace_torch_version,
    )

    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(RuntimeError, match=r"torch >= 2\.6"):
        check_graph_trace_torch_version(_FakeModel(2))


@pytest.mark.parametrize(
    ("version", "n_attn"),
    [
        ("2.5.1", 0),  # old torch OK without attention (backed symbols only)
        ("2.6.0", 2),  # floor version with attention
        ("2.10.0+cu126", 2),  # current torch with attention, local suffix
    ],
)
def test_graph_trace_version_guard_passes(monkeypatch, version, n_attn) -> None:
    """No-attention models and torch >= 2.6 pass the guard silently."""
    import torch

    from deepmd.pt_expt.utils.serialization import (
        check_graph_trace_torch_version,
    )

    monkeypatch.setattr(torch, "__version__", version)
    check_graph_trace_torch_version(_FakeModel(n_attn))


def test_graph_trace_version_guard_tolerates_no_descriptor(monkeypatch) -> None:
    """Composite models without a single descriptor pass (dense route anyway)."""
    import torch

    from deepmd.pt_expt.utils.serialization import (
        check_graph_trace_torch_version,
    )

    class _NoDesc:
        pass

    monkeypatch.setattr(torch, "__version__", "2.5.1")
    check_graph_trace_torch_version(_NoDesc())


@pytest.mark.parametrize(
    ("repformer_overrides", "should_raise"),
    [
        # nlayers >= 2 so a non-last layer actually consumes compact pairs
        # (the LAST layer is built with update_chnnl_2=False, which forces
        # its g2/h2 updates off).
        ({"nlayers": 2}, True),  # default update_g2_has_attn=True
        ({"nlayers": 2, "update_g2_has_attn": False, "update_h2": True}, True),
        (
            {"nlayers": 2, "update_g2_has_attn": False, "update_h2": False},
            False,
        ),  # no pair consumers on any layer
        # nlayers=1: the only layer is the last -> NO effective compact-pair
        # consumer even with the arguments enabled; torch 2.5 stays usable.
        ({"nlayers": 1}, False),
    ],
    ids=["g2_attn_2layers", "update_h2_2layers", "no_pair_consumers", "single_layer"],
)
def test_graph_trace_version_guard_dpa2_compact_pairs(
    monkeypatch, repformer_overrides, should_raise
) -> None:
    """A default graph-eligible DPA2 must trip the torch < 2.6 guard.

    Regression (OutisLi review): the guard keyed on dpa1's
    ``get_numb_attn_layer``, which DPA2 does not implement, so every DPA2
    passed and compiled training / graph freeze failed deep inside
    ``make_fx`` instead of the fast version error.  The guard now keys on
    the descriptor capability ``uses_compact_edge_pairs()``: DPA2's
    ``update_g2_has_attn`` (default True) and ``update_h2`` both run the
    compact ``center_edge_pairs`` realization; with both off the lower
    traces backed symbols only and old torch stays usable.  The capability
    reads the EFFECTIVE per-layer flags: the last layer's g2/h2 updates
    are structurally off (``update_chnnl_2=False``), so a single-layer
    repformer never builds compact pairs and must NOT be rejected.
    """
    import torch

    from deepmd.dpmodel.model.model import (
        get_model,
    )
    from deepmd.pt_expt.utils.serialization import (
        check_graph_trace_torch_version,
    )

    cfg = copy.deepcopy(DPA2_GUARD_CONFIG)
    cfg["descriptor"]["repformer"].update(repformer_overrides)
    model = get_model(cfg)
    assert model.atomic_model.descriptor.uses_graph_lower() is True

    monkeypatch.setattr(torch, "__version__", "2.5.1")
    if should_raise:
        with pytest.raises(RuntimeError, match=r"torch >= 2\.6"):
            check_graph_trace_torch_version(model)
    else:
        check_graph_trace_torch_version(model)


# Small graph-eligible dpa2 for the version-guard regression above.
DPA2_GUARD_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa2",
        "repinit": {
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "nsel": 10,
            "neuron": [4, 8],
            "axis_neuron": 2,
        },
        "repformer": {
            "rcut": 3.0,
            "rcut_smth": 0.5,
            "nsel": 6,
            "nlayers": 1,
            "g1_dim": 8,
            "g2_dim": 4,
        },
    },
    "fitting_net": {"neuron": [8, 8], "seed": 1},
}


def _build_model(model_kind: str) -> torch.nn.Module:
    """Build a small pt_expt model for ``_needs_with_comm_artifact`` tests.

    No AOTI compile is involved — the caller only inspects the returned
    model's descriptor capability methods.

    Parameters
    ----------
    model_kind : str
        ``"dpa4"`` (bridging-free SeZM, config shared with
        ``test_dpa4_export.py``) or ``"dpa2"`` (``DPA2_GUARD_CONFIG`` above).

    Returns
    -------
    torch.nn.Module
        The constructed pt_expt model, on CPU, in eval mode.
    """
    from deepmd.pt_expt.model.get_model import (
        get_model as get_pt_expt_model,
    )

    if model_kind == "dpa4":
        from ..model.test_dpa4_export import (
            _DPA4_CONFIG,
        )

        config = _DPA4_CONFIG
    elif model_kind == "dpa2":
        config = DPA2_GUARD_CONFIG
    else:
        raise ValueError(f"unknown model_kind {model_kind!r}")
    model = get_pt_expt_model(copy.deepcopy(config))
    model.to("cpu")
    model.eval()
    return model


@pytest.mark.parametrize(
    "model_kind,lower_kind,expected",
    [
        ("dpa4", "graph", True),  # graph lower has real border exchange now
        (
            "dpa4",
            "nlist",
            False,
        ),  # dense lower is comm-less: no artifact, no trace crash
        (
            "dpa2",
            "nlist",
            True,
        ),  # dense with-comm is dpa2's production MP path — unchanged
        ("dpa2", "graph", True),  # graph with-comm unchanged
    ],
)
def test_needs_with_comm_artifact_kind_aware(model_kind, lower_kind, expected) -> None:
    """``_needs_with_comm_artifact`` is lower-kind-aware for DPA4, unchanged for dpa2.

    DPA4's graph lower carries a real per-layer ``border_op`` exchange, but
    its dense (nlist) lower adapter raises on ``comm_dict`` — so the dense
    kind must not request a with-comm artifact (it would crash the trace).
    dpa2 implements comm on both lowers (no ``dense_lower_supports_comm``
    override), so both kinds stay ``True``.
    """
    model = _build_model(model_kind)
    assert _needs_with_comm_artifact(model, lower_kind) is expected
