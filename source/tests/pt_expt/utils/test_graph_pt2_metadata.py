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

from deepmd.pt_expt.utils.serialization import (
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
    # B2.0: the edge axis is DYNAMIC (Dim("nedge", min=2)); there is no static
    # capacity baked into the AOTI artifact, so no ``edge_capacity`` is persisted.
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
    # edge_capacity is a graph-only artifact constant; the dense path omits it.
    assert "edge_capacity" not in meta


def test_neighbor_graph_method_rejected_on_nlist_artifact(dpa1_dpmodel_data) -> None:
    """A non-default ``neighbor_graph_method`` on a NLIST-form artifact raises.

    The knob is consumed only by graph-form ``.pt2`` eval; silently ignoring
    it on nlist-form artifacts misled users into thinking they selected an
    O(N) builder (OutisLi review, #5714). The nlist-path knob is
    ``nlist_backend``.
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
        # the default stays accepted (no behavior change)
        DeepPot(p)


class _FakeDesc:
    def __init__(self, n_attn: int) -> None:
        self._n = n_attn

    def get_numb_attn_layer(self) -> int:
        return self._n


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
