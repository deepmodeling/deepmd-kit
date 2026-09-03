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
from types import (
    SimpleNamespace,
)

import numpy as np
import pytest
import torch

from deepmd.entrypoints.convert_backend import (
    convert_backend,
)
from deepmd.pt_expt.model.graph_lower import (
    graph_edge_dtype,
)
from deepmd.pt_expt.utils.serialization import (
    _graph_reads_source_csr,
    _needs_with_comm_artifact,
    _resolve_target_lower_kind,
    _supports_graph_export,
    deserialize_to_file,
    serialize_from_file,
)


def _exported_graph(
    placeholder_names: tuple[str, ...], used_name: str | None
) -> SimpleNamespace:
    """Build the graph surface consumed by ``_graph_reads_source_csr``."""
    graph = torch.fx.Graph()
    placeholders = {name: graph.placeholder(name) for name in placeholder_names}
    graph.output(placeholders[used_name] if used_name is not None else 0)
    return SimpleNamespace(graph_module=SimpleNamespace(graph=graph))


@pytest.mark.parametrize(
    ("placeholder_names", "used_name", "expected"),
    [
        ((), None, True),
        (("source_order", "source_row_ptr"), "source_order", True),
        (("source_order", "source_row_ptr"), "source_row_ptr", True),
        (("source_order", "source_row_ptr"), None, False),
        (("edge_index", "edge_vec"), None, True),
    ],
    ids=[
        "unrecognized_graph",
        "source_order_used",
        "source_row_ptr_used",
        "source_csr_unused",
        "source_csr_absent",
    ],
)
def test_graph_reads_source_csr(
    placeholder_names: tuple[str, ...], used_name: str | None, expected: bool
) -> None:
    """Source-CSR metadata follows actual graph users and fails safe."""
    assert (
        _graph_reads_source_csr(_exported_graph(placeholder_names, used_name))
        is expected
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


@pytest.mark.parametrize(
    "lower_input_kind",
    ["nlist", "graph", "dpa1_canonical", "dpa4c_canonical", "edge_vec"],
)
def test_pt2_serialization_preserves_lower_input_kind(
    tmp_path, lower_input_kind: str
) -> None:
    """The interchange dictionary exposes the artifact's lower semantics."""
    model_file = tmp_path / "model.pt2"
    with zipfile.ZipFile(model_file, "w") as zf:
        zf.writestr("model/extra/model.json", json.dumps({"model": {}}))
        zf.writestr(
            "model/extra/metadata.json",
            json.dumps({"lower_input_kind": lower_input_kind}),
        )

    data = serialize_from_file(str(model_file))

    assert data["lower_input_kind"] == lower_input_kind


@pytest.mark.parametrize(
    "lower_input_kind",
    ["nlist", "graph", "dpa1_canonical", "dpa4c_canonical", "edge_vec"],
)
def test_pte_serialization_preserves_lower_input_kind(
    tmp_path, monkeypatch: pytest.MonkeyPatch, lower_input_kind: str
) -> None:
    """PTE extra metadata has the same interchange contract as PT2."""

    def load_exported_program(_model_file: str, *, extra_files: dict[str, str]) -> None:
        extra_files["model.json"] = json.dumps({"model": {}})
        extra_files["model_def_script.json"] = ""
        extra_files["metadata.json"] = json.dumps(
            {"lower_input_kind": lower_input_kind}
        )

    monkeypatch.setattr(torch.export, "load", load_exported_program)

    data = serialize_from_file(str(tmp_path / "model.pte"))

    assert data["lower_input_kind"] == lower_input_kind


@pytest.mark.parametrize(
    ("embedded_lower_input_kind", "expected"),
    [("graph", "graph"), (None, "nlist")],
)
def test_pt2_serialization_legacy_lower_input_kind_fallback(
    tmp_path,
    embedded_lower_input_kind: str | None,
    expected: str,
) -> None:
    """Legacy PT2 archives use embedded metadata, then dense fallback."""
    model_file = tmp_path / "model.pt2"
    model_data: dict[str, object] = {"model": {}}
    if embedded_lower_input_kind is not None:
        model_data["lower_input_kind"] = embedded_lower_input_kind
    with zipfile.ZipFile(model_file, "w") as zf:
        zf.writestr("model/extra/model.json", json.dumps(model_data))

    data = serialize_from_file(str(model_file))

    assert data["lower_input_kind"] == expected


@pytest.mark.parametrize(
    ("embedded_lower_input_kind", "expected"),
    [("graph", "graph"), (None, "nlist")],
)
def test_pte_serialization_legacy_lower_input_kind_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    embedded_lower_input_kind: str | None,
    expected: str,
) -> None:
    """Legacy PTE archives use embedded metadata, then dense fallback."""
    model_data: dict[str, object] = {"model": {}}
    if embedded_lower_input_kind is not None:
        model_data["lower_input_kind"] = embedded_lower_input_kind

    def load_exported_program(_model_file: str, *, extra_files: dict[str, str]) -> None:
        extra_files["model.json"] = json.dumps(model_data)
        extra_files["model_def_script.json"] = ""
        extra_files["metadata.json"] = ""

    monkeypatch.setattr(torch.export, "load", load_exported_program)

    data = serialize_from_file(str(tmp_path / "model.pte"))

    assert data["lower_input_kind"] == expected


@pytest.fixture(scope="module")
def dpa1_dpmodel_data() -> dict:
    return _build_dpa1_data()


def test_convert_regular_pt_dpa1_preserves_dense_semantics(tmp_path) -> None:
    """A nonzero-davg PT artifact remains numerically dense after conversion."""
    from deepmd.infer import (
        DeepPot,
    )
    from deepmd.pt.utils.serialization import deserialize_to_file as deserialize_to_pt
    from deepmd.pt.utils.serialization import serialize_from_file as serialize_from_pt

    data = _build_dpa1_data()
    descriptor_variables = data["model"]["descriptor"]["@variables"]
    descriptor_variables["davg"] = np.full_like(descriptor_variables["davg"], 0.01)
    source_model = tmp_path / "model.pth"
    converted_model = tmp_path / "model.pt2"
    deserialize_to_pt(str(source_model), copy.deepcopy(data))

    source_data = serialize_from_pt(str(source_model))
    assert source_data["lower_input_kind"] == "nlist"

    convert_backend(INPUT=str(source_model), OUTPUT=str(converted_model))
    assert _read_metadata(str(converted_model))["lower_input_kind"] == "nlist"

    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.1],
            [0.3, 1.4, 0.2],
            [1.2, 1.1, 0.8],
        ],
        dtype=np.float64,
    )[None, ...]
    atype = np.array([[0, 1, 0, 1]], dtype=np.int32)
    source_result = DeepPot(str(source_model), auto_batch_size=False).eval(
        coord, None, atype
    )
    converted_result = DeepPot(str(converted_model), auto_batch_size=False).eval(
        coord, None, atype
    )
    np.testing.assert_allclose(
        converted_result[0], source_result[0], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        converted_result[1], source_result[1], rtol=1e-10, atol=1e-10
    )


def test_convert_pt_dpa4_through_dp_maps_edge_vec_to_graph(tmp_path) -> None:
    """A schema-neutral DPModel container preserves PT SeZM's edge-list ABI."""
    from deepmd.dpmodel.utils.serialization import (
        load_dp_model,
    )
    from deepmd.pt.model.model import get_model as get_pt_model
    from deepmd.pt.train.wrapper import (
        ModelWrapper,
    )
    from deepmd.pt.utils.serialization import serialize_from_file as serialize_from_pt

    from ..model.test_dpa4_export import (
        _DPA4_CONFIG,
    )

    config = copy.deepcopy(_DPA4_CONFIG)
    source_model = tmp_path / "model.pt"
    intermediate_model = tmp_path / "model.dp"
    converted_model = tmp_path / "model.pte"
    model = get_pt_model(config)
    wrapper = ModelWrapper(model, model_params=config)
    torch.save({"model": wrapper.state_dict()}, source_model)

    source_data = serialize_from_pt(str(source_model))
    assert source_data["lower_input_kind"] == "edge_vec"

    convert_backend(INPUT=str(source_model), OUTPUT=str(intermediate_model))
    assert load_dp_model(str(intermediate_model))["lower_input_kind"] == "edge_vec"

    convert_backend(INPUT=str(intermediate_model), OUTPUT=str(converted_model))
    converted_data = serialize_from_file(str(converted_model))
    assert converted_data["lower_input_kind"] == "graph"


def test_edge_vec_uses_dense_lower_for_non_energy_target(tmp_path) -> None:
    """Target model capabilities, not the source spelling, select graph export."""
    from deepmd.dpmodel.utils.serialization import (
        save_dp_model,
    )
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )

    from ..model.test_dpa4_export import (
        _DPA4_CONFIG,
    )

    config = copy.deepcopy(_DPA4_CONFIG)
    config.pop("type")
    config["fitting_net"] = {
        "type": "property",
        "task_dim": 2,
        "neuron": [16],
        "precision": "float64",
        "seed": 1,
    }
    model = get_model(config)
    source_model = tmp_path / "property.dp"
    converted_model = tmp_path / "property.pte"
    save_dp_model(
        str(source_model),
        {"model": model.serialize(), "lower_input_kind": "edge_vec"},
    )

    convert_backend(INPUT=str(source_model), OUTPUT=str(converted_model))

    assert serialize_from_file(str(converted_model))["lower_input_kind"] == "nlist"


def test_native_spin_auto_pte_reports_target_constraint(tmp_path) -> None:
    """An unbound native-spin container reports why automatic PTE export fails."""
    from deepmd.dpmodel.utils.serialization import (
        save_dp_model,
    )

    from ..model.test_dpa4_native_spin import (
        _build_native_spin_model_cpu,
    )

    source_model = tmp_path / "native_spin.dp"
    data = {"model": _build_native_spin_model_cpu().serialize()}
    assert _resolve_target_lower_kind("native_spin.pt2", data, "auto") == "graph"
    save_dp_model(
        str(source_model),
        data,
    )

    with pytest.raises(
        ValueError,
        match=r"automatic lower selection for native-spin models requires a \.pt2 output",
    ):
        convert_backend(
            INPUT=str(source_model), OUTPUT=str(tmp_path / "native_spin.pte")
        )


def test_deserialize_rejects_unknown_lower_kind(dpa1_dpmodel_data, tmp_path) -> None:
    """The target serializer owns validation of its supported lower ABIs."""
    with pytest.raises(ValueError, match="Unsupported lower_kind 'unknown'"):
        deserialize_to_file(
            str(tmp_path / "model.pt2"),
            copy.deepcopy(dpa1_dpmodel_data),
            lower_kind="unknown",
        )


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
    assert meta["graph_source_csr"] is True
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
    """Compressed DPA1 graph geometry follows descriptor compute precision.

    The stub borrows the REAL capability methods (dpmodel
    ``DescrptDPA1.graph_edge_dtype``, pt_expt
    ``DescrptDPA1.supports_graph_export``) and the real DPAtomicModel-style
    delegation, so the helpers are tested through the capability seam they
    consume in production (issue #5906 Task 4).
    """
    from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as _DPDescrptDPA1
    from deepmd.pt_expt.descriptor.dpa1 import DescrptDPA1 as _PEDescrptDPA1

    class _Descriptor:
        geo_compress = True

        class _Block:
            mean = torch.empty(0, dtype=statistics_dtype)

        se_atten = _Block()

        def _fused_eligible(self, backend: str) -> bool:
            return backend == "cuda" and self.se_atten.mean.dtype == torch.float32

        graph_edge_dtype = _DPDescrptDPA1.graph_edge_dtype
        supports_graph_export = _PEDescrptDPA1.supports_graph_export

    class _AtomicModel:
        descriptor = _Descriptor()

        def graph_edge_dtype(self) -> str:
            return str(self.descriptor.graph_edge_dtype())

        def supports_graph_export(self) -> bool:
            return bool(self.descriptor.supports_graph_export())

    class _Model:
        atomic_model = _AtomicModel()

    assert graph_edge_dtype(_Model(), "graph") == expected
    assert graph_edge_dtype(_Model(), "nlist") == "float64"
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

    def uses_compact_edge_pairs(self) -> bool:
        # mirrors DPAtomicModel's capability delegation (issue #5906 Task 4)
        return self.descriptor.uses_compact_edge_pairs()


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


def test_graph_trace_version_guard_checks_compositions(monkeypatch) -> None:
    """Compositions answer by aggregation and are NOT exempt (issue #5906).

    The old defensive fallthrough silently passed any model without a
    single ``.descriptor``; a linear composition whose child emits compact
    pairs must now trip the torch < 2.6 guard like the child itself would.
    """
    import torch

    from deepmd.pt_expt.utils.serialization import (
        check_graph_trace_torch_version,
    )

    class _FakeLinearAtomicModel:
        def __init__(self, children) -> None:
            self.models = children

        def uses_compact_edge_pairs(self) -> bool:
            return any(m.uses_compact_edge_pairs() for m in self.models)

    class _FakeLinearModel:
        def __init__(self, n_attns) -> None:
            self.atomic_model = _FakeLinearAtomicModel(
                [_FakeAtomicModel(n) for n in n_attns]
            )

    monkeypatch.setattr(torch, "__version__", "2.5.1")
    with pytest.raises(RuntimeError, match=r"torch >= 2\.6"):
        check_graph_trace_torch_version(_FakeLinearModel([0, 2]))
    check_graph_trace_torch_version(_FakeLinearModel([0, 0]))


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
        ``test_dpa4_export.py``), ``"dpa2"`` (``DPA2_GUARD_CONFIG`` above),
        or ``"linear-two-dpa2"`` (a linear composition of two
        ``DPA2_GUARD_CONFIG`` children -- capability aggregation,
        issue #5906 Task 4).

    Returns
    -------
    torch.nn.Module
        The constructed pt_expt model, on CPU, in eval mode.
    """
    from deepmd.pt_expt.model.get_model import get_model as get_pt_expt_model

    if model_kind == "dpa4":
        from ..model.test_dpa4_export import (
            _DPA4_CONFIG,
        )

        config = _DPA4_CONFIG
    elif model_kind == "dpa2":
        config = DPA2_GUARD_CONFIG
    elif model_kind == "linear-two-dpa2":
        from deepmd.pt_expt.model.get_model import (
            get_linear_model,
        )

        child = {
            "descriptor": DPA2_GUARD_CONFIG["descriptor"],
            "fitting_net": DPA2_GUARD_CONFIG["fitting_net"],
        }
        config = {
            "type_map": DPA2_GUARD_CONFIG["type_map"],
            "models": [copy.deepcopy(child), copy.deepcopy(child)],
            "weights": "mean",
        }
        model = get_linear_model(config)
        model.to("cpu")
        model.eval()
        return model
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
        (
            "linear-two-dpa2",
            "graph",
            True,
        ),  # composition aggregates children (issue #5906 Task 4)
        (
            "linear-two-dpa2",
            "nlist",
            True,
        ),  # dpa2 children's dense lower supports comm -> composition does
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
