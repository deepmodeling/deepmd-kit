# SPDX-License-Identifier: LGPL-3.0-or-later
"""Request-aware source CSR omission for the compact DPA4C canonical graph.

The source CSR of a compact canonical graph serves the source-attributed
per-atom virial. A short-range single-block DPA4C deployment that requests no
``DERV_C`` output therefore needs only two-entry placeholders. These tests pin
the graph-building contract and the request propagation of
:class:`deepmd.pt_expt.infer.deep_eval.DeepEval` without a GPU, a compiled
``.pt2`` archive, or any model weights.
"""

import inspect
from unittest import (
    mock,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.output_def import (
    OutputVariableCategory,
    OutputVariableDef,
)
from deepmd.pt_expt.infer.deep_eval import DeepEval as PtExptDeepEval
from deepmd.pt_expt.utils.canonical_graph import (
    canonical_graph_from_pairs,
    validate_canonical_graph_shapes,
)
from deepmd.pt_expt.utils.vesin_graph_builder import (
    build_canonical_graph_vesin,
)

# Pair construction and builder forwarding.

_GEOMETRY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]


def _nodes(count: int) -> torch.Tensor:
    return torch.tensor([count], dtype=torch.int64)


def _pairs(center, source, count):
    return (
        torch.tensor(center, dtype=torch.int64),
        torch.tensor(source, dtype=torch.int64),
        torch.tensor(_GEOMETRY[:count], dtype=torch.float64).reshape(count, 3),
    )


def _build(center, source, vec, node_count, *, with_source_csr):
    return canonical_graph_from_pairs(
        center,
        source,
        vec,
        _nodes(node_count),
        node_count,
        with_source_csr=with_source_csr,
    )


@pytest.mark.parametrize("edge_count", [0, 1, 2, 3, 4])
def test_default_matches_explicit_true(edge_count: int) -> None:
    """Omitting the keyword equals ``with_source_csr=True`` exactly."""
    pairs = _pairs(list(range(edge_count)), [0] * edge_count, edge_count)
    node_count = max(edge_count, 1)
    default = canonical_graph_from_pairs(*pairs, _nodes(node_count), node_count)
    explicit = canonical_graph_from_pairs(
        *pairs, _nodes(node_count), node_count, with_source_csr=True
    )
    for field in (
        "n_node",
        "n_local",
        "source",
        "edge_vec",
        "destination_row_ptr",
        "source_row_ptr",
        "source_order",
    ):
        torch.testing.assert_close(
            getattr(default, field), getattr(explicit, field), rtol=0, atol=0
        )


@pytest.mark.parametrize("edge_count", [0, 1, 2, 3, 4])
def test_false_keeps_graph_and_only_drops_source_csr(edge_count: int) -> None:
    """``False`` changes nothing except the source CSR contents."""
    pairs = _pairs([0, 0, 1, 2][:edge_count], [1, 2, 2, 0][:edge_count], edge_count)
    node_count = 3
    full = _build(*pairs, node_count, with_source_csr=True)
    lean = _build(*pairs, node_count, with_source_csr=False)

    # Destination grouping, geometry, and node counts stay bit-identical.
    for field in ("n_node", "n_local", "source", "edge_vec", "destination_row_ptr"):
        torch.testing.assert_close(
            getattr(full, field), getattr(lean, field), rtol=0, atol=0
        )
    assert lean.n_node.dtype == torch.int64 and lean.n_local.dtype == torch.int64

    # The placeholder is a contiguous two-entry int64 zero on the index device.
    for tensor in (lean.source_row_ptr, lean.source_order):
        assert tensor.dtype == torch.int64
        assert tensor.shape == (2,)
        assert tensor.device == lean.source.device
        assert tensor.is_contiguous()
        torch.testing.assert_close(tensor, torch.zeros(2, dtype=torch.int64))

    # Both branches still satisfy the exported-shape contract.
    validate_canonical_graph_shapes(full, node_count)
    validate_canonical_graph_shapes(lean, node_count)


@pytest.mark.parametrize("edge_count", [0, 1])
def test_false_placeholder_is_not_repadded_for_short_graphs(edge_count: int) -> None:
    """Guard slots grow the edge axis, never the two-entry placeholder."""
    pairs = _pairs([0] * edge_count, [1] * edge_count, edge_count)
    lean = _build(*pairs, 2, with_source_csr=False)
    assert lean.source_row_ptr.shape == (2,)
    assert lean.source_order.shape == (2,)
    # The edge axis still carries the two guard slots of the deployment ABI.
    assert lean.source.shape == (2,)
    assert lean.edge_vec.shape == (2, 3)
    # And the full branch does pad its real source order there.
    full = _build(*pairs, 2, with_source_csr=True)
    assert full.source_order.shape == (2,)


@pytest.mark.parametrize(
    "physical_edges", ["empty", "one", "self-images"], ids=lambda name: name
)
def test_n1_storage2_boundary(physical_edges: str) -> None:
    """One node and two edge slots: placeholder and full CSR share their shape.

    ``E = 2`` uses the two periodic self-image edges of a lone node with
    opposite displacements. At this boundary the two-entry placeholder
    occupies exactly the slots the complete source CSR would, so the shape
    says nothing about the physical edge count and a consumer may still
    allocate its tiny scratch buffers.
    """
    vectors = {
        "empty": [],
        "one": [[1.0, 0.0, 0.0]],
        "self-images": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    }[physical_edges]
    count = len(vectors)
    center = torch.zeros(count, dtype=torch.int64)
    source = torch.zeros(count, dtype=torch.int64)
    vec = torch.tensor(vectors, dtype=torch.float64).reshape(count, 3)
    full = canonical_graph_from_pairs(center, source, vec, _nodes(1), 1)
    lean = canonical_graph_from_pairs(
        center, source, vec, _nodes(1), 1, with_source_csr=False
    )

    # Node counts, destination CSR, source indices and geometry are identical.
    for field in ("n_node", "n_local", "source", "edge_vec", "destination_row_ptr"):
        torch.testing.assert_close(
            getattr(full, field), getattr(lean, field), rtol=0, atol=0
        )
    # Both edge axes carry exactly two slots; the physical count is in the
    # row pointer, not in the axis length.
    assert full.source.shape == lean.source.shape == (2,)
    assert full.edge_vec.shape == lean.edge_vec.shape == (2, 3)
    assert int(lean.destination_row_ptr[-1]) == count

    # The placeholder is indistinguishable in shape from a complete source CSR
    # at this boundary, but its contents are zero on both entries.
    for tensor in (lean.source_row_ptr, lean.source_order):
        assert tensor.dtype == torch.int64 and tensor.is_contiguous()
        torch.testing.assert_close(tensor, torch.zeros(2, dtype=torch.int64))
    # With two real edges the complete branch is not a placeholder: it orders
    # both self-images under the single node.
    if count == 2:
        torch.testing.assert_close(
            full.source_order, torch.tensor([0, 1], dtype=torch.int64)
        )
        assert full.source_row_ptr.tolist() == [0, 2]
    validate_canonical_graph_shapes(full, 1)
    validate_canonical_graph_shapes(lean, 1)


def test_false_multi_node_and_multi_frame_counts() -> None:
    """Node counts are untouched by the omission across frames."""
    center = torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.int64)
    source = torch.tensor([1, 2, 0, 3, 0, 2], dtype=torch.int64)
    vec = torch.tensor([[1.0, 0.0, 0.0]] * 6, dtype=torch.float64)
    n_node = torch.tensor([2, 2], dtype=torch.int64)
    full = canonical_graph_from_pairs(center, source, vec, n_node, 4)
    lean = canonical_graph_from_pairs(
        center, source, vec, n_node, 4, with_source_csr=False
    )
    for field in ("n_node", "n_local", "source", "edge_vec", "destination_row_ptr"):
        torch.testing.assert_close(
            getattr(full, field), getattr(lean, field), rtol=0, atol=0
        )
    assert lean.source_row_ptr.tolist() == [0, 0]
    assert lean.source_row_ptr.device == full.source_row_ptr.device


def test_builder_forwards_the_option() -> None:
    """The vesin builder passes the option through to the pair assembler."""
    sentinel = mock.sentinel.graph
    # An empty node axis reaches the assembler without a neighbor search.
    with mock.patch(
        "deepmd.pt_expt.utils.vesin_graph_builder.canonical_graph_from_pairs",
        return_value=sentinel,
    ) as assembler:
        result = build_canonical_graph_vesin(
            torch.zeros((1, 0, 3), dtype=torch.float64),
            None,
            2.0,
            mock.Mock(),
            with_source_csr=False,
        )
    assert result is sentinel
    assert assembler.call_args.kwargs["with_source_csr"] is False
    # The default still forwards the complete-graph request.
    with mock.patch(
        "deepmd.pt_expt.utils.vesin_graph_builder.canonical_graph_from_pairs",
        return_value=sentinel,
    ) as assembler:
        build_canonical_graph_vesin(
            torch.zeros((1, 0, 3), dtype=torch.float64), None, 2.0, mock.Mock()
        )
    assert assembler.call_args.kwargs["with_source_csr"] is True


# Request propagation and capability gates.


def _odef(category: OutputVariableCategory) -> OutputVariableDef:
    return OutputVariableDef("var", [1], category=category.value)


_DERV_C = _odef(OutputVariableCategory.DERV_C)
_DERV_C_REDU = _odef(OutputVariableCategory.DERV_C_REDU)
_OUT = _odef(OutputVariableCategory.OUT)


def _make_evaluator(**metadata) -> PtExptDeepEval:
    """A DeepEval shell carrying only what the propagation decision reads."""
    evaluator = object.__new__(PtExptDeepEval)
    evaluator.metadata = {
        "dim_fparam": 0,
        "dim_aparam": 0,
        "dim_chg_spin": 0,
        **metadata,
    }
    evaluator._dpmodel = None
    evaluator._is_spin = False
    evaluator._charge_state_fold = None
    return evaluator


@pytest.mark.parametrize(
    "metadata",
    [
        {"lower_input_kind": "dpa1_canonical"},
        {"lower_input_kind": "dpa4c_canonical", "topologies": ("pme",)},
        {"lower_input_kind": "dpa4c_canonical", "has_message_passing": True},
        {"graph_edge_dtype": "float32"},
    ],
    ids=["dpa1", "long-range-topology", "message-passing", "generic-graph"],
)
def test_ineligible_layouts_keep_source_csr(metadata: dict) -> None:
    """DPA1, topologies, message passing, and generic graphs are untouched."""
    evaluator = _make_evaluator(canonical_requires_source_csr=False, **metadata)
    assert evaluator._wants_source_csr(False) is True
    assert evaluator._wants_source_csr(True) is True


def test_spin_keeps_source_csr() -> None:
    """The spin path keeps the default builder behaviour."""
    evaluator = _make_evaluator(
        lower_input_kind="dpa4c_canonical", canonical_requires_source_csr=False
    )
    evaluator._is_spin = True
    assert evaluator._wants_source_csr(False) is True


@pytest.mark.parametrize(
    "metadata",
    [
        {"lower_input_kind": "dpa4c_canonical"},
        {"lower_input_kind": "dpa4c_canonical", "canonical_requires_source_csr": True},
    ],
    ids=["field-missing", "capability-required"],
)
def test_missing_or_required_capability_keeps_source_csr(metadata: dict) -> None:
    """Only an exported ``False`` permits placeholders; absence is not a grant.

    An artifact that predates the field, and one whose lower reads the source
    CSR on every call, both keep the complete source CSR.
    """
    evaluator = _make_evaluator(**metadata)
    assert evaluator._wants_source_csr(False) is True
    assert evaluator._wants_source_csr(True) is True


@pytest.mark.parametrize("capability", [None, True, "false", 0])
def test_only_explicit_false_is_a_grant(capability) -> None:
    """No other value of the capability field opens the placeholder path."""
    evaluator = _make_evaluator(
        lower_input_kind="dpa4c_canonical", canonical_requires_source_csr=capability
    )
    assert evaluator._wants_source_csr(False) is True


def test_spin_entry_point_never_asks_for_omission() -> None:
    """``_eval_model_graph_spin`` leaves ``do_atomic_virial`` at its default."""
    evaluator = _make_evaluator(lower_input_kind="dpa4c_canonical")
    evaluator._is_spin = True
    evaluator._is_pt2 = False
    evaluator.exported_module = mock.Mock(return_value={})
    nframes, natoms = 1, 2
    with mock.patch.object(
        PtExptDeepEval,
        "_prepare_graph_inputs",
        return_value=((), nframes, natoms),
    ) as prepare:
        evaluator._eval_model_graph_spin(
            np.zeros((nframes, natoms, 3)),
            None,
            np.zeros((nframes, natoms), dtype=np.int32),
            np.zeros((nframes, natoms, 3)),
            None,
            None,
            [_OUT],
            None,
        )
    assert "do_atomic_virial" not in prepare.call_args.kwargs


def test_short_range_dpa4c_follows_the_request() -> None:
    """The eligible layout drops and restores the source CSR per call."""
    evaluator = _make_evaluator(
        lower_input_kind="dpa4c_canonical", canonical_requires_source_csr=False
    )
    assert evaluator._wants_source_csr(True) is True
    assert evaluator._wants_source_csr(False) is False
    # No cached state: the decision is a pure function of the arguments.
    assert evaluator._wants_source_csr(True) is True
    assert evaluator._wants_source_csr(False) is False


def _run(evaluator, request_defs) -> bool:
    """Evaluate request defs through ``_eval_model_graph``, capturing the flag.

    Only the graph build and the exported forward are stubbed; the request
    inspection inside ``_eval_model_graph`` runs for real.
    """
    nframes, natoms = 1, 2
    coords = np.zeros((nframes, natoms, 3))
    atom_types = np.zeros((nframes, natoms), dtype=np.int32)
    graph = mock.MagicMock()
    graph.n_node = torch.tensor([natoms], dtype=torch.int64)
    graph.n_local = graph.n_node
    graph.source = torch.zeros(2, dtype=torch.uint32)
    graph.edge_vec = torch.zeros((2, 3), dtype=torch.float32)
    graph.destination_row_ptr = torch.zeros(natoms + 1, dtype=torch.int64)
    graph.source_row_ptr = torch.zeros(2, dtype=torch.int64)
    graph.source_order = torch.zeros(2, dtype=torch.int64)
    evaluator._is_pt2 = False
    evaluator.exported_module = mock.Mock(return_value={})
    with mock.patch.object(
        PtExptDeepEval,
        "_build_device_graph",
        return_value=(graph, torch.zeros(natoms, dtype=torch.int64), nframes, natoms),
    ) as build:
        evaluator._eval_model_graph(
            coords, None, atom_types, None, None, request_defs, None
        )
    return build.call_args.kwargs["with_source_csr"]


def _eligible_evaluator() -> PtExptDeepEval:
    """A DPA4C evaluator whose export declines the source CSR."""
    return _make_evaluator(
        lower_input_kind="dpa4c_canonical", canonical_requires_source_csr=False
    )


def test_eval_model_graph_omits_for_frame_requests() -> None:
    """A frame-level request (no ``DERV_C``) omits the source CSR."""
    assert _run(_eligible_evaluator(), [_OUT, _DERV_C_REDU]) is False


def test_eval_model_graph_keeps_for_atomic_requests() -> None:
    """A per-atom virial request keeps the complete source CSR."""
    assert _run(_eligible_evaluator(), [_OUT, _DERV_C]) is True


def test_switching_requests_leaves_no_state() -> None:
    """False -> True -> False on one instance follows the current request only."""
    evaluator = _eligible_evaluator()
    assert _run(evaluator, [_OUT]) is False
    assert _run(evaluator, [_OUT, _DERV_C]) is True
    assert _run(evaluator, [_OUT]) is False


def test_eval_model_graph_keeps_without_the_capability() -> None:
    """Architecture alone never permits placeholders."""
    for metadata in (
        {"lower_input_kind": "dpa4c_canonical"},
        {"lower_input_kind": "dpa4c_canonical", "canonical_requires_source_csr": True},
    ):
        assert _run(_make_evaluator(**metadata), [_OUT, _DERV_C_REDU]) is True


def test_new_keywords_are_optional_and_default_to_true() -> None:
    """The builder and evaluator entry points stay callable without keywords."""
    center = torch.tensor([0], dtype=torch.int64)
    graph = canonical_graph_from_pairs(
        center, center, torch.zeros((1, 3), dtype=torch.float64), _nodes(1), 1
    )
    assert graph.source_row_ptr.shape == (2,)

    for method, keyword in (
        (PtExptDeepEval._prepare_graph_inputs, "do_atomic_virial"),
        (PtExptDeepEval._build_device_graph, "with_source_csr"),
        (PtExptDeepEval._build_eval_graph, "with_source_csr"),
        (build_canonical_graph_vesin, "with_source_csr"),
        (canonical_graph_from_pairs, "with_source_csr"),
    ):
        assert inspect.signature(method).parameters[keyword].default is True


# ------------------------------------------------------- GPU operator regression


def _operator_available() -> bool:
    """Whether the compiled DPA4C CUDA operator is registered."""
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        op_available,
    )

    return op_available()


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not _operator_available(),
    reason="CUDA and the compiled DPA4C operator are required",
)


@_GPU
@pytest.mark.parametrize("physical_edges", [0, 1, 2], ids=["E0", "E1", "E2"])
def test_gpu_frame_outputs_match_with_placeholders(
    physical_edges: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frame-level outputs of the fused operator ignore the source CSR.

    A single node with two edge storage slots is the boundary at which a
    two-entry placeholder and a complete source CSR have the same shape, so
    the operator cannot tell them apart by shape; its tiny scratch buffers may
    still be allocated. Comparing the frame energy, force and virial of the
    complete graph with those of the placeholders checks that the omission
    changes no frame-level result at ``E = 0, 1, 2``. The ``E = 2`` case uses
    the two periodic self-image edges of a lone node, with opposite
    displacements.
    """
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        build_compression_artifacts,
        ensure_registered,
    )

    from ..descriptor.test_dpa4c_cuda import (
        _build_descriptor,
        _energy_fitting,
        _fused_arguments,
    )

    ensure_registered()
    operator = torch.ops.deepmd.dpa4c_canonical_compress_energy_force
    descriptor = _build_descriptor(8)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    fitting = _energy_fitting(descriptor)
    vectors = {
        0: [],
        1: [[1.0, 0.0, 0.0]],
        2: [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    }[physical_edges]
    center = torch.zeros(physical_edges, dtype=torch.int64, device="cuda")
    source = torch.zeros(physical_edges, dtype=torch.int64, device="cuda")
    vec = torch.tensor(vectors, dtype=torch.float32, device="cuda").reshape(
        physical_edges, 3
    )
    atype = torch.zeros(1, dtype=torch.int64, device="cuda")

    results = []
    for with_source_csr in (True, False):
        graph = canonical_graph_from_pairs(
            center,
            source,
            vec,
            torch.ones(1, dtype=torch.int64, device="cuda"),
            1,
            with_source_csr=with_source_csr,
        )
        arguments = _fused_arguments(
            monkeypatch,
            descriptor,
            fitting,
            graph,
            atype,
            fitting.bias_atom_e[:, 0],
        )
        energy, _atom_energy, force, virial, _atom_virial, _mag, _mask = operator(
            *arguments
        )
        results.append((energy, force, virial))

    full, lean = results
    for actual, expected in zip(lean, full, strict=True):
        torch.testing.assert_close(actual, expected)
    # The two graphs differ only in the placeholder contents, not in shape, so
    # this compares equal-shape runs rather than an empty-graph shortcut.
    assert lean[0].shape == (1, 1)
