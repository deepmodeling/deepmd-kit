# SPDX-License-Identifier: LGPL-3.0-or-later
"""Do not transfer unused types to the direct canonical Vesin builder."""

from unittest.mock import (
    patch,
)

import numpy as np
import pytest
import torch

from deepmd.pt_expt.infer.deep_eval import (
    DeepEval,
)


def make_evaluator(method):
    evaluator = object.__new__(DeepEval)
    evaluator._neighbor_graph_method = method
    evaluator._rcut = 3.0
    evaluator._vesin_search = object()
    evaluator.metadata = {"graph_edge_dtype": "float32"}
    return evaluator


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "virtual",
        "empty",
        "strided",
        "broadcast",
        "readonly",
        "list",
        "cpu_tensor",
        "virtual_cpu_tensor",
    ],
)
def test_direct_canonical_only_transfers_consumed_types(dtype, case):
    n = 0 if case == "empty" else 4
    atom_types = np.zeros((1, n), dtype=dtype)
    if case in ("virtual", "virtual_cpu_tensor"):
        atom_types[0, -1] = -1
    elif case == "strided":
        atom_types = np.zeros((1, 2 * n), dtype=dtype)[:, ::2]
    elif case == "broadcast":
        atom_types = np.broadcast_to(np.array([[1]], dtype=dtype), (1, n))
    elif case == "readonly":
        atom_types.flags.writeable = False
    elif case == "list":
        atom_types = atom_types.tolist()
    if case in ("cpu_tensor", "virtual_cpu_tensor"):
        atom_types = torch.from_numpy(atom_types)
    before = np.asarray(atom_types).copy()
    evaluator = make_evaluator("vesin")
    transfers = []
    masks = []
    original = torch.as_tensor

    def record(value, *args, **kwargs):
        if kwargs.get("dtype") == torch.int64:
            transfers.append(np.array(value, copy=True))
        if kwargs.get("dtype") == torch.bool:
            masks.append(np.array(value, copy=True))
        return original(value, *args, **kwargs)

    with (
        patch.object(DeepEval, "_model_pair_excl", return_value=None),
        patch("torch.as_tensor", side_effect=record),
        patch(
            "deepmd.pt_expt.utils.vesin_graph_builder.build_canonical_graph_vesin"
        ) as builder,
    ):
        result = evaluator._build_eval_graph(
            np.zeros((1, n, 3)), atom_types, None, torch.device("cpu"), canonical=True
        )
    assert result is builder.return_value
    assert not transfers
    if case in ("virtual", "virtual_cpu_tensor"):
        assert len(masks) == 1
        np.testing.assert_array_equal(masks[0], before >= 0)
        assert torch.equal(
            builder.call_args.kwargs["node_mask"], torch.tensor(before >= 0)
        )
    else:
        assert not masks
        assert builder.call_args.kwargs["node_mask"] is None
    np.testing.assert_array_equal(atom_types, before)


@pytest.mark.parametrize(
    "canonical,excluded", [(False, False), (False, True), (True, True)]
)
def test_generic_or_excluded_vesin_still_receives_types(canonical, excluded):
    evaluator = make_evaluator("vesin")
    atom_types = np.array([[0, 1, -1]], dtype=np.int32)
    exclusion = object() if excluded else None
    with (
        patch.object(DeepEval, "_model_pair_excl", return_value=exclusion),
        patch(
            "deepmd.pt_expt.utils.vesin_graph_builder.build_neighbor_graph_vesin"
        ) as builder,
    ):
        evaluator._build_eval_graph(
            np.zeros((1, 3, 3)),
            atom_types,
            None,
            torch.device("cpu"),
            canonical=canonical,
        )
    assert builder.call_args.args[1].dtype == torch.int64
    assert torch.equal(
        builder.call_args.args[1], torch.tensor(atom_types, dtype=torch.int64)
    )
    assert builder.call_args.kwargs["pair_excl"] is exclusion


@pytest.mark.parametrize("method", ["cell", "nv"])
@pytest.mark.parametrize("canonical", [False, True])
def test_other_device_builders_still_receive_types(method, canonical):
    evaluator = make_evaluator(method)
    atom_types = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    target = (
        f"deepmd.pt_expt.utils.{method}_graph_builder.build_neighbor_graph_{method}"
    )
    with (
        patch.object(DeepEval, "_model_pair_excl", return_value=None),
        patch(target) as builder,
    ):
        evaluator._build_eval_graph(
            np.zeros((2, 3, 3)),
            atom_types,
            None,
            torch.device("cpu"),
            canonical=canonical,
        )
    assert torch.equal(builder.call_args.args[1], torch.tensor(atom_types))


def test_single_frame_fused_cell_keeps_its_type_input():
    evaluator = make_evaluator("cell")
    atom_types = np.array([[0, 1, 0]], dtype=np.int64)
    with (
        patch.object(DeepEval, "_model_pair_excl", return_value=None),
        patch(
            "deepmd.pt_expt.utils.cell_graph_builder.build_neighbor_graph_fused"
        ) as builder,
    ):
        evaluator._build_eval_graph(
            np.zeros((1, 3, 3)), atom_types, None, torch.device("cpu"), canonical=True
        )
    assert torch.equal(builder.call_args.args[1], torch.tensor(atom_types.reshape(-1)))
