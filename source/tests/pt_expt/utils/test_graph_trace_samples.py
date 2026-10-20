# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph trace coverage must not depend on dense neighbor capacities."""

from unittest.mock import patch

import pytest
import torch

from deepmd.pt_expt.utils.serialization import (
    build_synthetic_graph_inputs,
    count_synthetic_graph_edges,
)


class _GraphSampleModel:
    def __init__(self, ntypes=118):
        self.ntypes = ntypes

    def get_rcut(self):
        return 4.0

    def get_type_map(self):
        return [f"T{i}" for i in range(self.ntypes)]

    def get_sel(self):
        raise AssertionError("Graph samples must not query dense capacities")

    def get_dim_fparam(self):
        return 0

    def get_dim_aparam(self):
        return 0

    def get_dim_chg_spin(self):
        return 0


@pytest.mark.parametrize("nloc", [4, 7])
@pytest.mark.parametrize("ntypes", [1, 118])
@pytest.mark.parametrize("canonicalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_graph_sample_covers_sparse_topology_and_high_types(
    nloc, ntypes, canonicalize, dtype
):
    model = _GraphSampleModel(ntypes)
    kwargs = {"nframes": 2, "nloc": nloc, "dtype": dtype, "device": torch.device("cpu")}
    with patch(
        "deepmd.pt_expt.utils.serialization.build_neighbor_list",
        side_effect=AssertionError("Graph samples must not build dense lists"),
    ):
        real = count_synthetic_graph_edges(model, **kwargs)
        sample = build_synthetic_graph_inputs(
            model, real + 5, canonicalize=canonicalize, **kwargs
        )
        repeated = build_synthetic_graph_inputs(
            model, real + 5, canonicalize=canonicalize, **kwargs
        )
    for actual, expected in zip(sample, repeated, strict=True):
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    atype, n_node, n_local, edge_index, edge_vec, edge_mask = sample[:6]
    assert atype.shape == (2 * nloc,)
    assert atype[0] == atype[nloc] == ntypes - 1
    assert (atype < ntypes - 1).any() == (ntypes > 1)
    assert n_node.tolist() == [nloc, nloc]
    assert n_local.tolist() == [nloc - 1, nloc - 1]
    assert edge_mask.sum() == real
    assert (~edge_mask).sum() >= 5
    real_edges = edge_index[:, edge_mask]
    # An owned center in each frame has no incoming or outgoing real edge.
    assert not torch.isin(real_edges, torch.tensor([1, nloc + 1])).any()
    assert 1 < n_local.min()
    # The highest-type node has two guaranteed neighbors and occurs at
    # both ends of directed edges.
    assert (real_edges[0] == 0).sum() >= 2
    assert (real_edges[1] == 0).sum() >= 2
    assert torch.isfinite(edge_vec).all()
    assert all(value is not None for value in sample[6:10]) == canonicalize


def test_graph_export_high_type_sparse_runtime():
    from deepmd.pt_expt.model import (
        get_model,
    )
    from deepmd.pt_expt.utils import (
        env,
    )
    from deepmd.pt_expt.utils.serialization import (
        _trace_and_export,
    )

    config = {
        "type_map": [f"T{i}" for i in range(118)],
        "descriptor": {
            "type": "se_atten",
            "sel": 416,
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "neuron": [3, 6],
            "axis_neuron": 2,
            "attn_layer": 0,
            "seed": 42,
        },
        "fitting_net": {"neuron": [8, 8], "seed": 42},
    }
    model = get_model(config).to(env.DEVICE).eval()
    exported, metadata, _, _ = _trace_and_export(
        {"model": model.serialize()}, model_json_override=None, lower_kind="graph"
    )
    # Dense compatibility metadata remains available to existing C++ readers.
    assert metadata["sel"] == model.get_sel()
    assert metadata["nnei"] == sum(model.get_sel())
    loaded = exported.module()
    for nframes, nloc in ((1, 1), (1, 4), (3, 8)):
        inputs = list(
            build_synthetic_graph_inputs(
                model,
                None,
                nframes=nframes,
                nloc=nloc,
                dtype=torch.float64,
                device=env.DEVICE,
            )
        )
        inputs[2] = inputs[1].clone()  # Runtime may own every atom.
        actual = loaded(*inputs)
        reference = model.forward_common_lower_graph(*inputs[:10])
        torch.testing.assert_close(
            actual["energy"], reference["energy_redu"], rtol=1e-10, atol=1e-10
        )
        for key, ref_key in (
            ("force", "energy_derv_r"),
            ("virial", "energy_derv_c_redu"),
        ):
            torch.testing.assert_close(
                actual[key],
                reference[ref_key].reshape(actual[key].shape),
                rtol=1e-10,
                atol=1e-10,
            )
