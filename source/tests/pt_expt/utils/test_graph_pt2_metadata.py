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


def _build_dpa1_data() -> dict:
    """Build a serialized dpmodel data dict for a dpa1(attn_layer=0) energy model."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    model = get_model(copy.deepcopy(DPA1_CONFIG))
    return {
        "model": model.serialize(),
        "model_def_script": copy.deepcopy(DPA1_CONFIG),
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
    # the static edge axis is baked into the AOTI artifact; E_max must be
    # persisted so the C++ conversion hub (PR-B B2) pads runtime edges to it.
    # E_max = ceil(1.25 * nloc_sample(7) * nnei(sum(sel)=30)) = 263.
    assert meta["edge_capacity"] == 263


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
