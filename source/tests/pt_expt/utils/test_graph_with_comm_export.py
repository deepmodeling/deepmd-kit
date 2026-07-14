# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-form ``.pt2`` export: the with-comm nested artifact (task 10).

Message-passing graph descriptors (dpa2's repformer block) need a SECOND
AOTI artifact -- ``model/extra/forward_lower_with_comm.pt2`` -- that accepts
8 extra positional comm tensors and drives per-layer cross-rank ghost-atom
exchange via ``deepmd_export::border_op``. This mirrors the dense
with-comm flow (``test_graph_pt2_metadata.py`` covers the plain-graph
``lower_input_kind`` branch; this file covers the *comm* branch on top of
it).
"""

import copy
import json
import zipfile

import pytest

from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)

# Small graph-eligible dpa2 descriptor: tebd_input_mode defaults to
# "concat", use_three_body defaults to False -> uses_graph_lower() is True,
# and has_message_passing_across_ranks() is unconditionally True for any
# DPA2 (delegates to DescrptBlockRepformers, which always needs cross-rank
# g1 exchange) -- so this model always gets a with-comm artifact.
DPA2_CONFIG = {
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

# dpa1 with attn_layer == 0: graph-eligible but NOT message-passing
# (has_message_passing_across_ranks() is False for a single se_atten
# descriptor) -- the regression pin that the with-comm artifact stays
# absent for non-GNN graph-eligible descriptors.
DPA1_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_atten",
        "sel": 10,
        "rcut_smth": 2.0,
        "rcut": 4.0,
        "neuron": [2, 4],
        "axis_neuron": 2,
        "attn": 4,
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
        "neuron": [4, 4],
        "resnet_dt": True,
        "seed": 1,
    },
}


def _build_data(config: dict) -> dict:
    """Build a serialized dpmodel data dict (same shape as ``dp freeze`` input)."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )

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
def dpa2_dpmodel_data() -> dict:
    return _build_data(DPA2_CONFIG)


@pytest.fixture(scope="module")
def dpa1_dpmodel_data() -> dict:
    return _build_data(DPA1_CONFIG)


def test_dpa2_graph_pt2_embeds_with_comm_artifact(dpa2_dpmodel_data, tmp_path) -> None:
    """dpa2 (message-passing) graph ``.pt2`` embeds the nested with-comm artifact."""
    p = str(tmp_path / "m_dpa2_graph.pt2")
    deserialize_to_file(
        p,
        copy.deepcopy(dpa2_dpmodel_data),
        do_atomic_virial=True,
        lower_kind="graph",
    )
    with zipfile.ZipFile(p, "r") as zf:
        names = zf.namelist()
    assert "model/extra/forward_lower_with_comm.pt2" in names

    meta = _read_metadata(p)
    assert meta["lower_input_kind"] == "graph"
    assert meta["has_comm_artifact"] is True
    assert meta["has_message_passing"] is True


def test_dpa2_graph_with_comm_aparam_freeze(tmp_path) -> None:
    """A ``numb_aparam > 0`` message-passing model freezes to the graph
    ``.pt2`` with the nested with-comm artifact.

    Regression (OutisLi review): the with-comm dynamic-shape spec gave
    ``aparam``'s node axis an independent ``nloc`` Dim while the graph ABI
    carries ``aparam`` on the same flat node axis as ``atype``; the graph
    fitting views ``aparam`` against the flat node count, so
    ``torch.export`` proved the two axes equal and rejected every
    ``numb_aparam > 0`` with-comm graph freeze with ``Constraints violated
    (nloc): aparam.size(1) and atype.size(0) must always be equal``. The
    aparam node axis now IS ``atype``'s ``n_node_total`` axis (flat
    ``(N, nda)`` ABI), so both the regular and the with-comm graph traces
    of this freeze must succeed.
    """
    cfg = copy.deepcopy(DPA2_CONFIG)
    cfg["fitting_net"] = {**cfg["fitting_net"], "numb_aparam": 1}
    data = _build_data(cfg)
    p = str(tmp_path / "m_dpa2_graph_aparam.pt2")
    deserialize_to_file(
        p,
        data,
        do_atomic_virial=True,
        lower_kind="graph",
    )
    with zipfile.ZipFile(p, "r") as zf:
        names = zf.namelist()
    assert "model/extra/forward_lower_with_comm.pt2" in names

    meta = _read_metadata(p)
    assert meta["lower_input_kind"] == "graph"
    assert meta["has_comm_artifact"] is True


def test_dpa1_graph_pt2_has_no_comm_artifact(dpa1_dpmodel_data, tmp_path) -> None:
    """dpa1 (non-message-passing) graph ``.pt2`` regression pin: no nested entry."""
    p = str(tmp_path / "m_dpa1_graph.pt2")
    deserialize_to_file(
        p,
        copy.deepcopy(dpa1_dpmodel_data),
        do_atomic_virial=True,
        lower_kind="graph",
    )
    with zipfile.ZipFile(p, "r") as zf:
        names = zf.namelist()
    assert "model/extra/forward_lower_with_comm.pt2" not in names

    meta = _read_metadata(p)
    assert meta["lower_input_kind"] == "graph"
    assert meta["has_comm_artifact"] is False
