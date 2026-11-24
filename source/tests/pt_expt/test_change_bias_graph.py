# SPDX-License-Identifier: LGPL-3.0-or-later
"""A bias update preserves the frozen graph's deployable interface."""

import copy
import zipfile

import numpy as np
import pytest

from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.pt_expt.entrypoints.main import (
    change_bias,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)

from ..common.dpmodel.test_dpa4_native_spin_model import (
    NATIVE_SPIN_CONFIG,
)
from ..dpa4_fixtures import (
    jitter_zero_arrays,
)


@pytest.mark.parametrize("spin", [False, True])
def test_graph_round_trip(tmp_path, spin):
    cfg = copy.deepcopy(NATIVE_SPIN_CONFIG)
    if not spin:
        del cfg["spin"]
    rng = np.random.default_rng(5959)
    model = jitter_zero_arrays(get_model(cfg).serialize(), rng)
    before, after = (str(tmp_path / name) for name in ["before.pt2", "after.pt2"])
    deserialize_to_file(before, {"model": model}, lower_kind="graph", with_comm=spin)
    change_bias(before, bias_value=[0.25, -0.5], output=after)
    atype = np.array([0, 1, 0, 1])
    coords = rng.uniform(1, 3, (1, 4, 3))
    kwargs = {"spin": rng.normal(size=coords.shape)} if spin else {}
    outputs = []
    for path in (before, after):
        data = serialize_from_file(path)
        assert data["lower_input_kind"] == "graph"
        assert data["do_atomic_virial"]
        assert data["has_comm_artifact"] == spin
        with zipfile.ZipFile(path) as archive:
            assert (
                "model/extra/forward_lower_with_comm.pt2" in archive.namelist()
            ) == spin
        outputs.append(
            DeepPot(path, auto_batch_size=False).eval(
                coords, np.eye(3) * 8, atype, **kwargs
            )
        )
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )

    old_bias = (
        BaseModel.deserialize(model).get_out_bias().detach().cpu().numpy().ravel()
    )
    delta = (np.array([0.25, -0.5]) - old_bias)[atype].sum()
    np.testing.assert_allclose(outputs[1][0] - outputs[0][0], delta, atol=1e-6)
    for old, new in zip(outputs[0][1:], outputs[1][1:], strict=True):
        np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("portable", [False, True])
def test_edge_lower_fails_before_output(tmp_path, portable):
    import json

    before = tmp_path / "edge.pt2"
    after = tmp_path / "after.pt2"
    with zipfile.ZipFile(before, "w") as archive:
        archive.writestr(
            "model/extra/metadata.json", json.dumps({"lower_input_kind": "edge_vec"})
        )
        if portable:
            archive.writestr("model/extra/model.json", "{}")
    with pytest.raises(ValueError, match="edge_vec"):
        change_bias(str(before), bias_value=[0.25], output=str(after))
    assert not after.exists()
