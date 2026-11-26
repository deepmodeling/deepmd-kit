# SPDX-License-Identifier: LGPL-3.0-or-later
"""Safety checks for the explicit same-run C++ model fixture reuse."""

import copy
import json
import zipfile

import numpy as np
import pytest

from .gen_common import (
    derive_pair_exclude_pt2,
    reuse_pt2,
)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from deepmd.pt_expt.utils import (
        serialization,
    )

    src = tmp_path / "source.pt2"
    dst = tmp_path / "target.pt2"
    src.write_bytes(b"a complete compiled fixture")
    data = {
        "model": {
            "type": "ener",
            "descriptor": {"sel": 8},
            "weights": np.array([0.25, 1.0], dtype=np.float64),
        },
        "time": "source provenance",
        "lower_input_kind": "nlist",
        "do_atomic_virial": True,
    }
    monkeypatch.setattr(serialization, "serialize_from_file", lambda path: data)
    return src, dst, data


def test_reuse_checks_model_not_provenance(archive):
    src, dst, source = archive
    expected = copy.deepcopy(source)
    expected["time"] = "different provenance"
    reuse_pt2(str(src), str(dst), expected)
    assert dst.read_bytes() == src.read_bytes()


@pytest.mark.parametrize("change", ["weights", "dtype", "architecture", "keys"])
def test_reuse_rejects_changed_model(archive, change):
    src, dst, source = archive
    expected = copy.deepcopy(source)
    model = expected["model"]
    if change == "weights":
        model["weights"][0] += 0.1
    elif change == "dtype":
        model["weights"] = model["weights"].astype(np.float32)
    elif change == "architecture":
        model["descriptor"]["sel"] = 16
    else:
        model["pair_exclude_types"] = [[0, 1]]
    with pytest.raises(ValueError, match="architecture or weights"):
        reuse_pt2(str(src), str(dst), expected)
    assert not dst.exists()


@pytest.mark.parametrize(
    "field,value", [("lower_input_kind", "graph"), ("do_atomic_virial", False)]
)
def test_reuse_rejects_changed_abi(archive, field, value):
    src, dst, source = archive
    expected = copy.deepcopy(source)
    source[field] = value
    with pytest.raises(ValueError, match="lower/atomic-virial contract"):
        reuse_pt2(str(src), str(dst), expected)
    assert not dst.exists()


@pytest.mark.parametrize("in_place", [False, True])
def test_pair_exclusion_changes_only_metadata(tmp_path, in_place):
    source = tmp_path / "source.pt2"
    target = source if in_place else tmp_path / "derived.pt2"
    contents = {
        "model/extra/metadata.json": {"pair_exclude_types": []},
        "model/extra/model.json": {"model": {"pair_exclude_types": []}},
        "model/extra/model_def_script.json": {"type_map": ["O", "H"]},
    }
    with zipfile.ZipFile(source, "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, json.dumps(value))
        archive.writestr("model/kernel.so", b"compiled bytes must not change")
    derive_pair_exclude_pt2(str(source), str(target), [[0, 1]])
    with zipfile.ZipFile(target) as archive:
        assert archive.read("model/kernel.so") == b"compiled bytes must not change"
        for name in contents:
            value = json.loads(archive.read(name))
            if name.endswith("/model.json"):
                value = value["model"]
            assert value["pair_exclude_types"] == [[0, 1]]


@pytest.mark.parametrize("lower_kind", ["nlist", "graph"])
@pytest.mark.parametrize("exclude_types", [[], [[0, 1]]])
def test_dpa1_serialization_keeps_pair_exclusion_metadata(lower_kind, exclude_types):
    pytest.importorskip("torch")
    from deepmd.dpmodel.model.model import (
        get_model,
    )
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.utils.serialization import (
        _collect_metadata,
    )

    # The CI fixtures reuse compiled lowers. Independently retain the real
    # dpmodel -> pt_expt -> C++ metadata export contract without another AOT
    # compile, including the no-exclusion control and both ingestion routes.
    config = {
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
            "seed": 1,
        },
        "fitting_net": {"neuron": [5, 5, 5], "resnet_dt": True, "seed": 1},
        "pair_exclude_types": exclude_types,
    }
    serialized = get_model(copy.deepcopy(config)).serialize()
    assert serialized["pair_exclude_types"] == exclude_types
    restored = BaseModel.deserialize(copy.deepcopy(serialized))
    assert restored.serialize()["pair_exclude_types"] == exclude_types
    metadata = json.loads(
        json.dumps(_collect_metadata(restored, lower_kind=lower_kind))
    )
    assert metadata["pair_exclude_types"] == exclude_types
    assert metadata["lower_input_kind"] == lower_kind
