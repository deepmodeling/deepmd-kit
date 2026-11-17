# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check fixture selection without compiling CUDA-only inference operators."""

import importlib
from pathlib import (
    Path,
)

import pytest


@pytest.fixture
def generator(monkeypatch):
    pytest.importorskip("torch")
    # Standalone gen scripts import their sibling helpers by their short names.
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    module = importlib.import_module(".gen_dpa4c_long_range", __package__)
    from deepmd.pt_expt.utils import (
        serialization,
    )

    calls = {"graph": [], "compressed": [], "mesh": [], "metadata": [], "reference": []}
    monkeypatch.setattr(module, "ensure_inductor_compiler", lambda: None)
    monkeypatch.setattr(module, "load_custom_ops", lambda: None)
    monkeypatch.setattr(module, "_model_dict", lambda cfg: {"config": cfg})
    monkeypatch.setattr(serialization, "mesh_freeze_supported", lambda: True)
    monkeypatch.setattr(
        serialization,
        "deserialize_to_file",
        lambda path, data, **kwargs: calls["graph"].append((path, data, kwargs)),
    )
    for name, key in (
        ("_export_compressed", "compressed"),
        ("_write_mesh_ref", "mesh"),
        ("_check_metadata", "metadata"),
        ("_write_ref", "reference"),
    ):
        monkeypatch.setattr(
            module, name, lambda *args, key=key: calls[key].append(args)
        )
    return module, calls


@pytest.mark.parametrize("compressed_only", [False, True])
def test_kokkos_selection_keeps_both_compressed_models(
    generator, monkeypatch, compressed_only
):
    module, calls = generator
    monkeypatch.setenv("DP_VARIANT", "cuda")
    module.main(compressed_only=compressed_only)
    assert [Path(path).name for path, _ in calls["compressed"]] == [
        "deeppot_dpa4c_long_range_compressed.pt2",
        "deeppot_dpa4c_long_range_periodic_compressed.pt2",
    ]
    assert calls["compressed"][0][1] == module.COMPRESSED_CONFIG
    assert calls["compressed"][1][1] == module._periodic(module.COMPRESSED_CONFIG)
    assert len(calls["graph"]) == (0 if compressed_only else 2)
    assert len(calls["mesh"]) == (0 if compressed_only else 1)
    assert len(calls["metadata"]) == (2 if compressed_only else 4)
    assert len(calls["reference"]) == (2 if compressed_only else 4)


def test_cpu_default_preserves_graph_and_mesh_fixtures(generator, monkeypatch):
    module, calls = generator
    monkeypatch.delenv("DP_VARIANT", raising=False)
    module.main()
    assert len(calls["graph"]) == 2
    assert len(calls["mesh"]) == 1
    assert not calls["compressed"]
    assert all(
        call[2] == {"do_atomic_virial": True, "lower_kind": "graph"}
        for call in calls["graph"]
    )


def test_compressed_only_rejects_non_cuda_generation(generator, monkeypatch):
    module, calls = generator
    monkeypatch.delenv("DP_VARIANT", raising=False)
    with pytest.raises(ValueError, match="requires DP_VARIANT=cuda"):
        module.main(compressed_only=True)
    assert not any(calls.values())


def test_old_torch_keeps_mesh_reference_and_compressed_periodic_model(
    generator, monkeypatch
):
    module, calls = generator
    from deepmd.pt_expt.utils import (
        serialization,
    )

    monkeypatch.setenv("DP_VARIANT", "cuda")
    monkeypatch.setattr(serialization, "mesh_freeze_supported", lambda: False)
    module.main()
    assert len(calls["graph"]) == 1
    assert len(calls["mesh"]) == 1
    assert len(calls["compressed"]) == 2
    assert len(calls["reference"]) == 3
