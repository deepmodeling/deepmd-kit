# SPDX-License-Identifier: LGPL-3.0-or-later
"""Preserve spin artifacts and diagnostics without compiling an AOT model."""

import importlib
import json
import runpy
import sys
import zipfile
from pathlib import (
    Path,
)

import numpy as np
import pytest

from .gen_common import (
    reuse_pt2,
)

_DPA3_MODELS = ("deeppot_dpa3_spin.pt2", "deeppot_dpa3_spin_mpi.pt2")


@pytest.fixture
def generator(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    module = importlib.import_module(".gen_spin", __package__)
    from deepmd import (
        infer,
    )
    from deepmd.dpmodel.utils import (
        serialization,
    )
    from deepmd.entrypoints import (
        convert_backend,
    )

    monkeypatch.setattr(module, "__file__", str(tmp_path / "gen_spin.py"))
    monkeypatch.setattr(module, "ensure_inductor_compiler", lambda: None)
    monkeypatch.setattr(module, "load_custom_ops", lambda: None)
    for name in ("deeppot_dpa_spin.pt2", *_DPA3_MODELS):
        (tmp_path / name).with_suffix(".yaml").write_text("fixture YAML")
    (tmp_path / "deeppot_dpa_spin_md0.pt2").write_bytes(b"model 0 generated this run")
    data = {"model": {"weights": np.array([0.25], dtype=np.float64)}}
    calls = {
        name: []
        for name in (
            "exports",
            "reuse",
            "loads",
            "evals",
            "refs",
            "events",
            "yaml_loads",
        )
    }

    def export(**kwargs):
        calls["exports"].append(kwargs)
        calls["events"].append("export")
        Path(kwargs["OUTPUT"]).write_bytes(b"separately exported model")

    def reuse(src, dst, model_data):
        calls["reuse"].append((src, dst, model_data))
        calls["events"].append("reuse")
        Path(dst).write_bytes(Path(src).read_bytes())

    def load_yaml(path):
        calls["yaml_loads"].append(path)
        return data

    class DeepPot:
        def __init__(self, path):
            self.path = Path(path).name
            calls["loads"].append(self.path)

        def eval(self, coord, box, atype, *, atomic, spin):
            natoms = len(atype)
            offset = 100.0 if box is None else 10.0
            # Distinct arrays catch switched fields, and distinct PBC/NoPBC
            # values catch writing both sidecar sections from the same eval.
            outputs = tuple(
                np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
                + offset
                + index
                for index, shape in enumerate(
                    (
                        (1, 1),
                        (1, natoms, 3),
                        (1, 9),
                        (1, natoms, 1),
                        (1, natoms, 9),
                        (1, natoms, 3),
                        (1, natoms, 1),
                    )
                )
            )
            calls["evals"].append(
                (
                    self.path,
                    coord.copy(),
                    None if box is None else box.copy(),
                    list(atype),
                    atomic,
                    spin.copy(),
                    outputs,
                )
            )
            calls["events"].append("eval")
            return outputs

    write_ref = module.write_expected_ref

    def write_expected(path, **kwargs):
        calls["refs"].append((path, kwargs))
        calls["events"].append("reference")
        write_ref(path, **kwargs)

    monkeypatch.setattr(convert_backend, "convert_backend", export)
    monkeypatch.setattr(serialization, "load_dp_model", load_yaml)
    monkeypatch.setattr(module, "reuse_pt2", reuse)
    monkeypatch.setattr(infer, "DeepPot", DeepPot)
    monkeypatch.setattr(module, "write_expected_ref", write_expected)
    return module, calls, tmp_path, data


def _assert_diagnostics(calls, directory):
    coord = np.array(
        [
            [12.83, 2.56, 2.18],
            [12.09, 2.87, 2.74],
            [0.25, 3.32, 1.68],
            [3.36, 3.00, 1.81],
            [3.51, 2.51, 2.60],
            [4.27, 3.22, 1.56],
        ]
    ).reshape(-1)
    spin = np.zeros((6, 3))
    spin[0] = [0.13, 0.02, 0.03]
    spin[3] = [0.14, 0.10, 0.12]
    assert calls["loads"] == ["deeppot_dpa_spin.pt2", "deeppot_dpa_spin.pth"]
    assert len(calls["evals"]) == 3
    for index, (path, actual_coord, box, atype, atomic, actual_spin, _) in enumerate(
        calls["evals"]
    ):
        assert path == f"deeppot_dpa_spin.{('pt2', 'pt2', 'pth')[index]}"
        assert atomic is True
        assert atype == [0, 1, 1, 0, 1, 1]
        np.testing.assert_array_equal(actual_coord, coord)
        np.testing.assert_array_equal(actual_spin, spin.reshape(-1))
        assert actual_coord.dtype == actual_spin.dtype == np.float64
        if index == 1:
            assert box is None
        else:
            np.testing.assert_array_equal(box, (np.eye(3) * 13.0).reshape(-1))
            assert box.dtype == np.float64

    assert len(calls["refs"]) == 1
    path, kwargs = calls["refs"][0]
    assert Path(path) == directory / "deeppot_dpa_spin.expected"
    assert kwargs["source_script"] == "source/tests/infer/gen_spin.py"
    assert list(kwargs["sections"]) == ["pbc", "nopbc"]
    for index, section in enumerate(kwargs["sections"].values()):
        _, force, virial, atom_e, atom_v, force_m, _ = calls["evals"][index][-1]
        expected = {
            "expected_e": atom_e[0, :, 0],
            "expected_f": force[0],
            "expected_fm": force_m[0],
            "expected_tot_v": virial[0],
            "expected_atom_v": atom_v[0],
        }
        assert section.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(section[key], expected[key])
    assert "[pbc]" in Path(path).read_text()
    assert "[nopbc]" in Path(path).read_text()


def _export(directory, name):
    args = {
        "INPUT": str((directory / name).with_suffix(".yaml")),
        "OUTPUT": str(directory / name),
    }
    if name.endswith(".pt2"):
        args["atomic_virial"] = True
    return args


def _complete_dpa3(directory):
    for name in _DPA3_MODELS:
        (directory / name).write_bytes(name.encode())


def test_standalone_preserves_all_exports_and_diagnostics(generator):
    module, calls, directory, _ = generator
    module.main()
    assert calls["exports"] == [
        _export(directory, name)
        for name in ("deeppot_dpa_spin.pth", "deeppot_dpa_spin.pt2", *_DPA3_MODELS)
    ]
    assert not calls["reuse"]
    assert not calls["yaml_loads"]
    _assert_diagnostics(calls, directory)


def test_dpa3_only_exports_both_variants_without_baseline_or_diagnostics(generator):
    module, calls, directory, _ = generator
    # A missing baseline must not trigger its construction in this mode.
    (directory / "deeppot_dpa_spin.yaml").unlink()
    module.main(dpa3_only=True)
    assert calls["exports"] == [_export(directory, name) for name in _DPA3_MODELS]
    for key in ("reuse", "loads", "evals", "refs", "yaml_loads"):
        assert not calls[key]
    for extension in ("yaml", "pth", "pt2", "expected"):
        assert not (directory / f"deeppot_dpa_spin.{extension}").exists()


def test_reuse_copies_checked_model0_then_exports_pth_and_keeps_diagnostics(generator):
    module, calls, directory, data = generator
    _complete_dpa3(directory)
    module.main(reuse_model_devi=True)
    assert calls["exports"] == [_export(directory, "deeppot_dpa_spin.pth")]
    assert len(calls["reuse"]) == 1
    src, dst, actual_data = calls["reuse"][0]
    assert Path(src) == directory / "deeppot_dpa_spin_md0.pt2"
    assert Path(dst) == directory / "deeppot_dpa_spin.pt2"
    assert actual_data is data
    assert calls["yaml_loads"] == [str(directory / "deeppot_dpa_spin.yaml")]
    assert calls["events"] == ["reuse", "export", "eval", "eval", "reference", "eval"]
    assert Path(dst).read_bytes() == Path(src).read_bytes()
    for name in _DPA3_MODELS:
        assert (directory / name).read_bytes() == name.encode()
    _assert_diagnostics(calls, directory)


@pytest.mark.parametrize("missing", (*_DPA3_MODELS, "deeppot_dpa_spin_md0.pt2"))
def test_reuse_requires_both_completed_producers_without_exporting(generator, missing):
    module, calls, directory, _ = generator
    _complete_dpa3(directory)
    (directory / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        module.main(reuse_model_devi=True)
    assert not any(calls.values())


def test_reuse_rejection_does_not_export_or_evaluate(generator, monkeypatch):
    module, calls, directory, _ = generator
    _complete_dpa3(directory)

    def reject(*args):
        raise ValueError("Model architecture or weights differ")

    monkeypatch.setattr(module, "reuse_pt2", reject)
    with pytest.raises(ValueError, match="architecture or weights"):
        module.main(reuse_model_devi=True)
    for key in ("exports", "loads", "evals", "refs"):
        assert not calls[key]
    assert not (directory / "deeppot_dpa_spin.pt2").exists()


def test_modes_are_mutually_exclusive(generator):
    module, calls, _, _ = generator
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.main(dpa3_only=True, reuse_model_devi=True)
    assert not any(calls.values())


def test_cli_modes_are_mutually_exclusive(monkeypatch):
    script = Path(__file__).with_name("gen_spin.py")
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, "argv", [str(script), "--dpa3-only", "--reuse-model-devi"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2


def test_committed_model0_reuses_for_baseline_but_seed2_is_rejected(tmp_path):
    pytest.importorskip("torch")
    from deepmd.dpmodel.utils.serialization import (
        load_dp_model,
    )
    from deepmd.pt_expt.utils.serialization import (
        _numpy_to_json_serializable,
    )

    directory = Path(__file__).parent
    base = load_dp_model(str(directory / "deeppot_dpa_spin.yaml"))
    md0 = load_dp_model(str(directory / "deeppot_dpa_spin_md0.yaml"))
    md1 = load_dp_model(str(directory / "deeppot_dpa_spin_md1.yaml"))
    assert base["time"] != md0["time"]
    assert base["model_def_script"] == md0["model_def_script"]

    def write_archive(path, data):
        # Use the real JSON encoding and archive reader. The placeholder
        # executable is never loaded or compiled.
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "model/extra/model.json", json.dumps(_numpy_to_json_serializable(data))
            )
            archive.writestr(
                "model/extra/metadata.json",
                json.dumps({"lower_input_kind": "nlist", "do_atomic_virial": True}),
            )
            archive.writestr(
                "model/extra/model_def_script.json",
                json.dumps(data["model_def_script"]),
            )
            archive.writestr("model/kernel.so", b"placeholder, never loaded")

    src = tmp_path / "md0.pt2"
    dst = tmp_path / "baseline.pt2"
    write_archive(src, md0)
    # This checks every serialized weight's dtype, shape and value, not only
    # seed or architecture, in the same direction used by gen_spin.py.
    reuse_pt2(str(src), str(dst), base)
    assert dst.read_bytes() == src.read_bytes()
    wrong = tmp_path / "md1.pt2"
    rejected = tmp_path / "rejected.pt2"
    write_archive(wrong, md1)
    with pytest.raises(ValueError, match="architecture or weights"):
        reuse_pt2(str(wrong), str(rejected), base)
    assert not rejected.exists()
