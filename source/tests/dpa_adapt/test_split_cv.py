# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for train_test_split() and cross_validate()."""

import json
import os
import tempfile
from pathlib import (
    Path,
)

import numpy as np
import pytest

from deepmd.dpa_adapt.cv import (
    _build_fold_groups,
    _extract_formula,
    _formula_to_group,
    cross_validate,
    train_test_split,
)
from deepmd.dpa_adapt.data.loader import (
    load_data,
)


def _write_system(
    root: str,
    natoms: int = 2,
    nframes: int = 3,
    label_key: str = "energy",
    elements: list[str] = None,
):
    """Create a deepmd/npy system dir, load it, return dpdata.System."""
    if elements is None:
        elements = ["H", "O"]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n_atoms = len(elements)
    (root / "type.raw").write_text(
        "\n".join(str(i % n_atoms) for i in range(natoms)) + "\n"
    )
    (root / "type_map.raw").write_text("\n".join(elements) + "\n")
    sdir = root / "set.000"
    sdir.mkdir(exist_ok=True)
    np.save(sdir / "coord.npy", np.zeros((nframes, natoms * 3)))
    np.save(sdir / "box.npy", np.tile(np.eye(3).ravel(), (nframes, 1)))
    np.save(sdir / f"{label_key}.npy", np.ones((nframes, 1)))
    return load_data(str(root))[0]


def _write_oer_tree(
    tmpdir: str, formulas: list[str], nsets: int = 3, label_key: str = "energy"
) -> list:
    """Create an OER-style tree and return loaded dpdata.System objects."""
    systems = []
    for formula in formulas:
        for s in range(1, nsets + 1):
            sysdir = Path(tmpdir) / f"set_{s:02d}" / formula / "353"
            sys = _write_system(str(sysdir), natoms=10, nframes=3, label_key=label_key)
            systems.append(sys)
    return sorted(systems, key=lambda s: s._dpa_source)


def _make_manifest(
    formula_parts: list[list[str]], test: list[str], tag: str = "ni"
) -> str:
    m = {
        "meta": {"mode": "stratified", "k": len(formula_parts), "seed": 123},
        "co": {"test": [], "parts": []},
        tag: {"test": test, "parts": formula_parts},
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    Path(path).write_text(json.dumps(m))
    return path


class TestExtractFormula:
    def test_oer_path_from_dpa_source(self, tmp_path):
        sys = _write_system(str(tmp_path / "set_01" / "Ni0.5Fe0.5O2H1" / "353"))
        assert "Ni0.5Fe0.5O2H1" in _extract_formula(sys)

    def test_formula_to_group(self, tmp_path):
        s1 = _write_system(str(tmp_path / "set_01" / "A" / "1"))
        s2 = _write_system(str(tmp_path / "set_02" / "A" / "1"))
        s3 = _write_system(str(tmp_path / "set_01" / "B" / "1"))
        groups = _formula_to_group([s1, s2, s3])
        assert groups == ["A", "A", "B"]


class TestBuildFoldGroups:
    def test_three_folds(self):
        parts = [["A", "B"], ["C", "D"], ["E"]]
        path = _make_manifest(parts, test=["F"])
        folds, test = _build_fold_groups(path)
        assert len(folds) == 3
        assert folds[0] == {"A", "B"}
        assert test == {"F"}


class TestTrainTestSplit:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp = tmp_path
        formulas = [f"Comp{i}" for i in range(10)]
        self.systems = _write_oer_tree(
            str(tmp_path), formulas, nsets=2, label_key="energy"
        )

    def test_manifest_split(self):
        parts = [
            ["Comp0", "Comp1"],
            ["Comp2", "Comp3"],
            ["Comp4", "Comp5"],
            ["Comp6", "Comp7"],
            ["Comp8"],
        ]
        mpath = _make_manifest(parts, test=["Comp9"])
        train, valid, test = train_test_split(self.systems, manifest=mpath)
        assert len(train) == 16, f"got {len(train)}"
        assert len(valid) == 2
        assert len(test) == 2
        t = set(_formula_to_group(train))
        v = set(_formula_to_group(valid))
        e = set(_formula_to_group(test))
        assert len(t & v) == 0
        assert len(t & e) == 0
        assert "Comp9" in e
        assert "Comp8" in v

    def test_group_by_formula(self):
        train, valid, test = train_test_split(
            self.systems,
            group_by="formula",
            test_size=0.1,
            valid_size=0.2,
            seed=42,
        )
        t = set(_formula_to_group(train))
        v = set(_formula_to_group(valid))
        e = set(_formula_to_group(test))
        assert len(t & v) == 0
        assert len(t & e) == 0
        assert len(v & e) == 0

    def test_group_by_explicit_list(self):
        groups = _formula_to_group(self.systems)
        train, valid, test = train_test_split(
            self.systems,
            group_by=groups,
            test_size=0.1,
            valid_size=0.1,
            seed=42,
        )
        t = set(_formula_to_group(train))
        v = set(_formula_to_group(valid))
        assert len(t & v) == 0

    def test_no_group_by_raises(self):
        with pytest.raises(ValueError, match="Either manifest"):
            train_test_split(self.systems)


class TestCrossValidate:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp = tmp_path
        formulas = [f"Comp{i}" for i in range(5)]
        self.systems = _write_oer_tree(
            str(tmp_path), formulas, nsets=2, label_key="energy"
        )

    def test_expensive_cv_guard(self):
        class FakeModel:
            strategy = "finetune"
            pretrained = None
            model_branch = None
            pooling = "mean"

        with pytest.raises(ValueError, match="allow_expensive_cv"):
            cross_validate(
                FakeModel(),
                self.systems,
                label_key="energy",
                cv=3,
                group_by="formula",
            )

    def test_invalid_granularity(self):
        class FakeModel:
            strategy = "frozen_sklearn"
            pretrained = None
            model_branch = None
            pooling = "mean"

        with pytest.raises(ValueError, match="granularity"):
            cross_validate(
                FakeModel(),
                self.systems,
                label_key="energy",
                cv=5,
                group_by="formula",
                granularity="invalid",
            )

    def test_invalid_cv_value(self):
        class FakeModel:
            strategy = "frozen_sklearn"
            pretrained = None
            model_branch = None
            pooling = "mean"

        with pytest.raises(ValueError, match="cv must be"):
            cross_validate(
                FakeModel(),
                self.systems,
                label_key="energy",
                cv=1,
                group_by="formula",
            )


class TestStandardScalerConsistency:
    def test_same_predictions_on_same_data(self):
        from sklearn.linear_model import (
            Ridge,
        )
        from sklearn.pipeline import (
            make_pipeline,
        )
        from sklearn.preprocessing import (
            StandardScaler,
        )

        from deepmd.dpa_adapt.cv import (
            _build_sklearn_head,
        )

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 32))
        y = rng.normal(size=(100,))

        head1 = make_pipeline(StandardScaler(), _build_sklearn_head("ridge", seed=42))
        head1.fit(X, y)
        pred1 = head1.predict(X)

        head2 = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
        head2.fit(X, y)
        pred2 = head2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestDeterministicCV:
    """Ensures cross_validate with frozen_sklearn + GroupKFold is deterministic."""

    def test_deterministic_folds_same_result_twice(self, tmp_path, monkeypatch):
        formulas = [f"Comp{i}" for i in range(4)]
        systems = _write_oer_tree(str(tmp_path), formulas, nsets=2, label_key="energy")

        rng = np.random.default_rng(42)
        n_total = len(systems) * 3  # 3 frames each
        n_total = sum(1 for _ in tmp_path.rglob("set.000"))
        raise pytest.skip("needs real DPA checkpoint to extract descriptors")

    def test_manifest_folds(self, tmp_path, monkeypatch):
        raise pytest.skip("needs real DPA checkpoint to extract descriptors")
