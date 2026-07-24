# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for grouped descriptor aggregation."""

from __future__ import (
    annotations,
)

import numpy as np

from dpa_adapt.finetuner import (
    DPAFineTuner,
)
from dpa_adapt.grouped._aggregation import (
    aggregate_weighted_groups,
)
from dpa_adapt.grouped._offline import (
    GroupedDataset,
)


def _write_system(
    root,
    name,
    label=2.5,
    group_id=None,
    weight=None,
    n_atoms=2,
    n_frames=1,
):
    sys_dir = root / name
    sys_dir.mkdir(parents=True)
    type_rows = "\n".join(str(i % 2) for i in range(n_atoms)) + "\n"
    (sys_dir / "type.raw").write_text(type_rows)
    (sys_dir / "type_map.raw").write_text("H\nO\n")
    set_dir = sys_dir / "set.000"
    set_dir.mkdir()
    np.save(set_dir / "coord.npy", np.zeros((n_frames, n_atoms * 3)))
    np.save(set_dir / "box.npy", np.tile(np.eye(3).ravel(), (n_frames, 1)))
    np.save(set_dir / "energy.npy", np.full((n_frames,), label, dtype=float))
    np.save(set_dir / "property.npy", np.full((n_frames,), label, dtype=float))
    if group_id is not None:
        np.save(
            set_dir / "group_id.npy", np.full((n_frames,), group_id, dtype=np.int64)
        )
    if weight is not None:
        np.save(set_dir / "weight.npy", np.asarray(weight, dtype=float))
    return sys_dir


def test_aggregate_weighted_groups_core():
    features = np.array([[1.0, 0.0], [3.0, 2.0], [10.0, 5.0]])
    group_ids = np.array([2, 1, 2])
    weights = np.array([0.25, 1.0, 0.75])
    labels = np.array([8.0, 4.0, 8.0])

    embeddings, group_labels, ordered_ids = aggregate_weighted_groups(
        features, group_ids, weights, labels
    )

    np.testing.assert_array_equal(ordered_ids, np.array([1, 2]))
    np.testing.assert_allclose(embeddings, np.array([[3.0, 2.0], [7.75, 3.75]]))
    np.testing.assert_allclose(group_labels, np.array([4.0, 8.0]))


def test_grouped_dataset_weighted_embedding(monkeypatch, tmp_path):
    parent = tmp_path / "data"
    sys_a = _write_system(parent, "a", group_id=0, weight=[0.7, 0.3], n_frames=2)
    expected_rows = {
        str(sys_a.resolve()): np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
    }

    def fake_load_or_extract(systems, **kwargs):
        return np.vstack([expected_rows[system._dpa_source] for system in systems])

    monkeypatch.setattr(
        "dpa_adapt.grouped._offline.load_or_extract",
        fake_load_or_extract,
    )

    dataset = GroupedDataset(str(parent), pretrained="fake.pt")

    np.testing.assert_allclose(
        dataset.get_embeddings(),
        np.array([[0.7, 1.4, 2.1]]) + np.array([[3.0, 6.0, 9.0]]),
    )
    np.testing.assert_allclose(dataset.get_labels(), np.array([2.5]))


def test_grouped_dataset_target_key_list_stacks_multi_property_labels(
    monkeypatch, tmp_path
):
    """The CLI's --target-key a,b (comma-separated) reaches GroupedDataset as
    target_key=["a", "b"]; each key must be read from its own set.*/{key}.npy
    and stacked into one (n_groups, 2) label column per group, the same
    convention the non-grouped _load_labels() multi-property path uses.
    """
    parent = tmp_path / "data"
    sys_a = _write_system(parent, "a", group_id=0, weight=[1.0], n_frames=1)
    # two extra label columns beyond the default energy/property ones
    np.save(sys_a / "set.000" / "gap.npy", np.array([1.5]))
    np.save(sys_a / "set.000" / "homo.npy", np.array([-4.0]))

    def fake_load_or_extract(systems, **kwargs):
        return np.ones((1, 2))

    monkeypatch.setattr(
        "dpa_adapt.grouped._offline.load_or_extract",
        fake_load_or_extract,
    )

    dataset = GroupedDataset(
        str(parent), pretrained="fake.pt", target_key=["gap", "homo"]
    )

    assert dataset.get_labels().shape == (1, 2)
    np.testing.assert_allclose(dataset.get_labels(), np.array([[1.5, -4.0]]))


def test_grouped_dataset_group_ids_are_scoped_per_system(monkeypatch, tmp_path):
    parent = tmp_path / "data"
    sys_a = _write_system(parent, "a", label=1.0, group_id=0, weight=[0.7])
    sys_b = _write_system(parent, "b", label=2.0, group_id=0, weight=[0.3])
    rows = {
        str(sys_a.resolve()): np.array([[1.0, 2.0]]),
        str(sys_b.resolve()): np.array([[10.0, 20.0]]),
    }

    def fake_load_or_extract(systems, **kwargs):
        return np.vstack([rows[system._dpa_source] for system in systems])

    monkeypatch.setattr(
        "dpa_adapt.grouped._offline.load_or_extract",
        fake_load_or_extract,
    )

    dataset = GroupedDataset(str(parent), pretrained="fake.pt")

    np.testing.assert_allclose(dataset.get_embeddings(), [[0.7, 1.4], [3.0, 6.0]])
    np.testing.assert_allclose(dataset.get_labels(), [1.0, 2.0])


def test_grouped_dataset_missing_weight_defaults_to_one(monkeypatch, tmp_path):
    parent = tmp_path / "data"
    sys_a = _write_system(parent, "a", group_id=0, weight=None, n_frames=2)
    rows = {
        str(sys_a.resolve()): np.array([[1.0, 2.0], [10.0, 20.0]]),
    }

    def fake_load_or_extract(systems, **kwargs):
        return np.vstack([rows[system._dpa_source] for system in systems])

    monkeypatch.setattr(
        "dpa_adapt.grouped._offline.load_or_extract",
        fake_load_or_extract,
    )

    dataset = GroupedDataset(str(parent), pretrained="fake.pt")

    np.testing.assert_allclose(dataset.get_embeddings(), [[11.0, 22.0]])
    np.testing.assert_allclose(dataset.get_labels(), [2.5])


def test_grouped_fit_and_predict(monkeypatch, tmp_path):
    parent = tmp_path / "data"
    sys_a = _write_system(
        parent, "a", label=4.0, group_id=0, weight=[0.7, 0.3], n_frames=2
    )
    rows = {
        str(sys_a.resolve()): np.array([[2.0, 0.0], [0.0, 8.0]]),
    }
    calls = []

    def fake_load_or_extract(systems, **kwargs):
        calls.append(kwargs)
        return np.vstack([rows[system._dpa_source] for system in systems])

    monkeypatch.setattr(
        "dpa_adapt.grouped._offline.load_or_extract",
        fake_load_or_extract,
    )

    model = DPAFineTuner(
        pretrained="fake.pt", strategy="frozen_sklearn", predictor="linear"
    )
    model.fit(str(parent))
    result = model.predict(str(parent))

    assert result.predictions.shape == (1, 1)
    np.testing.assert_allclose(
        model.predictor[:-1].transform([[1.4, 2.4]]), [[0.0, 0.0]]
    )
    assert calls[0]["pooling"] == "mean"


def test_plain_input_keeps_existing_fit_path(monkeypatch, tmp_path):
    sys_dir = _write_system(tmp_path, "plain", n_frames=2)
    called = []

    def fake_fit(self, data, type_map=None, target_key=None, labels=None, fmt=None):
        called.append((data, target_key))
        self._fitted = True

    monkeypatch.setattr(DPAFineTuner, "_fit_sklearn", fake_fit)

    model = DPAFineTuner(pretrained="fake.pt", strategy="frozen_sklearn")
    model.fit(str(sys_dir), target_key="energy")

    assert called == [(str(sys_dir), "energy")]
