# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the mixed-type -> grouped marker converter (mark_groups).

Pure numpy / filesystem; no compiled deepmd backend needed, so these run
anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpa_adapt import mark_groups


def _write_mixed_system(
    set_dir,
    *,
    natoms: int = 5,
    masked_per_frame=(2, 1, 0),
    overpotential: float = 324.9,
) -> None:
    """Write an OER-style mixed-type set.000 (real_atom_types with -1, one label)."""
    set_dir.mkdir(parents=True, exist_ok=True)
    nframes = len(masked_per_frame)
    coord = np.random.default_rng(0).random((nframes, natoms * 3))
    box = np.tile((np.eye(3) * 20.0).reshape(9), (nframes, 1))
    real = np.zeros((nframes, natoms), dtype=np.int32)
    real[:, 0] = 1  # a second element, just to be non-trivial
    for frame, k in enumerate(masked_per_frame):
        if k:
            real[frame, natoms - k :] = -1
    label = np.full((nframes, 1), float(overpotential), dtype=np.float64)
    np.save(set_dir / "coord.npy", coord)
    np.save(set_dir / "box.npy", box)
    np.save(set_dir / "real_atom_types.npy", real)
    np.save(set_dir / "overpotential.npy", label)
    (set_dir.parent / "type.raw").write_text("0\n" * natoms)
    (set_dir.parent / "type_map.raw").write_text("Ni\nO\n")


def test_group_by_system_writes_group_id_and_pool_mask(tmp_path):
    sysdir = tmp_path / "Fe0.2Ni0.8" / "05"
    _write_mixed_system(sysdir / "set.000", natoms=5, masked_per_frame=(2, 1, 0))

    results = mark_groups(tmp_path, target="overpotential")
    assert len(results) == 1
    r = results[0]
    assert r.n_frames == 3
    assert r.n_groups == 1
    assert r.wrote_group_id and r.wrote_pool_mask

    set_dir = sysdir / "set.000"
    gid = np.load(set_dir / "group_id.npy")
    assert gid.dtype == np.int64
    assert gid.shape == (3,)
    assert gid.tolist() == [0, 0, 0]  # one group per system

    pool_mask = np.load(set_dir / "pool_mask.npy")
    assert pool_mask.dtype == np.float64
    assert pool_mask.shape == (3, 5)
    real = np.load(set_dir / "real_atom_types.npy")
    np.testing.assert_array_equal(pool_mask, (real >= 0).astype(np.float64))
    # O*/OH*/OOH*: 2 / 1 / 0 masked -> 3 / 4 / 5 pooled atoms
    assert pool_mask.sum(axis=1).tolist() == [3.0, 4.0, 5.0]


def test_discovers_deeply_nested_systems(tmp_path):
    # mimic dpdata/set_XX/{equation}/{natoms}/set.000 layout
    for si in (1, 2):
        for eq in ("Fe0.2Ni0.8", "Co1.0"):
            _write_mixed_system(
                tmp_path / "dpdata" / f"set_{si:02d}" / eq / "05" / "set.000"
            )
    results = mark_groups(tmp_path, target="overpotential")
    assert len(results) == 4
    assert all(r.wrote_group_id for r in results)
    # every leaf now has both markers
    for gid in tmp_path.rglob("group_id.npy"):
        assert np.load(gid).tolist() == [0, 0, 0]


def test_group_by_label_splits_distinct_labels(tmp_path):
    # one system holding two triplets with different overpotentials
    set_dir = tmp_path / "merged" / "set.000"
    set_dir.mkdir(parents=True)
    real = np.zeros((6, 4), dtype=np.int32)
    real[0, -2:] = -1  # a couple of masked atoms so pool_mask is exercised
    np.save(set_dir / "coord.npy", np.zeros((6, 12)))
    np.save(set_dir / "real_atom_types.npy", real)
    np.save(set_dir / "overpotential.npy", np.array([[1.0]] * 3 + [[2.0]] * 3))

    mark_groups(tmp_path, group_by="label", target="overpotential")
    gid = np.load(set_dir / "group_id.npy")
    assert gid.tolist() == [0, 0, 0, 1, 1, 1]


def test_group_by_int_chunks(tmp_path):
    set_dir = tmp_path / "chunked" / "set.000"
    set_dir.mkdir(parents=True)
    np.save(set_dir / "coord.npy", np.zeros((7, 3)))  # 7 frames, 1 atom
    mark_groups(tmp_path, group_by=3)
    gid = np.load(set_dir / "group_id.npy")
    assert gid.tolist() == [0, 0, 0, 1, 1, 1, 2]  # trailing remainder is its own group


def test_no_masked_atoms_skips_pool_mask(tmp_path):
    set_dir = tmp_path / "homog" / "set.000"
    set_dir.mkdir(parents=True)
    np.save(set_dir / "coord.npy", np.zeros((3, 9)))
    np.save(set_dir / "real_atom_types.npy", np.zeros((3, 3), dtype=np.int32))
    r = mark_groups(tmp_path)[0]
    assert r.wrote_group_id
    assert not r.wrote_pool_mask  # pool_mask defaults to 1.0 at train time
    assert not (set_dir / "pool_mask.npy").exists()


def test_overwrite_false_preserves_existing(tmp_path):
    set_dir = tmp_path / "sys" / "set.000"
    _write_mixed_system(set_dir, masked_per_frame=(1, 0, 0))
    np.save(set_dir / "group_id.npy", np.array([7, 7, 7], dtype=np.int64))

    mark_groups(tmp_path, target="overpotential", overwrite=False)
    assert np.load(set_dir / "group_id.npy").tolist() == [7, 7, 7]  # untouched

    mark_groups(tmp_path, target="overpotential", overwrite=True)
    assert np.load(set_dir / "group_id.npy").tolist() == [0, 0, 0]  # regenerated


def test_dry_run_writes_nothing(tmp_path):
    set_dir = tmp_path / "sys" / "set.000"
    _write_mixed_system(set_dir, masked_per_frame=(2, 1, 0))
    results = mark_groups(tmp_path, target="overpotential", dry_run=True)
    assert results[0].wrote_group_id and results[0].wrote_pool_mask
    assert not (set_dir / "group_id.npy").exists()
    assert not (set_dir / "pool_mask.npy").exists()


def test_no_systems_found_raises(tmp_path):
    from dpa_adapt.data.errors import DPADataError

    (tmp_path / "empty").mkdir()
    with pytest.raises(DPADataError):
        mark_groups(tmp_path / "empty")
