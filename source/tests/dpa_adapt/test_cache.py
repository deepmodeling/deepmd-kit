# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for descriptor cache (desc_cache.py)."""

import numpy as np

from dpa_adapt.data.desc_cache import (
    _cache_dir,
    _cache_key,
    _data_fingerprint,
    _per_system_cache_path,
    _system_fingerprint,
)
from dpa_adapt.data.loader import (
    load_data,
)
from dpa_adapt.finetuner import (
    ensure_per_system_cache,
)


def _make_system(tmp_path, name="sys", natoms=2, nframes=3, elements=None):
    """Create a minimal deepmd/npy system dir and load it via dpdata."""
    if elements is None:
        elements = ["H", "O"]
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "type.raw").write_text(
        "\n".join(str(i % len(elements)) for i in range(natoms)) + "\n"
    )
    (root / "type_map.raw").write_text("\n".join(elements) + "\n")
    sd = root / "set.000"
    sd.mkdir(exist_ok=True)
    np.save(sd / "coord.npy", np.random.default_rng().random((nframes, natoms * 3)))
    np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (nframes, 1)))
    return load_data(str(root))[0]


class TestSystemFingerprint:
    def test_same_data_same_fp(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        fp1 = _system_fingerprint(s)
        fp2 = _system_fingerprint(s)
        assert fp1 == fp2

    def test_different_data_different_fp(self, tmp_path):
        s1 = _make_system(tmp_path, "s1", nframes=3)
        s2 = _make_system(tmp_path, "s2", nframes=5)
        assert _system_fingerprint(s1) != _system_fingerprint(s2)

    def test_different_elements_different_fp(self, tmp_path):
        s1 = _make_system(tmp_path, "s1", elements=["H", "O"])
        s2 = _make_system(tmp_path, "s2", elements=["Cu", "O"])
        assert _system_fingerprint(s1) != _system_fingerprint(s2)

    def test_different_real_atom_types_different_fp(self, tmp_path):
        # Grouped systems: type.raw/atom_types is a uniform placeholder, so
        # coords/atom_types/cells alone are identical across these two
        # systems -- only set.000/real_atom_types.npy (per-frame, with -1
        # padding) differs. The fingerprint must still change.
        s1 = _make_system(tmp_path, "s1", natoms=3, nframes=2)
        s2 = _make_system(tmp_path, "s2", natoms=3, nframes=2)
        # force identical coords/box so only real_atom_types.npy differs
        coord = np.load(tmp_path / "s1" / "set.000" / "coord.npy")
        np.save(tmp_path / "s2" / "set.000" / "coord.npy", coord)
        s1, s2 = load_data(str(tmp_path / "s1"))[0], load_data(str(tmp_path / "s2"))[0]
        assert _system_fingerprint(s1) == _system_fingerprint(s2)  # sanity

        np.save(
            tmp_path / "s1" / "set.000" / "real_atom_types.npy",
            np.array([[0, 1, -1], [0, 1, 2]]),
        )
        np.save(
            tmp_path / "s2" / "set.000" / "real_atom_types.npy",
            np.array([[0, 1, 2], [0, 1, 2]]),
        )
        assert _system_fingerprint(s1) != _system_fingerprint(s2)

    def test_different_pool_mask_different_fp(self, tmp_path):
        s1 = _make_system(tmp_path, "s1", natoms=3, nframes=2)
        s2 = _make_system(tmp_path, "s2", natoms=3, nframes=2)

        np.save(
            tmp_path / "s1" / "set.000" / "pool_mask.npy",
            np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
        )
        np.save(
            tmp_path / "s2" / "set.000" / "pool_mask.npy",
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        )
        assert _system_fingerprint(s1) != _system_fingerprint(s2)


class TestFingerprint:
    def test_identical_data_same_fp(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        fp1 = _data_fingerprint([s])
        fp2 = _data_fingerprint([s])
        assert fp1 == fp2

    def test_different_data_different_fp(self, tmp_path):
        s1 = _make_system(tmp_path, "s1", nframes=3)
        s2 = _make_system(tmp_path, "s2", nframes=5)
        fp1 = _data_fingerprint([s1])
        fp2 = _data_fingerprint([s2])
        assert fp1 != fp2

    def test_system_order_changes_fp(self, tmp_path):
        s1 = _make_system(tmp_path, "s1", nframes=3)
        s2 = _make_system(tmp_path, "s2", nframes=5)
        fp1 = _data_fingerprint([s1, s2])
        fp2 = _data_fingerprint([s2, s1])
        assert fp1 != fp2


class TestCacheKey:
    def test_same_inputs_same_key(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        k1 = _cache_key([s], str(ckpt), None, "mean")
        k2 = _cache_key([s], str(ckpt), None, "mean")
        assert k1 == k2

    def test_different_pooling_different_key(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        k1 = _cache_key([s], str(ckpt), None, "mean")
        k2 = _cache_key([s], str(ckpt), None, "mean+std")
        assert k1 != k2

    def test_different_branch_different_key(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        k1 = _cache_key([s], str(ckpt), "Omat24", "mean")
        k2 = _cache_key([s], str(ckpt), "Domains_Drug", "mean")
        assert k1 != k2

    def test_different_checkpoint_different_key(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt1 = tmp_path / "dummy1.pt"
        ckpt2 = tmp_path / "dummy2.pt"
        ckpt1.write_text("dummy")
        ckpt2.write_text("different")
        k1 = _cache_key([s], str(ckpt1), None, "mean")
        k2 = _cache_key([s], str(ckpt2), None, "mean")
        assert k1 != k2

    def test_different_type_map_different_key(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        k1 = _cache_key([s], str(ckpt), None, "mean", type_map=("H", "O"))
        k2 = _cache_key([s], str(ckpt), None, "mean", type_map=("O", "H"))
        assert k1 != k2

    def test_different_system_order_different_key(self, tmp_path):
        s1 = _make_system(tmp_path, "s1", nframes=3)
        s2 = _make_system(tmp_path, "s2", nframes=5)
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        k1 = _cache_key([s1, s2], str(ckpt), None, "mean")
        k2 = _cache_key([s2, s1], str(ckpt), None, "mean")
        assert k1 != k2


class TestCacheDir:
    def test_respects_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        d = _cache_dir()
        assert str(tmp_path) in str(d)
        assert "dpa_adapt" in str(d)


class TestPerSystemCachePath:
    def test_uses_hash_not_path(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        path = _per_system_cache_path(s, str(ckpt))
        # Should be under the cache dir, not next to the original data
        assert "dpa_adapt" in str(path)
        assert path.suffix == ".npy"

    def test_includes_type_map(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        p1 = _per_system_cache_path(s, str(ckpt), type_map=("H", "O"))
        p2 = _per_system_cache_path(s, str(ckpt), type_map=("O", "H"))
        assert p1 != p2


class TestEnsurePerSystemCache:
    def _write_dummy_desc_cache(self, system, pretrained, feat_dim=8, nframes=2):
        cache_path = _per_system_cache_path(system, pretrained)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.zeros((nframes, feat_dim)))

    def test_all_cached_does_not_load_model(self, tmp_path, monkeypatch):
        s1 = _make_system(tmp_path, "sys1")
        s2 = _make_system(tmp_path, "sys2")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        self._write_dummy_desc_cache(s1, str(ckpt))
        self._write_dummy_desc_cache(s2, str(ckpt))

        called = []

        class FakeFineTuner:
            def __init__(self, **kwargs):
                called.append(True)

            def _extract_features(self, systems):
                return np.zeros((2, 8))

        monkeypatch.setattr(
            "dpa_adapt.finetuner.DPAFineTuner",
            FakeFineTuner,
        )
        ensure_per_system_cache(
            [s1, s2],
            pretrained=str(ckpt),
            pooling="mean",
        )
        assert called == [], "DPAFineTuner was called but all systems were cached"

    def test_some_missing_loads_model(self, tmp_path, monkeypatch):
        s1 = _make_system(tmp_path, "sys1")
        s2 = _make_system(tmp_path, "sys2")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        self._write_dummy_desc_cache(s1, str(ckpt))

        called = []

        class FakeFineTuner:
            def __init__(self, **kwargs):
                called.append(True)

            def _extract_features(self, systems):
                return np.zeros((2, 8))

            _device = None

        monkeypatch.setattr(
            "dpa_adapt.finetuner.DPAFineTuner",
            FakeFineTuner,
        )
        ensure_per_system_cache(
            [s1, s2],
            pretrained=str(ckpt),
            pooling="mean",
        )
        assert len(called) == 1, (
            "DPAFineTuner should be called exactly once for the missing system"
        )
