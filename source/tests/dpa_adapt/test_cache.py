# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for descriptor cache (desc_cache.py)."""

import numpy as np

from deepmd.dpa_adapt.data.desc_cache import (
    _cache_dir,
    _cache_key,
    _data_fingerprint,
    _per_system_cache_path,
    _system_fingerprint,
    ensure_per_system_cache,
)
from deepmd.dpa_adapt.data.loader import (
    load_data,
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
    np.save(sd / "coord.npy", np.random.rand(nframes, natoms * 3))
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


class TestCacheKey:
    def test_same_inputs_same_key(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        k1 = _cache_key([s], str(ckpt), "mean")
        k2 = _cache_key([s], str(ckpt), "mean")
        assert k1 == k2

    def test_different_pooling_different_key(self, tmp_path):
        s = _make_system(tmp_path, "s1")
        ckpt = tmp_path / "dummy.pt"
        ckpt.write_text("dummy")
        k1 = _cache_key([s], str(ckpt), "mean")
        k2 = _cache_key([s], str(ckpt), "mean+std")
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
        path = _per_system_cache_path(s)
        # Should be under the cache dir, not next to the original data
        assert "dpa_adapt" in str(path)
        assert path.suffix == ".npy"


class TestEnsurePerSystemCache:
    def _write_dummy_desc_cache(self, system, feat_dim=8, nframes=2):
        cache_path = _per_system_cache_path(system)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.zeros((nframes, feat_dim)))

    def test_all_cached_does_not_load_model(self, tmp_path, monkeypatch):
        s1 = _make_system(tmp_path, "sys1")
        s2 = _make_system(tmp_path, "sys2")
        self._write_dummy_desc_cache(s1)
        self._write_dummy_desc_cache(s2)

        called = []

        class FakeFineTuner:
            def __init__(inner_self, **kwargs):
                called.append(True)

            def _extract_features(inner_self, systems):
                return np.zeros((2, 8))

        monkeypatch.setattr(
            "deepmd.dpa_adapt.finetuner.DPAFineTuner",
            FakeFineTuner,
        )
        ensure_per_system_cache(
            [s1, s2],
            pretrained="/nonexistent/dummy.pt",
            pooling="mean",
        )
        assert called == [], "DPAFineTuner was called but all systems were cached"

    def test_some_missing_loads_model(self, tmp_path, monkeypatch):
        s1 = _make_system(tmp_path, "sys1")
        s2 = _make_system(tmp_path, "sys2")
        self._write_dummy_desc_cache(s1)

        called = []

        class FakeFineTuner:
            def __init__(inner_self, **kwargs):
                called.append(True)

            def _extract_features(inner_self, systems):
                return np.zeros((2, 8))

            _device = None

        monkeypatch.setattr(
            "deepmd.dpa_adapt.finetuner.DPAFineTuner",
            FakeFineTuner,
        )
        ensure_per_system_cache(
            [s1, s2],
            pretrained="/nonexistent/dummy.pt",
            pooling="mean",
        )
        assert len(called) == 1, (
            "DPAFineTuner should be called exactly once for the missing system"
        )
