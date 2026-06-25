# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for load_dataset()."""

import logging
from pathlib import (
    Path,
)

import numpy as np
import pytest

from dpa_adapt.data.dataset import (
    load_dataset,
)
from dpa_adapt.data.errors import (
    DPADataError,
)
from dpa_adapt.data.loader import (
    load_data,
)


def _write_system(
    root: str,
    natoms: int = 2,
    nframes: int = 3,
    label_key: str = "energy",
    elements: list[str] | None = None,
) -> Path:
    """Create a minimal deepmd/npy system directory. Returns its Path."""
    if elements is None:
        elements = ["H", "O"]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "type.raw").write_text(
        "\n".join(str(i % len(elements)) for i in range(natoms)) + "\n"
    )
    (root / "type_map.raw").write_text("\n".join(elements) + "\n")
    sdir = root / "set.000"
    sdir.mkdir(exist_ok=True)
    np.save(sdir / "coord.npy", np.zeros((nframes, natoms * 3)))
    np.save(sdir / "box.npy", np.tile(np.eye(3).ravel(), (nframes, 1)))
    np.save(sdir / f"{label_key}.npy", np.zeros((nframes, 1)))
    return root


class TestLoadDataset:
    def test_label_filter(self, tmp_path):
        root = _write_system(str(tmp_path / "sys1"), label_key="energy")
        # load_dataset resolves "energy" → "energies" via alias
        systems = load_dataset(str(root), label_key="energy")
        assert len(systems) == 1

    def test_label_filter_skips_missing(self, tmp_path, caplog):
        root = _write_system(str(tmp_path / "sys1"), label_key="energy")
        caplog.set_level(logging.WARNING, logger="dpa_adapt.data.dataset")
        with pytest.raises(DPADataError, match="no valid systems"):
            load_dataset(str(root), label_key="nonexistent")

    def test_explicit_list(self, tmp_path):
        s1 = load_data(str(_write_system(str(tmp_path / "s1"), label_key="energy")))[0]
        s2 = load_data(str(_write_system(str(tmp_path / "s2"), label_key="energy")))[0]
        systems = load_dataset([s1, s2], label_key="energy")
        assert len(systems) == 2

    def test_single_path(self, tmp_path):
        root = _write_system(str(tmp_path / "s1"), label_key="energy")
        systems = load_dataset(str(root), label_key="energy")
        assert len(systems) == 1

    def test_no_label_filter_raises_when_all_skipped(self, tmp_path):
        root = _write_system(str(tmp_path / "s1"), label_key="energy")
        with pytest.raises(DPADataError):
            load_dataset(str(root), label_key="bandgap")
