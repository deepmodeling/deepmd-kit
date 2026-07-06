# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for ``data/raw/raw_to_set.sh``.

The script splits ``*.raw`` label files into per-set chunks and converts each
chunk into ``*.npy`` inside ``set.<pi>/``. Historically the per-set *move* block
omitted the global ``dipole.raw``/``polarizability.raw`` chunks, so datasets
carrying global dipole/polarizability labels silently lost them (the ``*.npy``
files were never generated). These tests lock in the move/convert symmetry for
every tensor label the script splits.
"""

import shutil
import subprocess
from pathlib import (
    Path,
)

import numpy as np
import pytest

# repo root: source/tests/common/test_raw_to_set.py -> parents[3]
RAW_TO_SET = Path(__file__).parents[3] / "data" / "raw" / "raw_to_set.sh"


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("split") is None,
    reason="raw_to_set.sh requires bash and split on PATH",
)
@pytest.mark.parametrize(
    "label,ncol",
    [
        ("dipole", 3),  # global dipole (regression: previously dropped)
        ("polarizability", 9),  # global polarizability (regression: previously dropped)
        ("atomic_dipole", 3),  # already-working path, documents intended symmetry
        ("atomic_polarizability", 9),  # already-working path
    ],
)
def test_raw_to_set_preserves_tensor_labels(
    tmp_path: Path, label: str, ncol: int
) -> None:
    """Every split tensor label must be converted to ``set.<pi>/<label>.npy``."""
    assert RAW_TO_SET.is_file(), f"missing script: {RAW_TO_SET}"

    nframes = 4
    nline_per_set = 2  # force multiple sets, exercising the per-set split/move loop
    rng = np.random.default_rng(0)

    # minimal always-required raw files
    box = rng.random((nframes, 9))
    coord = rng.random((nframes, 6))  # 2 atoms
    values = rng.random((nframes, ncol))
    np.savetxt(tmp_path / "box.raw", box)
    np.savetxt(tmp_path / "coord.raw", coord)
    np.savetxt(tmp_path / f"{label}.raw", values)

    subprocess.run(
        ["bash", str(RAW_TO_SET), str(nline_per_set)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    nset = -(-nframes // nline_per_set)  # ceil division
    recovered = []
    for ii in range(nset):
        set_dir = tmp_path / f"set.{ii:03d}"
        npy = set_dir / f"{label}.npy"
        assert npy.is_file(), f"{label}.npy was not generated in {set_dir.name}"
        recovered.append(np.load(npy))

    # round-trip: concatenated per-set values must match the original input
    np.testing.assert_allclose(np.concatenate(recovered, axis=0), values)

    # no orphaned split chunks should be left behind in the raw directory
    leftovers = list(tmp_path.glob(f"{label}.raw[0-9]*"))
    assert not leftovers, f"orphaned split chunks left in raw dir: {leftovers}"
