# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mixed-system batching must pad Hessian labels as square blocks.

``DeepmdDataSystem.get_batch()`` merges frames from systems with different
atom counts.  A Hessian is a ``(3 * natoms, 3 * natoms)`` matrix, so neither
the non-atomic concatenate nor the atomic flat-prefix copy can pad it: the
former raises on ragged systems and the latter would scatter the rows.  The
JAX trainer calls ``get_batch()`` directly, so this is reachable from a
``batch_size: "mixed:N"`` run with Hessian supervision.
"""

import shutil
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

NATOMS = (2, 3)
NFRAMES = 2
# One constant per system, so a mis-scattered block is visible as a stray value.
HESSIAN_VALUES = (1.0, 2.0)


def _write_system(root: Path, natoms: int, value: float) -> None:
    """Write a one-set system whose Hessian is a constant square block."""
    root.mkdir(parents=True)
    (root / "type.raw").write_text("\n".join(["0"] * natoms) + "\n")
    (root / "type_map.raw").write_text("A\n")
    set_dir = root / "set.000"
    set_dir.mkdir()
    rng = np.random.default_rng(natoms)
    np.save(set_dir / "coord.npy", rng.random((NFRAMES, natoms * 3)))
    np.save(set_dir / "box.npy", np.tile(np.eye(3).reshape(9) * 20.0, (NFRAMES, 1)))
    np.save(set_dir / "energy.npy", rng.random((NFRAMES, 1)))
    dof = natoms * 3
    np.save(
        set_dir / "hessian.npy",
        np.full((NFRAMES, dof * dof), value, dtype=np.float64),
    )


class TestMixedBatchHessian(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        systems = []
        for natoms, value in zip(NATOMS, HESSIAN_VALUES, strict=True):
            path = self.tmpdir / f"sys_{natoms}"
            _write_system(path, natoms, value)
            systems.append(str(path))
        self.ds = DeepmdDataSystem(systems, "mixed:2", 1, 2.0)
        self.ds.add("energy", 1, atomic=False, must=True, high_prec=True)
        self.ds.add("hessian", 1, atomic=False, must=True, special_shape="hessian")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _ragged_batch(self) -> dict:
        """Merge exactly one frame from each system.

        ``get_batch_mixed`` picks systems at random, so it does not reliably
        produce the ragged case this covers.
        """
        batch_data = []
        for sys_idx in range(self.ds.nsystems):
            bb_data = self.ds.data_systems[sys_idx].get_batch(1)
            bb_data["natoms_vec"] = self.ds.natoms_vec[sys_idx]
            bb_data["default_mesh"] = self.ds.default_mesh[sys_idx]
            batch_data.append(bb_data)
        return self.ds._merge_batch_data(batch_data)

    def test_mixed_batch_embeds_each_hessian_block(self) -> None:
        """Each frame's block sits in the top-left of the padded square."""
        batch = self._ragged_batch()
        max_natoms = int(batch["natoms_vec"][0])
        self.assertEqual(max_natoms, max(NATOMS))
        padded_dof = max_natoms * 3

        self.assertEqual(batch["hessian"].shape, (len(NATOMS), padded_dof * padded_dof))
        hessian = batch["hessian"].reshape(-1, padded_dof, padded_dof)

        for frame, (natoms, value) in enumerate(
            zip(NATOMS, HESSIAN_VALUES, strict=True)
        ):
            frame_dof = natoms * 3
            self.assertEqual(int(batch["real_natoms_vec"][frame, 0]), natoms)
            np.testing.assert_array_equal(hessian[frame, :frame_dof, :frame_dof], value)
            # Everything outside the block is padding and must stay zero.
            np.testing.assert_array_equal(hessian[frame, frame_dof:, :], 0.0)
            np.testing.assert_array_equal(hessian[frame, :, frame_dof:], 0.0)

    def test_get_batch_accepts_a_mixed_hessian_batch(self) -> None:
        """The public entry point must not raise for any sampled combination."""
        for _ in range(20):
            batch = self.ds.get_batch()
            padded_dof = int(batch["natoms_vec"][0]) * 3
            self.assertEqual(batch["hessian"].shape[1], padded_dof * padded_dof)


if __name__ == "__main__":
    unittest.main()
