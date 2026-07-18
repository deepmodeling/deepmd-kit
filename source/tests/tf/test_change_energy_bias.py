# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for ``change_energy_bias_lower`` frame selection (``ntest``).

``change-bias --numb-batch`` is threaded through as ``ntest``; ``0`` must mean
"use all frames per system" rather than selecting zero frames.
"""

import shutil
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.tf.fit.ener import (
    change_energy_bias_lower,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


class _MockDP:
    """Minimal DeepEval stub that records how many frames it is asked to predict."""

    def __init__(self, natoms: int) -> None:
        self._natoms = natoms
        self.frames_evaluated: list[int] = []

    def get_dim_fparam(self) -> int:
        return 0

    def get_dim_aparam(self) -> int:
        return 0

    def eval(self, coord, box, atype, mixed_type=False, fparam=None, aparam=None):
        n = coord.shape[0]
        self.frames_evaluated.append(n)
        return (
            np.zeros([n, 1]),
            np.zeros([n, self._natoms, 3]),
            np.zeros([n, 9]),
        )


class TestChangeEnergyBiasNumbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.nframes = 10
        self.test_size = 4  # < nframes so multiple test frames are available
        self.natoms = 3
        self.type_map = ["O", "H"]
        atype = np.array([0, 1, 1])

        set_dir = self.tmp / "set.000"
        set_dir.mkdir(parents=True)
        rng = np.random.default_rng(0)
        np.save(set_dir / "coord.npy", rng.random((self.nframes, self.natoms * 3)))
        np.save(
            set_dir / "box.npy",
            np.tile((np.eye(3) * 10.0).reshape(9), (self.nframes, 1)),
        )
        np.save(set_dir / "energy.npy", rng.random((self.nframes, 1)))
        (self.tmp / "type.raw").write_text("\n".join(str(t) for t in atype))
        (self.tmp / "type_map.raw").write_text("\n".join(self.type_map))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build_data(self) -> DeepmdDataSystem:
        data = DeepmdDataSystem(
            [str(self.tmp)], 1, self.test_size, 6.0, type_map=self.type_map
        )
        data.add("energy", 1, atomic=False, must=True, high_prec=True)
        return data

    def _frames_evaluated(self, ntest: int) -> list[int]:
        data = self._build_data()
        dp = _MockDP(self.natoms)
        change_energy_bias_lower(
            data,
            dp,
            self.type_map,
            self.type_map,
            np.zeros(len(self.type_map)),
            bias_adjust_mode="change-by-statistic",
            ntest=ntest,
        )
        return dp.frames_evaluated

    def test_numb_batch_controls_frames(self) -> None:
        # number of test frames actually available for this system
        nframes_test = self._build_data().data_systems[0].get_test()["box"].shape[0]
        self.assertGreater(nframes_test, 1)

        # a positive ntest caps to that many frames per system
        self.assertEqual(self._frames_evaluated(1), [1])
        # ntest == 0 means all frames in the system
        self.assertEqual(self._frames_evaluated(0), [nframes_test])
