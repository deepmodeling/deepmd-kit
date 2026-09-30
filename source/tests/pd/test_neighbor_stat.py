# SPDX-License-Identifier: LGPL-3.0-or-later
import shutil
import unittest

import dpdata
import numpy as np
import paddle

from deepmd.entrypoints.neighbor_stat import (
    neighbor_stat,
)
from deepmd.pd.utils.env import (
    DEVICE,
)
from deepmd.pd.utils.neighbor_stat import (
    NeighborStatOP,
)

from ..seed import (
    GLOBAL_SEED,
)


def gen_sys(nframes):
    rng = np.random.default_rng(GLOBAL_SEED)
    natoms = 1000
    data = {}
    X, Y, Z = np.mgrid[0:2:3j, 0:2:3j, 0:2:3j]
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # + 0.1
    data["coords"] = np.repeat(positions[np.newaxis, :, :], nframes, axis=0)
    data["forces"] = rng.random([nframes, natoms, 3])
    data["cells"] = np.array([3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0]).reshape(
        1, 3, 3
    )
    data["energies"] = rng.random([nframes, 1])
    data["atom_names"] = ["TYPE"]
    data["atom_numbs"] = [27]
    data["atom_types"] = np.repeat(0, 27)
    return data


class TestNeighborStat(unittest.TestCase):
    def setUp(self):
        data0 = gen_sys(1)
        sys0 = dpdata.LabeledSystem()
        sys0.data = data0
        sys0.to_deepmd_npy("system_0", set_size=1)

    def tearDown(self):
        shutil.rmtree("system_0")

    def test_neighbor_stat(self):
        for rcut in (0.0, 1.0, 2.0, 4.0):
            for mixed_type in (True, False):
                with self.subTest(rcut=rcut, mixed_type=mixed_type):
                    rcut += 1e-3  # prevent numerical errors
                    min_nbor_dist, max_nbor_size = neighbor_stat(
                        system="system_0",
                        rcut=rcut,
                        type_map=["TYPE", "NO_THIS_TYPE"],
                        mixed_type=mixed_type,
                        backend="paddle",
                    )
                    upper = np.ceil(rcut) + 1
                    X, Y, Z = np.mgrid[-upper:upper, -upper:upper, -upper:upper]
                    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                    # distance to (0,0,0)
                    distance = np.linalg.norm(positions, axis=1)
                    expected_neighbors = np.count_nonzero(
                        np.logical_and(distance > 0, distance <= rcut)
                    )
                    self.assertAlmostEqual(min_nbor_dist, 1.0, 6)
                    ret = [expected_neighbors]
                    if not mixed_type:
                        ret.append(0)
                    np.testing.assert_array_equal(max_nbor_size, ret)


class TestNeighborStatOP(unittest.TestCase):
    def test_virtual_atoms_do_not_affect_statistics(self) -> None:
        """Ignore virtual atoms as both statistics centers and neighbors."""
        # Atom 1 is virtual and overlaps atom 0. Without both masks, it either
        # sets the minimum distance to zero or inflates the maximum type-0 count.
        coord = paddle.to_tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ]
            ],
            dtype=paddle.float64,
            place=DEVICE,
        ).reshape([1, -1])
        atype = paddle.to_tensor([[0, -1, 0, 1]], dtype=paddle.int64, place=DEVICE)
        expected_min_rr2 = np.array([[1.0, np.inf, 1.0, 4.0]])

        for cell in (
            None,
            10.0 * paddle.eye(3, dtype=paddle.float64).reshape([1, 9]).to(DEVICE),
        ):
            for mixed_types in (False, True):
                with self.subTest(cell=cell is not None, mixed_types=mixed_types):
                    min_rr2, max_nnei = NeighborStatOP(
                        ntypes=2,
                        rcut=1.1,
                        mixed_types=mixed_types,
                    )(coord, atype, cell)

                    np.testing.assert_allclose(min_rr2.numpy(), expected_min_rr2)
                    expected_max_nnei = [[1]] if mixed_types else [[1, 0]]
                    np.testing.assert_array_equal(max_nnei.numpy(), expected_max_nnei)
