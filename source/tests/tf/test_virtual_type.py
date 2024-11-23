# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test virtual atomic type."""

import os
import unittest

import numpy as np

from deepmd.tf.infer import (
    DeepPot,
)
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.tf.utils.neighbor_stat import (
    NeighborStat,
)

from .common import (
    gen_data,
    infer_path,
    j_loader,
)


class TestVirtualType(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("virtual_type.pbtxt")),
            "virtual_type.pb",
        )
        cls.dp = DeepPot("virtual_type.pb")
        os.remove("virtual_type.pb")

    def setUp(self) -> None:
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ]
        )
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = None

    def test_virtual_type(self) -> None:
        nloc = len(self.atype)
        nghost = 10
        e1, f1, v1, ae1, av1 = self.dp.eval(
            self.coords.reshape([1, -1]), self.box, self.atype, atomic=True
        )
        e2, f2, v2, ae2, av2 = self.dp.eval(
            np.concatenate(
                [self.coords.reshape([1, -1]), np.zeros((1, nghost * 3))], axis=1
            ),
            self.box,
            self.atype + [-1] * nghost,
            atomic=True,
        )
        np.testing.assert_almost_equal(e1, e2)
        np.testing.assert_almost_equal(f1, f2[:, :nloc])
        np.testing.assert_almost_equal(v1, v2)
        np.testing.assert_almost_equal(ae1, ae2[:, :nloc])
        np.testing.assert_almost_equal(av1, av2[:, :nloc])

    def test_infer_mixed_type(self) -> None:
        nloc = len(self.atype)
        nghost = 10
        e, f, v, ae, av = self.dp.eval(
            np.concatenate(
                [
                    self.coords.reshape([1, -1]),
                    np.zeros((1, nghost * 3)),
                    np.zeros((1, nghost * 3)),
                    self.coords.reshape([1, -1]),
                ],
                axis=1,
            ).reshape(2, -1),
            None,
            np.array(self.atype + [-1] * nghost + [-1] * nghost + self.atype).reshape(
                2, -1
            ),
            atomic=True,
            mixed_type=True,
        )
        e1 = e[0]
        f1 = f[0]
        v1 = v[0]
        ae1 = ae[0]
        av1 = av[0]
        e2 = e[1]
        f2 = f[1]
        v2 = v[1]
        ae2 = ae[1]
        av2 = av[1]
        np.testing.assert_almost_equal(e1, e2)
        np.testing.assert_almost_equal(f1[:nloc], f2[nghost:])
        np.testing.assert_almost_equal(v1, v2)
        np.testing.assert_almost_equal(ae1[:nloc], ae2[nghost:])
        np.testing.assert_almost_equal(av1[:nloc], av2[nghost:])


class TestTrainVirtualType(unittest.TestCase):
    def setUp(self) -> None:
        gen_data(mixed_type=True, virtual_type=True)

    def test_data_mixed_type(self) -> None:
        jfile = "water_se_atten_mixed_type.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        batch_size = 1
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]
        type_map = jdata["model"]["type_map"]

        data = DeepmdDataSystem(systems, batch_size, test_size, rcut, type_map=type_map)
        data.get_batch()
        # neighbor stat
        nei_stat = NeighborStat(len(type_map), rcut, mixed_type=True)
        min_nbor_dist, max_nbor_size = nei_stat.get_stat(data)
