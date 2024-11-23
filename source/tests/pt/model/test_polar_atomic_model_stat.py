# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)
from typing import (
    NoReturn,
    Optional,
)

import h5py
import numpy as np
import torch

from deepmd.pt.model.atomic_model import (
    BaseAtomicModel,
    DPPolarAtomicModel,
)
from deepmd.pt.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt.model.task.polarizability import (
    PolarFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.path import (
    DPPath,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class FooFitting(PolarFittingNet):
    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ):
        nf, nloc, _ = descriptor.shape
        ret = {}
        ret["polarizability"] = (
            torch.Tensor(
                [
                    [
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                        [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]],
                    ],
                    [
                        [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
                        [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
                        [[6.0, 6.0, 6.0], [4.0, 4.0, 4.0], [2.0, 2.0, 2.0]],
                    ],
                ]
            )
            .view([nf, nloc, *self.output_def()["polarizability"].shape])
            .to(env.GLOBAL_PT_FLOAT_PRECISION)
            .to(env.DEVICE)
        )

        return ret


class TestAtomicModelStat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.merged_output_stat = [
            {
                "coord": to_torch_tensor(np.zeros([2, 3, 3])),
                "atype": to_torch_tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
                ),
                "atype_ext": to_torch_tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32)
                ),
                "box": to_torch_tensor(np.zeros([2, 3, 3])),
                "natoms": to_torch_tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32)
                ),
                # bias of foo: 5, 6
                "atom_polarizability": to_torch_tensor(
                    np.array(
                        [
                            [
                                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
                                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
                                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
                            ],
                            [
                                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
                                [[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]],
                                [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0], [7.0, 7.0, 7.0]],
                            ],
                        ]
                    ).reshape(2, 3, 3, 3)
                ),
                "find_atom_polarizability": np.float32(1.0),
            },
            {
                "coord": to_torch_tensor(np.zeros([2, 3, 3])),
                "atype": to_torch_tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
                ),
                "atype_ext": to_torch_tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32)
                ),
                "box": to_torch_tensor(np.zeros([2, 3, 3])),
                "natoms": to_torch_tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32)
                ),
                # bias of foo: 5, 6 from atomic label.
                "polarizability": to_torch_tensor(
                    np.array(
                        [
                            [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
                            [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0], [7.0, 7.0, 7.0]],
                        ]
                    ).reshape(2, 3, 3)
                ),
                "find_polarizability": np.float32(1.0),
            },
        ]
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.stat_file_path = DPPath(h5file, "a")

    def test_output_stat(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft = FooFitting(self.nt, 1, 1).to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = DPPolarAtomicModel(
            ds,
            ft,
            type_map=type_map,
        ).to(env.DEVICE)
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        def cvt_ret(x):
            return {kk: to_numpy_array(vv) for kk, vv in x.items()}

        # 1. test run without bias
        # nf x na x odim
        ret0 = md0.forward_common_atomic(*args)
        ret0 = cvt_ret(ret0)
        expected_ret0 = {}
        expected_ret0["polarizability"] = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                    [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]],
                ],
                [
                    [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
                    [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
                    [[6.0, 6.0, 6.0], [4.0, 4.0, 4.0], [2.0, 2.0, 2.0]],
                ],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["polarizability"].shape])

        np.testing.assert_almost_equal(
            ret0["polarizability"], expected_ret0["polarizability"]
        )

        # 2. test bias is applied
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args)
        ret1 = cvt_ret(ret1)
        expected_std = np.zeros(
            (1, 2, 9), dtype=np.float64
        )  # 1 keys, 2 atypes, 9 max dims.
        expected_std[:, 1, :] = np.ones(9, dtype=np.float64) * 0.8164966  # updating std
        # nt x odim (dia)
        diagnoal_bias = np.array(
            [
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
            ]
        ).reshape(2, 3, 3)
        expected_ret1 = {}
        expected_ret1["polarizability"] = ret0["polarizability"] + diagnoal_bias[at]
        np.testing.assert_almost_equal(
            ret1["polarizability"], expected_ret1["polarizability"]
        )
        np.testing.assert_almost_equal(to_numpy_array(md0.out_std), expected_std)

        # 3. test bias load from file
        def raise_error() -> NoReturn:
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args)
        ret2 = cvt_ret(ret2)
        np.testing.assert_almost_equal(ret1["polarizability"], ret2["polarizability"])
        np.testing.assert_almost_equal(to_numpy_array(md0.out_std), expected_std)

        # 4. test change bias
        BaseAtomicModel.change_out_bias(
            md0, self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        args = [
            to_torch_tensor(ii)
            for ii in [
                self.coord_ext,
                to_numpy_array(self.merged_output_stat[0]["atype_ext"]),
                self.nlist,
            ]
        ]
        ret3 = md0.forward_common_atomic(*args)
        ret3 = cvt_ret(ret3)

        expected_ret3 = {}
        expected_std = np.array(
            [
                [
                    [
                        1.4142136,
                        1.4142136,
                        1.4142136,
                        1.2472191,
                        1.2472191,
                        1.2472191,
                        1.2472191,
                        1.2472191,
                        1.2472191,
                    ],
                    [
                        0.4714045,
                        0.4714045,
                        0.4714045,
                        0.8164966,
                        0.8164966,
                        0.8164966,
                        2.6246693,
                        2.6246693,
                        2.6246693,
                    ],
                ]
            ]
        )
        # new bias [[[3.0000, -, -, -, 2.6667, -, -, -, 2.3333],
        # [1.6667, -, -, -, 2.0000, -, -, -, 1.3333]]]
        # which yields [2.667, 1.667]
        expected_ret3["polarizability"] = np.array(
            [
                [
                    [[3.6667, 1.0, 1.0], [1.0, 3.6667, 1.0], [1.0, 1.0, 3.6667]],
                    [[3.6667, 1.0, 1.0], [2.0, 4.6667, 2.0], [3.0, 3.0, 5.6667]],
                    [[4.6667, 3.0, 3.0], [3.0, 4.6667, 3.0], [6.0, 6.0, 7.6667]],
                ],
                [
                    [[6.6667, 4.0, 4.0], [4.0, 6.6667, 4.0], [4.0, 4.0, 6.6667]],
                    [[5.6667, 4.0, 4.0], [5.0, 6.6667, 5.0], [6.0, 6.0, 7.6667]],
                    [[7.6667, 6.0, 6.0], [4.0, 5.6667, 4.0], [2.0, 2.0, 3.6667]],
                ],
            ]
        ).reshape(2, 3, 3, 3)
        np.testing.assert_almost_equal(
            ret3["polarizability"], expected_ret3["polarizability"], decimal=4
        )
        np.testing.assert_almost_equal(to_numpy_array(md0.out_std), expected_std)
