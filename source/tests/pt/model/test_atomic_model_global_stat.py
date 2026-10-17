# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)
from typing import (
    NoReturn,
)

import h5py
import numpy as np
import torch

from deepmd.dpmodel.atomic_model import DPAtomicModel as DPDPAtomicModel
from deepmd.dpmodel.model.ener_model import EnergyModel as DPEnergyModel
from deepmd.dpmodel.model.model import get_model as get_dp_model
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
    save_dp_model,
)
from deepmd.pt.model.atomic_model import (
    BaseAtomicModel,
    DPAtomicModel,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA1,
    DescrptSeA,
)
from deepmd.pt.model.model import (
    EnergyModel,
    get_model,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
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


class FooFitting(torch.nn.Module, BaseFitting):
    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "foo",
                    [1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
                OutputVariableDef(
                    "pix",
                    [1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
                OutputVariableDef(
                    "bar",
                    [1, 2],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ]
        )

    def serialize(self) -> dict:
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        raise NotImplementedError

    def get_type_map(self) -> list[str]:
        raise NotImplementedError

    def get_dim_fparam(self) -> int:
        return 1

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ):
        nloc = descriptor.shape[1]
        # The frame parameter selects predictions independently of batching.
        assert fparam is not None
        frame_index = fparam[:, 0].to(torch.long)
        ret = {}
        ret["foo"] = (
            torch.Tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            )
            .view([-1, nloc, *self.output_def()["foo"].shape])
            .to(env.GLOBAL_PT_FLOAT_PRECISION)
            .to(env.DEVICE)
            .index_select(0, frame_index)
        )
        ret["pix"] = (
            torch.Tensor(
                [
                    [3.0, 2.0, 1.0],
                    [6.0, 5.0, 4.0],
                ]
            )
            .view([-1, nloc, *self.output_def()["pix"].shape])
            .to(env.GLOBAL_PT_FLOAT_PRECISION)
            .to(env.DEVICE)
            .index_select(0, frame_index)
        )
        ret["bar"] = (
            torch.Tensor(
                [
                    [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                    [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
                ]
            )
            .view([-1, nloc, *self.output_def()["bar"].shape])
            .to(env.GLOBAL_PT_FLOAT_PRECISION)
            .to(env.DEVICE)
            .index_select(0, frame_index)
        )
        return ret


class TestAtomicModelStat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.fparam = to_torch_tensor(
            np.arange(self.nf, dtype=np.float64).reshape(-1, 1)
        )
        nf, nloc, nnei = self.nlist.shape
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
                # bias of foo: 1, 3
                "foo": to_torch_tensor(np.array([5.0, 7.0]).reshape(2, 1)),
                # no bias of pix
                # bias of bar: [1, 5], [3, 2]
                "bar": to_torch_tensor(
                    np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2)
                ),
                "find_foo": np.float32(1.0),
                "find_bar": np.float32(1.0),
            }
        ]
        for system in self.merged_output_stat:
            system["fparam"] = self.fparam
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
        ft = FooFitting().to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
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
        ret0 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret0 = cvt_ret(ret0)

        expected_ret0 = {}
        expected_ret0["foo"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["foo"].shape)  # noqa: RUF005
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["pix"].shape)  # noqa: RUF005
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["bar"].shape)  # noqa: RUF005
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])

        # 2. test bias is applied
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret1 = cvt_ret(ret1)
        expected_std = np.array(
            [[[0, 1], [0, 1]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]]
        )  # 3 keys, 2 atypes, 2 max dims.
        # nt x odim
        foo_bias = np.array([1.0, 3.0]).reshape(2, 1)
        bar_bias = np.array([1.0, 5.0, 3.0, 2.0]).reshape(2, 1, 2)
        expected_ret1 = {}
        expected_ret1["foo"] = ret0["foo"] + foo_bias[at]
        expected_ret1["pix"] = ret0["pix"]
        expected_ret1["bar"] = ret0["bar"] + bar_bias[at]
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], expected_ret1[kk])
        np.testing.assert_almost_equal(to_numpy_array(md0.out_std), expected_std)

        # 3. test bias load from file
        def raise_error() -> NoReturn:
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret2 = cvt_ret(ret2)
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])
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
        ret3 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret3 = cvt_ret(ret3)
        ## model output on foo: [[2, 3, 6], [5, 8, 9]] given bias [1, 3]
        ## foo sumed: [11, 22] compared with [5, 7], fit target is [-6, -15]
        ## fit bias is [1, -8]
        ## old bias + fit bias [2, -5]
        ## new model output is [[3, 4, -2], [6, 0, 1]], which sumed to [5, 7]
        expected_ret3 = {}
        expected_ret3["foo"] = np.array([[3, 4, -2], [6, 0, 1]]).reshape(2, 3, 1)
        expected_ret3["pix"] = ret0["pix"]
        for kk in ["foo", "pix"]:
            np.testing.assert_almost_equal(ret3[kk], expected_ret3[kk])
        # bar is too complicated to be manually computed.
        np.testing.assert_almost_equal(to_numpy_array(md0.out_std), expected_std)

    def test_preset_bias(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft = FooFitting().to(env.DEVICE)
        type_map = ["foo", "bar"]
        # both types occur in the data, so an assigned output assigns both
        preset_out_bias = {
            "foo": [3, 2],
            "bar": np.array([7.0, 5.0, 13.0, 11.0]).reshape(2, 1, 2),
        }
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
            preset_out_bias=preset_out_bias,
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
        ret0 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret0 = cvt_ret(ret0)
        expected_ret0 = {}
        expected_ret0["foo"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["foo"].shape)  # noqa: RUF005
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["pix"].shape)  # noqa: RUF005
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["bar"].shape)  # noqa: RUF005
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])
        # 2. the preset fixes foo and bar; pix has no label and keeps a zero bias
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret1 = cvt_ret(ret1)
        foo_bias = np.array(preset_out_bias["foo"], dtype=np.float64).reshape(2, 1)
        bar_bias = preset_out_bias["bar"]
        expected_ret1 = {}
        expected_ret1["foo"] = ret0["foo"] + foo_bias[at]
        expected_ret1["pix"] = ret0["pix"]
        expected_ret1["bar"] = ret0["bar"] + bar_bias[at]
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], expected_ret1[kk])
        # 3. change-by-statistic keeps every assigned type at its preset
        BaseAtomicModel.change_out_bias(
            md0, self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        ret3 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret3 = cvt_ret(ret3)
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret3[kk], ret1[kk])
        out_bias, _ = md0._fetch_out_stat(["foo", "bar"])
        np.testing.assert_almost_equal(to_numpy_array(out_bias["foo"]), foo_bias)
        np.testing.assert_almost_equal(to_numpy_array(out_bias["bar"]), bar_bias)
        # 4. a preset that leaves an observed type unassigned is rejected
        md1 = DPAtomicModel(
            ds,
            FooFitting().to(env.DEVICE),
            type_map=type_map,
            preset_out_bias={"foo": [None, 2]},
        ).to(env.DEVICE)
        with self.assertRaisesRegex(ValueError, "foo"):
            md1.compute_or_load_out_stat(self.merged_output_stat)

    def test_preset_bias_all_none(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft = FooFitting().to(env.DEVICE)
        type_map = ["foo", "bar"]
        preset_out_bias = {
            "foo": [None, None],
        }
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
            preset_out_bias=preset_out_bias,
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
        ret0 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret0 = cvt_ret(ret0)
        expected_ret0 = {}
        expected_ret0["foo"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["foo"].shape)  # noqa: RUF005
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["pix"].shape)  # noqa: RUF005
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["bar"].shape)  # noqa: RUF005
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])

        # 2. test bias is applied
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args, fparam=self.fparam)
        ret1 = cvt_ret(ret1)
        # nt x odim
        foo_bias = np.array([1.0, 3.0]).reshape(2, 1)
        bar_bias = np.array([1.0, 5.0, 3.0, 2.0]).reshape(2, 1, 2)
        expected_ret1 = {}
        expected_ret1["foo"] = ret0["foo"] + foo_bias[at]
        expected_ret1["pix"] = ret0["pix"]
        expected_ret1["bar"] = ret0["bar"] + bar_bias[at]
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], expected_ret1[kk])

    def test_serialize(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = InvarFitting(
            "foo",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["A", "B"]
        md0 = DPAtomicModel(
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

        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret0 = md0.forward_common_atomic(*args)
        ret0 = cvt_ret(ret0)
        md1 = DPAtomicModel.deserialize(md0.serialize())
        ret1 = md1.forward_common_atomic(*args)
        ret1 = cvt_ret(ret1)

        for kk in ["foo"]:
            np.testing.assert_almost_equal(ret0[kk], ret1[kk])

        md2 = DPDPAtomicModel.deserialize(md0.serialize())
        args = [self.coord_ext, self.atype_ext, self.nlist]
        ret2 = md2.forward_common_atomic(*args)
        for kk in ["foo"]:
            np.testing.assert_almost_equal(ret0[kk], ret2[kk])


class TestModelPresetBias(unittest.TestCase):
    """Preset bias through the model-level bias change used by fine-tuning."""

    def setUp(self) -> None:
        # a mixed-types descriptor supports change_type_map
        self.model = get_model(
            {
                "type_map": ["O", "H", "B"],
                "descriptor": {
                    "type": "dpa1",
                    "sel": 20,
                    "rcut_smth": 0.5,
                    "rcut": 4.0,
                    "neuron": [4, 8],
                    "axis_neuron": 2,
                    "attn_layer": 0,
                    "seed": 1,
                },
                "fitting_net": {"neuron": [8, 8], "seed": 1},
                "preset_out_bias": {"energy": {"O": -10.0, "H": -13.6}},
            }
        ).to(env.DEVICE)
        rng = np.random.default_rng(20260913)
        # O and H occur in the data and are assigned; B is neither
        self.sampled = [
            {
                "coord": to_torch_tensor(rng.uniform(-1.5, 1.5, size=(2, 3, 3))),
                "atype": to_torch_tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
                ),
                "natoms": to_torch_tensor(
                    np.array([[3, 3, 2, 1, 0], [3, 3, 1, 2, 0]], dtype=np.int32)
                ),
                "energy": to_torch_tensor(np.array([-30.0, -20.0]).reshape(2, 1)),
                "find_energy": np.float32(1.0),
            }
        ]

    def out_bias(self) -> np.ndarray:
        return to_numpy_array(self.model.get_out_bias()).reshape(-1)

    def test_preset_pinned_in_both_modes(self) -> None:
        preset = np.array([-10.0, -13.6, 0.0])
        self.model.change_out_bias(self.sampled, bias_adjust_mode="set-by-statistic")
        np.testing.assert_allclose(self.out_bias(), preset)
        # the output std of a fully preset output keeps its stored value
        out_std = to_numpy_array(self.model.atomic_model.out_std).reshape(-1)
        np.testing.assert_allclose(out_std, 1.0)
        self.model.change_out_bias(self.sampled, bias_adjust_mode="change-by-statistic")
        np.testing.assert_allclose(self.out_bias(), preset)

    def test_spin_model(self) -> None:
        # the virtual spin types are excluded from the output and need no preset
        model = get_model(
            {
                "type_map": ["O", "H", "B"],
                "descriptor": {
                    "type": "dpa1",
                    "sel": 20,
                    "rcut_smth": 0.5,
                    "rcut": 4.0,
                    "neuron": [4, 8],
                    "axis_neuron": 2,
                    "attn_layer": 0,
                    "seed": 1,
                },
                "fitting_net": {"neuron": [8, 8], "seed": 1},
                "spin": {"use_spin": [True, False, False], "virtual_scale": [0.3]},
                "preset_out_bias": {"energy": {"O": -10.0, "H": -13.6}},
            }
        ).to(env.DEVICE)
        rng = np.random.default_rng(1)
        sampled = [
            {**self.sampled[0], "spin": to_torch_tensor(rng.normal(size=(2, 3, 3)))}
        ]
        for mode in ("set-by-statistic", "change-by-statistic"):
            model.change_out_bias(sampled, bias_adjust_mode=mode)
            out_bias = to_numpy_array(model.get_out_bias()).reshape(-1)
            np.testing.assert_allclose(out_bias[:3], [-10.0, -13.6, 0.0])

    def test_preset_needs_no_data(self) -> None:
        def raise_error() -> NoReturn:
            raise RuntimeError

        self.model.atomic_model.change_out_bias(
            raise_error, bias_adjust_mode="set-by-statistic", observed_type=["O", "H"]
        )
        np.testing.assert_allclose(self.out_bias(), [-10.0, -13.6, 0.0])

    def test_missing_element_rejected(self) -> None:
        model = get_model(
            {
                "type_map": ["O", "H", "B"],
                "descriptor": {
                    "type": "dpa1",
                    "sel": 20,
                    "rcut_smth": 0.5,
                    "rcut": 4.0,
                    "neuron": [4, 8],
                    "axis_neuron": 2,
                    "attn_layer": 0,
                    "seed": 1,
                },
                "fitting_net": {"neuron": [8, 8], "seed": 1},
                "preset_out_bias": {"energy": {"H": -13.6, "B": 3.0}},
            }
        ).to(env.DEVICE)
        with self.assertRaisesRegex(ValueError, "O"):
            model.change_out_bias(self.sampled, bias_adjust_mode="set-by-statistic")

    def test_dp_file_round_trip(self) -> None:
        self.model.change_out_bias(self.sampled, bias_adjust_mode="set-by-statistic")
        with tempfile.TemporaryDirectory() as tmp:
            filename = str(Path(tmp) / "model.dp")
            save_dp_model(filename, {"model": self.model.serialize()})
            data = load_dp_model(filename)["model"]
        # the file is read back by the pt backend and by the dpmodel backend
        for loaded in (EnergyModel.deserialize(data), DPEnergyModel.deserialize(data)):
            np.testing.assert_allclose(
                to_numpy_array(loaded.get_out_bias()),
                to_numpy_array(self.model.get_out_bias()),
            )
            self.assertEqual(
                loaded.atomic_model.preset_out_bias,
                {"energy": [[-10.0], [-13.6], None]},
            )

    def test_nonfinite_assigned_bias_from_checkpoint(self) -> None:
        # the preset replaces a non-finite stored bias of an assigned type
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.model.atomic_model.out_bias[0, :2, 0] = value
                loaded = EnergyModel.deserialize(self.model.serialize())
                loaded.change_out_bias(
                    self.sampled, bias_adjust_mode="change-by-statistic"
                )
                bias = to_numpy_array(loaded.get_out_bias()).reshape(-1)
                np.testing.assert_allclose(bias, [-10.0, -13.6, 0.0])
                loaded.change_out_bias(
                    self.sampled, bias_adjust_mode="change-by-statistic"
                )
                np.testing.assert_allclose(
                    to_numpy_array(loaded.get_out_bias()).reshape(-1), bias, atol=1e-10
                )

    def test_change_type_map_remaps_preset(self) -> None:
        self.model.change_out_bias(self.sampled, bias_adjust_mode="set-by-statistic")
        # H is dropped, O keeps its preset, B and C are unassigned
        self.model.change_type_map(["B", "O", "C"])
        self.assertEqual(
            self.model.atomic_model.preset_out_bias, {"energy": [None, [-10.0], None]}
        )
        sampled = [
            {
                **self.sampled[0],
                "atype": to_torch_tensor(
                    np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int32)
                ),
                "natoms": to_torch_tensor(
                    np.array([[3, 3, 0, 2, 1], [3, 3, 0, 1, 2]], dtype=np.int32)
                ),
            }
        ]
        # C occurs in the new data without a preset
        with self.assertRaisesRegex(ValueError, "C"):
            self.model.change_out_bias(sampled, bias_adjust_mode="change-by-statistic")

    def test_dipole_preset_rejected(self) -> None:
        params = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [4, 4],
                "neuron": [4, 8],
                "axis_neuron": 2,
                "rcut": 3.0,
                "rcut_smth": 2.5,
            },
            "fitting_net": {"type": "dipole", "neuron": [8]},
            "preset_out_bias": {"dipole": {"H": [0.0, 1.0, 2.0]}},
        }
        for builder in (get_model, get_dp_model):
            with self.subTest(builder=builder.__module__):
                with self.assertRaisesRegex(ValueError, "do not apply an output bias"):
                    builder(params)

    def test_unknown_output_rejected(self) -> None:
        params = {
            "type_map": ["O", "H", "B"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [8, 8, 8],
                "rcut_smth": 0.5,
                "rcut": 4.0,
            },
            "fitting_net": {"neuron": [8]},
            "preset_out_bias": {"enrgy": {"H": -13.6}},
        }
        with self.assertRaisesRegex(ValueError, "enrgy"):
            get_model(params)

    def test_no_distinguish_rejected(self) -> None:
        params = {
            "type_map": ["O", "H", "B"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [8, 8, 8],
                "rcut_smth": 0.5,
                "rcut": 4.0,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_prop",
                "task_dim": 1,
                "neuron": [8],
                "distinguish_types": False,
            },
            "preset_out_bias": {"band_prop": {"H": 1.0}},
        }
        model = get_model(params).to(env.DEVICE)
        with self.assertRaisesRegex(ValueError, "distinguish"):
            model.change_out_bias(self.sampled, bias_adjust_mode="set-by-statistic")
