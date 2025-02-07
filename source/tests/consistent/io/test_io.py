# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import shutil
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.backend.backend import (
    Backend,
)
from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)

from ...utils import (
    CI,
    DP_TEST_TF2_ONLY,
    TEST_DEVICE,
)

infer_path = Path(__file__).parent.parent.parent / "infer"


class IOTest:
    data: dict

    def get_data_from_model(self, model_file: str) -> dict:
        """Get data from a model file.

        Parameters
        ----------
        model_file : str
            The model file.

        Returns
        -------
        dict
            The data from the model file.
        """
        inp_backend: Backend = Backend.detect_backend_by_model(model_file)()
        inp_hook = inp_backend.serialize_hook
        return inp_hook(model_file)

    def save_data_to_model(self, model_file: str, data: dict) -> None:
        """Save data to a model file.

        Parameters
        ----------
        model_file : str
            The model file.
        data : dict
            The data to save.
        """
        out_backend: Backend = Backend.detect_backend_by_model(model_file)()
        out_hook = out_backend.deserialize_hook
        out_hook(model_file, data)

    def tearDown(self) -> None:
        prefix = "test_consistent_io_" + self.__class__.__name__.lower()
        for ii in Path(".").glob(prefix + ".*"):
            if Path(ii).is_file():
                Path(ii).unlink()
            elif Path(ii).is_dir():
                shutil.rmtree(ii)

    @unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
    def test_data_equal(self) -> None:
        prefix = "test_consistent_io_" + self.__class__.__name__.lower()
        for backend_name, suffix_idx in (
            ("tensorflow", 0) if not DP_TEST_TF2_ONLY else ("jax", 0),
            ("pytorch", 0),
            ("dpmodel", 0),
        ):
            with self.subTest(backend_name=backend_name):
                backend = Backend.get_backend(backend_name)()
                if not backend.is_available():
                    continue
                reference_data = copy.deepcopy(self.data)
                self.save_data_to_model(
                    prefix + backend.suffixes[suffix_idx], reference_data
                )
                data = self.get_data_from_model(prefix + backend.suffixes[suffix_idx])
                data = data.copy()
                reference_data = self.data.copy()
                # some keys are not expected to be not the same
                for kk in [
                    "backend",
                    "tf_version",
                    "pt_version",
                    "jax_version",
                    "@variables",
                    # dpmodel only
                    "software",
                    "version",
                    "time",
                ]:
                    data.pop(kk, None)
                    reference_data.pop(kk, None)
                np.testing.assert_equal(data, reference_data)

    def test_deep_eval(self) -> None:
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
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        natoms = self.atype.shape[1]
        nframes = self.atype.shape[0]
        prefix = "test_consistent_io_" + self.__class__.__name__.lower()
        rets = []
        rets_nopbc = []
        for backend_name, suffix_idx in (
            # unfortunately, jax2tf cannot work with tf v1 behaviors
            ("jax", 2) if DP_TEST_TF2_ONLY else ("tensorflow", 0),
            ("pytorch", 0),
            ("dpmodel", 0),
            ("jax", 0) if DP_TEST_TF2_ONLY else (None, None),
        ):
            if backend_name is None:
                continue
            backend = Backend.get_backend(backend_name)()
            if not backend.is_available():
                continue
            reference_data = copy.deepcopy(self.data)
            self.save_data_to_model(
                prefix + backend.suffixes[suffix_idx], reference_data
            )
            deep_eval = DeepEval(prefix + backend.suffixes[suffix_idx])
            if deep_eval.get_dim_fparam() > 0:
                fparam = np.ones((nframes, deep_eval.get_dim_fparam()))
            else:
                fparam = None
            if deep_eval.get_dim_aparam() > 0:
                aparam = np.ones((nframes, natoms, deep_eval.get_dim_aparam()))
            else:
                aparam = None
            ret = deep_eval.eval(
                self.coords,
                self.box,
                self.atype,
                fparam=fparam,
                aparam=aparam,
            )
            rets.append(ret)
            ret = deep_eval.eval(
                self.coords,
                self.box,
                self.atype,
                fparam=fparam,
                aparam=aparam,
                atomic=True,
            )
            rets.append(ret)
            ret = deep_eval.eval(
                self.coords,
                None,
                self.atype,
                fparam=fparam,
                aparam=aparam,
            )
            rets_nopbc.append(ret)
            ret = deep_eval.eval(
                self.coords,
                None,
                self.atype,
                fparam=fparam,
                aparam=aparam,
                atomic=True,
            )
            rets_nopbc.append(ret)
        for ret in rets[1:]:
            for vv1, vv2 in zip(rets[0], ret):
                if np.isnan(vv2).all():
                    # expect all nan if not supported
                    continue
                np.testing.assert_allclose(vv1, vv2, rtol=1e-12, atol=1e-12)

        for idx, ret in enumerate(rets_nopbc[1:]):
            for vv1, vv2 in zip(rets_nopbc[0], ret):
                if np.isnan(vv2).all():
                    # expect all nan if not supported
                    continue
                np.testing.assert_allclose(
                    vv1, vv2, rtol=1e-12, atol=1e-12, err_msg=f"backend {idx + 1}"
                )


class TestDeepPot(unittest.TestCase, IOTest):
    def setUp(self) -> None:
        model_def_script = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
                "rcut_smth": 0.50,
                "rcut": 6.00,
                "neuron": [
                    3,
                    6,
                ],
                "resnet_dt": False,
                "axis_neuron": 2,
                "precision": "float64",
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "type": "ener",
                "neuron": [
                    5,
                    5,
                ],
                "resnet_dt": True,
                "precision": "float64",
                "atom_ener": [],
                "seed": 1,
            },
        }
        model = get_model(copy.deepcopy(model_def_script))
        self.data = {
            "model": model.serialize(),
            "backend": "test",
            "model_def_script": model_def_script,
        }

    def tearDown(self) -> None:
        IOTest.tearDown(self)


class TestDeepPotFparamAparam(unittest.TestCase, IOTest):
    def setUp(self) -> None:
        model_def_script = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
                "rcut_smth": 0.50,
                "rcut": 6.00,
                "neuron": [
                    3,
                    6,
                ],
                "resnet_dt": False,
                "axis_neuron": 2,
                "precision": "float64",
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "type": "ener",
                "neuron": [
                    5,
                    5,
                ],
                "resnet_dt": True,
                "precision": "float64",
                "atom_ener": [],
                "seed": 1,
                "numb_fparam": 2,
                "numb_aparam": 2,
            },
        }
        model = get_model(copy.deepcopy(model_def_script))
        self.data = {
            "model": model.serialize(),
            "backend": "test",
            "model_def_script": model_def_script,
        }

    def tearDown(self) -> None:
        IOTest.tearDown(self)
