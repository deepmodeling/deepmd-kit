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

    def tearDown(self):
        prefix = "test_consistent_io_" + self.__class__.__name__.lower()
        for ii in Path(".").glob(prefix + ".*"):
            if Path(ii).is_file():
                Path(ii).unlink()
            elif Path(ii).is_dir():
                shutil.rmtree(ii)

    @unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
    def test_data_equal(self):
        prefix = "test_consistent_io_" + self.__class__.__name__.lower()
        for backend_name, suffix_idx in (
            ("tensorflow", 0),
            ("pytorch", 0),
            ("dpmodel", 0),
            ("jax", 0),
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
                data = copy.deepcopy(data)
                reference_data = copy.deepcopy(self.data)
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

    def test_deep_eval(self):
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
        prefix = "test_consistent_io_" + self.__class__.__name__.lower()
        rets = []
        for backend_name in ("tensorflow", "pytorch", "dpmodel", "jax"):
            backend = Backend.get_backend(backend_name)()
            if not backend.is_available():
                continue
            reference_data = copy.deepcopy(self.data)
            self.save_data_to_model(prefix + backend.suffixes[0], reference_data)
            deep_eval = DeepEval(prefix + backend.suffixes[0])
            ret = deep_eval.eval(
                self.coords,
                self.box,
                self.atype,
            )
            rets.append(ret)
        for ret in rets[1:]:
            for vv1, vv2 in zip(rets[0], ret):
                if np.isnan(vv2).all():
                    # expect all nan if not supported
                    continue
                np.testing.assert_allclose(vv1, vv2, rtol=1e-12, atol=1e-12)


class TestDeepPot(unittest.TestCase, IOTest):
    def setUp(self):
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

    def tearDown(self):
        IOTest.tearDown(self)
