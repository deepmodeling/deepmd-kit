# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
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

    def test_data_equal(self):
        prefix = "test_consistent_io_" + self.__class__.__name__.lower()
        for backend_name in ("tensorflow", "pytorch", "dpmodel"):
            with self.subTest(backend_name=backend_name):
                backend = Backend.get_backend(backend_name)()
                if not backend.is_available:
                    continue
                reference_data = copy.deepcopy(self.data)
                self.save_data_to_model(prefix + backend.suffixes[0], reference_data)
                data = self.get_data_from_model(prefix + backend.suffixes[0])
                data = copy.deepcopy(data)
                reference_data = copy.deepcopy(self.data)
                # some keys are not expected to be not the same
                for kk in [
                    "backend",
                    "tf_version",
                    "pt_version",
                    "@variables",
                    # dpmodel only
                    "software",
                    "version",
                    "time",
                ]:
                    data.pop(kk, None)
                    reference_data.pop(kk, None)
                np.testing.assert_equal(data, reference_data)


class TestDeepPot(unittest.TestCase, IOTest):
    def setUp(self):
        param = {
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
        model = get_model(copy.deepcopy(param))
        self.data = {
            "model": model.serialize(),
            "backend": "test",
            "model_param": param,
        }
