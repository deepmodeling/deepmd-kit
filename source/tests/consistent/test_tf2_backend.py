# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest
from pathlib import (
    Path,
)

import numpy as np
import pytest
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not tf.executing_eagerly():
    pytest.skip("TF2 backend requires eager execution", allow_module_level=True)

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.model.model import get_model as get_dp_model
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.tf2.common import (
    unwrap_value,
)
from deepmd.tf2.make_model import (
    model_call_from_call_lower,
)
from deepmd.tf2.model.model import get_model as get_tf2_model


def _model_data() -> dict:
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [20, 20],
            "rcut_smth": 0.50,
            "rcut": 6.00,
            "neuron": [3, 6],
            "resnet_dt": False,
            "axis_neuron": 2,
            "precision": "float64",
            "type_one_side": True,
            "seed": 1,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [5, 5],
            "resnet_dt": True,
            "precision": "float64",
            "atom_ener": [],
            "seed": 1,
        },
    }


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
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
    atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
    box = np.array(
        [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
        dtype=GLOBAL_NP_FLOAT_PRECISION,
    ).reshape(1, 9)
    idx_map = np.argsort(atype.ravel())
    return coords[:, idx_map], atype[:, idx_map], box


def _numpy_ret(ret: dict) -> dict[str, np.ndarray]:
    return {
        key: value.numpy() if isinstance(value, tf.Tensor) else to_numpy_array(value)
        for key, value in ret.items()
        if value is not None
    }


class TestTF2Backend(unittest.TestCase):
    def setUp(self) -> None:
        self.coords, self.atype, self.box = _inputs()
        self.data = _model_data()

    def test_eager_consistent_with_dpmodel(self) -> None:
        dp_model = get_dp_model(self.data)
        tf2_model = get_tf2_model(self.data)

        expected = dp_model(
            self.coords,
            self.atype,
            box=self.box,
            do_atomic_virial=False,
        )
        actual = tf2_model(
            self.coords,
            self.atype,
            box=self.box,
            do_atomic_virial=False,
        )

        expected = _numpy_ret(expected)
        actual = _numpy_ret(actual)
        for key in ("atom_energy", "energy"):
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=1e-12, atol=1e-12
            )

    def test_tf_function_call_from_lower(self) -> None:
        model = get_tf2_model(self.data)
        direct = _numpy_ret(
            model(
                self.coords,
                self.atype,
                box=self.box,
                do_atomic_virial=False,
            )
        )
        fparam = np.empty((1, model.get_dim_fparam()), dtype=np.float64)
        aparam = np.empty(
            (1, self.atype.shape[1], model.get_dim_aparam()), dtype=np.float64
        )

        def call_lower(coord, atype, nlist, mapping, fparam, aparam):
            return unwrap_value(
                model.call_common_lower(
                    coord,
                    atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    do_atomic_virial=False,
                )
            )

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, None, 3], tf.float64),
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, 9], tf.float64),
                tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
                tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
            ],
        )
        def call(coord, atype, box, fparam, aparam):
            return model_call_from_call_lower(
                call_lower=call_lower,
                rcut=model.get_rcut(),
                sel=model.get_sel(),
                mixed_types=model.mixed_types(),
                model_output_def=model.model_output_def(),
                coord=coord,
                atype=atype,
                box=box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=False,
            )

        graph = _numpy_ret(call(self.coords, self.atype, self.box, fparam, aparam))
        np.testing.assert_allclose(
            graph["energy"], direct["atom_energy"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            graph["energy_redu"], direct["energy"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            np.squeeze(graph["energy_derv_r"], axis=2),
            direct["force"],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.squeeze(graph["energy_derv_c_redu"], axis=1),
            direct["virial"],
            rtol=1e-12,
            atol=1e-12,
        )
