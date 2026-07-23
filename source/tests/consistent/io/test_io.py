# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import shutil
import unittest
from pathlib import (
    Path,
)
from typing import (
    ClassVar,
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
    # backends that cannot represent this model type (e.g. tf v1 has no
    # property model), skipped by the cross-backend round trips below.
    skip_backends: ClassVar[set[str]] = set()

    def _has_fparam_aparam(self) -> bool:
        """Whether the serialized fitting requires both parameter families."""
        fitting = self.data.get("model_def_script", {}).get("fitting_net", {})
        return (
            isinstance(fitting, dict)
            and fitting.get("numb_fparam", 0) > 0
            and fitting.get("numb_aparam", 0) > 0
        )

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
                if backend_name in self.skip_backends:
                    continue
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
        rets_atomic = []
        rets_nopbc = []
        rets_nopbc_atomic = []
        for backend_name, suffix_idx in (
            # unfortunately, jax2tf cannot work with tf v1 behaviors
            ("jax", 2) if DP_TEST_TF2_ONLY else ("tensorflow", 0),
            ("tf2", 0) if DP_TEST_TF2_ONLY else (None, None),
            ("pytorch", 0),
            ("paddle", 1) if self._has_fparam_aparam() else (None, None),
            ("dpmodel", 0),
            ("jax", 0) if DP_TEST_TF2_ONLY else (None, None),
        ):
            if backend_name is None or backend_name in self.skip_backends:
                continue
            backend = Backend.get_backend(backend_name)()
            if not backend.is_available():
                continue
            reference_data = copy.deepcopy(self.data)
            model_file = prefix + backend.suffixes[suffix_idx]
            self.save_data_to_model(model_file, reference_data)
            deep_eval = DeepEval(model_file)
            self.assertIsInstance(deep_eval.get_model_def_script(), dict)
            if not model_file.endswith((".savedmodel", ".savedmodeltf")):
                # SavedModel formats store an executable graph, not a lossless model dict.
                serialized_data = self.get_data_from_model(model_file)
                np.testing.assert_equal(deep_eval.serialize(), serialized_data["model"])
            if deep_eval.get_dim_fparam() > 0:
                fparam = np.ones((nframes, deep_eval.get_dim_fparam()))
            else:
                fparam = None
            if deep_eval.get_dim_aparam() > 0:
                aparam = np.ones((nframes, natoms, deep_eval.get_dim_aparam()))
            else:
                aparam = None
            if backend_name in {"pytorch", "jax", "tf2", "paddle"} and (
                deep_eval.get_dim_fparam() > 0 and deep_eval.get_dim_aparam() > 0
            ):
                self._assert_backend_parameter_shorthand(model_file, deep_eval)
            ret = deep_eval.eval(
                self.coords,
                self.box,
                self.atype,
                fparam=fparam,
                aparam=aparam,
            )
            # the non-atomic eval returns exactly the global outputs; the atomic
            # eval appends the per-atom outputs. Split by that count so this
            # generalizes across model types (energy: 3 global, property: 1).
            n_global = len(ret)
            rets.append(ret)
            ret = deep_eval.eval(
                self.coords,
                self.box,
                self.atype,
                fparam=fparam,
                aparam=aparam,
                atomic=True,
            )
            rets.append(ret[:n_global])
            rets_atomic.append(ret[n_global:])
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
            rets_nopbc.append(ret[:n_global])
            rets_nopbc_atomic.append(ret[n_global:])

        for rets_idx, rets_x in enumerate(
            (rets, rets_atomic, rets_nopbc, rets_nopbc_atomic)
        ):
            for idx, ret in enumerate(rets_x[1:]):
                for vv1, vv2 in zip(rets_x[0], ret, strict=True):
                    if np.isnan(vv2).all():
                        # expect all nan if not supported
                        continue
                    np.testing.assert_allclose(
                        vv1,
                        vv2,
                        rtol=1e-12,
                        atol=1e-12,
                        err_msg=f"backend {idx + 1} for rets_idx {rets_idx}",
                    )

    def _assert_backend_parameter_shorthand(
        self, model_file: str, deep_eval: DeepEval
    ) -> None:
        """Compare backend-direct shorthand with explicit frame-major inputs.

        Calling ``deep_eval.deep_eval`` deliberately bypasses the public
        ``_standard_input`` normalization. A one-frame auto-batch size also
        proves that shared per-atom parameters are expanded before the batcher
        can mistake their atom axis for a frame axis.
        """
        natoms = self.atype.shape[1]
        nframes = 2
        coords = np.repeat(self.coords, nframes, axis=0)
        boxes = np.repeat(self.box, nframes, axis=0)
        atom_types = self.atype.reshape(-1)
        fparam_shared = np.ones(deep_eval.get_dim_fparam())
        aparam_per_atom = np.ones((natoms, deep_eval.get_dim_aparam()))
        fparam_full = np.tile(fparam_shared, (nframes, 1))
        aparam_full = np.tile(aparam_per_atom, (nframes, 1, 1))
        backend = DeepEval(model_file, auto_batch_size=natoms).deep_eval

        expected = backend.eval(
            coords,
            boxes,
            atom_types,
            fparam=fparam_full,
            aparam=aparam_full,
        )
        shorthand_cases = (
            (fparam_shared.tolist(), aparam_per_atom),
            (
                fparam_shared,
                np.ones(deep_eval.get_dim_aparam()),
            ),
        )
        for fparam, aparam in shorthand_cases:
            actual = backend.eval(
                coords,
                boxes,
                atom_types,
                fparam=fparam,
                aparam=aparam,
            )
            self.assertEqual(actual.keys(), expected.keys())
            for name in actual:
                np.testing.assert_allclose(
                    actual[name],
                    expected[name],
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                    err_msg=f"backend-direct shorthand output {name}",
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


@unittest.skipIf(
    not DP_TEST_TF2_ONLY,
    "pair_exclude_types is not supported by the TF v1 backend; it is validated "
    "in the TF2-only job, where test_deep_eval also exercises the jax2tf "
    "'.savedmodel' export path (see the backend table in test_deep_eval).",
)
class TestDeepPotPairExclude(unittest.TestCase, IOTest):
    """Model-level ``pair_exclude_types`` is a nlist-BUILD transform (decision
    #18/A4). Every backend folds it in where the neighbor list is built (the
    jax2tf ``.savedmodel`` export reuses the dpmodel
    ``apply_pair_exclusion_nlist`` via the ``ndtensorflow`` namespace), so the
    exported models must still eval-agree across backends.
    """

    def setUp(self) -> None:
        model_def_script = {
            "type_map": ["O", "H"],
            "pair_exclude_types": [[0, 1]],
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


class TestDeepProperty(unittest.TestCase, IOTest):
    # tf v1 has no property model, so the property dict cannot round trip
    # through the tensorflow backend.
    skip_backends: ClassVar[set[str]] = {"tensorflow"}

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
                "type": "property",
                "neuron": [
                    5,
                    5,
                ],
                "property_name": "foo",
                "task_dim": 3,
                "resnet_dt": True,
                "precision": "float64",
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
