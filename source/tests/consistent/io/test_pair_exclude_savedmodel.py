# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate that the jax2tf ``.savedmodel`` export reuses the canonical dpmodel
model-level pair-exclusion transform.

Model-level ``pair_exclude_types`` is a nlist-BUILD transform (decision
#18/A4). The jax2tf SavedModel wrapper folds it into the neighbor list by
calling :func:`deepmd.dpmodel.utils.nlist.apply_pair_exclusion_nlist` through
the vendored ``ndtensorflow`` array-API namespace, rather than a hand-written
TensorFlow twin. These tests prove that reuse:

* the exported SavedModel traces (the reused array-API code is convertible
  under ``tf.saved_model.save``), and
* it matches the dpmodel reference to fp64 tolerance (same math, applied at the
  same nlist-BUILD seam),
* while a no-exclusion baseline exercises the identity branch and proves the
  exclusion is genuinely active (excluded energy differs from baseline).
"""

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

_JAX_BACKEND = Backend.get_backend("jax")()


def _model_def_script(pair_exclude_types: list[list[int]]) -> dict:
    md = {
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
    if pair_exclude_types:
        md["pair_exclude_types"] = pair_exclude_types
    return md


@unittest.skipIf(
    not _JAX_BACKEND.is_available(),
    "jax2tf SavedModel export requires the jax backend (jax + tensorflow).",
)
class TestPairExcludeSavedModel(unittest.TestCase):
    """jax2tf ``.savedmodel`` export of model-level pair exclusion."""

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
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self._artifacts: list[Path] = []

    def tearDown(self) -> None:
        for path in self._artifacts:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def _eval(self, backend_name: str, md: dict, atomic: bool = False) -> tuple:
        """Serialize *md*, write it through *backend_name*, DeepEval it."""
        backend = Backend.get_backend(backend_name)()
        suffix = ".savedmodel" if backend_name == "jax" else backend.suffixes[0]
        prefix = f"test_pairexcl_savedmodel_{backend_name}_{len(self._artifacts)}"
        model_file = prefix + suffix
        model = get_model(copy.deepcopy(md))
        data = {
            "model": model.serialize(),
            "backend": "test",
            "model_def_script": md,
        }
        backend.deserialize_hook(model_file, data)
        self._artifacts.append(Path(model_file))
        deep_eval = DeepEval(model_file)
        return deep_eval.eval(self.coords, self.box, self.atype, atomic=atomic)

    def test_savedmodel_matches_reference_excluded(self) -> None:
        """SavedModel export matches the references when [0, 1] is excluded.

        Energy is compared against the dpmodel reference: both apply the
        exclusion at the nlist-BUILD seam, so the neighbor list fed to the
        lower model is identical and the energy must match to fp64. Force and
        virial are compared against the pytorch backend instead, because the
        numpy dpmodel DeepEval does not compute usable forces (returns NaN)
        regardless of exclusion; pytorch applies the same exclusion
        (idempotently) and produces finite gradients.
        """
        md = _model_def_script([[0, 1]])
        e_ref = self._eval("dpmodel", md)[0]
        e_sm, f_sm, v_sm = self._eval("jax", md)[:3]
        np.testing.assert_allclose(e_sm, e_ref, atol=1e-10)
        if Backend.get_backend("pytorch")().is_available():
            _, f_pt, v_pt = self._eval("pytorch", md)[:3]
            np.testing.assert_allclose(f_sm, f_pt, atol=1e-10)
            np.testing.assert_allclose(v_sm, v_pt, atol=1e-10)

    def test_savedmodel_matches_dpmodel_no_exclusion(self) -> None:
        """Identity branch: no exclusion still round-trips and matches dpmodel."""
        md = _model_def_script([])
        e_ref = self._eval("dpmodel", md)[0]
        e_sm = self._eval("jax", md)[0]
        np.testing.assert_allclose(e_sm, e_ref, atol=1e-10)

    def test_exclusion_is_active(self) -> None:
        """Excluding [0, 1] changes the SavedModel energy vs the baseline."""
        e_excl = self._eval("jax", _model_def_script([[0, 1]]))[0]
        e_none = self._eval("jax", _model_def_script([]))[0]
        # O-H pairs dominate this water-like cluster; excluding them moves the
        # energy well above the fp64 tolerance.
        self.assertGreater(float(np.abs(e_excl - e_none).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
