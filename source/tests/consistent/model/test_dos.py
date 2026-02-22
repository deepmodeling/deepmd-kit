# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.model.dos_model import DOSModel as DOSModelDP
from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    INSTALLED_TF,
    CommonTest,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.dos_model import DOSModel as DOSModelPT
else:
    DOSModelPT = None
if INSTALLED_TF:
    from deepmd.tf.model.dos import DOSModel as DOSModelTF
else:
    DOSModelTF = None
if INSTALLED_JAX:
    from deepmd.jax.model.dos_model import DOSModel as DOSModelJAX
    from deepmd.jax.model.model import get_model as get_model_jax
else:
    DOSModelJAX = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.model import DOSModel as DOSModelPTExpt
else:
    DOSModelPTExpt = None
from deepmd.utils.argcheck import (
    model_args,
)


class TestDOS(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        return {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
                "rcut_smth": 1.8,
                "rcut": 6.0,
                "neuron": [2, 4, 8],
                "resnet_dt": False,
                "axis_neuron": 8,
                "precision": "float64",
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "type": "dos",
                "numb_dos": 2,
                "neuron": [4, 4, 4],
                "resnet_dt": True,
                "numb_fparam": 0,
                "precision": "float64",
                "seed": 1,
            },
        }

    tf_class = DOSModelTF
    dp_class = DOSModelDP
    pt_class = DOSModelPT
    pt_expt_class = DOSModelPTExpt
    jax_class = DOSModelJAX
    args = model_args()

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_tf:
            return self.RefBackend.TF
        if not self.skip_pt_expt and self.pt_expt_class is not None:
            return self.RefBackend.PT_EXPT
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    @property
    def skip_tf(self):
        return not INSTALLED_TF

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is DOSModelDP:
            return get_model_dp(data)
        elif cls is DOSModelPT:
            model = get_model_pt(data)
            model.atomic_model.out_bias.uniform_()
            return model
        elif cls is DOSModelPTExpt:
            dp_model = get_model_dp(data)
            return DOSModelPTExpt.deserialize(dp_model.serialize())
        elif cls is DOSModelJAX:
            return get_model_jax(data)
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
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
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

        # TF requires the atype to be sort
        idx_map = np.argsort(self.atype.ravel())
        self.atype = self.atype[:, idx_map]
        self.coords = self.coords[:, idx_map]

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        return self.build_tf_model(
            obj, self.natoms, self.coords, self.atype, self.box, suffix, ret_key="dos"
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_model(
            dp_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_model(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        return self.eval_pt_expt_model(
            pt_expt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_model(
            jax_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        if backend is self.RefBackend.TF:
            return (
                ret[0].ravel(),
                ret[1].ravel(),
            )
        elif backend in {
            self.RefBackend.DP,
            self.RefBackend.PT,
            self.RefBackend.PT_EXPT,
            self.RefBackend.JAX,
        }:
            return (
                ret["dos"].ravel(),
                ret["atom_dos"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")


@unittest.skipUnless(INSTALLED_PT and INSTALLED_PT_EXPT, "PyTorch is not installed")
class TestDOSModelAPIs(unittest.TestCase):
    """Test translated_output_def consistency across dp, pt, and pt_expt backends."""

    def setUp(self) -> None:
        data = model_args().normalize_value(
            {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": [20, 20],
                    "rcut_smth": 1.8,
                    "rcut": 6.0,
                    "neuron": [2, 4, 8],
                    "resnet_dt": False,
                    "axis_neuron": 8,
                    "precision": "float64",
                    "type_one_side": True,
                    "seed": 1,
                },
                "fitting_net": {
                    "type": "dos",
                    "numb_dos": 2,
                    "neuron": [4, 4, 4],
                    "resnet_dt": True,
                    "numb_fparam": 0,
                    "precision": "float64",
                    "seed": 1,
                },
            },
            trim_pattern="_*",
        )
        self.dp_model = get_model_dp(data)
        serialized = self.dp_model.serialize()
        self.pt_model = DOSModelPT.deserialize(serialized)
        self.pt_expt_model = DOSModelPTExpt.deserialize(serialized)

    def test_translated_output_def(self) -> None:
        """translated_output_def should return the same keys on dp, pt, and pt_expt."""
        dp_def = self.dp_model.translated_output_def()
        pt_def = self.pt_model.translated_output_def()
        pt_expt_def = self.pt_expt_model.translated_output_def()
        self.assertEqual(set(dp_def.keys()), set(pt_def.keys()))
        self.assertEqual(set(dp_def.keys()), set(pt_expt_def.keys()))
        for key in dp_def:
            self.assertEqual(dp_def[key].shape, pt_def[key].shape)
            self.assertEqual(dp_def[key].shape, pt_expt_def[key].shape)
