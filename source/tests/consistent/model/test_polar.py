# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.dpmodel.model.polar_model import PolarModel as PolarModelDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.polar_model import PolarModel as PolarModelPT
else:
    PolarModelPT = None
if INSTALLED_TF:
    from deepmd.tf.model.tensor import PolarModel as PolarModelTF
else:
    PolarModelTF = None
if INSTALLED_JAX:
    from deepmd.jax.model.model import get_model as get_model_jax
    from deepmd.jax.model.polar_model import PolarModel as PolarModelJAX
else:
    PolarModelJAX = None
from deepmd.utils.argcheck import (
    model_args,
)


class TestPolar(CommonTest, ModelTest, unittest.TestCase):
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
                "type": "polar",
                "neuron": [4, 4, 4],
                "resnet_dt": True,
                "numb_fparam": 0,
                "precision": "float64",
                "seed": 1,
            },
        }

    tf_class = PolarModelTF
    dp_class = PolarModelDP
    pt_class = PolarModelPT
    jax_class = PolarModelJAX
    args = model_args()

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_tf:
            return self.RefBackend.TF
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    @property
    def skip_tf(self):
        return True  # need to fix tf consistency

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is PolarModelDP:
            return get_model_dp(data)
        elif cls is PolarModelPT:
            model = get_model_pt(data)
            model.atomic_model.out_bias.uniform_()
            return model
        elif cls is PolarModelJAX:
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
            obj, self.natoms, self.coords, self.atype, self.box, suffix, ret_key="polar"
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
        if backend in {self.RefBackend.DP, self.RefBackend.JAX}:
            return (
                ret["polarizability_redu"].ravel(),
                ret["polarizability"].ravel(),
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["global_polar"].ravel(),
                ret["polar"].ravel(),
            )
        elif backend is self.RefBackend.TF:
            return (
                ret[0].ravel(),
                ret[1].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")
