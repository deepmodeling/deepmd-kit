# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    SKIP_FLAG,
    CommonTest,
    parameterized,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.dp_linear_model import (
        LinearEnergyModel as LinearEnergyModelPT,
    )
else:
    LinearEnergyModelPT = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.model.dp_linear_model import (
        LinearEnergyModel as LinearEnergyModelPTExpt,
    )
    from deepmd.pt_expt.model.get_model import (
        get_linear_model as get_linear_model_pt_expt,
    )
else:
    LinearEnergyModelPTExpt = None
from deepmd.utils.argcheck import (
    model_args,
)


@parameterized(
    (
        [],
        [[0, 1]],
    ),
    (
        [],
        [1],
    ),
)
class TestLinearEner(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        pair_exclude_types, atom_exclude_types = self.param
        return {
            "type": "linear_ener",
            "type_map": ["O", "H", "B"],
            "models": [
                {
                    "descriptor": {
                        "type": "se_atten",
                        "sel": 40,
                        "rcut_smth": 0.5,
                        "rcut": 4.0,
                        "neuron": [3, 6],
                        "axis_neuron": 2,
                        "attn": 8,
                        "attn_layer": 2,
                        "attn_dotr": True,
                        "attn_mask": False,
                        "activation_function": "tanh",
                        "scaling_factor": 1.0,
                        "normalize": False,
                        "temperature": 1.0,
                        "set_davg_zero": True,
                        "type_one_side": True,
                        "seed": 1,
                    },
                    "fitting_net": {
                        "neuron": [5, 5],
                        "resnet_dt": True,
                        "seed": 1,
                    },
                },
                {
                    "descriptor": {
                        "type": "se_atten",
                        "sel": 40,
                        "rcut_smth": 0.5,
                        "rcut": 4.0,
                        "neuron": [3, 6],
                        "axis_neuron": 2,
                        "attn": 8,
                        "attn_layer": 2,
                        "attn_dotr": True,
                        "attn_mask": False,
                        "activation_function": "tanh",
                        "scaling_factor": 1.0,
                        "normalize": False,
                        "temperature": 1.0,
                        "set_davg_zero": True,
                        "type_one_side": True,
                        "seed": 2,
                    },
                    "fitting_net": {
                        "neuron": [5, 5],
                        "resnet_dt": True,
                        "seed": 2,
                    },
                },
            ],
            "weights": "mean",
            "pair_exclude_types": pair_exclude_types,
            "atom_exclude_types": atom_exclude_types,
        }

    dp_class = None
    pt_class = LinearEnergyModelPT
    pt_expt_class = LinearEnergyModelPTExpt
    args = model_args()

    def get_reference_backend(self):
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_pt_expt and self.pt_expt_class is not None:
            return self.RefBackend.PT_EXPT
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    @property
    def skip_tf(self) -> bool:
        return True

    @property
    def skip_jax(self) -> bool:
        return True

    @property
    def skip_dp(self) -> bool:
        return True

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is LinearEnergyModelPT:
            return get_model_pt(data)
        elif cls is LinearEnergyModelPTExpt:
            return get_linear_model_pt_expt(data)
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

        idx_map = np.argsort(self.atype.ravel())
        self.atype = self.atype[:, idx_map]
        self.coords = self.coords[:, idx_map]

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        return self.build_tf_model(
            obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            suffix,
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

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        if backend is self.RefBackend.DP:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend in {
            self.RefBackend.PT,
            self.RefBackend.PT_EXPT,
        }:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["force"].ravel(),
                ret["virial"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")
