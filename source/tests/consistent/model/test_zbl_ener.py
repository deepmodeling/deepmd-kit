# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.model.dp_zbl_model import DPZBLModel as DPZBLModelDP
from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_JAX,
    INSTALLED_PT,
    SKIP_FLAG,
    CommonTest,
    parameterized,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.dp_zbl_model import DPZBLModel as DPZBLModelPT
else:
    DPZBLModelPT = None
if INSTALLED_JAX:
    from deepmd.jax.model.dp_zbl_model import DPZBLModel as DPZBLModelJAX
    from deepmd.jax.model.model import get_model as get_model_jax
else:
    DPZBLModelJAX = None
import os

from deepmd.utils.argcheck import (
    model_args,
)

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


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
class TestEner(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        pair_exclude_types, atom_exclude_types = self.param
        return {
            "type_map": ["O", "H", "B"],
            "use_srtab": f"{TESTS_DIR}/pt/water/data/zbl_tab_potential/H2O_tab_potential.txt",
            "smin_alpha": 0.1,
            "sw_rmin": 0.2,
            "sw_rmax": 4.0,
            "pair_exclude_types": pair_exclude_types,
            "atom_exclude_types": atom_exclude_types,
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
        }

    dp_class = DPZBLModelDP
    pt_class = DPZBLModelPT
    jax_class = DPZBLModelJAX
    args = model_args()

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_tf:
            return self.RefBackend.TF
        if not self.skip_jax:
            return self.RefBackend.JAX
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    @property
    def skip_tf(self) -> bool:
        return True

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is DPZBLModelDP:
            return get_model_dp(data)
        elif cls is DPZBLModelPT:
            return get_model_pt(data)
        elif cls is DPZBLModelJAX:
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
        if backend is self.RefBackend.DP:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["force"].ravel(),
                ret["virial"].ravel(),
            )
        elif backend is self.RefBackend.TF:
            return (ret[0].ravel(), ret[1].ravel(), ret[2].ravel(), ret[3].ravel())
        elif backend is self.RefBackend.JAX:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                ret["energy_derv_r"].ravel(),
                ret["energy_derv_c_redu"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")
