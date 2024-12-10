# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.model.ener_model import EnergyModel as EnergyModelDP
from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_JAX,
    INSTALLED_PD,
    INSTALLED_PT,
    INSTALLED_TF,
    SKIP_FLAG,
    CommonTest,
    parameterized,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.ener_model import EnergyModel as EnergyModelPT
else:
    EnergyModelPT = None
if INSTALLED_TF:
    from deepmd.tf.model.ener import EnerModel as EnergyModelTF
else:
    EnergyModelTF = None
from deepmd.utils.argcheck import (
    model_args,
)

if INSTALLED_PD:
    from deepmd.pd.model.model import get_model as get_model_pd
    from deepmd.pd.model.model.ener_model import EnergyModel as EnergyModelPD
else:
    EnergyModelPD = None
if INSTALLED_JAX:
    from deepmd.jax.model.ener_model import EnergyModel as EnergyModelJAX
    from deepmd.jax.model.model import get_model as get_model_jax
else:
    EnergyModelJAX = None


@parameterized(
    ("strip", "concat"),  # tebd_input_mode
    # strip + smooth is inconsistent
    (False,),  # smooth
)
class TestDPA1Ener(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            tebd_input_mode,
            smooth_type_embedding,
        ) = self.param
        return {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_atten",
                "sel": 40,
                "rcut_smth": 0.50,
                "rcut": 6.00,
                "neuron": [
                    3,
                    6,
                ],
                "resnet_dt": False,
                "axis_neuron": 2,
                "seed": 1,
                "attn": 128,
                "attn_layer": 0,
                "precision": "float64",
                "tebd_input_mode": tebd_input_mode,
                "smooth_type_embedding": smooth_type_embedding,
            },
            "fitting_net": {
                "neuron": [
                    5,
                    5,
                ],
                "resnet_dt": True,
                "precision": "float64",
                "seed": 1,
            },
        }

    tf_class = EnergyModelTF
    dp_class = EnergyModelDP
    pt_class = EnergyModelPT
    pd_class = EnergyModelPD
    jax_class = EnergyModelJAX
    args = model_args()

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_tf:
            return self.RefBackend.TF
        if not self.skip_pd:
            return self.RefBackend.PD
        if not self.skip_jax:
            return self.RefBackend.JAX
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is EnergyModelDP:
            return get_model_dp(data)
        elif cls is EnergyModelPT:
            return get_model_pt(data)
        elif cls is EnergyModelPD:
            return get_model_pd(data)
        elif cls is EnergyModelJAX:
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

    def eval_pd(self, pd_obj: Any) -> Any:
        return self.eval_pd_model(
            pd_obj,
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
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["force"].ravel(),
                ret["virial"].ravel(),
                ret["atom_virial"].ravel(),
            )
        elif backend is self.RefBackend.TF:
            return (
                ret[0].ravel(),
                ret[1].ravel(),
                ret[2].ravel(),
                ret[3].ravel(),
                ret[4].ravel(),
            )
        elif backend is self.RefBackend.PD:
            return (
                ret["energy"].flatten(),
                ret["atom_energy"].flatten(),
                ret["force"].flatten(),
                ret["virial"].flatten(),
                ret["atom_virial"].flatten(),
            )
        elif backend is self.RefBackend.JAX:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                ret["energy_derv_r"].ravel(),
                ret["energy_derv_c_redu"].ravel(),
                ret["energy_derv_c"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")
