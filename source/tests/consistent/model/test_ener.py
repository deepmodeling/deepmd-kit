# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA as DescrptSeADP
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnerFittingDP
from deepmd.dpmodel.model.dp_model import DPModel as EnergyModelDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import (
        get_model,
    )
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


# @parameterized(
#     (True, False),  # resnet_dt
#     (True, False),  # type_one_side
#     ([], [[0, 1]]),  # excluded_types
# )
class TestSeA(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        return {
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
    args = model_args()

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is EnergyModelDP:
            # should not do it here...
            data["descriptor"].pop("type")
            data["fitting_net"].pop("type")
            descriptor = DescrptSeADP(
                **data["descriptor"],
            )
            fitting = EnerFittingDP(
                ntypes=descriptor.get_ntypes(),
                dim_descrpt=descriptor.get_dim_out(),
                **data["fitting_net"],
            )
            return cls(
                descriptor=descriptor,
                fitting=fitting,
                type_map=data["type_map"],
            )
        elif cls is EnergyModelPT:
            return get_model(data)
        return cls(**data, **self.addtional_data)

    def setUp(self):
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

    def build_tf(self, obj: Any, suffix: str) -> Tuple[list, dict]:
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

    def extract_ret(self, ret: Any, backend) -> Tuple[np.ndarray, ...]:
        if backend is self.RefBackend.DP:
            return (ret["energy_redu"].ravel(), ret["energy"])
        elif backend is self.RefBackend.PT:
            return (ret["energy"].ravel(), ret["atom_energy"])
        elif backend is self.RefBackend.TF:
            return (ret[0], ret[1])
        raise ValueError(f"Unknown backend: {backend}")
