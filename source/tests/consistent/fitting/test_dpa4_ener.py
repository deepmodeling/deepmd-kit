# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.fitting.dpa4_ener import SeZMEnergyFittingNet as SeZMEnerFittingDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.argcheck import (
    fitting_sezm_ener,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    CommonTest,
    parameterized_cases,
)
from .common import (
    FittingTest,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.task.sezm_ener import SeZMEnergyFittingNet as SeZMEnerFittingPT
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
else:
    SeZMEnerFittingPT = None
if INSTALLED_PT_EXPT:
    import torch

    from deepmd.pt_expt.fitting.dpa4_ener import (
        SeZMEnergyFittingNet as SeZMEnerFittingPTExpt,
    )
    from deepmd.pt_expt.utils.env import DEVICE as PT_EXPT_DEVICE
else:
    SeZMEnerFittingPTExpt = None
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict

    from ...array_api_strict.fitting.dpa4_ener import (
        SeZMEnergyFittingNet as SeZMEnerFittingStrict,
    )
else:
    SeZMEnerFittingStrict = None

# not implemented
SeZMEnerFittingTF = None


DPA4_ENER_FITTING_CURATED_CASES = (
    ("float64", [0]),
    ("float64", [16, 16]),
    ("float32", [0]),
    ("float32", [16, 16]),
)


@parameterized_cases(*DPA4_ENER_FITTING_CURATED_CASES)
class TestDPA4Ener(CommonTest, FittingTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            precision,
            neuron,
        ) = self.param
        return {
            "neuron": neuron,
            "precision": precision,
            "seed": 20251208,
            "activation_function": "silu",
        }

    @property
    def skip_pt(self) -> bool:
        return CommonTest.skip_pt

    skip_dp = False
    skip_tf = True
    skip_jax = True
    skip_pd = True
    skip_pt_expt = not INSTALLED_PT_EXPT
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

    tf_class = SeZMEnerFittingTF
    dp_class = SeZMEnerFittingDP
    pt_class = SeZMEnerFittingPT
    pt_expt_class = SeZMEnerFittingPTExpt
    jax_class = None
    pd_class = None
    array_api_strict_class = SeZMEnerFittingStrict
    args = fitting_sezm_ener()

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        rng = np.random.default_rng(20251208)
        self.inputs = rng.normal(size=(1, 6, 20)).astype(GLOBAL_NP_FLOAT_PRECISION)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        # inconsistent if not sorted
        self.atype.sort()

    @property
    def additional_data(self) -> dict:
        return {
            "ntypes": self.ntypes,
            "dim_descrpt": self.inputs.shape[-1],
            "mixed_types": True,
        }

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError("dpa4_ener is not implemented in TensorFlow")

    def eval_dp(self, dp_obj: Any) -> Any:
        return dp_obj(
            self.inputs,
            self.atype.reshape(1, -1),
        )["energy"]

    def eval_pt(self, pt_obj: Any) -> Any:
        return (
            pt_obj(
                torch.from_numpy(self.inputs).to(device=PT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_DEVICE),
            )["energy"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        return (
            pt_expt_obj(
                torch.from_numpy(self.inputs).to(device=PT_EXPT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_EXPT_DEVICE),
            )["energy"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        return to_numpy_array(
            array_api_strict_obj(
                array_api_strict.asarray(self.inputs),
                array_api_strict.asarray(self.atype.reshape(1, -1)),
            )["energy"]
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret,)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            precision,
            _neuron,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        (
            precision,
            _neuron,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
