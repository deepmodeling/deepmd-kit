# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.fitting.dos_fitting import DOSFittingNet as DOSFittingDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)
from .common import (
    FittingTest,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.task.dos import DOSFittingNet as DOSFittingPT
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
else:
    DOSFittingPT = object
if INSTALLED_TF:
    from deepmd.tf.fit.dos import DOSFitting as DOSFittingTF
else:
    DOSFittingTF = object
from deepmd.utils.argcheck import (
    fitting_dos,
)

if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
    from deepmd.jax.fitting.fitting import DOSFittingNet as DOSFittingJAX
else:
    DOSFittingJAX = object
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict

    from ...array_api_strict.fitting.fitting import DOSFittingNet as DOSFittingStrict
else:
    DOSFittingStrict = object


@parameterized(
    (True, False),  # resnet_dt
    ("float64", "float32"),  # precision
    (True, False),  # mixed_types
    (0, 1),  # numb_fparam
    (0, 1),  # numb_aparam
    (10, 20),  # numb_dos
)
class TestDOS(CommonTest, FittingTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        return {
            "neuron": [5, 5, 5],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "numb_fparam": numb_fparam,
            "numb_aparam": numb_aparam,
            "seed": 20240217,
            "numb_dos": numb_dos,
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    @property
    def skip_array_api_strict(self) -> bool:
        return not INSTALLED_ARRAY_API_STRICT

    tf_class = DOSFittingTF
    dp_class = DOSFittingDP
    pt_class = DOSFittingPT
    jax_class = DOSFittingJAX
    array_api_strict_class = DOSFittingStrict
    args = fitting_dos()

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        self.inputs = np.ones((1, 6, 20), dtype=GLOBAL_NP_FLOAT_PRECISION)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        # inconsistent if not sorted
        self.atype.sort()
        self.fparam = -np.ones((1,), dtype=GLOBAL_NP_FLOAT_PRECISION)
        self.aparam = np.zeros_like(
            self.atype, dtype=GLOBAL_NP_FLOAT_PRECISION
        ).reshape(-1, 1)

    @property
    def additional_data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        return {
            "ntypes": self.ntypes,
            "dim_descrpt": self.inputs.shape[-1],
            "mixed_types": mixed_types,
        }

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        return self.build_tf_fitting(
            obj,
            self.inputs.ravel(),
            self.natoms,
            self.atype,
            self.fparam if numb_fparam else None,
            self.aparam if numb_aparam else None,
            suffix,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        return (
            pt_obj(
                torch.from_numpy(self.inputs).to(device=PT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_DEVICE),
                fparam=torch.from_numpy(self.fparam).to(device=PT_DEVICE)
                if numb_fparam
                else None,
                aparam=torch.from_numpy(self.aparam).to(device=PT_DEVICE)
                if numb_aparam
                else None,
            )["dos"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        return dp_obj(
            self.inputs,
            self.atype.reshape(1, -1),
            fparam=self.fparam if numb_fparam else None,
            aparam=self.aparam if numb_aparam else None,
        )["dos"]

    def eval_jax(self, jax_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        return np.asarray(
            jax_obj(
                jnp.asarray(self.inputs),
                jnp.asarray(self.atype.reshape(1, -1)),
                fparam=jnp.asarray(self.fparam) if numb_fparam else None,
                aparam=jnp.asarray(self.aparam) if numb_aparam else None,
            )["dos"]
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        return to_numpy_array(
            array_api_strict_obj(
                array_api_strict.asarray(self.inputs),
                array_api_strict.asarray(self.atype.reshape(1, -1)),
                fparam=array_api_strict.asarray(self.fparam) if numb_fparam else None,
                aparam=array_api_strict.asarray(self.aparam) if numb_aparam else None,
            )["dos"]
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        if backend == self.RefBackend.TF:
            # shape is not same
            ret = ret[0].reshape(-1, self.natoms[0], 1)
        return (ret,)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
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
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            numb_aparam,
            numb_dos,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
