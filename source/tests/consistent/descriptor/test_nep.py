# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.descriptor.nep import DescrptNep as DescrptNepDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.argcheck import (
    descrpt_nep_args,
)

from ..common import (
    INSTALLED_JAX,
    INSTALLED_PT_EXPT,
    CommonTest,
    parameterized,
)
from .common import (
    DescriptorAPITest,
    DescriptorTest,
)

if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.descriptor.nep import DescrptNep as DescrptNepPTExpt
else:
    DescrptNepPTExpt = None
if INSTALLED_JAX:
    from deepmd.jax.descriptor.nep import DescrptNep as DescrptNepJAX
else:
    DescrptNepJAX = None


@parameterized(
    ("float32", "float64"),  # precision
)
class TestNep(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (precision,) = self.param
        return {
            "sel": [9, 10],
            "rcut_radial": 6.00,
            "rcut_angular": 4.00,
            "n_max_radial": 3,
            "n_max_angular": 2,
            "basis_size_radial": 4,
            "basis_size_angular": 3,
            "l_max": 2,
            "l_max_4body": 2,
            "l_max_5body": 1,
            "precision": precision,
            "seed": 1145141919810,
        }

    # NEP is implemented only for the dpmodel reference and the JAX / pt_expt
    # backends; TF, PT (hard-coded), PD, and array-api-strict are not provided.
    skip_tf = True
    skip_pt = True
    skip_pd = True
    skip_array_api_strict = True

    @property
    def skip_dp(self) -> bool:
        return CommonTest.skip_dp

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    @property
    def skip_pt_expt(self) -> bool:
        return not INSTALLED_PT_EXPT

    tf_class = None
    dp_class = DescrptNepDP
    pt_class = None
    pt_expt_class = DescrptNepPTExpt
    jax_class = DescrptNepJAX
    array_api_strict_class = None
    args = descrpt_nep_args()

    def setUp(self) -> None:
        CommonTest.setUp(self)
        self.ntypes = 2
        self.coords = np.array(
            [
                12.83, 2.56, 2.18,
                12.09, 2.87, 2.74,
                0.25, 3.32, 1.68,
                3.36, 3.00, 1.81,
                3.51, 2.51, 2.60,
                4.27, 3.22, 1.56,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        # NEP has no TensorFlow backend; this branch is always skipped.
        raise NotImplementedError

    def eval_pt(self, pt_obj: Any) -> Any:
        # NEP has no hard-coded PyTorch backend; this branch is always skipped.
        raise NotImplementedError

    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_descriptor(
            dp_obj, self.natoms, self.coords, self.atype, self.box
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        return self.eval_pt_expt_descriptor(
            pt_expt_obj, self.natoms, self.coords, self.atype, self.box
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_descriptor(
            jax_obj, self.natoms, self.coords, self.atype, self.box
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        # The angular invariants square the harmonic sums, which amplifies the
        # float32 reduction-order differences between numpy and JAX/XLA; float64
        # remains tight.
        (precision,) = self.param
        return 1e-10 if precision == "float64" else 1e-3

    @property
    def atol(self) -> float:
        (precision,) = self.param
        return 1e-10 if precision == "float64" else 1e-3


@parameterized(
    ("float64",),  # precision
)
class TestNepDescriptorAPI(DescriptorAPITest, unittest.TestCase):
    """Test consistency of BaseDescriptor API methods across backends."""

    dp_class = DescrptNepDP
    pt_class = None
    pt_expt_class = DescrptNepPTExpt
    args = descrpt_nep_args()

    @property
    def data(self) -> dict:
        (precision,) = self.param
        return {
            "sel": [9, 10],
            "rcut_radial": 6.00,
            "rcut_angular": 4.00,
            "n_max_radial": 3,
            "n_max_angular": 2,
            "basis_size_radial": 4,
            "basis_size_angular": 3,
            "l_max": 2,
            "precision": precision,
            "seed": 1145141919810,
        }

    @property
    def skip_pt(self) -> bool:
        return True

    @property
    def skip_pt_expt(self) -> bool:
        return not INSTALLED_PT_EXPT
