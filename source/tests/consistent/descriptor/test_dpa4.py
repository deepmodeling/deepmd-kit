# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    ClassVar,
)

import numpy as np
from dargs import (
    Argument,
)

from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4DP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.argcheck import (
    descrpt_se_zm_args,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    CommonTest,
    parameterized_cases,
)
from .common import (
    DescriptorTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.descriptor.sezm import DescrptSeZM as DescrptDPA4PT
else:
    DescrptDPA4PT = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4PTExpt
else:
    DescrptDPA4PTExpt = None
if INSTALLED_JAX:
    from deepmd.jax.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4JAX
else:
    DescrptDPA4JAX = None
if INSTALLED_ARRAY_API_STRICT:
    from ...array_api_strict.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4Strict
else:
    DescrptDPA4Strict = None

# not implemented
DescrptDPA4TF = None


DPA4_CASE_FIELDS = (
    "precision",
    "grid_branch",
    "s2_activation",
    "basis_type",
    "ffn_so3_grid",
    "message_node_so3",
    "grid_mlp",
    "so3_readout",
)

DPA4_BASELINE_CASE = {
    "precision": "float64",
    "grid_branch": [1, 1, 1],
    "s2_activation": [False, True],
    "basis_type": "bessel",
    "ffn_so3_grid": False,
    "message_node_so3": False,
    "grid_mlp": False,
    "so3_readout": "none",
}


def dpa4_case(**overrides: Any) -> tuple:
    unknown = set(overrides) - set(DPA4_BASELINE_CASE)
    if unknown:
        raise KeyError(f"Unknown DPA4 case override(s): {sorted(unknown)}")
    case = dict(DPA4_BASELINE_CASE)
    case.update(overrides)
    return tuple(case[field] for field in DPA4_CASE_FIELDS)


# curated cases (one-factor-at-a-time, dpa3 precedent) instead of full
# cross product to keep CI runtime sane
DPA4_CURATED_CASES = (
    # baseline coverage
    dpa4_case(),
    # grid branch disabled
    dpa4_case(grid_branch=[0, 0, 0]),
    # no S2 activation in any FFN
    dpa4_case(s2_activation=[False, False]),
    # gaussian radial basis
    dpa4_case(basis_type="gaussian"),
    # float32 baseline
    dpa4_case(precision="float32"),
    # float32 mixed high-risk path
    dpa4_case(
        precision="float32",
        grid_branch=[0, 0, 0],
        s2_activation=[False, False],
        basis_type="gaussian",
    ),
    # SO(3) Wigner-D FFN grid (example-config flag)
    dpa4_case(ffn_so3_grid=True),
    # post-aggregation SO(3) cross-grid message (example-config flag)
    dpa4_case(message_node_so3=True),
    # both SO(3) grid paths on (mirrors examples/water/dpa4/input.json)
    dpa4_case(ffn_so3_grid=True, message_node_so3=True),
    # polynomial grid MLP op (grid_branch=0 so grid_mlp takes effect)
    dpa4_case(grid_mlp=True, grid_branch=[0, 0, 0]),
    # SO(3) readout: GLU grid product folds l>0 into the l=0 output
    dpa4_case(so3_readout="glu"),
    # SO(3) readout: point-wise grid MLP folds l>0 into the l=0 output
    dpa4_case(so3_readout="mlp"),
)


@parameterized_cases(*DPA4_CURATED_CASES)
class TestDPA4(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            precision,
            grid_branch,
            s2_activation,
            basis_type,
            ffn_so3_grid,
            message_node_so3,
            grid_mlp,
            so3_readout,
        ) = self.param
        return {
            "ntypes": self.ntypes,
            "sel": 10,
            "rcut": 4.0,
            "channels": 16,
            "n_radial": 8,
            "basis_type": basis_type,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "grid_branch": grid_branch,
            "s2_activation": s2_activation,
            "ffn_so3_grid": ffn_so3_grid,
            "message_node_so3": message_node_so3,
            "grid_mlp": grid_mlp,
            "so3_readout": so3_readout,
            "random_gamma": False,
            # JAX currently supports DPA4 without the backend-specific AMP
            # policy. Keep cross-backend consistency cases within that shared
            # feature subset; AMP behavior has dedicated backend tests.
            "use_amp": False,
            "precision": precision,
            "trainable": False,
            "seed": 20251208,
        }

    @property
    def skip_pt(self) -> bool:
        return CommonTest.skip_pt

    skip_dp = False
    skip_tf = True
    skip_jax = not INSTALLED_JAX or DescrptDPA4JAX is None
    skip_pd = True
    skip_pt_expt = not INSTALLED_PT_EXPT
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

    tf_class = DescrptDPA4TF
    dp_class = DescrptDPA4DP
    pt_class = DescrptDPA4PT
    pt_expt_class = DescrptDPA4PTExpt
    jax_class = DescrptDPA4JAX
    pd_class = None
    array_api_strict_class = DescrptDPA4Strict
    args: ClassVar[list] = [
        *descrpt_se_zm_args(),
        Argument("ntypes", int, optional=False),
    ]

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
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError("DPA4 is not implemented in TensorFlow")

    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_descriptor(
            dp_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_descriptor(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        return self.eval_pt_expt_descriptor(
            pt_expt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_descriptor(
            jax_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        return self.eval_array_api_strict_descriptor(
            array_api_strict_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        precision = self.param[0]
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        precision = self.param[0]
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
