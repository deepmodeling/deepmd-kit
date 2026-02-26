# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnerFittingDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PD,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)
from .common import (
    FittingTest,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.task.ener import EnergyFittingNet as EnerFittingPT
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
else:
    EnerFittingPT = object
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet as EnerFittingPTExpt,
    )
    from deepmd.pt_expt.utils.env import DEVICE as PT_EXPT_DEVICE
else:
    EnerFittingPTExpt = None
if INSTALLED_TF:
    from deepmd.tf.fit.ener import EnerFitting as EnerFittingTF
else:
    EnerFittingTF = object
if INSTALLED_PD:
    import paddle

    from deepmd.pd.model.task.ener import EnergyFittingNet as EnerFittingPD
    from deepmd.pd.utils.env import DEVICE as PD_DEVICE
else:
    EnerFittingPD = object
from deepmd.utils.argcheck import (
    fitting_ener,
)

if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
    from deepmd.jax.fitting.fitting import EnergyFittingNet as EnerFittingJAX
else:
    EnerFittingJAX = object
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict

    from ...array_api_strict.fitting.fitting import (
        EnergyFittingNet as EnerFittingStrict,
    )
else:
    EnerFittingStrict = None


@parameterized(
    (True, False),  # resnet_dt
    ("float64", "float32", "bfloat16"),  # precision
    (True, False),  # mixed_types
    ((0, None), (1, None), (1, [1.0])),  # (numb_fparam, default_fparam)
    ((0, False), (1, False), (1, True)),  # (numb_aparam, use_aparam_as_mask)
    ([], [-12345.6, None]),  # atom_ener
)
class TestEner(CommonTest, FittingTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return {
            "neuron": [5, 5, 5],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "numb_fparam": numb_fparam,
            "numb_aparam": numb_aparam,
            "default_fparam": default_fparam,
            "seed": 20240217,
            "atom_ener": atom_ener,
            "use_aparam_as_mask": use_aparam_as_mask,
            "activation_function": "relu",
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return CommonTest.skip_pt

    skip_jax = not INSTALLED_JAX

    @property
    def skip_array_api_strict(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        # TypeError: The array_api_strict namespace does not support the dtype 'bfloat16'
        return not INSTALLED_ARRAY_API_STRICT or precision == "bfloat16"

    @property
    def skip_pd(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        # Paddle do not support "bfloat16" in some kernels,
        # so skip this in CI test
        return not INSTALLED_PD or precision == "bfloat16" or default_fparam is not None

    @property
    def skip_tf(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return not INSTALLED_TF or default_fparam is not None

    @property
    def skip_pt_expt(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        # PyTorch does not support bfloat16 for some operations
        return CommonTest.skip_pt_expt or precision == "bfloat16"

    tf_class = EnerFittingTF
    dp_class = EnerFittingDP
    pt_class = EnerFittingPT
    pt_expt_class = EnerFittingPTExpt
    jax_class = EnerFittingJAX
    pd_class = EnerFittingPD
    array_api_strict_class = EnerFittingStrict
    args = fitting_ener()

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
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
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
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
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
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return (
            pt_obj(
                torch.from_numpy(self.inputs).to(device=PT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_DEVICE),
                fparam=(
                    torch.from_numpy(self.fparam).to(device=PT_DEVICE)
                    if (numb_fparam and default_fparam is None)  # test default_fparam
                    else None
                ),
                aparam=(
                    torch.from_numpy(self.aparam).to(device=PT_DEVICE)
                    if numb_aparam
                    else None
                ),
            )["energy"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return (
            pt_expt_obj(
                torch.from_numpy(self.inputs).to(device=PT_EXPT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_EXPT_DEVICE),
                fparam=(
                    torch.from_numpy(self.fparam).to(device=PT_EXPT_DEVICE)
                    if (numb_fparam and default_fparam is None)  # test default_fparam
                    else None
                ),
                aparam=(
                    torch.from_numpy(self.aparam).to(device=PT_EXPT_DEVICE)
                    if numb_aparam
                    else None
                ),
            )["energy"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return dp_obj(
            self.inputs,
            self.atype.reshape(1, -1),
            fparam=self.fparam if (numb_fparam and default_fparam is None) else None,
            aparam=self.aparam if numb_aparam else None,
        )["energy"]

    def eval_jax(self, jax_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return np.asarray(
            jax_obj(
                jnp.asarray(self.inputs),
                jnp.asarray(self.atype.reshape(1, -1)),
                fparam=jnp.asarray(self.fparam)
                if (numb_fparam and default_fparam is None)
                else None,
                aparam=jnp.asarray(self.aparam) if numb_aparam else None,
            )["energy"]
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return to_numpy_array(
            array_api_strict_obj(
                array_api_strict.asarray(self.inputs),
                array_api_strict.asarray(self.atype.reshape(1, -1)),
                fparam=array_api_strict.asarray(self.fparam)
                if (numb_fparam and default_fparam is None)
                else None,
                aparam=array_api_strict.asarray(self.aparam) if numb_aparam else None,
            )["energy"]
        )

    def eval_pd(self, pd_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return (
            pd_obj(
                paddle.to_tensor(self.inputs).to(device=PD_DEVICE),
                paddle.to_tensor(self.atype.reshape([1, -1])).to(device=PD_DEVICE),
                fparam=(
                    paddle.to_tensor(self.fparam).to(device=PD_DEVICE)
                    if numb_fparam
                    else None
                ),
                aparam=(
                    paddle.to_tensor(self.aparam).to(device=PD_DEVICE)
                    if numb_aparam
                    else None
                ),
            )["energy"]
            .detach()
            .cpu()
            .numpy()
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
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        elif precision == "bfloat16":
            return 1e-1
        else:
            raise ValueError(f"Unknown precision: {precision}")

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        elif precision == "bfloat16":
            return 1e-1
        else:
            raise ValueError(f"Unknown precision: {precision}")


@parameterized(
    (True,),  # resnet_dt
    ("float64",),  # precision
    (True,),  # mixed_types
    ((3, None),),  # (numb_fparam, default_fparam)
    ((3, False),),  # (numb_aparam, use_aparam_as_mask)
    ([],),  # atom_ener
)
class TestEnerStat(CommonTest, FittingTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        return {
            "neuron": [5, 5, 5],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "numb_fparam": numb_fparam,
            "numb_aparam": numb_aparam,
            "default_fparam": default_fparam,
            "seed": 20240217,
            "atom_ener": atom_ener,
            "use_aparam_as_mask": use_aparam_as_mask,
        }

    @property
    def skip_pt(self) -> bool:
        return CommonTest.skip_pt

    @property
    def skip_pt_expt(self) -> bool:
        return CommonTest.skip_pt_expt

    @property
    def skip_tf(self) -> bool:
        return True

    skip_jax = not INSTALLED_JAX

    @property
    def skip_array_api_strict(self) -> bool:
        return not INSTALLED_ARRAY_API_STRICT

    @property
    def skip_pd(self) -> bool:
        return not INSTALLED_PD

    tf_class = EnerFittingTF
    dp_class = EnerFittingDP
    pt_class = EnerFittingPT
    pt_expt_class = EnerFittingPTExpt
    jax_class = EnerFittingJAX
    pd_class = EnerFittingPD
    array_api_strict_class = EnerFittingStrict
    args = fitting_ener()

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        self.inputs = np.ones((1, 6, 20), dtype=GLOBAL_NP_FLOAT_PRECISION)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        # inconsistent if not sorted
        self.atype.sort()

        # Prepare data for compute_input_stats
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param

        # Create fparam and aparam with correct dimensions
        rng = np.random.default_rng(20240217)
        self.fparam = (
            rng.normal(size=(1, numb_fparam)).astype(GLOBAL_NP_FLOAT_PRECISION)
            if numb_fparam > 0
            else None
        )
        self.aparam = (
            rng.normal(size=(1, 6, numb_aparam)).astype(GLOBAL_NP_FLOAT_PRECISION)
            if numb_aparam > 0
            else None
        )

        self.stat_data = [
            {
                "fparam": rng.normal(size=(2, numb_fparam)).astype(
                    GLOBAL_NP_FLOAT_PRECISION
                ),
                "aparam": rng.normal(size=(2, 6, numb_aparam)).astype(
                    GLOBAL_NP_FLOAT_PRECISION
                ),
            },
            {
                "fparam": rng.normal(size=(3, numb_fparam)).astype(
                    GLOBAL_NP_FLOAT_PRECISION
                ),
                "aparam": rng.normal(size=(3, 6, numb_aparam)).astype(
                    GLOBAL_NP_FLOAT_PRECISION
                ),
            },
        ]

    @property
    def additional_data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
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
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
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
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        # Convert stat_data to torch tensors for pt backend
        pt_stat_data = [
            {
                "fparam": torch.from_numpy(d["fparam"]).to(PT_DEVICE),
                "aparam": torch.from_numpy(d["aparam"]).to(PT_DEVICE),
            }
            for d in self.stat_data
        ]
        pt_obj.compute_input_stats(pt_stat_data, protection=1e-2)
        return (
            pt_obj(
                torch.from_numpy(self.inputs).to(device=PT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_DEVICE),
                fparam=(
                    torch.from_numpy(self.fparam).to(device=PT_DEVICE)
                    if self.fparam is not None
                    else None
                ),
                aparam=(
                    torch.from_numpy(self.aparam).to(device=PT_DEVICE)
                    if self.aparam is not None
                    else None
                ),
            )["energy"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        # dpmodel's compute_input_stats accepts numpy arrays
        pt_expt_obj.compute_input_stats(self.stat_data, protection=1e-2)
        return (
            pt_expt_obj(
                torch.from_numpy(self.inputs).to(device=PT_EXPT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_EXPT_DEVICE),
                fparam=(
                    torch.from_numpy(self.fparam).to(device=PT_EXPT_DEVICE)
                    if self.fparam is not None
                    else None
                ),
                aparam=(
                    torch.from_numpy(self.aparam).to(device=PT_EXPT_DEVICE)
                    if self.aparam is not None
                    else None
                ),
            )["energy"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        dp_obj.compute_input_stats(self.stat_data, protection=1e-2)
        return dp_obj(
            self.inputs,
            self.atype.reshape(1, -1),
            fparam=self.fparam,
            aparam=self.aparam,
        )["energy"]

    def eval_jax(self, jax_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        # Convert stat_data to jax arrays
        jax_stat_data = [
            {
                "fparam": jnp.asarray(d["fparam"]),
                "aparam": jnp.asarray(d["aparam"]),
            }
            for d in self.stat_data
        ]
        jax_obj.compute_input_stats(jax_stat_data, protection=1e-2)
        return np.asarray(
            jax_obj(
                jnp.asarray(self.inputs),
                jnp.asarray(self.atype.reshape(1, -1)),
                fparam=jnp.asarray(self.fparam) if self.fparam is not None else None,
                aparam=jnp.asarray(self.aparam) if self.aparam is not None else None,
            )["energy"]
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        # Convert stat_data to array_api_strict arrays
        strict_stat_data = [
            {
                "fparam": array_api_strict.asarray(d["fparam"]),
                "aparam": array_api_strict.asarray(d["aparam"]),
            }
            for d in self.stat_data
        ]
        array_api_strict_obj.compute_input_stats(strict_stat_data, protection=1e-2)
        return to_numpy_array(
            array_api_strict_obj(
                array_api_strict.asarray(self.inputs),
                array_api_strict.asarray(self.atype.reshape(1, -1)),
                fparam=array_api_strict.asarray(self.fparam)
                if self.fparam is not None
                else None,
                aparam=array_api_strict.asarray(self.aparam)
                if self.aparam is not None
                else None,
            )["energy"]
        )

    def eval_pd(self, pd_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        # Convert stat_data to paddle tensors
        pd_stat_data = [
            {
                "fparam": paddle.to_tensor(d["fparam"]).to(PD_DEVICE),
                "aparam": paddle.to_tensor(d["aparam"]).to(PD_DEVICE),
            }
            for d in self.stat_data
        ]
        pd_obj.compute_input_stats(pd_stat_data, protection=1e-2)
        return (
            pd_obj(
                paddle.to_tensor(self.inputs).to(device=PD_DEVICE),
                paddle.to_tensor(self.atype.reshape([1, -1])).to(device=PD_DEVICE),
                fparam=(
                    paddle.to_tensor(self.fparam).to(device=PD_DEVICE)
                    if self.fparam is not None
                    else None
                ),
                aparam=(
                    paddle.to_tensor(self.aparam).to(device=PD_DEVICE)
                    if self.aparam is not None
                    else None
                ),
            )["energy"]
            .detach()
            .cpu()
            .numpy()
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
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
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
            (numb_fparam, default_fparam),
            (numb_aparam, use_aparam_as_mask),
            atom_ener,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
