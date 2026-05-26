# Test Patterns for Descriptors

## 8a. dpmodel self-consistency test

**Create** `source/tests/common/dpmodel/test_descriptor_<name>.py`

```python
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import DescrptYourName

from ...seed import GLOBAL_SEED
from .case_single_frame_with_nlist import TestCaseSingleFrameWithNlist


class TestDescrptYourName(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(
            size=(self.nt, nnei, 4)
        )  # 4 for full env mat, 1 for radial-only
        dstd = 0.1 + np.abs(
            rng.normal(size=(self.nt, nnei, 4))
        )  # 4 for full env mat, 1 for radial-only

        em0 = DescrptYourName(self.rcut, self.rcut_smth, self.sel)
        em0.davg = davg
        em0.dstd = dstd
        em1 = DescrptYourName.deserialize(em0.serialize())
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist)
        mm1 = em1.call(self.coord_ext, self.atype_ext, self.nlist)
        for ii in [0, 4]:  # descriptor and sw
            np.testing.assert_allclose(mm0[ii], mm1[ii])
```

Reference: `source/tests/common/dpmodel/test_descriptor_se_t.py`

## 8b. pt_expt unit tests

**Create** `source/tests/pt_expt/descriptor/test_<name>.py`

Three test types: consistency, exportable, make_fx. Use `pytest.mark.parametrize` with trailing comments explaining each parameter. Do **not** inherit from `unittest.TestCase`. Use `setup_method` instead of `setUp`.

```python
import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from deepmd.dpmodel.descriptor import DescrptYourName as DPDescrptYourName
from deepmd.pt_expt.descriptor.your_name import DescrptYourName
from deepmd.pt_expt.utils import env
from deepmd.pt_expt.utils.env import PRECISION_DICT

from ...pt.model.test_env_mat import TestCaseSingleFrameWithNlist
from ...pt.model.test_mlp import get_tols
from ...seed import GLOBAL_SEED


class TestDescrptYourName(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("idt", [False, True])  # resnet_dt
    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_consistency(self, idt, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        err_msg = f"idt={idt} prec={prec}"
        dd0 = DescrptYourName(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            resnet_dt=idt,
            seed=GLOBAL_SEED,
        ).to(self.device)
        dd0.davg = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)
        rd0, _, _, _, sw0 = dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        # Serialize/deserialize round-trip
        dd1 = DescrptYourName.deserialize(dd0.serialize())
        rd1, _, _, _, sw1 = dd1(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd1.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )
        np.testing.assert_allclose(
            sw0.detach().cpu().numpy(),
            sw1.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )
        # Permutation equivariance
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy()[0][self.perm[: self.nloc]],
            rd0.detach().cpu().numpy()[1],
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )
        # Compare with dpmodel
        dd2 = DPDescrptYourName.deserialize(dd0.serialize())
        rd2, _, _, _, sw2 = dd2.call(
            self.coord_ext,
            self.atype_ext,
            self.nlist,
        )
        np.testing.assert_allclose(
            rd1.detach().cpu().numpy(), rd2, rtol=rtol, atol=atol, err_msg=err_msg
        )
        np.testing.assert_allclose(
            sw1.detach().cpu().numpy(), sw2, rtol=rtol, atol=atol, err_msg=err_msg
        )

    @pytest.mark.parametrize("idt", [False, True])  # resnet_dt
    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_exportable(self, idt, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        dd0 = DescrptYourName(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            resnet_dt=idt,
            seed=GLOBAL_SEED,
        ).to(self.device)
        dd0.davg = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0 = dd0.eval()
        inputs = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        torch.export.export(dd0, inputs)

    @pytest.mark.parametrize("prec", ["float64"])  # precision — float64 only
    def test_make_fx(self, prec) -> None:
        """Verify make_fx traces forward + autograd (for forward_lower)."""
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        dd0 = DescrptYourName(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)
        dd0.davg = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0 = dd0.eval()

        coord_ext = torch.tensor(self.coord_ext, dtype=dtype, device=self.device)
        atype_ext = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)

        def fn(coord_ext, atype_ext, nlist):
            coord_ext = coord_ext.detach().requires_grad_(True)
            rd = dd0(coord_ext, atype_ext, nlist)[0]
            grad = torch.autograd.grad(rd.sum(), coord_ext, create_graph=False)[0]
            return rd, grad

        rd_eager, grad_eager = fn(coord_ext, atype_ext, nlist)
        traced = make_fx(fn)(coord_ext, atype_ext, nlist)
        rd_traced, grad_traced = traced(coord_ext, atype_ext, nlist)
        np.testing.assert_allclose(
            rd_eager.detach().cpu().numpy(),
            rd_traced.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            grad_eager.detach().cpu().numpy(),
            grad_traced.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
```

Reference: `source/tests/pt_expt/descriptor/test_se_t.py`

## 8e. array_api_strict wrapper

**Create** `source/tests/array_api_strict/descriptor/<name>.py`

```python
from typing import Any

from deepmd.dpmodel.descriptor.your_name import DescrptYourName as DescrptYourNameDP

from ..common import to_array_api_strict_array
from ..utils.exclude_mask import PairExcludeMask
from ..utils.network import NetworkCollection
from .base_descriptor import BaseDescriptor


@BaseDescriptor.register("your_name")
class DescrptYourName(DescrptYourNameDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"dstd", "davg"}:
            value = to_array_api_strict_array(value)
        elif name in {"embeddings"}:
            if value is not None:
                value = NetworkCollection.deserialize(value.serialize())
        elif name == "env_mat":
            pass
        elif name == "emask":
            value = PairExcludeMask(value.ntypes, value.exclude_types)
        return super().__setattr__(name, value)
```

**Edit** `source/tests/array_api_strict/descriptor/__init__.py` — add import and `__all__` entry.

Reference: `source/tests/array_api_strict/descriptor/se_e2_r.py`

## 8h. Cross-backend consistency test

**Create** `source/tests/consistent/descriptor/test_<name>.py`

Two test classes: one for numerical consistency (`CommonTest`), one for API consistency (`DescriptorAPITest`).

```python
import unittest
from typing import Any

import numpy as np

from deepmd.dpmodel.descriptor.your_name import DescrptYourName as DescrptYourNameDP
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.utils.argcheck import descrpt_your_name_args

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)
from .common import DescriptorAPITest, DescriptorTest

# Conditional imports for each backend.
# Omit any backend that has no implementation for this descriptor.
if INSTALLED_PT:
    from deepmd.pt.model.descriptor.your_name import DescrptYourName as YourNamePT
else:
    YourNamePT = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.descriptor.your_name import DescrptYourName as YourNamePTExpt
else:
    YourNamePTExpt = None
if INSTALLED_TF:
    from deepmd.tf.descriptor.your_name import DescrptYourName as YourNameTF
else:
    YourNameTF = None
if INSTALLED_JAX:
    from deepmd.jax.descriptor.your_name import DescrptYourName as YourNameJAX
else:
    YourNameJAX = None
if INSTALLED_ARRAY_API_STRICT:
    from ...array_api_strict.descriptor.your_name import (
        DescrptYourName as YourNameStrict,
    )
else:
    YourNameStrict = None


@parameterized(
    (True, False),  # resnet_dt
    ("float32", "float64"),  # precision
)
class TestYourName(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        resnet_dt, precision = self.param
        return {
            "sel": [9, 10],
            "rcut_smth": 5.80,
            "rcut": 6.00,
            "neuron": [6, 12, 24],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "seed": 1145141919810,
            "activation_function": "relu",
        }

    @property
    def skip_pt(self) -> bool:
        return CommonTest.skip_pt

    @property
    def skip_pt_expt(self) -> bool:
        # Add parameter-based skips here if needed, e.g.:
        # return (not some_supported_param) or CommonTest.skip_pt_expt
        return CommonTest.skip_pt_expt

    @property
    def skip_dp(self) -> bool:
        return CommonTest.skip_dp

    @property
    def skip_jax(self) -> bool:
        return CommonTest.skip_jax

    @property
    def skip_array_api_strict(self) -> bool:
        return CommonTest.skip_array_api_strict

    tf_class = YourNameTF
    dp_class = DescrptYourNameDP
    pt_class = YourNamePT
    pt_expt_class = YourNamePTExpt
    jax_class = YourNameJAX
    array_api_strict_class = YourNameStrict
    args = descrpt_your_name_args()

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
                0.25,
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

    # Implement eval_* methods using self.eval_*_descriptor() helpers.
    # For mixed_types descriptors (dpa1, dpa2, dpa3, se_atten_v2),
    # pass mixed_types=True to each eval call.
    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_descriptor(
            dp_obj, self.natoms, self.coords, self.atype, self.box
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_descriptor(
            pt_obj, self.natoms, self.coords, self.atype, self.box
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        return self.eval_pt_expt_descriptor(
            pt_expt_obj, self.natoms, self.coords, self.atype, self.box
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_descriptor(
            jax_obj, self.natoms, self.coords, self.atype, self.box
        )

    def eval_array_api_strict(self, obj: Any) -> Any:
        return self.eval_array_api_strict_descriptor(
            obj, self.natoms, self.coords, self.atype, self.box
        )

    def extract_ret(self, ret: Any, backend: Any) -> tuple:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        _, precision = self.param
        return 1e-10 if precision == "float64" else 1e-4

    @property
    def atol(self) -> float:
        _, precision = self.param
        return 1e-10 if precision == "float64" else 1e-4


@parameterized(
    ("float64",),  # precision — API test only needs one precision
)
class TestYourNameAPI(DescriptorAPITest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (precision,) = self.param
        return {
            "sel": [9, 10],
            "rcut_smth": 5.80,
            "rcut": 6.00,
            "neuron": [6, 12, 24],
            "precision": precision,
            "seed": 1145141919810,
        }

    dp_class = DescrptYourNameDP
    pt_class = YourNamePT
    pt_expt_class = YourNamePTExpt
    args = descrpt_your_name_args()
    ntypes = 2

    @property
    def skip_pt(self) -> bool:
        return not INSTALLED_PT

    @property
    def skip_pt_expt(self) -> bool:
        return not INSTALLED_PT_EXPT
```

Reference: `source/tests/consistent/descriptor/test_se_t.py`
