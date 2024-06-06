# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestInvarFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ):
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        rd0, _, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
        )
        atype = torch.tensor(self.atype_ext[:, :nloc], dtype=int, device=env.DEVICE)

        for od, mixed_types, nfp, nap, et, nn in itertools.product(
            [1, 3],
            [True, False],
            [0, 3],
            [0, 4],
            [[], [0], [1]],
            [[4, 4, 4], []],
        ):
            ft0 = InvarFitting(
                "foo",
                self.nt,
                dd0.dim_out,
                od,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
                exclude_types=et,
                neuron=nn,
            ).to(env.DEVICE)
            ft1 = DPInvarFitting.deserialize(ft0.serialize())
            ft2 = InvarFitting.deserialize(ft0.serialize())

            if nfp > 0:
                ifp = torch.tensor(
                    rng.normal(size=(self.nf, nfp)), dtype=dtype, device=env.DEVICE
                )
            else:
                ifp = None
            if nap > 0:
                iap = torch.tensor(
                    rng.normal(size=(self.nf, self.nloc, nap)),
                    dtype=dtype,
                    device=env.DEVICE,
                )
            else:
                iap = None

            ret0 = ft0(rd0, atype, fparam=ifp, aparam=iap)
            ret1 = ft1(
                rd0.detach().cpu().numpy(),
                atype.detach().cpu().numpy(),
                fparam=to_numpy_array(ifp),
                aparam=to_numpy_array(iap),
            )
            ret2 = ft2(rd0, atype, fparam=ifp, aparam=iap)
            np.testing.assert_allclose(
                to_numpy_array(ret0["foo"]),
                ret1["foo"],
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["foo"]),
                to_numpy_array(ret2["foo"]),
            )
            self.assertEqual(ft0.get_sel_type(), ft1.get_sel_type())

    def test_new_old(
        self,
    ):
        nf, nloc, nnei = self.nlist.shape
        dd = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        rd0, _, _, _, _ = dd(
            torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
        )
        atype = torch.tensor(self.atype_ext[:, :nloc], dtype=int, device=env.DEVICE)

        od = 1
        for foo, mixed_types in itertools.product(
            [True],
            [True, False],
        ):
            ft0 = EnergyFittingNet(
                self.nt,
                dd.dim_out,
                mixed_types=mixed_types,
            ).to(env.DEVICE)
            ft1 = EnergyFittingNet(
                self.nt,
                dd.dim_out,
                mixed_types=mixed_types,
                old_impl=True,
            ).to(env.DEVICE)
            dd0 = ft0.state_dict()
            dd1 = ft1.state_dict()
            for kk, vv in dd1.items():
                new_kk = kk
                new_kk = new_kk.replace("filter_layers_old", "filter_layers.networks")
                new_kk = new_kk.replace("deep_layers", "layers")
                new_kk = new_kk.replace("final_layer", "layers.3")
                dd1[kk] = dd0[new_kk]
                if kk.split(".")[-1] in ["idt", "bias"]:
                    dd1[kk] = dd1[kk].unsqueeze(0)
            dd1["bias_atom_e"] = dd0["bias_atom_e"]
            ft1.load_state_dict(dd1)
            ret0 = ft0(rd0, atype)
            ret1 = ft1(rd0, atype)
            np.testing.assert_allclose(
                to_numpy_array(ret0["energy"]),
                to_numpy_array(ret1["energy"]),
            )

    def test_jit(
        self,
    ):
        for od, mixed_types, nfp, nap, et in itertools.product(
            [1, 3],
            [True, False],
            [0, 3],
            [0, 4],
            [[], [0]],
        ):
            ft0 = InvarFitting(
                "foo",
                self.nt,
                9,
                od,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
                exclude_types=et,
            ).to(env.DEVICE)
            torch.jit.script(ft0)

    def test_get_set(self):
        ifn0 = InvarFitting(
            "energy",
            self.nt,
            3,
            1,
        )
        rng = np.random.default_rng(GLOBAL_SEED)
        foo = rng.normal([3, 4])
        for ii in [
            "bias_atom_e",
            "fparam_avg",
            "fparam_inv_std",
            "aparam_avg",
            "aparam_inv_std",
        ]:
            ifn0[ii] = torch.tensor(foo, dtype=dtype, device=env.DEVICE)
            np.testing.assert_allclose(
                foo, np.reshape(ifn0[ii].detach().cpu().numpy(), foo.shape)
            )
