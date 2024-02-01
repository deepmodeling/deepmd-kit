# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

try:
    from deepmd.model_format import DescrptDPA1 as DPDescrptDPA1

    support_se_atten = True
except ModuleNotFoundError:
    support_se_atten = False
except ImportError:
    support_se_atten = False

from deepmd.pt.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


@unittest.skipIf(not support_se_atten, "EnvMat not supported")
class TestDescrptSeAtten(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ):
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec in itertools.product(
            [False, True],
            ["float64", "float32"],
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} prec={prec}"
            # dpa1 new impl
            dd0 = DescrptDPA1(
                self.rcut,
                self.rcut_smth,
                self.sel,
                self.nt,
                attn_layer=0,  # TODO add support for non-zero layer
                # precision=prec,
                # resnet_dt=idt,
                old_impl=False,
            ).to(env.DEVICE)
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.se_atten.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptDPA1.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
            # dp impl
            dd2 = DPDescrptDPA1.deserialize(dd0.serialize())
            rd2, _, _, _, _ = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd2,
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
            # dp impl serialization
            dd3 = DPDescrptDPA1.deserialize(dd2.serialize())
            rd3, _, _, _, _ = dd3.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd3,
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
            # old impl
            if idt is False and prec == "float64":
                dd4 = DescrptDPA1(
                    self.rcut,
                    self.rcut_smth,
                    self.sel,
                    self.nt,
                    attn_layer=0,  # TODO add support for non-zero layer
                    # precision=prec,
                    # resnet_dt=idt,
                    old_impl=True,
                ).to(env.DEVICE)
                dd0_state_dict = dd0.se_atten.state_dict()
                dd4_state_dict = dd4.se_atten.state_dict()

                dd0_state_dict_attn = dd0.se_atten.dpa1_attention.state_dict()
                dd4_state_dict_attn = dd4.se_atten.dpa1_attention.state_dict()
                for i in dd4_state_dict:
                    dd4_state_dict[i] = (
                        dd0_state_dict[
                            i.replace(".deep_layers.", ".layers.")
                            .replace("filter_layers_old.", "filter_layers._networks.")
                            .replace(
                                ".attn_layer_norm.weight", ".attn_layer_norm.matrix"
                            )
                        ]
                        .detach()
                        .clone()
                    )
                    if ".bias" in i and "attn_layer_norm" not in i:
                        dd4_state_dict[i] = dd4_state_dict[i].unsqueeze(0)
                dd4.se_atten.load_state_dict(dd4_state_dict)

                dd0_state_dict_tebd = dd0.type_embedding.state_dict()
                dd4_state_dict_tebd = dd4.type_embedding_old.state_dict()
                for i in dd4_state_dict_tebd:
                    dd4_state_dict_tebd[i] = (
                        dd0_state_dict_tebd[i.replace("embedding.weight", "matrix")]
                        .detach()
                        .clone()
                    )
                dd4.type_embedding_old.load_state_dict(dd4_state_dict_tebd)

                rd4, _, _, _, _ = dd4(
                    torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                    torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                    torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                )
                np.testing.assert_allclose(
                    rd0.detach().cpu().numpy(),
                    rd4.detach().cpu().numpy(),
                    rtol=rtol,
                    atol=atol,
                    err_msg=err_msg,
                )

    def test_jit(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec in itertools.product(
            [False, True],
            ["float64", "float32"],
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} prec={prec}"
            # sea new impl
            dd0 = DescrptDPA1(
                self.rcut,
                self.rcut_smth,
                self.sel,
                self.nt,
                # precision=prec,
                # resnet_dt=idt,
                old_impl=False,
            )
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.se_atten.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            # dd1 = DescrptDPA1.deserialize(dd0.serialize())
            model = torch.jit.script(dd0)
            # model = torch.jit.script(dd1)
