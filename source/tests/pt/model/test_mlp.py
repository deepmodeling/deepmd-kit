# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.utils import EmbeddingNet as DPEmbeddingNet
from deepmd.dpmodel.utils import FittingNet as DPFittingNet
from deepmd.dpmodel.utils import (
    NativeLayer,
    NativeNet,
)
from deepmd.pt.model.network.mlp import (
    MLP,
    EmbeddingNet,
    FittingNet,
    MLPLayer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)


def get_tols(prec):
    if prec in ["single", "float32"]:
        rtol, atol = 0.0, 1e-4
    elif prec in ["double", "float64"]:
        rtol, atol = 0.0, 1e-12
    # elif prec in ["half", "float16"]:
    #   rtol, atol=1e-2, 0
    else:
        raise ValueError(f"unknown prec {prec}")
    return rtol, atol


class TestMLPLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.test_cases = itertools.product(
            [(5, 5), (5, 10), (5, 8), (8, 5)],  # inp, out
            [True, False],  # bias
            [True, False],  # use time step
            ["tanh", "none"],  # activation
            [True, False],  # resnet
            [None, [4], [3, 2]],  # prefix shapes
            ["float32", "double"],  # precision
        )

    def test_match_native_layer(
        self,
    ) -> None:
        for (ninp, nout), bias, ut, ac, resnet, ashp, prec in self.test_cases:
            # input
            inp_shap = [ninp]
            if ashp is not None:
                inp_shap = ashp + inp_shap
            rtol, atol = get_tols(prec)
            dtype = PRECISION_DICT[prec]
            xx = torch.arange(np.prod(inp_shap), dtype=dtype, device=env.DEVICE).view(
                inp_shap
            )
            # def mlp layer
            ml = MLPLayer(ninp, nout, bias, ut, ac, resnet, precision=prec).to(
                env.DEVICE
            )
            # check consistency
            nl = NativeLayer.deserialize(ml.serialize())
            np.testing.assert_allclose(
                ml.forward(xx).detach().cpu().numpy(),
                nl.call(xx.detach().cpu().numpy()),
                rtol=rtol,
                atol=atol,
                err_msg=f"(i={ninp}, o={nout}) bias={bias} use_dt={ut} act={ac} resnet={resnet} prec={prec}",
            )
            # check self-consistency
            ml1 = MLPLayer.deserialize(ml.serialize()).to(env.DEVICE)
            np.testing.assert_allclose(
                ml.forward(xx).detach().cpu().numpy(),
                ml1.forward(xx).detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"(i={ninp}, o={nout}) bias={bias} use_dt={ut} act={ac} resnet={resnet} prec={prec}",
            )

    def test_jit(self) -> None:
        for (ninp, nout), bias, ut, ac, resnet, _, prec in self.test_cases:
            ml = MLPLayer(ninp, nout, bias, ut, ac, resnet, precision=prec)
            model = torch.jit.script(ml)
            ml1 = MLPLayer.deserialize(ml.serialize())
            model = torch.jit.script(ml1)


class TestMLP(unittest.TestCase):
    def setUp(self) -> None:
        self.test_cases = itertools.product(
            [[2, 2, 4, 8], [1, 3, 3]],  # inp and hiddens
            [True, False],  # bias
            [True, False],  # use time step
            ["tanh", "none"],  # activation
            [True, False],  # resnet
            [None, [4], [3, 2]],  # prefix shapes
            ["float32", "double"],  # precision
        )

    def test_match_native_net(
        self,
    ) -> None:
        for ndims, bias, ut, ac, resnet, ashp, prec in self.test_cases:
            # input
            inp_shap = [ndims[0]]
            if ashp is not None:
                inp_shap = ashp + inp_shap
            rtol, atol = get_tols(prec)
            dtype = PRECISION_DICT[prec]
            xx = torch.arange(np.prod(inp_shap), dtype=dtype, device=env.DEVICE).view(
                inp_shap
            )
            # def MLP
            layers = []
            for ii in range(1, len(ndims)):
                layers.append(
                    MLPLayer(
                        ndims[ii - 1], ndims[ii], bias, ut, ac, resnet, precision=prec
                    ).serialize()
                )
            ml = MLP(layers).to(env.DEVICE)
            # check consistency
            nl = NativeNet.deserialize(ml.serialize())
            np.testing.assert_allclose(
                ml.forward(xx).detach().cpu().numpy(),
                nl.call(xx.detach().cpu().numpy()),
                rtol=rtol,
                atol=atol,
                err_msg=f"net={ndims} bias={bias} use_dt={ut} act={ac} resnet={resnet} prec={prec}",
            )
            # check self-consistency
            ml1 = MLP.deserialize(ml.serialize()).to(env.DEVICE)
            np.testing.assert_allclose(
                ml.forward(xx).detach().cpu().numpy(),
                ml1.forward(xx).detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"net={ndims} bias={bias} use_dt={ut} act={ac} resnet={resnet} prec={prec}",
            )

    def test_jit(self) -> None:
        for ndims, bias, ut, ac, resnet, _, prec in self.test_cases:
            layers = []
            for ii in range(1, len(ndims)):
                ml = layers.append(
                    MLPLayer(
                        ndims[ii - 1], ndims[ii], bias, ut, ac, resnet, precision=prec
                    ).serialize()
                )
            ml = MLP(ml)
            model = torch.jit.script(ml)
            ml1 = MLP.deserialize(ml.serialize())
            model = torch.jit.script(ml1)


class TestEmbeddingNet(unittest.TestCase):
    def setUp(self) -> None:
        self.test_cases = itertools.product(
            [1, 3],  # inp
            [[24, 48, 96], [24, 36]],  # and hiddens
            ["tanh", "none"],  # activation
            [True, False],  # resnet_dt
            ["float32", "double"],  # precision
        )

    def test_match_embedding_net(
        self,
    ) -> None:
        for idim, nn, act, idt, prec in self.test_cases:
            # input
            rtol, atol = get_tols(prec)
            dtype = PRECISION_DICT[prec]
            xx = torch.arange(idim, dtype=dtype, device=env.DEVICE)
            # def MLP
            ml = EmbeddingNet(idim, nn, act, idt, prec).to(env.DEVICE)
            # check consistency
            nl = DPEmbeddingNet.deserialize(ml.serialize())
            np.testing.assert_allclose(
                ml.forward(xx).detach().cpu().numpy(),
                nl.call(xx.detach().cpu().numpy()),
                rtol=rtol,
                atol=atol,
                err_msg=f"idim={idim} nn={nn} use_dt={idt} act={act} prec={prec}",
            )
            # check self-consistency
            ml1 = EmbeddingNet.deserialize(ml.serialize()).to(env.DEVICE)
            np.testing.assert_allclose(
                ml.forward(xx).detach().cpu().numpy(),
                ml1.forward(xx).detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"idim={idim} nn={nn} use_dt={idt} act={act} prec={prec}",
            )

    def test_jit(
        self,
    ) -> None:
        for idim, nn, act, idt, prec in self.test_cases:
            # def MLP
            ml = EmbeddingNet(idim, nn, act, idt, prec).to(env.DEVICE)
            ml1 = EmbeddingNet.deserialize(ml.serialize()).to(env.DEVICE)
            model = torch.jit.script(ml)
            model = torch.jit.script(ml1)


class TestFittingNet(unittest.TestCase):
    def setUp(self) -> None:
        self.test_cases = itertools.product(
            [1, 3],  # inp
            [1, 5],  # out
            [[24, 48, 96], [24, 36]],  # and hiddens
            ["tanh", "none"],  # activation
            [True, False],  # resnet_dt
            ["float32", "double"],  # precision
            [True, False],  # bias_out
        )

    def test_match_fitting_net(
        self,
    ) -> None:
        for idim, odim, nn, act, idt, prec, ob in self.test_cases:
            # input
            rtol, atol = get_tols(prec)
            dtype = PRECISION_DICT[prec]
            xx = torch.arange(idim, dtype=dtype, device=env.DEVICE)
            # def MLP
            ml = FittingNet(
                idim,
                odim,
                neuron=nn,
                activation_function=act,
                resnet_dt=idt,
                precision=prec,
                bias_out=ob,
            ).to(env.DEVICE)
            # check consistency
            nl = DPFittingNet.deserialize(ml.serialize())
            np.testing.assert_allclose(
                ml.forward(xx).detach().cpu().numpy(),
                nl.call(xx.detach().cpu().numpy()),
                rtol=rtol,
                atol=atol,
                err_msg=f"idim={idim} nn={nn} use_dt={idt} act={act} prec={prec}",
            )
            # check self-consistency
            ml1 = FittingNet.deserialize(ml.serialize()).to(env.DEVICE)
            np.testing.assert_allclose(
                ml.forward(xx).detach().cpu().numpy(),
                ml1.forward(xx).detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"idim={idim} nn={nn} use_dt={idt} act={act} prec={prec}",
            )

    def test_jit(
        self,
    ) -> None:
        for idim, odim, nn, act, idt, prec, ob in self.test_cases:
            # def MLP
            ml = FittingNet(
                idim,
                odim,
                neuron=nn,
                activation_function=act,
                resnet_dt=idt,
                precision=prec,
                bias_out=ob,
            ).to(env.DEVICE)
            ml1 = FittingNet.deserialize(ml.serialize()).to(env.DEVICE)
            model = torch.jit.script(ml)
            model = torch.jit.script(ml1)
