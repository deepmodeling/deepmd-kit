# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import unittest
from unittest import (
    mock,
)

import numpy as np
import torch

from deepmd.dpmodel.utils import EmbeddingNet as DPEmbeddingNet
from deepmd.dpmodel.utils import FittingNet as DPFittingNet
from deepmd.dpmodel.utils import (
    NativeLayer,
    NativeNet,
)
from deepmd.pt.model.network import mlp as mlp_module
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

from ...common.test_mixins import (
    get_tols,
)


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

    def test_thin_so2_eval_linear_traces_without_addmm(self) -> None:
        from torch.fx.experimental.proxy_tensor import (
            make_fx,
        )

        layer = MLPLayer(
            5,
            8,
            bias=True,
            activation_function="none",
            precision="float32",
            trainable=False,
        ).to(env.DEVICE)
        layer.eval()
        mlp_module.enable_neo_cute_compile_visible_linears(layer)
        value = torch.randn(7, 5, device=env.DEVICE, dtype=torch.float32)
        environment = {
            "DP_CUTE_INFER": "1",
            "DP_CUTE_SO2_THIN_WRAPPER": "1",
        }

        with mock.patch.dict(os.environ, environment, clear=False):
            graph = make_fx(layer)(value)
            with mock.patch.object(
                mlp_module.F,
                "linear",
                side_effect=AssertionError("thin eval path reached aten::linear"),
            ):
                actual = layer(value)

        expected = mlp_module.F.linear(value, layer.matrix.t(), layer.bias)
        torch.testing.assert_close(actual, expected)
        targets = {
            node.target for node in graph.graph.nodes if node.op == "call_function"
        }
        self.assertIn(torch.ops.aten.mm.default, targets)
        self.assertNotIn(torch.ops.aten.addmm.default, targets)

    def test_thin_so2_eval_linear_is_scoped_to_marked_model(self) -> None:
        layer = MLPLayer(
            5,
            8,
            bias=True,
            activation_function="none",
            precision="float32",
            trainable=False,
        ).to(env.DEVICE)
        layer.eval()
        value = torch.randn(7, 5, device=env.DEVICE, dtype=torch.float32)
        environment = {
            "DP_CUTE_INFER": "1",
            "DP_CUTE_SO2_THIN_WRAPPER": "1",
        }

        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(
                mlp_module,
                "_matmul_bias",
                side_effect=AssertionError("unmarked MLP reached Neo-only topology"),
            ),
        ):
            actual = layer(value)

        expected = mlp_module.F.linear(value, layer.matrix.t(), layer.bias)
        torch.testing.assert_close(actual, expected)

    def test_thin_so2_linear_defaults_on_only_for_sm80(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"DP_CUTE_INFER": "1"},
                clear=True,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            self.assertTrue(mlp_module._use_so2_compile_visible_linear())

        with (
            mock.patch.dict(
                os.environ,
                {
                    "DP_CUTE_INFER": "1",
                    "DP_CUTE_SO2_THIN_WRAPPER": "0",
                },
                clear=True,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            self.assertFalse(mlp_module._use_so2_compile_visible_linear())


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
