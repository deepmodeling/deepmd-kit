# SPDX-License-Identifier: LGPL-3.0-or-later
"""Torch-free unit tests for the dpmodel DPA4 (SeZM) GLU energy fitting."""

import subprocess
import sys

import numpy as np
import pytest

from deepmd.dpmodel.fitting.dpa4_ener import (
    GLUFittingNet,
    SeZMEnergyFittingNet,
    _resolve_auto_neuron,
)


def make_fitting(**overrides) -> SeZMEnergyFittingNet:
    kwargs = {
        "ntypes": 2,
        "dim_descrpt": 12,
        "neuron": [16],
        "precision": "float64",
        "seed": 7,
    }
    kwargs.update(overrides)
    return SeZMEnergyFittingNet(**kwargs)


def make_inputs(nf=2, nloc=5, dim=12, ntypes=2, seed=0):
    rng = np.random.default_rng(seed)
    descriptor = rng.normal(size=(nf, nloc, dim))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    atype[0, 0], atype[0, 1] = 0, 1
    return descriptor, atype


class TestGuards:
    def test_dim_case_embd_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="dim_case_embd"):
            make_fitting(dim_case_embd=2)

    def test_case_film_embd_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="case_film_embd"):
            make_fitting(case_film_embd=True)

    def test_glu_net_case_film_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="case_film_embd"):
            GLUFittingNet(8, 1, [16], dim_case_embd=2, case_film_embd=True)

    def test_negative_neuron_raises(self) -> None:
        with pytest.raises(ValueError, match="neuron"):
            make_fitting(neuron=[-1])


class TestAutoNeuron:
    def test_auto_width_marker(self) -> None:
        # dim_in=12 -> 32*ceil(8*12/3/32) = 32
        assert _resolve_auto_neuron(
            [0],
            dim_descrpt=12,
            numb_fparam=0,
            numb_aparam=0,
            dim_case_embd=0,
            case_film_embd=False,
            use_aparam_as_mask=False,
        ) == [32]

    def test_no_marker_passthrough(self) -> None:
        assert _resolve_auto_neuron(
            [16, 16],
            dim_descrpt=12,
            numb_fparam=0,
            numb_aparam=0,
            dim_case_embd=0,
            case_film_embd=False,
            use_aparam_as_mask=False,
        ) == [16, 16]

    def test_none_means_single_auto(self) -> None:
        fit = make_fitting(neuron=None)
        assert fit.neuron == [32]


class TestRoundtrip:
    @pytest.mark.parametrize("bias_out", [False, True])  # output-layer bias
    @pytest.mark.parametrize("resnet_dt", [False, True])  # serialized flag only
    @pytest.mark.parametrize(
        "neuron", [[], [16], [16, 16]]
    )  # direct linear / shallow / deep
    def test_serialize_roundtrip_exact(self, neuron, bias_out, resnet_dt) -> None:
        fit = make_fitting(neuron=neuron, bias_out=bias_out, resnet_dt=resnet_dt)
        data = fit.serialize()
        assert data["type"] == "sezm_ener"
        assert data["@class"] == "Fitting"
        fit2 = SeZMEnergyFittingNet.deserialize(data)
        assert set(fit2.serialize().keys()) == set(data.keys())
        descriptor, atype = make_inputs()
        out1 = fit.call(descriptor, atype)["energy"]
        out2 = fit2.call(descriptor, atype)["energy"]
        np.testing.assert_array_equal(out1, out2)

    def test_glu_net_roundtrip_exact(self) -> None:
        net = GLUFittingNet(8, 1, [16], precision="float64", bias_out=True, seed=3)
        net2 = GLUFittingNet.deserialize(net.serialize())
        x = np.random.default_rng(1).normal(size=(4, 8))
        np.testing.assert_array_equal(net.call(x), net2.call(x))
        # state keys follow the pt state_dict contract
        assert set(net.serialize()["@variables"]) == {
            "hidden_layers.0.linear.matrix",
            "hidden_layers.0.linear.bias",
            "output_layer.matrix",
            "output_layer.bias",
        }

    def test_glu_net_no_bias_out_keys(self) -> None:
        net = GLUFittingNet(8, 1, [16], precision="float64", bias_out=False, seed=3)
        assert "output_layer.bias" not in net.serialize()["@variables"]


class TestForward:
    @pytest.mark.parametrize("mixed_types", [True, False])  # shared vs per-type nets
    def test_output_shape(self, mixed_types) -> None:
        fit = make_fitting(mixed_types=mixed_types)
        descriptor, atype = make_inputs()
        out = fit.call(descriptor, atype)["energy"]
        assert out.shape == (2, 5, 1)

    def test_bias_atom_e_added(self) -> None:
        fit = make_fitting()
        bias = np.array([[1.5], [-2.5]])
        fit["bias_atom_e"] = bias
        descriptor, atype = make_inputs()
        out0 = make_fitting().call(descriptor, atype)["energy"]
        out1 = fit.call(descriptor, atype)["energy"]
        np.testing.assert_allclose(out1 - out0, bias[atype], rtol=1e-12, atol=1e-14)

    def test_exclude_types_zeroed(self) -> None:
        fit = make_fitting(exclude_types=[0])
        descriptor, atype = make_inputs()
        out = fit.call(descriptor, atype)["energy"]
        assert np.all(out[atype == 0] == 0.0)
        assert np.all(out[atype == 1] != 0.0)

    def test_default_fparam_matches_explicit(self) -> None:
        fit = make_fitting(numb_fparam=2, default_fparam=[0.25, -0.75])
        descriptor, atype = make_inputs()
        out_default = fit.call(descriptor, atype)["energy"]
        fparam = np.tile(np.array([[0.25, -0.75]]), (2, 1))
        out_explicit = fit.call(descriptor, atype, fparam=fparam)["energy"]
        np.testing.assert_array_equal(out_default, out_explicit)

    def test_trainable_list_accepted(self) -> None:
        fit = make_fitting(trainable=[True, False])
        descriptor, atype = make_inputs()
        assert fit.call(descriptor, atype)["energy"].shape == (2, 5, 1)


class TestNoTorchImport:
    def test_dpa4_ener_does_not_import_torch(self) -> None:
        code = (
            "import sys; "
            "import deepmd.dpmodel.fitting.dpa4_ener; "
            "print('torch' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"
