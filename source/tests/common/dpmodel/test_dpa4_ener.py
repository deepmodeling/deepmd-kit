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
    def test_case_film_without_case_dimension_is_inactive(self) -> None:
        fitting = make_fitting(case_film_embd=True)
        assert not fitting.case_film_embd

    def test_negative_neuron_raises(self) -> None:
        with pytest.raises(ValueError, match="neuron"):
            make_fitting(neuron=[-1])

    @pytest.mark.parametrize(
        "key",
        [
            (0, 0),  # wrong length (ndim=1)
            (-1,),  # negative type index
            (2,),  # type index >= ntypes
            5,  # int index out of bounds
        ],
    )  # invalid network-collection keys
    def test_network_collection_invalid_key(self, key) -> None:
        fitting = make_fitting()
        # __getitem__ raises KeyError per the lookup-method convention
        with pytest.raises(KeyError):
            fitting.nets[key]


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
    @pytest.mark.parametrize("trainable", [False, True, [True, False], [False, True]])
    def test_glu_trainability_roundtrip(self, trainable: bool | list[bool]) -> None:
        """The serialized GLU policy controls every affine map."""
        net = GLUFittingNet(2, 1, [3], trainable=trainable, seed=3)
        expected = all(trainable) if isinstance(trainable, list) else trainable
        record = net.serialize()
        assert record["trainable"] is expected
        restored = GLUFittingNet.deserialize(record)
        assert all(
            layer.trainable is expected
            for layer in [*restored.hidden_layers, restored.output_layer]
        )

    def test_standalone_legacy_glu_defaults_to_trainable(self) -> None:
        """A standalone legacy record has no outer freeze policy to inherit."""
        record = GLUFittingNet(2, 1, [3], trainable=False, seed=3).serialize()
        record.pop("trainable")
        restored = GLUFittingNet.deserialize(record)
        assert restored.trainable

    @pytest.mark.parametrize("mixed_types", [False, True])
    def test_legacy_collection_inherits_freezing_without_mutating_record(
        self,
        mixed_types: bool,
    ) -> None:
        fitting = make_fitting(trainable=[True, False], mixed_types=mixed_types)
        record = fitting.serialize()
        for network in record["nets"]["networks"]:
            network.pop("trainable")
        restored = SeZMEnergyFittingNet.deserialize(record)
        assert all(not network.trainable for network in restored.nets.networks)
        assert all("trainable" not in network for network in record["nets"]["networks"])

    @pytest.mark.parametrize("film", [False, True])
    def test_case_embedding_roundtrip(self, film: bool) -> None:
        fitting = make_fitting(dim_case_embd=2, case_film_embd=film)
        fitting.set_case_embd(1)
        descriptor, atype = make_inputs()
        if film:
            # The projectors start at zero, where the modulation is the
            # identity; random projectors make the round trip carry them.
            plain = fitting(descriptor, atype)["energy"]
            random = np.random.default_rng(19)
            for weight in fitting.nets[()].case_film.projectors:
                weight[:] = random.normal(size=weight.shape)
            assert np.abs(fitting(descriptor, atype)["energy"] - plain).max() > 1e-6
        restored = SeZMEnergyFittingNet.deserialize(fitting.serialize())
        np.testing.assert_array_equal(
            fitting(descriptor, atype)["energy"],
            restored(descriptor, atype)["energy"],
        )

    @pytest.mark.parametrize("film", [False, True])
    @pytest.mark.parametrize("neuron", [[], [8], [8, 4]])
    def test_fold_case_embedding_preserves_outputs(
        self, film: bool, neuron: list[int]
    ) -> None:
        fitting = make_fitting(dim_case_embd=2, case_film_embd=film, neuron=neuron)
        fitting.set_case_embd(1)
        if film:
            random = np.random.default_rng(17)
            for weight in fitting.nets[()].case_film.projectors:
                weight[:] = random.normal(size=weight.shape)
        descriptor, atype = make_inputs()
        expected = fitting(descriptor, atype)["energy"]
        fitting.fold_case_embedding()
        assert fitting.dim_case_embd == 0
        restored = SeZMEnergyFittingNet.deserialize(fitting.serialize())
        np.testing.assert_allclose(
            restored(descriptor, atype)["energy"], expected, rtol=1e-12, atol=1e-12
        )

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
