# SPDX-License-Identifier: LGPL-3.0-or-later
"""GroupPropertyFittingNet used to accept and silently ignore several
property-schema options (seed, numb_aparam, resnet_dt, ...) via a bare
``**kwargs``, and collapsed a per-layer ``trainable`` list to
``all(trainable)``. Config authors setting any of those got a model that
looked accepted but behaved differently than requested. Cover: an
unsupported option now raises instead of being swallowed, ``seed`` actually
makes initialization reproducible, and a per-layer ``trainable`` list
freezes exactly those layers.
"""

from __future__ import (
    annotations,
)

import pytest
import torch

pytest.importorskip("deepmd.lib")

from deepmd.pt.model.task.group_property import (
    GroupPropertyFittingNet,
)


def _make(**overrides):
    kwargs = {"ntypes": 3, "dim_descrpt": 4, "property_name": "y", "neuron": [8]}
    kwargs.update(overrides)
    return GroupPropertyFittingNet(**kwargs)


def test_injected_type_and_mixed_types_are_accepted():
    # the generic model-building path always passes these two; they must not
    # crash construction even though the fitting net itself doesn't use them.
    _make(type="group_property", mixed_types=True)


def test_unsupported_property_options_are_rejected_not_ignored():
    for bad_kwarg in (
        "numb_aparam",
        "default_fparam",
        "dim_case_embd",
        "resnet_dt",
        "intensive",
        "distinguish_types",
        "some_future_typo",
    ):
        with pytest.raises(TypeError):
            _make(**{bad_kwarg: True})


def test_seed_makes_initialization_reproducible():
    fn1 = _make(seed=42)
    fn2 = _make(seed=42)
    w1 = fn1.network[0].weight.detach()
    w2 = fn2.network[0].weight.detach()
    assert torch.equal(w1, w2)


def test_different_seeds_give_different_initialization():
    fn1 = _make(seed=42)
    fn2 = _make(seed=43)
    assert not torch.equal(
        fn1.network[0].weight.detach(), fn2.network[0].weight.detach()
    )


def test_unseeded_construction_advances_the_global_rng_stream():
    """Without an explicit seed, initialization must still draw from (and
    advance) the caller's global RNG stream, same as an unseeded
    torch.nn.Linear always has -- two back-to-back unseeded constructions
    should not silently produce identical weights.
    """
    torch.manual_seed(0)
    fn_a = _make()
    fn_b = _make()
    assert not torch.equal(
        fn_a.network[0].weight.detach(), fn_b.network[0].weight.detach()
    )


def test_trainable_list_freezes_only_the_named_layers():
    # neuron=[8] -> 2 Linear layers (hidden, output); freeze only the first.
    fn = _make(trainable=[False, True])
    linear_layers = [m for m in fn.network if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 2
    assert not any(p.requires_grad for p in linear_layers[0].parameters())
    assert all(p.requires_grad for p in linear_layers[1].parameters())


def test_trainable_list_wrong_length_raises():
    with pytest.raises(ValueError, match="trainable"):
        _make(trainable=[True])


def test_trainable_bool_still_applies_uniformly():
    fn = _make(trainable=False)
    assert not any(p.requires_grad for p in fn.parameters())
