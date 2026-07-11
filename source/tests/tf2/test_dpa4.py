# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for TF2 DPA4 trainable and trackable state."""

import os

import numpy as np
import pytest

if os.environ.get("DP_TEST_TF2_ONLY") != "1":
    pytest.skip(
        "TF2 tests require DP_TEST_TF2_ONLY=1",
        allow_module_level=True,
    )

from deepmd.tf2.descriptor.dpa4 import (
    DescrptDPA4,
    _iter_object_tree,
)
from deepmd.tf2.env import (
    tf,
)


def _make_trainable_descriptor() -> DescrptDPA4:
    """Build a small descriptor that enables the optional trainable leaves."""
    return DescrptDPA4(
        ntypes=2,
        sel=4,
        rcut=4.0,
        channels=4,
        n_radial=4,
        lmax=1,
        mmax=1,
        n_blocks=1,
        grid_branch=0,
        layer_scale=True,
        message_node_so3=True,
        random_gamma=False,
        precision="float64",
        trainable=True,
        seed=20260711,
    )


def _assert_optional_weights_are_tracked(descriptor: DescrptDPA4) -> None:
    """Assert optional DPA4 variables are trainable TensorFlow trackables."""
    modules = list(_iter_object_tree(descriptor))
    tracked_ids = {id(variable) for variable in descriptor.trainable_variables}

    frame_modules = [
        module
        for module in modules
        if type(module).__name__ in {"FrameContract", "FrameExpand"}
    ]
    assert {type(module).__name__ for module in frame_modules} == {
        "FrameContract",
        "FrameExpand",
    }
    for module in frame_modules:
        variable = object.__getattribute__(module, "_tf2_weight_variable")
        assert isinstance(variable, tf.Variable)
        assert variable.trainable
        assert id(variable) in tracked_ids

    interaction_blocks = [
        module for module in modules if type(module).__name__ == "SeZMInteractionBlock"
    ]
    assert interaction_blocks
    for block in interaction_blocks:
        variables = object.__getattribute__(
            block,
            "_tf2_adam_ffn_layer_scales_variables",
        )
        assert variables
        assert all(variable.trainable for variable in variables)
        assert all(id(variable) in tracked_ids for variable in variables)


def test_optional_dpa4_weights_are_tf2_trainable_variables() -> None:
    """Optional cross-grid and FFN LayerScale weights must receive gradients."""
    _assert_optional_weights_are_tracked(_make_trainable_descriptor())


def test_dpa4_deserialize_refreshes_trackable_state() -> None:
    """Serialization must preserve values and nested TensorFlow trackables."""
    descriptor = _make_trainable_descriptor()
    serialized = descriptor.serialize()

    restored = DescrptDPA4.deserialize(serialized)

    np.testing.assert_equal(restored.serialize(), serialized)
    _assert_optional_weights_are_tracked(restored)
