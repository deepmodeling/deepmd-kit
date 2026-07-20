# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for TF2 DPA4 descriptor trainable and trackable state."""

import os

import numpy as np
import pytest

if os.environ.get("DP_TEST_TF2_ONLY") != "1":
    pytest.skip(
        "TF2 tests require DP_TEST_TF2_ONLY=1",
        allow_module_level=True,
    )

from deepmd.tf2.common import (
    to_tf_tensor,
    wrap_tensor,
)
from deepmd.tf2.descriptor.dpa4 import (
    DescrptDPA4,
    DynamicRadialDegreeMixer,
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


def _make_frozen_descriptor(seed: int) -> DescrptDPA4:
    """Build a frozen descriptor whose complete state must remain trackable."""
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
        random_gamma=False,
        precision="float64",
        trainable=False,
        seed=seed,
    )


def test_frozen_descriptor_tracks_and_restores_every_parameter(tmp_path) -> None:
    """Frozen leaves are non-trainable variables included in checkpoints."""
    source = _make_frozen_descriptor(20260712)
    target = _make_frozen_descriptor(20260713)
    source_embedding = object.__getattribute__(
        source.type_embedding, "_tf2_adam_type_embedding_variable"
    )
    target_embedding = object.__getattribute__(
        target.type_embedding, "_tf2_adam_type_embedding_variable"
    )
    assert not np.array_equal(source_embedding.numpy(), target_embedding.numpy())
    assert source.variables
    assert not source.trainable_variables

    checkpoint_path = tf.train.Checkpoint(descriptor=source).save(
        str(tmp_path / "descriptor")
    )
    tf.train.Checkpoint(descriptor=target).restore(checkpoint_path).assert_consumed()

    np.testing.assert_array_equal(target_embedding.numpy(), source_embedding.numpy())


def test_promoted_parameters_release_public_tensor_shadows() -> None:
    """Variable-backed attributes must not retain their original eager tensors."""
    descriptor = _make_trainable_descriptor()
    for module in _iter_object_tree(descriptor):
        raw_attrs = object.__getattribute__(module, "__dict__")
        for name in getattr(module, "_tf2_array_variable_attrs", ()):
            assert name not in raw_attrs
        for name in getattr(module, "_tf2_array_variable_list_attrs", ()):
            assert name not in raw_attrs


def test_random_gamma_fails_fast_until_graph_safe_rng_is_supported() -> None:
    """TF2 must not silently disable the default random-roll augmentation."""
    with pytest.raises(NotImplementedError, match="random_gamma"):
        DescrptDPA4(
            ntypes=2,
            sel=4,
            rcut=4.0,
            channels=4,
            n_radial=4,
            lmax=1,
            mmax=1,
            n_blocks=1,
            precision="float64",
            random_gamma=True,
            seed=20260712,
        )


def test_dynamic_radial_mixer_accepts_unknown_rank_tensor_specs() -> None:
    """Runtime rank and shape checks support fully unknown TensorSpecs."""
    mixer = DynamicRadialDegreeMixer(
        lmax=1,
        mmax=1,
        channels=4,
        mode="degree_channel",
        rank=0,
        precision="float64",
        seed=20260712,
        trainable=True,
    )

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
        )
    )
    def apply_mixer(x_local: tf.Tensor, radial_feat: tf.Tensor) -> tf.Tensor:
        return to_tf_tensor(mixer(wrap_tensor(x_local), wrap_tensor(radial_feat)))

    for nedge in (2, 3):
        inputs = tf.ones((nedge, mixer.reduced_dim, mixer.channels), tf.float64)
        output = apply_mixer(inputs, inputs)
        assert tuple(output.shape) == (nedge, mixer.reduced_dim, mixer.channels)
