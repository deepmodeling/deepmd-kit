# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for JAX DPA4 descriptor trainable-state conversion."""

from deepmd.jax.descriptor.dpa4 import (
    DescrptDPA4,
    _iter_object_tree,
)
from deepmd.jax.env import (
    nnx,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
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


def test_optional_dpa4_weights_are_jax_parameters() -> None:
    """Optional cross-grid and FFN LayerScale weights must receive gradients."""
    descriptor = _make_trainable_descriptor()
    modules = list(_iter_object_tree(descriptor))

    frame_modules = [
        module
        for module in modules
        if type(module).__name__ in {"FrameContract", "FrameExpand"}
    ]
    assert {type(module).__name__ for module in frame_modules} == {
        "FrameContract",
        "FrameExpand",
    }
    assert all(isinstance(module.weight, ArrayAPIParam) for module in frame_modules)

    interaction_blocks = [
        module for module in modules if type(module).__name__ == "SeZMInteractionBlock"
    ]
    assert interaction_blocks
    assert all(
        isinstance(scale, ArrayAPIParam)
        for block in interaction_blocks
        for scale in block.adam_ffn_layer_scales
    )


def test_frozen_descriptor_has_no_optimizer_visible_parameters() -> None:
    """The root freeze flag must demote every descendant parameter."""
    descriptor = DescrptDPA4(
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
        seed=20260712,
    )

    assert len(nnx.to_flat_state(nnx.state(descriptor, nnx.Param))) == 0
