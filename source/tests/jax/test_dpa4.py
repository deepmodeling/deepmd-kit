# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for JAX DPA4 trainable-state conversion."""

from deepmd.jax.descriptor.dpa4 import (
    DescrptDPA4,
    _iter_object_tree,
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
