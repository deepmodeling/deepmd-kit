# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for virtual types in dpmodel embedding gathers."""

from typing import (
    Any,
)

import array_api_strict
import numpy as np
import pytest

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.descriptor.dpa2 import (
    DescrptDPA2,
    RepformerArgs,
    RepinitArgs,
)
from deepmd.dpmodel.descriptor.dpa3 import (
    DescrptDPA3,
    RepFlowArgs,
)
from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
    SeZMTypeEmbedding,
)
from deepmd.dpmodel.descriptor.se_t_tebd import (
    DescrptSeTTebd,
)
from deepmd.dpmodel.utils.type_embed import (
    take_type_embedding,
)
from source.tests.array_api_strict.common import (
    convert_array_api_strict_value,
)


@pytest.mark.parametrize("namespace_name", ["numpy", "torch"])
def test_padded_type_embedding_maps_virtual_type(namespace_name: str) -> None:
    """Negative types select the explicit final zero row, including on Torch."""
    if namespace_name == "torch":
        import torch

        namespace = torch
        table = namespace.asarray(
            np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float64),
            device="cpu",
        )
        atype = namespace.asarray(np.array([0, -1, 1], dtype=np.int64), device="cpu")
    else:
        namespace = np
        table = namespace.asarray(
            np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
        )
        atype = namespace.asarray(np.array([0, -1, 1], dtype=np.int64))

    actual = take_type_embedding(table, atype)

    np.testing.assert_array_equal(
        to_numpy_array(actual),
        [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
    )


@pytest.mark.parametrize("namespace_name", ["numpy", "torch"])
def test_sezm_padded_embedding_maps_virtual_type(namespace_name: str) -> None:
    """SeZM uses the same padding-row contract at its gather boundary."""
    if namespace_name == "torch":
        import torch

        namespace = torch
        atype = namespace.asarray(np.array([0, -1, 1], dtype=np.int64), device="cpu")
    else:
        namespace = np
        atype = namespace.asarray(np.array([0, -1, 1], dtype=np.int64))
    embedding = SeZMTypeEmbedding(ntypes=2, embed_dim=2, padding=True, seed=1)
    embedding.adam_type_embedding = np.array(
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float64
    )
    actual = embedding(atype)

    np.testing.assert_array_equal(
        to_numpy_array(actual),
        [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
    )


@pytest.mark.parametrize(
    ("tebd_input_mode", "type_one_side"),
    [("concat", True), ("strip", True), ("strip", False)],
)
def test_dpa1_strict_virtual_type_matches_explicit_padding(
    tebd_input_mode: str, type_one_side: bool
) -> None:
    """Direct descriptor calls remap virtual types before all gather modes."""
    descriptor = convert_array_api_strict_value(
        DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[2, 2],
            ntypes=2,
            attn_layer=0,
            axis_neuron=2,
            neuron=[6, 12],
            tebd_input_mode=tebd_input_mode,
            type_one_side=type_one_side,
        )
    )
    coord = array_api_strict.asarray(
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    )
    nlist = array_api_strict.asarray(
        np.array([[[1, 2, -1, -1], [0, 2, -1, -1]]], dtype=np.int64)
    )
    virtual_atype = array_api_strict.asarray(np.array([[0, 1, -1]], dtype=np.int64))
    padding_atype = array_api_strict.asarray(np.array([[0, 1, 2]], dtype=np.int64))

    actual = descriptor._call_dense(coord, virtual_atype, nlist)
    expected = descriptor._call_dense(coord, padding_atype, nlist)

    for actual_value, expected_value in zip(actual, expected, strict=True):
        if actual_value is not None:
            np.testing.assert_allclose(
                to_numpy_array(actual_value), to_numpy_array(expected_value)
            )


def test_se_t_tebd_strip_strict_virtual_type_matches_explicit_padding() -> None:
    """Strip-mode pair indices remap virtual neighbors to the padding type."""
    descriptor = convert_array_api_strict_value(
        DescrptSeTTebd(
            rcut=4.0,
            rcut_smth=0.5,
            sel=2,
            ntypes=2,
            neuron=[4, 8],
            tebd_dim=2,
            tebd_input_mode="strip",
            concat_output_tebd=False,
            seed=7,
        )
    )
    coord = array_api_strict.asarray(
        np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=np.float64,
        )
    )
    # Atom 2 participates in both local environments so the strip-mode
    # type-pair lookup must consume its remapped padding index.
    nlist = array_api_strict.asarray(np.array([[[1, 2], [0, 2]]], dtype=np.int64))
    virtual_atype = array_api_strict.asarray(np.array([[0, 1, -1]], dtype=np.int64))
    padding_atype = array_api_strict.asarray(np.array([[0, 1, 2]], dtype=np.int64))

    actual = descriptor(coord, virtual_atype, nlist)
    expected = descriptor(coord, padding_atype, nlist)

    for actual_value, expected_value in zip(actual, expected, strict=True):
        if actual_value is not None:
            np.testing.assert_allclose(
                to_numpy_array(actual_value), to_numpy_array(expected_value)
            )


def test_dpa2_virtual_neighbor_matches_explicit_padding() -> None:
    """DPA2 gathers the padding embedding for a virtual extended atom."""
    descriptor = DescrptDPA2(
        ntypes=2,
        repinit=RepinitArgs(
            rcut=4.0,
            rcut_smth=0.5,
            nsel=4,
            neuron=[4, 8],
            axis_neuron=2,
            tebd_dim=2,
            tebd_input_mode="strip",
        ),
        repformer=RepformerArgs(
            rcut=3.0,
            rcut_smth=0.5,
            nsel=2,
            nlayers=1,
            g1_dim=8,
            g2_dim=4,
            axis_neuron=2,
            attn1_hidden=8,
            attn1_nhead=2,
            attn2_hidden=4,
            attn2_nhead=2,
        ),
        concat_output_tebd=False,
        seed=11,
    )
    coord = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    nlist = np.array([[[1, 2, -1, -1], [0, 2, -1, -1]]], dtype=np.int64)
    mapping = np.array([[0, 1, 0]], dtype=np.int64)

    actual = descriptor(
        coord,
        np.array([[0, 1, -1]], dtype=np.int64),
        nlist,
        mapping,
    )
    expected = descriptor(
        coord,
        np.array([[0, 1, 2]], dtype=np.int64),
        nlist,
        mapping,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value)


@pytest.mark.parametrize("use_loc_mapping", [True, False])
def test_dpa3_virtual_type_uses_padding_for_both_mapping_paths(
    use_loc_mapping: bool,
) -> None:
    """DPA3 remaps before either local-only or full extended gathering.

    A virtual ``-1`` is deliberately avoided for the sentinel: under both
    NumPy and array_api_strict, a raw ``take`` wraps ``-1`` to the final
    (padding) row, which would coincide with a correct remap and let a
    regression slip through. ``-2`` wraps to a real row instead, so the test
    only passes when the negative type is explicitly remapped to padding.
    """
    descriptor = convert_array_api_strict_value(
        DescrptDPA3(
            ntypes=2,
            repflow=RepFlowArgs(
                n_dim=8,
                e_dim=4,
                a_dim=4,
                nlayers=1,
                e_rcut=4.0,
                e_rcut_smth=0.5,
                e_sel=4,
                a_rcut=4.0,
                a_rcut_smth=0.5,
                a_sel=3,
                axis_neuron=2,
                update_angle=False,
            ),
            use_loc_mapping=use_loc_mapping,
            seed=17,
        )
    )
    table = np.vstack(
        (
            np.arange(descriptor.tebd_dim, dtype=np.float64),
            np.arange(descriptor.tebd_dim, dtype=np.float64) + 10.0,
            np.zeros(descriptor.tebd_dim),
        )
    )
    strict_table = array_api_strict.asarray(table)
    descriptor.type_embedding.call = lambda: strict_table

    class CaptureRepflows:
        """Capture DPA3's initial node embeddings without later env lookups."""

        node_ebd_ext: Any

        def __call__(
            self,
            nlist,
            coord_ext,
            atype_ext,
            node_ebd_ext,
            mapping,
            comm_dict=None,
        ):
            self.node_ebd_ext = node_ebd_ext
            nframes, nloc, nnei = nlist.shape
            return (
                node_ebd_ext[:, :nloc, :],
                np.zeros((nframes, nloc, nnei, 4)),
                np.zeros((nframes, nloc, nnei, 3)),
                np.zeros((nframes, nloc, 4, 3)),
                np.zeros((nframes, nloc, nnei)),
            )

    capture = CaptureRepflows()
    descriptor.repflows = capture
    coord = array_api_strict.asarray(
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    )
    nlist = array_api_strict.asarray(
        np.array([[[1, 2, -1, -1], [0, 2, -1, -1]]], dtype=np.int64)
    )
    mapping = array_api_strict.asarray(np.array([[0, 1, 0]], dtype=np.int64))
    atype = array_api_strict.asarray(
        np.array([[0, -2, 1] if use_loc_mapping else [0, 1, -2]], dtype=np.int64)
    )

    descriptor(coord, atype, nlist, mapping)

    expected_rows = [0, 2] if use_loc_mapping else [0, 1, 2]
    np.testing.assert_array_equal(
        to_numpy_array(capture.node_ebd_ext), table[expected_rows][None, ...]
    )
