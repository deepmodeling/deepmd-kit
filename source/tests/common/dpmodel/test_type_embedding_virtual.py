# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for virtual types in dpmodel embedding gathers."""

import array_api_strict
import numpy as np
import pytest

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
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


@pytest.mark.parametrize("namespace", [np, array_api_strict])
def test_padded_type_embedding_maps_virtual_type(namespace) -> None:
    """Negative types select the explicit final zero row on every backend."""
    table = namespace.asarray(
        np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
    )
    atype = namespace.asarray(np.array([0, -1, 1], dtype=np.int64))

    actual = take_type_embedding(table, atype)

    np.testing.assert_array_equal(
        to_numpy_array(actual),
        [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
    )


@pytest.mark.parametrize("namespace", [np, array_api_strict])
def test_sezm_padded_embedding_maps_virtual_type(namespace) -> None:
    """SeZM uses the same padding-row contract at its gather boundary."""
    embedding = SeZMTypeEmbedding(ntypes=2, embed_dim=2, padding=True, seed=1)
    embedding.adam_type_embedding = np.array(
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float64
    )
    atype = namespace.asarray(np.array([0, -1, 1], dtype=np.int64))

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
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 9.0, 9.0]]])
    )
    nlist = array_api_strict.asarray(
        np.array([[[1, -1, -1, -1], [0, -1, -1, -1]]], dtype=np.int64)
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
