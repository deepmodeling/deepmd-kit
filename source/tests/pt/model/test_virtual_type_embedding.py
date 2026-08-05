# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for virtual types in legacy PyTorch strip descriptors."""

import torch

from deepmd.pt.model.descriptor.se_atten import (
    DescrptBlockSeAtten,
)
from deepmd.pt.model.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd,
)
from deepmd.pt.utils import (
    env,
)


def _inputs(nnei: int):
    device = env.DEVICE
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    full_nlist = torch.tensor(
        [[[1, 2, -1, -1], [0, 2, -1, -1]]],
        dtype=torch.long,
        device=device,
    )
    virtual_atype = torch.tensor([[0, 1, -1]], dtype=torch.long, device=device)
    padding_atype = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    type_embedding = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    virtual_embedding = type_embedding[
        torch.where(
            virtual_atype >= 0,
            virtual_atype,
            torch.full_like(virtual_atype, 2),
        )
    ]
    return (
        coord,
        full_nlist[:, :, :nnei],
        virtual_atype,
        padding_atype,
        type_embedding,
        virtual_embedding,
        type_embedding[padding_atype],
    )


def _assert_outputs_close(actual, expected) -> None:
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if actual_value is not None:
            torch.testing.assert_close(actual_value, expected_value)


def _virtual_center_inputs(nnei: int):
    device = env.DEVICE
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    nlist = torch.tensor([[[1, 2, -1, -1]]], dtype=torch.long, device=device)[
        :, :, :nnei
    ]
    virtual_atype = torch.tensor([[-1, 1, 0]], dtype=torch.long, device=device)
    reference_atype = torch.tensor([[0, 1, 0]], dtype=torch.long, device=device)
    type_embedding = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    virtual_embedding = type_embedding[
        torch.where(
            virtual_atype >= 0,
            virtual_atype,
            torch.full_like(virtual_atype, 2),
        )
    ]
    return (
        coord,
        nlist,
        virtual_atype,
        reference_atype,
        type_embedding,
        virtual_embedding,
        type_embedding[reference_atype],
    )


def test_se_atten_strip_virtual_neighbor_matches_padding() -> None:
    """Two-side DPA1 pair indices remap a virtual neighbor before folding."""
    (
        coord,
        nlist,
        virtual_atype,
        padding_atype,
        type_embedding,
        virtual_embedding,
        padding_embedding,
    ) = _inputs(4)
    descriptor = DescrptBlockSeAtten(
        rcut=4.0,
        rcut_smth=0.5,
        sel=[2, 2],
        ntypes=2,
        attn_layer=0,
        axis_neuron=2,
        neuron=[6, 12],
        tebd_dim=2,
        tebd_input_mode="strip",
        type_one_side=False,
        precision="float64",
        seed=1,
    ).to(env.DEVICE)

    actual = descriptor(
        nlist,
        coord,
        virtual_atype,
        virtual_embedding,
        type_embedding=type_embedding,
    )
    expected = descriptor(
        nlist,
        coord,
        padding_atype,
        padding_embedding,
        type_embedding=type_embedding,
    )

    _assert_outputs_close(actual, expected)


def test_se_t_tebd_strip_virtual_neighbor_matches_padding() -> None:
    """SE_T type-pair indices remap virtual neighbors on both pair axes."""
    (
        coord,
        nlist,
        virtual_atype,
        padding_atype,
        type_embedding,
        virtual_embedding,
        padding_embedding,
    ) = _inputs(2)
    descriptor = DescrptBlockSeTTebd(
        rcut=4.0,
        rcut_smth=0.5,
        sel=2,
        ntypes=2,
        neuron=[4, 8],
        tebd_dim=2,
        tebd_input_mode="strip",
        precision="float64",
        seed=1,
    ).to(env.DEVICE)

    actual = descriptor(
        nlist,
        coord,
        virtual_atype,
        virtual_embedding,
        type_embedding=type_embedding,
    )
    expected = descriptor(
        nlist,
        coord,
        padding_atype,
        padding_embedding,
        type_embedding=type_embedding,
    )

    _assert_outputs_close(actual, expected)


def test_se_atten_virtual_center_uses_real_type_statistics() -> None:
    """DPA1 clamps a virtual center before indexing mean and stddev."""
    (
        coord,
        nlist,
        virtual_atype,
        reference_atype,
        type_embedding,
        virtual_embedding,
        reference_embedding,
    ) = _virtual_center_inputs(4)
    descriptor = DescrptBlockSeAtten(
        rcut=4.0,
        rcut_smth=0.5,
        sel=[2, 2],
        ntypes=2,
        attn_layer=0,
        axis_neuron=2,
        neuron=[6, 12],
        tebd_dim=2,
        tebd_input_mode="strip",
        type_one_side=True,
        precision="float64",
        seed=1,
    ).to(env.DEVICE)
    descriptor.mean[0, :, :] = 0.25
    descriptor.mean[1, :, :] = -0.5

    actual = descriptor(
        nlist,
        coord,
        virtual_atype,
        virtual_embedding,
        type_embedding=type_embedding,
    )
    expected = descriptor(
        nlist,
        coord,
        reference_atype,
        reference_embedding,
        type_embedding=type_embedding,
    )

    _assert_outputs_close(actual, expected)


def test_se_t_tebd_virtual_center_uses_real_type_statistics() -> None:
    """SE_T clamps a virtual center before indexing mean and stddev."""
    (
        coord,
        nlist,
        virtual_atype,
        reference_atype,
        type_embedding,
        virtual_embedding,
        reference_embedding,
    ) = _virtual_center_inputs(2)
    descriptor = DescrptBlockSeTTebd(
        rcut=4.0,
        rcut_smth=0.5,
        sel=2,
        ntypes=2,
        neuron=[4, 8],
        tebd_dim=2,
        tebd_input_mode="strip",
        precision="float64",
        seed=1,
    ).to(env.DEVICE)
    descriptor.mean[0, :, :] = 0.25
    descriptor.mean[1, :, :] = -0.5

    actual = descriptor(
        nlist,
        coord,
        virtual_atype,
        virtual_embedding,
        type_embedding=type_embedding,
    )
    expected = descriptor(
        nlist,
        coord,
        reference_atype,
        reference_embedding,
        type_embedding=type_embedding,
    )

    _assert_outputs_close(actual, expected)
