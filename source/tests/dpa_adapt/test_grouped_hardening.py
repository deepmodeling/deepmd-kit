# SPDX-License-Identifier: LGPL-3.0-or-later
"""Locally-runnable hardening checks (no compiled deepmd.lib).

- Task C: all-masked frame rejected at Assembly construction time.
- Task C: masked-mean formula (÷ mask sum, not natoms) — torch mirror of the model.
- Task B: group_reduce mean-vs-sum — torch mirror of the model reduction.
- Task F: shuffled-batch group-label aggregation invariance — torch mirror of the loss.

The model/loss/descriptor wiring versions live in source/tests/pt/ and need the
native extension (Bohrium/CI).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_zero_mask_rejected_at_construction():
    from dpa_adapt.data.errors import DPADataError
    from dpa_adapt.grouped import ComponentSpec

    comp = ComponentSpec.from_arrays(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], ["H", "O"], pool_mask=[0.0, 0.0]
    )
    with pytest.raises(DPADataError, match="all-zero"):
        comp.validate()


def test_zero_mask_rejected_through_writer(tmp_path):
    from dpa_adapt.data.errors import DPADataError
    from dpa_adapt.grouped import Assembly

    a = Assembly(target="y", type_map=["H", "O"])
    g = a.group(label=1.0)
    g.add([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], ["H", "O"], pool_mask=[0.0, 0.0])
    with pytest.raises(DPADataError, match="all-zero"):
        a.write(tmp_path / "out")


def test_masked_mean_divides_by_mask_sum():
    torch = pytest.importorskip("torch")
    # frame 0: atoms 0,1 kept (atom 2 masked); frame 1: only atom 0 kept
    descriptor = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[10.0, 10.0], [999.0, 999.0], [999.0, 999.0]],
        ]
    )
    pool_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # mirror GroupPropertyModel.forward masked mean
    mask_sum = pool_mask.sum(dim=1)
    assert not bool((mask_sum == 0).any())
    denom = mask_sum.clamp_min(1.0)
    keep = pool_mask[:, :, None] > 0
    desc = torch.where(keep, descriptor, torch.zeros_like(descriptor))
    frame_emb = (desc * pool_mask[:, :, None]).sum(dim=1) / denom[:, None]

    # frame0 = ([1,2]+[3,4])/2 = [2,3]; frame1 = [10,10]/1 = [10,10]
    assert torch.allclose(frame_emb, torch.tensor([[2.0, 3.0], [10.0, 10.0]]))
    # NOT divided by natoms (=3): that would give [4/3, 6/3]=[1.333,2] for frame0
    assert not torch.allclose(frame_emb[0], torch.tensor([4.0 / 3, 6.0 / 3]))


def _reduce(frame_emb, weight, group_id, mode):
    """Mirror of GroupPropertyModel.forward frame->group reduction."""
    import torch

    order, inverse = torch.unique(group_id, sorted=True, return_inverse=True)
    n = order.shape[0]
    ge = torch.zeros((n, frame_emb.shape[1]))
    ge.index_add_(0, inverse, frame_emb * weight[:, None])
    if mode == "mean":
        ws = torch.zeros((n, 1))
        ws.index_add_(0, inverse, weight[:, None])
        ge = ge / ws.clamp_min(1e-12)
    return ge


def test_group_reduce_mean_vs_sum():
    torch = pytest.importorskip("torch")
    frame_emb = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])  # 2 identical frames
    weight = torch.ones(2)
    group_id = torch.tensor([0, 0])  # one group

    mean = _reduce(frame_emb, weight, group_id, "mean")
    summ = _reduce(frame_emb, weight, group_id, "sum")

    assert torch.allclose(mean, frame_emb[:1])  # mean-reduce == the frame embedding
    assert torch.allclose(summ, 2.0 * frame_emb[:1])  # sum-reduce == 2 x


def _group_labels(frame_label, group_inverse, n_groups):
    """Mirror of GroupPropertyLoss._group_labels (uses shared inverse)."""
    import torch

    gl = frame_label.new_zeros((n_groups, frame_label.shape[1]))
    gl[group_inverse] = frame_label
    assert torch.allclose(gl[group_inverse], frame_label)  # consistency
    return gl


def test_shuffled_batch_group_labels_invariance():
    torch = pytest.importorskip("torch")
    # groups: 0->0.0, 1->10.0, 2->20.0; frames arrive interleaved
    order_sorted = torch.tensor([0, 1, 2])

    def group_labels_for(perm):
        group_id = torch.tensor([2, 0, 2, 1, 0])[perm]
        frame_label = torch.tensor([[20.0], [0.0], [20.0], [10.0], [0.0]])[perm]
        gorder, inverse = torch.unique(group_id, sorted=True, return_inverse=True)
        assert torch.equal(gorder, order_sorted)
        return _group_labels(frame_label, inverse, gorder.shape[0])

    ref = group_labels_for(torch.arange(5))
    shuffled = group_labels_for(torch.tensor([3, 0, 4, 1, 2]))
    assert torch.allclose(ref, shuffled)
    assert torch.allclose(ref, torch.tensor([[0.0], [10.0], [20.0]]))  # sorted by group id


def test_padding_coords_are_non_physical(tmp_path):
    """Task D data-writer: padding atoms get a large, spread-out offset (not 0,0,0)."""
    from dpa_adapt.grouped import Assembly

    a = Assembly(target="y", type_map=["C", "O", "H"])
    g = a.group(label=1.0)
    g.add([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], ["C", "O"], box=np.eye(3) * 20)  # 2 atoms
    g.add(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ["C", "O", "H"],
        box=np.eye(3) * 20,
    )  # 3 atoms -> frame 0 padded to 3
    res = a.write(tmp_path / "out")
    set_dir = tmp_path / "out" / res["systems"][0] / "set.000"
    coord = np.load(set_dir / "coord.npy").reshape(2, 3, 3)
    real = np.load(set_dir / "real_atom_types.npy")
    pad = real[0] < 0
    assert pad.sum() == 1  # frame 0 has one padding atom
    assert np.linalg.norm(coord[0][pad][0]) > 20.0  # far outside the 20 A box, not origin
