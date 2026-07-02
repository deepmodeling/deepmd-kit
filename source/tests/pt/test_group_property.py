# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("deepmd.lib")

from deepmd.pt.loss.group_property import GroupPropertyLoss
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
    GroupCompleteBatchSampler,
    GroupDistributedBatchSampler,
)
from deepmd.pt.utils.grouped import (
    GROUP_ID_KEY,
    GROUP_WEIGHT_KEY,
    POOL_MASK_KEY,
    distributed_grouped_frame_batches,
    group_data_requirements,
    normalize_group_id_tensor,
    normalize_pool_mask_tensor,
    normalize_weight_tensor,
)
from deepmd.utils.data import DataRequirementItem


class ToyGroupedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4, bias=False)
        self.head = torch.nn.Linear(4, 1)
        self.var_name = "target"

    def forward(
        self,
        coord,
        atype,
        box=None,
        do_atomic_virial=False,
        fparam=None,
        aparam=None,
        charge_spin=None,
        group_id=None,
        weight=None,
        pool_mask=None,
    ):
        del box, do_atomic_virial, fparam, aparam, charge_spin
        nframes, natoms = atype.shape
        coord = coord.reshape(nframes, natoms, 3)
        descriptor = self.backbone(coord)
        pool_mask = normalize_pool_mask_tensor(pool_mask, nframes, natoms).to(
            descriptor.dtype
        )
        frame_embedding = (descriptor * pool_mask[:, :, None]).sum(dim=1) / pool_mask.sum(
            dim=1
        ).clamp_min(1.0)[:, None]
        group_id = normalize_group_id_tensor(group_id, nframes)
        weight = normalize_weight_tensor(weight, nframes).to(descriptor.dtype).detach()
        group_order, inverse = torch.unique(group_id, sorted=True, return_inverse=True)
        group_embedding = torch.zeros(
            group_order.shape[0], frame_embedding.shape[1], dtype=frame_embedding.dtype
        )
        group_embedding.index_add_(0, inverse, frame_embedding * weight[:, None])
        return {"target": self.head(group_embedding), "group_id": group_order}


def test_group_property_loss_trains_backbone_and_detaches_weight() -> None:
    torch.manual_seed(7)
    coord = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.2, 0.1, 0.0], [0.0, 1.1, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.1, 0.0, 1.2], [1.1, 0.9, 0.0]],
        ]
    )
    atype = torch.zeros((4, 2), dtype=torch.long)
    weight = torch.tensor([[0.5], [0.5], [0.5], [0.5]], requires_grad=True)
    label = {
        "target": torch.tensor([[1.0], [1.0], [2.0], [2.0]]),
        GROUP_ID_KEY: torch.tensor([[0], [0], [1], [1]], dtype=torch.long),
        GROUP_WEIGHT_KEY: weight,
        POOL_MASK_KEY: torch.ones((4, 2, 1)),
        f"find_{GROUP_ID_KEY}": torch.ones(1),
    }
    input_dict = {
        "coord": coord,
        "atype": atype,
        "box": None,
        "do_atomic_virial": False,
        "fparam": None,
        "aparam": None,
        "charge_spin": None,
    }
    model = ToyGroupedModel()
    loss_fn = GroupPropertyLoss(task_dim=1, var_name="target", loss_func="mse")
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    before = model.backbone.weight.detach().clone()
    losses = []
    for _ in range(80):
        opt.zero_grad()
        _, loss, _ = loss_fn(input_dict, model, label, natoms=2)
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]
    assert not torch.allclose(before, model.backbone.weight.detach())
    assert weight.grad is None


def test_group_property_loss_rejects_inconsistent_group_labels() -> None:
    model = ToyGroupedModel()
    loss_fn = GroupPropertyLoss(task_dim=1, var_name="target", loss_func="mse")
    label = {
        "target": torch.tensor([[1.0], [1.5]]),
        GROUP_ID_KEY: torch.tensor([[0], [0]], dtype=torch.long),
        GROUP_WEIGHT_KEY: torch.ones((2, 1)),
        POOL_MASK_KEY: torch.ones((2, 1, 1)),
        f"find_{GROUP_ID_KEY}": torch.ones(1),
    }
    input_dict = {
        "coord": torch.ones((2, 1, 3)),
        "atype": torch.zeros((2, 1), dtype=torch.long),
        "box": None,
        "do_atomic_virial": False,
        "fparam": None,
        "aparam": None,
        "charge_spin": None,
    }
    try:
        loss_fn(input_dict, model, label, natoms=1)
    except ValueError as exc:
        assert "Inconsistent target labels" in str(exc)
    else:
        raise AssertionError("expected inconsistent group labels to raise")


def test_group_complete_batch_sampler_keeps_groups_intact() -> None:
    sampler = GroupCompleteBatchSampler(
        np.array([0, 0, 1, 1, 2]), max_frames=2, shuffle=False
    )
    assert list(sampler) == [[0, 1], [2, 3], [4]]



def test_distributed_group_batches_assign_whole_groups_to_one_rank() -> None:
    group_ids = np.array([0, 0, 1, 1, 2, 3, 3, 4])
    rank0 = distributed_grouped_frame_batches(
        group_ids, max_frames=3, num_replicas=2, rank=0, shuffle=False
    )
    rank1 = distributed_grouped_frame_batches(
        group_ids, max_frames=3, num_replicas=2, rank=1, shuffle=False
    )

    rank_frames = [set(sum(rank0, [])), set(sum(rank1, []))]
    assert rank_frames[0].isdisjoint(rank_frames[1])
    assert rank_frames[0] | rank_frames[1] == set(range(len(group_ids)))

    owner_by_group = {}
    for rank, frames in enumerate(rank_frames):
        for frame in frames:
            group = int(group_ids[frame])
            owner_by_group.setdefault(group, rank)
            assert owner_by_group[group] == rank


def test_group_distributed_batch_sampler_keeps_groups_intact_per_rank() -> None:
    group_ids = np.array([0, 0, 1, 1, 2, 3, 3, 4])
    samplers = [
        GroupDistributedBatchSampler(
            group_ids,
            max_frames=3,
            num_replicas=2,
            rank=rank,
            shuffle=False,
        )
        for rank in range(2)
    ]
    batches_by_rank = [list(sampler) for sampler in samplers]
    frames_by_rank = [set(sum(batches, [])) for batches in batches_by_rank]

    assert frames_by_rank[0].isdisjoint(frames_by_rank[1])
    assert frames_by_rank[0] | frames_by_rank[1] == set(range(len(group_ids)))
    for rank, frames in enumerate(frames_by_rank):
        for group in set(group_ids):
            group_frames = {ii for ii, gid in enumerate(group_ids) if gid == group}
            if frames & group_frames:
                assert group_frames <= frames

def test_dploaderset_enables_group_batches_for_group_requirements(tmp_path) -> None:
    system = tmp_path / "sys"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type.raw").write_text("0\n0\n")
    (system / "type_map.raw").write_text("H\n")
    np.save(set_dir / "coord.npy", np.zeros((5, 6), dtype=np.float64))
    np.save(set_dir / "box.npy", np.tile(np.eye(3).reshape(1, 9), (5, 1)))
    np.save(set_dir / "target.npy", np.array([[1.0], [1.0], [2.0], [2.0], [3.0]]))
    np.save(set_dir / f"{GROUP_ID_KEY}.npy", np.array([0, 0, 1, 1, 2]))
    np.save(set_dir / f"{GROUP_WEIGHT_KEY}.npy", np.ones(5))
    np.save(set_dir / f"{POOL_MASK_KEY}.npy", np.ones((5, 2)))

    data = DpLoaderSet([str(system)], batch_size=2, type_map=["H"], shuffle=False)
    req = [DataRequirementItem("target", ndof=1, atomic=False, must=True)]
    req.extend(group_data_requirements())
    data.add_data_requirement(req)
    batches = [batch[GROUP_ID_KEY].reshape(-1).tolist() for batch in data.dataloaders[0]]
    assert batches == [[0, 0], [1, 1], [2]]
