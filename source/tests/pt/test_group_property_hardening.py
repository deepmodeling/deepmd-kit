# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt-backend hardening tests for the grouped property path (Tasks B-F).

These require the compiled ``deepmd.lib`` extension and skip cleanly otherwise.
Run on Bohrium/CI with the DeePMD 3.1.3 env:

    pytest source/tests/pt/test_group_property_hardening.py -v
"""

from __future__ import (
    annotations,
)

import pytest

pytest.importorskip("torch")
try:
    import torch

    from deepmd.pt.loss.group_property import (
        GroupPropertyLoss,
    )
    from deepmd.pt.model.model.group_property_model import (
        GroupPropertyModel,
    )
    from deepmd.pt.model.task.group_property import (
        GroupPropertyFittingNet,
    )
    from deepmd.pt.utils.grouped import (
        distributed_grouped_frame_batches,
    )
except ImportError as exc:  # needs the compiled deepmd.lib extension
    pytest.skip(f"deepmd backend unavailable: {exc}", allow_module_level=True)


class _StubDescriptor(torch.nn.Module):
    """Per-atom feature = atom's local index (deterministic), so masked-mean /
    reduce results are hand-checkable.  Real atoms occupy the first ``natoms`` rows.
    """

    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        self.dim = dim

    def get_rcut(self) -> float:
        return 6.0

    def get_sel(self) -> list[int]:
        return [8]

    def mixed_types(self) -> bool:
        return True

    def forward(self, ext_coord, ext_atype, nlist, mapping=None, charge_spin=None):
        nframes, nall = ext_atype.shape
        idx = torch.arange(nall, dtype=ext_coord.dtype).reshape(1, nall, 1)
        desc = idx.expand(nframes, nall, self.dim).clone()
        return desc, None, None, None, None


def _fitting(dim=2, task_dim=1, neuron=(4,), numb_fparam=0, group_reduce="mean"):
    return GroupPropertyFittingNet(
        ntypes=2,
        dim_descrpt=dim,
        property_name="y",
        task_dim=task_dim,
        neuron=list(neuron),
        numb_fparam=numb_fparam,
        group_reduce=group_reduce,
    )


def _model(fitting, dim=2):
    return GroupPropertyModel(
        descriptor=_StubDescriptor(dim), fitting=fitting, type_map=["A", "B"]
    )


def _coords(nframes, natoms):
    return torch.rand(nframes, natoms, 3, dtype=torch.float64)


# --------------------------------------------------------------------------- C
def test_masked_mean_model_divides_by_mask_sum():
    model = _model(_fitting())
    coord = _coords(1, 3)
    atype = torch.tensor([[0, 1, 0]])
    pool_mask = torch.tensor([[1.0, 1.0, 0.0]])  # atom 2 excluded
    out = model(
        coord,
        atype,
        box=None,
        group_id=torch.tensor([[0]]),
        weight=torch.tensor([[1.0]]),
        pool_mask=pool_mask,
    )
    # per-atom features = [0,0],[1,1],[2,2]; masked mean over atoms 0,1 = [0.5,0.5]
    assert torch.allclose(
        out["frame_embedding"], torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    )


def test_zero_mask_rejected_by_model():
    model = _model(_fitting())
    with pytest.raises(ValueError, match="all-zero pool_mask"):
        model(
            _coords(1, 3),
            torch.tensor([[0, 1, 0]]),
            box=None,
            group_id=torch.tensor([[0]]),
            weight=torch.tensor([[1.0]]),
            pool_mask=torch.zeros(1, 3),
        )


# --------------------------------------------------------------------------- B
def test_group_reduce_mean_vs_sum_wiring():
    fitting = _fitting(group_reduce="mean")
    model = _model(fitting)
    atype = torch.tensor([[0, 1], [0, 1]])
    coord = _coords(2, 2)
    kw = {
        "box": None,
        "pool_mask": torch.ones(2, 2),
        "weight": torch.tensor([[1.0], [1.0]]),
    }

    # two identical frames in one group: mean-reduce == a single-frame group
    pred_two = model(coord, atype, group_id=torch.tensor([[0], [0]]), **kw)["y"]
    pred_one = model(
        coord[:1],
        atype[:1],
        box=None,
        pool_mask=torch.ones(1, 2),
        weight=torch.tensor([[1.0]]),
        group_id=torch.tensor([[0]]),
    )["y"]
    assert torch.allclose(pred_two, pred_one)  # mean is size-invariant

    fitting.group_reduce = "sum"  # same weights, sum now doubles the embedding
    pred_sum = model(coord, atype, group_id=torch.tensor([[0], [0]]), **kw)["y"]
    assert not torch.allclose(pred_sum, pred_two)


def test_single_frame_group_equivalence():
    fitting = _fitting(group_reduce="mean")
    model = _model(fitting)
    coord = _coords(3, 2)
    atype = torch.tensor([[0, 1], [0, 1], [0, 1]])
    out = model(
        coord,
        atype,
        box=None,
        group_id=torch.tensor([[0], [1], [2]]),  # each frame its own group
        weight=torch.ones(3, 1),
        pool_mask=torch.ones(3, 2),
    )
    # singleton groups + mean + weight 1 => grouping is a no-op:
    # prediction == fitting applied to each frame embedding directly.
    direct = fitting(out["frame_embedding"])
    assert torch.allclose(out["y"], direct)


# --------------------------------------------------------------------------- E
def test_group_output_bias_zero_initialized():
    fitting = _fitting()
    last = fitting.network[-1]
    assert isinstance(last, torch.nn.Linear)
    assert torch.allclose(last.bias, torch.zeros_like(last.bias))  # route 3: zero-init


# --------------------------------------------------------------------------- F
def test_shuffled_batch_loss_invariance():
    torch.manual_seed(0)
    fitting = _fitting()
    model = _model(fitting)
    loss_fn = GroupPropertyLoss(task_dim=1, var_name="y")

    natoms = 2
    nframes = 5
    coord = _coords(nframes, natoms)
    atype = torch.zeros(nframes, natoms, dtype=torch.long)
    group_id = torch.tensor([2, 0, 2, 1, 0])
    # consistent per-group labels
    label_of = {0: 0.0, 1: 10.0, 2: 20.0}
    y = torch.tensor([[label_of[int(g)]] for g in group_id], dtype=torch.float64)

    def run(perm):
        inp = {"coord": coord[perm], "atype": atype[perm], "box": None}
        lab = {
            "y": y[perm],
            "group_id": group_id[perm].reshape(-1, 1).double(),
            "weight": torch.ones(nframes, 1).double(),
            "pool_mask": torch.ones(nframes, natoms).double(),
        }
        _, loss, _ = loss_fn(inp, model, lab, natoms=natoms)
        return loss

    ref = run(torch.arange(nframes))
    shuffled = run(torch.tensor([3, 0, 4, 1, 2]))
    assert torch.allclose(ref, shuffled)


# --------------------------------------------------------------------------- serialize
def test_serialize_roundtrip_group_reduce_and_fparam():
    fitting = _fitting(numb_fparam=2, group_reduce="sum")
    sd = fitting.serialize()
    assert sd["group_reduce"] == "sum"
    assert sd["numb_fparam"] == 2
    rebuilt = GroupPropertyFittingNet(**{k: v for k, v in sd.items() if k != "type"})
    assert rebuilt.group_reduce == "sum"
    assert rebuilt.get_dim_fparam() == 2


# --------------------------------------------------------------------------- D (nlist)
def test_padding_invariance():
    """Real descriptor: appending virtual atoms (atype=-1) must not change the
    descriptors of real atoms, even when virtual atoms sit inside the cutoff.
    """
    from deepmd.pt.model.descriptor import (
        DescrptSeA,
    )
    from deepmd.pt.utils.nlist import (
        extend_input_and_build_neighbor_list,
    )

    desc = DescrptSeA(rcut=6.0, rcut_smth=5.0, sel=[20, 20], ntypes=2)

    def run(coord, atype, box):
        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord, atype, desc.get_rcut(), desc.get_sel(), mixed_types=True, box=box
        )
        return desc(ext_coord, ext_atype, nlist, mapping=mapping)[0]

    torch.manual_seed(0)
    base_coord = torch.rand(1, 8, 3, dtype=torch.float64) * 5.0
    base_atype = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
    box = None  # non-periodic
    ref = run(base_coord, base_atype, box)[:, :8, :]

    # adversarial: virtual atoms placed INSIDE the cutoff of real atoms
    pad = base_coord[:, :3, :].clone()  # 3 virtual atoms among the real cloud
    pad_coord = torch.cat([base_coord, pad], dim=1)
    pad_atype = torch.cat([base_atype, torch.full((1, 3), -1)], dim=1)
    got = run(pad_coord, pad_atype, box)[:, :8, :]
    assert torch.allclose(ref, got, atol=1e-10)

    # periodic case: nlist masks atype<0 regardless of coord wrapping
    box_p = (torch.eye(3, dtype=torch.float64) * 10.0).reshape(1, 9)
    ref_p = run(base_coord, base_atype, box_p)[:, :8, :]
    got_p = run(pad_coord, pad_atype, box_p)[:, :8, :]
    assert torch.allclose(ref_p, got_p, atol=1e-10)


# --------------------------------------------------------------------------- DDP
def test_group_not_split_across_ranks():
    # 5 groups of varying size, world_size=2; assign groups (not frames) to ranks.
    group_ids = torch.tensor([0, 0, 0, 1, 2, 2, 3, 4, 4, 4]).numpy()
    seen: dict[int, int] = {}
    for rank in range(2):
        batches = distributed_grouped_frame_batches(
            group_ids, num_replicas=2, rank=rank, max_frames=8
        )
        for batch in batches:
            for frame_idx in batch:
                gid = int(group_ids[frame_idx])
                assert seen.setdefault(gid, rank) == rank, (
                    f"group {gid} split across ranks"
                )
    # every group landed on exactly one rank, and all groups were covered
    assert set(seen) == {0, 1, 2, 3, 4}
