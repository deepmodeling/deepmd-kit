# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end checks for heterogeneous grouped assemblies.

These cover the full data path for groups whose components differ in size and
composition (e.g. OER O*/OH*/OOH*): the assembly writer pads every frame up to
the group's max atom count and emits the DeepMD ``mixed_type`` layout, and the
grouped model must aggregate the padded frames without letting the virtual
padding atoms corrupt the pooled embedding.

The ``deepmd.pt`` integration tests need the compiled ``deepmd.lib`` extension
and therefore only run in a full build (they skip cleanly otherwise); the
standalone pooling test runs anywhere ``torch`` is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpa_adapt import AssemblyDatasetBuilder, ComponentSpec, PoolMask


def _require_real_torch():
    """Return real torch, skipping if a sibling test leaked a mock stub.

    Several ``dpa_adapt`` tests do ``sys.modules.setdefault("torch", MagicMock())``
    at import time, which shadows real torch for the rest of the session when
    they are collected first.  These tests need the genuine library.
    """
    torch = pytest.importorskip("torch")
    if type(torch).__module__.startswith("unittest.mock"):
        pytest.skip("real torch shadowed by a leaked mock-torch stub in this session")
    return torch


def _heterogeneous_builder() -> AssemblyDatasetBuilder:
    builder = AssemblyDatasetBuilder(property_name="target", type_map=["C", "O", "H"])
    group = builder.group(key="oer", label=1.5)
    # component with 2 atoms -> padded to 3 with one virtual atom
    group.add_component(
        ComponentSpec.from_arrays(
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], ["C", "O"], box=np.eye(3) * 20
        )
    )
    # component with 3 atoms, capping H excluded from pooling
    group.add_component(
        ComponentSpec.from_arrays(
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ["C", "O", "H"],
            box=np.eye(3) * 20,
            pool_mask=PoolMask.exclude_indices([2]),
        )
    )
    return builder


def test_writer_to_dataloader_preserves_padding_and_group(tmp_path) -> None:
    _require_real_torch()
    try:
        from deepmd.pt.utils.dataloader import DpLoaderSet
        from deepmd.pt.utils.grouped import group_data_requirements
        from deepmd.utils.data import DataRequirementItem
    except ImportError as exc:  # needs the compiled deepmd.lib extension
        pytest.skip(f"deepmd backend unavailable: {exc}")

    builder = _heterogeneous_builder()
    result = builder.write_deepmd_npy(tmp_path)
    system = str(tmp_path / result["systems"][0])

    data = DpLoaderSet([system], batch_size=8, type_map=["C", "O", "H"], shuffle=False)
    req = [DataRequirementItem("target", ndof=1, atomic=False, must=True)]
    req.extend(group_data_requirements())
    data.add_data_requirement(req)

    batch = next(iter(data.dataloaders[0]))

    atype = batch["atype"]
    assert atype.shape == (2, 3)
    # the smaller component keeps a trailing virtual atom (type -1)
    assert atype[0].tolist() == [0, 1, -1]
    assert atype[1].tolist() == [0, 1, 2]
    # the whole group lands in a single batch (group completeness)
    assert batch["group_id"].reshape(-1).tolist() == [0, 0]
    # pool_mask stays aligned with coord: padding atom and excluded H are masked
    assert batch["pool_mask"].reshape(2, 3).tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    # padding atom coordinates are the written zeros
    assert batch["coord"].reshape(2, 3, 3)[0, 2].tolist() == [0.0, 0.0, 0.0]


def test_group_property_model_pool_is_nan_safe_for_virtual_atoms() -> None:
    torch = _require_real_torch()
    try:
        from deepmd.pt.model.model.group_property_model import GroupPropertyModel
        from deepmd.pt.model.task.group_property import GroupPropertyFittingNet
    except ImportError as exc:  # needs the compiled deepmd.lib extension
        pytest.skip(f"deepmd backend unavailable: {exc}")

    dim = 4

    class _NaNOnVirtualDescriptor(torch.nn.Module):
        """Descriptor stub that returns NaN for virtual (type < 0) atoms.

        This is the worst case for the masked mean pool: if padding rows were
        multiplied by a zero ``pool_mask`` (``0 * NaN``) the frame embedding
        would be NaN.  The model must instead drop those rows entirely.
        """

        def get_rcut(self) -> float:
            return 6.0

        def get_sel(self) -> list[int]:
            return [8]

        def mixed_types(self) -> bool:
            return True

        def forward(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping=None,
            charge_spin=None,
        ):
            nframes, nall = extended_atype.shape
            desc = torch.ones(nframes, nall, dim, dtype=extended_coord.dtype)
            desc = torch.where(
                (extended_atype < 0)[:, :, None],
                torch.full_like(desc, float("nan")),
                desc,
            )
            return desc, None, None, None, None

    fitting = GroupPropertyFittingNet(
        ntypes=3,
        dim_descrpt=dim,
        property_name="target",
        task_dim=1,
        neuron=[4],
    )
    model = GroupPropertyModel(
        descriptor=_NaNOnVirtualDescriptor(),
        fitting=fitting,
        type_map=["C", "O", "H"],
    )

    coord = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 0.0, 0.0]],  # 2 real + 1 virtual
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.0, 0.0, 0.0]],  # 3 real
        ]
    )
    atype = torch.tensor([[0, 1, -1], [0, 1, 2]])
    group_id = torch.tensor([[0], [0]])
    weight = torch.tensor([[0.5], [0.5]])
    pool_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    out = model(
        coord,
        atype,
        box=None,
        group_id=group_id,
        weight=weight,
        pool_mask=pool_mask,
    )
    # the virtual padding atom must not poison the pooled embedding/prediction
    assert torch.isfinite(out["frame_embedding"]).all()
    assert torch.isfinite(out["target"]).all()
    # the whole group aggregates to a single prediction row
    assert out["target"].shape[0] == 1


def test_masked_mean_pool_is_nan_safe() -> None:
    """Standalone check of the model's NaN-safe masked mean pool.

    Mirrors the aggregation in ``GroupPropertyModel.forward``: excluded atoms
    (``pool_mask`` == 0) -- padding/virtual atoms or capping H -- must be dropped
    before the weighted sum, so a non-finite descriptor on those rows cannot
    propagate through ``0 * NaN``.  A naive ``descriptor * pool_mask`` would.
    """
    torch = _require_real_torch()

    # atom 2 in frame 0 is excluded and carries a NaN descriptor
    descriptor = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [float("nan"), float("nan")]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ],
        requires_grad=True,
    )
    pool_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    # naive form poisons the pooled embedding
    naive = (descriptor * pool_mask[:, :, None]).sum(dim=1)
    assert torch.isnan(naive[0]).any()

    # hardened form (as used by the model) stays finite and correct
    denom = pool_mask.sum(dim=1).clamp_min(1.0)
    keep = pool_mask[:, :, None] > 0
    masked = torch.where(keep, descriptor, torch.zeros_like(descriptor))
    frame_embedding = (masked * pool_mask[:, :, None]).sum(dim=1) / denom[:, None]

    assert torch.isfinite(frame_embedding).all()
    # frame 0 mean over the two kept atoms: ([1,2] + [3,4]) / 2 = [2, 3]
    assert torch.allclose(frame_embedding[0], torch.tensor([2.0, 3.0]))

    # gradient flows to kept atoms and is zero (not NaN) on the excluded row
    frame_embedding.sum().backward()
    assert torch.isfinite(descriptor.grad).all()
    assert torch.equal(descriptor.grad[0, 2], torch.zeros(2))


def test_group_property_model_consumes_per_group_fparam() -> None:
    """numb_fparam widens the fitting input; the model concats the per-group
    fparam AFTER aggregation, so it bypasses the weighted sum over frames.
    """
    torch = _require_real_torch()
    try:
        from deepmd.pt.model.model.group_property_model import GroupPropertyModel
        from deepmd.pt.model.task.group_property import GroupPropertyFittingNet
    except ImportError as exc:  # needs the compiled deepmd.lib extension
        pytest.skip(f"deepmd backend unavailable: {exc}")

    dim, fdim = 4, 2

    fitting = GroupPropertyFittingNet(
        ntypes=2,
        dim_descrpt=dim,
        property_name="y",
        task_dim=1,
        neuron=[4],
        numb_fparam=fdim,
    )
    assert fitting.get_dim_fparam() == fdim
    first_linear = next(m for m in fitting.network if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == dim + fdim  # descriptor dim + fparam dim

    class _ConstDescriptor(torch.nn.Module):
        def get_rcut(self) -> float:
            return 6.0

        def get_sel(self) -> list[int]:
            return [8]

        def mixed_types(self) -> bool:
            return True

        def forward(
            self, extended_coord, extended_atype, nlist, mapping=None, charge_spin=None
        ):
            nframes, nall = extended_atype.shape
            desc = torch.ones(nframes, nall, dim, dtype=extended_coord.dtype)
            return desc, None, None, None, None

    model = GroupPropertyModel(
        descriptor=_ConstDescriptor(), fitting=fitting, type_map=["A", "B"]
    )
    coord = torch.rand(2, 3, 3, dtype=torch.float64)
    atype = torch.tensor([[0, 1, 0], [0, 1, 1]])
    group_id = torch.tensor([[0], [0]])  # both frames -> one group
    weight = torch.tensor([[0.5], [0.5]])
    pool_mask = torch.ones(2, 3)
    fparam = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)  # per-group

    out = model(
        coord,
        atype,
        box=None,
        fparam=fparam,
        group_id=group_id,
        weight=weight,
        pool_mask=pool_mask,
    )
    assert out["y"].shape[0] == 1  # aggregates to one group prediction
    assert torch.isfinite(out["y"]).all()
