# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the PyTorch mixed-nloc flat forward path."""

import pytest
import torch

from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA3,
)
from deepmd.pt.model.model import (
    EnergyModel,
)
from deepmd.pt.model.task import (
    EnergyFittingNet,
)
from deepmd.pt.utils.nlist import (
    build_precomputed_flat_graph,
)

from ....seed import (
    GLOBAL_SEED,
)


def _build_model() -> EnergyModel:
    repflow = RepFlowArgs(
        n_dim=8,
        e_dim=4,
        a_dim=4,
        nlayers=1,
        e_rcut=1.5,
        e_rcut_smth=0.2,
        e_sel=4,
        a_rcut=1.0,
        a_rcut_smth=0.1,
        a_sel=3,
        axis_neuron=2,
        update_angle=True,
        update_style="res_residual",
        update_residual_init="const",
        a_compress_rate=0,
        n_multi_edge_message=1,
        smooth_edge_update=True,
    )
    descriptor = DescrptDPA3(
        2,
        repflow=repflow,
        precision="float64",
        seed=GLOBAL_SEED,
        type_map=["O", "H"],
    )
    fitting = EnergyFittingNet(
        ntypes=2,
        dim_descrpt=descriptor.get_dim_out(),
        neuron=[8],
        mixed_types=descriptor.mixed_types(),
        type_map=["O", "H"],
        precision="float64",
        seed=GLOBAL_SEED,
    )
    return EnergyModel(descriptor, fitting, type_map=["O", "H"]).to("cpu").eval()


def _mixed_batch() -> tuple[torch.Tensor, ...]:
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
        ],
        dtype=torch.float64,
        device="cpu",
    )
    atype = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long, device="cpu")
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device="cpu")
    ptr = torch.tensor([0, 2, 5], dtype=torch.long, device="cpu")
    box = (
        torch.eye(3, dtype=torch.float64, device="cpu").reshape(1, 9).repeat(2, 1)
        * 10.0
    )
    return coord, atype, batch, ptr, box


def _flat_graph(
    coord: torch.Tensor,
    atype: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
    box: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return build_precomputed_flat_graph(
        coord,
        atype,
        batch,
        ptr,
        rcut=1.5,
        sel=[4],
        a_rcut=1.0,
        a_sel=3,
        mixed_types=True,
        box=box,
    )


def test_dpa3_flat_forward_matches_regular_per_frame_energy_and_force() -> None:
    model = _build_model()
    coord, atype, batch, ptr, box = _mixed_batch()
    graph = _flat_graph(coord, atype, batch, ptr, box)

    flat = model(
        coord,
        atype,
        box=box,
        mixed_batch={"batch": batch, "ptr": ptr, **graph},
    )
    regular = []
    for frame_idx in range(ptr.numel() - 1):
        start = int(ptr[frame_idx].item())
        end = int(ptr[frame_idx + 1].item())
        regular.append(
            model(
                coord[start:end].reshape(1, -1),
                atype[start:end].reshape(1, -1),
                box=box[frame_idx : frame_idx + 1],
            )
        )

    expected_energy = torch.cat([item["energy"] for item in regular], dim=0)
    expected_atom_energy = torch.cat(
        [item["atom_energy"].reshape(-1, 1) for item in regular], dim=0
    )
    expected_force = torch.cat(
        [item["force"].reshape(-1, 3) for item in regular], dim=0
    )

    torch.testing.assert_close(flat["energy"], expected_energy)
    torch.testing.assert_close(flat["atom_energy"], expected_atom_energy)
    torch.testing.assert_close(flat["force"], expected_force)
    torch.testing.assert_close(
        flat["energy"],
        torch.stack([flat["atom_energy"][:2].sum(0), flat["atom_energy"][2:].sum(0)]),
    )
    assert flat["virial"].shape == (2, 9)
    assert torch.isfinite(flat["virial"]).all()
    assert flat["mask"].tolist() == [1, 1, 1, 1, 1]


def test_dpa3_flat_forward_requires_precomputed_graph_fields() -> None:
    model = _build_model()
    coord, atype, batch, ptr, box = _mixed_batch()

    with pytest.raises(RuntimeError, match="precomputed graph fields"):
        model(coord, atype, box=box, mixed_batch={"batch": batch, "ptr": ptr})


def test_dpa3_flat_forward_rejects_atomic_virial() -> None:
    model = _build_model()
    coord, atype, batch, ptr, box = _mixed_batch()
    graph = _flat_graph(coord, atype, batch, ptr, box)

    with pytest.raises(NotImplementedError, match="Atomic virial"):
        model(
            coord,
            atype,
            box=box,
            do_atomic_virial=True,
            mixed_batch={"batch": batch, "ptr": ptr, **graph},
        )
