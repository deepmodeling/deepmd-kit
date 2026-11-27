# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 on the compact canonical lower: parity, halo rows and export.

The lower views the compact graph as a ``NeighborGraph`` and runs the graph
forward of the model, so its outputs equal those of the graph lower on the
same edges. A node set with halo rows refreshes them through its border plan
before every block that reads them, and its forces, folded onto the owners,
equal those of the folded frame.
"""

from __future__ import (
    annotations,
)

import copy
import dataclasses
import json
import zipfile
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    attach_edge_csr,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.graph_lower import (
    canonical_lower_evaluator,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.canonical_graph import (
    BorderPlan,
    CanonicalGraph,
    canonical_graph_from_neighbor_graph,
    canonical_index_bytes,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or env.DEVICE.type != "cuda",
    reason="the compact canonical lower serves CUDA devices",
)

#: A small DPA4.5-like model of three blocks: rising degrees, mixing layers,
#: the environment seed and a charge/spin condition read as an input.
_CONFIG = {
    "type_map": ["C", "N", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 3.0,
        "sel": 40,
        "channels": 16,
        "n_radial": 8,
        "l_schedule": [1, 2, 2],
        "mixing_layers": [2, 2, 2],
        "focus_dim": 16,
        "precision": "float32",
        "add_chg_spin_ebd": True,
        "default_chg_spin": [0.0, 1.0],
        "random_gamma": False,
        "seed": 3,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [16],
        "precision": "float32",
        "seed": 3,
    },
}

#: Box side in Å of the periodic frames, about three cutoffs.
_SIDE = 7.0


def _model(variant: str) -> torch.nn.Module:
    """The configuration of a variant with every zero parameter redrawn.

    DPA4 zero-initializes its residual output projections, so an untrained
    model reads no edge; redrawing them makes every block and condition change
    the outputs.
    """
    config = copy.deepcopy(_CONFIG)
    if variant == "spin":
        config["spin"] = {"scheme": "native", "use_spin": [True, True, False]}
    if variant == "zbl":
        config["bridging_method"] = "zbl"
    model = get_model(config).to(env.DEVICE).eval()
    generator = torch.Generator(device="cpu").manual_seed(11)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() and not bool((parameter != 0).any()):
                parameter.copy_(
                    0.1
                    * torch.randn(
                        parameter.shape, generator=generator, dtype=torch.float64
                    ).to(parameter)
                )
    return model


def _frame(owned: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions and types of a periodic frame of ``owned`` atoms."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    points = _SIDE * torch.rand(owned, 3, dtype=torch.float64, generator=generator)
    return points.to(env.DEVICE), (torch.arange(owned) % 3).to(env.DEVICE)


def _images(
    points: torch.Tensor, cut: float, swaps: str
) -> tuple[torch.Tensor, torch.Tensor, BorderPlan]:
    """Append the periodic images within reach of a frame as halo rows.

    ``"self"`` receives every image in one swap. ``"lammps"`` emulates the six
    border swaps of one process of a LAMMPS brick decomposition: the atoms
    within ``cut`` of the lower face of a dimension travel as images beyond its
    upper face, then those within ``cut`` of the upper face below the lower
    face, both chosen among the atoms present before the first swap of the
    dimension, owned or received in an earlier dimension.

    Returns the positions of every row, owned first, the owner of every row
    and the border plan that refreshes the halo rows.
    """
    owned = points.shape[0]
    if swaps == "self":
        shifts = torch.cartesian_prod(*[torch.tensor([-1.0, 0.0, 1.0])] * 3)
        shifts = _SIDE * shifts[shifts.abs().sum(-1) > 0].to(points)
        images = (points[None] + shifts[:, None]).reshape(-1, 3)
        near = (
            torch.cdist(images, points, compute_mode="donot_use_mm_for_euclid_dist")
            .amin(-1)
            .lt(cut)
        )
        local = torch.arange(owned, device=points.device)
        owner = torch.cat((local, local.repeat(shifts.shape[0])[near]))
        halo = int(near.sum())
        plan = BorderPlan(
            send_nodes=owner[owned:].to(torch.int32).contiguous(),
            swaps=torch.tensor([[0, 0, halo, halo, owned]], dtype=torch.int32),
            communicator=torch.zeros(1, dtype=torch.int64),
        )
        return torch.cat((points, images[near])), owner, plan
    rows = list(points.cpu().numpy())
    owner = list(range(owned))
    table, send = [], []
    for dimension in range(3):
        present = len(rows)
        for shift in (_SIDE, -_SIDE):
            chosen = [
                i
                for i in range(present)
                if (
                    rows[i][dimension] < cut
                    if shift > 0
                    else rows[i][dimension] >= _SIDE - cut
                )
            ]
            first = len(rows)
            for i in chosen:
                image = rows[i].copy()
                image[dimension] += shift
                rows.append(image)
                owner.append(owner[i])
            table.append([0, 0, len(chosen), len(chosen), first])
            send.extend(chosen)
    plan = BorderPlan(
        send_nodes=torch.tensor(send, dtype=torch.int32, device=points.device),
        swaps=torch.tensor(table, dtype=torch.int32),
        communicator=torch.zeros(1, dtype=torch.int64),
    )
    return (
        torch.as_tensor(np.array(rows), device=points.device),
        torch.tensor(owner, device=points.device),
        plan,
    )


def _neighbor_graph(
    points: torch.Tensor, owned: int, cut: float, owner: torch.Tensor | None
) -> NeighborGraph:
    """The destination-major graph of the rows of the first ``owned`` atoms.

    Every atom within ``cut`` of an owned atom is the source of one of its
    edges. The node set holds every atom, the rows of the others without
    edges, or, given the ``owner`` of every atom, the owned atoms alone, every
    edge then leading from the owner of its source.
    """
    # The direct distance keeps the separation of an atom from itself exact,
    # which the matrix-product form rounds to a small positive value.
    distance = torch.cdist(
        points[:owned], points, compute_mode="donot_use_mm_for_euclid_dist"
    )
    destination, source = torch.nonzero(
        (distance < cut) & (distance > 1e-8), as_tuple=True
    )
    edge_vec = (points[source] - points[destination]).to(torch.float32)
    node_count = points.shape[0]
    if owner is not None:
        source = owner[source]
        node_count = owned
    graph = NeighborGraph(
        n_node=torch.tensor([node_count], device=points.device),
        edge_index=torch.stack((source, destination)),
        edge_vec=edge_vec,
        edge_mask=torch.ones(source.shape[0], dtype=torch.bool, device=points.device),
        n_local=torch.tensor([owned], device=points.device),
    )
    return attach_edge_csr(graph, node_count, canonicalize=True)


def _compact_graph(
    points: torch.Tensor, owned: int, cut: float, owner: torch.Tensor | None
) -> CanonicalGraph:
    """The compact graph of :func:`_neighbor_graph`."""
    return canonical_graph_from_neighbor_graph(
        _neighbor_graph(points, owned, cut, owner)
    )


def _with_guards(graph: CanonicalGraph, count: int) -> CanonicalGraph:
    """The graph with ``count`` guard slots past its physical edges.

    The Kokkos builder hands over its edge arrays at their reserved extent: a
    guard slot carries a zero bond vector on node zero and lies outside every
    CSR row, the source order mapping it to itself.
    """
    storage = graph.source.shape[0]
    return dataclasses.replace(
        graph,
        source=torch.cat((graph.source, graph.source.new_zeros(count))),
        edge_vec=torch.cat((graph.edge_vec, graph.edge_vec.new_zeros(count, 3))),
        source_order=torch.cat(
            (
                graph.source_order,
                torch.arange(
                    storage, storage + count, device=graph.source_order.device
                ),
            )
        ),
    )


def _evaluate(
    model: torch.nn.Module,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    moment: torch.Tensor | None,
    plan: BorderPlan | None = None,
) -> dict[str, torch.Tensor]:
    """The compact canonical lower of a graph with the per-atom virial."""
    charge_spin = torch.tensor([[1.0, 2.0]], dtype=torch.float64, device=env.DEVICE)
    border = (
        {}
        if plan is None
        else {
            "communicator": plan.communicator,
            "halo_send": plan.send_nodes,
            "halo_swaps": plan.swaps,
        }
    )
    return model.forward_lower_canonical_graph(
        atype,
        graph.n_node,
        graph.n_local,
        graph.source,
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        do_atomic_virial=True,
        spin=moment,
        charge_spin=charge_spin,
        **border,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare at the precision of a float32 model, relative to the largest entry."""
    scale = max(expected.abs().max().item(), 1.0)
    torch.testing.assert_close(
        actual.double(), expected.double(), atol=2e-5 * scale, rtol=2e-5
    )


@_CUDA
def test_dpa4_is_eligible_for_the_compact_canonical_lower() -> None:
    """Single-precision DPA4 models without frame or atomic parameters take the
    compact lower, whose bond vectors are single precision, through their
    graph forward. A model-level pair exclusion, which the graph build
    applies and the compact graph does not carry, keeps the graph lower.
    """
    for variant in ("plain", "spin", "zbl"):
        assert canonical_lower_evaluator(_model(variant)) == "graph"
    for region, key, value in (
        ("fitting_net", "numb_fparam", 2),
        ("descriptor", "precision", "float64"),
        (None, "pair_exclude_types", [[0, 1]]),
    ):
        config = copy.deepcopy(_CONFIG)
        (config if region is None else config[region])[key] = value
        assert canonical_lower_evaluator(get_model(config)) is None


@_CUDA
@pytest.mark.parametrize("variant", ["plain", "spin", "zbl"])
def test_compact_lower_matches_graph_lower(variant: str) -> None:
    """The compact lower reproduces the graph lower on the same edges."""
    model = _model(variant)
    points, atype = _frame(40, seed=7)
    images, owner, _ = _images(points, model.get_rcut(), "self")
    neighbor = _neighbor_graph(images, 40, model.get_rcut(), owner)
    moment = None
    if variant == "spin":
        generator = torch.Generator(device="cpu").manual_seed(13)
        moment = torch.randn(40, 3, generator=generator).to(env.DEVICE)
    reference = model.forward_common_lower_graph(
        atype,
        neighbor.n_node,
        neighbor.n_local,
        neighbor.edge_index,
        neighbor.edge_vec,
        neighbor.edge_mask,
        neighbor.destination_order,
        neighbor.destination_row_ptr,
        neighbor.source_order,
        neighbor.source_row_ptr,
        destination_sorted=True,
        do_atomic_virial=True,
        charge_spin=torch.tensor([[1.0, 2.0]], dtype=torch.float64, device=env.DEVICE),
        spin=moment,
    )
    reference = model._translate_eager_call(reference, atype, do_atomic_virial=True)
    result = _evaluate(
        model, canonical_graph_from_neighbor_graph(neighbor), atype, moment
    )
    keys = ["energy", "atom_energy", "force", "virial", "atom_virial"]
    if variant == "spin":
        keys.append("force_mag")
    for key in keys:
        _assert_close(result[key], reference[key])


@_CUDA
@pytest.mark.parametrize("variant", ["plain", "spin"])
@pytest.mark.parametrize("swaps", ["self", "lammps"])
def test_halo_rows_match_the_folded_frame(variant: str, swaps: str) -> None:
    """A node set with halo rows and its border plan matches the folded frame.

    The owned atoms of a periodic frame keep complete rows, and every image
    within reach becomes a halo row without edges, which the border plan
    refreshes from its owner before every block that reads it: by one self
    swap of all images, or by the six swaps of one process of a LAMMPS brick
    decomposition, whose later swaps forward images received by earlier ones.
    The edge storage carries guard slots past the physical edges, as the
    Kokkos builder reserves them. The energy of the owned atoms, the virial
    and the forces of the halo rows, folded onto their owners, match those of
    the folded frame, in which every edge leads from the owner itself.
    """
    model = _model(variant)
    owned = 48
    points, atype = _frame(owned, seed=17)
    cut = model.get_rcut()
    rows, owner, plan = _images(points, cut, swaps)
    assert rows.shape[0] > owned
    moment = halo_moment = None
    if variant == "spin":
        generator = torch.Generator(device="cpu").manual_seed(19)
        moment = torch.randn(owned, 3, generator=generator).to(env.DEVICE)
        halo_moment = moment[owner]
    reference = _evaluate(model, _compact_graph(rows, owned, cut, owner), atype, moment)
    result = _evaluate(
        model,
        _with_guards(_compact_graph(rows, owned, cut, None), 5),
        atype[owner],
        halo_moment,
        plan,
    )
    _assert_close(result["energy"], reference["energy"])
    _assert_close(result["virial"], reference["virial"])
    _assert_close(result["atom_energy"][:owned], reference["atom_energy"])
    keys = ["force", "atom_virial"] + (["force_mag"] if variant == "spin" else [])
    for key in keys:
        value = result[key].reshape(rows.shape[0], -1)
        folded = value.new_zeros(owned, value.shape[1]).index_add_(0, owner, value)
        _assert_close(folded, reference[key].reshape(owned, -1))


@_CUDA
@pytest.mark.parametrize("variant", ["plain", "spin"])
def test_compact_canonical_freeze(
    tmp_path: Path, variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DPA4 model freezes with the compact canonical lower by default.

    The archive records the schema, its reads of the source CSR, of the
    charge/spin condition and of the halo exchange, and its program, given
    the graph with guard slots, reproduces the eager lower of a node set
    without halo nodes.
    """
    from torch._inductor import (
        aoti_load_package,
    )

    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    model = _model(variant)
    archive = tmp_path / "dpa4.pt2"
    deserialize_to_file(
        str(archive), {"model": model.serialize()}, lower_kind="auto", with_comm=False
    )
    with zipfile.ZipFile(archive) as package:
        name = next(n for n in package.namelist() if n.endswith("extra/metadata.json"))
        metadata = json.loads(package.read(name))
    assert metadata["lower_input_kind"] == "dpa_canonical"
    assert metadata["canonical_evaluator"] == "graph"
    assert metadata["graph_edge_dtype"] == "float32"
    assert metadata["canonical_requires_source_csr"]
    assert not metadata["canonical_requires_reciprocal"]
    assert metadata["has_message_passing"]
    assert metadata["dim_chg_spin"] == 2
    assert not metadata["has_comm_artifact"]
    points, atype = _frame(40, seed=23)
    images, owner, _ = _images(points, model.get_rcut(), "self")
    graph = _compact_graph(images, 40, model.get_rcut(), owner)
    moment = None
    if variant == "spin":
        generator = torch.Generator(device="cpu").manual_seed(29)
        moment = torch.randn(40, 3, generator=generator).to(env.DEVICE)
    expected = _evaluate(model, graph, atype, moment)
    graph = _with_guards(graph, 3)
    plan = BorderPlan.single_domain(env.DEVICE)
    inputs = [
        atype,
        graph.n_node,
        graph.n_local,
        canonical_index_bytes(graph.source),
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        *([] if moment is None else [moment]),
        torch.tensor([[1.0, 2.0]], dtype=torch.float64, device=env.DEVICE),
        plan.communicator,
        plan.send_nodes,
        plan.swaps,
    ]
    monkeypatch.setenv("AOTI_RUNTIME_CHECK_INPUTS", "1")
    runner = aoti_load_package(str(archive))
    outputs = runner(*inputs)
    torch.cuda.synchronize(env.DEVICE)
    keys = ["energy", "atom_energy", "force", "virial", "atom_virial"]
    if variant == "spin":
        keys.append("force_mag")
    for key in keys:
        _assert_close(outputs[key].reshape(expected[key].shape), expected[key])
