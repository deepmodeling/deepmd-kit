# SPDX-License-Identifier: LGPL-3.0-or-later
"""Density floor of DPA4 measured from the training neighborhoods."""

from __future__ import (
    annotations,
)

import itertools
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DPDescriptor
from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
    load_or_measure_envelope_mass,
    mean_envelope_mass,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    C3CutoffEnvelope,
)
from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    apply_pair_exclusion,
    build_neighbor_graph,
)
from deepmd.pt_expt.descriptor.dpa4 import DescrptDPA4 as ExportDescriptor
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.utils.path import (
    DPPath,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )

BOX = 7.0


def sample(natoms: int = 24, seed: int = 0) -> list[dict[str, np.ndarray]]:
    """Return two periodic frames of ``natoms`` atoms of two types."""
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, BOX, (2, natoms, 3))
    return [
        {
            "coord": coord.reshape(2, -1),
            "atype": rng.integers(0, 2, (2, natoms)),
            "box": np.tile(np.eye(3).reshape(1, 9) * BOX, (2, 1)),
        }
    ]


def config(rcut: float | list[float], density_floor: float | None = 2.0) -> dict:
    """Return a small two-block descriptor with attention."""
    return {
        "ntypes": 2,
        "sel": 80,
        "rcut": rcut,
        "channels": 8,
        "n_radial": 4,
        "lmax": 2,
        "l_schedule": [1, 2],
        "n_blocks": 2,
        "n_atten_head": 1,
        "random_gamma": False,
        "precision": "float64",
        "density_floor": density_floor,
        "seed": 3,
    }


def test_envelope_mass_matches_periodic_sum() -> None:
    """The measured mass is the mean squared-envelope sum over all images."""
    data = sample()
    radii = [3.5, 5.0]
    measured, count = mean_envelope_mass(
        data, radii, exponent=5, ntypes=2, exclude_types=[]
    )
    assert count == 48
    envelope = C3CutoffEnvelope(rcut=max(radii), exponent=5)
    coord = data[0]["coord"].reshape(2, 24, 3)
    shifts = np.array(list(itertools.product(range(-1, 2), repeat=3))) * BOX
    expected = np.zeros(len(radii))
    for frame in coord:
        separation = (
            frame[None, :, None, :] + shifts[None, None] - frame[:, None, None, :]
        )
        distance = np.linalg.norm(separation, axis=-1).reshape(24, -1)
        distance = distance[distance > 1e-9].reshape(24, -1)
        for index, radius in enumerate(radii):
            inside = distance < radius
            expected[index] += np.sum(
                envelope.call(distance, rcut=radius) ** 2 * inside
            )
    np.testing.assert_allclose(measured, expected / coord.shape[0] / 24, rtol=1e-12)


@pytest.mark.parametrize(
    "backend", [DPDescriptor, ExportDescriptor], ids=["dpmodel", "pt_expt"]
)
@pytest.mark.parametrize("rcut", [4.0, [3.0, 4.0]])
def test_statistics_set_floor_in_every_backend(
    backend: type, rcut: float | list[float]
) -> None:
    """Each backend measures the same masses and starts the null mass there."""
    data = sample()
    radii = rcut if isinstance(rcut, list) else [rcut, rcut]
    expected, _ = mean_envelope_mass(
        data, radii, exponent=5, ntypes=2, exclude_types=[]
    )
    descriptor = backend(**config(rcut))
    descriptor.compute_input_stats(data)
    np.testing.assert_allclose(
        to_numpy_array(descriptor.envelope_mass), expected, rtol=1e-12
    )
    for block, mass in zip(descriptor.blocks, expected, strict=True):
        raw = to_numpy_array(block.so2_conv.adamw_attn_z_bias_raw)
        null_mass = np.logaddexp(0.0, raw)
        np.testing.assert_allclose(null_mass, np.full_like(null_mass, 2.0 * mass))
    restored = backend.deserialize(descriptor.serialize())
    np.testing.assert_array_equal(
        to_numpy_array(restored.envelope_mass),
        to_numpy_array(descriptor.envelope_mass),
    )


def test_statistics_file_records_measurement(tmp_path: Path) -> None:
    """A recorded measurement is read back without sampling the data again."""
    data = sample()
    options = {"radii": [3.0, 4.0], "exponent": 5, "ntypes": 2, "exclude_types": []}
    path = DPPath(str(tmp_path), "a")
    first, count = load_or_measure_envelope_mass(lambda: data, path, **options)

    def unreachable() -> list:
        raise AssertionError("the recorded measurement must be reused")

    recorded, recorded_count = load_or_measure_envelope_mass(
        unreachable, path, **options
    )
    np.testing.assert_array_equal(recorded, first)
    assert recorded_count == count
    other, _ = load_or_measure_envelope_mass(
        lambda: data, path, **{**options, "radii": [3.5, 4.0]}
    )
    assert not np.array_equal(other, first)


def test_unset_floor_keeps_the_state() -> None:
    """Without ``density_floor`` the descriptor has no measured state."""
    descriptor = ExportDescriptor(**config(4.0, density_floor=None))
    descriptor.compute_input_stats(sample())
    assert "envelope_mass" not in descriptor.state_dict()
    portable = DPDescriptor(**config(4.0, density_floor=None))
    assert "envelope_mass" not in portable.serialize()["@variables"]


def test_envelope_mass_is_the_training_degree() -> None:
    """The measurement equals the mean degree of the forward pass.

    The model-level exclusion removes the (1, 2) pairs and the
    descriptor-level exclusion the (0, 1) pairs from the degree, and no
    neighbor capacity limits it.
    """
    rng = np.random.default_rng(5)
    coord = rng.uniform(0.0, BOX, (1, 30, 3))
    atype = rng.integers(0, 3, (1, 30))
    box = np.eye(3).reshape(1, 9) * BOX
    system = {
        "coord": coord.reshape(1, -1),
        "atype": atype,
        "box": box,
        "pair_exclude_types": [(1, 2)],
    }
    measured, count = mean_envelope_mass(
        [system], [4.0], exponent=5, ntypes=3, exclude_types=[(0, 1)]
    )
    options = {**config(4.0), "ntypes": 3, "sel": 4, "exclude_types": [(0, 1), (1, 2)]}
    descriptor = ExportDescriptor(**options).to(env.DEVICE).eval()
    numpy_graph = build_neighbor_graph(coord, atype, box, 4.0)
    graph = NeighborGraph(
        **{
            name: torch.as_tensor(getattr(numpy_graph, name), device=env.DEVICE)
            for name in ("n_node", "edge_index", "edge_vec", "edge_mask")
        }
    )
    seen = []

    def capture(module: torch.nn.Module, args: tuple) -> None:
        seen.append(args[1])

    handle = descriptor.blocks[0].register_forward_pre_hook(capture)
    try:
        descriptor.call_graph(
            graph, torch.as_tensor(atype.reshape(-1), device=env.DEVICE)
        )
    finally:
        handle.remove()
    assert count == 30
    np.testing.assert_allclose(measured[0], seen[0].deg.mean().item(), rtol=1e-10)


def _dense_envelope_mass(
    system: dict,
    radii: list[float],
    ntypes: int,
    exclude_types: list[tuple[int, int]],
) -> tuple[np.ndarray, int]:
    """Return the mean envelope mass read from the dense carry-all neighbor graph."""
    envelope = C3CutoffEnvelope(rcut=max(radii), exponent=5)
    atype = system["atype"]
    nframes, nloc = atype.shape
    excluded = PairExcludeMask(ntypes, [*exclude_types, *system["pair_exclude_types"]])
    graph = build_neighbor_graph(
        system["coord"].reshape(nframes, nloc, 3),
        atype,
        system.get("box"),
        max(radii),
    )
    graph = apply_pair_exclusion(graph, np.maximum(atype.reshape(-1), 0), excluded)
    keep = np.asarray(graph.edge_mask).astype(bool)
    center = np.asarray(graph.edge_index)[1][keep]
    distance = np.linalg.norm(np.asarray(graph.edge_vec)[keep], axis=-1)
    counted = atype.reshape(-1) >= 0
    mass = np.zeros(len(radii))
    for index, radius in enumerate(radii):
        weight = envelope.call(distance, rcut=radius) ** 2
        degree = np.bincount(center, weights=weight, minlength=nframes * nloc)
        mass[index] = degree[counted].sum()
    return mass / counted.sum(), int(counted.sum())


@pytest.mark.parametrize(
    ("cell", "padding"),
    [
        (np.diag([7.0, 7.0, 7.0]), 0),
        (np.array([[7.0, 0.0, 0.0], [1.5, 6.5, 0.0], [0.5, 1.0, 7.5]]), 3),
        (None, 2),
        (np.diag([3.0, 3.0, 3.0]), 1),
        (np.array([[3.2, 0.0, 0.0], [1.4, 2.8, 0.0], [0.9, 0.7, 3.1]]), 1),
    ],
    ids=[
        "orthorhombic",
        "triclinic_padded",
        "non_periodic_padded",
        "cell_below_cutoff",
        "triclinic_below_cutoff",
    ],
)
def test_envelope_mass_matches_dense_graph(
    cell: np.ndarray | None, padding: int
) -> None:
    """The k-d tree neighbor search reproduces the dense neighbor graph.

    Cells smaller than the cutoff need several periodic images per direction.

    Model-level and descriptor-level pair exclusions are both active, and the
    padding atoms (negative type) are neither centers nor neighbors.
    """
    rng = np.random.default_rng(1)
    nframes, nloc = 2, 24
    if cell is None:
        coord = rng.uniform(0.0, 8.0, (nframes, nloc, 3))
    else:
        coord = rng.uniform(0.0, 1.0, (nframes, nloc, 3)) @ cell
    atype = rng.integers(0, 3, (nframes, nloc))
    if padding:
        atype[:, -padding:] = -1
    system = {
        "coord": coord.reshape(nframes, -1),
        "atype": atype,
        "pair_exclude_types": [(1, 2)],
    }
    if cell is not None:
        system["box"] = np.tile(cell.reshape(1, 9), (nframes, 1))
    radii = [3.0, 4.5]
    measured, count = mean_envelope_mass(
        [system], radii, exponent=5, ntypes=3, exclude_types=[(0, 1)]
    )
    expected, expected_count = _dense_envelope_mass(system, radii, 3, [(0, 1)])
    assert count == expected_count
    np.testing.assert_allclose(measured, expected, rtol=1e-12)
