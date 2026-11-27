# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared DPA4 test fixtures used across test roots.

``jitter_zero_arrays`` is imported by ``source/tests/common/dpmodel``,
``source/tests/pt_expt`` test modules that need a message-sensitive DPA4
fixture. ``source/tests/infer/gen_dpa4.py`` keeps its own inline copy
(mirror of this module) because it is a standalone script that runs
outside pytest's package machinery.
"""

import numpy as np


def four_atom_spin_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinates, types, spins and complete non-self neighbour rows.

    Coordinates and spins have shape (1, 4, 3), types have shape (1, 4),
    and neighbour indices have shape (1, 4, 3). Floating arrays use float64;
    integer arrays use int64. Type 0 carries active magnetic moments.
    """
    coord = np.array(
        [[[0.0, 0.0, 0.0], [1.2, 0.3, 0.2], [-0.2, 1.5, 0.4], [0.1, -0.4, 1.3]]],
        dtype=np.float64,
    )
    atype = np.array([[0, 0, 1, 0]], dtype=np.int64)
    spin = np.array(
        [[[0.5, 0.3, -0.2], [-0.4, 0.7, 0.2], [0.1, 0.2, 0.3], [0.3, -0.7, 0.9]]],
        dtype=np.float64,
    )
    nlist = np.array([[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]], dtype=np.int64)
    return coord, atype, spin, nlist


def tetrahedral_spin_configurations() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return a tetrahedral star and two zero-total-moment configurations.

    The coordinates have shape (1, 5, 3), in angstrom. The centre has type 1
    and its four neighbours have type 0, with types of shape (1, 5). A cutoff
    of 4 angstrom includes the centre-neighbour bonds of length 3 angstrom
    and excludes the neighbour-neighbour bonds. The two spin arrays have
    shape (1, 5, 3), with a non-magnetic centre and unit neighbour moments.
    Their neighbour quadrupole sums have squared norms 4 and 16.
    """
    vertices = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        dtype=np.float64,
    )
    coord = np.concatenate([np.zeros((1, 3)), np.sqrt(3.0) * vertices])[None]
    atype = np.array([[1, 0, 0, 0, 0]], dtype=np.int64)
    spin_a = np.array(
        [[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]],
        dtype=np.float64,
    )
    spin_b = spin_a.copy()
    spin_b[0, 3:] = [[1, 0, 0], [-1, 0, 0]]
    return coord, atype, spin_a, spin_b


def jitter_zero_arrays(node, rng: np.random.Generator):
    """Return a copy of a serialized tree with every zero float array jittered.

    DPA4 deliberately zero-initializes several residual output projections
    (``SO2Convolution.post_focus_mix``, ``EquivariantFFN.so3_linear_2`` --
    both per-block and the top-level ``output_ffn`` -- see the "Zero-
    initialized so residual path starts near-identity" comments in
    ``dpa4_nn/so2.py``/``dpa4_nn/ffn.py``) so a freshly constructed,
    untrained descriptor is architecturally edge/message independent: its
    scalar read-out is exactly the type embedding regardless of geometry,
    neighbors, or ``exclude_types``. That makes a bare ``make_descriptor()``
    vacuous for an exclusion anti-vacuity check -- excluding pairs cannot
    change an output that never depended on edges.

    This function replaces *every* float array in the serialized weight tree
    that is exactly all-zero, wherever it occurs (not only the named residual
    projections above) -- it has no notion of which key it is perturbing, it
    only inspects array values. Non-zero arrays (e.g. the learned type
    embedding) are carried through untouched, and traversal order is a fixed
    depth-first walk, so two calls seeded identically produce bit-identical
    trees.

    This is a PURE rebuild -- it does not mutate ``node`` (nor anything it
    references), so callers must use the return value:
    ``data = jitter_zero_arrays(data, rng)``. (Mutating ``node`` in place
    tripped CodeQL's ``py/modification-of-default-value`` dataflow.)

    Parameters
    ----------
    node : dict, list, np.ndarray, or other
        Root (or sub-tree) of a serialized parameter tree. Not mutated.
    rng : np.random.Generator
        Seeded RNG used to draw the replacement noise.

    Returns
    -------
    dict, list, np.ndarray, or other
        A new tree of the same shape with zero float arrays replaced; leaves
        that are not zero float arrays are returned as-is.
    """
    if isinstance(node, dict):
        return {key: jitter_zero_arrays(value, rng) for key, value in node.items()}
    if isinstance(node, list):
        return [jitter_zero_arrays(value, rng) for value in node]
    if (
        isinstance(node, np.ndarray)
        and node.dtype.kind == "f"
        and node.size > 0
        and np.all(node == 0.0)
    ):
        return rng.normal(0.0, 0.05, size=node.shape).astype(node.dtype)
    return node
