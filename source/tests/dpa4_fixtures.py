# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared DPA4 test fixtures used across test roots.

``jitter_zero_arrays`` is imported by ``source/tests/common/dpmodel``,
``source/tests/pt_expt`` test modules that need a message-sensitive DPA4
fixture. ``source/tests/infer/gen_dpa4.py`` keeps its own inline copy
(mirror of this module) because it is a standalone script that runs
outside pytest's package machinery.
"""

import numpy as np


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
