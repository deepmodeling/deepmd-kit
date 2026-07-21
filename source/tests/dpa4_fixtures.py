# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared DPA4 test fixtures used across test roots.

``jitter_zero_arrays`` is imported by ``source/tests/common/dpmodel``,
``source/tests/pt_expt`` test modules that need a message-sensitive DPA4
fixture. ``source/tests/infer/gen_dpa4.py`` keeps its own inline copy
(mirror of this module) because it is a standalone script that runs
outside pytest's package machinery.
"""

import numpy as np


def jitter_zero_arrays(node, rng: np.random.Generator) -> None:
    """Recursively replace every exactly-zero float array with small noise.

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

    This function jitters *every* float array in the serialized weight
    tree that is exactly all-zero, wherever it occurs (not only the named
    residual projections above) -- it has no notion of which key it is
    perturbing, it only inspects array values. Non-zero arrays (e.g. the
    learned type embedding) are left untouched, and traversal order is a
    fixed depth-first walk, so two calls seeded identically produce
    bit-identical trees.

    Parameters
    ----------
    node : dict, list, np.ndarray, or other
        Root (or sub-tree) of a serialized parameter tree, mutated in
        place. Non-container, non-array leaves are ignored.
    rng : np.random.Generator
        Seeded RNG used to draw the replacement noise.
    """
    if isinstance(node, dict):
        items: object = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return
    for key, value in items:
        if (
            isinstance(value, np.ndarray)
            and value.dtype.kind == "f"
            and value.size > 0
            and np.all(value == 0.0)
        ):
            # Replace the zero array in its parent container rather than
            # mutating it in place. Assigning a fresh array into the caller's
            # (freshly serialized) dict avoids writing through an array that
            # CodeQL's dataflow can trace back to a mutable default value
            # (py/modification-of-default-value). The RNG draw order, shapes,
            # and dtype are unchanged, so seeded calls stay bit-identical.
            node[key] = rng.normal(0.0, 0.05, size=value.shape).astype(value.dtype)
        else:
            jitter_zero_arrays(value, rng)
