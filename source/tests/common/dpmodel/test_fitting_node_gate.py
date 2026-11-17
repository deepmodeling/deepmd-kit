# SPDX-License-Identifier: LGPL-3.0-or-later
"""The readout gate of a bridging window on the dpmodel fitting.

A bridged descriptor hands the fitting a per-atom source gate. The gate fades
the learned part of the output, the deviation of the output from the per-type
bias the fitting stores, so an atom whose gate is zero keeps that bias alone
and an atom whose gate is one is untouched. The same rule applies on the dense
call and on the graph-native one.
"""

import numpy as np
import pytest

from deepmd.dpmodel.fitting import (
    InvarFitting,
)

NF, NLOC, ND, NTYPES = 2, 4, 8, 2


def _fitting(vacuum_ref: bool) -> InvarFitting:
    return InvarFitting(
        "energy",
        NTYPES,
        ND,
        1,
        mixed_types=True,
        vacuum_ref=vacuum_ref,
        seed=3,
    )


def _inputs(rng: np.random.Generator, vacuum_ref: bool):
    descriptor = rng.normal(size=(NF, NLOC, ND))
    atype = rng.integers(0, NTYPES, size=(NF, NLOC))
    node_gate = rng.uniform(0.0, 1.0, size=(NF, NLOC, 1))
    vacuum = rng.normal(size=(NTYPES, ND)) if vacuum_ref else None
    return descriptor, atype, node_gate, vacuum


@pytest.mark.parametrize("vacuum_ref", [False, True])
def test_dense_call_fades_the_learned_part(vacuum_ref: bool) -> None:
    """``out = bias + gate * (out_without_gate - bias)`` atom by atom."""
    rng = np.random.default_rng(0)
    fitting = _fitting(vacuum_ref)
    bias = np.asarray(fitting.bias_atom_e).reshape(NTYPES, 1)
    descriptor, atype, node_gate, vacuum = _inputs(rng, vacuum_ref)

    ungated = fitting(descriptor, atype, vacuum_descriptor=vacuum)["energy"]
    gated = fitting(descriptor, atype, vacuum_descriptor=vacuum, node_gate=node_gate)[
        "energy"
    ]

    reference = bias[atype]
    np.testing.assert_allclose(
        gated, reference + node_gate * (ungated - reference), rtol=0.0, atol=1e-12
    )
    # the fixture must separate the two: a gate of one would make it vacuous
    assert np.abs(gated - ungated).max() > 1e-6


@pytest.mark.parametrize("vacuum_ref", [False, True])
def test_a_closed_gate_leaves_the_bias_and_an_open_one_changes_nothing(
    vacuum_ref: bool,
) -> None:
    """The two ends of the fade are the bias itself and the plain output."""
    rng = np.random.default_rng(1)
    fitting = _fitting(vacuum_ref)
    bias = np.asarray(fitting.bias_atom_e).reshape(NTYPES, 1)
    descriptor, atype, _, vacuum = _inputs(rng, vacuum_ref)

    closed = fitting(
        descriptor,
        atype,
        vacuum_descriptor=vacuum,
        node_gate=np.zeros((NF, NLOC, 1)),
    )["energy"]
    np.testing.assert_allclose(closed, bias[atype], rtol=0.0, atol=1e-12)

    ungated = fitting(descriptor, atype, vacuum_descriptor=vacuum)["energy"]
    opened = fitting(
        descriptor,
        atype,
        vacuum_descriptor=vacuum,
        node_gate=np.ones((NF, NLOC, 1)),
    )["energy"]
    np.testing.assert_allclose(opened, ungated, rtol=0.0, atol=1e-12)

    if vacuum_ref:
        # the fold moves the reference out of the bias; the gate still reads
        # the bias the atoms of a frozen pair fade to
        fitting.fold_vacuum_reference(vacuum)
        assert fitting.folded_reference is not None
        np.testing.assert_allclose(fitting.readout_reference(), bias, atol=1e-12)
        folded = fitting(descriptor, atype, node_gate=np.zeros((NF, NLOC, 1)))
        np.testing.assert_allclose(folded["energy"], closed, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("vacuum_ref", [False, True])
def test_call_graph_matches_the_dense_gate(vacuum_ref: bool) -> None:
    """The flat node axis carries the gate the same way the dense call does."""
    rng = np.random.default_rng(2)
    fitting = _fitting(vacuum_ref)
    descriptor, atype, node_gate, vacuum = _inputs(rng, vacuum_ref)

    dense = fitting(descriptor, atype, vacuum_descriptor=vacuum, node_gate=node_gate)[
        "energy"
    ]
    n_nodes = NF * NLOC
    flat = fitting.call_graph(
        descriptor.reshape(n_nodes, ND),
        atype.reshape(n_nodes),
        vacuum_descriptor=vacuum,
        node_gate=node_gate.reshape(n_nodes),
    )["energy"]
    assert flat.shape == (n_nodes, 1)
    np.testing.assert_allclose(flat, dense.reshape(n_nodes, 1), rtol=0.0, atol=1e-12)
