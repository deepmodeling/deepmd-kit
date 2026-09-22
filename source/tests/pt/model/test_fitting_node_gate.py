# SPDX-License-Identifier: LGPL-3.0-or-later
"""The readout gate of a bridging window on the pt fitting networks.

A bridged descriptor hands the fitting a per-atom source gate. The gate fades
the learned part of the output, the deviation of the output from the per-type
bias the fitting stores, so an atom whose gate is zero keeps that bias alone
and an atom whose gate is one is untouched. Both the generic energy fitting
and the SeZM one are covered, the latter on both of its branches: the plain
path and the one that conditions the network on a case embedding.
"""

import numpy as np
import pytest
import torch

from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.model.task.sezm_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)

NF, NLOC, ND, NTYPES = 2, 5, 8, 3


def _build(kind: str, vacuum_ref: bool) -> torch.nn.Module:
    """One fitting of each kind, in float64 on the test device."""
    if kind == "invar":
        fitting = InvarFitting(
            "energy",
            NTYPES,
            ND,
            1,
            mixed_types=True,
            precision="float64",
            vacuum_ref=vacuum_ref,
            seed=GLOBAL_SEED,
        )
    else:
        fitting = SeZMEnergyFittingNet(
            NTYPES,
            ND,
            neuron=[8, 8],
            precision="float64",
            mixed_types=True,
            dim_case_embd=2 if kind == "sezm_case_film" else 0,
            case_film_embd=kind == "sezm_case_film",
            vacuum_ref=vacuum_ref,
            seed=GLOBAL_SEED,
        )
    fitting = fitting.to(device=env.DEVICE, dtype=torch.float64)
    if fitting.dim_case_embd > 0:
        fitting.set_case_embd(0)
    return fitting


def _inputs(vacuum_ref: bool):
    rng = np.random.default_rng(GLOBAL_SEED)
    descriptor = torch.as_tensor(
        rng.normal(size=(NF, NLOC, ND)), dtype=torch.float64, device=env.DEVICE
    )
    atype = torch.as_tensor(rng.integers(0, NTYPES, size=(NF, NLOC)), device=env.DEVICE)
    node_gate = torch.as_tensor(
        rng.uniform(0.0, 1.0, size=(NF, NLOC, 1)),
        dtype=torch.float64,
        device=env.DEVICE,
    )
    kwargs = {}
    if vacuum_ref:
        kwargs["vacuum_descriptor"] = torch.as_tensor(
            rng.normal(size=(NTYPES, ND)), dtype=torch.float64, device=env.DEVICE
        )
    return descriptor, atype, node_gate, kwargs


def _constant_gate(value: float) -> torch.Tensor:
    return torch.full((NF, NLOC, 1), value, dtype=torch.float64, device=env.DEVICE)


@pytest.mark.parametrize("kind", ["invar", "sezm", "sezm_case_film"])
@pytest.mark.parametrize("vacuum_ref", [False, True])
def test_the_gate_fades_the_deviation_from_the_bias(
    kind: str, vacuum_ref: bool
) -> None:
    """``out = bias + gate * (out_without_gate - bias)`` atom by atom."""
    fitting = _build(kind, vacuum_ref)
    descriptor, atype, node_gate, kwargs = _inputs(vacuum_ref)

    ungated = fitting(descriptor, atype, **kwargs)["energy"]
    gated = fitting(descriptor, atype, node_gate=node_gate, **kwargs)["energy"]

    reference = fitting.bias_atom_e[atype].to(torch.float64)
    torch.testing.assert_close(
        gated,
        reference + node_gate * (ungated - reference),
        rtol=0.0,
        atol=1e-12,
    )
    # a gate that changed nothing would make the comparison vacuous
    assert (gated - ungated).abs().max().item() > 1e-6


@pytest.mark.parametrize("kind", ["invar", "sezm", "sezm_case_film"])
@pytest.mark.parametrize("vacuum_ref", [False, True])
def test_the_two_ends_of_the_fade(kind: str, vacuum_ref: bool) -> None:
    """A closed gate leaves the bias; an open one leaves the output alone."""
    fitting = _build(kind, vacuum_ref)
    descriptor, atype, _, kwargs = _inputs(vacuum_ref)

    closed = fitting(
        descriptor,
        atype,
        node_gate=_constant_gate(0.0),
        **kwargs,
    )["energy"]
    torch.testing.assert_close(
        closed, fitting.bias_atom_e[atype].to(torch.float64), rtol=0.0, atol=1e-12
    )

    opened = fitting(
        descriptor,
        atype,
        node_gate=_constant_gate(1.0),
        **kwargs,
    )["energy"]
    torch.testing.assert_close(
        opened, fitting(descriptor, atype, **kwargs)["energy"], rtol=0.0, atol=1e-12
    )

    if vacuum_ref:
        # the fold moves the reference out of the bias; the gate still reads
        # the bias the atoms of a frozen pair fade to
        bias = fitting.bias_atom_e.clone()
        fitting.fold_vacuum_reference(kwargs["vacuum_descriptor"])
        assert fitting.folded_reference is not None
        torch.testing.assert_close(
            fitting.readout_reference(), bias, rtol=0.0, atol=1e-12
        )
        folded = fitting(descriptor, atype, node_gate=_constant_gate(0.0))["energy"]
        torch.testing.assert_close(folded, closed, rtol=0.0, atol=1e-12)
