# SPDX-License-Identifier: LGPL-3.0-or-later
"""Folding the case embedding of a fitting into its first layer.

The case embedding is the same block of input columns for every atom, so the
fold moves its product with the first-layer weight into the bias. The folded
fitting carries no case embedding and gives the outputs of the original, also
after a serialization round trip. A first layer whose residual adds its input
to the output keeps the case embedding.
"""

import numpy as np
import pytest

from deepmd.dpmodel.fitting import (
    InvarFitting,
)

NF, NLOC, NTYPES, NCASE = 2, 5, 2, 3


def _fitting(
    dim_descrpt: int, mixed_types: bool, numb_fparam: int, numb_aparam: int
) -> InvarFitting:
    fitting = InvarFitting(
        "energy",
        NTYPES,
        dim_descrpt,
        1,
        neuron=[16, 16],
        resnet_dt=True,
        numb_fparam=numb_fparam,
        numb_aparam=numb_aparam,
        dim_case_embd=NCASE,
        mixed_types=mixed_types,
        seed=3,
    )
    fitting.set_case_embd(1)
    return fitting


@pytest.mark.parametrize(
    ("dim_descrpt", "mixed_types", "numb_fparam", "numb_aparam"),
    [
        (12, True, 0, 0),  # one network over the descriptor and the case
        (12, False, 2, 3),  # per-type networks behind frame and atomic parameters
        (16, True, 0, 0),  # the first layer turns square once folded
    ],
)
def test_the_folded_fitting_gives_the_outputs_of_the_original(
    dim_descrpt: int, mixed_types: bool, numb_fparam: int, numb_aparam: int
) -> None:
    rng = np.random.default_rng(0)
    fitting = _fitting(dim_descrpt, mixed_types, numb_fparam, numb_aparam)
    descriptor = rng.normal(size=(NF, NLOC, dim_descrpt))
    atype = rng.integers(0, NTYPES, size=(NF, NLOC))
    fparam = rng.normal(size=(NF, numb_fparam)) if numb_fparam else None
    aparam = rng.normal(size=(NF, NLOC, numb_aparam)) if numb_aparam else None
    expected = fitting(descriptor, atype, fparam=fparam, aparam=aparam)["energy"]

    fitting.fold_case_embedding()

    assert fitting.dim_case_embd == 0
    assert fitting.case_embd is None
    assert fitting.nets[0].in_dim == dim_descrpt + numb_fparam + numb_aparam
    restored = InvarFitting.deserialize(fitting.serialize())
    for folded in (fitting, restored):
        np.testing.assert_allclose(
            folded(descriptor, atype, fparam=fparam, aparam=aparam)["energy"],
            expected,
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "dim_input",
    [16, 8],  # the residual adds the input once, or twice side by side
)
def test_a_residual_over_the_case_columns_keeps_the_case_embedding(
    dim_input: int,
) -> None:
    # The descriptor and the case together are as wide as the first layer, or
    # half as wide, so its residual adds the case embedding to the output.
    fitting = _fitting(dim_input - NCASE, True, 0, 0)

    fitting.fold_case_embedding()

    assert fitting.dim_case_embd == NCASE
    assert fitting.nets[0][0].w.shape == (dim_input, 16)
