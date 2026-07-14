# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for metadata exposed by serialized JAX HLO models."""

from deepmd.jax.model.hlo import HLO


def test_hlo_get_nnei_uses_stored_selection() -> None:
    """Return the neighbor width encoded in an HLO model's selection metadata.

    Constructing a complete ``HLO`` instance requires valid serialized
    StableHLO artifacts.  This API only depends on the stored ``sel`` metadata,
    so bypass initialization to test that metadata contract directly.
    """
    model = HLO.__new__(HLO)
    model.sel = [6, 12, 1]

    assert model.get_nnei() == sum(model.sel)
