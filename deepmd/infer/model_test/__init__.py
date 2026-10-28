# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluation of a trained model against labelled data.

The machinery behind ``dp test``: a tester walks one system in chunks,
evaluates each chunk and combines the errors. Lazy data sources such as LMDB
therefore need not fit in memory; ordinary ``DeepmdData`` systems materialize
the complete test system before it is sliced into evaluation chunks.
:func:`build_tester` selects the tester of a model class; the command-line
entry point only discovers the systems and reports the run-level average.
"""

import logging
from typing import (
    Any,
)

from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.infer.deep_polar import (
    DeepGlobalPolar,
    DeepPolar,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.infer.model_test.base import (
    ChunkContext,
    ModelTester,
    save_txt_file,
    test_chunk_atoms,
)
from deepmd.infer.model_test.dos import (
    DosTester,
)
from deepmd.infer.model_test.ener import (
    EnerTester,
    SpinEnerTester,
)
from deepmd.infer.model_test.property import (
    PropertyTester,
)
from deepmd.infer.model_test.tensor import (
    DipoleTester,
    PolarTester,
    TensorTester,
)

log = logging.getLogger(__name__)

__all__ = [
    "ChunkContext",
    "DipoleTester",
    "DosTester",
    "EnerTester",
    "ModelTester",
    "PolarTester",
    "PropertyTester",
    "SpinEnerTester",
    "TensorTester",
    "build_tester",
    "save_txt_file",
    "test_chunk_atoms",
]


def build_tester(dp: Any, *, atomic: bool) -> ModelTester:
    """Return the tester of the model class under test.

    Parameters
    ----------
    dp : Any
        The evaluator of the model under test.
    atomic : bool
        Whether per-atom quantities are computed.

    Returns
    -------
    ModelTester
        A tester able to evaluate one system of that model class.

    Raises
    ------
    RuntimeError
        If no tester covers the model class.
    """
    if isinstance(dp, DeepPot):
        has_spin = dp.has_spin or dp.get_ntypes_spin() != 0
        tester = SpinEnerTester if has_spin else EnerTester
        return tester(dp, atomic=atomic)
    if isinstance(dp, DeepDOS):
        return DosTester(dp, atomic=atomic)
    if isinstance(dp, DeepProperty):
        return PropertyTester(dp, atomic=atomic)
    if isinstance(dp, DeepGlobalPolar):
        # A global polar model reports one tensor per frame, which is what a
        # polar model does when per-atom output is not requested.
        log.warning(
            "Global polar model is not currently supported. Please directly "
            "use the polar mode and change loss parameters."
        )
        return PolarTester(dp, atomic=False)
    if isinstance(dp, DeepPolar):
        return PolarTester(dp, atomic=atomic)
    if isinstance(dp, DeepDipole):
        return DipoleTester(dp, atomic=atomic)
    raise RuntimeError(f"Testing is not supported for {type(dp).__name__}.")
