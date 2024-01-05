# SPDX-License-Identifier: LGPL-3.0-or-later
"""Submodule that contains all the DeePMD-Kit entry point scripts."""

from ..infer.model_devi import (
    make_model_devi,
)
from .compress import (
    compress,
)
from .convert import (
    convert,
)
from .doc import (
    doc_train_input,
)
from .freeze import (
    freeze,
)
from .gui import (
    start_dpgui,
)
from .neighbor_stat import (
    neighbor_stat,
)
from .test import (
    test,
)

# import `train` as `train_dp` to avoid the conflict of the
# module name `train` and the function name `train`
from .train import train as train_dp
from .transfer import (
    transfer,
)

__all__ = [
    "doc_train_input",
    "freeze",
    "test",
    "train_dp",
    "transfer",
    "compress",
    "doc_train_input",
    "make_model_devi",
    "convert",
    "neighbor_stat",
    "start_dpgui",
]
