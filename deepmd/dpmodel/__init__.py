# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.utils.entry_point import (
    load_entry_point,
)

from .common import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from .model import (
    DPModelCommon,
)
from .output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
    fitting_check_output,
    get_deriv_name,
    get_hessian_name,
    get_reduce_name,
    model_check_output,
)

__all__ = [
    "DEFAULT_PRECISION",
    "PRECISION_DICT",
    "DPModelCommon",
    "FittingOutputDef",
    "ModelOutputDef",
    "NativeOP",
    "OutputVariableDef",
    "fitting_check_output",
    "get_deriv_name",
    "get_hessian_name",
    "get_reduce_name",
    "model_check_output",
]


load_entry_point("deepmd.dpmodel")
