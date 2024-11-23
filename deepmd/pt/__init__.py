# SPDX-License-Identifier: LGPL-3.0-or-later

# import customized OPs globally
from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)
from deepmd.utils.entry_point import (
    load_entry_point,
)

load_entry_point("deepmd.pt")

__all__ = [
    "ENABLE_CUSTOMIZED_OP",
]
