# SPDX-License-Identifier: LGPL-3.0-or-later

# import customized OPs globally
try:
    from deepmd.pd.cxx_op import (
        ENABLE_CUSTOMIZED_OP,
    )

    __all__ = [
        "ENABLE_CUSTOMIZED_OP",
    ]
except Exception as e:
    __all__ = []
