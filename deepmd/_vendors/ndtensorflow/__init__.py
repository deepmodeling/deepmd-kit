# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from typing import (
    Final,
)

from . import (
    fft,
    linalg,
)
from ._array import (
    Array,
)
from ._info import (
    __array_namespace_info__,
)
from ._namespace import *
from ._namespace import __all__ as _namespace_all

__array_api_version__: Final = "2025.12"

__all__ = sorted(
    set(_namespace_all)
    | {
        "Array",
        "__array_api_version__",
        "__array_namespace_info__",
        "fft",
        "linalg",
    }
)


def __dir__() -> list[str]:
    return __all__
