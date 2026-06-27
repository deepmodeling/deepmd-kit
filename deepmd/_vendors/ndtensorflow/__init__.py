from __future__ import annotations

from typing import Final

from ._array import Array
from ._namespace import *
from ._namespace import __all__ as _namespace_all
from ._info import __array_namespace_info__
from . import fft, linalg

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
