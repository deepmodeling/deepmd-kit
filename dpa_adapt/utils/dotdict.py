# SPDX-License-Identifier: LGPL-3.0-or-later
# utils/dotdict.py

from typing import (
    Any,
)


class DotDict(dict):
    """A dict subclass that allows attribute-style access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{name}'") from None
