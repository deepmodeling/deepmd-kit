# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training-time regularizer configuration for DPA-ADAPT."""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)


@dataclass(frozen=True)
class Regularizer:
    """Extra downstream-batch losses added during training.

    ``Regularizer`` deliberately does not model MFT auxiliary data.  External
    auxiliary tasks belong to ``strategy="mft"``; this object is reserved for
    losses on the current downstream training path.
    """

    descriptor_anchor: float = 0.0

    @classmethod
    def from_config(cls, value: Regularizer | Mapping[str, Any] | None) -> Regularizer:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "regularizer must be a Regularizer, a dict, or None; "
                f"got {type(value).__name__}."
            )
        unsupported = sorted(set(value) - {"descriptor_anchor"})
        if unsupported:
            raise ValueError(
                "Regularizer does not accept MFT-style auxiliary fields or "
                f"unknown options: {unsupported}. Use strategy='mft' for aux_data."
            )
        return cls(descriptor_anchor=float(value.get("descriptor_anchor", 0.0)))

    def __post_init__(self) -> None:
        if self.descriptor_anchor < 0.0:
            raise ValueError(
                f"descriptor_anchor must be non-negative; got {self.descriptor_anchor}."
            )

    @property
    def enabled(self) -> bool:
        return self.descriptor_anchor > 0.0

    def require_supported_backend(self) -> None:
        """Fail loudly until a backend implements these training-time losses."""
        if self.enabled:
            raise NotImplementedError(
                "Regularizer(descriptor_anchor=...) is part of the public API "
                "contract, but the dp --pt training backend is not wired yet. "
                "Use strategy='mft' for auxiliary-task regularization, or leave "
                "regularizer unset until descriptor-anchor loss is implemented."
            )
