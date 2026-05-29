# SPDX-License-Identifier: LGPL-3.0-or-later
"""Uni-Mol-tools-like helpers for DeePMD property tasks."""

from .predict import (
    PropertyPredict,
)
from .train import (
    PropertyTrain,
)

__all__ = ["PropertyPredict", "PropertyTrain"]
