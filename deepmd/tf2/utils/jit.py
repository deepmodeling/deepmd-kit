# SPDX-License-Identifier: LGPL-3.0-or-later
"""TensorFlow 2 JIT configuration helpers."""

from __future__ import (
    annotations,
)

import os


def env_flag(name: str) -> bool:
    """Return whether an environment flag is enabled."""
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def default_jit_compile() -> bool:
    """Return the default tf.function XLA setting for TF2 code paths."""
    return env_flag("DP_JIT")
