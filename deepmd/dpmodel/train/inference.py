# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent inference policy configuration."""

import os
from collections.abc import (
    Generator,
)
from contextlib import (
    contextmanager,
)
from typing import (
    Any,
)

#: Accepted ``DP_TF32_INFER`` values and the matmul precision each selects.
_TF32_INFER_PRECISIONS = {"0": "highest", "1": "high", "2": "medium"}
_BOOL_ENV_VALUES = {
    "1": True,
    "true": True,
    "yes": True,
    "on": True,
    "0": False,
    "false": False,
    "no": False,
    "off": False,
}


def infer_env_defaults(validating_params: dict[str, Any]) -> dict[str, str]:
    """
    Translate the eval-time policy options into environment defaults.

    Models sample these variables once, while they are being constructed, so
    the configuration has to reach them through the environment rather than
    through a constructor argument. A variable exported by the user takes
    precedence; see :func:`scoped_env_defaults`.

    Parameters
    ----------
    validating_params : dict[str, Any]
        The normalized ``validating`` section.

    Returns
    -------
    dict[str, str]
        The environment variables requested by the configuration.
    """
    flags = {
        "compiled_infer": "DP_COMPILE_INFER",
        "tf32_infer": "DP_TF32_INFER",
        "amp_infer": "DP_AMP_INFER",
    }
    return {
        name: "1" for flag, name in flags.items() if validating_params.get(flag, False)
    }


@contextmanager
def scoped_env_defaults(defaults: dict[str, str]) -> Generator[None, None, None]:
    """Temporarily set missing environment variables and restore them afterward."""
    previous = {key: os.environ.get(key) for key in defaults}
    try:
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def infer_matmul_precision() -> str:
    """
    Resolve the eval-time matmul precision from ``DP_TF32_INFER``.

    Returns
    -------
    str
        A backend-independent precision level: ``"highest"`` for
        ``0`` (the default), ``"high"`` for ``1``, ``"medium"`` for ``2``.

    Raises
    ------
    ValueError
        If ``DP_TF32_INFER`` is set to anything other than 0, 1 or 2.
    """
    value = os.environ.get("DP_TF32_INFER", "0").strip().lower()
    if value not in _TF32_INFER_PRECISIONS:
        raise ValueError(f"DP_TF32_INFER must be one of 0/1/2, got {value!r}")
    return _TF32_INFER_PRECISIONS[value]


def infer_compile_enabled() -> bool:
    """
    Resolve whether evaluation forwards use the compiled model.

    Returns
    -------
    bool
        Whether ``DP_COMPILE_INFER`` enables compiled evaluation.

    Raises
    ------
    ValueError
        If ``DP_COMPILE_INFER`` is not a supported boolean value.
    """
    value = os.environ.get("DP_COMPILE_INFER", "0").strip().lower()
    if value not in _BOOL_ENV_VALUES:
        choices = "/".join(_BOOL_ENV_VALUES)
        raise ValueError(f"DP_COMPILE_INFER must be one of {choices}, got {value!r}")
    return _BOOL_ENV_VALUES[value]


def infer_amp_enabled() -> bool:
    """Return whether inference requests bfloat16 matrix products."""
    return os.environ.get("DP_AMP_INFER", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
