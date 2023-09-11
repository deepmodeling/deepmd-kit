# SPDX-License-Identifier: LGPL-3.0-or-later
"""A PEP-517 backend to find TensorFlow."""
from typing import (
    List,
)

from scikit_build_core import build as _orig

from .find_tensorflow import (
    find_tensorflow,
)
from .read_env import (
    set_scikit_build_env,
)

__all__ = [
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
]


def __dir__() -> List[str]:
    return __all__


set_scikit_build_env()

prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_wheel = _orig.build_wheel
build_sdist = _orig.build_sdist
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
build_editable = _orig.build_editable


def get_requires_for_build_wheel(
    config_settings: dict,
) -> List[str]:
    return _orig.get_requires_for_build_wheel(config_settings) + find_tensorflow()[1]


def get_requires_for_build_editable(
    config_settings: dict,
) -> List[str]:
    return _orig.get_requires_for_build_editable(config_settings) + find_tensorflow()[1]
