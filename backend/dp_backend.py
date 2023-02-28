"""A PEP-517 backend to find TensorFlow."""
from typing import (
    List,
)

from find_tensorflow import (
    find_tensorflow,
)

# TODO: switch to scikit_build_core after it is available
from setuptools import build_meta as _orig

__all__ = [
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
]


def __dir__() -> List[str]:
    return __all__


prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_wheel = _orig.build_wheel
build_sdist = _orig.build_sdist
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist


def get_requires_for_build_wheel(
    config_settings: dict,
) -> List[str]:
    return _orig.get_requires_for_build_wheel(config_settings) + find_tensorflow()[1]


# TODO: export get_requires_for_build_editable, prepare_metadata_for_build_editable, build_editable
# after scikit-build is ready
# See https://github.com/scikit-build/scikit-build/issues/740
# Now we use the legacy-editable mode
