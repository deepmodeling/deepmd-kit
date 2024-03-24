# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
)

from .find_tensorflow import (
    get_tf_requirement,
)
from .read_env import (
    get_argument_from_env,
)

__all__ = ["dynamic_metadata"]


def __dir__() -> List[str]:
    return __all__


def dynamic_metadata(
    field: str,
    settings: Optional[Dict[str, object]] = None,
) -> str:
    assert field in ["optional-dependencies", "entry-points", "scripts"]
    _, _, find_libpython_requires, extra_scripts, tf_version = get_argument_from_env()
    if field == "scripts":
        return {
            "dp": "deepmd.main:main",
            **extra_scripts,
        }
    elif field == "optional-dependencies":
        return {
            "test": [
                "dpdata>=0.2.7",
                "ase",
                "pytest",
                "pytest-cov",
                "pytest-sugar",
                "dpgui",
            ],
            "docs": [
                "sphinx>=3.1.1",
                "sphinx_rtd_theme>=1.0.0rc1",
                "sphinx_markdown_tables",
                "myst-nb>=1.0.0rc0",
                "myst-parser>=0.19.2",
                "sphinx-design",
                "breathe",
                "exhale",
                "numpydoc",
                "ase",
                "deepmodeling-sphinx>=0.1.0",
                "dargs>=0.3.4",
                "sphinx-argparse",
                "pygments-lammps",
                "sphinxcontrib-bibtex",
            ],
            "lmp": [
                "lammps~=2023.8.2.3.0",
                *find_libpython_requires,
            ],
            "ipi": [
                "i-PI",
                *find_libpython_requires,
            ],
            "gui": [
                "dpgui",
            ],
            **get_tf_requirement(tf_version),
            "cu11": [
                "nvidia-cuda-runtime-cu11",
                "nvidia-cublas-cu11",
                "nvidia-cufft-cu11",
                "nvidia-curand-cu11",
                "nvidia-cusolver-cu11",
                "nvidia-cusparse-cu11",
                "nvidia-cudnn-cu11",
                "nvidia-cuda-nvcc-cu11",
            ],
            "cu12": [
                "nvidia-cuda-runtime-cu12",
                "nvidia-cublas-cu12",
                "nvidia-cufft-cu12",
                "nvidia-curand-cu12",
                "nvidia-cusolver-cu12",
                "nvidia-cusparse-cu12",
                "nvidia-cudnn-cu12",
                "nvidia-cuda-nvcc-cu12",
            ],
            "torch": [
                "torch>=2a",
            ],
        }
