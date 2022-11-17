"""Setup script for DeePMD-kit package."""

import os
import sys

from skbuild import setup

topdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(topdir, 'backend'))

from find_tensorflow import find_tensorflow, get_tf_requirement, get_tf_version


cmake_args = []
# get variant option from the environment varibles, available: cpu, cuda, rocm
dp_variant = os.environ.get("DP_VARIANT", "cpu").lower()
if dp_variant == "cpu" or dp_variant == "":
    pass
elif dp_variant == "cuda":
    cmake_args.append("-DUSE_CUDA_TOOLKIT:BOOL=TRUE")
    cuda_root = os.environ.get("CUDA_TOOLKIT_ROOT_DIR")
    if cuda_root:
        cmake_args.append(f"-DCUDA_TOOLKIT_ROOT_DIR:STRING={cuda_root}")
elif dp_variant == "rocm":
    cmake_args.append("-DUSE_ROCM_TOOLKIT:BOOL=TRUE")
    rocm_root = os.environ.get("ROCM_ROOT")
    if rocm_root:
        cmake_args.append(f"-DCMAKE_HIP_COMPILER_ROCM_ROOT:STRING={rocm_root}")
else:
    raise RuntimeError("Unsupported DP_VARIANT option: %s" % dp_variant)

if os.environ.get("DP_BUILD_TESTING", "0") == "1":
    cmake_args.append("-DBUILD_TESTING:BOOL=TRUE")

tf_install_dir, _ = find_tensorflow()
tf_version = get_tf_version(tf_install_dir)


# TODO: migrate packages and entry_points to pyproject.toml after scikit-build supports it
# See also https://scikit-build.readthedocs.io/en/latest/usage.html#setuptools-options
setup(
    packages=[
        "deepmd",
        "deepmd/descriptor",
        "deepmd/fit",
        "deepmd/infer",
        "deepmd/loss",
        "deepmd/utils",
        "deepmd/loggers",
        "deepmd/cluster",
        "deepmd/entrypoints",
        "deepmd/op",
        "deepmd/model",
        "deepmd/train",
        "deepmd/nvnmd",
        "deepmd/nvnmd/data",
        "deepmd/nvnmd/descriptor",
        "deepmd/nvnmd/entrypoints",
        "deepmd/nvnmd/fit",
        "deepmd/nvnmd/utils",
    ],
    cmake_args=[
        f"-DTENSORFLOW_ROOT:PATH={tf_install_dir}",
        "-DBUILD_PY_IF:BOOL=TRUE",
        "-DBUILD_CPP_IF:BOOL=FALSE",
        *cmake_args,
    ],
    cmake_source_dir="source",
    cmake_minimum_required_version="3.16",
    extras_require={
        "test": ["dpdata>=0.1.9", "ase", "pytest", "pytest-cov", "pytest-sugar"],
        "docs": [
            "sphinx>=3.1.1",
            "recommonmark",
            "sphinx_rtd_theme>=1.0.0rc1",
            "sphinx_markdown_tables",
            "myst-parser",
            "breathe",
            "exhale",
            "numpydoc",
            "ase",
            "deepmodeling-sphinx>=0.1.0",
            "dargs>=0.3.1",
            "sphinx-argparse",
            "pygments-lammps",
            ],
        **get_tf_requirement(tf_version),
    },
    entry_points={"console_scripts": ["dp = deepmd.entrypoints.main:main"]},
)
