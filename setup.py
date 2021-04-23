"""Setup script for DeePMD-kit package."""

import os
from distutils.util import get_platform
from importlib.machinery import FileFinder
from importlib.util import find_spec
from pathlib import Path
from sysconfig import get_path

from packaging.specifiers import SpecifierSet
from pkg_resources import Distribution
from skbuild import setup
from skbuild.cmaker import get_cmake_version
from skbuild.exceptions import SKBuildError

# define constants
INSTALL_REQUIRES = (Path(__file__).parent / "requirements.txt").read_text().splitlines()
setup_requires = ["setuptools_scm", "scikit-build"]

# read readme to markdown
readme_file = Path(__file__).parent / "README.md"
readme = readme_file.read_text()

tf_version = os.environ.get("TENSORFLOW_VERSION", "2.3")

if tf_version in SpecifierSet("<1.15") or tf_version in SpecifierSet(">=2.0,<2.1"):
    extras_require = {
        "cpu": [f"tensorflow=={tf_version}"],
        "gpu": [f"tensorflow-gpu=={tf_version}"],
    }
else:
    extras_require = {
        "cpu": [f"tensorflow-cpu=={tf_version}"],
        "gpu": [f"tensorflow=={tf_version}"],
    }

# get tensorflow spec
tf_spec = find_spec("tensorflow")
if not tf_spec:
    # purelib gets site-packages path
    site_packages = get_path("purelib")
    if site_packages:
        tf_spec = FileFinder(site_packages).find_spec("tensorflow")

# get install dir from spec
try:
    tf_install_dir = tf_spec.submodule_search_locations[0]  # type: ignore
    # AttributeError if ft_spec is None
    # TypeError if submodule_search_locations are None
    # IndexError if submodule_search_locations is an empty list
except (AttributeError, TypeError, IndexError):
    setup_requires.append(f"tensorflow=={tf_version}")
    dist = Distribution(
        project_name="tensorflow", version=tf_version, platform=get_platform()
    ).egg_name()
    tf_install_dir = Path(__file__).parent.resolve().joinpath(".egg", dist, "tensorflow").resolve()

# add cmake as a build requirement if cmake>3.7 is not installed
try:
    cmake_version = get_cmake_version()
except SKBuildError:
    setup_requires.append("cmake")
else:
    if cmake_version in SpecifierSet("<3.7"):
        setup_requires.append("cmake")

Path("deepmd").mkdir(exist_ok=True)

setup(
    name="deepmd-kit",
    setup_requires=setup_requires,
    use_scm_version={"write_to": "deepmd/_version.py"},
    author="Han Wang",
    author_email="wang_han@iapcm.ac.cn",
    description="A deep learning package for many-body potential energy representation and molecular dynamics",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/deepmodeling/deepmd-kit",
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
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    keywords="deepmd",
    install_requires=INSTALL_REQUIRES,
    cmake_args=[
        f"-DTENSORFLOW_ROOT:STRING={tf_install_dir}",
        "-DBUILD_PY_IF:BOOL=TRUE",
        "-DBUILD_CPP_IF:BOOL=FALSE",
        "-DFLOAT_PREC:STRING=high",
    ],
    cmake_source_dir="source",
    cmake_minimum_required_version="3.0",
    extras_require={
        "test": ["dpdata>=0.1.9", "ase", "pytest", "pytest-cov", "pytest-sugar"],
        "docs": ["sphinx", "recommonmark", "sphinx_rtd_theme"],
        **extras_require,
    },
    entry_points={"console_scripts": ["dp = deepmd.entrypoints.main:main"]},
)
