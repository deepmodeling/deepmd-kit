# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import os
import platform
import site
from functools import (
    lru_cache,
)
from importlib.machinery import (
    FileFinder,
)
from importlib.util import (
    find_spec,
)
from pathlib import (
    Path,
)
from sysconfig import (
    get_path,
)
from typing import (
    Optional,
    Union,
)

from packaging.specifiers import (
    SpecifierSet,
)
from packaging.version import (
    Version,
)


@lru_cache
def find_pytorch() -> tuple[Optional[str], list[str]]:
    """Find PyTorch library.

    Tries to find PyTorch in the order of:

    1. Environment variable `PYTORCH_ROOT` if set
    2. The current Python environment.
    3. user site packages directory if enabled
    4. system site packages directory (purelib)

    Considering the default PyTorch package still uses old CXX11 ABI, we
    cannot install it automatically.

    Returns
    -------
    str, optional
        PyTorch library path if found.
    list of str
        TensorFlow requirement if not found. Empty if found.
    """
    if os.environ.get("DP_ENABLE_PYTORCH", "0") == "0":
        return None, []
    requires = []
    pt_spec = None

    if (pt_spec is None or not pt_spec) and os.environ.get("PYTORCH_ROOT") is not None:
        site_packages = Path(os.environ.get("PYTORCH_ROOT")).parent.absolute()
        pt_spec = FileFinder(str(site_packages)).find_spec("torch")

    # get pytorch spec
    # note: isolated build will not work for backend
    if pt_spec is None or not pt_spec:
        pt_spec = find_spec("torch")

    if not pt_spec and site.ENABLE_USER_SITE:
        # first search TF from user site-packages before global site-packages
        site_packages = site.getusersitepackages()
        if site_packages:
            pt_spec = FileFinder(site_packages).find_spec("torch")

    if not pt_spec:
        # purelib gets site-packages path
        site_packages = get_path("purelib")
        if site_packages:
            pt_spec = FileFinder(site_packages).find_spec("torch")

    # get install dir from spec
    try:
        pt_install_dir = pt_spec.submodule_search_locations[0]  # type: ignore
        # AttributeError if ft_spec is None
        # TypeError if submodule_search_locations are None
        # IndexError if submodule_search_locations is an empty list
    except (AttributeError, TypeError, IndexError):
        pt_install_dir = None
        requires.extend(get_pt_requirement()["torch"])
    return pt_install_dir, requires


@lru_cache
def get_pt_requirement(pt_version: str = "") -> dict:
    """Get PyTorch requirement when PT is not installed.

    If pt_version is not given and the environment variable `PYTORCH_VERSION` is set, use it as the requirement.

    Parameters
    ----------
    pt_version : str, optional
        PT version

    Returns
    -------
    dict
        PyTorch requirement.
    """
    if pt_version is None:
        return {"torch": []}
    if (
        os.environ.get("CIBUILDWHEEL", "0") == "1"
        and platform.system() == "Linux"
        and platform.machine() == "x86_64"
    ):
        cuda_version = os.environ.get("CUDA_VERSION", "12.2")
        if cuda_version == "" or cuda_version in SpecifierSet(">=12,<13"):
            # CUDA 12.2, cudnn 9
            pt_version = "2.7.0"
        elif cuda_version in SpecifierSet(">=11,<12"):
            # CUDA 11.8, cudnn 8
            pt_version = "2.3.1"
        else:
            raise RuntimeError("Unsupported CUDA version") from None
    if pt_version == "":
        pt_version = os.environ.get("PYTORCH_VERSION", "")

    return {
        "torch": [
            # uv has different local version behaviors, i.e. `==2.3.1` cannot match `==2.3.1+cpu`
            # https://github.com/astral-sh/uv/blob/main/PIP_COMPATIBILITY.md#local-version-identifiers
            # luckily, .* (prefix matching) defined in PEP 440 can match any local version
            # https://peps.python.org/pep-0440/#version-matching
            f"torch=={Version(pt_version).base_version}.*"
            if pt_version != ""
            # https://github.com/pytorch/pytorch/commit/7e0c26d4d80d6602aed95cb680dfc09c9ce533bc
            else "torch>=2.1.0"
        ],
    }


@lru_cache
def get_pt_version(pt_path: Optional[Union[str, Path]]) -> str:
    """Get TF version from a TF Python library path.

    Parameters
    ----------
    pt_path : str or Path
        PT Python library path

    Returns
    -------
    str
        version
    """
    if pt_path is None or pt_path == "":
        return ""
    version_file = Path(pt_path) / "version.py"
    spec = importlib.util.spec_from_file_location("torch.version", version_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__
