# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import os
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


@lru_cache
def find_paddle() -> tuple[Optional[str], list[str]]:
    """Find PaddlePadle library.

    Tries to find PaddlePadle in the order of:

    1. Environment variable `PADDLE_ROOT` if set
    2. The current Python environment.
    3. user site packages directory if enabled
    4. system site packages directory (purelib)

    Considering the default PaddlePadle package still uses old CXX11 ABI, we
    cannot install it automatically.

    Returns
    -------
    str, optional
        PaddlePadle library path if found.
    list of str
        Paddle requirement if not found. Empty if found.
    """
    if os.environ.get("DP_ENABLE_PADDLE", "0") == "0":
        return None, []
    requires = []
    pd_spec = None

    if (pd_spec is None or not pd_spec) and os.environ.get("PADDLE_ROOT") is not None:
        site_packages = Path(os.environ.get("PADDLE_ROOT")).parent.absolute()
        pd_spec = FileFinder(str(site_packages)).find_spec("paddle")

    # get paddle spec
    # note: isolated build will not work for backend
    if pd_spec is None or not pd_spec:
        pd_spec = find_spec("paddle")

    if not pd_spec and site.ENABLE_USER_SITE:
        # first search TF from user site-packages before global site-packages
        site_packages = site.getusersitepackages()
        if site_packages:
            pd_spec = FileFinder(site_packages).find_spec("paddle")

    if not pd_spec:
        # purelib gets site-packages path
        site_packages = get_path("purelib")
        if site_packages:
            pd_spec = FileFinder(site_packages).find_spec("paddle")

    # get install dir from spec
    try:
        pd_install_dir = pd_spec.submodule_search_locations[0]  # type: ignore
        # AttributeError if ft_spec is None
        # TypeError if submodule_search_locations are None
        # IndexError if submodule_search_locations is an empty list
    except (AttributeError, TypeError, IndexError):
        pd_install_dir = None
        requires.extend(get_pd_requirement()["paddle"])
    return pd_install_dir, requires


@lru_cache
def get_pd_requirement(pd_version: str = "") -> dict:
    """Get PaddlePadle requirement when Paddle is not installed.

    If pd_version is not given and the environment variable `PADDLE_VERSION` is set, use it as the requirement.

    Parameters
    ----------
    pd_version : str, optional
        Paddle version

    Returns
    -------
    dict
        PaddlePadle requirement.
    """
    if pd_version is None:
        return {"paddle": []}
    if pd_version == "":
        pd_version = os.environ.get("PADDLE_VERSION", "")

    return {
        "paddle": [
            "paddlepaddle>=3.0.0",
        ],
    }


@lru_cache
def get_pd_version(pd_path: Optional[Union[str, Path]]) -> str:
    """Get Paddle version from a Paddle Python library path.

    Parameters
    ----------
    pd_path : str or Path
        Paddle Python library path, e.g. "/python3.10/site-packages/paddle/"

    Returns
    -------
    str
        version
    """
    if pd_path is None or pd_path == "":
        return ""
    version_file = Path(pd_path) / "version" / "__init__.py"
    spec = importlib.util.spec_from_file_location("paddle.version", version_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.full_version
