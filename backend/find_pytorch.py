# SPDX-License-Identifier: LGPL-3.0-or-later
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
)


@lru_cache
def find_pytorch() -> Optional[str]:
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
    """
    if os.environ.get("DP_ENABLE_PYTORCH", "1") == "0":
        return None
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
    return pt_install_dir
