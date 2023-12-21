import os
import site
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
    List,
    Optional,
    Tuple,
    Union,
)

from packaging.specifiers import (
    SpecifierSet,
)


def find_tensorflow() -> Tuple[Optional[str], List[str]]:
    """Find TensorFlow library.

    Tries to find TensorFlow in the order of:

    1. Environment variable `TENSORFLOW_ROOT` if set
    2. The current Python environment.
    3. user site packages directory if enabled
    4. system site packages directory (purelib)
    5. add as a requirement (detect TENSORFLOW_VERSION or the latest) and let pip install it

    Returns
    -------
    str
        TensorFlow library path if found.
    list of str
        TensorFlow requirement if not found. Empty if found.
    """
    requires = []

    tf_spec = None
    if os.environ.get("TENSORFLOW_ROOT") is not None:
        site_packages = Path(os.environ.get("TENSORFLOW_ROOT")).parent.absolute()
        tf_spec = FileFinder(str(site_packages)).find_spec("tensorflow")

    # get tensorflow spec
    # note: isolated build will not work for backend
    if tf_spec is None or not tf_spec:
        tf_spec = find_spec("tensorflow")

    if not tf_spec and site.ENABLE_USER_SITE:
        # first search TF from user site-packages before global site-packages
        site_packages = site.getusersitepackages()
        if site_packages:
            tf_spec = FileFinder(site_packages).find_spec("tensorflow")

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
        requires.extend(get_tf_requirement()["cpu"])
        # setuptools will re-find tensorflow after installing setup_requires
        tf_install_dir = None
    return tf_install_dir, requires


def get_tf_requirement(tf_version: str = "") -> dict:
    """Get TensorFlow requirement (CPU) when TF is not installed.

    If tf_version is not given and the environment variable `TENSORFLOW_VERSION` is set, use it as the requirement.

    Parameters
    ----------
    tf_version : str, optional
        TF version

    Returns
    -------
    dict
        TensorFlow requirement, including cpu and gpu.
    """
    if tf_version == "":
        tf_version = os.environ.get("TENSORFLOW_VERSION", "")

    if tf_version == "":
        return {
            "cpu": [
                "tensorflow-cpu; platform_machine!='aarch64'",
                "tensorflow; platform_machine=='aarch64'",
            ],
            "gpu": [
                "tensorflow; platform_machine!='aarch64'",
                "tensorflow; platform_machine=='aarch64'",
            ],
        }
    elif tf_version in SpecifierSet("<1.15") or tf_version in SpecifierSet(
        ">=2.0,<2.1"
    ):
        return {
            "cpu": [
                f"tensorflow=={tf_version}; platform_machine!='aarch64'",
                f"tensorflow=={tf_version}; platform_machine=='aarch64'",
            ],
            "gpu": [
                f"tensorflow-gpu=={tf_version}; platform_machine!='aarch64'",
                f"tensorflow=={tf_version}; platform_machine=='aarch64'",
            ],
        }
    else:
        return {
            "cpu": [
                f"tensorflow-cpu=={tf_version}; platform_machine!='aarch64'",
                f"tensorflow=={tf_version}; platform_machine=='aarch64'",
            ],
            "gpu": [
                f"tensorflow=={tf_version}; platform_machine!='aarch64'",
                f"tensorflow=={tf_version}; platform_machine=='aarch64'",
            ],
        }


def get_tf_version(tf_path: Union[str, Path]) -> str:
    """Get TF version from a TF Python library path.

    Parameters
    ----------
    tf_path : str or Path
        TF Python library path

    Returns
    -------
    str
        version
    """
    if tf_path is None or tf_path == "":
        return ""
    version_file = (
        Path(tf_path) / "include" / "tensorflow" / "core" / "public" / "version.h"
    )
    major = minor = patch = None
    with open(version_file) as f:
        for line in f:
            if line.startswith("#define TF_MAJOR_VERSION"):
                major = line.split()[-1]
            elif line.startswith("#define TF_MINOR_VERSION"):
                minor = line.split()[-1]
            elif line.startswith("#define TF_PATCH_VERSION"):
                patch = line.split()[-1]
            elif line.startswith("#define TF_VERSION_SUFFIX"):
                suffix = line.split()[-1].strip('"')
    if None in (major, minor, patch):
        raise RuntimeError("Failed to read TF version")
    return ".".join((major, minor, patch)) + suffix
