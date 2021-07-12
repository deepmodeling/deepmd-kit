"""This module will house cust Tf OPs after CMake installation."""

from pathlib import Path
import importlib
import logging

NOT_LOADABLE = ("__init__.py")
PACKAGE_BASE = "deepmd.op"

log = logging.getLogger(__name__)


def import_ops():
    """Import all custom TF ops that are present in this submodule.

    Note
    ----
    Initialy this subdir is unpopulated. CMake will install all the op module python
    files and shared libs.
    """
    for module_file in Path(__file__).parent.glob("*.py"):
        if module_file.name not in NOT_LOADABLE:
            module_name = f".{module_file.stem}"
            log.debug(f"importing op module: {module_name}")
            importlib.import_module(module_name, PACKAGE_BASE)


import_ops()
