"""This module will house cust Tf OPs after CMake installation."""

from pathlib import Path
import importlib

NOT_LOADABLE = ("__init__.py")
PACKAGE_BASE = "deepmd.op"


def import_ops():
    """Import all custom TF ops that are present in this submodule."""
    for module_file in Path(__file__).parent.glob("*.py"):
        if module_file.name not in NOT_LOADABLE:
            module_name = module_file.stem
            importlib.import_module(module_name, PACKAGE_BASE)


import_ops()
