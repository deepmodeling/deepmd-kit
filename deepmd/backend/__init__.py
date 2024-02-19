# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backends.

Avoid directly importing third-party libraries in this module for performance.
"""
# copy from dpdata
from importlib import (
    import_module,
    metadata,
)
from pathlib import (
    Path,
)

PACKAGE_BASE = "deepmd.backend"
NOT_LOADABLE = ("__init__.py",)

for module_file in Path(__file__).parent.glob("*.py"):
    if module_file.name not in NOT_LOADABLE:
        module_name = f".{module_file.stem}"
        import_module(module_name, PACKAGE_BASE)

# https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html
try:
    eps = metadata.entry_points(group="deepmd.backend")
except TypeError:
    eps = metadata.entry_points().get("deepmd.backend", [])
for ep in eps:
    plugin = ep.load()
