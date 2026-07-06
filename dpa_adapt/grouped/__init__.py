# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Grouped property data helpers.

Most users should keep data in ordinary ``deepmd/npy`` form and call
``mark_groups`` only when existing systems need grouped markers:

    from dpa_adapt import mark_groups

    mark_groups("oer/dpdata", target="overpotential", group_by="system")

``Assembly`` is the low-level writer for converter implementations or tests
that already have component arrays in memory:

    from dpa_adapt import Assembly

    a = Assembly(target="x")
    a.group(label=..., fparam={...}).add(coords, symbols, weight=...)
    a.write(PATH)

The DeepMD tensor names still use ``group_id`` internally because that is the
training primitive, and users describe assemblies and groups.
"""

_LAZY = {
    "Assembly": ("._core", "Assembly"),
    "ComponentSpec": ("._core", "ComponentSpec"),
    "GroupSpec": ("._core", "GroupSpec"),
    "PoolMask": ("._core", "PoolMask"),
    "SiteSelector": ("._core", "SiteSelector"),
    "SubstitutionSpec": ("._core", "SubstitutionSpec"),
    "GroupMarkerResult": ("._convert", "GroupMarkerResult"),
    "mark_groups": ("._convert", "mark_groups"),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib

        mod_name, attr_name = _LAZY[name]
        module = importlib.import_module(mod_name, __package__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
